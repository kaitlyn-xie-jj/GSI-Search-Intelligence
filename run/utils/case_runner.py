# run/utils/case_runner.py
import copy
import json
import os
import time
import traceback
import asyncio
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

# Setup project path before importing modules
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from modules.task_solver.unified_task_solver import UnifiedTaskSolver
from modules.task_solver.solver_factory import create_solver
from modules.platform.platform_factory import (
    initialize_platform,
    get_scene_graph,
    PlatformType
)
from modules.utils.trace_loader import ReplayTrace
from modules.events import start_global_event_bus
from modules.monitor.task_monitor import get_global_task_monitor
from modules.utils.system.var_dump import set_default_dump_dir, dump_var
from modules.config.system_config import config
from modules.dataset_loader.loader import DatasetLoader, load_unreal_goal
from modules.hitl.hitl_manager import get_hitl_manager, HITLManager
from modules.communication.unified_communicator import UnifiedCommunicator
from run.utils.common import reset_global_environment, find_matching_replay_trace, setup_project_path
from run.utils.log_token_parser import compute_token_stats

project_root = setup_project_path()


def resolve_dataset_root() -> Path:
    root = os.environ.get("GSI_DATASET_ROOT")
    if root:
        return Path(root).expanduser().resolve()
    return (project_root / "dataset").resolve()


def token_stats_enabled() -> bool:
    return os.environ.get("GSI_DISABLE_TOKEN_STATS", "1").strip().lower() in {
        "0",
        "false",
        "no",
        "off",
    }


class ExperimentRunner:
    """
    Single experiment runner (supports externally injected UnifiedTaskSolver to fix run_dir, etc.)
    """

    DEFAULT_SIMULATION_DELAY: float = 2.0

    def __init__(self) -> None:
        self.task_solver: Optional[UnifiedTaskSolver] = None
        self.running: bool = False
        self.ui_instances: Dict[str, Any] = {}

    async def run_task_solver(
            self,
            injected_solver: Optional[UnifiedTaskSolver] = None,
            save_dir: Optional[Path] = None,
            replay_trace: Optional[ReplayTrace] = None,
    ) -> Dict[str, Any]:
        """
        Return structure:
        {
            "time_steps": ...,
            "metrics": {...},
            "elapsed_sec": float
        }
        """
        try:
            print("=== Starting SGI Task Solver ===")
            print("Starting global event bus...")
            await start_global_event_bus()
            print("Global event bus started, all components will share this instance")

            print("Initializing SGI task solver...")
            self.task_solver = (
                injected_solver
                if injected_solver is not None
                else create_solver(
                    planner_mode=(
                        config.planner_mode if config.planner_mode else "phase"
                    ),
                    robot_type_list=(
                        config.default_robot_types
                        if config.default_robot_types
                        else ["UAV", "UGV", "Quadruped", "Humanoid"]
                    ),
                    enable_render=(
                        config.enable_visualization
                        if config.enable_visualization
                        else False
                    ),
                    run_dir=save_dir,
                    replay_trace=replay_trace,
                )
            )

            # Initialize platform services.
            print("Initializing platform service...")
            await self.task_solver.initialize()

            print("Starting task solving...")
            t0 = time.time()
            success = await self.task_solver.solve_task()
            t1 = time.time()
            elapsed = round(t1 - t0, 6)
            print(f"Task duration: {elapsed:.2f}s")
            metrics = self.task_solver.get_metrics() if self.task_solver else {}

            return {
                "success": success,
                "metrics": metrics,
                "elapsed_sec": elapsed,
            }

        except Exception as e:
            print(f"Error occurred while running task solver: {e}")
            traceback.print_exc()
            # Keep the return structure consistent.
            return {"time_steps": [], "metrics": {}, "elapsed_sec": 0.0}

    def _postprocess_and_save(
            self,
            run_result: Dict[str, Any],
            save_dir: Optional[Path] = None,
            extra_meta: Optional[Dict[str, Any]] = None,
    ) -> Path:
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        target_dir: Path

        if save_dir:
            target_dir = Path(save_dir)
        else:
            ws_root = None
            try:
                if self.task_solver and hasattr(self.task_solver, "path_manager"):
                    ws_root = getattr(
                        self.task_solver.path_manager, "workspace_root", None
                    )
            except Exception:
                ws_root = None

            if ws_root:
                target_dir = Path(ws_root)

        target_dir.mkdir(parents=True, exist_ok=True)

        metrics = run_result.get("metrics", {}) or {}
        elapsed = float(run_result.get("elapsed_sec", 0.0))

        # Write metrics.json with full raw metrics.
        metrics_path = target_dir / "metrics.json"
        try:
            with metrics_path.open("w", encoding="utf-8") as f:
                json.dump(metrics, f, ensure_ascii=False, indent=2)
            print(f"[Post-processing] Metrics saved: {metrics_path}")
        except Exception as e:
            print(f"[Post-processing] Failed to write metrics.json: {e}")

        # Build summary: common metrics plus algorithm-specific metrics.
        planning_durs = metrics.get("planning_durations", []) or []
        allocation_durs = metrics.get("allocation_durations", []) or []
        
        summary = {
            "timestamp": ts,
            # ---- Common metrics ----
            "elapsed_sec": elapsed,
            "success": bool(metrics.get("success", False)),
            "llm_calls": metrics.get("llm_calls", 0),
            "replans_full": metrics.get("replans_full", 0),
            "replans_partial": metrics.get("replans_partial", 0),
            "replans_total": metrics.get("replans_total", 0),
            "planning_duration_mean": (
                round(sum(planning_durs) / max(1, len(planning_durs)), 6)
            ),
            "planning_duration_std": (
                round(
                    (sum((x - sum(planning_durs) / max(1, len(planning_durs))) ** 2
                         for x in planning_durs) / max(1, len(planning_durs))) ** 0.5, 6
                ) if planning_durs else 0.0
            ),
            "planning_durations": planning_durs,
            "allocation_duration_mean": (
                round(sum(allocation_durs) / max(1, len(allocation_durs)), 6)
            ),
            "allocation_duration_std": (
                round(
                    (sum((x - sum(allocation_durs) / max(1, len(allocation_durs))) ** 2
                         for x in allocation_durs) / max(1, len(allocation_durs))) ** 0.5, 6
                ) if allocation_durs else 0.0
            ),
            "allocation_durations": allocation_durs,
            "total_energy": metrics.get("total_energy", 0.0),
            "newcase_total": metrics.get("newcase_total", 0),
            "newcase_total_orig": metrics.get("newcase_total_orig", 0),
            "newcase_top_types": sorted(
                (metrics.get("newcase_by_type", {}) or {}).items(),
                key=lambda kv: kv[1],
                reverse=True,
            )[:5],
        }
        
        # ---- Token statistics, parsed offline from log.md ----
        if token_stats_enabled():
            try:
                log_md_path = target_dir / "log.md"
                token_stats = compute_token_stats(log_md_path)
                if token_stats:
                    summary["prompt_tokens_mean"] = token_stats["prompt_tokens_mean"]
                    summary["response_tokens_mean"] = token_stats["response_tokens_mean"]
                    summary["prompt_tokens_total"] = token_stats["prompt_tokens_total"]
                    summary["response_tokens_total"] = token_stats["response_tokens_total"]
                    summary["prompt_tokens"] = token_stats["prompt_tokens"]
                    summary["response_tokens"] = token_stats["response_tokens"]
                    summary["llm_call_count_from_log"] = token_stats["llm_call_count_from_log"]
            except Exception as e:
                print(f"[Post-processing] Token statistics failed: {e}")
        else:
            print("[Post-processing] Token statistics skipped; set GSI_DISABLE_TOKEN_STATS=0 to enable")
        
        # ---- Algorithm-specific metrics ----
        # Automatically collect numeric and list metrics not in common metrics.
        common_keys = {
            "success", "llm_calls", "replans_full", "replans_partial",
            "replans_total", "planning_durations", "allocation_durations",
            "total_energy", "newcase_total", "newcase_total_orig", "newcase_by_type",
        }
        algo_specific = {}
        for key, value in metrics.items():
            if key in common_keys:
                continue
            if isinstance(value, (int, float, bool)):
                algo_specific[key] = value
            elif isinstance(value, list) and value:
                # List type: record mean.
                try:
                    nums = [float(v) for v in value]
                    algo_specific[f"{key}_mean"] = round(sum(nums) / len(nums), 6)
                except (TypeError, ValueError):
                    algo_specific[key] = value
        
        if algo_specific:
            summary["algorithm_specific"] = algo_specific

        if extra_meta:
            summary.update(extra_meta)

        summary_path = target_dir / "single_run_summary.json"
        try:
            with summary_path.open("w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            print(f"[Post-processing] Summary saved: {summary_path}")
        except Exception as e:
            print(f"[Post-processing] Failed to write single_run_summary.json: {e}")

        return target_dir

    async def run_experiment(
            self,
            enable_replanning: bool = True,
            enable_render: bool = True,
            injected_solver: Optional[UnifiedTaskSolver] = None,
            save_dir: Optional[Path] = None,
            replay_trace: Optional[ReplayTrace] = None,
    ) -> bool:
        try:
            self.running = True
            print("=== SGI Task Solver Experiment Started ===")

            # Run the task solver.
            run_result = await self.run_task_solver(
                injected_solver=injected_solver,
                save_dir=save_dir,
                replay_trace=replay_trace,
            )

            # Optional renderer stage.
            if enable_render:
                try:
                    await asyncio.sleep(1)
                except KeyboardInterrupt:
                    print("\nUser interrupted, cleaning up...")

            # Postprocess and save.
            try:
                # Extra metadata: get scene and goal information from global context.
                extra_meta = {}
                try:
                    tc = get_scene_graph()
                    extra_meta = {
                        "goal_id": getattr(getattr(tc, "_goal", None), "id", None),
                        "scene_nodes_count": len(getattr(tc, "_nodes", []) or []),
                        "scene_edges_count": len(getattr(tc, "_edges", []) or []),
                    }
                except Exception:
                    pass

                try:
                    pmode = None
                    rtypes = None
                    s_enable_render = None
                    if self.task_solver is not None:
                        pmode = getattr(self.task_solver, "planner_mode", None)
                        rtypes = getattr(self.task_solver, "robot_type_list", None)
                        s_enable_render = getattr(
                            self.task_solver, "enable_render", None
                        )
                    extra_meta["input_params"] = {
                        "planner_mode": pmode,
                        "robot_type_list": rtypes,
                        "enable_render": bool(s_enable_render),
                    }
                except Exception:
                    pass

                self._postprocess_and_save(
                    run_result, save_dir=save_dir, extra_meta=extra_meta
                )
            except Exception as e:
                print(f"[Post-processing] Save failed: {e}")

            return True

        except KeyboardInterrupt:
            print("\nExperiment interrupted by user")
            return False
        except Exception as e:
            print(f"Error occurred during experiment: {e}")
            traceback.print_exc()
            return False
        finally:
            await self.cleanup()

    async def cleanup(self) -> None:
        try:
            self.running = False
            if self.task_solver:
                await self.task_solver.cleanup()
        except Exception as e:
            print(f"Error occurred while cleaning up resources: {e}")

async def run_single_case_lifecycle(
        run_dir: Path,
        task_id: str,
        solver_kwargs: Dict[str, Any] = None,
        enable_replanning: bool = False,
        enable_render: bool = False,
        replay_config: Optional[Dict] = None,  # Replay config passed in.
) -> Dict[str, Any]:
    """
    Run the complete lifecycle of a single case:
    Reset -> Load Dataset -> Init Context -> Init HITL -> Init Solver -> Run -> Cleanup
    """
    t0 = time.time()
    
    # Robust task_id parsing
    if ':' in task_id:
        type_name, scenario_id, goal_id = task_id.split(':')
    elif '/' in task_id:
        type_name, scenario_id, goal_id = task_id.split('/')
    else:
        import re
        # Try to parse format like: cybertown_scenario_1_g_0
        match = re.match(r"(.*)_(scenario_.*)_(g_.*)", task_id)
        if match:
            type_name, scenario_id, goal_id = match.groups()
        else:
            # Fallback defaults if parsing fails
            type_name, scenario_id, goal_id = "cybertown", task_id, "unknown"

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    set_default_dump_dir(run_dir)

    # 1. Environment reset.
    await reset_global_environment()
    
    # HITL communicator and manager, initialized after environment reset.
    hitl_communicator: Optional[UnifiedCommunicator] = None
    hitl_manager: Optional[HITLManager] = None
    
    # 2. Handle replay logic.
    replay_trace = None
    replay_cfg = getattr(config, "replay_mode", {}) or {}
    if replay_cfg.get("enabled"):
        replay_trace = find_matching_replay_trace(task_id=task_id, project_root=project_root)

    # If replay is enabled but no trace is found, report an error and skip.
    if replay_cfg.get("enabled") and replay_trace is None:
        msg = (
            f"[replay] No matching replay data found (type={type_name}, sce={scenario_id}, goal={goal_id}), "
            f"skipping this experiment."
        )
        print(msg)
        return {
            "ok": False,
            "elapsed_sec": '0.0',
            "run_dir": str(run_dir),
            "type_name": str(type_name),
            "scenario_id": str(scenario_id),
            "goal_id": str(goal_id),
            "goal_type": None,
            "metrics": {},
            "error": msg,
        }

    # 3. Load data, choosing the loading path by platform type.
    # Get platform type from config, defaulting to semantic.
    platform_type_str = getattr(config, "platform_type", "semantic")
    
    if platform_type_str == "unreal":
        # Unreal platform: load only goal data; scene graph comes from UE5 in real time.
        goal_data = load_unreal_goal(goal_id=goal_id, type_name=type_name)
        if not goal_data:
            return {
                "ok": False, 
                "error": f"Goal not found: {goal_id}", 
                "run_dir": str(run_dir),
                "type_name": type_name,
                "scenario_id": scenario_id,
                "goal_id": goal_id,
            }
        
        # Extract information from normalized goal data.
        raw_goal = goal_data.get("goal_details", {})
        instruction = goal_data.get("instruction", "")
        scene_graph = {}  # Unreal gets the scene graph from UE5, so this is empty here.
        full_data = {
            "instruction": instruction,
            "goal_details": goal_data,
        }
    else:
        # Semantic platform: load the full task.
        dataset = DatasetLoader(
            local_path=str(resolve_dataset_root()),
            platform="semantic",
            use_local=True,
        )
        if not dataset:
            return {"ok": False, "error": "Dataset load failed", "run_dir": str(run_dir)}

        # Get task data with lazy=True to save memory.
        _full_data = dataset.get_task(task_id=task_id,
                                     include_goal=True,
                                     include_scenario=True,
                                     include_prompt=False,
                                     lazy=True)
        full_data = copy.deepcopy(_full_data)
        if not full_data:
            return {"ok": False, "error": f"Task not found: {task_id}", "run_dir": str(run_dir)}

        # Extract goal and scene graph from full data.
        raw_goal = full_data.get("goal_details", {}).get("goal_details", {})
        scene_graph = full_data.get("scene_graph", {})
    
    # 4. Initialize context.
    processed_goal = {
        "id": raw_goal.get("goal_id"),
        "goal_type": raw_goal.get("goal_type"),
        "goal_determinacy": raw_goal.get("goal_determinacy", "open"),
        "description": full_data.get("instruction") or raw_goal.get("description"),
        "success_condition": raw_goal.get("success_condition"),
        "context": raw_goal.get("core_params", {})
    }

    # Get platform type, already determined during data loading.
    platform_type = PlatformType.UNREAL if platform_type_str == "unreal" else PlatformType.SEMANTIC
    
    # Get platform-specific config.
    platform_kwargs = {}
    hitl_communicator = None  # Initialize as None to avoid NameError.
    hitl_manager = None
    
    # 6. Initialize HITL.
    try:
        hitl_config = getattr(config, "human_in_loop", {}) or {}
        unreal_config = getattr(config, "unreal_platform", {}) or {}
        if hitl_config.get("enabled", False):
            print("[HITL] Initializing Human-in-the-Loop system (before platform)...")
            
            # Create unified communicator, replacing the old BidirectionalCommunicator.
            hitl_communicator = UnifiedCommunicator(
                unreal_url=unreal_config.get("base_url", "http://localhost:8080"),
                server_port=hitl_config.get("server_port", 8081),
                timeout=hitl_config.get("timeout", 30.0),
                max_retries=hitl_config.get("retry_count", 3),
                retry_delay=hitl_config.get("retry_delay", 1.0),
                hitl_enabled=True,
            )
            
            # Initialize HITL manager.
            hitl_manager = get_hitl_manager()
            hitl_manager.initialize(
                config={"human_in_loop": hitl_config},
                communicator=hitl_communicator
            )
            
            print(f"[HITL] System initialized: enabled={hitl_manager.is_enabled}, "
                  f"instruction={hitl_manager.is_instruction_enabled}, "
                  f"review={hitl_manager.is_review_enabled}, "
                  f"decision={hitl_manager.is_decision_enabled}")
        else:
            print("[HITL] Human-in-the-Loop system is disabled")
    except Exception as e:
        print(f"[HITL] Failed to initialize HITL system: {e}")
        traceback.print_exc()
    
    # Configure platform parameters.
    if platform_type == PlatformType.UNREAL:
        unreal_config = getattr(config, "unreal_platform", {}) or {}
        hitl_config = getattr(config, "human_in_loop", {}) or {}
        platform_kwargs = {
            "base_url": unreal_config.get("base_url", "http://localhost:8080"),
            "server_port": hitl_config.get("server_port", 8081),
            "timeout": unreal_config.get("timeout", 30.0),
            "polling_interval": unreal_config.get("polling_interval", 0.5),
            "shared_communicator": hitl_communicator,
            "hitl_enabled": hitl_config.get("enabled", False),
        }

    await initialize_platform(
        platform_type=platform_type,
        initial_nodes=scene_graph.get("nodes", []),
        initial_edges=scene_graph.get("edges", []),
        initial_goal=processed_goal,
        **platform_kwargs
    )

    # Save snapshot.
    snapshot = config.make_run_input_snapshot(
        mode_label="default", type_name=type_name, scenario_id=scenario_id, goal_id=goal_id
    )
    dump_var("run_input_default", snapshot)

    # 5. Start components.
    await start_global_event_bus()
    _ = get_global_task_monitor()

    # 7. Run solver.
    solver_kwargs = solver_kwargs or {}
    solver_kwargs['run_dir'] = run_dir
    solver_kwargs['replay_trace'] = replay_trace
    solver_kwargs['enable_replanning'] = enable_replanning
    
    solver = create_solver(**solver_kwargs)

    # Execution logic.
    runner = ExperimentRunner()
    ok = await runner.run_experiment(
        enable_render=enable_render,
        injected_solver=solver
    )

    # 8. Cleanup.
    await solver.cleanup()
    
    # Clean up HITL resources.
    if hitl_communicator:
        try:
            await hitl_communicator.close()
            print("[HITL] Communication resources cleaned up")
        except Exception as e:
            print(f"[HITL] Error cleaning up communicator: {e}")
    
    if hitl_manager:
        try:
            hitl_manager.reset()
            HITLManager.reset_instance()
            print("[HITL] Manager reset")
        except Exception as e:
            print(f"[HITL] Error resetting manager: {e}")
    
    await reset_global_environment()

    dt = time.time() - t0
    
    # Offline token statistics parsed from log.md without affecting runtime performance.
    token_stats = {}
    if token_stats_enabled():
        try:
            token_stats = compute_token_stats(run_dir / "log.md")
        except Exception:
            pass
    
    return {
        "ok": ok,
        "elapsed_sec": round(dt, 3),
        "run_dir": str(run_dir),
        "type_name": type_name,
        "scenario_id": scenario_id,
        "goal_id": goal_id,
        "goal_type": processed_goal.get("goal_type"),
        "metrics": solver.get_metrics(),
        "token_stats": token_stats,
    }
