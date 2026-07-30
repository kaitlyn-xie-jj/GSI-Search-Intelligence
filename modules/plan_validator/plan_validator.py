#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plan validator module
Provides fine-grained reward feedback signals
Reuses existing static and dynamic validation logic

Main features:
1. Static validation:
    - Reuses validate_complete for format, syntax, dependency checks
    - Task allocation validation, checks if plan can be assigned to available robots
2. Dynamic validation: Reuses UnifiedTaskSolver for actual execution validation
3. Configurable feedback: Controls validator enablement and reward values via config file
"""

import asyncio
import copy
import gc
import json
import os
import re
import sys
import time
import traceback
import logging
import uuid
from functools import lru_cache
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from modules.task_solver.llm_framework.core.parser import parse_text
from modules.task_solver.unified_task_solver import UnifiedTaskSolver
from modules.platform.platform_factory import (
    initialize_platform,
    get_scene_graph,
    cleanup_platform,
    reset_platform,
    PlatformType
)
from modules.events import start_global_event_bus, stop_global_event_bus, set_global_event_bus
from modules.monitor.task_monitor import stop_task_monitoring, reset_task_monitoring
from modules.dataset_loader.loader import DatasetLoader
from modules.utils.system.root import PathManager
from modules.task_solver.llm_framework.file import Logger
from modules.task_solver.sgi_planner.utils.validate_plan import validate_complete
from modules.plan_validator.replan_state_store import load_replan_state

# -----------------------------------------------------------------------------
# Global resource cache (memory sharing with Fork mode)
# -----------------------------------------------------------------------------
_GLOBAL_DATASET_LOADER = None


def _get_or_init_global_loader():
    """Get or initialize global dataset loader (singleton to reuse index)"""
    global _GLOBAL_DATASET_LOADER
    if _GLOBAL_DATASET_LOADER is None:
        _GLOBAL_DATASET_LOADER = DatasetLoader(
            local_path=str(Path(__file__).resolve().parents[2] / "dataset"),
            use_local=True,
            platform="semantic",
        )
    return _GLOBAL_DATASET_LOADER


def load_scene_graph_data(file_path):
    """
    Read scene_graph.json file and extract nodes, edges, goal.
    Uses deepcopy to ensure returned data is fully independent.

    Args:
        file_path (str): Full path to json file

    Returns:
        dict: Dictionary containing 'nodes', 'edges', 'goal'. Returns None if read fails.
    """
    if not os.path.exists(file_path):
        print(f"❌ Error: File does not exist: {file_path}")
        return None

    try:
        # 1. Read JSON file
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 2. Extract and deepcopy required fields
        result = {
            "nodes": copy.deepcopy(data.get("nodes", [])),
            "edges": copy.deepcopy(data.get("edges", [])),
            # Default to empty dict if goal doesn't exist
            "goal": copy.deepcopy(data.get("goal", {}))
        }

        # Print brief log
        node_count = len(result['nodes'])
        edge_count = len(result['edges'])
        print(f"✅ Successfully loaded data: .../{os.path.basename(os.path.dirname(file_path))}/{os.path.basename(file_path)}")
        print(f"   -> Stats: {node_count} Nodes, {edge_count} Edges")

        return result

    except json.JSONDecodeError:
        print(f"❌ Error: JSON parsing failed: {file_path}")
        return None
    except Exception as e:
        print(f"❌ Unknown error while reading file: {e}")
        return None


# -----------------------------------------------------------------------------


def _build_first_plan_initialization(task_id: str, loader: DatasetLoader) -> Dict[str, Any]:
    """
    Build scene graph and goal for first-plan initialization.

    This function only prepares data, doesn't start platform; caller can choose to cache return value.
    """

    task_id = re.sub(r"^scenario", "cybertown_scenario", task_id)
    temp = re.sub(r"scenario_\d+", "scenario_1", task_id)
    tasks = loader.get_task(
        task_id=temp,
        include_scenario=True,
        include_goal=True,
        include_prompt=False,
        lazy=True,
    )
    if not tasks:
        raise RuntimeError(f"Load failed: task not found (task_id={temp})")

    raw_goal = tasks.get("goal_details", {}).get("goal_details", {})
    match = re.search(r"(scenario_\d+)", task_id)
    if not match:
        raise RuntimeError(f"Cannot extract scenario_id from task_id '{task_id}'")
    scenario_id = match.group(1)
    if scenario_id != "scenario_1":
        scenario_path = project_root / "dataset" / "scenarios" / "cybertown" / scenario_id / "scene_graph.json"
        scene_graph = load_scene_graph_data(str(scenario_path))
        processed_goal = scene_graph["goal"]
    else:
        scene_graph = tasks.get("scene_graph", {})
        processed_goal = {
            "id": raw_goal.get("goal_id"),
            "goal_type": raw_goal.get("goal_type"),
            "goal_determinacy": raw_goal.get("goal_determinacy", "open"),
            "description": tasks.get("instruction") or raw_goal.get("description"),
            "success_condition": raw_goal.get("success_condition"),
            "context": raw_goal.get("core_params", {}),
        }

    if not scene_graph:
        raise RuntimeError(f"Load failed: scene data missing (task_id={task_id})")
    return {
        "scene_graph": scene_graph,
        "processed_goal": processed_goal,
    }


@lru_cache(maxsize=4096)
def _load_first_plan_initialization(task_id: str) -> Dict[str, Any]:
    """
    Cache first-plan initialization data to avoid repeated DatasetLoader task lookups.

    Return object is read-only; caller must copy fields when modification needed.
    """

    return _build_first_plan_initialization(task_id, _get_or_init_global_loader())


def _copy_initialization_payload(init_data: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Return fresh mutable initialization data for one validation run.

    DatasetLoader and _load_first_plan_initialization intentionally cache scene
    graph structures. The semantic platform mutates nodes and edges during
    execution, so every validation must receive an isolated copy.
    """

    return copy.deepcopy(init_data["scene_graph"]), copy.deepcopy(init_data["processed_goal"])


class PlanValidator:
    """Plan validator"""

    def __init__(self, workspace_root: str = None, task_id: str = None, logger: logging.Logger = None,
                 dataset=None,
                 planner_mode: str = "full",
                 state_store: str = None,
                 state_id: str = None):
        """
        Initialize plan validator

        Args:
            workspace_root: Workspace root directory
            task_id: Task ID, format "type:scenario_id:goal_id" or "type/scenario_id/goal_id"
            logger: (Ignored, will use internal Logger)
            dataset: Optional dataset instance (DatasetLoader),
                    if provided uses this instance, otherwise loads from cached index
        """
        # Load config
        self.config = self._load_config()

        # Use workspace root from config, or default if not provided.
        # Validator may run many plans concurrently through /validate_batch.
        # TANGO writes alloc_config/*.yaml and alloc_results.yaml under the
        # workspace, so sharing one directory across requests corrupts allocator
        # inputs/results. Give each validation its own child directory.
        if workspace_root is None:
            base_workspace_root = os.environ.get("GSI_PLAN_VALIDATOR_WORKSPACE_ROOT") or self.config.get(
                "default_workspace_root", "./results/plan_validation_results"
            )
            safe_task_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(task_id or "task"))[:96]
            workspace_root = str(
                Path(base_workspace_root)
                / f"{safe_task_id}_{int(time.time() * 1000)}_{os.getpid()}_{uuid.uuid4().hex[:8]}"
            )

        self.path_manager = PathManager(base_results_dir=str(workspace_root))
        self.workspace_root = workspace_root
        self.task_id = task_id
        # Use provided dataset or get from cache (lazy load)
        self.dataset = dataset
        self._initialized = False
        self.solver: Optional[UnifiedTaskSolver] = None
        self.logger = Logger(log_file_dir=str(self.workspace_root))
        self.planner_mode = self.config.get("planner_mode", "full")
        self.state_store = state_store
        self.state_id = state_id

        # Validation stats
        self.stats = {
            "total_validations": 0,
            "static_failures": 0,
            "allocation_failures": 0,
            "dynamic_failures": 0,
            "successes": 0
        }

    def _load_config(self) -> Dict[str, Any]:
        """
        Load config file

        Returns:
            Config dictionary
        """
        config_file = Path(__file__).parent.parent.parent / "config" / "plan_validator.json"

        if config_file.exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Unable to load config file {config_file}: {e}")

        # Return default config if file doesn't exist or load fails
        return {
            "enabled_validators": {
                "planning_time_syntax": True,
                "planning_time_validity": True,
                "planning_time_capability": True,
                "planning_time_dependencies": True,
                "planning_time_edge_soft": True,
                "planning_time_empty": True,
                "planning_time_allocation": True,
                "execution_time_precondition": True,
                "post_execution_skill": True,
                "post_execution_time": True,
                "task_level_success": True,
                "task_level_failure": True
            },
            "feedback_rewards": {
                "planning_time_syntax": -10.0,
                "planning_time_validity": -8.0,
                "planning_time_capability": -8.0,
                "planning_time_dependencies": -7.0,
                "planning_time_edge_soft": -2.0,
                "planning_time_empty": -5.0,
                "planning_time_allocation": -9.0,
                "execution_time_precondition": -10.0,
                "post_execution_skill": -7.0,
                "post_execution_time": 0.0,
                "task_level_success": 10.0,
                "task_level_failure": -10.0
            }
        }

    def _is_validator_enabled(self, validator_name: str) -> bool:
        """
        Check if specified validator is enabled

        Args:
            validator_name: Validator name

        Returns:
            Whether enabled
        """
        enabled_validators = self.config.get("enabled_validators", {})
        return enabled_validators.get(validator_name, True)

    def _get_reward(self, validator_name: str) -> float:
        """
        Get reward value for specified validator

        Args:
            validator_name: Validator name

        Returns:
            Reward value
        """
        feedback_rewards = self.config.get("feedback_rewards", {})
        return feedback_rewards.get(validator_name, 0.0)

    def _is_soft_fixable_edge_consistency_failure(self, results: Dict[str, Any]) -> bool:
        """
        Treat normal-edge/tbd dataflow mismatches as soft-fixable.

        These plans can often execute correctly because runtime parameter filling
        resolves the tbd value after the producer task completes. For RLVR we
        still penalize the unclear graph semantics, but we should not stop before
        dynamic validation.
        """

        hard_levels = [
            "level1_format",
            "level1_adapter",
            "level1_meta_schema",
            "level1_schema",
            "level2_skills",
            "level2_dependencies",
            "level2_acyclicity",
        ]
        if any(not results.get(level, {}).get("valid") for level in hard_levels):
            return False

        edge_result = results.get("level2_edge_consistency") or {}
        errors = edge_result.get("errors") or []
        if edge_result.get("valid") or not errors:
            return False

        for error in errors:
            if not isinstance(error, dict) or error.get("error_type") != "edge_type_mismatch":
                return False
            message = str(error.get("message") or "")
            if "type is normal" not in message or "tbd variable" not in message:
                return False
        return True

    def _edge_soft_feedbacks(self, errors: List[Any]) -> List[Dict[str, Any]]:
        reward = self._get_reward("planning_time_edge_soft")
        feedbacks = []
        for error in errors or []:
            feedback = error.copy() if isinstance(error, dict) else {"message": str(error)}
            feedback["validator_name"] = "planning_time_edge_soft"
            feedback["reward"] = reward
            feedback["soft_penalty"] = True
            if "position" not in feedback:
                feedback["position"] = -1
            feedbacks.append(feedback)
        return feedbacks

    def _processed_goal_from_replan_record(self, record: Dict[str, Any], task_id: str = None) -> Dict[str, Any]:
        """
        Recover goal from replan state record.

        Input is full record from local state store; prioritizes world_state.goal as it represents current state for replan prompt.
        Output is passed directly to semantic platform initialization.
        """

        world_goal = copy.deepcopy((record.get("world_state") or {}).get("goal") or {})
        meta_goal = copy.deepcopy((record.get("meta") or {}).get("goal") or {})
        case = record.get("case") or {}
        if world_goal:
            return world_goal
        if meta_goal:
            return meta_goal
        return {
            "id": case.get("goal_id") or task_id,
            "goal_type": case.get("goal_type"),
            "goal_determinacy": "open",
            "description": "",
            "success_condition": None,
            "context": {},
        }

    async def _initialize_environment(self, task_id: str = None):
        """
        Initialize validation environment (uses global cache)
        """
        if self._initialized:
            return

        try:
            timing = getattr(self, "_current_timing", None)
            state_record = load_replan_state(self.state_store, self.state_id)
            if state_record is not None:
                if timing is not None:
                    timing["state_load_ms"] = timing.get("state_load_ms", 0.0)
                world_state = copy.deepcopy(state_record.get("world_state") or {})
                scene_graph = {
                    "nodes": world_state.get("nodes", []),
                    "edges": world_state.get("edges", []),
                }
                processed_goal = self._processed_goal_from_replan_record(state_record, task_id)
                if not scene_graph["nodes"]:
                    raise RuntimeError(
                        f"replan state has no nodes: state_store={self.state_store} state_id={self.state_id}"
                    )
                await initialize_platform(
                    platform_type=PlatformType.SEMANTIC,
                    initial_nodes=scene_graph.get("nodes", []),
                    initial_edges=scene_graph.get("edges", []),
                    initial_goal=processed_goal,
                )
                await start_global_event_bus()
                self._initialized = True
                return

            if self.dataset:
                init_data = _build_first_plan_initialization(task_id, self.dataset)
            else:
                init_data = _load_first_plan_initialization(task_id)
            scene_graph, processed_goal = _copy_initialization_payload(init_data)

            # Initialize platform (using platform factory)
            await initialize_platform(
                platform_type=PlatformType.SEMANTIC,
                initial_nodes=scene_graph.get("nodes", []),
                initial_edges=scene_graph.get("edges", []),
                initial_goal=processed_goal,
            )

            # Start event bus
            await start_global_event_bus()

            self._initialized = True
            # self.logger.info(f"Environment initialized: Task={task_id}, Nodes={len(scene_graph.get('nodes', []))}, Goal={processed_goal.get('id')}")

        except Exception as e:
            raise RuntimeError(f"Environment initialization failed: {e}")

    def _parse_plan(self, plan_str: str) -> Dict[str, Any]:
        """
        Parse plan string

        Args:
            plan_str: Plan string

        Returns:
            Parsed plan data
        """
        try:
            # Try parsing as JSON
            return json.loads(plan_str)
        except json.JSONDecodeError as e:
            # If not JSON format, raise specific error
            raise ValueError(f"Invalid JSON: Provide negative feedback at character index {e.pos}. Error details: {str(e)}")

    async def _validate_static(self, plan_str: str) -> Tuple[bool, List[Dict[str, Any]], Optional[Dict]]:
        """
        Perform static validation using validate_complete and convert to feedback format

        Args:
            plan_str: Plan string

        Returns:
            (is_valid, feedback_list, parsed_plan_data)
        """
        from modules.task_solver.sgi_planner.prompt import robot_skill_library

        feedbacks = []
        def _append_feedbacks(errors, validator_name: str, reward_key: str = None):
            reward = self._get_reward(reward_key or validator_name)
            for error in errors or []:
                feedback = error.copy() if isinstance(error, dict) else {"message": str(error)}
                feedback['validator_name'] = validator_name
                feedback['reward'] = reward
                if 'position' not in feedback:
                    feedback['position'] = -1
                feedbacks.append(feedback)

        try:
            plan_str = parse_text(plan_str, "json", all_matches=False)
        except Exception as e:
            plan_str = plan_str
        results = validate_complete(plan_str, robot_skill_library, enable_fix=False)

        soft_edge_feedbacks: List[Dict[str, Any]] = []
        if not results["overall_valid"] and self._is_soft_fixable_edge_consistency_failure(results):
            fixed_results = validate_complete(plan_str, robot_skill_library, enable_fix=True)
            if fixed_results["overall_valid"] and fixed_results.get("fixed_data"):
                soft_edge_feedbacks = self._edge_soft_feedbacks(
                    results.get("level2_edge_consistency", {}).get("errors", [])
                )
                results = fixed_results

        # Convert validation results to feedback format
        if not results['overall_valid']:
            # Format validation failed
            if not results['level1_format']['valid'] and self._is_validator_enabled("planning_time_syntax"):
                _append_feedbacks(results['level1_format']['errors'], 'planning_time_syntax')

            if not results['level1_adapter']['valid'] and self._is_validator_enabled("planning_time_syntax"):
                _append_feedbacks(results['level1_adapter']['errors'], 'planning_time_syntax')

            if not results['level1_meta_schema']['valid'] and self._is_validator_enabled("planning_time_syntax"):
                _append_feedbacks(results['level1_meta_schema']['errors'], 'planning_time_syntax')

            # Schema validation failed
            if not results['level1_schema']['valid'] and self._is_validator_enabled("planning_time_syntax"):
                _append_feedbacks(results['level1_schema']['errors'], 'planning_time_syntax')

            # Skill validity validation failed - now includes position info
            if not results['level2_skills']['valid'] and self._is_validator_enabled("planning_time_validity"):
                for error in results['level2_skills']['errors']:
                    feedback = error.copy() if isinstance(error, dict) else {}
                    feedback['validator_name'] = 'planning_time_validity'
                    feedback['reward'] = self._get_reward("planning_time_validity")
                    if 'position' not in feedback:
                        feedback['position'] = -1
                    feedbacks.append(feedback)

            # Dependency validation failed (dangling deps) - now includes position info
            if not results['level2_dependencies']['valid'] and self._is_validator_enabled("planning_time_dependencies"):
                for error in results['level2_dependencies']['errors']:
                    feedback = error.copy() if isinstance(error, dict) else {}
                    feedback['validator_name'] = 'planning_time_dependencies'
                    feedback['reward'] = self._get_reward("planning_time_dependencies")
                    if 'position' not in feedback:
                        feedback['position'] = -1
                    feedbacks.append(feedback)

            # Cyclic dependency validation failed
            if not results['level2_acyclicity']['valid'] and self._is_validator_enabled("planning_time_dependencies"):
                for error in results['level2_acyclicity']['errors']:
                    feedback = error.copy() if isinstance(error, dict) else {}
                    feedback['validator_name'] = 'planning_time_dependencies'
                    feedback['reward'] = self._get_reward("planning_time_dependencies")
                    if 'position' not in feedback:
                        feedback['position'] = -1
                    feedbacks.append(feedback)

            # Edge type and node content consistency validation failed
            if not results['level2_edge_consistency']['valid'] and self._is_validator_enabled("planning_time_dependencies"):
                for error in results['level2_edge_consistency']['errors']:
                    feedback = error.copy() if isinstance(error, dict) else {}
                    feedback['validator_name'] = 'planning_time_dependencies'
                    feedback['reward'] = self._get_reward("planning_time_dependencies")
                    if 'position' not in feedback:
                        feedback['position'] = -1
                    feedbacks.append(feedback)

            return False, feedbacks, None

        # Validation passed, parse plan data
        try:
            if results['fixed_data']:
                plan_data = json.loads(results['fixed_data'])
            else:
                plan_data = json.loads(plan_str)
        except:
            plan_data = None

        # Empty plan check
        if plan_data and self._is_validator_enabled("planning_time_empty"):
            atomic_tasks = plan_data.get('atomic_tasks')
            task_graph = plan_data.get('task_graph')
            task_graph_nodes = task_graph.get('nodes', []) if isinstance(task_graph, dict) else task_graph
            if (atomic_tasks is not None and not atomic_tasks) or (task_graph is not None and not task_graph_nodes):
                feedbacks.append({
                    "validator_name": "planning_time_empty",
                    "position": -1,
                    "reward": self._get_reward("planning_time_empty")
                })
                return False, feedbacks, None

        feedbacks.extend(soft_edge_feedbacks)
        return True, feedbacks, plan_data

    async def validate_plan(self, plan_str: str,
                            task_id: str = None,
                            ) -> Dict[str, Any]:
        """
        Validate single plan and calculate reward (supports full mode loop validation)
        Args:
            plan_str: Plan string
            task_id: Task ID, format "type:scenario_id:goal_id" or "type_scenario_id_goal_id"
        """
        import time
        from modules.task_solver.sgi_planner.task_graph_manager import TaskGraphManager
        from modules.task_solver.sgi_planner.feedback_processor import ReplanningStrategy

        timing_stats = {
            "static_ms": 0.0,
            "state_load_ms": 0.0,
            "env_init_ms": 0.0,
            "solver_init_ms": 0.0,
            "allocation_ms": 0.0,
            "execution_ms": 0.0,
            "cleanup_ms": 0.0,
        }
        self._current_timing = timing_stats
        self.stats["total_validations"] += 1
        start_time = datetime.now()

        # Internal function for unified return format
        def _finalize_result(is_valid: bool, reward: float, feedbacks: List[Dict], details: Optional[Dict] = None,
                             error_msg: Optional[str] = None):
            duration = (datetime.now() - start_time).total_seconds()
            total_reward = sum(f.get('reward', 0.0) for f in feedbacks)
            total_reward = max(total_reward, -10.0)
            token_rewards, error_positions = self._generate_token_rewards(
                plan_str, feedbacks, total_reward, is_valid
            )

            result = {
                "success": True,  # Whether validator ran successfully
                "valid": is_valid,
                "reward": total_reward,
                "token_rewards": token_rewards,
                "error_positions": error_positions,
                "feedbacks": feedbacks,
                "plan": plan_str,
                "validation_time": duration,
                "timestamp": start_time.isoformat(),
                "stats": dict(self.stats),
                "details": details or {}
            }
            result["details"].setdefault("timing", dict(timing_stats))
            if error_msg:
                result["error"] = error_msg
                result["success"] = False  # Validator itself errored

            return result

        try:
            # 1. Static format validation first: invalid JSON/schema shouldn't trigger platform and solver init.
            t0 = time.time()
            static_valid, static_feedbacks, plan_data = await self._validate_static(plan_str)
            timing_stats["static_ms"] += (time.time() - t0) * 1000.0
            if not static_valid:
                if not static_feedbacks:
                    static_feedbacks.append({
                        "validator_name": "planning_time_syntax",
                        "position": -1,
                        "reward": self._get_reward("planning_time_syntax"),
                        "message": "Static validation failed"
                    })
                self.stats["static_failures"] += 1
                self.logger.warning(f"Static validation failed")
                return _finalize_result(False, 0.0, static_feedbacks)

            # 2. Initialize environment
            t0 = time.time()
            await self._initialize_environment(task_id)
            timing_stats["env_init_ms"] += (time.time() - t0) * 1000.0

            # 3. Create Solver
            t0 = time.time()
            self.solver = UnifiedTaskSolver(
                solver_type="sgi",
                logger=self.logger,
                path_manager=self.path_manager,
                planner_mode=self.planner_mode,  # Use member variable
                enable_replanning=True,
                enable_render=False,
            )
            await self.solver.initialize()
            await self.solver._init_event_bus()
            self.solver._init_context_static()
            timing_stats["solver_init_ms"] += (time.time() - t0) * 1000.0

            # 4. Manually inject plan graph (skip LLM planning)
            try:
                self.solver.planning_layer.task_graph_manager = TaskGraphManager(
                    plan_data,
                    self.solver.logger,
                    self.solver.context,
                    self.solver.world_model_layer.world_model_manager,
                    self.solver.planner_mode
                )
            except Exception as e:
                self.stats["static_failures"] += 1
                reward = self._get_reward("planning_time_syntax")
                static_feedbacks.append({"position": -1, "reward": reward, "message": f"Task graph init failed: {e}"})
                return _finalize_result(False, 0.0, static_feedbacks, error_msg=f"Graph init failed: {e}")

            # 5. Initialize execution loop
            self.solver.planning_layer.feedback_processor.reset()
            self.solver.planning_layer.feedback_processor.replanning_strategy = ReplanningStrategy.NONE

            loop_count = 0
            max_loops = 50  # Prevent infinite loop
            total_feedbacks = list(static_feedbacks)  # Collect all feedbacks
            status = None
            first_batch = True

            # 6. Start execution loop: reuse UnifiedTaskSolver's execution and feedback evaluation path.
            #    Current validator only validates given plan itself; if FULL replan needed, mark as failed,
            #    but allow PARTIAL/local recovery consistent with normal execution.
            while loop_count <= max_loops:
                loop_count += 1

                fp = self.solver.planning_layer.feedback_processor
                if fp.replanning_strategy == ReplanningStrategy.FULL:
                    self.stats["dynamic_failures"] += 1
                    reward = self._get_reward("task_level_failure")
                    total_feedbacks.append({"position": -1, "reward": reward, "message": "Plan requires full replan, validation failed"})
                    return _finalize_result(False, 0.0, total_feedbacks, details={"requires_full_replan": True})

                # b. Get and allocate next batch
                if first_batch:
                    self.solver._refresh_context()
                    t0 = time.time()
                    allocation_plan = await self.solver.planning_layer._get_next_executable_batch()
                    timing_stats["allocation_ms"] += (time.time() - t0) * 1000.0
                    first_batch = False
                else:
                    feedback_data = fp.prepare_feedback_data(
                        task_graph_manager=self.solver.planning_layer.task_graph_manager
                    )
                    self.solver._refresh_context(feedback_data=feedback_data)
                    t0 = time.time()
                    allocation_plan = await self.solver.planning_layer.generate_plan()
                    timing_stats["allocation_ms"] += (time.time() - t0) * 1000.0

                # c. Check batch status
                if allocation_plan is None:
                    self.logger.warning("Task allocation failed.")
                    self.stats["allocation_failures"] += 1
                    reward = self._get_reward("planning_time_allocation")
                    total_feedbacks.append({"position": -1, "reward": reward, "message": "Task allocation failed"})
                    allocation_debug = (
                        getattr(self.solver, "context", None)._generated_text.get("allocation_debug", {})
                        if getattr(self.solver, "context", None) is not None
                        else {}
                    )
                    batch_debug = (
                        getattr(self.solver, "context", None)._generated_text.get("batch_debug", {})
                        if getattr(self.solver, "context", None) is not None
                        else {}
                    )
                    return _finalize_result(
                        False,
                        0.0,
                        total_feedbacks,
                        details={
                            "allocation_failed": True,
                            "workspace_root": str(self.workspace_root),
                            "allocation_debug": allocation_debug,
                            "batch_debug": batch_debug,
                        },
                    )

                if allocation_plan.get("status") == "mission_complete":
                    break  # Plan graph execution complete

                if not allocation_plan:
                    if self.solver.world_model_layer.is_goal_completed():
                        self.stats["successes"] += 1
                        reward = self._get_reward("task_level_success")
                        total_feedbacks.append({"position": -1, "reward": reward, "message": "Task succeeded"})
                        return _finalize_result(True, 0.0, total_feedbacks, details={"goal_completed": True})
                    if self.solver.planning_layer.task_graph_manager.has_pending_tasks():
                        self.logger.warning("Plan deadlock: pending tasks exist but none are ready.")
                        self.stats["static_failures"] += 1
                        reward = self._get_reward("planning_time_dependencies")
                        total_feedbacks.append({"position": -1, "reward": reward, "message": "Plan graph deadlock"})
                        return _finalize_result(False, 0.0, total_feedbacks)
                    else:
                        break  # No tasks and no pending -> another success case

                # d. Semantic processing
                skills_by_timestep = await self.solver._process_plan(allocation_plan)
                if not skills_by_timestep:
                    self.logger.warning("Plan batch skill conversion failed.")
                    self.stats["dynamic_failures"] += 1
                    reward = self._get_reward("task_level_failure")
                    total_feedbacks.append({"position": -1, "reward": reward, "message": "Plan batch skill conversion failed"})
                    return _finalize_result(False, 0.0, total_feedbacks)

                # e. Execute
                t0 = time.time()
                execution_result = await self.solver._execute_skills(skills_by_timestep)
                timing_stats["execution_ms"] += (time.time() - t0) * 1000.0

                status = execution_result.get('status')

                # f. Reuse normal solver's feedback evaluation path:
                #    - First handle new situation events (ROBOT_FAULT etc.)
                #    - Update world model / task graph
                #    - Decide NONE / PARTIAL / FULL
                termination_result = await self.solver._evaluate_execution_result_dispatch(execution_result)
                if termination_result is True:
                    self.stats["successes"] += 1
                    reward = self._get_reward("task_level_success")
                    total_feedbacks.append({"position": -1, "reward": reward, "message": "Task succeeded"})
                    return _finalize_result(True, 0.0, total_feedbacks, details={"goal_completed": True})
                if termination_result is False:
                    self.stats["dynamic_failures"] += 1
                    reward = self._get_reward("task_level_failure")
                    total_feedbacks.append({"position": -1, "reward": reward, "message": "Plan execution failed"})
                    return _finalize_result(False, 0.0, total_feedbacks, details={"goal_completed": False})

                fp = self.solver.planning_layer.feedback_processor
                if fp.replanning_strategy == ReplanningStrategy.FULL:
                    self.stats["dynamic_failures"] += 1
                    reward = self._get_reward("task_level_failure")
                    total_feedbacks.append({"position": -1, "reward": reward, "message": "Plan requires full replan, validation failed"})
                    return _finalize_result(False, 0.0, total_feedbacks, details={"requires_full_replan": True})
                self.solver._handle_replan_decision(fp)

            # 7. Loop ended, calculate final result
            if loop_count > max_loops:
                # self.logger.error("Validation timeout: too many loops.")
                self.stats["dynamic_failures"] += 1
                reward = self._get_reward("task_level_failure")
                total_feedbacks.append({"position": -1, "reward": reward, "message": "Validation timeout"})
                return _finalize_result(False, 0.0, total_feedbacks)

            # Check final goal
            final_goal_completed = self.solver.world_model_layer.is_goal_completed()
            if status == "completed" and final_goal_completed:
                self.stats["successes"] += 1
                reward = self._get_reward("task_level_success")
                total_feedbacks.append({"position": -1, "reward": reward, "message": "Task succeeded"})
                return _finalize_result(True, 0.0, total_feedbacks, details={"goal_completed": True})
            else:
                # self.logger.warning("Plan execution completed but final goal not achieved.")
                self.stats["dynamic_failures"] += 1
                # (If plan execution completes but goal not achieved, this is still plan failure)
                reward = self._get_reward("task_level_failure")
                total_feedbacks.append({"position": -1, "reward": reward, "message": "Plan execution completed but goal not achieved"})
                return _finalize_result(False, 0.0, total_feedbacks, details={"goal_completed": False})

        except ValueError as e:
            # JSON parsing error
            duration = (datetime.now() - start_time).total_seconds()
            self.logger.error(f"JSON parsing error: {e} [elapsed={duration:.2f}s]")
            self.stats["static_failures"] += 1
            reward = self._get_reward("planning_time_syntax")
            feedbacks = [{"position": -1, "reward": reward}]
            return _finalize_result(False, 0.0, feedbacks, error_msg=str(e))

        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            self.logger.error(f"Exception during validation [elapsed={duration:.2f}s]: {e}\n{traceback.format_exc()}")
            reward = self._get_reward("task_level_failure")
            feedbacks = [{"position": -1, "reward": reward}]
            return _finalize_result(False, 0.0, feedbacks, error_msg=str(e))

        finally:
            t0 = time.time()
            try:
                if self.solver:
                    try:
                        await self.solver.cleanup()
                    except Exception as e:
                        self.logger.error(f"Solver cleanup failed: {e}")
                if self._initialized or self.solver:
                    await self._cleanup_environment()
            finally:
                timing_stats["cleanup_ms"] += (time.time() - t0) * 1000.0
                self._current_timing = None

    async def _cleanup_environment(self):
        """
        Clean up and reset global state to prevent leaks in process pool
        """

        try:
            # 1. Stop event bus
            await stop_global_event_bus()
            set_global_event_bus(None)

            # 2. Clean up platform
            await cleanup_platform()
            reset_platform()

            # 3. Stop task monitoring
            await stop_task_monitoring()
            reset_task_monitoring()

            if self.solver:
                del self.solver
            self.solver = None

            gc.collect()


        except Exception as e:
            self.logger.error(f"Exception during environment cleanup: {e}")

        finally:
            self._initialized = False
            self.solver = None

    def _generate_token_rewards(
            self,
            plan_str: str,
            feedbacks: List[Dict[str, Any]],
            overall_reward: float,
            is_valid: bool
    ) -> Tuple[List[float], List[int]]:
        """
        Generate token-level reward list and error position list

        Args:
            plan_str: Plan string
            feedbacks: Feedback list, each feedback contains position and reward
                      position can be:
                      - -1: Global error/success
                      - int: Single character position (converted to [pos, pos])
                      - [low, high]: Position range
            overall_reward: Overall reward
            is_valid: Whether plan is valid

        Returns:
            (token_rewards, error_positions)
            - token_rewards: Reward for each character position
            - error_positions: Error character position list (flattened position list)
        """
        plan_length = len(plan_str)

        # Initialize: all token rewards to 0
        token_rewards = [0.0] * plan_length
        error_positions = []

        # Process each feedback, assign to corresponding token positions
        for feedback in feedbacks:
            position = feedback.get('position', -1)
            reward = feedback.get('reward', 0.0)

            # Normalize position to [low, high] format
            if position == -1:
                # Global error/success, distribute evenly to all tokens
                reward_per_token = reward / plan_length if plan_length > 0 else reward
                for i in range(plan_length):
                    token_rewards[i] += reward_per_token

            elif isinstance(position, list) and len(position) == 2:
                # Position range [low, high] - keep original position data, no expansion
                pos_low, pos_high = position

                # Use original positions directly, no boundary adjustment
                if pos_low <= pos_high and 0 <= pos_low < plan_length and 0 <= pos_high < plan_length:
                    # Calculate reward distribution within range
                    range_length = pos_high - pos_low + 1
                    reward_per_token = reward / range_length if range_length > 0 else reward

                    # Distribute reward to each token in range
                    for i in range(pos_low, pos_high + 1):
                        token_rewards[i] += reward_per_token

                    # Record error positions (only negative rewards)
                    if reward < 0:
                        for i in range(pos_low, pos_high + 1):
                            error_positions.append(i)

            elif isinstance(position, int) and position >= 0:
                # Single position, convert to [pos, pos] range
                if 0 <= position < plan_length:
                    token_rewards[position] += reward

                    # Only negative rewards recorded as error positions
                    if reward < 0:
                        error_positions.append(position)

        return token_rewards, error_positions


async def validate_single_plan_async(plan_str: str,
                                     task_id: str = None,
                                     state_store: str = None,
                                     state_id: str = None) -> Dict[str, Any]:
    """
    Async convenience function for validating single plan

    Args:
        plan_str: LLM output plan string
        task_id: Task ID, format "type:scenario_id:goal_id" or "type/scenario_id/goal_id"
        state_store: Optional local replan state store name
        state_id: Optional replan state record ID

    Returns:
        Validation result
    """
    validator = PlanValidator(task_id=task_id, state_store=state_store, state_id=state_id)
    return await validator.validate_plan(plan_str, task_id)


def validate_single_plan(plan_str: str,
                         task_id: str = None,
                         state_store: str = None,
                         state_id: str = None) -> Dict[str, Any]:
    """
    Convenience function for validating single plan

    Args:
        plan_str: LLM output plan string
        task_id: Task ID, format "type:scenario_id:goal_id" or "type/scenario_id/goal_id"
        state_store: Optional local replan state store name
        state_id: Optional replan state record ID

    Returns:
        Validation result
    """
    return asyncio.run(validate_single_plan_async(plan_str, task_id, state_store, state_id))


if __name__ == "__main__":

    # Example LLM raw output
    plan_input = """
    {
  "meta": {
    "reasoning": "The plan addresses the user's goal to guide a 'person in black with a suitcase' to a designated area. First, a UAV performs an aerial search across 'cybertown' to locate the target, producing their location if found. Upon successful detection, a Quadruped navigates to the target's location and then proceeds to guide them to the specified 'PointRadius_MarkedPoint_40m'. This strategy prioritizes efficient target localization using a UAV before engaging a Quadruped for the guidance task.",
    "shared_skill_groups": [
      ["T1.0", "T2.0"],
      ["T3.0", "T4.0"]
    ]
  },
  "task_graph": {
    "nodes": [
      {
        "task_id": "T1",
        "location": "current_location",
        "required_skills": ["UAV:take_off:1"]
      },
      {
        "task_id": "T2",
        "location": "cybertown",
        "required_skills": ["UAV:search<cybertown>_for<Person_BlackClothes_with_Suitcase>:1"],
        "produces": ["person_location"]
      },
      {
        "task_id": "T3",
        "location": "tbd:person_location",
        "required_skills": ["Quadruped:navigate<tbd:person_location>:1"]
      },
      {
        "task_id": "T4",
        "location": "PointRadius_MarkedPoint_40m",
        "required_skills": ["Quadruped:guide<Person_BlackClothes_with_Suitcase>_to<PointRadius_MarkedPoint_40m>:1"]
      }
    ],
    "edges": [
      "T1->T2",
      "T2->T3:person_location != null",
      "T3->T4"
    ]
  }
}""".strip()
    result = validate_single_plan(plan_input, task_id="cybertown_scenario_1_g_19874")
    print(json.dumps(result, ensure_ascii=False, indent=2))
