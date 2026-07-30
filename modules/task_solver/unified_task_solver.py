# -*- coding: utf-8 -*-
"""
UnifiedTaskSolver - unified task solving controller.
"""
import asyncio
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any, List

from modules.task_solver.llm_framework.core.context import WorkflowContext
from modules.task_solver.llm_framework.core.parser import parse_text
from modules.task_solver.llm_framework.file import Logger
from modules.task_solver.sgi_planner.planning_layer import PlanningLayer
from modules.task_solver.sgi_planner.feedback_processor import ReplanningStrategy
from modules.task_solver.world_model.world_model_layer import WorldModelLayer
from modules.task_solver.solver_context import SolverContextMixin
from modules.task_solver.solver_hitl import SolverHITLMixin
from modules.task_solver.baseline_planners.base_planner import BaselinePlanner
from modules.platform.platform_factory import get_scene_graph, create_platform_executor
from modules.platform.abstract_scene_graph import AbstractSceneGraph
from modules.platform.abstract_platform_executor import AbstractPlatformExecutor
from modules.utils.system.root import PathManager
from modules.utils.system.var_dump import dump_var, set_default_dump_dir
from modules.utils.system.logging_utils import dlog
from modules.utils.system.metrics_utils import MetricsManager
from modules.utils.trace_loader import ReplayTrace
from modules.utils.replan_recorder import ReplanDatasetRecorder, ReplanSampleCollected
from modules.config.system_config import config
from modules.monitor.task_monitor import get_global_task_monitor
from modules.hitl.hitl_manager import get_hitl_manager

logger = logging.getLogger(__name__)


class UnifiedTaskSolver(SolverContextMixin, SolverHITLMixin):
    """Unified task solving controller.

    Encapsulates shared logic for three-layer architecture initialization,
    event subscriptions, and context management. Different planning layer
    implementations are selected by solver_type.
    """

    def __init__(
        self,
        solver_type: str = "sgi",
        logger: Logger = None,
        path_manager: PathManager = None,
        planner_mode: str = "full",
        use_environment_model: bool = True,
        robot_type_list: list = ["UAV", "UGV", "Quadruped", "Humanoid"],
        enable_replan_dataset_capture: bool = False,
        enable_replanning: bool = None,
        run_dir: Optional[Path] = None,
        enable_render: bool = False,
        replay_trace: Optional[ReplayTrace] = None,
        solver_config: Optional[Dict[str, Any]] = None,
        replan_capture_options: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the unified task solver.

        Args:
            solver_type: Solver type.
            solver_config: Configuration dictionary for each method.
        """
        # Configuration parameters.
        self.solver_type = solver_type
        self.planner_mode = planner_mode
        self.use_environment_model = use_environment_model
        self.robot_type_list = robot_type_list
        self.enable_render = enable_render
        self.enable_recording = enable_render
        self.enable_replan_dataset_capture = enable_replan_dataset_capture
        self.replan_capture_options = replan_capture_options or {}
        self.is_replay_mode: bool = replay_trace is not None
        self.replay_trace: Optional[ReplayTrace] = replay_trace
        self.solver_config = solver_config or {}

        # Replanning control.
        self.enable_replanning = enable_replanning or config.get_config('enable_replanning', False)
        self.replanning_count = 0

        # Context.
        self.context: Optional[WorkflowContext] = None
        self._context_initialized: bool = False

        # Initialize components.
        self._init_basic_components(logger, path_manager, run_dir)
        self._init_monitor()
        self._init_hitl_manager()
        self._init_layers()
        self._setup_event_subscriptions()

        # Control flags.
        self._none_strategy_streak: int = 0
        self._services_initialized: bool = False

        # Event buffer.
        self._newcase_event_buffer: List[Dict[str, Any]] = []

    # =========================================================================
    # Initialization methods
    # =========================================================================

    def _init_basic_components(self, logger, path_manager, run_dir: Optional[Path] = None):
        """Initialize basic components."""
        if path_manager:
            self.path_manager = path_manager
        elif run_dir:
            self.path_manager = PathManager(workspace_root_override=run_dir)
        else:
            self.path_manager = PathManager(formatted_date=datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

        self.logger = logger or Logger(log_file_dir=str(self.path_manager.workspace_root))

        if self.enable_replan_dataset_capture:
            ReplanDatasetRecorder.enable(**self.replan_capture_options)

        set_default_dump_dir(self.path_manager.workspace_root)

        # Common metrics.
        self.metrics_manager = MetricsManager({
            "success": (False, 'set'),
            "replans_full": (0, 'set'),
            "replans_partial": (0, 'set'),
            "total_energy": (0.0, 'increment'),     # Total energy consumption from accumulated movement distance.
            "newcase_total": (0, 'increment'),
            "newcase_total_orig": (0, 'set'),
            "newcase_by_type": ({}, 'set'),
        })

    def _init_monitor(self):
        """Initialize the task monitor."""
        try:
            self.monitor = get_global_task_monitor()
        except Exception:
            dlog("Task monitor not available", logger=self.logger, level='error')
            self.monitor = None

    def _init_hitl_manager(self):
        """Initialize the HITL manager."""
        try:
            self.hitl_manager = get_hitl_manager()
            if self.hitl_manager.is_enabled:
                dlog("HITL Manager is enabled", logger=self.logger, level='info')
        except Exception as e:
            dlog(f"Failed to initialize HITL Manager: {e}", logger=self.logger, level='warning')
            self.hitl_manager = None

    def _init_layers(self):
        """Initialize the three-layer architecture and create planning_layer here."""
        self.context = WorkflowContext(path_manager=self.path_manager)
        scene_graph: AbstractSceneGraph = get_scene_graph()

        # World model layer.
        self.world_model_layer = WorldModelLayer(
            scene_graph=scene_graph,
            monitor=self.monitor,
            logger=self.logger,
            is_replay=self.is_replay_mode,
        )

        # Get robot labels.
        robot_nodes = scene_graph.get_nodes_by_category("robot") if scene_graph else []
        robot_labels = sorted(
            n.get("properties", {}).get("label", "")
            for n in robot_nodes
            if n.get("properties", {}).get("label") and n.get("properties", {}).get("type") in self.robot_type_list
        )

        # Create the corresponding planning layer by solver_type.
        if self.solver_type == "sgi":
            sgi_cfg = self.solver_config.get("sgi", {})
            self.planning_layer = PlanningLayer(
                logger=self.logger,
                path_manager=self.path_manager,
                planner_mode=self.planner_mode,
                use_environment_model=self.use_environment_model,
                robot_type_list=self.robot_type_list,
                world_model=self.world_model_layer.world_model_manager,
                context=self.context,
                max_steps=sgi_cfg.get("max_steps", 15),
                validate_plan=sgi_cfg.get("validate_plan", True),
            )
        elif self.solver_type == "llamar":
            from modules.task_solver.baseline_planners.llamar.llamar_planning_layer import LLaMARPlanningLayer

            llamar_cfg = self.solver_config.get("llamar", {})
            self.planning_layer = LLaMARPlanningLayer(
                max_steps=llamar_cfg.get("max_steps", 20),
                world_model_manager=self.world_model_layer.world_model_manager,
                context=self.context,
                robot_labels=robot_labels,
                logger=self.logger,
                model_family=llamar_cfg.get("model_family"),
                model_name_override=llamar_cfg.get("model_name_override"),
                use_few_shot=llamar_cfg.get("use_few_shot", False),
            )
        elif self.solver_type == "spine":
            from modules.task_solver.baseline_planners.spine.spine_planning_layer import SPINEPlanningLayer

            spine_cfg = self.solver_config.get("spine", {})
            self.planning_layer = SPINEPlanningLayer(
                max_steps=spine_cfg.get("max_steps", 20),
                world_model_manager=self.world_model_layer.world_model_manager,
                context=self.context,
                robot_labels=robot_labels,
                logger=self.logger,
                model_family=spine_cfg.get("model_family"),
                model_name_override=spine_cfg.get("model_name_override"),
                n_attempts=spine_cfg.get("n_attempts", 3),
                use_few_shot=spine_cfg.get("use_few_shot", False),
            )
        elif self.solver_type == "lipllm":
            from modules.task_solver.baseline_planners.lipllm.lipllm_planning_layer import LipLLMPlanningLayer

            lipllm_cfg = self.solver_config.get("lipllm", {})
            self.planning_layer = LipLLMPlanningLayer(
                max_steps=lipllm_cfg.get("max_steps", 20),
                world_model_manager=self.world_model_layer.world_model_manager,
                context=self.context,
                robot_labels=robot_labels,
                logger=self.logger,
                model_family=lipllm_cfg.get("model_family"),
                model_name_override=lipllm_cfg.get("model_name_override"),
                use_few_shot=lipllm_cfg.get("use_few_shot", False),
                n_attempts=lipllm_cfg.get("n_attempts", 5),
                max_iterations=lipllm_cfg.get("max_iterations", 15),
                alpha=lipllm_cfg.get("alpha", 0.3),
            )
        elif self.solver_type == "smartllm":
            from modules.task_solver.baseline_planners.smartllm.smartllm_planning_layer import SmartLLMPlanningLayer

            smartllm_cfg = self.solver_config.get("smartllm", {})
            self.planning_layer = SmartLLMPlanningLayer(
                max_steps=smartllm_cfg.get("max_steps", 20),
                world_model_manager=self.world_model_layer.world_model_manager,
                context=self.context,
                robot_labels=robot_labels,
                logger=self.logger,
                model_family=smartllm_cfg.get("model_family"),
                model_name_override=smartllm_cfg.get("model_name_override"),
                use_few_shot=smartllm_cfg.get("use_few_shot", False),
            )
        else:
            raise ValueError(f"Unknown solver_type: {self.solver_type}")

        # Execution layer.
        self.platform_executor: AbstractPlatformExecutor = create_platform_executor(
            scene_graph=scene_graph,
            enable_visualization=self.enable_render,
            enable_video_recording=self.enable_recording,
            video_output_path=self.path_manager.workspace_root / 'execution_demo.mp4' if self.path_manager else None,
            logger_instance=self.logger,
            is_replay=self.is_replay_mode,
        )

        self.platform_executor.set_incremental_outcome_handler(self._handle_incremental_outcomes)
        self._setup_review_hooks()
        self._setup_replan_hooks()

    def _setup_replan_hooks(self):
        """Inject the pre-replan callback into planning_layer, SGI only."""
        if self.solver_type != "sgi":
            return

        self.planning_layer.prepare_full_replan = self._prepare_full_replan

    async def _prepare_full_replan(self):
        """Pre-replan callback: refresh feedback data and context, SGI only."""
        feedback_data = self.planning_layer.feedback_processor.prepare_feedback_data(
            task_graph_manager=self.planning_layer.task_graph_manager
        )
        self._refresh_context(feedback_data=feedback_data)

    def _setup_event_subscriptions(self):
        """Set up event subscriptions."""
        try:
            from modules.config.events import EventType
            from modules.events.event_bus import subscribe_event

            self.newcase_subscription_id = subscribe_event(
                event_type=EventType.NEW_CASE.value,
                handler=self._handle_newcase_event,
                subscriber_id="unified_solver_newcase"
            )
        except Exception as e:
            dlog(f"Error setting up event subscriptions: {e}", logger=self.logger, level='error')
            self.newcase_subscription_id = None

    async def initialize(self) -> None:
        """Asynchronous initialization: start platform services."""
        if self._services_initialized:
            dlog("Services already initialized, skipping", logger=self.logger, level='debug')
            return

        try:
            await self.platform_executor.start_services()
            self._services_initialized = True
            dlog("Platform services started successfully", logger=self.logger, level='info')
        except Exception as e:
            dlog(f"Failed to start platform services: {e}", logger=self.logger, level='error')
            raise

    # =========================================================================
    # Main flow
    # =========================================================================

    async def solve_task(self, knowledge_scope: str = 'local') -> bool:
        """Main task solving method, used as the shared entry point.
        
        Args:
            knowledge_scope: Knowledge scope, either 'local' or 'global'.
            
        Returns:
            Whether the task completed successfully.
        """
        if not self._services_initialized:
            await self.initialize()

        # Set knowledge scope.
        self.world_model_layer.set_knowledge_scope(knowledge_scope)
        self.platform_executor.set_knowledge_scope(knowledge_scope)
        await self._init_event_bus()

        # Initialize.
        self._init_context_static()
        await self._wait_for_hitl_instruction()
        self.planning_layer.reset()

        # Get max steps.
        max_steps = self._get_max_steps()
        step = 0

        # Shared main loop.
        while step < max_steps:
            dlog(f"UnifiedTaskSolver: Step {step + 1}/{max_steps}", logger=self.logger, level='info')

            try:
                result = await self._execute_planning_cycle()

                if result is not None:
                    # Terminate the loop.
                    return self._finalize_and_return(result)

            except ReplanSampleCollected as e:
                # SGI-specific: replan sample collection exception.
                dlog(f"Replan sample collected: {e}", logger=self.logger, level='stage')
                return self._finalize_and_return(False)

            except Exception as e:
                # Error handling: if steps remain, set FULL replanning and continue; otherwise exit.
                dlog(f"Error in execution loop: {e}", logger=self.logger, level='error')
                if step + 1 < max_steps and hasattr(self.planning_layer, 'feedback_processor'):
                    self.planning_layer.feedback_processor.replanning_strategy = ReplanningStrategy.FULL
                else:
                    return self._finalize_and_return(False)

            # Continue the loop.
            step += 1

        # Reached max steps.
        dlog(f"Maximum steps ({max_steps}) reached", logger=self.logger, level='warning')

        return self._finalize_and_return(False)

    def _prepare_cycle(self) -> None:
        """Prepare the planning cycle and record logs.
        
        Shared method for all methods with feedback_processor.
        Replanning metrics are recorded by each planning layer.
        """
        fp = self.planning_layer.feedback_processor

        if fp.replanning_strategy == ReplanningStrategy.FULL:
            dlog(
                f"Starting FULL planning cycle (init plan / replan #{self.replanning_count + 1})",
                logger=self.logger, level="stage"
            )

    async def _execute_planning_cycle(self) -> Optional[bool]:
        """Execute a single planning cycle.
        
        1. Call _prepare_cycle to process replanning signals.
        2. Get skill sequences, from replay mode or normal mode.
        3. Execute skills.
        4. Evaluate results.
            
        Returns:
            None: Continue the loop.
            bool: Terminate the loop and return success or failure.
        """
        # Methods with feedback_processor handle replanning signals and record metrics.
        if hasattr(self.planning_layer, 'feedback_processor'):
            self._prepare_cycle()

        # 1. Get skill sequences.
        if self.is_replay_mode:
            # Replay mode: get skill sequences from the trace.
            skills_by_timestep = await self._get_replay_skills()
        else:
            # Normal mode: call _get_planned_skills.
            skills_by_timestep = await self._get_planned_skills()

        # Handle the skill sequence retrieval result.
        if skills_by_timestep is None:
            # Planning failed. Terminate the loop and return failure.
            return False
        
        if not skills_by_timestep:
            # Empty dict. Use ground truth to verify whether the task is truly complete.
            ground_truth_done = self.world_model_layer.is_goal_completed()
            if ground_truth_done:
                dlog("Plan is empty and goal is completed - task successful", 
                     logger=self.logger, level='info')
                return True
            else:
                dlog("Plan is empty but goal not completed - planner early termination", 
                     logger=self.logger, level='warning')
                return False

        # 2. Execute skills.
        execution_result = await self._execute_skills(skills_by_timestep)
        dump_var("execution_result", execution_result,
                 meta={"replanning_count": self.replanning_count} if hasattr(self.planning_layer, 'feedback_processor') else None)

        # 3. Evaluate results.
        termination_result = await self._evaluate_execution_result_dispatch(execution_result)

        if termination_result is not None:
            # Need to terminate the loop.
            return termination_result

        # Handle replanning decision.
        if hasattr(self.planning_layer, 'feedback_processor'):
            self._handle_replan_decision(self.planning_layer.feedback_processor)

        # Continue the loop.
        return None

    async def _get_planned_skills(self) -> Optional[Dict]:
        """Normal mode: get skill sequences through planning.
        
        Shared planning flow:
        1. Call _execute_planning to generate a plan.
        2. Call _process_plan to process the plan and convert it to skill sequences.
        
        Returns:
            Skill sequence dictionary:
            - None: planning failed.
            - {}: empty dictionary, meaning the plan is empty.
            - {...}: normal skill sequence.
        """
        # (1) Execute planning.
        plan = await self._execute_planning()

        if plan is None:
            # Planning failed.
            dlog("Failed to generate plan", logger=self.logger, level='warning')
            return None
        
        if not plan:
            # Plan is empty.
            dlog("Plan is empty", logger=self.logger, level='warning')
            return {}

        # (2) Process the plan.
        return await self._process_plan(plan)

    def _load_episode_data(self, episode: Dict) -> None:
        """Load episode data into context and initialize TaskGraphManager on the SGI replay path."""
        from modules.task_solver.sgi_planner.task_graph_manager import TaskGraphManager

        conversations = episode.get("conversations", [])
        for conv in conversations:
            if conv.get("prompt"):
                dlog("[Replay] Prompt:\n", conv["prompt"], logger=self.logger, level='debug')
            if conv.get("response"):
                dlog("[Replay] Response:\n", conv["response"], logger=self.logger, level='info')

        last_response = ""
        if conversations:
            last_response = conversations[-1].get("response", "")
            if last_response:
                parsed = parse_text(last_response, "json", True)
                if parsed:
                    self.context._generated_text["task_plan"] = parsed[0]

        dispatcher_result = episode.get("dispatcher_result")
        if dispatcher_result is not None:
            self.context._generated_text["alloc_results"] = dispatcher_result

        raw_plan = self.context._generated_text.get("task_plan")
        if raw_plan and last_response:
            try:
                plan_data = json.loads(raw_plan) if isinstance(raw_plan, str) else raw_plan
                if isinstance(plan_data, dict):
                    self.planning_layer.task_graph_manager = TaskGraphManager(
                        plan_data, self.logger, self.context,
                        self.world_model_layer.world_model_manager,
                        self.planner_mode,
                    )
                    dlog("[replay] TaskGraphManager initialized from episode data",
                            logger=self.logger, level='info')
            except Exception as e:
                dlog(f"[replay] Failed to init TaskGraphManager: {e}",
                        logger=self.logger, level='warning')

    async def _get_replay_skills(self) -> Optional[Dict]:
        """Replay mode: get skill sequences from the trace."""
        if not self.replay_trace:
            dlog("[replay] Replay trajectory is empty", logger=self.logger, level='error')
            return None

        episode = self.replay_trace.next_episode()
        if episode is None:
            dlog("[replay] No more episodes to replay", logger=self.logger, level='stage')
            return None

        self._load_episode_data(episode)
        await self._replay_hitl_review(episode)

        skills_by_timestep = episode.get("skills_by_timestep") or {}
        await self.platform_executor.rollback_last_dynamic_event()
        self.platform_executor.set_replay_meta({
            "current_plan_selection": episode.get("current_plan_selection"),
            "new_case_event": episode.get("new_case_event"),
            "execution_result": episode.get("execution_result"),
        })
        return skills_by_timestep

    def _process_outcome_feedback(
        self,
        outcomes: List[Dict],
        status: str,
        goal_completed: bool,
        hitl_result: Optional[Dict] = None
    ) -> None:
        """Process feedback from normal execution results.
        
        Flow:
        - Send terminal feedback.
        - Call feedback_processor.process_outcome_event.
        - Update local world model.
        - Update task graph.
        - feedback_processor automatically appends replanning signals internally.
        """
        # Send terminal feedback.
        self._emit_terminal_feedback(outcomes, status, goal_completed)
        
        if not hasattr(self.planning_layer, 'feedback_processor'):
            return

        fp = self.planning_layer.feedback_processor
        
        # Call process_outcome_event.
        if self.solver_type == "sgi":
            graph_is_ongoing = (
                hasattr(self.planning_layer, 'task_graph_manager') and
                self.planning_layer.task_graph_manager and
                self.planning_layer.task_graph_manager.has_pending_tasks()
            )
            fp.process_outcome_event(
                outcomes=outcomes,
                planner_mode=self.planner_mode,
                status=status,
                goal_completed=goal_completed,
                graph_is_ongoing=graph_is_ongoing,
                world_model_manager=self.world_model_layer.world_model_manager,
                goal_progress_monitor=self.world_model_layer.goal_progress_monitor,
                hitl_decision=hitl_result,
            )
        else:
            fp.process_outcome_event(
                outcomes=outcomes,
                status=status,
                goal_completed=goal_completed,
                world_model_manager=self.world_model_layer.world_model_manager,
            )
        
        # Update local world model.
        self.world_model_layer.world_model_manager.update_from_outcomes(outcomes)
        
        # Update task graph.
        if hasattr(self.planning_layer, 'task_graph_manager') and self.planning_layer.task_graph_manager:
            self.planning_layer.update_graph_from_feedback()
        
        # Append replanning signals.
        if fp.replanning_requested:
            fp.add_replan_signal("evaluation", fp.last_event)
    
    def _process_newcase_feedback(self, outcomes: List[Dict[str, Any]]) -> bool:
        """Process cached new situation events, shared by methods with feedback_processor.
        
        Drain events from the buffer and process each one with fp.process_newcase_event.
        outcomes are stored directly in last_event when the event is built.
        
        Args:
            outcomes: Outcomes list for the current execution cycle.
            
        Returns:
            Whether at least one new situation event was processed.
        """
        if not hasattr(self.planning_layer, 'feedback_processor'):
            return False
        
        fp = self.planning_layer.feedback_processor
        events = self._drain_newcase_events()
        
        if not events:
            return False
        
        for event_content in events:
            fp.process_newcase_event(event_content, outcomes=outcomes)
            
            # Update local world model.
            self.world_model_layer.world_model_manager.update_from_events(
                fp.last_event.get('type'), fp.last_event.get('details', {})
            )

            # Update task graph.
            if hasattr(self.planning_layer, 'task_graph_manager') and self.planning_layer.task_graph_manager:
                self.planning_layer.update_graph_from_feedback()
            
            # Append replanning signals.
            fp.add_replan_signal("newcase", fp.last_event)
        
        return True

    def _handle_replan_decision(self, fp):
        """Process replanning decisions, shared method.
        
        replanning_count only records FULL replanning count as a metric and does
        not participate in loop control.
        """
        fp.replanning_strategy = fp.aggregate_strategy(self.enable_replanning)
        if fp.replanning_strategy == ReplanningStrategy.NONE:
            self._none_strategy_streak += 1
            if self._none_strategy_streak >= 3:
                dlog("Forcing FULL replan after 3 NONE cycles", logger=self.logger, level="warning")
                fp.replanning_strategy = ReplanningStrategy.FULL
                self._none_strategy_streak = 0
        else:
            self._none_strategy_streak = 0

        if fp.replanning_strategy == ReplanningStrategy.FULL:
            self.replanning_count += 1

    # =========================================================================
    # Execution steps
    # =========================================================================

    async def _execute_planning(self) -> Optional[Dict]:
        """Execute planning with a shared interface and solver_type-specific branches.
        
        Returns:
            Plan dictionary, or None if planning failed.
        """
        if self.solver_type == "sgi":
            feedback_data = self.planning_layer.feedback_processor.prepare_feedback_data(
                task_graph_manager=self.planning_layer.task_graph_manager
            )
            self._refresh_context(feedback_data=feedback_data)
        elif self.solver_type == "spine":
            self._refresh_context()
            self.planning_layer.feedback_processor.prepare_feedback_data(
                world_model_manager=self.world_model_layer.world_model_manager,
                robot_labels=self.planning_layer.robot_labels,
                planning_agent=self.planning_layer.planning_agent,
            )
        elif self.solver_type == "lipllm":
            self._refresh_context()
            self.planning_layer.feedback_processor.prepare_feedback_data(
                world_model_manager=self.world_model_layer.world_model_manager,
                robot_labels=self.planning_layer.robot_labels,
                planning_layer=self.planning_layer,
            )
        elif self.solver_type == "smartllm":
            self._refresh_context()
            self.planning_layer.feedback_processor.prepare_feedback_data(
                world_model_manager=self.world_model_layer.world_model_manager,
                robot_labels=self.planning_layer.robot_labels,
            )
        else:
            # Baselines without feedback_processor, such as LLaMAR.
            self._refresh_context()
            
        # Call the planning layer to generate a plan.
        plan = await self.planning_layer.generate_plan()
        
        return plan

    async def _process_plan(self, plan: Dict) -> Optional[Dict]:
        """Process a plan and convert it to skill sequences.
        
        Args:
            plan: Plan generated by the planning layer.
            
        Returns:
            Skill sequence dictionary, or None if processing failed.
        """
        # Roll back the previous dynamic event.
        await self.platform_executor.rollback_last_dynamic_event()
        
        # Process the plan.
        result = self.world_model_layer.process_plan(
            plan=plan,
            goal_config=self.planning_layer.get_goal_config(),
            task_dependencies=self.planning_layer.get_dependencies(),
            area_boundaries=self.planning_layer.get_area_boundaries(),
            category_map=self.planning_layer.get_category_map(),
            runtime_params=self.planning_layer.get_runtime_params()
        )
        
        # Accumulate energy consumption.
        coordinator = self.world_model_layer.plan_translate_coordinator
        if hasattr(coordinator, 'get_last_plan_energy'):
            self.metrics_manager.record("total_energy", coordinator.get_last_plan_energy())
        
        return result

    async def _execute_skills(self, skills_by_timestep: Dict) -> Dict[str, Any]:
        """Execute skills and return normalized ExecutionResult format.
        
        Args:
            skills_by_timestep: Skill sequence dictionary.
            
        Returns:
            Normalized ExecutionResult format:
            {
                'status': 'completed' | 'failed',
                'outcomes': List[Dict[str, Any]]
            }
        """
        # Methods with feedback_processor, such as SGI and SPINE, reset it here.
        if hasattr(self.planning_layer, 'feedback_processor'):
            self.planning_layer.feedback_processor.reset()
        
        # Execute skills.
        exec_result = await self.platform_executor.execute_plan(skills_by_timestep)

        # Record the newcase_total_orig metric.
        if self.platform_executor.newcase_generator:
            self.metrics_manager.record("newcase_total_orig",
                int(self.platform_executor.newcase_generator._generated_count or 0))

        # Return normalized format.
        status = 'completed' if exec_result.get('success') else 'failed'
        return {'status': status, 'outcomes': exec_result.get('outcomes', [])}

    async def _evaluate_execution_result(self, execution_result: Dict) -> Optional[bool]:
        """Evaluate results for shared methods.
        
        Flow:
        - New situation event feedback through _process_newcase_feedback.
        - HITL decision.
        - Normal execution result feedback through _process_outcome_feedback.
        - Determine whether replanning is needed.
        
        Args:
            execution_result: Execution result dictionary in {'status': str, 'outcomes': List} format.
            
        Returns:
            None: Continue the loop.
            bool: Terminate the loop and return success or failure.
        """
        outcomes = execution_result.get('outcomes', []) or []
        status = execution_result.get('status')
        fp = self.planning_layer.feedback_processor

        # A new situation means the task is not complete. Trigger replanning directly.
        if self._process_newcase_feedback(outcomes):
            return None  # Continue the loop and enter replanning.

        # Normal path: HITL decision -> goal completion check -> outcome feedback processing.
        hitl_result = await self._request_hitl_decision()
        goal_completed = self._determine_goal_completion(hitl_result)
        self._process_outcome_feedback(outcomes, status, goal_completed, hitl_result)

        # Check whether replanning is needed.
        if not fp.replanning_requested:
            return True  # Task complete. Terminate the loop.

        return None  # Replanning needed. Continue the loop.
    
    def _determine_goal_completion(self, hitl_result: Optional[Dict]) -> bool:
        """Determine goal completion status, shared method.
        
        Args:
            hitl_result: HITL decision result.
            
        Returns:
            Whether the goal is complete.
        """
        if hitl_result is not None:
            selected_option = hitl_result.get("decision")
            if selected_option == "end_task":
                dlog("HITL decision: end_task", logger=self.logger, level='info')
                return True
            else:
                dlog("HITL decision: continue_task", logger=self.logger, level='info')
                return False
        else:
            return self.world_model_layer.is_goal_completed()

    async def _evaluate_execution_result_dispatch(self, execution_result: Dict) -> Optional[bool]:
        """Evaluate execution results, shared entry point.
        
        Args:
            execution_result: Execution result dictionary in {'status': str, 'outcomes': List} format.
            
        Returns:
            None: Continue the loop.
            bool: Terminate the loop and return success or failure.
        """
        if self.solver_type == "llamar":
            return await self._evaluate_execution_result_llamar(execution_result)
        else:
            # Shared by sgi, spine, and other methods.
            return await self._evaluate_execution_result(execution_result)

    def _emit_terminal_feedback(self, outcomes: List, status: str, goal_completed: bool):
        """Send terminal feedback, shared method."""
        scene_graph = get_scene_graph()
        goal = scene_graph.get_goal() if scene_graph else None
        if goal:
            fb = self.world_model_layer.generate_terminal_feedback(
                goal_id=goal["id"], outcomes=outcomes,
                plan_completed=(status == 'completed'), achieved=goal_completed
            )
            if fb:
                dlog(fb, logger=self.logger, level='debug')

    # =========================================================================
    # Event handling
    # =========================================================================

    async def _handle_newcase_event(self, event) -> None:
        """Process a new situation event by caching it only, with no extra processing.
        
        Only does two things:
        1. Record metrics.
        2. Store event.description directly in the buffer.
        
        Actual feedback processing happens later:
        - Methods with feedback_processor: processed in _process_newcase_feedback.
        - LLaMAR: passed to planning_layer.process_feedback after _drain_newcase_events.
        """
        try:
            event_content = event.description or {}
            kind = event_content.get("event_kind", "incident").lower()

            dump_var("feedback_event", event_content)
            self._bump_newcase_metrics(kind, event_content.get("reason"))

            # ReplanDatasetRecorder snapshot, if enabled.
            scene_graph = get_scene_graph()
            if self.monitor and scene_graph and ReplanDatasetRecorder.is_enabled():
                ReplanDatasetRecorder.snapshot_newcase(
                    context=self.context, scene_graph=scene_graph,
                    event_content=event_content
                )

            # Cache event.description directly without building extra structure.
            self._newcase_event_buffer.append(event_content)
        except Exception as e:
            dlog(f"Error handling NewCaseEvent: {e}",
                 logger=self.logger, level='error')

    def _bump_newcase_metrics(self, kind: str, trig_type: Optional[str]) -> None:
        """Update new situation metrics."""
        try:
            self.metrics_manager.record("newcase_total", 1)
            metrics_dict = self.metrics_manager.get_all()
            by_type_dict = metrics_dict.setdefault("newcase_by_type", {})
            key = f"{(kind or 'unknown').lower()}:{(trig_type or 'unknown').lower()}"
            by_type_dict[key] = int(by_type_dict.get(key, 0)) + 1
        except Exception:
            pass

    def _drain_newcase_events(self) -> List[Dict[str, Any]]:
        """Drain and clear the event buffer, shared method."""
        events = list(self._newcase_event_buffer)
        self._newcase_event_buffer.clear()
        return events

    async def _handle_incremental_outcomes(self, step_outcomes: List[Dict[str, Any]]) -> None:
        """Process incremental outcomes."""
        if not step_outcomes:
            return

        try:
            self.world_model_layer.process_outcomes(step_outcomes)
            
            if self.world_model_layer.world_model_manager:
                self.world_model_layer.world_model_manager.update_from_outcomes(step_outcomes)
            
            if hasattr(self.planning_layer, "task_graph_manager") and self.planning_layer.task_graph_manager:
                self.planning_layer.task_graph_manager._update_task_statuses(step_outcomes)
                
        except Exception as e:
            dlog(f"Error processing incremental outcomes: {e}", logger=self.logger, level='error')


    # =========================================================================
    # Utility methods
    # =========================================================================

    def _get_max_steps(self) -> int:
        """Get the maximum step limit.
        
        Shared logic:
        - Replay mode: unlimited steps.
        - enable_replanning is True: use planning_layer.max_steps.
        - enable_replanning is False: one step, with no replanning.
        
        Returns:
            Maximum step count.
        """
        if self.is_replay_mode:
            return int(1e9)
        if self.enable_replanning:
            return getattr(self.planning_layer, 'max_steps', 15)
        return 1

    async def _init_event_bus(self):
        """Initialize the event bus."""
        try:
            from modules.events.event_bus import get_global_event_bus, start_global_event_bus
            event_bus = get_global_event_bus()
            if event_bus and not event_bus.running:
                await start_global_event_bus()
        except Exception as e:
            dlog(f"Error initializing event bus: {e}", logger=self.logger, level='error')

    async def _evaluate_execution_result_llamar(
        self,
        execution_result: Dict
    ) -> Optional[bool]:
        """Evaluate results for baseline methods.
        
        Uses a dual judgment mechanism:
        1. planning_layer.is_task_completed() - the algorithm's own judgment.
        2. world_model_layer.is_goal_completed() - ground truth judgment.
        
        Flow:
        - Collect new situation events through _drain_newcase_events.
        - Call planning_layer.process_feedback.
        - Use the dual judgment mechanism in _evaluate_termination_baseline.
        - Record verifier correctness.
        
        Args:
            execution_result: Execution result dictionary in {'status': str, 'outcomes': List} format.
            
        Returns:
            None: Continue the loop.
            bool: Terminate the loop and return success or failure.
        """
        outcomes = execution_result.get('outcomes', []) or []
        
        # Collect new situation events.
        newcase_events = self._drain_newcase_events()
        
        # Let the planning layer process feedback.
        await self.planning_layer.process_feedback(
            execution_result,
            newcase_events,
            self.context
        )
        
        # Dual judgment mechanism.
        planner_says_done = self.planning_layer.is_task_completed()
        ground_truth_done = self.world_model_layer.is_goal_completed()
        
        # Evaluate termination conditions.
        termination_result = self._evaluate_termination_baseline(
            planner_says_done=planner_says_done,
            ground_truth_done=ground_truth_done,
        )
        
        if termination_result is not None:
            # Need to terminate the loop.
            success, verifier_correct = termination_result
            if verifier_correct is not None:
                self.planning_layer._metrics["verifier_correct"] = verifier_correct
            return success
        
        # Continue the loop.
        return None

    def _evaluate_termination_baseline(
        self,
        planner_says_done: bool,
        ground_truth_done: bool,
    ) -> Optional[tuple[bool, bool]]:
        """Evaluate baseline termination conditions with a dual judgment mechanism.
        
        Dual judgment cases:
        1. Planner complete + GT complete -> (True, True), correct judgment.
        2. Planner complete + GT incomplete -> (False, False), incorrect early stop.
        3. Planner incomplete + GT complete -> None, incorrect late stop, continue loop.
        4. Planner incomplete + GT incomplete -> None, continue loop.
        
        Args:
            planner_says_done: Whether planning_layer judges the task complete.
            ground_truth_done: Whether world_model judges the task complete.
            
        Returns:
            (success, verifier_correct), or None to continue the loop.
        """
        if planner_says_done:
            # Algorithm judges complete.
            if ground_truth_done:
                # Case 1: algorithm complete + GT complete -> correct judgment.
                dlog(
                    "Baseline: planner correctly identified task completion",
                    logger=self.logger, level='info'
                )
                return (True, True)
            else:
                # Case 2: algorithm complete + GT incomplete -> incorrect early stop.
                dlog(
                    "Baseline: planner incorrectly stopped (early termination)",
                    logger=self.logger, level='warning'
                )
                return (False, False)
        else:
            # Algorithm judges incomplete.
            if ground_truth_done:
                # Case 3: algorithm incomplete + GT complete -> incorrect judgment, continue loop.
                dlog(
                    "Baseline: planner failed to detect completion (late termination)",
                    logger=self.logger, level='warning'
                )
                return None
            else:
                # Case 4: algorithm incomplete + GT incomplete -> correct judgment, continue loop.
                return None

    def _finalize_metrics(
        self,
        success: bool,
    ) -> None:
        """Record final metrics.
        
        All algorithm-specific metrics are returned by each planning layer's
        get_metrics() and merged here.
        
        Args:
            success: Whether the task completed successfully.
        """
        # Merge all planning layer metrics.
        self.metrics_manager.merge_from(self.planning_layer.get_metrics())
        
        # Record common metrics.
        self.metrics_manager.record("success", success)
        
        # Compute derived metric: total replans.
        replans_full = self.metrics_manager.get("replans_full", 0)
        replans_partial = self.metrics_manager.get("replans_partial", 0)
        self.metrics_manager.record("replans_total", replans_full + replans_partial)
        
        # New situation statistics, shared.
        if self.platform_executor.newcase_controller:
            summary = self.platform_executor.newcase_controller.summary()
            self.metrics_manager.record("newcase_n_new_target", summary.get("n_new_target", 0))
            self.metrics_manager.record("newcase_generated_count", summary.get("generated_count", 0))
            
            # Record new situation statistics by skill.
            per_skill = summary.get("per_skill_counts", {})
            self.metrics_manager.record(
                "new_case_cnt_by_skill",
                dict(zip(per_skill.get("order", []), per_skill.get("counts", [])))
            )
            
            # Record new situation statistics by type.
            per_type = summary.get("per_newcase_type_counts", {})
            self.metrics_manager.record(
                "new_case_cnt_by_type",
                dict(zip(per_type.get("order", []), per_type.get("counts", [])))
            )

    def _finalize_and_return(self, success_status: bool) -> bool:
        """Finalize the task and return the result, shared method."""
        self._finalize_metrics(success_status)
        return success_status

    # =========================================================================
    # Cleanup and metrics
    # =========================================================================

    async def cleanup(self):
        """Clean up resources."""
        try:
            if self.newcase_subscription_id:
                from modules.events.event_bus import unsubscribe_event
                unsubscribe_event(self.newcase_subscription_id)

            if hasattr(self.platform_executor, 'stop_services'):
                await self.platform_executor.stop_services()
            if hasattr(self.platform_executor, 'cleanup'):
                self.platform_executor.cleanup()
            if hasattr(self.planning_layer, 'cleanup'):
                self.planning_layer.cleanup()

            dlog("All resources cleaned up", logger=self.logger)
        except Exception as e:
            dlog(f"Error during cleanup: {e}", logger=self.logger, level='error')

    def get_metrics(self) -> Dict[str, Any]:
        """Get metrics."""
        return self.metrics_manager.get_all()
