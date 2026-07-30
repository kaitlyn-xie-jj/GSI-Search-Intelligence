# -*- coding: utf-8 -*-
"""
LipLLMPlanningLayer - LipLLM planning layer implementing BaselinePlanner interface

Encapsulates the three-stage pipeline: skill list generation -> dependency graph generation -> LP task allocation.
Driven by UnifiedTaskSolver via generate_plan; feedback is handled through feedback_processor.
"""
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from modules.task_solver.baseline_planners.base_planner import BaselinePlanner
from modules.task_solver.baseline_planners.common.action_converter import ActionConverter
from modules.task_solver.baseline_planners.lipllm.lipllm_feedback_processor import (
    LipLLMFeedbackProcessor,
)
from modules.task_solver.baseline_planners.lipllm.task_allocator import TaskAllocator
from modules.task_solver.llm_framework.core.context import WorkflowContext
from modules.task_solver.llm_framework.file import Logger
from modules.utils.system.logging_utils import dlog

logger = logging.getLogger(__name__)


class LipLLMPlanningLayer(BaselinePlanner):
    """LipLLM planning layer — encapsulates the three-stage pipeline.

    Implements BaselinePlanner interface, driven by UnifiedTaskSolver:
    - generate_plan: SkillListAgent -> DependencyGraphAgent -> TaskAllocator -> dispatcher_result
    - feedback_processor: Called by UnifiedTaskSolver in _evaluate_execution_result to process outcomes
    - feedback_processor.prepare_feedback_data: Called by UnifiedTaskSolver before _execute_planning

    Attributes:
        max_steps: Maximum step limit.
        task_allocator: LP-based task allocator.
        action_converter: Action format converter.
        feedback_processor: LipLLM-specific feedback processor.
    """

    def __init__(
        self,
        max_steps: int = 20,
        world_model_manager=None,
        context: Optional[WorkflowContext] = None,
        robot_labels: Optional[List[str]] = None,
        logger: Logger = None,
        model_family: str = None,
        model_name_override: str = None,
        use_few_shot: bool = False,
        n_attempts: int = 5,
        max_iterations: int = 15,
        alpha: float = 0.3,
    ):
        self.max_steps = max_steps
        self.logger = logger
        self._world_model_manager = world_model_manager
        self._context = context

        # Robot labels
        self.robot_labels: List[str] = robot_labels or []

        # Internal components (non-LLM)
        self.task_allocator = TaskAllocator(alpha=alpha)
        self.action_converter = ActionConverter(step_prefix="lipllm_step")
        self.feedback_processor = LipLLMFeedbackProcessor(context=context)
        self.feedback_processor.set_planning_layer(self)

        # Agent config (lazy init)
        self._model_family = model_family
        self._model_name_override = model_name_override
        self._use_few_shot = use_few_shot
        self._n_attempts = n_attempts
        self._max_iterations = max_iterations
        self._agents_initialized = False
        self.skill_list_agent = None
        self.dependency_graph_agent = None

        # Internal state
        self._skill_list: List[str] = []
        self._dependency_edges: List[Tuple[str, str]] = []
        self._dependency_graph = None
        self._allocation_result: Optional[Dict] = None
        self._feedback_text: Optional[str] = None

        # Timestep queue: cached pending timesteps not yet dispatched
        self._pending_timesteps: List[Dict] = []
        self._current_step_num: int = 0  # Used for task_id generation

        # Metrics
        self._metrics: Dict[str, Any] = {
            "success": False,
            "total_steps": 0,
            "llm_calls": 0,
            "llm_calls_skill_list": 0,
            "llm_calls_dependency": 0,
            "replans_full": 0,
            "replans_partial": 0,
            "planning_durations": [],
            "allocation_durations": [],
        }

    # =========================================================================
    # Agent initialization
    # =========================================================================

    def _ensure_agents(self, context: WorkflowContext = None) -> None:
        """Ensure agents are initialized (lazy init, created on first call)."""
        if self._agents_initialized:
            return

        ctx = context or self._context
        if ctx is None:
            raise RuntimeError("LipLLMPlanningLayer requires a WorkflowContext")

        self._context = ctx

        from modules.task_solver.baseline_planners.lipllm.agents.skill_list_agent import (
            SkillListAgent,
        )
        from modules.task_solver.baseline_planners.lipllm.agents.dependency_graph_agent import (
            DependencyGraphAgent,
        )

        self.skill_list_agent = SkillListAgent(
            logger_=self.logger,
            context=ctx,
            robot_labels=self.robot_labels,
            model_family=self._model_family,
            model_name_override=self._model_name_override,
            use_few_shot=self._use_few_shot,
            max_iterations=self._max_iterations,
        )
        self.dependency_graph_agent = DependencyGraphAgent(
            logger_=self.logger,
            context=ctx,
            model_family=self._model_family,
            model_name_override=self._model_name_override,
            use_few_shot=self._use_few_shot,
        )
        self._agents_initialized = True

    # =========================================================================
    # BaselinePlanner interface
    # =========================================================================

    async def generate_plan(self) -> Optional[Dict]:
        """Timestep queue management: dispatch one timestep at a time.

        Logic:
        - If feedback_processor has newcase_triggered, clear queue and run full pipeline replan
        - If queue has pending timesteps, pop and return the next one (no LLM call)
        - If queue is empty, run full three-stage pipeline, cache all timesteps, pop the first one

        Returns:
            Single timestep dispatcher_result dict, or None on planning failure.
        """
        context = self._context
        self._ensure_agents(context)

        # Check if newcase triggered replan -> clear queue, force full replan
        if self.feedback_processor._newcase_triggered:
            self._pending_timesteps.clear()
            self.feedback_processor._newcase_triggered = False
            dlog(
                "LipLLM: newcase triggered, clearing pending timesteps for full replan",
                logger=self.logger,
                level="info",
            )

        # If queue has pending timesteps, pop the next one
        if self._pending_timesteps:
            return self._pop_next_timestep()

        # Queue empty, run full three-stage pipeline (timed)
        _t0 = time.time()
        full_result = await self._run_full_pipeline()
        self._metrics["planning_durations"].append(round(time.time() - _t0, 6))
        if full_result is None:
            return None

        # Split full result into individual timesteps and cache in queue
        self._enqueue_timesteps(full_result)

        if not self._pending_timesteps:
            return {}

        # Pop the first timestep
        return self._pop_next_timestep()

    def has_pending_steps(self) -> bool:
        """Whether there are pending timesteps to dispatch."""
        return len(self._pending_timesteps) > 0

    def clear_pending_steps(self) -> None:
        """Clear the pending timestep queue."""
        self._pending_timesteps.clear()

    def _pop_next_timestep(self) -> Dict:
        """Pop the next timestep from the queue, wrapped as dispatcher_result format."""
        ts_data = self._pending_timesteps.pop(0)
        dlog(
            f"LipLLM: dispatching timestep {ts_data['ts_key']} "
            f"({len(self._pending_timesteps)} remaining)",
            logger=self.logger,
            level="info",
        )
        return {"timestep_skills": {ts_data["ts_key"]: ts_data["skills"]}}

    def _enqueue_timesteps(self, full_result: Dict) -> None:
        """Split full dispatcher_result into individual timesteps and cache in queue."""
        timestep_skills = full_result.get("timestep_skills", {})
        if not timestep_skills:
            return

        # Sort by timestep key (numeric order)
        sorted_keys = sorted(timestep_skills.keys(), key=lambda k: int(k))
        for ts_key in sorted_keys:
            self._pending_timesteps.append({
                "ts_key": ts_key,
                "skills": timestep_skills[ts_key],
            })

        dlog(
            f"LipLLM: enqueued {len(self._pending_timesteps)} timesteps",
            logger=self.logger,
            level="info",
        )

    async def _run_full_pipeline(self) -> Optional[Dict]:
        """Full three-stage pipeline: skill_list -> dependency_graph -> allocation -> dispatcher_result.

        Returns:
            Full dispatcher_result dict (all timesteps), or None on planning failure.
        """
        context = self._context
        wmm = self._get_world_model_manager(context)

        step = self._metrics["total_steps"]
        dlog(
            f"LipLLMPlanningLayer: _run_full_pipeline step={step}",
            logger=self.logger,
            level="info",
        )

        # Record replan metrics: each full pipeline after the first plan counts as a full replan
        if step > 0:
            self._metrics["replans_full"] += 1

        # Get feedback text (used during replanning)
        feedback_text = self.feedback_processor.get_feedback_text()

        # ---- Stage 1: Skill list generation ----
        try:
            self.skill_list_agent.known_nodes = wmm.known_nodes
            self.skill_list_agent.known_edges = wmm.known_edges
            skill_list = await self.skill_list_agent.generate_skill_list(
                feedback_text=feedback_text,
            )
        except Exception as e:
            logger.error(f"SkillListAgent failed: {e}")
            return None

        # Record metrics: LLM calls for skill list generation (iterative, each iteration = one call)
        self._metrics["llm_calls"] += self.skill_list_agent.llm_call_count
        self._metrics["llm_calls_skill_list"] += 1
        self.skill_list_agent.llm_call_count = 0  # Reset agent counter

        if not skill_list:
            logger.warning("SkillListAgent returned empty skill list.")
            return None

        self._skill_list = skill_list
        dlog(
            f"LipLLM Stage 1: generated {len(skill_list)} skills: {skill_list}",
            logger=self.logger,
            level="info",
        )

        # ---- Stage 2: Dependency graph generation ----
        try:
            graph, edges = await self.dependency_graph_agent.generate_dependencies(
                skill_list=skill_list,
                n_attempts=self._n_attempts,
            )
        except Exception as e:
            logger.error(f"DependencyGraphAgent failed: {e}")
            return None

        # Record metrics: LLM calls for dependency graph (may include retries)
        self._metrics["llm_calls"] += self.dependency_graph_agent.llm_call_count
        self._metrics["llm_calls_dependency"] += 1
        self.dependency_graph_agent.llm_call_count = 0  # Reset agent counter

        self._dependency_graph = graph
        self._dependency_edges = edges
        dlog(
            f"LipLLM Stage 2: generated {len(edges)} dependency edges",
            logger=self.logger,
            level="info",
        )

        # ---- Stage 3: LP task allocation ----
        _t_alloc = time.time()
        real_time_pos_map = context._generated_text.get("real_time_pos_map", None)
        robot_positions = TaskAllocator.build_robot_positions(
            self.robot_labels, real_time_pos_map
        )
        skill_positions = TaskAllocator.build_skill_positions(
            skill_list, real_time_pos_map
        )

        # Build label -> type mapping for type constraints
        from modules.task_solver.baseline_planners.lipllm.prompt.lipllm_prompt import (
            build_robot_type_map,
        )
        robot_type_map = build_robot_type_map(self.robot_labels)

        try:
            allocation = self.task_allocator.allocate(
                graph=graph,
                robot_labels=self.robot_labels,
                robot_positions=robot_positions,
                skill_positions=skill_positions,
                robot_type_map=robot_type_map,
            )
        except Exception as e:
            logger.error(f"TaskAllocator failed: {e}")
            self._metrics["allocation_durations"].append(round(time.time() - _t_alloc, 6))
            return None

        self._metrics["allocation_durations"].append(round(time.time() - _t_alloc, 6))

        self._allocation_result = allocation
        dlog(
            f"LipLLM Stage 3: allocated into {len(allocation)} timesteps",
            logger=self.logger,
            level="info",
        )

        # ---- Convert to dispatcher_result ----
        dispatcher_result = self._convert_allocation_to_dispatcher_result(
            allocation, self._current_step_num
        )
        dlog(
            f"LipLLM: converted allocation to dispatcher_result: {dispatcher_result}",
            logger=self.logger,
            level="info",
        )

        self._metrics["total_steps"] = step + 1
        return dispatcher_result

    async def process_feedback(
        self,
        outcomes: List[Dict],
        newcase_events: List[Dict],
        context,
    ) -> None:
        """No-op; feedback is handled by feedback_processor.

        LipLLM uses the feedback_processor path (same as SPINE),
        driven by UnifiedTaskSolver's _evaluate_execution_result.
        """
        pass

    def is_task_completed(self) -> bool:
        """Always returns False; task completion is determined by WorldModelLayer."""
        return False

    def reset(self) -> None:
        """Reset internal state."""
        if self.skill_list_agent:
            self.skill_list_agent.reset()
        self.feedback_processor.reset()
        self._skill_list = []
        self._dependency_edges = []
        self._dependency_graph = None
        self._allocation_result = None
        self._feedback_text = None
        self._pending_timesteps = []
        self._current_step_num = 0
        self._metrics: Dict[str, Any] = {
            "success": False,
            "total_steps": 0,
            "llm_calls": 0,
            "llm_calls_skill_list": 0,
            "llm_calls_dependency": 0,
            "replans_full": 0,
            "replans_partial": 0,
            "planning_durations": [],
            "allocation_durations": [],
        }

    def get_metrics(self) -> Dict[str, Any]:
        """Get experiment metrics, format compatible with MetricsManager."""
        return dict(self._metrics)

    # =========================================================================
    # Helper methods
    # =========================================================================

    def _convert_allocation_to_dispatcher_result(
        self,
        allocation: Dict[int, Dict[str, str]],
        step_num: int,
    ) -> Optional[Dict]:
        """Convert TaskAllocator allocation result to dispatcher_result format.

        Typed skills (robot_type:skill_str) have the pure skill_str extracted here for ActionConverter.

        Args:
            allocation: {timestep_idx: {robot_label: typed_skill}} allocation result.
            step_num: Current step number, used for task_id generation.

        Returns:
            dispatcher_result dict (with timestep_skills), or empty dict if no allocation.
        """
        if not allocation:
            return {}

        from modules.task_solver.baseline_planners.lipllm.task_allocator import (
            extract_skill_str,
        )

        task_id = f"lipllm_step_{step_num}"
        timestep_skills: Dict[str, Dict[str, Dict[str, Any]]] = {}

        for ts_idx, robot_skills in allocation.items():
            ts_key = str(ts_idx)
            timestep_skills[ts_key] = {}
            for robot_label, typed_skill in robot_skills.items():
                # Extract pure skill_str, strip robot_type prefix
                skill_str = extract_skill_str(typed_skill)

                # Convert skill format via ActionConverter
                parsed = self.action_converter.parse_single_action(skill_str)
                if parsed == "sync_wait" and skill_str.lower() not in (
                    "sync_wait", "wait", "done", ""
                ):
                    logger.warning(
                        f"ActionConverter could not parse '{skill_str}', "
                        f"using original string as fallback."
                    )
                    parsed = skill_str

                timestep_skills[ts_key][robot_label] = {
                    "skill_str": parsed,
                    "task_id": task_id,
                }

        if not timestep_skills:
            return {}

        return {"timestep_skills": timestep_skills}

    def _get_world_model_manager(self, context):
        """Get WorldModelManager instance."""
        if self._world_model_manager is not None:
            return self._world_model_manager
        return getattr(context, "_world_model_manager", None)

    def set_world_model_manager(self, wmm) -> None:
        """Set WorldModelManager reference (called by UnifiedTaskSolver)."""
        self._world_model_manager = wmm

    def set_robot_labels(self, labels: List[str]) -> None:
        """Set robot label list (called by factory or UnifiedTaskSolver)."""
        self.robot_labels = labels
        if self.skill_list_agent:
            self.skill_list_agent.robot_labels = labels
