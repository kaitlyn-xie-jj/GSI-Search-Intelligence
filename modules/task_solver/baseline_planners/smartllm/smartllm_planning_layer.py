# -*- coding: utf-8 -*-
"""
SmartLLMPlanningLayer - SmartLLM planning layer implementing BaselinePlanner interface

Encapsulates the three-stage pipeline: task decomposition -> coalition formation -> task allocation.
Driven by UnifiedTaskSolver via generate_plan; feedback is handled through feedback_processor.
"""
import json
import logging
import time
from typing import Any, Dict, List, Optional

from modules.task_solver.baseline_planners.base_planner import BaselinePlanner
from modules.task_solver.baseline_planners.common.action_converter import ActionConverter
from modules.task_solver.baseline_planners.smartllm.smartllm_feedback_processor import (
    SmartLLMFeedbackProcessor,
)
from modules.task_solver.llm_framework.core.context import WorkflowContext
from modules.task_solver.llm_framework.file import Logger
from modules.utils.system.logging_utils import dlog

logger = logging.getLogger(__name__)


class SmartLLMPlanningLayer(BaselinePlanner):
    """SmartLLM planning layer — three-stage pipeline: decompose -> coalition -> allocate.

    Implements BaselinePlanner interface, driven by UnifiedTaskSolver:
    - generate_plan: TaskDecompositionAgent -> CoalitionFormationAgent -> TaskAllocationAgent -> dispatcher_result
    - feedback_processor: Called by UnifiedTaskSolver in _evaluate_execution_result to process outcomes
    - feedback_processor.prepare_feedback_data: Called by UnifiedTaskSolver before _execute_planning

    Attributes:
        max_steps: Maximum step limit.
        action_converter: Action format converter.
        feedback_processor: SmartLLM-specific feedback processor.
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
    ):
        self.max_steps = max_steps
        self.logger = logger
        self._world_model_manager = world_model_manager
        self._context = context

        # Robot labels
        self.robot_labels: List[str] = robot_labels or []

        # Internal components
        self.action_converter = ActionConverter(step_prefix="smartllm_step")
        self.feedback_processor = SmartLLMFeedbackProcessor(context=context)
        self.feedback_processor.set_planning_layer(self)

        # Agent config (lazy init)
        self._model_family = model_family
        self._model_name_override = model_name_override
        self._use_few_shot = use_few_shot
        self._agents_initialized = False
        self.decomposition_agent = None
        self.coalition_agent = None
        self.allocation_agent = None

        # Internal state
        self._initial_plan_done = False
        self._active_robot_labels: List[str] = []

        # Timestep queue: cached pending timesteps not yet dispatched
        self._pending_timesteps: List[Dict] = []

        # Metrics
        self._metrics: Dict[str, Any] = {
            "success": False,
            "total_steps": 0,
            "llm_calls": 0,
            "llm_calls_decomposition": 0,
            "llm_calls_coalition": 0,
            "llm_calls_allocation": 0,
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
            raise RuntimeError("SmartLLMPlanningLayer requires a WorkflowContext")

        self._context = ctx

        from modules.task_solver.baseline_planners.smartllm.agents.task_decomposition_agent import (
            TaskDecompositionAgent,
        )
        from modules.task_solver.baseline_planners.smartllm.agents.coalition_formation_agent import (
            CoalitionFormationAgent,
        )
        from modules.task_solver.baseline_planners.smartllm.agents.task_allocation_agent import (
            TaskAllocationAgent,
        )

        self.decomposition_agent = TaskDecompositionAgent(
            logger_=self.logger,
            context=ctx,
            robot_labels=self.robot_labels,
            model_family=self._model_family,
            model_name_override=self._model_name_override,
            use_few_shot=self._use_few_shot,
        )
        self.coalition_agent = CoalitionFormationAgent(
            logger_=self.logger,
            context=ctx,
            robot_labels=self.robot_labels,
            model_family=self._model_family,
            model_name_override=self._model_name_override,
            use_few_shot=self._use_few_shot,
        )
        self.allocation_agent = TaskAllocationAgent(
            logger_=self.logger,
            context=ctx,
            robot_labels=self.robot_labels,
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

        # Check if newcase triggered replan -> clear queue
        if self.feedback_processor._newcase_triggered:
            self._pending_timesteps.clear()
            self.feedback_processor._newcase_triggered = False
            dlog(
                "SmartLLM: newcase triggered, clearing pending timesteps for full replan",
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
            f"SmartLLM: dispatching timestep {ts_data['ts_key']} "
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

        sorted_keys = sorted(timestep_skills.keys(), key=lambda k: int(k))
        for ts_key in sorted_keys:
            self._pending_timesteps.append({
                "ts_key": ts_key,
                "skills": timestep_skills[ts_key],
            })

        dlog(
            f"SmartLLM: enqueued {len(self._pending_timesteps)} timesteps",
            logger=self.logger,
            level="info",
        )

    async def _run_full_pipeline(self) -> Optional[Dict]:
        """Full three-stage pipeline: decompose -> coalition -> allocate -> convert.

        Returns:
            Full dispatcher_result dict (all timesteps), or None on planning failure.
        """
        context = self._context
        wmm = self._get_world_model_manager(context)

        step = self._metrics["total_steps"]
        dlog(
            f"SmartLLMPlanningLayer: generate_plan step={step}",
            logger=self.logger,
            level="info",
        )

        # Record replan metrics: each full pipeline after the first plan counts as a full replan
        if step > 0:
            self._metrics["replans_full"] += 1

        # Get feedback text (used during replanning)
        feedback_text = self.feedback_processor.get_feedback_text()

        # Update agent observation state
        self.decomposition_agent.known_nodes = wmm.known_nodes
        self.decomposition_agent.known_edges = wmm.known_edges
        self.coalition_agent.known_nodes = wmm.known_nodes
        self.coalition_agent.known_edges = wmm.known_edges

        # ---- Stage 1: Task decomposition ----
        if feedback_text:
            self.decomposition_agent.set_feedback(feedback_text)
        else:
            self.decomposition_agent.set_feedback(None)

        try:
            await self.decomposition_agent.run(auto_next=False)
        except Exception as e:
            logger.error(f"TaskDecompositionAgent failed: {e}")
            self._metrics["llm_calls"] += 1
            self._metrics["llm_calls_decomposition"] += 1
            return None

        self._metrics["llm_calls"] += 1
        self._metrics["llm_calls_decomposition"] += 1

        decomposition = self.decomposition_agent.last_decomposition
        if decomposition is None:
            logger.warning("TaskDecompositionAgent returned no decomposition.")
            return None

        decomposition_text = json.dumps(decomposition, ensure_ascii=False)
        dlog(
            f"SmartLLM Stage 1: decomposition = {decomposition_text[:300]}",
            logger=self.logger,
            level="info",
        )

        # ---- Stage 2: Coalition formation ----
        self.coalition_agent.set_decomposition(decomposition_text)

        try:
            await self.coalition_agent.run(auto_next=False)
        except Exception as e:
            logger.error(f"CoalitionFormationAgent failed: {e}")
            self._metrics["llm_calls"] += 1
            self._metrics["llm_calls_coalition"] += 1
            return None

        self._metrics["llm_calls"] += 1
        self._metrics["llm_calls_coalition"] += 1

        coalition_text = self.coalition_agent.last_coalition
        if coalition_text is None:
            logger.warning("CoalitionFormationAgent returned no coalition.")
            return None

        dlog(
            f"SmartLLM Stage 2: coalition = {coalition_text[:300]}",
            logger=self.logger,
            level="info",
        )

        # ---- Stage 3: Task allocation ----
        _t_alloc = time.time()
        self.allocation_agent.set_inputs(
            decomposition_result=decomposition_text,
            coalition_result=coalition_text,
        )

        try:
            await self.allocation_agent.run(auto_next=False)
        except Exception as e:
            logger.error(f"TaskAllocationAgent failed: {e}")
            self._metrics["llm_calls"] += 1
            self._metrics["llm_calls_allocation"] += 1
            self._metrics["allocation_durations"].append(round(time.time() - _t_alloc, 6))
            return None

        self._metrics["llm_calls"] += 1
        self._metrics["llm_calls_allocation"] += 1
        self._metrics["allocation_durations"].append(round(time.time() - _t_alloc, 6))

        allocation = self.allocation_agent.last_allocation
        if allocation is None:
            logger.warning("TaskAllocationAgent returned no allocation.")
            return None

        dlog(
            f"SmartLLM Stage 3: allocation = {allocation}",
            logger=self.logger,
            level="info",
        )

        # ---- Convert to dispatcher_result ----
        dispatcher_result = self._convert_plan_to_dispatcher_result(allocation, step)

        self._initial_plan_done = True
        self._metrics["total_steps"] = step + 1
        return dispatcher_result

    async def process_feedback(
        self,
        outcomes: List[Dict],
        newcase_events: List[Dict],
        context,
    ) -> None:
        """No-op; feedback is handled by feedback_processor.

        SmartLLM uses the feedback_processor path (same as SPINE/LipLLM),
        driven by UnifiedTaskSolver's _evaluate_execution_result.
        """
        pass

    def is_task_completed(self) -> bool:
        """Always returns False; task completion is determined by WorldModelLayer."""
        return False

    def reset(self) -> None:
        """Reset internal state."""
        if self.decomposition_agent:
            self.decomposition_agent.reset()
        if self.coalition_agent:
            self.coalition_agent.reset()
        if self.allocation_agent:
            self.allocation_agent.reset()
        self.feedback_processor.reset()
        self._initial_plan_done = False
        self._active_robot_labels = []
        self._pending_timesteps = []
        self._metrics = {
            "success": False,
            "total_steps": 0,
            "llm_calls": 0,
            "llm_calls_decomposition": 0,
            "llm_calls_coalition": 0,
            "llm_calls_allocation": 0,
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

    def _convert_plan_to_dispatcher_result(
        self,
        plan: List[List[str]],
        step_num: int,
    ) -> Optional[Dict]:
        """Convert timestep-serialized plan to dispatcher_result format.

        plan is a list where each element is a timestep (sub-list),
        and each sub-list element is a "robot_label:skill" string.
        Timesteps execute sequentially; skills within a timestep execute in parallel.

        Args:
            plan: [["UGV-1:navigate<campus-2>", "Humanoid-1:navigate<campus-2>"],
                   ["Humanoid-1:place<medical_supply>_on<UGV-1>"], ...]
                  or empty list [] meaning no actions needed
            step_num: Current step number.

        Returns:
            dispatcher_result dict (with timestep_skills),
            empty dict {} if plan is empty (task complete, no actions needed),
            None on conversion failure.
        """
        if not plan:
            return {}

        task_id = f"smartllm_step_{step_num}"

        timestep_skills: Dict[str, Dict[str, Dict[str, Any]]] = {}

        for ts_idx, timestep in enumerate(plan):
            ts_key = str(ts_idx)
            ts_skills: Dict[str, Dict[str, Any]] = {}

            for entry in timestep:
                if ":" not in entry:
                    logger.warning(f"Malformed entry '{entry}' in timestep {ts_idx}, skipping.")
                    continue

                robot_label, skill_str = entry.split(":", 1)
                robot_label = robot_label.strip()
                skill_str = skill_str.strip()

                # Normalize skill string via ActionConverter
                parsed = self.action_converter.parse_single_action(skill_str)
                if parsed == "sync_wait" and skill_str.lower() not in (
                    "sync_wait", "stay idle", "done", "wait", ""
                ):
                    logger.warning(
                        f"ActionConverter could not parse '{skill_str}', "
                        f"using original string as fallback."
                    )
                    parsed = skill_str

                ts_skills[robot_label] = {
                    "skill_str": parsed,
                    "task_id": task_id,
                }

                # Track active robots
                if parsed.lower() not in ("sync_wait", "stay idle", "done", "wait", ""):
                    if robot_label not in self._active_robot_labels:
                        self._active_robot_labels.append(robot_label)

            if ts_skills:
                timestep_skills[ts_key] = ts_skills

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
        if self.decomposition_agent:
            self.decomposition_agent.robot_labels = labels
        if self.coalition_agent:
            self.coalition_agent.robot_labels = labels
        if self.allocation_agent:
            self.allocation_agent.robot_labels = labels
