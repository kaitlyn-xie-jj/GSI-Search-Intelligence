# -*- coding: utf-8 -*-
"""
SPINEPlanningLayer - SPINE planning layer implementing BaselinePlanner interface

Encapsulates single-agent multi-turn dialog planning logic,
driven by UnifiedTaskSolver via generate_plan and process_feedback.
SPINE uses a single LLM Agent for scene-graph-based reasoning,
achieving iterative plan-feedback loops through multi-turn dialog.
"""
import logging
import time
from typing import Any, Dict, List, Optional

from modules.task_solver.baseline_planners.base_planner import BaselinePlanner
from modules.task_solver.baseline_planners.spine.agents.spine_planning_agent import (
    SPINEPlanningAgent,
)
from modules.task_solver.baseline_planners.spine.spine_plan_validator import (
    SPINEPlanValidator,
)
from modules.task_solver.baseline_planners.common.action_converter import ActionConverter
from modules.task_solver.baseline_planners.spine.spine_feedback_processor import (
    SPINEFeedbackProcessor,
)
from modules.task_solver.sgi_planner.base_feedback_processor import (
    ReplanningStrategy,
)
from modules.task_solver.llm_framework.core.context import WorkflowContext
from modules.task_solver.llm_framework.file import Logger
from modules.utils.system.logging_utils import dlog

logger = logging.getLogger(__name__)


class SPINEPlanningLayer(BaselinePlanner):
    """SPINE planning layer — encapsulates single-agent multi-turn dialog planning.

    Implements BaselinePlanner interface, driven by UnifiedTaskSolver:
    - generate_plan: SPINEPlanningAgent -> SPINEPlanValidator validate -> ActionConverter convert -> dispatcher_result
    - feedback_processor: Called by UnifiedTaskSolver in _evaluate_execution_result to process outcomes
    - feedback_processor.prepare_feedback_data: Called before _execute_planning, formats update message for agent

    Attributes:
        max_steps: Maximum step limit.
        planning_agent: SPINE planning agent instance.
        plan_validator: SPINE plan validator.
        action_converter: Action format converter (reuses common/).
        feedback_processor: SPINE-specific feedback processor.
    """

    def __init__(
        self,
        max_steps: int = 30,
        world_model_manager=None,
        context: Optional[WorkflowContext] = None,
        robot_labels: Optional[List[str]] = None,
        logger: Logger = None,
        model_family: str = None,
        model_name_override: str = None,
        n_attempts: int = 3,
        use_few_shot: bool = True,
    ):
        self.max_steps = max_steps
        self.logger = logger
        self._world_model_manager = world_model_manager
        self._context = context

        # Robot labels
        self.robot_labels: List[str] = robot_labels or []

        # Internal components
        self.plan_validator = SPINEPlanValidator()
        self.action_converter = ActionConverter(step_prefix="spine_step")
        self.feedback_processor = SPINEFeedbackProcessor(context=context)

        # Agent (lazy init)
        self._model_family = model_family
        self._model_name_override = model_name_override
        self._n_attempts = n_attempts
        self._use_few_shot = use_few_shot
        self._agent_initialized = False
        self.planning_agent: Optional[SPINEPlanningAgent] = None

        # Control flags
        self._initial_plan_done = False
        self._active_robot_labels: List[str] = []
        self._replanning_strategy: str = "FULL"  # SPINE always uses FULL replanning

        # Metrics
        self._metrics: Dict[str, Any] = {
            "success": False,
            "total_steps": 0,
            "llm_calls": 0,
            "llm_calls_effective": 0,
            "plan_parse_failures": 0,
            "action_parse_failures": 0,
            "replans_full": 0,
            "replans_partial": 0,
            "planning_durations": [],
            "allocation_durations": [],
        }

    # =========================================================================
    # Agent initialization
    # =========================================================================

    def _ensure_agent(self, context: WorkflowContext = None) -> None:
        """Ensure agent is initialized (lazy init, created on first call)."""
        if self._agent_initialized:
            return

        ctx = context or self._context
        if ctx is None:
            raise RuntimeError("SPINEPlanningLayer requires a WorkflowContext")

        self._context = ctx

        self.planning_agent = SPINEPlanningAgent(
            logger_=self.logger,
            context=ctx,
            robot_labels=self.robot_labels,
            n_attempts=self._n_attempts,
            model_family=self._model_family,
            model_name_override=self._model_name_override,
            use_few_shot=self._use_few_shot,
        )
        self._agent_initialized = True

    # =========================================================================
    # BaselinePlanner interface
    # =========================================================================

    async def generate_plan(self) -> Optional[Dict]:
        """Generate a plan.

        Flow:
        1. On initial planning, set base_request to user instruction
        2. Update agent observation state
        3. Call SPINEPlanningAgent.generate_plan to get LLM response
        4. SPINEPlanValidator validates plan format (done inside agent)
        5. ActionConverter converts per-robot skill strings to dispatcher_result
        6. Return dispatcher_result

        Returns:
            dispatcher_result dict (on success), or None (on planning failure).
        """
        _t0 = time.time()
        result = await self._generate_plan_inner()
        _dur = round(time.time() - _t0, 6)
        self._metrics["planning_durations"].append(_dur)
        # SPINE decomposition and allocation are done in one LLM step; allocation duration equals planning duration
        self._metrics["allocation_durations"].append(_dur)
        return result

    async def _generate_plan_inner(self) -> Optional[Dict]:
        """Actual logic for generate_plan (wrapped with timing by generate_plan)."""
        context = self._context
        self._ensure_agent(context)
        wmm = self._get_world_model_manager(context)

        step = self._metrics["total_steps"]
        dlog(f"SPINEPlanningLayer: generate_plan step={step}",
             logger=self.logger, level='info')

        # Record replan metrics: after initial plan, record based on feedback_processor strategy
        if self._initial_plan_done:
            strategy = self.feedback_processor.replanning_strategy
            if strategy == ReplanningStrategy.FULL:
                self._metrics["replans_full"] += 1
            elif strategy == ReplanningStrategy.PARTIAL:
                self._metrics["replans_partial"] += 1

        # On initial planning, set user instruction as agent's base_request
        if not self._initial_plan_done:
            instruction = context._generated_text.get("instruction", "")
            self.planning_agent.base_request = instruction

        # Update agent observation state
        self.planning_agent.set_state(
            known_nodes=wmm.known_nodes,
            known_edges=wmm.known_edges,
            robot_labels=self.robot_labels,
        )

        # Call agent to generate plan (includes retry and validation logic)
        plan, success, logs = await self.planning_agent.generate_plan(
            plan_validator=self.plan_validator,
        )

        # llm_calls_total: count all LLM attempts (1 initial + retries from logs)
        num_attempts = len(logs) + (1 if success else 0)
        self._metrics["llm_calls"] += num_attempts

        if not success or plan is None:
            self._metrics["plan_parse_failures"] += 1
            dlog(
                f"SPINE planning failed. Logs: {logs}",
                logger=self.logger, level='error',
            )
            return None

        # llm_calls_effective: only count successful planning rounds
        self._metrics["llm_calls_effective"] += 1

        # Convert per-robot plan to dispatcher_result
        dispatcher_result = self._convert_plan_to_dispatcher_result(plan, step)
        if dispatcher_result is None:
            self._metrics["action_parse_failures"] += 1
            return None

        self._initial_plan_done = True
        self._metrics["total_steps"] = step + 1
        return dispatcher_result

    async def process_feedback(
        self,
        exec_result: Dict[str, Any],
        newcase_events: List[Dict],
        context,
    ) -> None:
        """Process execution feedback (SPINE does not use this interface).

        SPINE feedback processing is handled uniformly in UnifiedTaskSolver's _evaluate_execution_result:
        - Outcomes and newcase events are processed by feedback_processor in the solver
        - Update message formatting and set_request are done in feedback_processor.prepare_feedback_data
        """
        pass

    def is_task_completed(self) -> bool:
        """Determine task completion.

        SPINE does not self-determine task completion; always returns False.
        Task completion is determined by world_model_layer.is_goal_completed() in UnifiedTaskSolver.
        """
        return False

    def reset(self) -> None:
        """Reset internal state."""
        if self.planning_agent:
            self.planning_agent.reset_history()
        self.feedback_processor.reset()
        self._initial_plan_done = False
        self._active_robot_labels = []
        self._replanning_strategy = "FULL"
        self._metrics = {
            "success": False,
            "total_steps": 0,
            "llm_calls": 0,
            "llm_calls_effective": 0,
            "plan_parse_failures": 0,
            "action_parse_failures": 0,
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
        """Convert timestep list plan to dispatcher_result format.

        plan is a list where each element is a timestep (sub-list),
        and each sub-list element is a "robot_label:skill" string.
        Timesteps execute sequentially; skills within a timestep execute in parallel.

        Args:
            plan: [["UAV-1:take_off", "UGV-1:navigate<loc>"], ["UAV-1:search<a>_for<t>"], ...]
                  or empty list [] meaning no actions needed
            step_num: Current step number.

        Returns:
            dispatcher_result dict (with timestep_skills),
            empty dict {} if plan is empty (task complete, no actions needed),
            None on conversion failure (parse errors, etc.).
        """
        if not plan:
            return {}

        task_id = f"spine_step_{step_num}"
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
        return getattr(context, '_world_model_manager', None)

    def set_world_model_manager(self, wmm) -> None:
        """Set WorldModelManager reference (called by UnifiedTaskSolver)."""
        self._world_model_manager = wmm

    def set_robot_labels(self, labels: List[str]) -> None:
        """Set robot label list (called by factory or UnifiedTaskSolver)."""
        self.robot_labels = labels
        if self.planning_agent:
            self.planning_agent.robot_labels = labels
