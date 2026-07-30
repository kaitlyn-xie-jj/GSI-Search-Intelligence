# -*- coding: utf-8 -*-
"""
LipLLMFeedbackProcessor - LipLLM-specific feedback processor

Inherits BaseFeedbackProcessor, adds LipLLM-specific logic:
- process_outcome_event: Process outcomes; replanning strategy is always FULL
- _decide_strategy: Always returns ReplanningStrategy.FULL
- prepare_feedback_data: Extract feedback from replan signals, format as text for SkillListAgent
"""
import logging
from typing import Any, Dict, List, Optional

from modules.task_solver.sgi_planner.base_feedback_processor import (
    BaseFeedbackProcessor,
    ReplanningStrategy,
)

logger = logging.getLogger(__name__)


class LipLLMFeedbackProcessor(BaseFeedbackProcessor):
    """LipLLM feedback processor — inherits base class, adds LipLLM-specific logic.

    LipLLM-specific methods:
    - process_outcome_event: LipLLM version, distinguishes "dispatch next timestep" vs "full replan"
    - prepare_feedback_data: Drain signals, format as feedback text for SkillListAgent
    - get_feedback_text: Get formatted feedback text
    - accumulate_outcomes: Accumulate outcomes per timestep for use during full replan
    """

    def __init__(self, logger=None, context=None):
        super().__init__(logger=logger or logging.getLogger(__name__), context=context)
        self._feedback_text: Optional[str] = None
        self._planning_layer_ref = None  # Set externally, used to query has_pending_steps
        self._newcase_triggered: bool = False  # Whether this round's replan was triggered by newcase

    def set_planning_layer(self, planning_layer) -> None:
        """Set reference to LipLLMPlanningLayer for querying timestep queue state."""
        self._planning_layer_ref = planning_layer

    def reset(self):
        """Reset processor state including feedback text. Does not clear accumulated outcomes or newcase flag (preserved across timesteps)."""
        super().reset()
        self._feedback_text = None

    def reset_full(self):
        """Full reset including accumulated outcomes and newcase flag (called after full replan completes)."""
        self.reset()
        self._accumulated_outcomes = []
        self._newcase_triggered = False

    def _decide_strategy(self, event: Dict[str, Any]) -> ReplanningStrategy:
        """LipLLM always returns FULL replanning."""
        return ReplanningStrategy.FULL

    def prepare_feedback_data(
        self,
        world_model_manager: Any = None,
        robot_labels: Optional[List[str]] = None,
        planning_layer: Any = None,
        **kwargs,
    ) -> None:
        """Prepare feedback data for LipLLM.

        Called at the start of each main loop iteration, before generate_plan.

        Responsibilities:
        1. Consume replan_signals, detect newcase -> set _newcase_triggered flag
        2. Format feedback text for SkillListAgent (only meaningful during actual replan)
        3. Manage clearing of _accumulated_outcomes

        Three scenarios:
        - No signals (first plan): do nothing
        - Signals present but pending steps remain (dispatch next timestep): consume signals, no feedback text, keep outcomes
        - Signals present and no pending steps or newcase (actual replan): consume signals, generate feedback text, clear outcomes

        Args:
            world_model_manager: WorldModelManager instance
            robot_labels: List of available robot labels
            planning_layer: LipLLMPlanningLayer instance
        """
        consumed = self.drain_replan_signals()
        if not consumed:
            return

        # Detect newcase signals
        has_newcase = any(sig.get("source") == "newcase" for sig in consumed)
        if has_newcase:
            self._newcase_triggered = True

        # Determine if full replan is needed (newcase or no pending steps)
        has_pending = (
            self._planning_layer_ref is not None
            and self._planning_layer_ref.has_pending_steps()
        )
        needs_full_replan = has_newcase or not has_pending

        if not needs_full_replan:
            # Just dispatching next timestep, no feedback text needed, keep accumulated outcomes
            self._feedback_text = None
            return

        # Full replan path: generate feedback text
        event_lines = []
        outcome_lines = []
        outcomes = []

        for sig in consumed:
            source = sig.get("source", "")
            payload = sig.get("payload") or {}
            details = payload.get("details") or {}

            # Only take outcomes from the first signal (all signals share the same execution cycle)
            if not outcomes:
                outcomes = details.get("outcomes") or []

            # Newcase event: extract event info
            if source == "newcase":
                robot_info = details.get("robot") or {}
                robot_label = robot_info.get("label", "unknown")
                skill = payload.get("skill", details.get("payload", {}).get("skill", "unknown"))
                message = payload.get("message", details.get("message", ""))
                event_type = payload.get("type", "unknown")

                line = f"  {robot_label}: {skill} - Unexpected event ({event_type})"
                if message:
                    line += f" ({message})"
                event_lines.append(line)

        # Use accumulated outcomes (includes results from all executed timesteps)
        all_outcomes = self._accumulated_outcomes if self._accumulated_outcomes else outcomes

        for outcome in all_outcomes:
            meta = outcome.get("meta") or {}
            data = outcome.get("data") or {}

            robot = meta.get("robot_label", data.get("agent_id", "unknown"))
            skill = meta.get("skill", data.get("skill", "unknown"))
            success = meta.get("success")
            if success is None:
                success = data.get("success", True)
            message = meta.get("message", data.get("message", ""))

            status_str = "Success" if success else "Failed"
            line = f"  {robot}: {skill} - {status_str}"
            if message:
                line += f" ({message})"
            outcome_lines.append(line)

        # Assemble feedback text
        parts = []

        # Previous skill list
        prev_skills = (
            self._planning_layer_ref._skill_list
            if self._planning_layer_ref and self._planning_layer_ref._skill_list
            else []
        )
        if prev_skills:
            skills_str = ", ".join(prev_skills)
            parts.append(f"Previous plan:\n  [{skills_str}]")

        if outcome_lines:
            parts.append("Execution feedback:\n" + "\n".join(outcome_lines))
        if event_lines:
            parts.append("New events:\n" + "\n".join(event_lines))

        if parts:
            self._feedback_text = "\n".join(parts)
        else:
            self._feedback_text = None

        # Full replan: clear accumulated outcomes (already included in feedback text)
        self._accumulated_outcomes = []

    def get_feedback_text(self) -> Optional[str]:
        """Get formatted feedback text."""
        return self._feedback_text

    def process_outcome_event(
        self,
        outcomes: List[Dict[str, Any]],
        status: str,
        goal_completed: bool,
        world_model_manager: Any,
    ) -> None:
        """Process execution results, distinguishing "dispatch next timestep" vs "full replan".

        Logic:
        - If there is an unconsumed event (e.g. newcase_event), skip
        - Accumulate current outcomes
        - If goal completed -> no replan needed
        - If pending timesteps remain -> set replanning_requested=True to continue main loop,
          but this is not a real replan, just dispatching the next timestep
        - If no pending timesteps -> actual full replan

        Args:
            outcomes: List of outcomes returned by the platform execution layer.
            status: Execution status (e.g. 'completed', 'failed').
            goal_completed: Whether the goal has been achieved.
            world_model_manager: WorldModelManager instance.
        """
        # Skip if there is an unconsumed event (e.g. newcase_event)
        if self.last_event is not None:
            return

        # Process outcomes and update runtime_params
        self._process_outcomes(outcomes, world_model_manager)

        # Accumulate current outcomes
        if outcomes:
            self._accumulated_outcomes.extend(outcomes)

        # No replan needed when goal is completed
        if goal_completed:
            self.replanning_strategy = ReplanningStrategy.NONE
            self.replanning_requested = False
            self._build_evaluation_event(outcomes, status, goal_completed, "goal_achieved")
            return

        # Check if there are pending timesteps
        has_pending = (
            self._planning_layer_ref is not None
            and self._planning_layer_ref.has_pending_steps()
        )

        if has_pending:
            # Pending timesteps remain: set replanning_requested=True to continue main loop
            # This is not a real replan, just dispatching the next timestep
            self.replanning_requested = True
            self.replanning_strategy = ReplanningStrategy.FULL
            self._build_evaluation_event(
                outcomes, status, goal_completed, "timestep_completed_pending_next"
            )
        else:
            # All timesteps executed: actual full replan
            self.replanning_requested = True
            self.replanning_strategy = ReplanningStrategy.FULL
            reason = (
                "plan_completed_but_goal_unachieved" if status == "completed"
                else "execution_failed"
            )
            self._build_evaluation_event(outcomes, status, goal_completed, reason)

    def _build_evaluation_event(
        self,
        outcomes: List[Dict[str, Any]],
        status: str,
        goal_completed: bool,
        reason: str,
    ) -> None:
        """Build evaluation event (internal helper)."""
        self.last_event = {
            "type": "EVALUATION_TRIGGER",
            "message": f"Execution: status={status}, goal_completed={goal_completed}",
            "severity": "info",
            "category": "evaluation",
            "details": {
                "status": status,
                "goal_completed": goal_completed,
                "reason": reason,
                "discovery_feedback": self.last_discovery_feedback,
                "outcomes": outcomes,
            },
        }
