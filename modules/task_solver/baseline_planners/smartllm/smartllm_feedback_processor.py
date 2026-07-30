# -*- coding: utf-8 -*-
"""
SmartLLMFeedbackProcessor - SmartLLM-specific feedback processor

Inherits BaseFeedbackProcessor, adds SmartLLM-specific logic:
- process_outcome_event: Distinguishes "dispatch next timestep" vs "full replan"
- _decide_strategy: Always returns ReplanningStrategy.FULL
- prepare_feedback_data: Extract feedback from replan signals, format as text for TaskDecompositionAgent
"""
import json
import logging
from typing import Any, Dict, List, Optional

from modules.task_solver.sgi_planner.base_feedback_processor import (
    BaseFeedbackProcessor,
    ReplanningStrategy,
)

logger = logging.getLogger(__name__)


class SmartLLMFeedbackProcessor(BaseFeedbackProcessor):
    """SmartLLM feedback processor — inherits base class, adds SmartLLM-specific logic.

    Supports timestep queue management:
    - Accumulate outcomes per timestep
    - Distinguish "dispatch next timestep" vs "full replan"
    - Include previous decomposition in feedback text
    """

    def __init__(self, logger=None, context=None):
        super().__init__(logger=logger or logging.getLogger(__name__), context=context)
        self._feedback_text: Optional[str] = None
        self._planning_layer_ref = None
        self._newcase_triggered: bool = False

    def set_planning_layer(self, planning_layer) -> None:
        """Set reference to SmartLLMPlanningLayer for querying timestep queue state."""
        self._planning_layer_ref = planning_layer

    def reset(self):
        """Reset processor state. Does not clear accumulated outcomes or newcase flag (preserved across timesteps)."""
        super().reset()
        self._feedback_text = None

    def reset_full(self):
        """Full reset including accumulated outcomes and newcase flag."""
        self.reset()
        self._accumulated_outcomes = []
        self._newcase_triggered = False

    def _decide_strategy(self, event: Dict[str, Any]) -> ReplanningStrategy:
        """SmartLLM always returns FULL replanning."""
        return ReplanningStrategy.FULL

    def prepare_feedback_data(
        self,
        world_model_manager: Any = None,
        robot_labels: Optional[List[str]] = None,
        **kwargs,
    ) -> None:
        """Prepare feedback data for SmartLLM.

        Called at the start of each main loop iteration, before generate_plan.

        Three scenarios:
        - No signals (first plan): do nothing
        - Signals present but pending steps remain (dispatch next timestep): consume signals, no feedback text, keep outcomes
        - Signals present and no pending steps or newcase (actual replan): consume signals, generate feedback text, clear outcomes

        Args:
            world_model_manager: WorldModelManager instance
            robot_labels: List of available robot labels
        """
        consumed = self.drain_replan_signals()
        if not consumed:
            return

        # Detect newcase signals
        has_newcase = any(sig.get("source") == "newcase" for sig in consumed)
        if has_newcase:
            self._newcase_triggered = True

        # Determine if full replan is needed
        has_pending = (
            self._planning_layer_ref is not None
            and self._planning_layer_ref.has_pending_steps()
        )
        needs_full_replan = has_newcase or not has_pending

        if not needs_full_replan:
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

            if not outcomes:
                outcomes = details.get("outcomes") or []

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

        # Previous decomposition
        prev_decomposition = (
            self._planning_layer_ref.decomposition_agent.last_decomposition
            if self._planning_layer_ref
            and self._planning_layer_ref.decomposition_agent
            and self._planning_layer_ref.decomposition_agent.last_decomposition
            else None
        )
        if prev_decomposition:
            parts.append(f"Previous decomposition:\n  {json.dumps(prev_decomposition, ensure_ascii=False)}")

        if outcome_lines:
            parts.append("Execution feedback:\n" + "\n".join(outcome_lines))
        if event_lines:
            parts.append("New events:\n" + "\n".join(event_lines))

        if parts:
            self._feedback_text = "\n".join(parts)
        else:
            self._feedback_text = None

        # Full replan: clear accumulated outcomes
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

        Args:
            outcomes: List of outcomes returned by the platform execution layer.
            status: Execution status.
            goal_completed: Whether the goal has been achieved.
            world_model_manager: WorldModelManager instance.
        """
        if self.last_event is not None:
            return

        self._process_outcomes(outcomes, world_model_manager)

        if outcomes:
            self._accumulated_outcomes.extend(outcomes)

        if goal_completed:
            self.replanning_strategy = ReplanningStrategy.NONE
            self.replanning_requested = False
            self._build_evaluation_event(outcomes, status, goal_completed, "goal_achieved")
            return

        has_pending = (
            self._planning_layer_ref is not None
            and self._planning_layer_ref.has_pending_steps()
        )

        if has_pending:
            self.replanning_requested = True
            self.replanning_strategy = ReplanningStrategy.FULL
            self._build_evaluation_event(
                outcomes, status, goal_completed, "timestep_completed_pending_next"
            )
        else:
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
        """Build evaluation event."""
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
