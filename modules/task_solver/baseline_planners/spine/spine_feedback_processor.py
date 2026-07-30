# -*- coding: utf-8 -*-
"""
SPINEFeedbackProcessor - SPINE-specific feedback processor

Inherits BaseFeedbackProcessor, adds SPINE-specific logic:
- process_outcome_event: Process outcomes; replanning strategy is always FULL
- _decide_strategy: Always returns ReplanningStrategy.FULL
- prepare_feedback_data: Extract feedback from replan signals, format as update message for agent
"""
import logging
from typing import Any, Dict, List, Optional

from modules.task_solver.sgi_planner.base_feedback_processor import (
    BaseFeedbackProcessor,
    ReplanningStrategy,
)

logger = logging.getLogger(__name__)


class SPINEFeedbackProcessor(BaseFeedbackProcessor):
    """SPINE feedback processor — inherits base class, adds SPINE-specific logic.

    SPINE-specific methods:
    - process_outcome_event: SPINE version, replanning strategy is always FULL
    - prepare_feedback_data: Format feedback as update message (supports both outcomes and newcase sources)
    """

    def __init__(self, logger=None, context=None):
        super().__init__(logger=logger or logging.getLogger(__name__), context=context)

    def _decide_strategy(self, event: Dict[str, Any]) -> ReplanningStrategy:
        """SPINE always returns FULL replanning."""
        return ReplanningStrategy.FULL

    def prepare_feedback_data(
        self,
        world_model_manager: Any = None,
        robot_labels: Optional[List[str]] = None,
        planning_agent: Any = None,
        **kwargs,
    ) -> None:
        """Prepare feedback data for SPINE.

        Drains replan signals internally, extracts event info and outcomes,
        formats as update message and passes to SPINEPlanningAgent.

        Event info is extracted per-signal; outcomes are taken once (shared across signals).

        Args:
            world_model_manager: WorldModelManager instance
            robot_labels: List of available robot labels
            planning_agent: SPINEPlanningAgent instance
        """
        consumed = self.drain_replan_signals()
        if not consumed:
            return

        event_lines = []
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

        # Process outcomes once
        outcome_lines = []
        for outcome in outcomes:
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

        # Assemble: event lines first, then outcomes
        all_lines = event_lines + outcome_lines
        if all_lines:
            update_message = "Execution results:\n" + "\n".join(all_lines)
        else:
            update_message = "Execution results:\n  No execution results available."

        if planning_agent:
            planning_agent.set_request(f"updates: {update_message}")

    def process_outcome_event(
        self,
        outcomes: List[Dict[str, Any]],
        status: str,
        goal_completed: bool,
        world_model_manager: Any,
    ) -> None:
        """Process execution results, always set FULL replanning strategy.

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

        # SPINE always requests FULL replanning
        self.replanning_requested = True
        self.replanning_strategy = ReplanningStrategy.FULL

        # No replan needed when goal is completed
        if goal_completed:
            self.replanning_strategy = ReplanningStrategy.NONE
            self.replanning_requested = False

        # Build evaluation event
        reason = "goal_achieved" if goal_completed else (
            "plan_completed_but_goal_unachieved" if status == "completed"
            else "execution_failed"
        )
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
