# -*- coding: utf-8 -*-
"""
LLaMAR Feedback Collector - Aggregate platform execution results and unexpected event feedback

Collects feedback from two sources:
1. Outcome list returned by the platform execution layer (normal execution results)
2. Unexpected event list (newcase events)

Aggregates both into per-robot feedback strings for Action Agent and Verifier Agent.
"""
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class FeedbackCollector:
    """Aggregate normal execution results and unexpected event feedback.

    Usage:
    1. Newcase events are passed via the ``_pending_newcase_events`` list (filled by caller).
    2. After each step, call ``collect_feedback`` with outcomes to get merged
       per-robot feedback dictionary.
    """

    def __init__(self):
        self._pending_newcase_events: List[Dict[str, Any]] = []

    def collect_feedback(
        self,
        outcomes: List[Dict[str, Any]],
        robot_labels: List[str],
        active_robot_labels: Optional[List[str]] = None,
    ) -> Dict[str, str]:
        """Aggregate outcomes and newcase events into per-robot feedback strings.

        For each robot:
        - Extract execution result summary from outcomes.
        - Extract event descriptions from pending newcase events.
        - Concatenate both parts into a single feedback string.

        Automatically clears consumed newcase events after the call.

        Args:
            outcomes: Outcome list returned by the platform execution layer.
            robot_labels: All robot labels in the current scene.
            active_robot_labels: Subset of robots dispatched in the previous step.

        Returns:
            ``{robot_label: feedback_string}`` dictionary.
        """
        target_robots = active_robot_labels if active_robot_labels else robot_labels
        per_robot: Dict[str, List[str]] = {label: [] for label in target_robots}

        # --- 1. Process outcomes ---
        for outcome in outcomes:
            meta = outcome.get("meta") or {}
            data = outcome.get("data") or {}

            robot = meta.get("robot_label", "")
            skill = meta.get("skill", data.get("skill", "unknown"))
            success = meta.get("success")
            if success is None:
                success = data.get("success", True)
            message = meta.get("message", data.get("message", ""))

            status = "Success" if success else "Failed"
            parts = [f"{skill} - {status}"]
            if message:
                parts.append(str(message))
            feedback_line = ": ".join(parts)

            if robot in per_robot:
                per_robot[robot].append(feedback_line)
            elif not robot or robot not in robot_labels:
                for label in target_robots:
                    per_robot[label].append(feedback_line)

        # --- 2. Process newcase events ---
        newcase_events = list(self._pending_newcase_events)
        self._pending_newcase_events.clear()

        for evt in newcase_events:
            details = evt.get("details") or {}
            message = details.get("message") or evt.get("message", "Unknown event")
            robot_info = details.get("robot") or evt.get("robot") or {}
            robot_label = robot_info.get("label", "")

            event_text = f"Unexpected event: {message}"

            if robot_label and robot_label in per_robot:
                per_robot[robot_label].append(event_text)
            else:
                for label in target_robots:
                    per_robot[label].append(event_text)

        # --- 3. Assemble final strings ---
        result: Dict[str, str] = {}
        for label in target_robots:
            lines = per_robot[label]
            result[label] = "; ".join(lines) if lines else "No feedback"

        return result
