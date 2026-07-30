# -*- coding: utf-8 -*-
"""
LLaMAR Prompt Utilities - Shared prompt formatting helper functions
"""
from typing import Dict, List, Optional


def format_subtasks(
    open_subtasks: Optional[List[str]],
    closed_subtasks: Optional[List[str]],
) -> str:
    """Format open/closed subtask lists into prompt text.

    Args:
        open_subtasks: Pending subtask list; None means no plan generated yet.
        closed_subtasks: Completed subtask list; None means nothing completed yet.

    Returns:
        Formatted subtask text.
    """
    open_text = str(open_subtasks) if open_subtasks is not None else "None"
    closed_text = str(closed_subtasks) if closed_subtasks is not None else "None"
    return (
        f"Robots' open subtasks: {open_text}\n"
        f"Robots' completed subtasks: {closed_text}"
    )


def format_feedback(per_robot_feedback: Dict[str, str]) -> str:
    """Format per-robot feedback dictionary into prompt text.

    Args:
        per_robot_feedback: {robot_label: feedback_string} dictionary.

    Returns:
        Formatted feedback text. Returns "No previous feedback." if empty.
    """
    if not per_robot_feedback:
        return "No previous feedback."

    lines = []
    for robot, feedback in per_robot_feedback.items():
        lines.append(f"{robot}: {feedback}")
    return "\n".join(lines)


def format_previous_actions(
    previous_actions: Dict[str, str],
    previous_successes: Dict[str, bool],
    active_robot_labels: Optional[List[str]] = None,
) -> str:
    """Format previous action info into prompt text, including only active robots.

    Args:
        previous_actions: {robot_label: action_string} dictionary.
        previous_successes: {robot_label: success_bool} dictionary.
        active_robot_labels: Subset of robots dispatched in the previous step.
            If None or empty, returns empty string.

    Returns:
        Formatted previous action text.
    """
    if not previous_actions or not active_robot_labels:
        return ""

    lines = []
    for label in active_robot_labels:
        action = previous_actions.get(label, "unknown action")
        success = previous_successes.get(label, False)
        status = "Success" if success else "Failed"
        lines.append(f"{label}'s previous action: {action} - {status}")
    return "\n".join(lines)
