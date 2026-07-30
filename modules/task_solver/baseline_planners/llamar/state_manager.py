# -*- coding: utf-8 -*-
"""
LLaMAR State Manager - Manages LLaMAR internal state across loop iterations

Maintains open/closed subtasks, memory, action history, etc.,
ensuring each agent receives the correct accumulated state at every step.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class LLaMARState:
    """Data container for LLaMAR internal state."""
    open_subtasks: Optional[List[str]] = None
    closed_subtasks: Optional[List[str]] = None
    memory: str = ""
    previous_actions: Dict[str, str] = field(default_factory=dict)
    previous_successes: Dict[str, bool] = field(default_factory=dict)
    previous_feedback: Dict[str, str] = field(default_factory=dict)
    step_num: int = 0
    current_subtask: str = ""


class LLaMARStateManager:
    """Manages LLaMAR internal state, maintaining consistency across loop iterations."""

    def __init__(self):
        self._state = LLaMARState()

    # -- Property access --

    @property
    def open_subtasks(self) -> Optional[List[str]]:
        return self._state.open_subtasks

    @property
    def closed_subtasks(self) -> Optional[List[str]]:
        return self._state.closed_subtasks

    @property
    def memory(self) -> str:
        return self._state.memory

    @property
    def previous_actions(self) -> Dict[str, str]:
        return self._state.previous_actions

    @property
    def previous_successes(self) -> Dict[str, bool]:
        return self._state.previous_successes

    @property
    def previous_feedback(self) -> Dict[str, str]:
        return self._state.previous_feedback

    @property
    def step_num(self) -> int:
        return self._state.step_num

    @property
    def current_subtask(self) -> str:
        return self._state.current_subtask

    # -- State update methods --

    def update_plan(self, new_plan: List[str]) -> None:
        """Update open_subtasks with the new plan from Planner Agent.

        Args:
            new_plan: List of pending subtasks output by Planner Agent.
        """
        self._state.open_subtasks = list(new_plan)

    def update_completed(self, completed: List[str]) -> None:
        """Move Verifier-reported completed subtasks from open to closed.

        Args:
            completed: List of newly completed subtasks this step.
        """
        if not completed:
            return

        if self._state.closed_subtasks is None:
            self._state.closed_subtasks = []

        completed_set = set(completed)
        self._state.closed_subtasks.extend(
            s for s in completed if s not in set(self._state.closed_subtasks)
        )

        if self._state.open_subtasks is not None:
            self._state.open_subtasks = [
                s for s in self._state.open_subtasks if s not in completed_set
            ]

    def update_memory(self, memory: str) -> None:
        """Update scene memory returned by Action Agent.

        Args:
            memory: New memory string.
        """
        self._state.memory = memory

    def update_current_subtask(self, subtask: str) -> None:
        """Update the currently executing subtask.

        Args:
            subtask: Current subtask description.
        """
        self._state.current_subtask = subtask

    def record_actions(
        self,
        actions: Dict[str, str],
        successes: Dict[str, bool],
    ) -> None:
        """Record each robot's action and execution result for this step.

        Args:
            actions: {robot_label: action_string}
            successes: {robot_label: success_bool}
        """
        self._state.previous_actions = dict(actions)
        self._state.previous_successes = dict(successes)

    def update_feedback(self, feedback: Dict[str, str]) -> None:
        """Update per-robot feedback info.

        Args:
            feedback: {robot_label: feedback_string}
        """
        self._state.previous_feedback = dict(feedback)

    def increment_step(self) -> None:
        """Increment step count by one."""
        self._state.step_num += 1

    def reset(self) -> None:
        """Reset all state to initial defaults, for use before a new task."""
        self._state = LLaMARState()
