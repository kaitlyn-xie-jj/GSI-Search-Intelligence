"""Common interface implemented by all search policies."""

from abc import ABC, abstractmethod
from typing import Any, Mapping, Optional, Tuple

from ..contracts import SearchState, Viewpoint


class SearchPolicy(ABC):
    """Select observation viewpoints from a platform-neutral search state."""

    @abstractmethod
    def plan(self, state: SearchState) -> Tuple[Viewpoint, ...]:
        """Return the remaining ordered viewpoints for the current state."""

    def select_next(self, state: SearchState) -> Optional[Viewpoint]:
        """Return the next viewpoint, or ``None`` when the policy is complete."""
        if state.exhausted_budget is not None:
            return None
        remaining = self.plan(state)
        return remaining[0] if remaining else None

    def decision_metadata(
        self,
        state: SearchState,
        viewpoint: Viewpoint,
    ) -> Mapping[str, Any]:
        """Return common diagnostics for a selected viewpoint."""
        return {
            "policy_name": type(self).__name__,
            "step_index": state.step_index,
            "selected_viewpoint_key": viewpoint.key,
        }
