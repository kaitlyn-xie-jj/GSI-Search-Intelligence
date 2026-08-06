"""Success-first supervision with a late global-recovery mode."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Tuple

from ..contracts import SearchState, Viewpoint
from .base import SearchPolicy


@dataclass(frozen=True)
class SuccessConstrainedSupervisorPolicy(SearchPolicy):
    """Keep active search unless the remaining budget requires recovery.

    This policy deliberately excludes random and legacy-active fallbacks. It
    changes mode only when the estimated number of executable actions falls to
    a configured reserve while effective global coverage remains insufficient.
    """

    default_policy: SearchPolicy
    recovery_policy: SearchPolicy
    recovery_reserve_actions: int = 2
    required_quality_coverage: float = 0.6
    high_quality_threshold: float = 0.5
    estimated_action_time_s: float = 6.0

    def __post_init__(self) -> None:
        if self.recovery_reserve_actions < 0:
            raise ValueError("recovery_reserve_actions must not be negative")
        for name in ("required_quality_coverage", "high_quality_threshold"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be within [0, 1]")
        if not math.isfinite(self.estimated_action_time_s) or self.estimated_action_time_s <= 0:
            raise ValueError("estimated_action_time_s must be finite and positive")

    def plan(self, state: SearchState) -> Tuple[Viewpoint, ...]:
        mode, _, _ = self._mode(state)
        policy = self.recovery_policy if mode == "global_recovery" else self.default_policy
        plan = policy.plan(state)
        if plan:
            return plan
        if mode == "global_recovery":
            return self.default_policy.plan(state)
        return ()

    def decision_metadata(
        self,
        state: SearchState,
        viewpoint: Viewpoint,
    ) -> Mapping[str, Any]:
        mode, reason, signals = self._mode(state)
        policy = self.recovery_policy if mode == "global_recovery" else self.default_policy
        if not policy.plan(state) and mode == "global_recovery":
            policy = self.default_policy
            mode = "active_search"
            reason = "recovery_route_exhausted"
        previous_mode = state.policy_metadata.get("success_supervisor_mode")
        delegated = dict(policy.decision_metadata(state, viewpoint))
        return {
            "policy_name": type(self).__name__,
            "step_index": state.step_index,
            "selected_viewpoint_key": viewpoint.key,
            "success_supervisor_mode": mode,
            "success_supervisor_previous_mode": previous_mode,
            "success_supervisor_mode_switched": previous_mode not in (None, mode),
            "success_supervisor_reason": reason,
            "success_supervisor_signals": signals,
            "delegated_policy_metadata": delegated,
        }

    def _mode(self, state: SearchState):
        signals = self._signals(state)
        if state.observations and state.observations[-1].detections:
            return "active_search", "positive_detection_confirmation", signals
        if (
            self.recovery_reserve_actions > 0
            and signals["estimated_remaining_actions"]
            <= self.recovery_reserve_actions
            and signals["high_quality_coverage_fraction"]
            < self.required_quality_coverage
        ):
            return "global_recovery", "coverage_reserve_reached", signals
        return "active_search", "success_first_default", signals

    def _signals(self, state: SearchState):
        high_quality_cells = set()
        for observation in state.observations:
            effective_quality = (
                observation.observation_quality
                * observation.visibility_probability
            )
            if effective_quality >= self.high_quality_threshold:
                high_quality_cells.update(observation.visible_cell_ids)
        belief_cells = set(state.belief)
        coverage = (
            len(high_quality_cells & belief_cells) / len(belief_cells)
            if belief_cells else 0.0
        )
        viewpoint_remaining = math.inf
        if state.task.budget.max_viewpoints is not None:
            viewpoint_remaining = max(
                0,
                state.task.budget.max_viewpoints - state.step_index,
            )
        time_remaining_actions = math.inf
        if state.task.budget.time_limit_s is not None:
            time_remaining_actions = max(
                0,
                math.floor(
                    (state.task.budget.time_limit_s - state.elapsed_time_s)
                    / self.estimated_action_time_s
                ),
            )
        estimated_remaining = min(viewpoint_remaining, time_remaining_actions)
        if math.isinf(estimated_remaining):
            estimated_remaining = self.recovery_reserve_actions + 1
        return {
            "estimated_remaining_actions": int(estimated_remaining),
            "viewpoint_actions_remaining": (
                None if math.isinf(viewpoint_remaining) else int(viewpoint_remaining)
            ),
            "time_actions_remaining": (
                None if math.isinf(time_remaining_actions) else int(time_remaining_actions)
            ),
            "high_quality_coverage_fraction": coverage,
            "required_quality_coverage": self.required_quality_coverage,
        }
