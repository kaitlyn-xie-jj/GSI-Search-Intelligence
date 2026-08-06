"""Explainable mode supervisor over the frozen Search Skill policies."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

from ..contracts import SearchObservation, SearchState, Viewpoint
from .base import SearchPolicy


@dataclass(frozen=True)
class HybridModeDecision:
    """One auditable supervisor decision."""

    mode: str
    reason: str
    previous_mode: Optional[str]
    switched: bool
    mode_entered_step: int
    signals: Mapping[str, Any]


@dataclass(frozen=True)
class HybridSearchSupervisorPolicy(SearchPolicy):
    """Select a frozen policy for one or more complete viewpoint actions.

    Improved active search remains the default. Coverage provides a global
    fallback, random search escapes stagnation, and original active search is
    used briefly when repeated rejected negatives make the visibility model
    unreliable. The supervisor does not blend incomparable policy scores.
    """

    improved_policy: SearchPolicy
    coverage_policy: SearchPolicy
    random_policy: SearchPolicy
    visibility_fallback_policy: SearchPolicy
    minimum_mode_steps: int = 2
    signal_window: int = 3
    visibility_rejection_threshold: float = 0.5
    low_effective_quality_threshold: float = 0.35
    high_quality_cell_threshold: float = 0.5
    coverage_trigger_progress: float = 0.35
    coverage_target_fraction: float = 0.55
    entropy_stagnation_ratio: float = 0.8
    kl_stagnation_threshold: float = 0.005
    fallback_cooldown_steps: int = 3

    def __post_init__(self) -> None:
        if self.minimum_mode_steps <= 0:
            raise ValueError("minimum_mode_steps must be positive")
        if self.signal_window <= 0:
            raise ValueError("signal_window must be positive")
        for name in (
            "visibility_rejection_threshold",
            "low_effective_quality_threshold",
            "high_quality_cell_threshold",
            "coverage_trigger_progress",
            "coverage_target_fraction",
            "entropy_stagnation_ratio",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be within [0, 1]")
        if self.kl_stagnation_threshold < 0:
            raise ValueError("kl_stagnation_threshold must not be negative")
        if self.fallback_cooldown_steps < 0:
            raise ValueError("fallback_cooldown_steps must not be negative")

    def plan(self, state: SearchState) -> Tuple[Viewpoint, ...]:
        decision = self.mode_decision(state)
        policy = self._policy_for_mode(decision.mode)
        plan = policy.plan(state)
        if plan:
            return plan
        if decision.mode != "improved_active":
            return self.improved_policy.plan(state)
        return ()

    def decision_metadata(
        self,
        state: SearchState,
        viewpoint: Viewpoint,
    ) -> Mapping[str, Any]:
        decision = self.mode_decision(state)
        policy = self._policy_for_mode(decision.mode)
        delegated_plan = policy.plan(state)
        if not delegated_plan and decision.mode != "improved_active":
            policy = self.improved_policy
            decision = HybridModeDecision(
                mode="improved_active",
                reason=f"{decision.mode}_route_exhausted",
                previous_mode=decision.previous_mode,
                switched=decision.previous_mode != "improved_active",
                mode_entered_step=state.step_index,
                signals=decision.signals,
            )
        delegated = dict(policy.decision_metadata(state, viewpoint))
        metadata = {
            "policy_name": type(self).__name__,
            "step_index": state.step_index,
            "selected_viewpoint_key": viewpoint.key,
            "hybrid_mode": decision.mode,
            "hybrid_previous_mode": decision.previous_mode,
            "hybrid_mode_switched": decision.switched,
            "hybrid_switch_reason": decision.reason,
            "hybrid_mode_entered_step": decision.mode_entered_step,
            "hybrid_mode_age_steps": (
                state.step_index - decision.mode_entered_step
            ),
            "hybrid_signals": dict(decision.signals),
            "delegated_policy_metadata": delegated,
        }
        for key in (
            "hybrid_last_visibility_fallback_step",
            "hybrid_last_random_escape_step",
        ):
            if key in decision.signals:
                metadata[key] = decision.signals[key]
        return metadata

    def mode_decision(self, state: SearchState) -> HybridModeDecision:
        signals = self._signals(state)
        previous = _optional_string(state.policy_metadata.get("hybrid_mode"))
        entered_step = int(state.policy_metadata.get(
            "hybrid_mode_entered_step",
            state.step_index,
        ))
        age = max(0, state.step_index - entered_step)

        if signals["recent_positive_detection"]:
            return self._decision(
                "improved_active",
                "positive_detection_confirmation",
                previous,
                entered_step,
                state.step_index,
                signals,
            )
        if previous is not None and age < self.minimum_mode_steps:
            return HybridModeDecision(
                mode=previous,
                reason="minimum_mode_residence",
                previous_mode=previous,
                switched=False,
                mode_entered_step=entered_step,
                signals=signals,
            )

        visibility_unreliable = (
            signals["recent_observation_count"] >= 2
            and (
                signals["negative_rejection_rate"]
                >= self.visibility_rejection_threshold
                or signals["mean_effective_observation_quality"]
                < self.low_effective_quality_threshold
            )
        )
        if visibility_unreliable and self._cooldown_elapsed(
            state,
            "hybrid_last_visibility_fallback_step",
        ):
            signals = dict(signals)
            signals["hybrid_last_visibility_fallback_step"] = state.step_index
            return self._decision(
                "visibility_fallback",
                "visibility_model_unreliable",
                previous,
                entered_step,
                state.step_index,
                signals,
            )

        needs_global_coverage = (
            signals["budget_progress"] >= self.coverage_trigger_progress
            and signals["high_quality_coverage_fraction"]
            < self.coverage_target_fraction
            and signals["entropy_ratio"] >= self.entropy_stagnation_ratio
        )
        if needs_global_coverage:
            return self._decision(
                "coverage_fallback",
                "global_coverage_deficit",
                previous,
                entered_step,
                state.step_index,
                signals,
            )

        stagnating = (
            state.step_index >= self.signal_window
            and signals["recent_new_high_quality_cells"] == 0
            and signals["last_kl_divergence_nats"]
            <= self.kl_stagnation_threshold
        )
        if stagnating and self._cooldown_elapsed(
            state,
            "hybrid_last_random_escape_step",
        ):
            signals = dict(signals)
            signals["hybrid_last_random_escape_step"] = state.step_index
            return self._decision(
                "random_escape",
                "belief_and_coverage_stagnation",
                previous,
                entered_step,
                state.step_index,
                signals,
            )

        return self._decision(
            "improved_active",
            "default_improved_active",
            previous,
            entered_step,
            state.step_index,
            signals,
        )

    def _signals(self, state: SearchState) -> Dict[str, Any]:
        recent = state.observations[-self.signal_window:]
        effective_quality = tuple(
            observation.observation_quality * observation.visibility_probability
            for observation in recent
        )
        rejected = tuple(
            observation for observation in recent
            if observation.negative_update_rejection_reason is not None
            or observation.negative_update_strength <= 0.0
        )
        recent_cells = _high_quality_cells(
            recent,
            self.high_quality_cell_threshold,
        )
        older_cells = _high_quality_cells(
            state.observations[:-self.signal_window],
            self.high_quality_cell_threshold,
        )
        all_high_quality_cells = recent_cells | older_cells
        belief_cell_count = len(state.belief)
        initial_entropy = float(state.policy_metadata.get(
            "initial_belief_entropy_nats",
            0.0,
        ))
        current_entropy = float(state.policy_metadata.get(
            "belief_entropy_nats",
            initial_entropy,
        ))
        return {
            "budget_progress": _budget_progress(state),
            "belief_max_probability": max(state.belief.values(), default=0.0),
            "entropy_ratio": (
                current_entropy / initial_entropy if initial_entropy > 0 else 0.0
            ),
            "last_kl_divergence_nats": float(state.policy_metadata.get(
                "last_kl_divergence_nats",
                0.0,
            )),
            "recent_observation_count": len(recent),
            "mean_effective_observation_quality": (
                sum(effective_quality) / len(effective_quality)
                if effective_quality else 1.0
            ),
            "negative_rejection_rate": (
                len(rejected) / len(recent) if recent else 0.0
            ),
            "high_quality_coverage_fraction": (
                len(all_high_quality_cells & set(state.belief)) / belief_cell_count
                if belief_cell_count else 0.0
            ),
            "recent_new_high_quality_cells": len(recent_cells - older_cells),
            "recent_positive_detection": any(
                observation.detections for observation in recent[-1:]
            ),
        }

    def _decision(
        self,
        mode: str,
        reason: str,
        previous: Optional[str],
        previous_entered_step: int,
        step_index: int,
        signals: Mapping[str, Any],
    ) -> HybridModeDecision:
        switched = previous != mode
        return HybridModeDecision(
            mode=mode,
            reason=reason,
            previous_mode=previous,
            switched=switched,
            mode_entered_step=(step_index if switched else previous_entered_step),
            signals=signals,
        )

    def _cooldown_elapsed(self, state: SearchState, key: str) -> bool:
        previous = state.policy_metadata.get(key)
        return previous is None or (
            state.step_index - int(previous) >= self.fallback_cooldown_steps
        )

    def _policy_for_mode(self, mode: str) -> SearchPolicy:
        policies = {
            "improved_active": self.improved_policy,
            "coverage_fallback": self.coverage_policy,
            "random_escape": self.random_policy,
            "visibility_fallback": self.visibility_fallback_policy,
        }
        try:
            return policies[mode]
        except KeyError as error:
            raise ValueError(f"unsupported hybrid mode: {mode}") from error


def _high_quality_cells(
    observations: Tuple[SearchObservation, ...],
    threshold: float,
) -> set[str]:
    cells = set()
    for observation in observations:
        quality = observation.observation_quality * observation.visibility_probability
        if quality >= threshold:
            cells.update(observation.visible_cell_ids)
    return cells


def _budget_progress(state: SearchState) -> float:
    progress = []
    budget = state.task.budget
    if budget.time_limit_s:
        progress.append(state.elapsed_time_s / budget.time_limit_s)
    if budget.distance_limit_m:
        progress.append(state.distance_travelled_m / budget.distance_limit_m)
    if budget.energy_limit:
        progress.append(state.energy_used / budget.energy_limit)
    if budget.max_viewpoints:
        progress.append(state.step_index / budget.max_viewpoints)
    return min(1.0, max(progress, default=0.0))


def _optional_string(value: object) -> Optional[str]:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None
