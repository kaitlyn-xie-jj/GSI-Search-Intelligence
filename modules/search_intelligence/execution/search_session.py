"""Platform-neutral closed loop joining all search data contracts."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Mapping, Optional, Tuple

from ..belief import BayesianBeliefUpdater, BeliefMap, BeliefUpdate

from ..contracts import (
    SearchObservation,
    SearchOutcome,
    SearchOutcomeStatus,
    SearchState,
    SearchTask,
    TargetDetection,
    Viewpoint,
)
from ..policies import SearchPolicy
from ..search_space import SearchGrid


class SearchSession:
    """Advance a SearchPolicy with observations until success or termination."""

    def __init__(
        self,
        task: SearchTask,
        policy: SearchPolicy,
        *,
        initial_belief: Optional[Mapping[str, float]] = None,
        current_viewpoint: Optional[Viewpoint] = None,
        initial_policy_metadata: Optional[Mapping[str, object]] = None,
        search_grid: Optional[SearchGrid] = None,
        belief_updater: Optional[BayesianBeliefUpdater] = None,
    ) -> None:
        if (search_grid is None) != (belief_updater is None):
            raise ValueError("search_grid and belief_updater must be provided together")
        self.policy = policy
        self.search_grid = search_grid
        self.belief_updater = belief_updater
        initial_belief_data = dict(initial_belief or {})
        metadata = dict(initial_policy_metadata or {})
        if search_grid is not None:
            belief_map = BeliefMap.for_grid(
                search_grid,
                initial_belief_data if initial_belief_data else None,
            )
            initial_belief_data = dict(belief_map.probabilities)
            metadata.update({
                "initial_belief_entropy_nats": belief_map.entropy_nats,
                "belief_entropy_nats": belief_map.entropy_nats,
                "belief_effective_cell_count": belief_map.effective_cell_count,
                "belief_max_probability": belief_map.maximum_probability,
                "belief_most_likely_cell_id": belief_map.most_likely_cell_id,
                "belief_update_count": 0,
                "cumulative_entropy_reduction_nats": 0.0,
                "cumulative_kl_divergence_nats": 0.0,
            })
        self.state = SearchState.initial(
            task,
            belief=initial_belief_data,
            current_viewpoint=current_viewpoint,
            policy_metadata=metadata,
        )
        self.outcome: Optional[SearchOutcome] = None
        self._pending_viewpoint: Optional[Viewpoint] = None
        self._pending_policy_metadata: Dict[str, object] = {}
        self._belief_updates: list[BeliefUpdate] = []
        self._policy_decisions: list[Mapping[str, object]] = []

    @property
    def completed(self) -> bool:
        return self.outcome is not None

    @property
    def belief_updates(self) -> Tuple[BeliefUpdate, ...]:
        return tuple(self._belief_updates)

    @property
    def policy_decisions(self) -> Tuple[Mapping[str, object], ...]:
        return tuple(dict(item) for item in self._policy_decisions)

    def remaining_plan(self) -> Tuple[Viewpoint, ...]:
        """Return policy viewpoints that have not yet produced observations."""
        return () if self.completed else self.policy.plan(self.state)

    def next_viewpoint(self) -> Optional[Viewpoint]:
        """Select the next action and finalize when no action remains."""
        if self.completed:
            return None
        if self._pending_viewpoint is not None:
            return self._pending_viewpoint

        self._pending_viewpoint = self.policy.select_next(self.state)
        if self._pending_viewpoint is not None:
            self._pending_policy_metadata = dict(self.policy.decision_metadata(
                self.state,
                self._pending_viewpoint,
            ))
            if self._pending_policy_metadata:
                self._policy_decisions.append(dict(self._pending_policy_metadata))
        if self._pending_viewpoint is None:
            budget = self.state.exhausted_budget
            status = (
                SearchOutcomeStatus.BUDGET_EXHAUSTED
                if budget is not None
                else SearchOutcomeStatus.NOT_FOUND
            )
            reason = f"{budget} budget exhausted" if budget else "coverage route exhausted"
            self.outcome = SearchOutcome.from_state(self.state, status=status, reason=reason)
        return self._pending_viewpoint

    def record_observation(
        self,
        observation: SearchObservation,
        *,
        belief: Optional[Mapping[str, float]] = None,
        policy_metadata: Optional[Mapping[str, object]] = None,
    ) -> SearchState:
        """Apply one observation, then evaluate success and resource limits."""
        if self.completed:
            raise RuntimeError("cannot record an observation after the search session completed")
        if self._pending_viewpoint is None:
            raise RuntimeError("next_viewpoint must be called before record_observation")
        action_viewpoint_key = (
            observation.action_viewpoint_key or observation.viewpoint.key
        )
        if action_viewpoint_key != self._pending_viewpoint.key:
            raise ValueError("observation viewpoint does not match the pending policy action")

        next_belief = belief
        next_metadata = dict(self._pending_policy_metadata)
        next_metadata.update(policy_metadata or {})
        if belief is None and self.belief_updater is not None:
            assert self.search_grid is not None
            update = self.belief_updater.update(
                BeliefMap.from_mapping(
                    self.state.belief,
                    update_index=len(self._belief_updates),
                ),
                observation,
                self.search_grid,
                min_detection_confidence=self.state.task.success_criteria.min_confidence,
                max_localization_error_m=(
                    self.state.task.success_criteria.max_localization_error_m
                ),
            )
            self._belief_updates.append(update)
            next_belief = update.posterior.probabilities
            next_metadata.update(update.to_policy_metadata())
            next_metadata.update({
                "cumulative_entropy_reduction_nats": sum(
                    item.entropy_reduction_nats for item in self._belief_updates
                ),
                "cumulative_kl_divergence_nats": sum(
                    item.kl_divergence_nats for item in self._belief_updates
                ),
            })
        elif belief is not None and self.search_grid is not None:
            next_belief = BeliefMap.for_grid(
                self.search_grid,
                belief,
                update_index=len(self._belief_updates),
            ).probabilities

        self.state = self.state.advance(
            observation,
            belief=next_belief,
            policy_metadata=next_metadata,
        )
        self._pending_viewpoint = None
        self._pending_policy_metadata = {}

        confirmed = self._confirmed_detections()
        if confirmed:
            self.outcome = SearchOutcome.from_state(
                self.state,
                status=SearchOutcomeStatus.FOUND,
                reason="search success criteria met",
                detections=confirmed,
            )
        elif self.state.exhausted_budget is not None:
            self.outcome = SearchOutcome.from_state(
                self.state,
                status=SearchOutcomeStatus.BUDGET_EXHAUSTED,
                reason=f"{self.state.exhausted_budget} budget exhausted",
            )
        return self.state

    def abort(self, reason: str) -> SearchOutcome:
        """Stop a session because execution was externally interrupted."""
        if self.completed:
            assert self.outcome is not None
            return self.outcome
        self.outcome = SearchOutcome.from_state(
            self.state,
            status=SearchOutcomeStatus.ABORTED,
            reason=reason,
        )
        self._pending_viewpoint = None
        self._pending_policy_metadata = {}
        return self.outcome

    def fail(self, reason: str) -> SearchOutcome:
        """Stop a session because the platform or policy failed."""
        if self.completed:
            assert self.outcome is not None
            return self.outcome
        self.outcome = SearchOutcome.from_state(
            self.state,
            status=SearchOutcomeStatus.ERROR,
            reason=reason,
        )
        self._pending_viewpoint = None
        self._pending_policy_metadata = {}
        return self.outcome

    def _confirmed_detections(self) -> Tuple[TargetDetection, ...]:
        criteria = self.state.task.success_criteria
        grouped: Dict[str, list[Tuple[float, TargetDetection]]] = defaultdict(list)
        for observation in self.state.observations:
            for detection in observation.matching_detections(criteria.min_confidence):
                if not self._localization_is_acceptable(detection):
                    continue
                key = detection.entity_id or detection.label.strip().lower()
                grouped[key].append((observation.timestamp_s, detection))

        for group in grouped.values():
            if len(group) < criteria.min_confirmations:
                continue
            timestamps = [timestamp for timestamp, _ in group]
            if max(timestamps) - min(timestamps) < criteria.min_persistence_s:
                continue
            return tuple(detection for _, detection in group)
        return ()

    def _localization_is_acceptable(self, detection: TargetDetection) -> bool:
        maximum = self.state.task.success_criteria.max_localization_error_m
        if maximum is None:
            return True
        error = detection.attributes.get("localization_error_m")
        return error is not None and float(error) <= maximum
