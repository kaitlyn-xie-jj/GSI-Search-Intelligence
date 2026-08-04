"""Candidate-viewpoint baselines and belief-aware active search policy."""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from ..belief import BinarySensorModel
from ..contracts import SearchState, Viewpoint
from ..search_space import ViewpointCandidate
from .base import SearchPolicy


@dataclass(frozen=True)
class ViewpointScore:
    """Auditable utility decomposition for one active-search candidate."""

    candidate_id: str
    viewpoint: Viewpoint
    belief_mass_visible: float
    unobserved_belief_mass: float
    detection_probability: float
    information_gain_nats: float
    travel_distance_m: float
    normalized_travel_cost: float
    utility: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RandomPolicy(SearchPolicy):
    """Deterministic seeded random ordering for a candidate-viewpoint baseline."""

    candidates: Tuple[ViewpointCandidate, ...]
    seed: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidates", _validated_candidates(self.candidates))

    def plan(self, state: SearchState) -> Tuple[Viewpoint, ...]:
        remaining = _remaining_candidates(self.candidates, state)
        ordered = sorted(
            remaining,
            key=lambda candidate: hashlib.sha256(
                f"{self.seed}:{candidate.candidate_id}".encode("utf-8")
            ).digest(),
        )
        return _cap_viewpoint_budget(
            tuple(candidate.viewpoint for candidate in ordered),
            state,
        )


@dataclass(frozen=True)
class GreedyPriorPolicy(SearchPolicy):
    """Select the candidate covering the most fixed initial-prior mass."""

    candidates: Tuple[ViewpointCandidate, ...]
    prior: Mapping[str, float]
    distance_weight: float = 0.0
    distance_scale_m: float = 100.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidates", _validated_candidates(self.candidates))
        prior = {str(cell_id): float(value) for cell_id, value in self.prior.items()}
        if any(value < 0 or not math.isfinite(value) for value in prior.values()):
            raise ValueError("GreedyPriorPolicy prior must be finite and non-negative")
        if self.distance_weight < 0 or self.distance_scale_m <= 0:
            raise ValueError("distance weight must be non-negative and scale positive")
        object.__setattr__(self, "prior", prior)

    def plan(self, state: SearchState) -> Tuple[Viewpoint, ...]:
        scored = []
        for candidate in _remaining_candidates(self.candidates, state):
            mass = sum(self.prior.get(cell_id, 0.0) for cell_id in candidate.visible_cell_ids)
            distance = _travel_distance(state.current_viewpoint, candidate.viewpoint)
            utility = mass - self.distance_weight * distance / self.distance_scale_m
            scored.append((utility, mass, candidate))
        scored.sort(key=lambda item: (-item[0], -item[1], item[2].candidate_id))
        return _cap_viewpoint_budget(
            tuple(item[2].viewpoint for item in scored),
            state,
        )


@dataclass(frozen=True)
class ActiveSearchPolicy(SearchPolicy):
    """Rank viewpoints by detection, mutual information, novelty, and travel cost."""

    candidates: Tuple[ViewpointCandidate, ...]
    sensor_model: BinarySensorModel = field(default_factory=BinarySensorModel)
    observation_quality: float = 1.0
    detection_weight: float = 1.0
    information_gain_weight: float = 1.0
    novelty_weight: float = 0.25
    travel_weight: float = 0.1
    distance_scale_m: float = 100.0
    minimum_utility: Optional[float] = None
    verification_followup_limit: Optional[int] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidates", _validated_candidates(self.candidates))
        weights = (
            self.detection_weight,
            self.information_gain_weight,
            self.novelty_weight,
            self.travel_weight,
        )
        if any(not math.isfinite(weight) or weight < 0 for weight in weights):
            raise ValueError("active-search utility weights must be finite and non-negative")
        if not 0.0 <= self.observation_quality <= 1.0:
            raise ValueError("observation_quality must be within [0, 1]")
        if self.distance_scale_m <= 0:
            raise ValueError("distance_scale_m must be positive")
        if self.minimum_utility is not None and not math.isfinite(self.minimum_utility):
            raise ValueError("minimum_utility must be finite")
        if (
            self.verification_followup_limit is not None
            and self.verification_followup_limit <= 0
        ):
            raise ValueError("verification_followup_limit must be positive")

    def score_candidates(self, state: SearchState) -> Tuple[ViewpointScore, ...]:
        scores = tuple(
            self._score(candidate, state)
            for candidate in _remaining_candidates(self.candidates, state)
            if candidate.visible_cell_ids
        )
        return tuple(sorted(
            scores,
            key=lambda score: (-score.utility, score.candidate_id),
        ))

    def plan(self, state: SearchState) -> Tuple[Viewpoint, ...]:
        scores = self.score_candidates(state)
        if self.minimum_utility is not None:
            scores = tuple(
                score for score in scores if score.utility >= self.minimum_utility
            )
        verification_cell_id = self._pending_verification_cell_id(state)
        if verification_cell_id is not None:
            verification_candidates = sorted(
                (
                    candidate
                    for candidate in _remaining_candidates(self.candidates, state)
                    if verification_cell_id in candidate.visible_cell_ids
                ),
                key=lambda candidate: (
                    _travel_distance(state.current_viewpoint, candidate.viewpoint),
                    candidate.candidate_id,
                ),
            )
            verification_keys = {
                candidate.viewpoint.key for candidate in verification_candidates
            }
            viewpoints = tuple(
                candidate.viewpoint for candidate in verification_candidates
            ) + tuple(
                score.viewpoint
                for score in scores
                if score.viewpoint.key not in verification_keys
            )
        else:
            viewpoints = tuple(score.viewpoint for score in scores)
        return _cap_viewpoint_budget(
            viewpoints,
            state,
        )

    def decision_metadata(
        self,
        state: SearchState,
        viewpoint: Viewpoint,
    ) -> Mapping[str, Any]:
        metadata = dict(super().decision_metadata(state, viewpoint))
        verification_cell_id = self._pending_verification_cell_id(state)
        verification_mode = verification_cell_id is not None and any(
            candidate.viewpoint.key == viewpoint.key
            and verification_cell_id in candidate.visible_cell_ids
            for candidate in _remaining_candidates(self.candidates, state)
        )
        metadata.update({
            "verification_mode": verification_mode,
            "verification_cell_id": verification_cell_id,
        })
        score = next(
            (
                candidate_score
                for candidate_score in self.score_candidates(state)
                if candidate_score.viewpoint.key == viewpoint.key
            ),
            None,
        )
        if score is None:
            return metadata
        metadata.update({
            "selected_candidate_id": score.candidate_id,
            "selected_viewpoint_score": score.to_dict(),
        })
        return metadata

    def _pending_verification_cell_id(
        self,
        state: SearchState,
    ) -> Optional[str]:
        criteria = state.task.success_criteria
        if criteria.min_confirmations <= 1 and criteria.min_persistence_s <= 0.0:
            return None

        grouped: Dict[str, list[Tuple[float, int, int, Optional[str]]]] = {}
        order = 0
        for observation_index, observation in enumerate(state.observations):
            for detection in observation.matching_detections(criteria.min_confidence):
                key = detection.entity_id or detection.label.strip().lower()
                localized_cell_id = detection.attributes.get("localized_cell_id")
                grouped.setdefault(key, []).append((
                    observation.timestamp_s,
                    observation_index,
                    order,
                    str(localized_cell_id) if localized_cell_id is not None else None,
                ))
                order += 1

        pending = []
        for detections in grouped.values():
            timestamps = [item[0] for item in detections]
            confirmed = (
                len(detections) >= criteria.min_confirmations
                and max(timestamps) - min(timestamps) >= criteria.min_persistence_s
            )
            localized = [item for item in detections if item[3] is not None]
            if confirmed or not localized:
                continue
            latest = max(localized, key=lambda item: item[2])
            followup_count = sum(
                latest[3] in observation.visible_cell_ids
                for observation in state.observations[latest[1] + 1:]
            )
            if (
                self.verification_followup_limit is None
                or followup_count < self.verification_followup_limit
            ):
                pending.append(latest)
        if not pending:
            return None
        return max(pending, key=lambda item: item[2])[3]

    def _score(
        self,
        candidate: ViewpointCandidate,
        state: SearchState,
    ) -> ViewpointScore:
        visible_ids = set(candidate.visible_cell_ids)
        visible_mass = sum(
            probability
            for cell_id, probability in state.belief.items()
            if cell_id in visible_ids
        )
        unobserved_mass = sum(
            probability
            * (1.0 - state.observed_cell_quality.get(cell_id, 0.0))
            for cell_id, probability in state.belief.items()
            if cell_id in visible_ids
        )
        effective_detection_probability = (
            self.sensor_model.effective_detection_probability(
                self.observation_quality
            )
        )
        detection_probability = (
            visible_mass * effective_detection_probability
            + (1.0 - visible_mass)
            * self.sensor_model.false_positive_probability
        )
        information_gain = _binary_expected_information_gain(
            visible_mass,
            effective_detection_probability,
            self.sensor_model.false_positive_probability,
        )
        distance = _travel_distance(state.current_viewpoint, candidate.viewpoint)
        normalized_cost = distance / self.distance_scale_m
        utility = (
            self.detection_weight * detection_probability
            + self.information_gain_weight * information_gain
            + self.novelty_weight * unobserved_mass
            - self.travel_weight * normalized_cost
        )
        return ViewpointScore(
            candidate_id=candidate.candidate_id,
            viewpoint=candidate.viewpoint,
            belief_mass_visible=visible_mass,
            unobserved_belief_mass=unobserved_mass,
            detection_probability=detection_probability,
            information_gain_nats=information_gain,
            travel_distance_m=distance,
            normalized_travel_cost=normalized_cost,
            utility=utility,
        )


@dataclass(frozen=True)
class AdaptiveWeightState:
    """Normalized search context and effective utility weights for one decision."""

    entropy_ratio: float
    normalized_entropy: float
    budget_progress: float
    coverage_fraction: float
    belief_max_probability: float
    prior_confidence: float
    base_weights: Mapping[str, float]
    multipliers: Mapping[str, float]
    adaptive_weights: Mapping[str, float]
    adaptation_components: Mapping[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LookaheadBranchValue:
    """Probability and best continuation value for one binary observation branch."""

    observation: str
    probability: float
    posterior_entropy_nats: float
    best_candidate_id: Optional[str]
    continuation_utility: float


@dataclass(frozen=True)
class LookaheadViewpointScore:
    """Horizon-2 value decomposition for one first-stage candidate."""

    candidate_id: str
    viewpoint: Viewpoint
    immediate_score: ViewpointScore
    branches: Tuple[LookaheadBranchValue, ...]
    expected_continuation_utility: float
    discount_factor: float
    utility: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AdaptiveActiveSearchPolicy(ActiveSearchPolicy):
    """Adapt active-search utility weights from belief and resource state."""

    def adaptive_weight_state(self, state: SearchState) -> AdaptiveWeightState:
        current_entropy = _belief_entropy(state.belief)
        initial_entropy = _finite_metadata_value(
            state.policy_metadata,
            "initial_belief_entropy_nats",
            current_entropy,
        )
        entropy_ratio = _clamp01(
            current_entropy / initial_entropy if initial_entropy > 0.0 else 0.0
        )
        cell_count = len(state.belief)
        maximum_entropy = math.log(cell_count) if cell_count > 1 else 0.0
        normalized_entropy = _clamp01(
            current_entropy / maximum_entropy if maximum_entropy > 0.0 else 0.0
        )
        coverage_fraction = _clamp01(
            sum(state.observed_cell_quality.get(cell_id, 0.0) for cell_id in state.belief)
            / cell_count
            if cell_count else 0.0
        )
        belief_max_probability = max(state.belief.values(), default=0.0)
        prior_confidence = _clamp01(_finite_metadata_value(
            state.policy_metadata,
            "prior_confidence",
            0.5,
        ))
        budget_progress = _budget_progress(state)

        uncertainty = normalized_entropy
        low_prior_confidence = 1.0 - prior_confidence
        unexplored_fraction = 1.0 - coverage_fraction
        concentration = 1.0 - normalized_entropy
        urgency = budget_progress * budget_progress
        multipliers = {
            "detection": 0.8 + concentration + 0.6 * budget_progress,
            "information_gain": (
                0.7
                + 1.2 * uncertainty * (1.0 - 0.5 * budget_progress)
                + 0.5 * low_prior_confidence
            ),
            "novelty": (
                0.6
                + 1.5 * uncertainty * unexplored_fraction
                + 0.5 * low_prior_confidence
            ),
            "travel": 0.7 + 1.8 * urgency + 0.5 * coverage_fraction,
        }
        base_weights = {
            "detection": self.detection_weight,
            "information_gain": self.information_gain_weight,
            "novelty": self.novelty_weight,
            "travel": self.travel_weight,
        }
        weighted = {
            name: base_weights[name] * multipliers[name]
            for name in base_weights
        }
        total = sum(weighted.values())
        adaptive_weights = (
            {name: value / total for name, value in weighted.items()}
            if total > 0.0
            else dict(weighted)
        )
        return AdaptiveWeightState(
            entropy_ratio=entropy_ratio,
            normalized_entropy=normalized_entropy,
            budget_progress=budget_progress,
            coverage_fraction=coverage_fraction,
            belief_max_probability=belief_max_probability,
            prior_confidence=prior_confidence,
            base_weights=base_weights,
            multipliers=multipliers,
            adaptive_weights=adaptive_weights,
            adaptation_components={
                "uncertainty": uncertainty,
                "concentration": concentration,
                "low_prior_confidence": low_prior_confidence,
                "unexplored_fraction": unexplored_fraction,
                "urgency": urgency,
            },
        )

    def score_candidates(self, state: SearchState) -> Tuple[ViewpointScore, ...]:
        weights = self.adaptive_weight_state(state).adaptive_weights
        scores = tuple(
            self._score_with_weights(candidate, state, weights)
            for candidate in _remaining_candidates(self.candidates, state)
            if candidate.visible_cell_ids
        )
        return tuple(sorted(
            scores,
            key=lambda score: (-score.utility, score.candidate_id),
        ))

    def decision_metadata(
        self,
        state: SearchState,
        viewpoint: Viewpoint,
    ) -> Mapping[str, Any]:
        metadata = dict(super().decision_metadata(state, viewpoint))
        metadata["adaptive_weight_state"] = self.adaptive_weight_state(state).to_dict()
        return metadata

    def _score_with_weights(
        self,
        candidate: ViewpointCandidate,
        state: SearchState,
        weights: Mapping[str, float],
    ) -> ViewpointScore:
        score = super()._score(candidate, state)
        utility = (
            weights["detection"] * score.detection_probability
            + weights["information_gain"] * score.information_gain_nats
            + weights["novelty"] * score.unobserved_belief_mass
            - weights["travel"] * score.normalized_travel_cost
        )
        return ViewpointScore(
            candidate_id=score.candidate_id,
            viewpoint=score.viewpoint,
            belief_mass_visible=score.belief_mass_visible,
            unobserved_belief_mass=score.unobserved_belief_mass,
            detection_probability=score.detection_probability,
            information_gain_nats=score.information_gain_nats,
            travel_distance_m=score.travel_distance_m,
            normalized_travel_cost=score.normalized_travel_cost,
            utility=utility,
        )


@dataclass(frozen=True)
class BeliefLookaheadPolicy(ActiveSearchPolicy):
    """Horizon-2 belief-space planner with binary observation branching."""

    discount_factor: float = 1.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if not 0.0 <= self.discount_factor <= 1.0:
            raise ValueError("discount_factor must be within [0, 1]")

    def score_candidates(self, state: SearchState) -> Tuple[LookaheadViewpointScore, ...]:
        scores = tuple(
            self._lookahead_score(candidate, state)
            for candidate in _remaining_candidates(self.candidates, state)
            if candidate.visible_cell_ids
        )
        return tuple(sorted(
            scores,
            key=lambda score: (-score.utility, score.candidate_id),
        ))

    def _lookahead_score(
        self,
        candidate: ViewpointCandidate,
        state: SearchState,
    ) -> LookaheadViewpointScore:
        immediate = self._score(candidate, state)
        branch_probabilities = {
            "positive": immediate.detection_probability,
            "negative": 1.0 - immediate.detection_probability,
        }
        branches = []
        for observation, probability in branch_probabilities.items():
            posterior = _binary_observation_posterior(
                state.belief,
                candidate.visible_cell_ids,
                self.sensor_model.effective_detection_probability(
                    self.observation_quality
                ),
                self.sensor_model.false_positive_probability,
                positive=observation == "positive",
                minimum_likelihood=self.sensor_model.minimum_likelihood,
            )
            future_state = _hypothetical_state_after_candidate(
                state,
                candidate,
                posterior,
                self.observation_quality,
                immediate.travel_distance_m,
            )
            continuation_scores = ActiveSearchPolicy.score_candidates(self, future_state)
            best = continuation_scores[0] if continuation_scores else None
            branches.append(LookaheadBranchValue(
                observation=observation,
                probability=probability,
                posterior_entropy_nats=_belief_entropy(posterior),
                best_candidate_id=best.candidate_id if best is not None else None,
                continuation_utility=best.utility if best is not None else 0.0,
            ))
        expected_continuation = sum(
            branch.probability * branch.continuation_utility
            for branch in branches
        )
        return LookaheadViewpointScore(
            candidate_id=candidate.candidate_id,
            viewpoint=candidate.viewpoint,
            immediate_score=immediate,
            branches=tuple(branches),
            expected_continuation_utility=expected_continuation,
            discount_factor=self.discount_factor,
            utility=(
                immediate.utility
                + self.discount_factor * expected_continuation
            ),
        )


def _validated_candidates(
    candidates: Sequence[ViewpointCandidate],
) -> Tuple[ViewpointCandidate, ...]:
    normalized = tuple(candidates)
    if len({candidate.candidate_id for candidate in normalized}) != len(normalized):
        raise ValueError("candidate IDs must be unique")
    if len({candidate.viewpoint.key for candidate in normalized}) != len(normalized):
        raise ValueError("candidate viewpoints must be unique")
    return normalized


def _remaining_candidates(
    candidates: Sequence[ViewpointCandidate],
    state: SearchState,
) -> Tuple[ViewpointCandidate, ...]:
    visited = set(state.visited_viewpoint_keys)
    return tuple(
        candidate
        for candidate in candidates
        if candidate.viewpoint.key not in visited
    )


def _cap_viewpoint_budget(
    viewpoints: Tuple[Viewpoint, ...],
    state: SearchState,
) -> Tuple[Viewpoint, ...]:
    limit = state.task.budget.max_viewpoints
    if limit is None:
        return viewpoints
    return viewpoints[:max(0, limit - state.step_index)]


def _travel_distance(
    current: Optional[Viewpoint],
    candidate: Viewpoint,
) -> float:
    if current is None:
        return 0.0
    return math.sqrt(
        (current.x - candidate.x) ** 2
        + (current.y - candidate.y) ** 2
        + (current.z - candidate.z) ** 2
    )


def _binary_expected_information_gain(
    visible_belief_mass: float,
    detection_probability: float,
    false_positive_probability: float,
) -> float:
    visible_belief_mass = max(0.0, min(1.0, visible_belief_mass))
    observation_probability = (
        visible_belief_mass * detection_probability
        + (1.0 - visible_belief_mass) * false_positive_probability
    )
    return max(0.0, (
        _binary_entropy(observation_probability)
        - visible_belief_mass * _binary_entropy(detection_probability)
        - (1.0 - visible_belief_mass)
        * _binary_entropy(false_positive_probability)
    ))


def _binary_entropy(probability: float) -> float:
    if probability <= 0.0 or probability >= 1.0:
        return 0.0
    return -(
        probability * math.log(probability)
        + (1.0 - probability) * math.log(1.0 - probability)
    )


def _belief_entropy(belief: Mapping[str, float]) -> float:
    return -sum(
        probability * math.log(probability)
        for probability in belief.values()
        if probability > 0.0
    )


def _binary_observation_posterior(
    belief: Mapping[str, float],
    visible_cell_ids: Sequence[str],
    detection_probability: float,
    false_positive_probability: float,
    *,
    positive: bool,
    minimum_likelihood: float,
) -> Dict[str, float]:
    visible = set(visible_cell_ids)
    mass = {}
    for cell_id, probability in belief.items():
        event_probability = (
            detection_probability
            if cell_id in visible
            else false_positive_probability
        )
        likelihood = max(
            minimum_likelihood,
            event_probability if positive else 1.0 - event_probability,
        )
        mass[cell_id] = probability * likelihood
    total = sum(mass.values())
    if total <= 0.0:
        return dict(belief)
    return {cell_id: value / total for cell_id, value in mass.items()}


def _hypothetical_state_after_candidate(
    state: SearchState,
    candidate: ViewpointCandidate,
    posterior: Mapping[str, float],
    observation_quality: float,
    travel_distance_m: float,
) -> SearchState:
    coverage = dict(state.observed_cell_quality)
    for cell_id in candidate.visible_cell_ids:
        coverage[cell_id] = max(coverage.get(cell_id, 0.0), observation_quality)
    return replace(
        state,
        belief=dict(posterior),
        current_viewpoint=candidate.viewpoint,
        visited_viewpoint_keys=(
            state.visited_viewpoint_keys + (candidate.viewpoint.key,)
        ),
        observed_cell_quality=coverage,
        distance_travelled_m=state.distance_travelled_m + travel_distance_m,
        step_index=state.step_index + 1,
    )


def _finite_metadata_value(
    metadata: Mapping[str, Any],
    key: str,
    default: float,
) -> float:
    try:
        value = float(metadata.get(key, default))
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _budget_progress(state: SearchState) -> float:
    budget = state.task.budget
    fractions = []
    if budget.max_viewpoints is not None:
        fractions.append(state.step_index / budget.max_viewpoints)
    if budget.time_limit_s is not None:
        fractions.append(state.elapsed_time_s / budget.time_limit_s)
    if budget.distance_limit_m is not None:
        fractions.append(state.distance_travelled_m / budget.distance_limit_m)
    if budget.energy_limit is not None:
        fractions.append(state.energy_used / budget.energy_limit)
    return _clamp01(max(fractions, default=0.0))


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))
