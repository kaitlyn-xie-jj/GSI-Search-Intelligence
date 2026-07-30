"""Candidate-viewpoint baselines and belief-aware active search policy."""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass, field
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
    detection_weight: float = 1.0
    information_gain_weight: float = 1.0
    novelty_weight: float = 0.25
    travel_weight: float = 0.1
    distance_scale_m: float = 100.0
    minimum_utility: Optional[float] = None

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
        if self.distance_scale_m <= 0:
            raise ValueError("distance_scale_m must be positive")
        if self.minimum_utility is not None and not math.isfinite(self.minimum_utility):
            raise ValueError("minimum_utility must be finite")

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
        return _cap_viewpoint_budget(
            tuple(score.viewpoint for score in scores),
            state,
        )

    def decision_metadata(
        self,
        state: SearchState,
        viewpoint: Viewpoint,
    ) -> Mapping[str, Any]:
        metadata = dict(super().decision_metadata(state, viewpoint))
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
        detection_probability = (
            visible_mass * self.sensor_model.detection_probability
            + (1.0 - visible_mass)
            * self.sensor_model.false_positive_probability
        )
        information_gain = _binary_expected_information_gain(
            visible_mass,
            self.sensor_model.detection_probability,
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
