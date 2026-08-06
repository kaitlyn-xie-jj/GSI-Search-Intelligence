"""Bayesian target-location belief and observation update models."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple

from .contracts import SearchObservation
from .search_space import SearchGrid


@dataclass(frozen=True)
class BeliefMap:
    """Categorical probability that one target occupies each searchable cell."""

    probabilities: Mapping[str, float]
    update_index: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        probabilities = {
            str(cell_id): float(probability)
            for cell_id, probability in self.probabilities.items()
        }
        if any(not math.isfinite(value) or value < 0 for value in probabilities.values()):
            raise ValueError("belief probabilities must be finite and non-negative")
        total = sum(probabilities.values())
        if probabilities and not math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError("belief probabilities must sum to 1")
        if self.update_index < 0:
            raise ValueError("BeliefMap.update_index must not be negative")
        object.__setattr__(self, "probabilities", probabilities)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @classmethod
    def from_mapping(
        cls,
        probabilities: Mapping[str, float],
        *,
        update_index: int = 0,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> "BeliefMap":
        """Normalize arbitrary non-negative mass into a BeliefMap."""
        normalized = {
            str(cell_id): float(probability)
            for cell_id, probability in probabilities.items()
        }
        if any(not math.isfinite(value) or value < 0 for value in normalized.values()):
            raise ValueError("belief mass must be finite and non-negative")
        total = sum(normalized.values())
        if normalized and total <= 0:
            raise ValueError("belief mass must contain a positive value")
        if normalized:
            normalized = {
                cell_id: probability / total
                for cell_id, probability in normalized.items()
            }
        return cls(
            probabilities=normalized,
            update_index=update_index,
            metadata=dict(metadata or {}),
        )

    @classmethod
    def for_grid(
        cls,
        grid: SearchGrid,
        probabilities: Optional[Mapping[str, float]] = None,
        *,
        update_index: int = 0,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> "BeliefMap":
        """Build a full grid-aligned distribution, filling omitted cells with zero."""
        searchable_ids = tuple(cell.cell_id for cell in grid.searchable_cells)
        searchable_id_set = set(searchable_ids)
        supplied = dict(probabilities or {})
        unknown = set(supplied) - searchable_id_set
        if unknown:
            raise ValueError(f"belief contains cells outside the searchable grid: {sorted(unknown)}")
        if not searchable_ids:
            return cls({}, update_index=update_index, metadata=dict(metadata or {}))
        if probabilities is None:
            mass = {cell_id: 1.0 for cell_id in searchable_ids}
        else:
            mass = {cell_id: float(supplied.get(cell_id, 0.0)) for cell_id in searchable_ids}
        return cls.from_mapping(
            mass,
            update_index=update_index,
            metadata=metadata,
        )

    @property
    def entropy_nats(self) -> float:
        return -sum(
            probability * math.log(probability)
            for probability in self.probabilities.values()
            if probability > 0
        )

    @property
    def effective_cell_count(self) -> float:
        return math.exp(self.entropy_nats)

    @property
    def most_likely_cell_id(self) -> Optional[str]:
        return max(
            self.probabilities,
            key=self.probabilities.get,
            default=None,
        )

    @property
    def maximum_probability(self) -> float:
        return max(self.probabilities.values(), default=0.0)

    def probability_in(self, cell_ids: Tuple[str, ...]) -> float:
        return sum(self.probabilities.get(cell_id, 0.0) for cell_id in cell_ids)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data.update({
            "entropy_nats": self.entropy_nats,
            "effective_cell_count": self.effective_cell_count,
            "most_likely_cell_id": self.most_likely_cell_id,
            "maximum_probability": self.maximum_probability,
        })
        return data


@dataclass(frozen=True)
class BinarySensorModel:
    """Target-conditioned detector reliability for one viewpoint observation."""

    detection_probability: float = 0.85
    false_positive_probability: float = 0.05
    minimum_likelihood: float = 1e-9

    def __post_init__(self) -> None:
        if not 0 < self.detection_probability <= 1:
            raise ValueError("detection_probability must be within (0, 1]")
        if not 0 <= self.false_positive_probability < 1:
            raise ValueError("false_positive_probability must be within [0, 1)")
        if self.detection_probability <= self.false_positive_probability:
            raise ValueError("detection_probability must exceed false_positive_probability")
        if not 0 < self.minimum_likelihood < 1:
            raise ValueError("minimum_likelihood must be within (0, 1)")

    def effective_detection_probability(self, quality: float) -> float:
        """Interpolate to an uninformative sensor as observation quality approaches zero."""
        quality = max(0.0, min(1.0, float(quality)))
        return (
            self.false_positive_probability
            + quality * (
                self.detection_probability - self.false_positive_probability
            )
        )


@dataclass(frozen=True)
class BeliefUpdate:
    """Posterior plus diagnostics for one Bayesian observation update."""

    posterior: BeliefMap
    evidence_type: str
    visible_cell_ids: Tuple[str, ...]
    evidence_cell_ids: Tuple[str, ...]
    likelihoods: Mapping[str, float]
    prior_entropy_nats: float
    posterior_entropy_nats: float
    entropy_reduction_nats: float
    kl_divergence_nats: float
    effective_detection_probability: float
    visibility_probability: float = 1.0
    negative_update_strength: float = 1.0
    rejection_reason: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "visible_cell_ids", tuple(self.visible_cell_ids))
        object.__setattr__(self, "evidence_cell_ids", tuple(self.evidence_cell_ids))
        object.__setattr__(self, "likelihoods", dict(self.likelihoods))

    def to_policy_metadata(self) -> Dict[str, Any]:
        return {
            "belief_update_count": self.posterior.update_index,
            "belief_entropy_nats": self.posterior_entropy_nats,
            "belief_effective_cell_count": self.posterior.effective_cell_count,
            "belief_max_probability": self.posterior.maximum_probability,
            "belief_most_likely_cell_id": self.posterior.most_likely_cell_id,
            "last_evidence_type": self.evidence_type,
            "last_entropy_reduction_nats": self.entropy_reduction_nats,
            "last_kl_divergence_nats": self.kl_divergence_nats,
            "last_visibility_probability": self.visibility_probability,
            "last_effective_detection_probability": (
                self.effective_detection_probability
            ),
            "last_negative_update_strength": self.negative_update_strength,
            "last_update_rejection_reason": self.rejection_reason,
        }

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BayesianBeliefUpdater:
    """Apply positive or negative target-conditioned evidence to a BeliefMap."""

    sensor_model: BinarySensorModel = field(default_factory=BinarySensorModel)
    confidence_gating_enabled: bool = True

    def update(
        self,
        prior: BeliefMap,
        observation: SearchObservation,
        grid: SearchGrid,
        *,
        min_detection_confidence: float = 0.5,
        max_localization_error_m: Optional[float] = None,
    ) -> BeliefUpdate:
        if not 0 <= min_detection_confidence <= 1:
            raise ValueError("min_detection_confidence must be within [0, 1]")
        if max_localization_error_m is not None and max_localization_error_m < 0:
            raise ValueError("max_localization_error_m must be non-negative")
        if not prior.probabilities:
            applied_visibility = (
                observation.visibility_probability
                if self.confidence_gating_enabled else 1.0
            )
            applied_negative_strength = (
                observation.negative_update_strength
                if self.confidence_gating_enabled else 1.0
            )
            return BeliefUpdate(
                posterior=BeliefMap({}, update_index=prior.update_index + 1),
                evidence_type="empty_search_space",
                visible_cell_ids=(),
                evidence_cell_ids=(),
                likelihoods={},
                prior_entropy_nats=0.0,
                posterior_entropy_nats=0.0,
                entropy_reduction_nats=0.0,
                kl_divergence_nats=0.0,
                effective_detection_probability=self.sensor_model.effective_detection_probability(
                    observation.observation_quality
                ),
                visibility_probability=applied_visibility,
                negative_update_strength=applied_negative_strength,
                rejection_reason=(
                    observation.negative_update_rejection_reason
                    if self.confidence_gating_enabled else None
                ),
            )

        grid_aligned = BeliefMap.for_grid(
            grid,
            prior.probabilities,
            update_index=prior.update_index,
            metadata=prior.metadata,
        )
        supported_ids = set(grid_aligned.probabilities)
        visible_ids = tuple(dict.fromkeys(
            cell_id
            for cell_id in observation.visible_cell_ids
            if cell_id in supported_ids
        ))
        detections = tuple(
            detection
            for detection in observation.matching_detections(
                min_detection_confidence
            )
            if _localization_is_acceptable(
                detection,
                max_localization_error_m,
            )
        )
        localized_ids = self._localized_detection_cells(detections, grid, supported_ids)

        if detections:
            evidence_type = "positive_localized" if localized_ids else "positive_unlocalized"
            evidence_ids = localized_ids or visible_ids
            confidence = max(detection.confidence for detection in detections)
            # A positive detection is direct evidence that the target was visible.
            quality = observation.observation_quality * confidence
            effective_detection = self.sensor_model.effective_detection_probability(quality)
            likelihoods = {
                cell_id: (
                    effective_detection
                    if cell_id in evidence_ids
                    else self.sensor_model.false_positive_probability
                )
                for cell_id in grid_aligned.probabilities
            }
        else:
            rejection_reason = (
                observation.negative_update_rejection_reason
                if self.confidence_gating_enabled else None
            )
            negative_strength = (
                observation.negative_update_strength
                if self.confidence_gating_enabled else 1.0
            )
            visibility_probability = (
                observation.visibility_probability
                if self.confidence_gating_enabled else 1.0
            )
            evidence_type = "negative" if negative_strength > 0 else "negative_rejected"
            evidence_ids = visible_ids
            effective_detection = self.sensor_model.effective_detection_probability(
                observation.observation_quality
                * visibility_probability
                * negative_strength
            )
            likelihoods = {
                cell_id: (
                    1.0 - effective_detection
                    if cell_id in visible_ids
                    else 1.0 - self.sensor_model.false_positive_probability
                )
                for cell_id in grid_aligned.probabilities
            }

        likelihoods = {
            cell_id: max(self.sensor_model.minimum_likelihood, likelihood)
            for cell_id, likelihood in likelihoods.items()
        }
        posterior_mass = {
            cell_id: probability * likelihoods[cell_id]
            for cell_id, probability in grid_aligned.probabilities.items()
        }
        posterior = BeliefMap.from_mapping(
            posterior_mass,
            update_index=prior.update_index + 1,
            metadata={
                **dict(prior.metadata),
                "last_evidence_type": evidence_type,
                "last_update_rejection_reason": (
                    rejection_reason if not detections else None
                ),
            },
        )
        prior_entropy = grid_aligned.entropy_nats
        posterior_entropy = posterior.entropy_nats
        return BeliefUpdate(
            posterior=posterior,
            evidence_type=evidence_type,
            visible_cell_ids=visible_ids,
            evidence_cell_ids=evidence_ids,
            likelihoods=likelihoods,
            prior_entropy_nats=prior_entropy,
            posterior_entropy_nats=posterior_entropy,
            entropy_reduction_nats=prior_entropy - posterior_entropy,
            kl_divergence_nats=self._kl_divergence(posterior, grid_aligned),
            effective_detection_probability=effective_detection,
            visibility_probability=(
                observation.visibility_probability
                if detections or self.confidence_gating_enabled else 1.0
            ),
            negative_update_strength=(
                (
                    observation.negative_update_strength
                    if self.confidence_gating_enabled else 1.0
                )
                if not detections else 0.0
            ),
            rejection_reason=(
                observation.negative_update_rejection_reason
                if not detections and self.confidence_gating_enabled else None
            ),
        )

    @staticmethod
    def _localized_detection_cells(
        detections: Tuple[Any, ...],
        grid: SearchGrid,
        supported_ids: set[str],
    ) -> Tuple[str, ...]:
        localized = []
        for detection in detections:
            if detection.estimated_position is None:
                continue
            cell = grid.cell_at(
                detection.estimated_position[0],
                detection.estimated_position[1],
            )
            if cell is not None and cell.cell_id in supported_ids and cell.searchable:
                localized.append(cell.cell_id)
        return tuple(dict.fromkeys(localized))

    @staticmethod
    def _kl_divergence(posterior: BeliefMap, prior: BeliefMap) -> float:
        divergence = 0.0
        for cell_id, posterior_probability in posterior.probabilities.items():
            prior_probability = prior.probabilities.get(cell_id, 0.0)
            if posterior_probability > 0 and prior_probability > 0:
                divergence += posterior_probability * math.log(
                    posterior_probability / prior_probability
                )
        return divergence


def _localization_is_acceptable(
    detection: Any,
    maximum_error_m: Optional[float],
) -> bool:
    if maximum_error_m is None:
        return True
    error = detection.attributes.get("localization_error_m")
    return error is not None and float(error) <= maximum_error_m
