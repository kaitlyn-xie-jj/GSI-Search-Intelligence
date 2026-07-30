"""Terminal result contract returned to GSI's platform and world model."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, Mapping, Optional, Tuple

from .search_observation import TargetDetection
from .search_state import SearchState


class SearchOutcomeStatus(str, Enum):
    FOUND = "found"
    NOT_FOUND = "not_found"
    BUDGET_EXHAUSTED = "budget_exhausted"
    ABORTED = "aborted"
    ERROR = "error"


@dataclass(frozen=True)
class SearchOutcome:
    """Terminal search result with task metrics and optional observed target."""

    task_id: str
    status: SearchOutcomeStatus
    reason: str
    detections: Tuple[TargetDetection, ...] = ()
    estimated_target_position: Optional[Tuple[float, float, float]] = None
    confidence: Optional[float] = None
    steps: int = 0
    elapsed_time_s: float = 0.0
    distance_travelled_m: float = 0.0
    energy_used: float = 0.0
    final_belief: Mapping[str, float] = field(default_factory=dict)
    metrics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.task_id.strip():
            raise ValueError("SearchOutcome.task_id must not be empty")
        if not self.reason.strip():
            raise ValueError("SearchOutcome.reason must not be empty")
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be within [0, 1]")
        if self.estimated_target_position is not None and len(self.estimated_target_position) != 3:
            raise ValueError("estimated_target_position must contain three values")
        if self.steps < 0 or min(self.elapsed_time_s, self.distance_travelled_m, self.energy_used) < 0:
            raise ValueError("steps and resource metrics must not be negative")
        object.__setattr__(self, "detections", tuple(self.detections))
        object.__setattr__(self, "final_belief", dict(self.final_belief))
        object.__setattr__(self, "metrics", dict(self.metrics))

    @property
    def found(self) -> bool:
        return self.status == SearchOutcomeStatus.FOUND

    @classmethod
    def from_state(
        cls,
        state: SearchState,
        *,
        status: SearchOutcomeStatus,
        reason: str,
        detections: Tuple[TargetDetection, ...] = (),
        metrics: Optional[Mapping[str, Any]] = None,
    ) -> "SearchOutcome":
        """Build a terminal outcome from the last policy state."""
        best = max(detections, key=lambda detection: detection.confidence, default=None)
        observed_cell_ids = set(state.observed_cell_quality)
        belief_cell_ids = set(state.belief)
        covered_belief_cells = observed_cell_ids & belief_cell_ids
        derived_metrics: Dict[str, Any] = {
            "observed_cell_count": len(observed_cell_ids),
            "belief_cell_count": len(belief_cell_ids),
            "coverage_fraction": (
                len(covered_belief_cells) / len(belief_cell_ids)
                if belief_cell_ids
                else None
            ),
        }
        for key in (
            "initial_belief_entropy_nats",
            "belief_entropy_nats",
            "belief_effective_cell_count",
            "belief_max_probability",
            "belief_most_likely_cell_id",
            "belief_update_count",
            "cumulative_entropy_reduction_nats",
            "cumulative_kl_divergence_nats",
            "last_evidence_type",
        ):
            if key in state.policy_metadata:
                derived_metrics[key] = state.policy_metadata[key]
        derived_metrics.update(metrics or {})
        return cls(
            task_id=state.task.task_id,
            status=status,
            reason=reason,
            detections=detections,
            estimated_target_position=best.estimated_position if best else None,
            confidence=best.confidence if best else None,
            steps=state.step_index,
            elapsed_time_s=state.elapsed_time_s,
            distance_travelled_m=state.distance_travelled_m,
            energy_used=state.energy_used,
            final_belief=state.belief,
            metrics=derived_metrics,
        )

    def to_platform_result(self) -> Dict[str, Any]:
        """Adapt to the result shape consumed by GSI's SkillExecutor."""
        # Repeated observations of the same entity are confirmation evidence,
        # not multiple discovered targets.
        target_ids = list(dict.fromkeys(
            detection.entity_id
            for detection in self.detections
            if detection.entity_id
        ))
        return {
            "success": self.found,
            "outcome": self.status.value,
            "reason": self.reason,
            "targets_found": target_ids,
            "estimated_target_position": self.estimated_target_position,
            "confidence": self.confidence,
            "final_belief": dict(self.final_belief),
            "search_metrics": {
                "steps": self.steps,
                "elapsed_time_s": self.elapsed_time_s,
                "distance_travelled_m": self.distance_travelled_m,
                "energy_used": self.energy_used,
                **dict(self.metrics),
            },
        }

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable representation."""
        data = asdict(self)
        data["status"] = self.status.value
        data["found"] = self.found
        return data
