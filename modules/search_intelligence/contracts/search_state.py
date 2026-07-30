"""Immutable state passed to search policies at each decision step."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from math import isclose
from typing import Any, Dict, Mapping, Optional, Tuple

from .search_observation import SearchObservation, Viewpoint
from .search_task import SearchTask


@dataclass(frozen=True)
class SearchState:
    """Current belief, history, coverage, pose, and resource usage."""

    task: SearchTask
    belief: Mapping[str, float]
    current_viewpoint: Optional[Viewpoint] = None
    observations: Tuple[SearchObservation, ...] = ()
    visited_viewpoint_keys: Tuple[str, ...] = ()
    observed_cell_quality: Mapping[str, float] = field(default_factory=dict)
    elapsed_time_s: float = 0.0
    distance_travelled_m: float = 0.0
    energy_used: float = 0.0
    step_index: int = 0
    policy_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        belief = {str(key): float(value) for key, value in self.belief.items()}
        if any(value < 0.0 for value in belief.values()):
            raise ValueError("belief values must not be negative")
        if belief and not isclose(sum(belief.values()), 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError("belief values must sum to 1")
        if min(self.elapsed_time_s, self.distance_travelled_m, self.energy_used) < 0:
            raise ValueError("resource usage must not be negative")
        if self.step_index < 0:
            raise ValueError("step_index must not be negative")

        quality = {str(key): float(value) for key, value in self.observed_cell_quality.items()}
        if any(not 0.0 <= value <= 1.0 for value in quality.values()):
            raise ValueError("observed cell quality must be within [0, 1]")

        object.__setattr__(self, "belief", belief)
        object.__setattr__(self, "observations", tuple(self.observations))
        object.__setattr__(self, "visited_viewpoint_keys", tuple(self.visited_viewpoint_keys))
        object.__setattr__(self, "observed_cell_quality", quality)
        object.__setattr__(self, "policy_metadata", dict(self.policy_metadata))

    @classmethod
    def initial(
        cls,
        task: SearchTask,
        belief: Mapping[str, float],
        *,
        current_viewpoint: Optional[Viewpoint] = None,
        policy_metadata: Optional[Mapping[str, Any]] = None,
    ) -> "SearchState":
        """Create a state and normalize a non-empty initial belief."""
        normalized = {str(key): float(value) for key, value in belief.items()}
        total = sum(normalized.values())
        if normalized and total <= 0:
            raise ValueError("initial belief must contain positive probability mass")
        if normalized:
            normalized = {key: value / total for key, value in normalized.items()}
        return cls(
            task=task,
            belief=normalized,
            current_viewpoint=current_viewpoint,
            policy_metadata=dict(policy_metadata or {}),
        )

    def advance(
        self,
        observation: SearchObservation,
        *,
        belief: Optional[Mapping[str, float]] = None,
        policy_metadata: Optional[Mapping[str, Any]] = None,
    ) -> "SearchState":
        """Return the state after applying one observation and optional belief update."""
        coverage = dict(self.observed_cell_quality)
        for cell_id in observation.visible_cell_ids:
            coverage[cell_id] = max(coverage.get(cell_id, 0.0), observation.observation_quality)

        metadata = dict(self.policy_metadata)
        if policy_metadata:
            metadata.update(policy_metadata)

        return replace(
            self,
            belief=dict(self.belief if belief is None else belief),
            current_viewpoint=observation.viewpoint,
            observations=self.observations + (observation,),
            visited_viewpoint_keys=self.visited_viewpoint_keys + (
                observation.action_viewpoint_key or observation.viewpoint.key,
            ),
            observed_cell_quality=coverage,
            elapsed_time_s=self.elapsed_time_s + observation.travel_time_s,
            distance_travelled_m=self.distance_travelled_m + observation.travel_distance_m,
            energy_used=self.energy_used + observation.energy_used,
            step_index=self.step_index + 1,
            policy_metadata=metadata,
        )

    @property
    def exhausted_budget(self) -> Optional[str]:
        """Return the first exhausted budget name, or ``None``."""
        budget = self.task.budget
        if budget.time_limit_s is not None and self.elapsed_time_s >= budget.time_limit_s:
            return "time"
        if budget.distance_limit_m is not None and self.distance_travelled_m >= budget.distance_limit_m:
            return "distance"
        if budget.energy_limit is not None and self.energy_used >= budget.energy_limit:
            return "energy"
        if budget.max_viewpoints is not None and self.step_index >= budget.max_viewpoints:
            return "viewpoints"
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)
