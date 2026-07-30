"""Data contracts for reproducible search-policy benchmarks."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple

from ..belief import BeliefMap, BinarySensorModel
from ..contracts import SearchTask, Viewpoint
from ..search_space import SearchGrid


SUPPORTED_POLICIES = ("coverage", "random", "greedy_prior", "active")


@dataclass(frozen=True)
class SearchBenchmarkScenario:
    """One target placement and task-conditioned prior evaluated by every policy."""

    scenario_id: str
    task: SearchTask
    grid: SearchGrid
    target_cell_id: str
    initial_belief: Mapping[str, float]
    start_xy: Tuple[float, float]
    prior_condition: str = "unspecified"
    target_entity_id: str = "benchmark-target"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.scenario_id.strip():
            raise ValueError("scenario_id must not be empty")
        if len(self.start_xy) != 2:
            raise ValueError("start_xy must contain two coordinates")
        if not all(math.isfinite(float(value)) for value in self.start_xy):
            raise ValueError("start_xy coordinates must be finite")
        searchable_ids = {cell.cell_id for cell in self.grid.searchable_cells}
        if self.target_cell_id not in searchable_ids:
            raise ValueError("target_cell_id must identify a searchable grid cell")
        belief = BeliefMap.for_grid(self.grid, self.initial_belief)
        object.__setattr__(self, "initial_belief", dict(belief.probabilities))
        object.__setattr__(
            self,
            "start_xy",
            (float(self.start_xy[0]), float(self.start_xy[1])),
        )
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def target_cell(self):
        return next(
            cell for cell in self.grid.searchable_cells
            if cell.cell_id == self.target_cell_id
        )

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["task"] = self.task.to_dict()
        data["grid"] = self.grid.to_dict()
        return data


@dataclass(frozen=True)
class SearchBenchmarkConfig:
    """Shared policy, sensor, and resource settings for one benchmark suite."""

    policy_names: Tuple[str, ...] = SUPPORTED_POLICIES
    repetitions: int = 10
    base_seed: int = 0
    altitude_m: float = 30.0
    footprint_radius_m: float = 20.0
    speed_mps: float = 10.0
    observation_time_s: float = 1.0
    energy_per_m: float = 0.05
    observation_energy: float = 0.5
    observation_quality: float = 1.0
    detection_confidence: float = 1.0
    sensor_model: BinarySensorModel = field(
        default_factory=lambda: BinarySensorModel(
            detection_probability=0.85,
            false_positive_probability=0.01,
        )
    )
    coverage_pass_spacing_m: float = 20.0
    coverage_observation_spacing_m: Optional[float] = 20.0
    candidate_stride_cells: int = 1
    max_candidates: Optional[int] = None
    detection_weight: float = 1.0
    information_gain_weight: float = 1.0
    novelty_weight: float = 0.25
    travel_weight: float = 0.1
    distance_scale_m: float = 100.0
    verification_followup_limit: Optional[int] = None

    def __post_init__(self) -> None:
        names = tuple(str(name).strip().lower() for name in self.policy_names)
        if not names or len(set(names)) != len(names):
            raise ValueError("policy_names must be non-empty and unique")
        unsupported = set(names) - set(SUPPORTED_POLICIES)
        if unsupported:
            raise ValueError(f"unsupported benchmark policies: {sorted(unsupported)}")
        if self.repetitions <= 0:
            raise ValueError("repetitions must be positive")
        for name in (
            "altitude_m",
            "speed_mps",
            "coverage_pass_spacing_m",
            "distance_scale_m",
        ):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if (
            self.coverage_observation_spacing_m is not None
            and (
                not math.isfinite(self.coverage_observation_spacing_m)
                or self.coverage_observation_spacing_m <= 0
            )
        ):
            raise ValueError(
                "coverage_observation_spacing_m must be finite and positive"
            )
        for name in (
            "footprint_radius_m",
            "observation_time_s",
            "energy_per_m",
            "observation_energy",
        ):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        for name in (
            "observation_quality",
            "detection_confidence",
        ):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be within [0, 1]")
        for name in (
            "detection_weight",
            "information_gain_weight",
            "novelty_weight",
            "travel_weight",
        ):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.candidate_stride_cells <= 0:
            raise ValueError("candidate_stride_cells must be positive")
        if self.max_candidates is not None and self.max_candidates <= 0:
            raise ValueError("max_candidates must be positive")
        if (
            self.verification_followup_limit is not None
            and self.verification_followup_limit <= 0
        ):
            raise ValueError("verification_followup_limit must be positive")
        object.__setattr__(self, "policy_names", names)

    def start_viewpoint(self, scenario: SearchBenchmarkScenario) -> Viewpoint:
        return Viewpoint(
            x=scenario.start_xy[0],
            y=scenario.start_xy[1],
            z=self.altitude_m,
            yaw=0.0,
        )

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["sensor_model"] = asdict(self.sensor_model)
        return data


@dataclass(frozen=True)
class SearchEpisodeResult:
    """Ground-truth-aware result of one policy on one scenario repetition."""

    scenario_id: str
    prior_condition: str
    policy_name: str
    repetition: int
    seed: int
    terminal_status: str
    declared_found: bool
    target_found: bool
    false_positive: bool
    steps: int
    elapsed_time_s: float
    distance_travelled_m: float
    energy_used: float
    coverage_fraction: float
    spl: float
    shortest_detection_distance_m: float
    initial_entropy_nats: float
    final_entropy_nats: float
    entropy_reduction_nats: float
    policy_trace: Tuple[Mapping[str, Any], ...] = ()
    belief_entropy_trace: Tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if self.repetition < 0 or self.steps < 0:
            raise ValueError("repetition and steps must not be negative")
        for name in (
            "elapsed_time_s",
            "distance_travelled_m",
            "energy_used",
            "shortest_detection_distance_m",
            "initial_entropy_nats",
            "final_entropy_nats",
        ):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        for name in ("coverage_fraction", "spl"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be within [0, 1]")
        if not math.isfinite(self.entropy_reduction_nats):
            raise ValueError("entropy_reduction_nats must be finite")
        object.__setattr__(
            self,
            "policy_trace",
            tuple(dict(item) for item in self.policy_trace),
        )
        object.__setattr__(self, "belief_entropy_trace", tuple(self.belief_entropy_trace))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MetricEstimate:
    """Mean and normal-approximation 95% confidence interval."""

    mean: Optional[float]
    ci95_low: Optional[float]
    ci95_high: Optional[float]
    sample_count: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PolicyAggregate:
    """Policy-level aggregate used for benchmark comparison tables."""

    policy_name: str
    prior_condition: str
    episode_count: int
    success_rate: MetricEstimate
    declared_found_rate: MetricEstimate
    false_positive_rate: MetricEstimate
    spl: MetricEstimate
    steps: MetricEstimate
    elapsed_time_s: MetricEstimate
    distance_travelled_m: MetricEstimate
    energy_used: MetricEstimate
    coverage_fraction: MetricEstimate
    entropy_reduction_nats: MetricEstimate
    successful_elapsed_time_s: MetricEstimate
    successful_distance_m: MetricEstimate

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        for key, value in self.__dict__.items():
            if isinstance(value, MetricEstimate):
                data[key] = value.to_dict()
        return data


@dataclass(frozen=True)
class SearchBenchmarkReport:
    """Complete benchmark output with episode records and policy aggregates."""

    config: SearchBenchmarkConfig
    scenario_ids: Tuple[str, ...]
    episodes: Tuple[SearchEpisodeResult, ...]
    aggregates: Tuple[PolicyAggregate, ...]
    condition_aggregates: Tuple[PolicyAggregate, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "scenario_ids": list(self.scenario_ids),
            "episodes": [episode.to_dict() for episode in self.episodes],
            "aggregates": [aggregate.to_dict() for aggregate in self.aggregates],
            "condition_aggregates": [
                aggregate.to_dict() for aggregate in self.condition_aggregates
            ],
        }
