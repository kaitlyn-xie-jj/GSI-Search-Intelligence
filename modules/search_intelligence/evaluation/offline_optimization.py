"""Leakage-resistant offline calibration for active-search utility weights."""

from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from ..belief import BinarySensorModel
from .contracts import (
    SearchBenchmarkConfig,
    SearchBenchmarkScenario,
    SearchEpisodeResult,
)
from .runner import SearchBenchmarkRunner
from .stress import stress_benchmark_scenarios


@dataclass(frozen=True)
class UtilityWeights:
    """Non-negative weights for the auditable active-search utility."""

    detection: float
    information_gain: float
    novelty: float
    travel: float

    def __post_init__(self) -> None:
        values = tuple(self.to_dict().values())
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("utility weights must be finite and non-negative")
        if sum(values) <= 0:
            raise ValueError("at least one utility weight must be positive")

    @property
    def total(self) -> float:
        return sum(self.to_dict().values())

    def normalized(self) -> "UtilityWeights":
        total = self.total
        return UtilityWeights(*(value / total for value in self.as_tuple()))

    def as_tuple(self) -> Tuple[float, float, float, float]:
        return (
            self.detection,
            self.information_gain,
            self.novelty,
            self.travel,
        )

    def to_dict(self) -> Dict[str, float]:
        return {
            "detection": self.detection,
            "information_gain": self.information_gain,
            "novelty": self.novelty,
            "travel": self.travel,
        }


DEFAULT_UTILITY_WEIGHTS = UtilityWeights(1.0, 1.0, 0.25, 0.1).normalized()


@dataclass(frozen=True)
class OfflineOptimizationConfig:
    """Search space, split, sensor, and objective settings for calibration."""

    candidate_count: int = 64
    validation_candidate_count: int = 8
    repetitions: int = 5
    base_seed: int = 0
    train_layouts: Tuple[str, ...] = ("compact_rectangle",)
    validation_layouts: Tuple[str, ...] = ("large_rectangle",)
    test_layouts: Tuple[str, ...] = ("l_shape",)
    sensor_model: BinarySensorModel = field(
        default_factory=lambda: BinarySensorModel(0.85, 0.01)
    )
    observation_quality: float = 1.0
    persistent_distractor_probability: float = 0.0
    false_alarm_correlation: float = 0.0
    correlated_false_alarm_shared_identity: bool = False
    localization_error_std_m: float = 0.0
    success_weight: float = 1.0
    spl_weight: float = 0.30
    budget_weight: float = 0.15
    false_positive_weight: float = 0.50
    minimum_validation_improvement: float = 0.01

    def __post_init__(self) -> None:
        if self.candidate_count <= 0:
            raise ValueError("candidate_count must be positive")
        if not 0 < self.validation_candidate_count <= self.candidate_count:
            raise ValueError(
                "validation_candidate_count must be within candidate_count"
            )
        if self.repetitions <= 0:
            raise ValueError("repetitions must be positive")
        split_layouts = (
            tuple(self.train_layouts),
            tuple(self.validation_layouts),
            tuple(self.test_layouts),
        )
        if any(not layouts for layouts in split_layouts):
            raise ValueError("train, validation, and test layouts must be non-empty")
        flattened = tuple(layout for layouts in split_layouts for layout in layouts)
        if len(set(flattened)) != len(flattened):
            raise ValueError("layout splits must be disjoint")
        for name in (
            "observation_quality",
            "persistent_distractor_probability",
            "false_alarm_correlation",
        ):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be within [0, 1]")
        if self.localization_error_std_m < 0:
            raise ValueError("localization_error_std_m must be non-negative")
        for name in (
            "success_weight",
            "spl_weight",
            "budget_weight",
            "false_positive_weight",
            "minimum_validation_improvement",
        ):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and non-negative")

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["sensor_model"] = asdict(self.sensor_model)
        return data


@dataclass(frozen=True)
class OptimizationScore:
    """Scalar objective and its auditable component metrics."""

    split: str
    objective: float
    episode_count: int
    success_rate: float
    false_positive_rate: float
    mean_spl: float
    mean_budget_fraction: float
    mean_steps: float
    mean_elapsed_time_s: float
    mean_distance_m: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PairedValidationComparison:
    """Paired candidate-minus-default validation objective difference."""

    mean_difference: float
    ci95_low: float
    ci95_high: float
    pair_count: int
    confidently_better: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WeightCandidateResult:
    """Train score and optional validation score for one weight candidate."""

    candidate_id: str
    weights: UtilityWeights
    train: OptimizationScore
    validation: Optional[OptimizationScore] = None
    validation_vs_default: Optional[PairedValidationComparison] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "weights": self.weights.to_dict(),
            "train": self.train.to_dict(),
            "validation": (
                self.validation.to_dict() if self.validation is not None else None
            ),
            "validation_vs_default": (
                self.validation_vs_default.to_dict()
                if self.validation_vs_default is not None else None
            ),
        }


@dataclass(frozen=True)
class OfflineOptimizationResult:
    """Complete result with isolated test evaluation after model selection."""

    config: OfflineOptimizationConfig
    split_scenario_ids: Mapping[str, Tuple[str, ...]]
    candidates: Tuple[WeightCandidateResult, ...]
    selected_candidate_id: str
    selected_weights: UtilityWeights
    selected_scores: Mapping[str, OptimizationScore]
    default_scores: Mapping[str, OptimizationScore]
    selected_episodes: Mapping[str, Tuple[SearchEpisodeResult, ...]]
    default_episodes: Mapping[str, Tuple[SearchEpisodeResult, ...]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": "gsi-offline-utility-optimization-v1",
            "config": self.config.to_dict(),
            "split_scenario_ids": {
                key: list(value) for key, value in self.split_scenario_ids.items()
            },
            "candidates": [item.to_dict() for item in self.candidates],
            "selected_candidate_id": self.selected_candidate_id,
            "selected_weights": self.selected_weights.to_dict(),
            "selected_scores": {
                key: value.to_dict() for key, value in self.selected_scores.items()
            },
            "default_weights": DEFAULT_UTILITY_WEIGHTS.to_dict(),
            "default_scores": {
                key: value.to_dict() for key, value in self.default_scores.items()
            },
            "test_isolation": (
                "The test split is evaluated only after selection by validation score."
            ),
            "selection_rule": (
                "Replace the default only when the paired validation objective "
                "difference has a strictly positive 95 percent lower bound and "
                "meets the configured minimum practical improvement."
            ),
        }


def default_offline_splits(
    config: OfflineOptimizationConfig,
) -> Mapping[str, Tuple[SearchBenchmarkScenario, ...]]:
    """Build disjoint layout-held-out splits from the stress scenario suite."""
    scenarios = stress_benchmark_scenarios()
    requested = {
        "train": set(config.train_layouts),
        "validation": set(config.validation_layouts),
        "test": set(config.test_layouts),
    }
    available = {str(item.metadata.get("layout")) for item in scenarios}
    unknown = set().union(*requested.values()) - available
    if unknown:
        raise ValueError(f"unknown optimization layouts: {sorted(unknown)}")
    splits = {
        split: tuple(
            item for item in scenarios
            if item.metadata.get("layout") in layouts
        )
        for split, layouts in requested.items()
    }
    scenario_ids = [
        item.scenario_id for items in splits.values() for item in items
    ]
    if len(set(scenario_ids)) != len(scenario_ids):
        raise ValueError("optimization scenario splits must be disjoint")
    return splits


def generate_weight_candidates(
    count: int,
    seed: int,
) -> Tuple[UtilityWeights, ...]:
    """Generate deterministic simplex candidates including the current default."""
    if count <= 0:
        raise ValueError("candidate count must be positive")
    anchors = (
        DEFAULT_UTILITY_WEIGHTS,
        UtilityWeights(1.0, 0.0, 0.0, 0.0),
        UtilityWeights(0.0, 1.0, 0.0, 0.0),
        UtilityWeights(0.0, 0.0, 1.0, 0.0),
        UtilityWeights(0.0, 0.0, 0.0, 1.0),
        UtilityWeights(1.0, 1.0, 1.0, 1.0).normalized(),
    )
    candidates = []
    seen = set()

    def append(weights: UtilityWeights) -> None:
        normalized = weights.normalized()
        key = tuple(round(value, 12) for value in normalized.as_tuple())
        if key not in seen and len(candidates) < count:
            seen.add(key)
            candidates.append(normalized)

    for anchor in anchors:
        append(anchor)
    generator = random.Random(seed)
    while len(candidates) < count:
        draws = tuple(-math.log(max(generator.random(), 1e-12)) for _ in range(4))
        append(UtilityWeights(*draws))
    return tuple(candidates)


class OfflineUtilityOptimizer:
    """Calibrate weights on train/validation and touch test only after selection."""

    def __init__(self, config: OfflineOptimizationConfig) -> None:
        self.config = config

    def run(
        self,
        splits: Optional[
            Mapping[str, Sequence[SearchBenchmarkScenario]]
        ] = None,
    ) -> OfflineOptimizationResult:
        split_items = self._validated_splits(
            splits if splits is not None else default_offline_splits(self.config)
        )
        weights = generate_weight_candidates(
            self.config.candidate_count,
            self.config.base_seed,
        )
        train_results = tuple(
            WeightCandidateResult(
                candidate_id=f"weights-{index:04d}",
                weights=item,
                train=self._evaluate(item, split_items["train"], "train"),
            )
            for index, item in enumerate(weights)
        )
        ranked_train = sorted(train_results, key=self._train_rank, reverse=True)
        # The pre-registered production default is always a finalist. This
        # no-regression guard prevents train-only ranking from excluding the
        # incumbent before validation can compare against it.
        validation_ids = {"weights-0000"}
        for item in ranked_train:
            if len(validation_ids) >= self.config.validation_candidate_count:
                break
            validation_ids.add(item.candidate_id)
        default_validation, default_validation_episodes = self._evaluate_with_episodes(
            DEFAULT_UTILITY_WEIGHTS,
            split_items["validation"],
            "validation",
        )
        candidate_items = []
        for item in train_results:
            validation = None
            comparison = None
            if item.candidate_id in validation_ids:
                if item.candidate_id == "weights-0000":
                    validation = default_validation
                    validation_episodes = default_validation_episodes
                else:
                    validation, validation_episodes = self._evaluate_with_episodes(
                        item.weights,
                        split_items["validation"],
                        "validation",
                    )
                comparison = self._paired_validation_comparison(
                    validation_episodes,
                    default_validation_episodes,
                    split_items["validation"],
                )
            candidate_items.append(WeightCandidateResult(
                candidate_id=item.candidate_id,
                weights=item.weights,
                train=item.train,
                validation=validation,
                validation_vs_default=comparison,
            ))
        candidates = tuple(candidate_items)
        finalists = tuple(item for item in candidates if item.validation is not None)
        challengers = tuple(
            item for item in finalists
            if item.validation_vs_default is not None
            and item.validation_vs_default.confidently_better
        )
        selected = (
            max(challengers, key=self._validation_rank)
            if challengers else candidates[0]
        )

        selected_pairs = {
            split: self._evaluate_with_episodes(
                selected.weights,
                scenarios,
                split,
            )
            for split, scenarios in split_items.items()
        }
        default_pairs = {
            split: self._evaluate_with_episodes(
                DEFAULT_UTILITY_WEIGHTS,
                scenarios,
                split,
            )
            for split, scenarios in split_items.items()
        }
        selected_scores = {split: pair[0] for split, pair in selected_pairs.items()}
        selected_episodes = {split: pair[1] for split, pair in selected_pairs.items()}
        default_scores = {split: pair[0] for split, pair in default_pairs.items()}
        default_episodes = {split: pair[1] for split, pair in default_pairs.items()}
        return OfflineOptimizationResult(
            config=self.config,
            split_scenario_ids={
                split: tuple(item.scenario_id for item in scenarios)
                for split, scenarios in split_items.items()
            },
            candidates=candidates,
            selected_candidate_id=selected.candidate_id,
            selected_weights=selected.weights,
            selected_scores=selected_scores,
            default_scores=default_scores,
            selected_episodes=selected_episodes,
            default_episodes=default_episodes,
        )

    def _evaluate(
        self,
        weights: UtilityWeights,
        scenarios: Sequence[SearchBenchmarkScenario],
        split: str,
    ) -> OptimizationScore:
        score, _ = self._evaluate_with_episodes(weights, scenarios, split)
        return score

    def _evaluate_with_episodes(
        self,
        weights: UtilityWeights,
        scenarios: Sequence[SearchBenchmarkScenario],
        split: str,
    ) -> Tuple[OptimizationScore, Tuple[SearchEpisodeResult, ...]]:
        config = SearchBenchmarkConfig(
            policy_names=("active",),
            repetitions=self.config.repetitions,
            base_seed=self.config.base_seed,
            sensor_model=self.config.sensor_model,
            observation_quality=self.config.observation_quality,
            detection_weight=weights.detection,
            information_gain_weight=weights.information_gain,
            novelty_weight=weights.novelty,
            travel_weight=weights.travel,
            distance_scale_mode="map_diagonal",
            persistent_distractor_probability=(
                self.config.persistent_distractor_probability
            ),
            false_alarm_correlation=self.config.false_alarm_correlation,
            correlated_false_alarm_shared_identity=(
                self.config.correlated_false_alarm_shared_identity
            ),
            localization_error_std_m=self.config.localization_error_std_m,
        )
        episodes = SearchBenchmarkRunner(config).run(scenarios).episodes
        return self._score(split, episodes, scenarios), episodes

    def _score(
        self,
        split: str,
        episodes: Sequence[SearchEpisodeResult],
        scenarios: Sequence[SearchBenchmarkScenario],
    ) -> OptimizationScore:
        if not episodes:
            raise ValueError("optimization scoring requires at least one episode")
        scenario_by_id = {item.scenario_id: item for item in scenarios}
        budget_fractions = tuple(
            episode.steps
            / max(1, scenario_by_id[episode.scenario_id].task.budget.max_viewpoints or 1)
            for episode in episodes
        )
        success_rate = mean(float(item.target_found) for item in episodes)
        false_positive_rate = mean(float(item.false_positive) for item in episodes)
        mean_spl = mean(item.spl for item in episodes)
        mean_budget_fraction = mean(budget_fractions)
        objective = mean(
            self._episode_objective(episode, scenario_by_id[episode.scenario_id])
            for episode in episodes
        )
        return OptimizationScore(
            split=split,
            objective=objective,
            episode_count=len(episodes),
            success_rate=success_rate,
            false_positive_rate=false_positive_rate,
            mean_spl=mean_spl,
            mean_budget_fraction=mean_budget_fraction,
            mean_steps=mean(item.steps for item in episodes),
            mean_elapsed_time_s=mean(item.elapsed_time_s for item in episodes),
            mean_distance_m=mean(item.distance_travelled_m for item in episodes),
        )

    def _paired_validation_comparison(
        self,
        candidate_episodes: Sequence[SearchEpisodeResult],
        default_episodes: Sequence[SearchEpisodeResult],
        scenarios: Sequence[SearchBenchmarkScenario],
    ) -> PairedValidationComparison:
        scenario_by_id = {item.scenario_id: item for item in scenarios}
        default_by_key = {
            (item.scenario_id, item.repetition): item
            for item in default_episodes
        }
        candidate_by_key = {
            (item.scenario_id, item.repetition): item
            for item in candidate_episodes
        }
        if set(candidate_by_key) != set(default_by_key):
            raise ValueError("paired validation episodes do not align")
        differences = tuple(
            self._episode_objective(candidate_by_key[key], scenario_by_id[key[0]])
            - self._episode_objective(default_by_key[key], scenario_by_id[key[0]])
            for key in sorted(default_by_key)
        )
        center = mean(differences)
        half_width = (
            1.96 * stdev(differences) / math.sqrt(len(differences))
            if len(differences) > 1 else 0.0
        )
        low = center - half_width
        high = center + half_width
        return PairedValidationComparison(
            mean_difference=center,
            ci95_low=low,
            ci95_high=high,
            pair_count=len(differences),
            confidently_better=(
                low > 0.0
                and center >= self.config.minimum_validation_improvement
            ),
        )

    def _episode_objective(
        self,
        episode: SearchEpisodeResult,
        scenario: SearchBenchmarkScenario,
    ) -> float:
        budget_fraction = episode.steps / max(
            1,
            scenario.task.budget.max_viewpoints or 1,
        )
        return (
            self.config.success_weight * float(episode.target_found)
            + self.config.spl_weight * episode.spl
            - self.config.budget_weight * budget_fraction
            - self.config.false_positive_weight * float(episode.false_positive)
        )

    @staticmethod
    def _train_rank(item: WeightCandidateResult) -> Tuple[float, ...]:
        score = item.train
        return (
            score.objective,
            score.success_rate,
            score.mean_spl,
            -score.false_positive_rate,
            -score.mean_budget_fraction,
        )

    @staticmethod
    def _validation_rank(item: WeightCandidateResult) -> Tuple[float, ...]:
        assert item.validation is not None
        score = item.validation
        return (
            score.objective,
            score.success_rate,
            score.mean_spl,
            -score.false_positive_rate,
            -score.mean_budget_fraction,
            item.train.objective,
        )

    @staticmethod
    def _validated_splits(
        splits: Mapping[str, Sequence[SearchBenchmarkScenario]],
    ) -> Mapping[str, Tuple[SearchBenchmarkScenario, ...]]:
        if set(splits) != {"train", "validation", "test"}:
            raise ValueError("splits must contain exactly train, validation, and test")
        normalized = {key: tuple(value) for key, value in splits.items()}
        if any(not value for value in normalized.values()):
            raise ValueError("optimization splits must not be empty")
        ids = [item.scenario_id for values in normalized.values() for item in values]
        if len(set(ids)) != len(ids):
            raise ValueError("optimization split scenario IDs must be disjoint")
        return normalized


def write_offline_optimization_result(
    result: OfflineOptimizationResult,
    output_directory: str,
) -> Mapping[str, str]:
    """Persist the full audit trail and a compact deployable policy artifact."""
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "offline_optimization_report.json"
    candidate_path = output / "offline_optimization_candidates.csv"
    split_path = output / "offline_optimization_split_summary.csv"
    episode_path = output / "offline_optimization_selected_episodes.csv"
    policy_path = output / "selected_active_search_policy.json"

    report_path.write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_csv(candidate_path, tuple(_candidate_row(item) for item in result.candidates))
    _write_csv(split_path, tuple(
        _split_row(method, split, weights, score)
        for method, weights, scores in (
            ("default", DEFAULT_UTILITY_WEIGHTS, result.default_scores),
            ("selected", result.selected_weights, result.selected_scores),
        )
        for split, score in scores.items()
    ))
    _write_csv(episode_path, tuple(
        _optimization_episode_row(method, split, episode)
        for method, episodes_by_split in (
            ("default", result.default_episodes),
            ("selected", result.selected_episodes),
        )
        for split, episodes in episodes_by_split.items()
        for episode in episodes
    ))
    policy_path.write_text(json.dumps({
        "schema_version": "gsi-active-search-policy-weights-v1",
        "selection": "validation",
        "candidate_id": result.selected_candidate_id,
        "weights": result.selected_weights.to_dict(),
        "distance_scale_mode": "map_diagonal",
        "sensor_model": asdict(result.config.sensor_model),
        "ros_parameters": {
            "active_detection_weight": result.selected_weights.detection,
            "active_information_gain_weight": (
                result.selected_weights.information_gain
            ),
            "active_novelty_weight": result.selected_weights.novelty,
            "active_travel_weight": result.selected_weights.travel,
            "active_distance_scale_mode": "map_diagonal",
        },
    }, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "report_json": str(report_path),
        "candidates_csv": str(candidate_path),
        "split_summary_csv": str(split_path),
        "selected_episodes_csv": str(episode_path),
        "selected_policy_json": str(policy_path),
    }


def _candidate_row(item: WeightCandidateResult) -> Dict[str, Any]:
    row: Dict[str, Any] = {"candidate_id": item.candidate_id, **item.weights.to_dict()}
    for split, score in (("train", item.train), ("validation", item.validation)):
        if score is not None:
            row.update({f"{split}_{key}": value for key, value in score.to_dict().items()})
    if item.validation_vs_default is not None:
        row.update({
            f"validation_vs_default_{key}": value
            for key, value in item.validation_vs_default.to_dict().items()
        })
    return row


def _split_row(
    method: str,
    split: str,
    weights: UtilityWeights,
    score: OptimizationScore,
) -> Dict[str, Any]:
    return {
        "method": method,
        "split": split,
        **weights.to_dict(),
        **score.to_dict(),
    }


def _optimization_episode_row(
    method: str,
    split: str,
    episode: SearchEpisodeResult,
) -> Dict[str, Any]:
    row = episode.to_dict()
    row.pop("policy_trace", None)
    row.pop("belief_entropy_trace", None)
    row.pop("sensor_trace", None)
    return {"method": method, "split": split, **row}


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = tuple(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
