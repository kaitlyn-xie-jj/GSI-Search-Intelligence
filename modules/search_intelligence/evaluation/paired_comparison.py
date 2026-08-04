"""Paired policy comparisons over identical scenario and repetition keys."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from statistics import mean, stdev
from typing import Callable, Dict, Mapping, Sequence, Tuple

from .contracts import SearchBenchmarkScenario, SearchEpisodeResult


@dataclass(frozen=True)
class PairedMetricComparison:
    """Candidate-minus-baseline difference with an oriented improvement interval."""

    metric: str
    higher_is_better: bool
    baseline_mean: float
    candidate_mean: float
    mean_difference: float
    difference_ci95_low: float
    difference_ci95_high: float
    mean_improvement: float
    improvement_ci95_low: float
    improvement_ci95_high: float
    relative_improvement: float | None
    pair_count: int

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def compare_paired_policy_results(
    baseline_episodes: Sequence[SearchEpisodeResult],
    candidate_episodes: Sequence[SearchEpisodeResult],
    scenarios: Sequence[SearchBenchmarkScenario],
) -> Mapping[str, PairedMetricComparison]:
    """Compare policies using identical scenario/repetition pairs."""
    baseline = _by_episode_key(baseline_episodes)
    candidate = _by_episode_key(candidate_episodes)
    if not baseline or set(baseline) != set(candidate):
        raise ValueError("paired policy episodes must be non-empty and aligned")
    scenario_by_id = {item.scenario_id: item for item in scenarios}
    if any(key[0] not in scenario_by_id for key in baseline):
        raise ValueError("every paired episode must have a matching scenario")

    metrics: Tuple[Tuple[str, bool, Callable], ...] = (
        ("success_rate", True, lambda episode, _: float(episode.target_found)),
        ("false_positive_rate", False, lambda episode, _: float(episode.false_positive)),
        ("spl", True, lambda episode, _: episode.spl),
        ("elapsed_time_s", False, lambda episode, _: episode.elapsed_time_s),
        ("distance_travelled_m", False, lambda episode, _: episode.distance_travelled_m),
        ("energy_used", False, lambda episode, _: episode.energy_used),
        ("steps", False, lambda episode, _: float(episode.steps)),
        (
            "viewpoint_budget_fraction",
            False,
            lambda episode, scenario: episode.steps / max(
                1,
                scenario.task.budget.max_viewpoints or 1,
            ),
        ),
    )
    comparisons = {}
    for name, higher_is_better, metric in metrics:
        baseline_values = tuple(
            metric(baseline[key], scenario_by_id[key[0]])
            for key in sorted(baseline)
        )
        candidate_values = tuple(
            metric(candidate[key], scenario_by_id[key[0]])
            for key in sorted(candidate)
        )
        differences = tuple(
            candidate_value - baseline_value
            for baseline_value, candidate_value in zip(
                baseline_values,
                candidate_values,
            )
        )
        center, low, high = _difference_interval(differences)
        direction = 1.0 if higher_is_better else -1.0
        improvement = direction * center
        improvement_low = direction * (low if higher_is_better else high)
        improvement_high = direction * (high if higher_is_better else low)
        baseline_mean = mean(baseline_values)
        comparisons[name] = PairedMetricComparison(
            metric=name,
            higher_is_better=higher_is_better,
            baseline_mean=baseline_mean,
            candidate_mean=mean(candidate_values),
            mean_difference=center,
            difference_ci95_low=low,
            difference_ci95_high=high,
            mean_improvement=improvement,
            improvement_ci95_low=improvement_low,
            improvement_ci95_high=improvement_high,
            relative_improvement=(
                improvement / abs(baseline_mean)
                if baseline_mean != 0.0 else None
            ),
            pair_count=len(differences),
        )
    return comparisons


def _by_episode_key(
    episodes: Sequence[SearchEpisodeResult],
) -> Mapping[Tuple[str, int], SearchEpisodeResult]:
    items = {
        (episode.scenario_id, episode.repetition): episode
        for episode in episodes
    }
    if len(items) != len(episodes):
        raise ValueError("policy episodes contain duplicate scenario/repetition keys")
    return items


def _difference_interval(values: Sequence[float]) -> Tuple[float, float, float]:
    center = mean(values)
    half_width = (
        1.96 * stdev(values) / math.sqrt(len(values))
        if len(values) > 1 else 0.0
    )
    return center, center - half_width, center + half_width
