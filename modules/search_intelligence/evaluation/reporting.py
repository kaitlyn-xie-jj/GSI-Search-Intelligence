"""Aggregation and artifact writers for search-policy benchmarks."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, Iterable, Mapping, Sequence, Tuple

from .contracts import (
    MetricEstimate,
    PolicyAggregate,
    SearchBenchmarkReport,
    SearchEpisodeResult,
)


def estimate(values: Iterable[float], *, bounded: bool = False) -> MetricEstimate:
    """Return a mean and normal-approximation 95% confidence interval."""
    items = tuple(float(value) for value in values)
    if not items:
        return MetricEstimate(None, None, None, 0)
    center = mean(items)
    half_width = (
        1.96 * stdev(items) / math.sqrt(len(items))
        if len(items) > 1 else 0.0
    )
    low = center - half_width
    high = center + half_width
    if bounded:
        low = max(0.0, low)
        high = min(1.0, high)
    return MetricEstimate(center, low, high, len(items))


def aggregate_policy_results(
    episodes: Sequence[SearchEpisodeResult],
    policy_names: Sequence[str],
    *,
    prior_condition: str = "all",
) -> Tuple[PolicyAggregate, ...]:
    aggregates = []
    for policy_name in policy_names:
        items = tuple(
            episode for episode in episodes
            if episode.policy_name == policy_name
            and (
                prior_condition == "all"
                or episode.prior_condition == prior_condition
            )
        )
        successful = tuple(episode for episode in items if episode.target_found)
        aggregates.append(PolicyAggregate(
            policy_name=policy_name,
            prior_condition=prior_condition,
            episode_count=len(items),
            success_rate=estimate(
                (float(item.target_found) for item in items), bounded=True
            ),
            declared_found_rate=estimate(
                (float(item.declared_found) for item in items), bounded=True
            ),
            false_positive_rate=estimate(
                (float(item.false_positive) for item in items), bounded=True
            ),
            spl=estimate((item.spl for item in items), bounded=True),
            steps=estimate(item.steps for item in items),
            elapsed_time_s=estimate(item.elapsed_time_s for item in items),
            distance_travelled_m=estimate(
                item.distance_travelled_m for item in items
            ),
            energy_used=estimate(item.energy_used for item in items),
            coverage_fraction=estimate(
                (item.coverage_fraction for item in items), bounded=True
            ),
            entropy_reduction_nats=estimate(
                item.entropy_reduction_nats for item in items
            ),
            successful_elapsed_time_s=estimate(
                item.elapsed_time_s for item in successful
            ),
            successful_distance_m=estimate(
                item.distance_travelled_m for item in successful
            ),
        ))
    return tuple(aggregates)


def write_benchmark_report(
    report: SearchBenchmarkReport,
    output_directory: str,
) -> Mapping[str, str]:
    """Write JSON, per-episode CSV, and aggregate CSV artifacts."""
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "search_benchmark_report.json"
    episode_path = output / "search_benchmark_episodes.csv"
    aggregate_path = output / "search_benchmark_aggregates.csv"

    json_path.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    episode_rows = [_episode_row(item) for item in report.episodes]
    _write_csv(episode_path, episode_rows)
    aggregate_rows = [
        _aggregate_row(item)
        for item in report.aggregates + report.condition_aggregates
    ]
    _write_csv(aggregate_path, aggregate_rows)
    return {
        "report_json": str(json_path),
        "episodes_csv": str(episode_path),
        "aggregates_csv": str(aggregate_path),
    }


def _episode_row(item: SearchEpisodeResult) -> Dict[str, object]:
    data = item.to_dict()
    data.pop("policy_trace", None)
    data.pop("belief_entropy_trace", None)
    return data


def _aggregate_row(item: PolicyAggregate) -> Dict[str, object]:
    row: Dict[str, object] = {
        "policy_name": item.policy_name,
        "prior_condition": item.prior_condition,
        "episode_count": item.episode_count,
    }
    for name, value in item.__dict__.items():
        if isinstance(value, MetricEstimate):
            row[f"{name}_mean"] = value.mean
            row[f"{name}_ci95_low"] = value.ci95_low
            row[f"{name}_ci95_high"] = value.ci95_high
            row[f"{name}_sample_count"] = value.sample_count
    return row


def _write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
