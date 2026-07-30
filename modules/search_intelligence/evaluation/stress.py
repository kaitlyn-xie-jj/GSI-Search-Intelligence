"""Parameterized stress scenarios and artifact writers for search policies."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

from ..belief import BinarySensorModel
from ..contracts import SearchTask
from ..search_space import SearchGrid
from .contracts import (
    SUPPORTED_POLICIES,
    MetricEstimate,
    SearchBenchmarkConfig,
    SearchBenchmarkReport,
    SearchBenchmarkScenario,
    SearchEpisodeResult,
)
from .defaults import focused_grid_belief
from .reporting import estimate, write_benchmark_report
from .runner import SearchBenchmarkRunner


@dataclass(frozen=True)
class SearchStressProfile:
    """One sensor/resource condition applied to the shared stress scenarios."""

    profile_id: str
    description: str
    sensor_model: BinarySensorModel = field(default_factory=BinarySensorModel)
    observation_quality: float = 1.0
    budget_scale: float = 1.0
    min_confirmations: int = 1

    def __post_init__(self) -> None:
        if not self.profile_id.strip():
            raise ValueError("profile_id must not be empty")
        if not 0.0 <= self.observation_quality <= 1.0:
            raise ValueError("observation_quality must be within [0, 1]")
        if not math.isfinite(self.budget_scale) or self.budget_scale <= 0:
            raise ValueError("budget_scale must be finite and positive")
        if self.min_confirmations <= 0:
            raise ValueError("min_confirmations must be positive")

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["sensor_model"] = asdict(self.sensor_model)
        return data


@dataclass(frozen=True)
class SearchStressRun:
    """Benchmark report plus the profile and scenario metadata that produced it."""

    profile: SearchStressProfile
    scenarios: Tuple[SearchBenchmarkScenario, ...]
    report: SearchBenchmarkReport


@dataclass(frozen=True)
class _LayoutSpec:
    layout_id: str
    geometry: Mapping[str, Any]
    resolution_m: float
    start_xy: Tuple[float, float]
    base_max_viewpoints: int
    target_cells: Mapping[str, Tuple[int, int]]


def default_stress_profiles() -> Tuple[SearchStressProfile, ...]:
    """Return paired nominal, sensing, and resource stress conditions."""
    return (
        SearchStressProfile(
            "nominal",
            "Nominal detector, observation quality, and viewpoint budget.",
            BinarySensorModel(0.85, 0.01),
        ),
        SearchStressProfile(
            "degraded_sensor",
            "Lower recall and moderately elevated false-positive probability.",
            BinarySensorModel(0.65, 0.05),
        ),
        SearchStressProfile(
            "low_observation_quality",
            "Nominal detector observed through low-quality sensor frames.",
            BinarySensorModel(0.85, 0.01),
            observation_quality=0.5,
        ),
        SearchStressProfile(
            "high_false_alarm",
            "Nominal recall with a high false-positive probability.",
            BinarySensorModel(0.85, 0.15),
        ),
        SearchStressProfile(
            "tight_budget",
            "Nominal sensing with 45 percent of the normal viewpoint budget.",
            BinarySensorModel(0.85, 0.01),
            budget_scale=0.45,
        ),
    )


def verification_stress_profiles() -> Tuple[SearchStressProfile, ...]:
    """Return paired profiles that isolate two-observation target verification."""
    return (
        SearchStressProfile(
            "verified_nominal",
            "Nominal sensing with two independent target confirmations.",
            BinarySensorModel(0.85, 0.01),
            min_confirmations=2,
        ),
        SearchStressProfile(
            "verified_high_false_alarm",
            "High false alarms with two independent target confirmations.",
            BinarySensorModel(0.85, 0.15),
            min_confirmations=2,
        ),
    )


def stress_benchmark_scenarios(
    *,
    budget_scale: float = 1.0,
    min_confirmations: int = 1,
) -> Tuple[SearchBenchmarkScenario, ...]:
    """Build 24 matched scenarios spanning map, target, and prior conditions."""
    if not math.isfinite(budget_scale) or budget_scale <= 0:
        raise ValueError("budget_scale must be finite and positive")
    if min_confirmations <= 0:
        raise ValueError("min_confirmations must be positive")
    scenarios = []
    for layout in _layout_specs():
        for target_position, target_index in layout.target_cells.items():
            for prior_condition in ("correct", "diffuse", "uniform", "misleading"):
                scenarios.append(_stress_scenario(
                    layout,
                    target_position,
                    target_index,
                    prior_condition,
                    budget_scale,
                    min_confirmations,
                ))
    return tuple(scenarios)


def run_stress_benchmark(
    profiles: Iterable[SearchStressProfile],
    *,
    repetitions: int = 20,
    base_seed: int = 0,
    policy_names: Sequence[str] = SUPPORTED_POLICIES,
) -> Tuple[SearchStressRun, ...]:
    """Run every profile over a freshly budgeted copy of the shared scenarios."""
    runs = []
    for profile in tuple(profiles):
        scenarios = stress_benchmark_scenarios(
            budget_scale=profile.budget_scale,
            min_confirmations=profile.min_confirmations,
        )
        config = SearchBenchmarkConfig(
            policy_names=tuple(policy_names),
            repetitions=repetitions,
            base_seed=base_seed,
            sensor_model=profile.sensor_model,
            observation_quality=profile.observation_quality,
        )
        runs.append(SearchStressRun(
            profile=profile,
            scenarios=scenarios,
            report=SearchBenchmarkRunner(config).run(scenarios),
        ))
    return tuple(runs)


def write_stress_benchmark_results(
    runs: Sequence[SearchStressRun],
    output_directory: str,
) -> Mapping[str, str]:
    """Write combined long-form data plus full trace reports for each profile."""
    if not runs:
        raise ValueError("at least one stress benchmark run is required")
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    episode_path = output / "stress_benchmark_episodes.csv"
    summary_path = output / "stress_benchmark_summary.csv"
    manifest_path = output / "stress_benchmark_manifest.json"

    episode_rows = []
    profile_artifacts: Dict[str, Mapping[str, str]] = {}
    for run in runs:
        profile_artifacts[run.profile.profile_id] = write_benchmark_report(
            run.report,
            str(output / "profiles" / run.profile.profile_id),
        )
        scenarios = {item.scenario_id: item for item in run.scenarios}
        for episode in run.report.episodes:
            episode_rows.append(_stress_episode_row(
                run.profile,
                scenarios[episode.scenario_id],
                episode,
            ))

    summary_rows = _stress_summary_rows(runs)
    _write_rows(episode_path, episode_rows)
    _write_rows(summary_path, summary_rows)
    manifest = {
        "schema_version": "gsi-search-stress-v1",
        "profiles": [run.profile.to_dict() for run in runs],
        "profile_artifacts": profile_artifacts,
        "episode_count": len(episode_rows),
        "summary_row_count": len(summary_rows),
        "episodes_csv": str(episode_path),
        "summary_csv": str(summary_path),
        "runs": [
            {
                "profile_id": run.profile.profile_id,
                "config": run.report.config.to_dict(),
                "scenarios": [scenario.to_dict() for scenario in run.scenarios],
            }
            for run in runs
        ],
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return {
        "manifest_json": str(manifest_path),
        "episodes_csv": str(episode_path),
        "summary_csv": str(summary_path),
    }


def _layout_specs() -> Tuple[_LayoutSpec, ...]:
    return (
        _LayoutSpec(
            "compact_rectangle",
            {
                "kind": "rectangle",
                "coords": [[0, 0], [100, 0], [100, 80], [0, 80]],
            },
            20.0,
            (10.0, 10.0),
            20,
            {"near": (1, 1), "far": (3, 4)},
        ),
        _LayoutSpec(
            "large_rectangle",
            {
                "kind": "rectangle",
                "coords": [[0, 0], [200, 0], [200, 160], [0, 160]],
            },
            20.0,
            (10.0, 10.0),
            48,
            {"near": (2, 2), "far": (7, 9)},
        ),
        _LayoutSpec(
            "l_shape",
            {
                "kind": "area",
                "coords": [
                    [0, 0], [160, 0], [160, 60],
                    [80, 60], [80, 160], [0, 160],
                ],
            },
            20.0,
            (10.0, 10.0),
            30,
            {"near": (2, 2), "far": (7, 3)},
        ),
    )


def _stress_scenario(
    layout: _LayoutSpec,
    target_position: str,
    target_index: Tuple[int, int],
    prior_condition: str,
    budget_scale: float,
    min_confirmations: int,
) -> SearchBenchmarkScenario:
    scenario_id = f"{layout.layout_id}-{target_position}-{prior_condition}"
    area_id = f"stress-{scenario_id}"
    max_viewpoints = max(1, int(math.ceil(
        layout.base_max_viewpoints * budget_scale
    )))
    task = SearchTask.from_skill_params({
        "task_id": scenario_id,
        "area_token": area_id,
        "area": layout.geometry,
        "target_token": "yellow-van",
        "max_viewpoints": max_viewpoints,
        "conf_ge": 0.5,
        "min_confirmations": min_confirmations,
    })
    grid = SearchGrid.from_task(task, resolution_m=layout.resolution_m)
    target_cell = grid.cell(*target_index)
    if target_cell is None or not target_cell.searchable:
        raise ValueError(f"target {target_index} is invalid for {layout.layout_id}")
    ordered_near = sorted(
        grid.searchable_cells,
        key=lambda cell: (
            math.hypot(
                cell.center[0] - target_cell.center[0],
                cell.center[1] - target_cell.center[1],
            ),
            cell.cell_id,
        ),
    )
    if prior_condition == "uniform":
        belief = grid.uniform_belief()
        focus_ids: Tuple[str, ...] = ()
        focus_mass = 0.0
    elif prior_condition == "correct":
        focus_ids = tuple(cell.cell_id for cell in ordered_near[:4])
        focus_mass = 0.75
        belief = focused_grid_belief(grid, focus_ids, focus_mass)
    elif prior_condition == "diffuse":
        focus_ids = tuple(cell.cell_id for cell in ordered_near[:8])
        focus_mass = 0.45
        belief = focused_grid_belief(grid, focus_ids, focus_mass)
    elif prior_condition == "misleading":
        focus_ids = tuple(cell.cell_id for cell in reversed(ordered_near[-4:]))
        focus_mass = 0.75
        belief = focused_grid_belief(grid, focus_ids, focus_mass)
    else:
        raise ValueError(f"unsupported prior condition: {prior_condition}")
    return SearchBenchmarkScenario(
        scenario_id=scenario_id,
        task=task,
        grid=grid,
        target_cell_id=target_cell.cell_id,
        initial_belief=belief,
        start_xy=layout.start_xy,
        prior_condition=prior_condition,
        metadata={
            "layout": layout.layout_id,
            "target_position": target_position,
            "grid_resolution_m": layout.resolution_m,
            "searchable_cell_count": len(grid.searchable_cells),
            "prior_focus_cell_ids": focus_ids,
            "prior_focus_mass": focus_mass,
            "max_viewpoints": max_viewpoints,
            "budget_scale": budget_scale,
            "min_confirmations": min_confirmations,
        },
    )


def _stress_episode_row(
    profile: SearchStressProfile,
    scenario: SearchBenchmarkScenario,
    episode: SearchEpisodeResult,
) -> Dict[str, object]:
    row: Dict[str, object] = episode.to_dict()
    row.pop("policy_trace", None)
    row.pop("belief_entropy_trace", None)
    row.update({
        "profile_id": profile.profile_id,
        "layout": scenario.metadata.get("layout", "unspecified"),
        "target_position": scenario.metadata.get("target_position", "unspecified"),
        "searchable_cell_count": scenario.metadata.get("searchable_cell_count"),
        "max_viewpoints": scenario.task.budget.max_viewpoints,
        "observation_quality": profile.observation_quality,
        "detection_probability": profile.sensor_model.detection_probability,
        "false_positive_probability": (
            profile.sensor_model.false_positive_probability
        ),
        "min_confirmations": profile.min_confirmations,
    })
    return row


def _stress_summary_rows(
    runs: Sequence[SearchStressRun],
) -> Sequence[Dict[str, object]]:
    rows = []
    for run in runs:
        scenarios = {item.scenario_id: item for item in run.scenarios}
        for policy_name in run.report.config.policy_names:
            policy_episodes = tuple(
                item for item in run.report.episodes
                if item.policy_name == policy_name
            )
            groups: Dict[Tuple[str, str, str, str], list] = {
                ("overall", "all", "all", "all"): list(policy_episodes)
            }
            for episode in policy_episodes:
                scenario = scenarios[episode.scenario_id]
                layout = str(scenario.metadata.get("layout", "unspecified"))
                target = str(scenario.metadata.get(
                    "target_position", "unspecified"
                ))
                prior = episode.prior_condition
                keys = (
                    ("prior", "all", "all", prior),
                    ("layout", layout, "all", "all"),
                    ("target_position", "all", target, "all"),
                    ("layout_prior", layout, "all", prior),
                )
                for key in keys:
                    groups.setdefault(key, []).append(episode)
            for (scope, layout, target, prior), episodes in groups.items():
                rows.append(_summary_row(
                    run.profile.profile_id,
                    policy_name,
                    scope,
                    layout,
                    target,
                    prior,
                    episodes,
                ))
    return rows


def _summary_row(
    profile_id: str,
    policy_name: str,
    scope: str,
    layout: str,
    target_position: str,
    prior_condition: str,
    episodes: Sequence[SearchEpisodeResult],
) -> Dict[str, object]:
    row: Dict[str, object] = {
        "profile_id": profile_id,
        "policy_name": policy_name,
        "scope": scope,
        "layout": layout,
        "target_position": target_position,
        "prior_condition": prior_condition,
        "episode_count": len(episodes),
    }
    metrics = {
        "success_rate": estimate(
            (float(item.target_found) for item in episodes), bounded=True
        ),
        "false_positive_rate": estimate(
            (float(item.false_positive) for item in episodes), bounded=True
        ),
        "spl": estimate((item.spl for item in episodes), bounded=True),
        "elapsed_time_s": estimate(item.elapsed_time_s for item in episodes),
        "distance_travelled_m": estimate(
            item.distance_travelled_m for item in episodes
        ),
        "energy_used": estimate(item.energy_used for item in episodes),
        "entropy_reduction_nats": estimate(
            item.entropy_reduction_nats for item in episodes
        ),
    }
    for metric_name, metric in metrics.items():
        _add_estimate(row, metric_name, metric)
    return row


def _add_estimate(
    row: Dict[str, object],
    name: str,
    metric: MetricEstimate,
) -> None:
    row[f"{name}_mean"] = metric.mean
    row[f"{name}_ci95_low"] = metric.ci95_low
    row[f"{name}_ci95_high"] = metric.ci95_high
    row[f"{name}_sample_count"] = metric.sample_count


def _write_rows(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
