"""Unified A/B/C/D Search Skill acceptance benchmark."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Mapping, Sequence, Tuple

from ..contracts import SearchTask
from ..search_space import SearchGrid
from .contracts import SearchBenchmarkConfig, SearchBenchmarkScenario, SearchEpisodeResult
from .defaults import focused_grid_belief
from .reporting import estimate
from .runner import SearchBenchmarkRunner


UNIFIED_POLICIES = ("coverage", "random", "active", "improved_active")
BASELINE_LABELS = {
    "coverage": "A_full_coverage_lawn_mower",
    "random": "B_random_search",
    "active": "C_original_active_search",
    "improved_active": "D_improved_active_search",
}

ENVIRONMENTS = {
    "open_area": {"visibility": 0.95, "risk": 0.05, "target": (0, 4)},
    "street_edge": {"visibility": 0.65, "risk": 0.25, "target": (1, 5)},
    "woodland": {"visibility": 0.35, "risk": 0.7, "target": (4, 3)},
    "building_passage": {"visibility": 0.2, "risk": 0.55, "target": (3, 4)},
}
PRIOR_CONDITIONS = ("correct", "wrong", "uniform")
SENSOR_CONDITIONS = {
    "normal": 1.0,
    "reduced_quality": 0.55,
}


def unified_benchmark_config(
    *,
    repetitions: int = 20,
    base_seed: int = 0,
) -> SearchBenchmarkConfig:
    return SearchBenchmarkConfig(
        policy_names=UNIFIED_POLICIES,
        repetitions=repetitions,
        base_seed=base_seed,
        altitude_m=30.0,
        footprint_radius_m=25.0,
        speed_mps=10.0,
        observation_time_s=1.0,
        coverage_pass_spacing_m=20.0,
        coverage_observation_spacing_m=20.0,
        detection_weight=1.0,
        information_gain_weight=0.8,
        novelty_weight=0.4,
        travel_weight=0.12,
        revisit_weight=0.2,
        risk_weight=0.15,
        distance_scale_mode="map_diagonal",
        lookahead_discount_factor=0.7,
        lookahead_candidate_limit=16,
        completion_time_reserve_s=3.0,
    )


def search_skill_scenario_matrix() -> Tuple[SearchBenchmarkScenario, ...]:
    scenarios = []
    for environment, environment_data in ENVIRONMENTS.items():
        for prior_condition in PRIOR_CONDITIONS:
            for sensor_condition, sensor_quality in SENSOR_CONDITIONS.items():
                scenarios.append(_matrix_scenario(
                    environment,
                    prior_condition,
                    sensor_condition,
                    float(environment_data["visibility"]),
                    float(environment_data["risk"]),
                    tuple(environment_data["target"]),
                    sensor_quality,
                ))
    return tuple(scenarios)


def run_unified_benchmark(
    *,
    repetitions: int = 20,
    base_seed: int = 0,
) -> Mapping[str, object]:
    config = unified_benchmark_config(
        repetitions=repetitions,
        base_seed=base_seed,
    )
    scenarios = search_skill_scenario_matrix()
    report = SearchBenchmarkRunner(config).run(scenarios)
    return _evaluation_payload(config, scenarios, report.episodes)


def write_unified_evaluation_report(
    payload: Mapping[str, object],
    output_path: str,
) -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return str(path)


def _matrix_scenario(
    environment: str,
    prior_condition: str,
    sensor_condition: str,
    visibility_probability: float,
    risk_score: float,
    target_index: Tuple[int, int],
    sensor_quality: float,
) -> SearchBenchmarkScenario:
    scenario_id = f"{environment}__{prior_condition}__{sensor_condition}"
    area_id = f"acceptance-{scenario_id}"
    task = SearchTask.from_skill_params({
        "task_id": scenario_id,
        "area_token": area_id,
        "area": {
            "kind": "rectangle",
            "coords": [[0, 0], [120, 0], [120, 100], [0, 100]],
        },
        "target_token": "yellow-van",
        "max_viewpoints": 10,
        "time_budget_s": 55.0,
        "min_confirmations": 2,
        "conf_ge": 0.5,
    })
    grid = SearchGrid.from_task(task, resolution_m=20.0)
    target_cell = grid.cell(*target_index)
    if target_cell is None:
        raise ValueError(f"target index is outside matrix grid: {target_index}")
    if prior_condition == "correct":
        focus = tuple(
            cell.cell_id for cell in grid.searchable_cells
            if abs(cell.row - target_cell.row) + abs(cell.column - target_cell.column) <= 1
        )
        belief = focused_grid_belief(grid, focus, 0.75)
    elif prior_condition == "wrong":
        focus = tuple(cell.cell_id for cell in grid.searchable_cells[:4])
        belief = focused_grid_belief(grid, focus, 0.8)
    else:
        belief = grid.uniform_belief()
    semantic_region_by_cell = {
        cell.cell_id: f"{environment}:sector-{cell.row // 2}-{cell.column // 2}"
        for cell in grid.searchable_cells
    }
    return SearchBenchmarkScenario(
        scenario_id=scenario_id,
        task=task,
        grid=grid,
        target_cell_id=target_cell.cell_id,
        initial_belief=belief,
        start_xy=(10.0, 10.0),
        prior_condition=prior_condition,
        metadata={
            "environment": environment,
            "sensor_condition": sensor_condition,
            "sensor_quality": sensor_quality,
            "visibility_probability": visibility_probability,
            "risk_score": risk_score,
            "semantic_region_by_cell": semantic_region_by_cell,
        },
    )


def _evaluation_payload(
    config: SearchBenchmarkConfig,
    scenarios: Sequence[SearchBenchmarkScenario],
    episodes: Sequence[SearchEpisodeResult],
) -> Mapping[str, object]:
    scenario_by_id = {scenario.scenario_id: scenario for scenario in scenarios}
    policy_results = {
        policy_name: _policy_metrics(
            tuple(item for item in episodes if item.policy_name == policy_name),
            scenarios,
        )
        for policy_name in UNIFIED_POLICIES
    }
    scenario_results = []
    for scenario in scenarios:
        scenario_results.append({
            "scenario_id": scenario.scenario_id,
            "environment": scenario.metadata["environment"],
            "prior_condition": scenario.prior_condition,
            "sensor_condition": scenario.metadata["sensor_condition"],
            "visibility_probability": scenario.metadata["visibility_probability"],
            "sensor_quality": scenario.metadata["sensor_quality"],
            "policies": {
                policy_name: _compact_metrics(tuple(
                    item for item in episodes
                    if item.scenario_id == scenario.scenario_id
                    and item.policy_name == policy_name
                ))
                for policy_name in UNIFIED_POLICIES
            },
        })
    improved = policy_results["improved_active"]
    coverage = policy_results["coverage"]
    original_active = policy_results["active"]
    return {
        "schema_version": "gsi-search-skill-evaluation-v1",
        "baseline_id": "search-skill-baseline-2026-08-05",
        "baselines": BASELINE_LABELS,
        "method_isolation": {
            "coverage": {
                "implementation": "CoveragePolicy",
                "confidence_gating_enabled": False,
            },
            "random": {
                "implementation": "RandomPolicy",
                "confidence_gating_enabled": False,
            },
            "active": {
                "implementation": "OriginalActiveSearchPolicy",
                "confidence_gating_enabled": False,
            },
            "improved_active": {
                "implementation": "AdaptiveBeliefLookaheadPolicy",
                "confidence_gating_enabled": True,
            },
        },
        "configuration": {
            "repetitions_per_condition": config.repetitions,
            "seed_count": config.repetitions,
            "scenario_condition_count": len(scenarios),
            "episode_count": len(episodes),
            "shared_map_size_m": [120.0, 100.0],
            "shared_grid_resolution_m": 20.0,
            "shared_speed_mps": config.speed_mps,
            "shared_time_budget_s": 55.0,
            "shared_max_viewpoints": 10,
            "shared_sensor_model": asdict(config.sensor_model),
            "policy_config": config.to_dict(),
        },
        "policy_results": policy_results,
        "scenario_results": scenario_results,
        "comparison_against_coverage": {
            "success_rate_delta": (
                improved["success_rate"]["mean"]
                - coverage["success_rate"]["mean"]
            ),
            "successful_distance_ratio": _optional_ratio(
                improved["mean_detection_distance_m"]["mean"],
                coverage["mean_detection_distance_m"]["mean"],
            ),
            "success_per_km_ratio": _optional_ratio(
                improved["success_per_km"],
                coverage["success_per_km"],
            ),
        },
        "comparison_against_original_active": {
            "success_rate_delta": (
                improved["success_rate"]["mean"]
                - original_active["success_rate"]["mean"]
            ),
            "successful_distance_ratio": _optional_ratio(
                improved["mean_detection_distance_m"]["mean"],
                original_active["mean_detection_distance_m"]["mean"],
            ),
            "mean_total_distance_ratio": _optional_ratio(
                improved["mean_distance_m"]["mean"],
                original_active["mean_distance_m"]["mean"],
            ),
            "mean_replan_ratio": _optional_ratio(
                improved["replans"]["mean"],
                original_active["replans"]["mean"],
            ),
            "belief_brier_ratio": _optional_ratio(
                improved["belief_calibration_brier"]["mean"],
                original_active["belief_calibration_brier"]["mean"],
            ),
        },
        "failure_categories": {
            policy_name: dict(sorted(Counter(
                item.failure_category for item in episodes
                if item.policy_name == policy_name
            ).items()))
            for policy_name in UNIFIED_POLICIES
        },
        "scenario_index": {
            scenario_id: {
                "environment": scenario_by_id[scenario_id].metadata["environment"],
                "prior_condition": scenario_by_id[scenario_id].prior_condition,
                "sensor_condition": scenario_by_id[scenario_id].metadata[
                    "sensor_condition"
                ],
            }
            for scenario_id in scenario_by_id
        },
    }


def _policy_metrics(
    episodes: Sequence[SearchEpisodeResult],
    scenarios: Sequence[SearchBenchmarkScenario],
) -> Mapping[str, object]:
    successful = tuple(item for item in episodes if item.target_found)
    scenario_success = []
    for scenario in scenarios:
        items = tuple(item for item in episodes if item.scenario_id == scenario.scenario_id)
        scenario_success.append(sum(item.target_found for item in items) / len(items))
    total_km = sum(item.distance_travelled_m for item in episodes) / 1000.0
    return {
        "label": BASELINE_LABELS[episodes[0].policy_name],
        "episode_count": len(episodes),
        "success_rate": estimate(
            (float(item.target_found) for item in episodes),
            bounded=True,
        ).to_dict(),
        "mean_detection_distance_m": estimate(
            item.distance_travelled_m for item in successful
        ).to_dict(),
        "mean_detection_time_s": estimate(
            item.elapsed_time_s for item in successful
        ).to_dict(),
        "success_per_km": (
            len(successful) / total_km if total_km > 0 else None
        ),
        "replans": estimate(item.replan_count for item in episodes).to_dict(),
        "belief_calibration_brier": estimate(
            item.belief_brier_score for item in episodes
        ).to_dict(),
        "worst_case_scenario_success_rate": min(scenario_success),
        "mean_distance_m": estimate(
            item.distance_travelled_m for item in episodes
        ).to_dict(),
    }


def _compact_metrics(
    episodes: Sequence[SearchEpisodeResult],
) -> Mapping[str, object]:
    successful = tuple(item for item in episodes if item.target_found)
    return {
        "success_rate": estimate(
            (float(item.target_found) for item in episodes),
            bounded=True,
        ).to_dict(),
        "mean_detection_distance_m": estimate(
            item.distance_travelled_m for item in successful
        ).to_dict(),
        "mean_detection_time_s": estimate(
            item.elapsed_time_s for item in successful
        ).to_dict(),
        "mean_replans": estimate(item.replan_count for item in episodes).to_dict(),
        "mean_brier_score": estimate(
            item.belief_brier_score for item in episodes
        ).to_dict(),
        "failure_categories": dict(sorted(Counter(
            item.failure_category for item in episodes
        ).items())),
    }


def _optional_ratio(first: object, second: object):
    if first is None or second is None or float(second) == 0.0:
        return None
    return float(first) / float(second)
