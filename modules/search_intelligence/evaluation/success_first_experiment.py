"""Tune and validate a success-first policy with an upgraded camera profile."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from ..belief import BinarySensorModel
from ..policies import SuccessConstrainedSupervisorPolicy
from .contracts import SearchBenchmarkConfig, SearchEpisodeResult
from .hybrid_experiment import _paired_comparison, _policy_metrics
from .runner import SearchEpisodeRunner
from .unified_benchmark import search_skill_scenario_matrix, unified_benchmark_config


@dataclass(frozen=True)
class SuccessFirstVariant:
    variant_id: str
    detection_weight: float
    information_gain_weight: float
    novelty_weight: float
    travel_weight: float
    recovery_reserve_actions: int
    required_quality_coverage: float = 0.6
    recovery_novelty_weight: float = 1.0


HIGH_RES_CAMERA_PROFILE = {
    "profile_id": "high-resolution-camera-v1",
    "assumed_resolution_px": [3840, 2160],
    "detection_probability": 0.96,
    "false_positive_probability": 0.005,
    "effective_recognition_radius_m": 30.0,
    "detection_confidence": 0.98,
    "camera_and_compute_power_model_included": False,
    "scope_note": (
        "Synthetic sensor-profile assumption; it does not replace real-camera "
        "pixel, weather, motion-blur, calibration, or power validation."
    ),
}


SUCCESS_FIRST_VARIANTS = (
    SuccessFirstVariant("highres_d_frozen_weights", 1.0, 0.8, 0.4, 0.12, 0),
    SuccessFirstVariant("detection_priority", 1.4, 0.65, 0.4, 0.08, 0),
    SuccessFirstVariant("balanced_success", 1.2, 0.75, 0.65, 0.08, 0),
    SuccessFirstVariant("late_recovery_2", 1.2, 0.75, 0.65, 0.08, 2, 0.6, 1.0),
    SuccessFirstVariant("late_recovery_3", 1.2, 0.75, 0.65, 0.08, 3, 0.6, 1.0),
    SuccessFirstVariant(
        "late_recovery_2_high_coverage",
        1.4,
        0.65,
        0.4,
        0.08,
        2,
        0.7,
        1.3,
    ),
)


class _VariantEpisodeRunner(SearchEpisodeRunner):
    def __init__(
        self,
        config: SearchBenchmarkConfig,
        variant: SuccessFirstVariant,
    ) -> None:
        super().__init__(config)
        self.variant = variant

    def _policy(self, policy_name, scenario, candidates, seed):
        if policy_name != "success_constrained":
            return super()._policy(policy_name, scenario, candidates, seed)
        default_policy = super()._policy(
            "improved_active",
            scenario,
            candidates,
            seed,
        )
        recovery_policy = replace(
            default_policy,
            detection_weight=0.8,
            information_gain_weight=0.4,
            novelty_weight=self.variant.recovery_novelty_weight,
            travel_weight=0.05,
            exploitation_fraction=0.2,
            exploration_fraction=0.6,
            semantic_fraction=0.2,
        )
        return SuccessConstrainedSupervisorPolicy(
            default_policy=default_policy,
            recovery_policy=recovery_policy,
            recovery_reserve_actions=self.variant.recovery_reserve_actions,
            required_quality_coverage=self.variant.required_quality_coverage,
        )


def run_success_first_experiment(
    *,
    tuning_repetitions: int = 5,
    tuning_base_seed: int = 5,
    validation_repetitions: int = 20,
    validation_base_seed: int = 10,
) -> Mapping[str, object]:
    scenarios = search_skill_scenario_matrix()
    tuning_results = []
    for variant in SUCCESS_FIRST_VARIANTS:
        episodes = _run_success_variant(
            variant,
            scenarios,
            repetitions=tuning_repetitions,
            base_seed=tuning_base_seed,
        )
        metrics = _extended_metrics(episodes, scenarios)
        tuning_results.append({
            "variant": asdict(variant),
            "metrics": metrics,
            "lexicographic_rank_key": list(_rank_key(metrics)),
        })
    selected = max(
        tuning_results,
        key=lambda item: tuple(item["lexicographic_rank_key"]),
    )
    selected_variant = next(
        variant for variant in SUCCESS_FIRST_VARIANTS
        if variant.variant_id == selected["variant"]["variant_id"]
    )

    current_d = _run_reference_d(
        scenarios,
        repetitions=validation_repetitions,
        base_seed=validation_base_seed,
        high_resolution_camera=False,
    )
    high_res_d = _run_reference_d(
        scenarios,
        repetitions=validation_repetitions,
        base_seed=validation_base_seed,
        high_resolution_camera=True,
    )
    tuned_e2 = _run_success_variant(
        selected_variant,
        scenarios,
        repetitions=validation_repetitions,
        base_seed=validation_base_seed,
    )
    current_metrics = _extended_metrics(current_d, scenarios)
    high_res_metrics = _extended_metrics(high_res_d, scenarios)
    tuned_metrics = _extended_metrics(tuned_e2, scenarios)
    hardware_comparison = _comparison_labels(_paired_comparison(
        current_d,
        high_res_d,
        current_metrics,
        high_res_metrics,
    ))
    policy_comparison = _comparison_labels(_paired_comparison(
        high_res_d,
        tuned_e2,
        high_res_metrics,
        tuned_metrics,
    ))
    return {
        "schema_version": "gsi-search-skill-success-first-v1",
        "experiment_id": "success-first-high-resolution-camera-2026-08-06",
        "baseline_reference": {
            "baseline_id": "search-skill-baseline-2026-08-05",
            "baseline_modified": False,
        },
        "objective_order": [
            "success_rate",
            "worst_case_success_rate",
            "success_per_km",
            "distance_per_success",
        ],
        "camera_profile": HIGH_RES_CAMERA_PROFILE,
        "tuning": {
            "seed_block": {
                "base_seed": tuning_base_seed,
                "repetitions": tuning_repetitions,
            },
            "candidate_count": len(SUCCESS_FIRST_VARIANTS),
            "results": tuning_results,
            "selected_variant": selected_variant.variant_id,
            "selection_rule": (
                "Lexicographic maximum: success rate, worst-case success, "
                "success per kilometer, then negative distance per success."
            ),
        },
        "held_out_validation": {
            "seed_block": {
                "base_seed": validation_base_seed,
                "repetitions": validation_repetitions,
            },
            "condition_count": len(scenarios),
            "episodes_per_method": len(current_d),
            "total_episode_count": len(current_d) + len(high_res_d) + len(tuned_e2),
            "methods": {
                "D_current_camera": current_metrics,
                "D_high_res_camera": high_res_metrics,
                "E2_tuned_high_res": tuned_metrics,
            },
            "hardware_effect_D_high_res_vs_D_current": hardware_comparison,
            "policy_effect_E2_vs_D_high_res": policy_comparison,
            "promotion_decision": _promotion_decision(
                high_res_metrics,
                tuned_metrics,
                policy_comparison,
            ),
            "scenario_results": _validation_scenarios(
                scenarios,
                current_d,
                high_res_d,
                tuned_e2,
            ),
            "e2_mode_usage": _success_mode_usage(tuned_e2),
        },
    }


def write_success_first_experiment(
    payload: Mapping[str, object],
    output_path: str,
) -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return str(path)


def _high_res_config(
    repetitions: int,
    base_seed: int,
    *,
    policy_name: str,
    variant: SuccessFirstVariant | None = None,
) -> SearchBenchmarkConfig:
    config = unified_benchmark_config(
        repetitions=repetitions,
        base_seed=base_seed,
    )
    changes: Dict[str, Any] = {
        "policy_names": (policy_name,),
        "sensor_model": BinarySensorModel(
            detection_probability=HIGH_RES_CAMERA_PROFILE["detection_probability"],
            false_positive_probability=HIGH_RES_CAMERA_PROFILE[
                "false_positive_probability"
            ],
        ),
        "footprint_radius_m": HIGH_RES_CAMERA_PROFILE[
            "effective_recognition_radius_m"
        ],
        "detection_confidence": HIGH_RES_CAMERA_PROFILE["detection_confidence"],
    }
    if variant is not None:
        changes.update({
            "detection_weight": variant.detection_weight,
            "information_gain_weight": variant.information_gain_weight,
            "novelty_weight": variant.novelty_weight,
            "travel_weight": variant.travel_weight,
        })
    return replace(config, **changes)


def _run_success_variant(
    variant,
    scenarios,
    *,
    repetitions,
    base_seed,
):
    config = _high_res_config(
        repetitions,
        base_seed,
        policy_name="success_constrained",
        variant=variant,
    )
    runner = _VariantEpisodeRunner(config, variant)
    return tuple(
        runner.run(scenario, "success_constrained", repetition)
        for scenario in scenarios
        for repetition in range(repetitions)
    )


def _run_reference_d(
    scenarios,
    *,
    repetitions,
    base_seed,
    high_resolution_camera,
):
    if high_resolution_camera:
        config = _high_res_config(
            repetitions,
            base_seed,
            policy_name="improved_active",
        )
    else:
        config = replace(
            unified_benchmark_config(
                repetitions=repetitions,
                base_seed=base_seed,
            ),
            policy_names=("improved_active",),
        )
    runner = SearchEpisodeRunner(config)
    return tuple(
        runner.run(scenario, "improved_active", repetition)
        for scenario in scenarios
        for repetition in range(repetitions)
    )


def _extended_metrics(episodes, scenarios):
    metrics = dict(_policy_metrics(episodes, scenarios))
    successes = sum(item.target_found for item in episodes)
    metrics.update({
        "distance_per_success_m": (
            sum(item.distance_travelled_m for item in episodes) / successes
            if successes else None
        ),
        "energy_per_success": (
            sum(item.energy_used for item in episodes) / successes
            if successes else None
        ),
        "restricted_mean_search_time_s": (
            sum(item.elapsed_time_s for item in episodes) / len(episodes)
        ),
    })
    return metrics


def _rank_key(metrics):
    return (
        metrics["success_rate"]["mean"],
        metrics["worst_case_scenario_success_rate"],
        metrics["success_per_km"] or 0.0,
        -(metrics["distance_per_success_m"] or float("inf")),
    )


def _promotion_decision(high_res_d, tuned_e2, comparison):
    success_delta = comparison["success_rate_delta"]
    success_non_decreasing = success_delta >= 0.0
    worst_non_decreasing = (
        tuned_e2["worst_case_scenario_success_rate"]
        >= high_res_d["worst_case_scenario_success_rate"]
    )
    resource_improved = (
        comparison["success_per_km_ratio"] is not None
        and comparison["success_per_km_ratio"] >= 1.1
    )
    promote = (
        success_non_decreasing
        and worst_non_decreasing
        and (success_delta > 0.0 or resource_improved)
    )
    return {
        "decision": "PROMOTE_E2" if promote else "DO_NOT_PROMOTE_E2",
        "success_non_decreasing": success_non_decreasing,
        "worst_case_non_decreasing": worst_non_decreasing,
        "success_improved": success_delta > 0.0,
        "resource_improved_at_equal_success": resource_improved,
    }


def _comparison_labels(comparison):
    normalized = dict(comparison)
    normalized["candidate_only_successes"] = normalized.pop(
        "hybrid_only_successes"
    )
    normalized["reference_only_successes"] = normalized.pop(
        "improved_only_successes"
    )
    return normalized


def _validation_scenarios(scenarios, current_d, high_res_d, tuned_e2):
    methods = {
        "D_current_camera": current_d,
        "D_high_res_camera": high_res_d,
        "E2_tuned_high_res": tuned_e2,
    }
    rows = []
    for scenario in scenarios:
        row = {
            "scenario_id": scenario.scenario_id,
            "environment": scenario.metadata["environment"],
            "prior_condition": scenario.prior_condition,
            "sensor_condition": scenario.metadata["sensor_condition"],
            "success_rate": {},
        }
        for method, episodes in methods.items():
            selected = tuple(
                item for item in episodes if item.scenario_id == scenario.scenario_id
            )
            row["success_rate"][method] = (
                sum(item.target_found for item in selected) / len(selected)
            )
        rows.append(row)
    return tuple(rows)


def _success_mode_usage(episodes: Sequence[SearchEpisodeResult]):
    counts: Dict[str, int] = {}
    switches = 0
    for episode in episodes:
        for decision in episode.policy_trace:
            mode = str(decision.get("success_supervisor_mode", "unknown"))
            counts[mode] = counts.get(mode, 0) + 1
            switches += bool(decision.get("success_supervisor_mode_switched", False))
    total = sum(counts.values())
    return {
        "actions_by_mode": dict(sorted(counts.items())),
        "action_fraction_by_mode": {
            mode: count / total for mode, count in sorted(counts.items())
        } if total else {},
        "mean_switches_per_episode": switches / len(episodes) if episodes else 0.0,
    }
