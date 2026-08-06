"""Paired pilot experiment for improved active search and hybrid supervision."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

from .contracts import SearchEpisodeResult
from .reporting import estimate
from .runner import SearchBenchmarkRunner
from .unified_benchmark import search_skill_scenario_matrix, unified_benchmark_config


HYBRID_EXPERIMENT_POLICIES = ("improved_active", "hybrid_supervisor")


def run_hybrid_search_experiment(
    *,
    repetitions: int = 5,
    base_seed: int = 0,
) -> Mapping[str, object]:
    """Run a small paired D/E pilot without changing the frozen report."""
    config = replace(
        unified_benchmark_config(
            repetitions=repetitions,
            base_seed=base_seed,
        ),
        policy_names=HYBRID_EXPERIMENT_POLICIES,
    )
    scenarios = search_skill_scenario_matrix()
    report = SearchBenchmarkRunner(config).run(scenarios)
    episodes = report.episodes
    by_policy = {
        policy_name: tuple(
            episode for episode in episodes
            if episode.policy_name == policy_name
        )
        for policy_name in HYBRID_EXPERIMENT_POLICIES
    }
    improved = _policy_metrics(by_policy["improved_active"], scenarios)
    hybrid = _policy_metrics(by_policy["hybrid_supervisor"], scenarios)
    comparison = _paired_comparison(
        by_policy["improved_active"],
        by_policy["hybrid_supervisor"],
        improved,
        hybrid,
    )
    return {
        "schema_version": "gsi-search-skill-hybrid-pilot-v1",
        "experiment_id": (
            f"hybrid-supervisor-pilot-seeds-{base_seed}-"
            f"{base_seed + repetitions - 1}"
        ),
        "baseline_reference": {
            "baseline_id": "search-skill-baseline-2026-08-05",
            "frozen_method": "D_improved_active_search",
            "frozen_report": (
                "results/search_skill_acceptance/evaluation_report.json"
            ),
            "baseline_modified": False,
        },
        "hypothesis": (
            "A rule-based supervisor can improve worst-case discovery and "
            "resource efficiency by delegating complete viewpoint actions to "
            "the frozen A/B/C/D policies."
        ),
        "configuration": {
            "pilot": repetitions < 20,
            "repetitions_per_condition": repetitions,
            "base_seed": base_seed,
            "condition_count": len(scenarios),
            "episode_count": len(episodes),
            "paired_policies": HYBRID_EXPERIMENT_POLICIES,
            "shared_policy_config": config.to_dict(),
        },
        "method_e": {
            "implementation": "HybridSearchSupervisorPolicy",
            "default_mode": "improved_active",
            "modes": {
                "improved_active": "D default and confirmation",
                "coverage_fallback": "A global coverage fallback",
                "random_escape": "B stagnation escape",
                "visibility_fallback": "C visibility-model fallback",
            },
            "score_fusion": False,
            "confidence_gating_enabled": True,
            "minimum_mode_residence_actions": 2,
        },
        "policy_results": {
            "improved_active": improved,
            "hybrid_supervisor": hybrid,
        },
        "paired_comparison": comparison,
        "pilot_decision": _pilot_decision(improved, hybrid, comparison),
        "hybrid_mode_usage": _mode_usage(by_policy["hybrid_supervisor"]),
        "scenario_results": _scenario_results(episodes, scenarios),
    }


def write_hybrid_search_experiment(
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


def _policy_metrics(
    episodes: Sequence[SearchEpisodeResult],
    scenarios: Sequence[object],
) -> Mapping[str, object]:
    successful = tuple(episode for episode in episodes if episode.target_found)
    scenario_rates = []
    for scenario in scenarios:
        items = tuple(
            episode for episode in episodes
            if episode.scenario_id == scenario.scenario_id
        )
        scenario_rates.append(sum(item.target_found for item in items) / len(items))
    total_km = sum(item.distance_travelled_m for item in episodes) / 1000.0
    return {
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
        "mean_total_distance_m": estimate(
            item.distance_travelled_m for item in episodes
        ).to_dict(),
        "success_per_km": len(successful) / total_km if total_km > 0 else None,
        "replans": estimate(item.replan_count for item in episodes).to_dict(),
        "belief_calibration_brier": estimate(
            item.belief_brier_score for item in episodes
        ).to_dict(),
        "worst_case_scenario_success_rate": min(scenario_rates),
        "failure_categories": dict(sorted(Counter(
            item.failure_category for item in episodes
        ).items())),
    }


def _paired_comparison(
    improved_episodes: Sequence[SearchEpisodeResult],
    hybrid_episodes: Sequence[SearchEpisodeResult],
    improved_metrics: Mapping[str, Any],
    hybrid_metrics: Mapping[str, Any],
) -> Mapping[str, object]:
    key = lambda item: (item.scenario_id, item.repetition)
    improved_by_key = {key(item): item for item in improved_episodes}
    hybrid_by_key = {key(item): item for item in hybrid_episodes}
    paired_keys = tuple(sorted(set(improved_by_key) & set(hybrid_by_key)))
    e_only = sum(
        hybrid_by_key[item].target_found and not improved_by_key[item].target_found
        for item in paired_keys
    )
    d_only = sum(
        improved_by_key[item].target_found and not hybrid_by_key[item].target_found
        for item in paired_keys
    )
    return {
        "pair_count": len(paired_keys),
        "success_rate_delta": (
            hybrid_metrics["success_rate"]["mean"]
            - improved_metrics["success_rate"]["mean"]
        ),
        "hybrid_only_successes": e_only,
        "improved_only_successes": d_only,
        "matched_outcomes": len(paired_keys) - e_only - d_only,
        "mean_total_distance_ratio": _optional_ratio(
            hybrid_metrics["mean_total_distance_m"]["mean"],
            improved_metrics["mean_total_distance_m"]["mean"],
        ),
        "successful_detection_distance_ratio": _optional_ratio(
            hybrid_metrics["mean_detection_distance_m"]["mean"],
            improved_metrics["mean_detection_distance_m"]["mean"],
        ),
        "successful_detection_time_ratio": _optional_ratio(
            hybrid_metrics["mean_detection_time_s"]["mean"],
            improved_metrics["mean_detection_time_s"]["mean"],
        ),
        "success_per_km_ratio": _optional_ratio(
            hybrid_metrics["success_per_km"],
            improved_metrics["success_per_km"],
        ),
        "replan_ratio": _optional_ratio(
            hybrid_metrics["replans"]["mean"],
            improved_metrics["replans"]["mean"],
        ),
        "brier_ratio": _optional_ratio(
            hybrid_metrics["belief_calibration_brier"]["mean"],
            improved_metrics["belief_calibration_brier"]["mean"],
        ),
    }


def _mode_usage(episodes: Sequence[SearchEpisodeResult]) -> Mapping[str, object]:
    mode_counts = Counter()
    reason_counts = Counter()
    switch_count = 0
    for episode in episodes:
        for decision in episode.policy_trace:
            mode_counts[str(decision.get("hybrid_mode", "unknown"))] += 1
            reason_counts[str(decision.get("hybrid_switch_reason", "unknown"))] += 1
            switch_count += bool(
                decision.get("hybrid_mode_switched", False)
                and decision.get("hybrid_previous_mode") is not None
            )
    total = sum(mode_counts.values())
    return {
        "action_count": total,
        "switch_count": switch_count,
        "mean_switches_per_episode": (
            switch_count / len(episodes) if episodes else 0.0
        ),
        "actions_by_mode": dict(sorted(mode_counts.items())),
        "action_fraction_by_mode": {
            mode: count / total for mode, count in sorted(mode_counts.items())
        } if total else {},
        "decisions_by_reason": dict(sorted(reason_counts.items())),
    }


def _scenario_results(
    episodes: Sequence[SearchEpisodeResult],
    scenarios: Sequence[object],
) -> Tuple[Mapping[str, object], ...]:
    results = []
    for scenario in scenarios:
        policies: Dict[str, object] = {}
        for policy_name in HYBRID_EXPERIMENT_POLICIES:
            items = tuple(
                episode for episode in episodes
                if episode.scenario_id == scenario.scenario_id
                and episode.policy_name == policy_name
            )
            policies[policy_name] = {
                "success_rate": sum(item.target_found for item in items) / len(items),
                "mean_distance_m": sum(
                    item.distance_travelled_m for item in items
                ) / len(items),
            }
        results.append({
            "scenario_id": scenario.scenario_id,
            "environment": scenario.metadata["environment"],
            "prior_condition": scenario.prior_condition,
            "sensor_condition": scenario.metadata["sensor_condition"],
            "policies": policies,
        })
    return tuple(results)


def _optional_ratio(first: object, second: object):
    if first is None or second is None or float(second) == 0.0:
        return None
    return float(first) / float(second)


def _pilot_decision(
    improved: Mapping[str, Any],
    hybrid: Mapping[str, Any],
    comparison: Mapping[str, Any],
) -> Mapping[str, object]:
    gates = {
        "non_decreasing_success_rate": comparison["success_rate_delta"] >= 0.0,
        "non_decreasing_success_per_km": (
            comparison["success_per_km_ratio"] is not None
            and comparison["success_per_km_ratio"] >= 1.0
        ),
        "non_increasing_replan_proxy": (
            comparison["replan_ratio"] is not None
            and comparison["replan_ratio"] <= 1.0
        ),
        "non_worsening_brier": (
            comparison["brier_ratio"] is not None
            and comparison["brier_ratio"] <= 1.0
        ),
        "improved_worst_case_success": (
            hybrid["worst_case_scenario_success_rate"]
            > improved["worst_case_scenario_success_rate"]
        ),
    }
    return {
        "decision": "PROMOTE_TO_20_SEED_ABLATION" if all(gates.values()) else "DO_NOT_PROMOTE",
        "gates": gates,
        "passed_gate_count": sum(gates.values()),
        "gate_count": len(gates),
        "interpretation": (
            "Pilot evidence is directional only; failed gates reject the current "
            "supervisor rule set but do not establish a publication result."
        ),
    }
