#!/usr/bin/env python3
"""Run paired fixed-versus-adaptive active-search development experiments."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from modules.search_intelligence import (  # noqa: E402
    compare_paired_policy_results,
    default_stress_profiles,
    realism_stress_profiles,
    run_stress_benchmark,
    write_stress_benchmark_results,
)


def main() -> int:
    profiles = {
        item.profile_id: item
        for item in default_stress_profiles() + realism_stress_profiles()
    }
    parser = argparse.ArgumentParser(
        description="Paired comparison of active and adaptive_active policies."
    )
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=("nominal", "tight_budget", "verified_combined_realism"),
        choices=tuple(profiles),
    )
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--seed", type=int, default=3107)
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "results" / "adaptive_weights_benchmark"),
    )
    args = parser.parse_args()

    selected = tuple(profiles[profile_id] for profile_id in args.profiles)
    runs = run_stress_benchmark(
        selected,
        repetitions=args.repetitions,
        base_seed=args.seed,
        policy_names=("active", "adaptive_active"),
    )
    output = Path(args.output_dir)
    artifacts = dict(write_stress_benchmark_results(runs, str(output)))
    comparisons = []
    for run in runs:
        for prior_condition in (
            "all",
            "correct",
            "diffuse",
            "uniform",
            "misleading",
        ):
            scenarios = tuple(
                scenario for scenario in run.scenarios
                if prior_condition == "all"
                or scenario.prior_condition == prior_condition
            )
            scenario_ids = {scenario.scenario_id for scenario in scenarios}
            baseline = tuple(
                episode for episode in run.report.episodes
                if episode.policy_name == "active"
                and episode.scenario_id in scenario_ids
            )
            candidate = tuple(
                episode for episode in run.report.episodes
                if episode.policy_name == "adaptive_active"
                and episode.scenario_id in scenario_ids
            )
            metrics = compare_paired_policy_results(baseline, candidate, scenarios)
            for metric in metrics.values():
                comparisons.append({
                    "profile_id": run.profile.profile_id,
                    "prior_condition": prior_condition,
                    **metric.to_dict(),
                })

    comparison_json = output / "adaptive_paired_comparison.json"
    comparison_csv = output / "adaptive_paired_comparison.csv"
    comparison_json.write_text(json.dumps({
        "schema_version": "gsi-adaptive-weight-comparison-v1",
        "evidence_status": "development_not_held_out",
        "baseline_policy": "active",
        "candidate_policy": "adaptive_active",
        "repetitions": args.repetitions,
        "base_seed": args.seed,
        "comparisons": comparisons,
    }, indent=2, sort_keys=True), encoding="utf-8")
    with comparison_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)
    artifacts.update({
        "paired_comparison_json": str(comparison_json),
        "paired_comparison_csv": str(comparison_csv),
    })
    overall = {
        run.profile.profile_id: {
            item["metric"]: {
                "baseline_mean": item["baseline_mean"],
                "candidate_mean": item["candidate_mean"],
                "mean_improvement": item["mean_improvement"],
                "improvement_ci95": (
                    item["improvement_ci95_low"],
                    item["improvement_ci95_high"],
                ),
            }
            for item in comparisons
            if item["profile_id"] == run.profile.profile_id
            and item["prior_condition"] == "all"
        }
        for run in runs
    }
    print(json.dumps({"overall": overall, "artifacts": artifacts}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
