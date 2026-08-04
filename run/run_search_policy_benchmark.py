#!/usr/bin/env python3
"""Run the standalone M6 benchmark for search policies."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from modules.search_intelligence import (  # noqa: E402
    SearchBenchmarkConfig,
    SearchBenchmarkRunner,
    default_benchmark_scenarios,
    write_benchmark_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare fixed, adaptive, and baseline search policies."
    )
    parser.add_argument(
        "--policies",
        nargs="+",
        default=("coverage", "random", "greedy_prior", "active", "adaptive_active", "lookahead_active"),
        choices=("coverage", "random", "greedy_prior", "active", "adaptive_active", "lookahead_active"),
    )
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "results" / "search_policy_benchmark"),
    )
    args = parser.parse_args()

    config = SearchBenchmarkConfig(
        policy_names=tuple(args.policies),
        repetitions=args.repetitions,
        base_seed=args.seed,
    )
    report = SearchBenchmarkRunner(config).run(default_benchmark_scenarios())
    artifacts = write_benchmark_report(report, args.output_dir)

    summary = {
        aggregate.policy_name: {
            "episodes": aggregate.episode_count,
            "success_rate": aggregate.success_rate.mean,
            "false_positive_rate": aggregate.false_positive_rate.mean,
            "mean_spl": aggregate.spl.mean,
            "mean_distance_m": aggregate.distance_travelled_m.mean,
            "successful_mean_time_s": aggregate.successful_elapsed_time_s.mean,
        }
        for aggregate in report.aggregates
    }
    print(json.dumps({
        "summary": summary,
        "artifacts": artifacts,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
