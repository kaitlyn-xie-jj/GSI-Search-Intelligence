#!/usr/bin/env python3
"""Run the parameterized search-policy stress benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from modules.search_intelligence import (  # noqa: E402
    default_stress_profiles,
    run_stress_benchmark,
    verification_stress_profiles,
    write_stress_benchmark_results,
)


def main() -> int:
    default_profiles = default_stress_profiles()
    available_profiles = {
        profile.profile_id: profile
        for profile in default_profiles + verification_stress_profiles()
    }
    parser = argparse.ArgumentParser(
        description=(
            "Stress-test search policies across maps, target positions, priors, "
            "sensor conditions, and resource budgets."
        )
    )
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=tuple(profile.profile_id for profile in default_profiles),
        choices=tuple(available_profiles),
    )
    parser.add_argument(
        "--policies",
        nargs="+",
        default=("coverage", "random", "greedy_prior", "active"),
        choices=("coverage", "random", "greedy_prior", "active"),
    )
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "results" / "search_stress_benchmark"),
    )
    args = parser.parse_args()

    selected_profiles = tuple(
        available_profiles[profile_id] for profile_id in args.profiles
    )
    runs = run_stress_benchmark(
        selected_profiles,
        repetitions=args.repetitions,
        base_seed=args.seed,
        policy_names=tuple(args.policies),
    )
    artifacts = write_stress_benchmark_results(runs, args.output_dir)
    payload = {
        "profiles": [profile.profile_id for profile in selected_profiles],
        "scenario_count_per_profile": len(runs[0].scenarios),
        "episode_count": sum(len(run.report.episodes) for run in runs),
        "artifacts": artifacts,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
