#!/usr/bin/env python3
"""Tune and validate the success-first high-resolution Search Skill."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from modules.search_intelligence import (  # noqa: E402
    run_success_first_experiment,
    write_success_first_experiment,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Tune on development seeds and validate on held-out seeds."
    )
    parser.add_argument("--tuning-seeds", type=int, default=5)
    parser.add_argument("--tuning-base-seed", type=int, default=5)
    parser.add_argument("--validation-seeds", type=int, default=20)
    parser.add_argument("--validation-base-seed", type=int, default=10)
    parser.add_argument(
        "--output",
        default=str(
            PROJECT_ROOT
            / "results"
            / "search_skill_success_first"
            / "experiment_report.json"
        ),
    )
    args = parser.parse_args()
    if min(args.tuning_seeds, args.validation_seeds) <= 0:
        parser.error("seed counts must be positive")
    payload = run_success_first_experiment(
        tuning_repetitions=args.tuning_seeds,
        tuning_base_seed=args.tuning_base_seed,
        validation_repetitions=args.validation_seeds,
        validation_base_seed=args.validation_base_seed,
    )
    path = write_success_first_experiment(payload, args.output)
    validation = payload["held_out_validation"]
    print(json.dumps({
        "experiment_report": path,
        "selected_variant": payload["tuning"]["selected_variant"],
        "methods": validation["methods"],
        "hardware_effect": validation[
            "hardware_effect_D_high_res_vs_D_current"
        ],
        "policy_effect": validation["policy_effect_E2_vs_D_high_res"],
        "promotion_decision": validation["promotion_decision"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
