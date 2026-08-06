#!/usr/bin/env python3
"""Run the paired D/E hybrid supervisor pilot experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from modules.search_intelligence import (  # noqa: E402
    run_hybrid_search_experiment,
    write_hybrid_search_experiment,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a paired improved-active versus hybrid pilot."
    )
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument(
        "--output",
        default=str(
            PROJECT_ROOT
            / "results"
            / "search_skill_hybrid_pilot"
            / "experiment_report.json"
        ),
    )
    args = parser.parse_args()
    if args.seeds <= 0:
        parser.error("--seeds must be positive")
    payload = run_hybrid_search_experiment(
        repetitions=args.seeds,
        base_seed=args.base_seed,
    )
    output = write_hybrid_search_experiment(payload, args.output)
    print(json.dumps({
        "experiment_report": output,
        "pilot": payload["configuration"]["pilot"],
        "policy_results": payload["policy_results"],
        "paired_comparison": payload["paired_comparison"],
        "hybrid_mode_usage": payload["hybrid_mode_usage"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
