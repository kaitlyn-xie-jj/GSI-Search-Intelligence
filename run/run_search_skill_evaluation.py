#!/usr/bin/env python3
"""Run the unified A/B/C/D Search Skill acceptance benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from modules.search_intelligence import (  # noqa: E402
    run_unified_benchmark,
    write_unified_evaluation_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the unified Search Skill acceptance benchmark."
    )
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument(
        "--output",
        default=str(
            PROJECT_ROOT
            / "results"
            / "search_skill_acceptance"
            / "evaluation_report.json"
        ),
    )
    args = parser.parse_args()
    if args.seeds < 20:
        parser.error("--seeds must be at least 20 for acceptance evaluation")
    payload = run_unified_benchmark(
        repetitions=args.seeds,
        base_seed=args.base_seed,
    )
    path = write_unified_evaluation_report(payload, args.output)
    print(json.dumps({
        "evaluation_report": path,
        "comparison_against_coverage": payload["comparison_against_coverage"],
        "policy_results": payload["policy_results"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
