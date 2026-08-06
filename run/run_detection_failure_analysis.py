#!/usr/bin/env python3
"""Generate the trace-based high-resolution detection failure taxonomy."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from modules.search_intelligence import (  # noqa: E402
    analyze_detection_failures,
    write_detection_failure_analysis,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze frozen D-high-res traces.")
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--base-seed", type=int, default=10)
    parser.add_argument(
        "--output",
        default=str(
            PROJECT_ROOT
            / "results"
            / "search_skill_success_first"
            / "failure_analysis.json"
        ),
    )
    args = parser.parse_args()
    if args.seeds <= 0:
        parser.error("--seeds must be positive")
    payload = analyze_detection_failures(
        repetitions=args.seeds,
        base_seed=args.base_seed,
    )
    path = write_detection_failure_analysis(payload, args.output)
    print(json.dumps({
        "failure_analysis": path,
        "success_rate": payload["success_rate"],
        "probability_decomposition": payload["probability_decomposition"],
        "failure_summary": payload["failure_summary"],
        "path_to_70_percent": payload["path_to_70_percent"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
