#!/usr/bin/env python3
"""Calibrate active-search utility weights on disjoint scenario splits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from modules.search_intelligence import (  # noqa: E402
    OfflineOptimizationConfig,
    OfflineUtilityOptimizer,
    write_offline_optimization_result,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Tune ActiveSearch utility weights on train/validation layouts and "
            "evaluate the selected weights once on a held-out layout."
        )
    )
    parser.add_argument("--candidates", type=int, default=64)
    parser.add_argument("--validation-candidates", type=int, default=8)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "results" / "search_offline_optimization"),
    )
    args = parser.parse_args()

    config = OfflineOptimizationConfig(
        candidate_count=args.candidates,
        validation_candidate_count=args.validation_candidates,
        repetitions=args.repetitions,
        base_seed=args.seed,
    )
    result = OfflineUtilityOptimizer(config).run()
    artifacts = write_offline_optimization_result(result, args.output_dir)
    print(json.dumps({
        "selected_candidate_id": result.selected_candidate_id,
        "selected_weights": result.selected_weights.to_dict(),
        "selected_scores": {
            split: score.to_dict()
            for split, score in result.selected_scores.items()
        },
        "default_scores": {
            split: score.to_dict()
            for split, score in result.default_scores.items()
        },
        "artifacts": artifacts,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
