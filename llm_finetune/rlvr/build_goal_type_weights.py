#!/usr/bin/env python3
"""Convert goal_type success rates into sampling weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm_finetune.rlvr.goal_type_sampling import build_difficulty_weights_from_success_rates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="JSON file containing goal_type -> success_rate percent")
    parser.add_argument("--output", required=True, help="Output JSON file containing goal_type -> weight")
    parser.add_argument("--curve", choices=["linear", "power", "logit", "exp"], default="power")
    parser.add_argument("--gamma", type=float, default=2.0, help="Used by power curve.")
    parser.add_argument("--temperature", type=float, default=2.0, help="Used by exp curve.")
    parser.add_argument("--epsilon", type=float, default=1e-3, help="Clamp to avoid infinities.")
    parser.add_argument("--scale", type=float, default=100.0, help="Multiply final weights by this constant.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    success_rates = json.loads(Path(args.input).read_text(encoding="utf-8"))
    if not isinstance(success_rates, dict):
        raise ValueError("input must be a JSON object")
    weights = build_difficulty_weights_from_success_rates(
        success_rates,
        curve=args.curve,
        gamma=args.gamma,
        temperature=args.temperature,
        epsilon=args.epsilon,
        scale=args.scale,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(weights, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(weights, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
