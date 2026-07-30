#!/usr/bin/env python3
"""Convert GSI replan collection JSONL into VeRL parquet plus local state store."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import pandas as pd


PARQUET_COLUMNS = ["prompt", "data_source", "ability", "reward_model", "extra_info"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build replan state store and VeRL parquet files.")
    parser.add_argument("--input", required=True, help="Source replan JSONL.")
    parser.add_argument("--output-dir", required=True, help="Output RLVR dataset directory.")
    parser.add_argument("--state-store", default="", help="State store name. Defaults to output directory name.")
    parser.add_argument("--seed", type=int, default=20260504)
    parser.add_argument("--train-ratio", type=float, default=0.90)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    return parser.parse_args()


def _task_id_from_case(case: dict[str, Any]) -> str:
    scenario_id = str(case.get("scenario_id") or "scenario_1")
    goal_id = str(case.get("goal_id") or "")
    return f"{scenario_id}_{goal_id}" if goal_id else scenario_id


def _extra_info(record: dict[str, Any], *, state_id: str, state_store: str, split: str) -> dict[str, Any]:
    case = record.get("case") or {}
    event = record.get("event") or {}
    details = event.get("details") or {}
    return {
        "task_id": _task_id_from_case(case),
        "state_store": state_store,
        "state_id": state_id,
        "data_tag": state_store,
        "split": split,
        "goal_type": case.get("goal_type", ""),
        "type_name": case.get("type_name", "cybertown"),
        "scenario_id": case.get("scenario_id", ""),
        "goal_id": case.get("goal_id", ""),
        "event_phase": event.get("phase", ""),
        "event_kind": event.get("event_kind", ""),
        "event_reason": event.get("reason", ""),
        "event_category": details.get("category", ""),
        "event_type": details.get("type", ""),
        "event_severity": details.get("severity", ""),
    }


def build_state_store(
    *,
    input_path: Path,
    output_dir: Path,
    state_store: str,
    seed: int,
    train_ratio: float,
    val_ratio: float,
) -> dict[str, Any]:
    if not 0.0 <= train_ratio <= 1.0:
        raise ValueError(f"train_ratio must be between 0 and 1, got {train_ratio}")
    if not 0.0 <= val_ratio <= 1.0:
        raise ValueError(f"val_ratio must be between 0 and 1, got {val_ratio}")
    if train_ratio + val_ratio > 1.0:
        raise ValueError("train_ratio + val_ratio must be <= 1")

    output_dir.mkdir(parents=True, exist_ok=True)
    state_store = state_store or output_dir.name

    records: list[dict[str, Any]] = []
    index: dict[str, int] = {}
    states_path = output_dir / "states.jsonl"
    with input_path.open("r", encoding="utf-8") as src, states_path.open("wb") as dst:
        for i, line in enumerate(src):
            if not line.strip():
                continue
            record = json.loads(line)
            state_id = f"{state_store}.{i:06d}"
            record["state_id"] = state_id
            offset = dst.tell()
            dst.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n")
            index[state_id] = offset
            records.append({"state_id": state_id, "record": record})

    rng = random.Random(seed)
    rng.shuffle(records)
    total = len(records)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    split_records = {
        "train": records[:train_end],
        "val": records[train_end:val_end],
        "test": records[val_end:],
    }

    for split, items in split_records.items():
        rows = []
        for item in items:
            record = item["record"]
            rows.append(
                {
                    "prompt": [{"role": "user", "content": str(record.get("prompt") or "")}],
                    "data_source": "cybertown",
                    "ability": "planning",
                    "reward_model": {"style": "custom", "ground_truth": ""},
                    "extra_info": _extra_info(record, state_id=item["state_id"], state_store=state_store, split=split),
                }
            )
        pd.DataFrame(rows, columns=PARQUET_COLUMNS).to_parquet(output_dir / f"{split}.parquet", index=False)

    (output_dir / "states.index.json").write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")
    manifest = {
        "source": str(input_path),
        "state_store": state_store,
        "states_path": str(states_path),
        "index_path": str(output_dir / "states.index.json"),
        "total": total,
        "splits": {key: len(value) for key, value in split_records.items()},
        "schema": PARQUET_COLUMNS,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> int:
    args = parse_args()
    manifest = build_state_store(
        input_path=Path(args.input),
        output_dir=Path(args.output_dir),
        state_store=args.state_store,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
