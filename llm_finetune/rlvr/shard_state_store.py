#!/usr/bin/env python3
"""Split a single-file RLVR state store into multiple JSONL shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, help="Dataset/state-store directory containing states.jsonl")
    parser.add_argument("--num-shards", type=int, default=10)
    parser.add_argument(
        "--remove-original",
        action="store_true",
        help="Remove states.jsonl after shards and sharded index are written successfully.",
    )
    return parser.parse_args()


def _state_id_from_record(record: dict[str, Any], fallback: str) -> str:
    for key in ("state_id", "id"):
        value = record.get(key)
        if value:
            return str(value)
    extra = record.get("extra_info")
    if isinstance(extra, dict) and extra.get("state_id"):
        return str(extra["state_id"])
    return fallback


def _load_legacy_index(index_path: Path) -> dict[str, int]:
    raw = json.loads(index_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"invalid index: {index_path}")
    legacy: dict[str, int] = {}
    for state_id, entry in raw.items():
        if isinstance(entry, dict):
            raise ValueError(f"{index_path} is already sharded; entry {state_id!r} is an object")
        legacy[str(state_id)] = int(entry)
    return legacy


def shard_state_store(input_dir: Path, *, num_shards: int, remove_original: bool = False) -> dict[str, Any]:
    if num_shards < 1:
        raise ValueError("--num-shards must be >= 1")

    states_path = input_dir / "states.jsonl"
    index_path = input_dir / "states.index.json"
    if not states_path.exists():
        raise FileNotFoundError(f"states.jsonl not found: {states_path}")
    if not index_path.exists():
        raise FileNotFoundError(f"states.index.json not found: {index_path}")

    legacy_index = _load_legacy_index(index_path)
    state_ids_by_offset = {offset: state_id for state_id, offset in legacy_index.items()}

    shard_dir = input_dir / "states"
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_paths = [shard_dir / f"states_{idx:05d}.jsonl" for idx in range(num_shards)]
    for shard_path in shard_paths:
        if shard_path.exists():
            shard_path.unlink()

    handles = [path.open("wb") for path in shard_paths]
    new_index: dict[str, dict[str, int | str]] = {}
    total = 0
    try:
        with states_path.open("rb") as source:
            while True:
                offset = source.tell()
                line = source.readline()
                if not line:
                    break
                state_id = state_ids_by_offset.get(offset)
                if state_id is None:
                    record = json.loads(line.decode("utf-8"))
                    state_id = _state_id_from_record(record, f"row_{total:08d}")
                shard_idx = total % num_shards
                shard_handle = handles[shard_idx]
                shard_offset = shard_handle.tell()
                shard_handle.write(line)
                new_index[state_id] = {
                    "file": shard_paths[shard_idx].relative_to(input_dir).as_posix(),
                    "offset": shard_offset,
                }
                total += 1
    finally:
        for handle in handles:
            handle.close()

    missing = sorted(set(legacy_index) - set(new_index))
    if missing:
        raise ValueError(f"{len(missing)} indexed states were not found in states.jsonl; first={missing[0]}")

    tmp_index_path = input_dir / "states.index.sharded.json.tmp"
    new_index_path = input_dir / "states.index.json"
    tmp_index_path.write_text(json.dumps(new_index, ensure_ascii=False), encoding="utf-8")
    tmp_index_path.replace(new_index_path)

    manifest = {
        "input_dir": str(input_dir),
        "source_states_jsonl": str(states_path),
        "num_shards": num_shards,
        "total_states": total,
        "index_format": "state_id -> {file, offset}",
        "remove_original": remove_original,
        "shard_dir": str(shard_dir),
        "shards": [path.name for path in shard_paths],
    }
    (input_dir / "states.shards.manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if remove_original:
        states_path.unlink()

    return manifest


def main() -> int:
    args = parse_args()
    manifest = shard_state_store(
        Path(args.input_dir),
        num_shards=args.num_shards,
        remove_original=args.remove_original,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
