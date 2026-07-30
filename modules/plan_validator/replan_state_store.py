"""
Local replan state store reader.

Business semantics:
- RLVR reward only passes `state_store/state_id` via HTTP, not full world_state.
- Validator worker uses index seek to corresponding JSONL line locally, reads world_state/event/meta/case to initialize environment.

Current limitations:
- State store defaults to `data/rlvr_gsi/<state_store>` in current GSI repo.
- Can override state root with `GSI_REPLAN_STATE_ROOT`.
- `GSI_REPLAN_STATE_ROOT` can point directly to a state-store/snapshot directory containing `states.index.json`, which supports loading directly from a Hugging Face dataset snapshot.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, NamedTuple


class StateIndexEntry(NamedTuple):
    path: Path
    offset: int


def _default_state_root() -> Path:
    env = os.environ.get("GSI_REPLAN_STATE_ROOT", "").strip()
    if env:
        return Path(env)
    # modules/plan_validator/replan_state_store.py -> repo root
    return Path(__file__).resolve().parents[2] / "data" / "rlvr_gsi"


def _resolve_store_dir(state_store: str) -> Path:
    root = _default_state_root()
    direct_index = root / "states.index.json"
    if direct_index.exists():
        return root
    return root / state_store


@lru_cache(maxsize=16)
def _load_index(state_store: str) -> dict[str, StateIndexEntry]:
    store_dir = _resolve_store_dir(state_store)
    states_path = store_dir / "states.jsonl"
    index_path = store_dir / "states.index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"states.index.json not found for state_store={state_store}: {index_path}")
    raw_index = json.loads(index_path.read_text(encoding="utf-8"))
    if not isinstance(raw_index, dict):
        raise ValueError(f"invalid states.index.json for state_store={state_store}: expected object")

    parsed: dict[str, StateIndexEntry] = {}
    for state_id, raw_entry in raw_index.items():
        if isinstance(raw_entry, dict):
            shard_file = str(raw_entry.get("file") or "").strip()
            if not shard_file:
                raise ValueError(f"invalid sharded index entry for {state_store}/{state_id}: missing file")
            offset = int(raw_entry["offset"])
            state_path = store_dir / shard_file
        else:
            if not states_path.exists():
                raise FileNotFoundError(f"states.jsonl not found for state_store={state_store}: {states_path}")
            offset = int(raw_entry)
            state_path = states_path
        if not state_path.exists():
            raise FileNotFoundError(f"state shard not found for state_store={state_store}: {state_path}")
        parsed[str(state_id)] = StateIndexEntry(path=state_path, offset=offset)
    return parsed


@lru_cache(maxsize=4096)
def _load_state_record(state_store: str, state_id: str) -> dict[str, Any]:
    """
    Read and cache parsed state record.

    Caller should not modify returned object in-place; copy necessary fields when modification needed.
    """

    index = _load_index(str(state_store))
    entry = index.get(str(state_id))
    if entry is None:
        raise KeyError(f"state_id not found in {state_store}: {state_id}")
    with entry.path.open("rb") as handle:
        handle.seek(entry.offset)
        line = handle.readline()
    if not line:
        raise ValueError(f"empty state record for {state_store}/{state_id}")
    return json.loads(line.decode("utf-8"))


def load_replan_state(state_store: str | None, state_id: str | None) -> dict[str, Any] | None:
    """
    Read full replan state by `state_store/state_id`.

    Returns None if no replan state requested, caller should use first-plan original scenario path.
    """

    if not state_store or not state_id:
        return None
    return _load_state_record(str(state_store), str(state_id))


def get_replan_state_cache_info() -> dict[str, Any]:
    """Return state store index/record cache metrics for validator stats exposure."""

    return {
        "index": _load_index.cache_info()._asdict(),
        "record": _load_state_record.cache_info()._asdict(),
    }
