#!/usr/bin/env bash

resolve_hf_model_path() {
  local explicit_path="${1:-}"
  local repo_id="${2:-}"
  local label="${3:-model}"

  if [[ -n "$explicit_path" ]]; then
    printf '%s\n' "$explicit_path"
    return 0
  fi
  if [[ -z "$repo_id" ]]; then
    echo "[ERROR] No ${label} path or repo id was provided." >&2
    return 1
  fi

  python - "$repo_id" "$label" <<'PY'
import os
import sys
from huggingface_hub import snapshot_download

repo_id, label = sys.argv[1], sys.argv[2]
local_files_only = os.environ.get("HF_HUB_OFFLINE") == "1" or os.environ.get("TRANSFORMERS_OFFLINE") == "1"
try:
    print(snapshot_download(repo_id=repo_id, local_files_only=local_files_only))
except Exception as exc:
    mode = "cached " if local_files_only else ""
    raise SystemExit(f"[ERROR] Could not resolve {mode}{label} repo {repo_id!r}: {exc}")
PY
}

resolve_sft_data_path() {
  local explicit_path="${1:-}"
  local repo_id="${2:-}"
  local filename="${3:-train.jsonl}"

  if [[ -n "$explicit_path" ]]; then
    printf '%s\n' "$explicit_path"
    return 0
  fi
  if [[ -z "$repo_id" ]]; then
    echo "[ERROR] No SFT data path or dataset repo id was provided." >&2
    return 1
  fi

  python - "$repo_id" "$filename" <<'PY'
import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

repo_id, filename = sys.argv[1], sys.argv[2]
local_files_only = os.environ.get("HF_HUB_OFFLINE") == "1" or os.environ.get("HF_DATASETS_OFFLINE") == "1"
try:
    snapshot = Path(snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_files_only=local_files_only,
        allow_patterns=[filename, "*.jsonl", "*.json"],
    ))
except Exception as exc:
    mode = "cached " if local_files_only else ""
    raise SystemExit(f"[ERROR] Could not resolve {mode}SFT dataset repo {repo_id!r}: {exc}")

candidate = snapshot / filename
if not candidate.exists():
    jsonl_files = sorted(snapshot.glob("*.jsonl"))
    if len(jsonl_files) == 1:
        candidate = jsonl_files[0]
if not candidate.exists():
    mode = "cached " if local_files_only else ""
    raise SystemExit(f"[ERROR] SFT data file not found in {mode}repo {repo_id!r}: {filename}")
print(candidate)
PY
}

resolve_rlvr_data_dir() {
  local explicit_dir="${1:-}"
  local repo_id="${2:-}"
  local state_store_name="${3:-}"

  if [[ -n "$explicit_dir" ]]; then
    printf '%s\n' "$explicit_dir"
    return 0
  fi
  if [[ -z "$repo_id" ]]; then
    echo "[ERROR] No RLVR data dir or dataset repo id was provided." >&2
    return 1
  fi

  python - "$repo_id" "$state_store_name" <<'PY'
import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

repo_id, state_store_name = sys.argv[1], sys.argv[2]
local_files_only = os.environ.get("HF_HUB_OFFLINE") == "1" or os.environ.get("HF_DATASETS_OFFLINE") == "1"
try:
    snapshot = Path(snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_files_only=local_files_only,
        allow_patterns=[
            "train.parquet",
            "val.parquet",
            "manifest.json",
            "states.index.json",
            "states.shards.manifest.json",
            "states/*.jsonl",
        ],
    ))
except Exception as exc:
    mode = "cached " if local_files_only else ""
    raise SystemExit(f"[ERROR] Could not resolve {mode}RLVR dataset repo {repo_id!r}: {exc}")

required = ["train.parquet", "val.parquet", "states.index.json"]
missing = [name for name in required if not (snapshot / name).exists()]
if missing:
    mode = "Cached " if local_files_only else ""
    raise SystemExit(f"[ERROR] {mode}RLVR dataset repo {repo_id!r} is missing: {missing}")
if not (snapshot / "states").is_dir() and not (snapshot / "states.jsonl").exists():
    mode = "Cached " if local_files_only else ""
    raise SystemExit(f"[ERROR] {mode}RLVR dataset repo {repo_id!r} has no states/ shards or states.jsonl")

print(snapshot)
PY
}
