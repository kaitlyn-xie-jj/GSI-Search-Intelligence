#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/GSI}"
SNAPSHOT="replan3000_asm50_nc4_promptstate_20260522_132738"
RLVR_SNAPSHOT="${SNAPSHOT}_snapshot"
DEFAULT_MODEL_PATH="/models/Qwen3-0.6B"
DEFAULT_SFT_DATA_PATH="${ROOT_DIR}/data/sft/replan3000_asm50_nc4_promptstate_20260522_132738_sft_curated_all_snapshot.jsonl"
DEFAULT_RLVR_DATA_DIR="${ROOT_DIR}/data/rlvr_gsi/replan3000_asm50_nc4_promptstate_20260522_132738_snapshot"

cd "${ROOT_DIR}"

export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/llm_finetune/verl_scripts/verl:${PYTHONPATH:-}"

MODEL_PATH="${MODEL_PATH:-${SFT_MODEL_PATH:-${RLVR_MODEL_PATH:-${DEFAULT_MODEL_PATH}}}}"
SFT_DATA_PATH="${SFT_DATA_PATH:-${DEFAULT_SFT_DATA_PATH}}"
RLVR_DATA_DIR="${RLVR_DATA_DIR:-${DEFAULT_RLVR_DATA_DIR}}"

require_file() {
  if [ ! -f "$1" ]; then
    echo "Missing file: $1" >&2
    exit 1
  fi
}

require_dir() {
  if [ ! -d "$1" ]; then
    echo "Missing directory: $1" >&2
    exit 1
  fi
}

require_file "${ROOT_DIR}/llm_finetune/scripts/runtime/internal/run_sft.sh"
require_file "${ROOT_DIR}/llm_finetune/scripts/runtime/internal/run_gsi_rlvr.sh"
require_file "${ROOT_DIR}/llm_finetune/scripts/runtime/internal/serve_gsi_validator.sh"
require_file "${ROOT_DIR}/llm_finetune/rlvr/gsi_reward_manager.py"
require_file "${ROOT_DIR}/llm_finetune/verl_scripts/verl/verl/trainer/main_ppo.py"

if [ "${GSI_REQUIRE_DATA:-0}" = "1" ]; then
  require_file "${SFT_DATA_PATH}"
  require_file "${RLVR_DATA_DIR}/train.parquet"
  require_file "${RLVR_DATA_DIR}/val.parquet"
  require_file "${RLVR_DATA_DIR}/states.index.json"
  if [ ! -f "${RLVR_DATA_DIR}/states.jsonl" ] && [ ! -d "${RLVR_DATA_DIR}/states" ]; then
    echo "Missing state records: expected ${RLVR_DATA_DIR}/states.jsonl or ${RLVR_DATA_DIR}/states/" >&2
    exit 1
  fi
fi

if [ "${GSI_REQUIRE_MODEL:-0}" = "1" ]; then
  require_dir "${MODEL_PATH}"
fi

python - <<'PY'
import importlib
modules = [
    "torch",
    "transformers",
    "datasets",
    "peft",
    "trl",
    "verl",
    "vllm",
    "ray",
    "huggingface_hub",
    "safetensors",
    "fastapi",
    "uvicorn",
    "pydantic",
    "aiohttp",
    "openai",
    "httpx",
    "jiter",
    "requests",
    "tenacity",
    "pandas",
    "pyarrow",
    "shapely",
    "pyscipopt",
    "pulp",
    "gurobipy",
    "tensorboard",
]

for module_name in modules:
    module = importlib.import_module(module_name)
    version = getattr(module, "__version__", "unknown")
    print(f"{module_name} OK {version}")

runtime_entrypoints = [
    "run.run_exp_multi_method",
    "run.run_collect_replan_dataset",
    "run.plan_validation_server",
    "modules.task_solver.baseline_planners.lipllm.task_allocator",
]

for module_name in runtime_entrypoints:
    importlib.import_module(module_name)
    print(f"{module_name} import OK")

import torch

if torch.cuda.is_available():
    module = importlib.import_module("unsloth")
    print(f"unsloth OK {getattr(module, '__version__', 'unknown')}")
else:
    print("unsloth SKIP no CUDA accelerator visible")

PY

echo "GSI runtime/training environment OK"
