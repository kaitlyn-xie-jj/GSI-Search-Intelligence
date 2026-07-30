#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/GSI}"
DEFAULT_RLVR_MODEL_REPO_ID="WindyLab/Qwen3-0.6B-cybertown-SFT"
DEFAULT_RLVR_DATA_REPO_ID="WindyLab/Qwen3-0.6B-cybertown-RLVR-data"
DEFAULT_RLVR_STATE_STORE_NAME="train6_val10_initial30_replan70_val15_20260528"
RUN_RLVR_SCRIPT="${RUN_RLVR_SCRIPT:-${ROOT_DIR}/llm_finetune/scripts/runtime/internal/run_gsi_rlvr.sh}"
source "${ROOT_DIR}/llm_finetune/scripts/runtime/hf_cache_utils.sh"

cd "${ROOT_DIR}"

export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/llm_finetune/verl_scripts/verl:${PYTHONPATH:-}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-lo}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"
export NO_PROXY="127.0.0.1,localhost,${NO_PROXY:-}"
export no_proxy="127.0.0.1,localhost,${no_proxy:-}"

RLVR_DATA_DIR_INPUT="${RLVR_DATA_DIR:-}"
RLVR_DATA_REPO_ID="${RLVR_DATA_REPO_ID:-}"
if [[ -z "${RLVR_DATA_DIR_INPUT}" && -z "${RLVR_DATA_REPO_ID}" ]]; then
  RLVR_DATA_REPO_ID="${DEFAULT_RLVR_DATA_REPO_ID}"
fi
RLVR_STATE_STORE_NAME="${RLVR_STATE_STORE_NAME:-${DEFAULT_RLVR_STATE_STORE_NAME}}"
RLVR_DATA_DIR="$(resolve_rlvr_data_dir "${RLVR_DATA_DIR_INPUT}" "${RLVR_DATA_REPO_ID}" "${RLVR_STATE_STORE_NAME}")"
export RLVR_DATA_DIR

if [ -z "${GSI_REPLAN_STATE_ROOT:-}" ]; then
  GSI_REPLAN_STATE_ROOT="${RLVR_DATA_DIR}"
fi
export GSI_REPLAN_STATE_ROOT

RLVR_MODEL_PATH_INPUT="${RLVR_MODEL_PATH:-}"
RLVR_MODEL_REPO_ID="${RLVR_MODEL_REPO_ID:-}"
if [[ -z "${RLVR_MODEL_PATH_INPUT}" && -z "${RLVR_MODEL_REPO_ID}" ]]; then
  RLVR_MODEL_REPO_ID="${DEFAULT_RLVR_MODEL_REPO_ID}"
fi
RLVR_MODEL_PATH="$(resolve_hf_model_path "${RLVR_MODEL_PATH_INPUT}" "${RLVR_MODEL_REPO_ID}" "RLVR base model")"
export RLVR_MODEL_PATH
export RLVR_OUTPUT_DIR="${RLVR_OUTPUT_DIR:-${ROOT_DIR}/outputs/rlvr_gsi/qwen3_0_6b_cybertown_rlvr}"

TRAIN_CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-all}}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-6144}"
VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-12288}"

export GSI_VALIDATOR_HOST="${GSI_VALIDATOR_HOST:-127.0.0.1}"
export GSI_VALIDATOR_PORT="${GSI_VALIDATOR_PORT:-8000}"
export GSI_VALIDATOR_URL="${GSI_VALIDATOR_URL:-http://${GSI_VALIDATOR_HOST}:${GSI_VALIDATOR_PORT}/validate}"
export GSI_VALIDATOR_BATCH_URL="${GSI_VALIDATOR_BATCH_URL:-http://${GSI_VALIDATOR_HOST}:${GSI_VALIDATOR_PORT}/validate_batch}"
export GSI_VALIDATOR_MAX_WORKERS="${GSI_VALIDATOR_MAX_WORKERS:-4}"
export GSI_VALIDATOR_BATCH_CHUNK_SIZE="${GSI_VALIDATOR_BATCH_CHUNK_SIZE:-4}"
export GSI_VALIDATOR_TIMEOUT="${GSI_VALIDATOR_TIMEOUT:-180}"
export GSI_REWARD_HTTP_TIMEOUT="${GSI_REWARD_HTTP_TIMEOUT:-240}"
export GSI_VALIDATOR_TIMEOUT_REWARD="${GSI_VALIDATOR_TIMEOUT_REWARD:--10}"

GSI_PREFLIGHT="${GSI_PREFLIGHT:-0}"
GSI_PREFLIGHT_CLEAN_VLLM="${GSI_PREFLIGHT_CLEAN_VLLM:-1}"
GSI_PREFLIGHT_STOP_VALIDATOR="${GSI_PREFLIGHT_STOP_VALIDATOR:-1}"
GSI_PREFLIGHT_STOP_RAY="${GSI_PREFLIGHT_STOP_RAY:-0}"
GSI_PREFLIGHT_MIN_GPU_FREE_MB="${GSI_PREFLIGHT_MIN_GPU_FREE_MB:-20000}"
GSI_PREFLIGHT_MIN_OUTPUT_FREE_GB="${GSI_PREFLIGHT_MIN_OUTPUT_FREE_GB:-100}"

require_file() {
  local path="$1"
  local label="$2"
  if [[ ! -f "$path" ]]; then
    echo "[ERROR] ${label} not found: ${path}" >&2
    exit 1
  fi
}

command_exists() {
  command -v "$1" >/dev/null 2>&1
}

count_selected_gpus() {
  local selected="$1"
  if [[ -z "$selected" || "$selected" == "all" ]]; then
    nvidia-smi -L | wc -l
    return 0
  fi

  local count=0
  local gpu
  IFS=',' read -r -a gpus <<< "$selected"
  for gpu in "${gpus[@]}"; do
    gpu="${gpu//[[:space:]]/}"
    if [[ -n "$gpu" ]]; then
      count=$((count + 1))
    fi
  done
  echo "$count"
}

gpu_is_selected() {
  local gpu_idx="$1"
  local selected="$2"
  if [[ -z "$selected" || "$selected" == "all" ]]; then
    return 0
  fi

  local gpu
  IFS=',' read -r -a gpus <<< "$selected"
  for gpu in "${gpus[@]}"; do
    gpu="${gpu//[[:space:]]/}"
    if [[ "$gpu" == "$gpu_idx" ]]; then
      return 0
    fi
  done
  return 1
}

cleanup_before_train() {
  if [[ "$GSI_PREFLIGHT_CLEAN_VLLM" == "1" ]]; then
    echo "[PREFLIGHT] Stopping stale vLLM API servers."
    pkill -TERM -f 'vllm.entrypoints.openai.api_server|vllm.*api_server' || true
    sleep 3
    pkill -KILL -f 'vllm.entrypoints.openai.api_server|vllm.*api_server' || true
  fi

  if [[ "$GSI_PREFLIGHT_STOP_VALIDATOR" == "1" ]]; then
    echo "[PREFLIGHT] Stopping stale GSI validator so train_rlvr.sh restarts it with this environment."
    ROOT_DIR="$ROOT_DIR" \
    GSI_REPLAN_STATE_ROOT="$GSI_REPLAN_STATE_ROOT" \
    GSI_VALIDATOR_PORT="${GSI_VALIDATOR_PORT:-8000}" \
    "${ROOT_DIR}/llm_finetune/scripts/runtime/serve_validator.sh" stop || true
  fi

  if [[ "$GSI_PREFLIGHT_STOP_RAY" == "1" ]]; then
    echo "[PREFLIGHT] Stopping existing Ray runtime."
    if command_exists ray; then
      if command_exists timeout; then
        timeout 60 ray stop --force || true
      else
        ray stop --force || true
      fi
    else
      echo "[PREFLIGHT] ray command not found; skipping Ray cleanup."
    fi
  fi
}

check_gpu_free_memory() {
  if ! command_exists nvidia-smi; then
    echo "[ERROR] nvidia-smi not found; cannot verify GPU availability." >&2
    exit 1
  fi

  local selected_count
  local expected_count
  selected_count="$(count_selected_gpus "$TRAIN_CUDA_VISIBLE_DEVICES")"
  expected_count="${VERL_N_GPUS_PER_NODE:-4}"
  if (( selected_count < expected_count )); then
    echo "[ERROR] TRAIN_CUDA_VISIBLE_DEVICES=${TRAIN_CUDA_VISIBLE_DEVICES} exposes ${selected_count} GPU(s), but VERL_N_GPUS_PER_NODE=${expected_count}." >&2
    exit 1
  fi

  local found=0
  local bad=0
  local idx free_mb total_mb
  echo "[PREFLIGHT] GPU memory:"
  while IFS=',' read -r idx free_mb total_mb; do
    idx="${idx//[[:space:]]/}"
    free_mb="${free_mb//[[:space:]]/}"
    total_mb="${total_mb//[[:space:]]/}"
    if ! gpu_is_selected "$idx" "$TRAIN_CUDA_VISIBLE_DEVICES"; then
      continue
    fi
    found=$((found + 1))
    echo "  gpu=${idx} free=${free_mb}MiB total=${total_mb}MiB"
    if (( free_mb < GSI_PREFLIGHT_MIN_GPU_FREE_MB )); then
      bad=1
    fi
  done < <(nvidia-smi --query-gpu=index,memory.free,memory.total --format=csv,noheader,nounits)

  if (( found == 0 )); then
    echo "[ERROR] No selected GPUs found in nvidia-smi output: TRAIN_CUDA_VISIBLE_DEVICES=${TRAIN_CUDA_VISIBLE_DEVICES}" >&2
    exit 1
  fi
  if (( bad != 0 )); then
    echo "[ERROR] At least one selected GPU has less than ${GSI_PREFLIGHT_MIN_GPU_FREE_MB}MiB free. Set GSI_PREFLIGHT_MIN_GPU_FREE_MB to override." >&2
    exit 1
  fi
}

check_output_disk_space() {
  mkdir -p "$RLVR_OUTPUT_DIR"

  local free_gb
  free_gb="$(df -Pk "$RLVR_OUTPUT_DIR" | awk 'NR == 2 {printf "%d", $4 / 1024 / 1024}')"
  if [[ -z "$free_gb" ]]; then
    echo "[ERROR] Could not check free disk space for ${RLVR_OUTPUT_DIR}" >&2
    exit 1
  fi

  echo "[PREFLIGHT] Output filesystem free space: ${free_gb}GiB at ${RLVR_OUTPUT_DIR}"
  if (( free_gb < GSI_PREFLIGHT_MIN_OUTPUT_FREE_GB )); then
    echo "[ERROR] Output filesystem has ${free_gb}GiB free, below GSI_PREFLIGHT_MIN_OUTPUT_FREE_GB=${GSI_PREFLIGHT_MIN_OUTPUT_FREE_GB}." >&2
    exit 1
  fi
}

check_state_records() {
  if [[ ! -s "${RLVR_DATA_DIR}/states.index.json" ]]; then
    echo "[ERROR] RLVR states.index.json is empty: ${RLVR_DATA_DIR}/states.index.json" >&2
    exit 1
  fi
  if [[ -d "${RLVR_DATA_DIR}/states" ]]; then
    local shard_count
    shard_count="$(find -L "${RLVR_DATA_DIR}/states" -type f -name '*.jsonl' | wc -l)"
    if (( shard_count == 0 )); then
      echo "[ERROR] RLVR states directory has no jsonl shards: ${RLVR_DATA_DIR}/states" >&2
      exit 1
    fi
    echo "[PREFLIGHT] RLVR sharded state records: ${shard_count} shard(s)."
  elif [[ -s "${RLVR_DATA_DIR}/states.jsonl" ]]; then
    echo "[PREFLIGHT] RLVR legacy state records: ${RLVR_DATA_DIR}/states.jsonl"
  else
    echo "[ERROR] RLVR state records are empty or missing under ${RLVR_DATA_DIR}" >&2
    exit 1
  fi
}

check_replan_state_load() {
  PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/llm_finetune/verl_scripts/verl:${PYTHONPATH:-}" \
  GSI_REPLAN_STATE_ROOT="$GSI_REPLAN_STATE_ROOT" \
  RLVR_DATA_DIR="$RLVR_DATA_DIR" \
  python - <<'PY'
import os
import sys

import pandas as pd

from modules.plan_validator.replan_state_store import load_replan_state

data_dir = os.environ["RLVR_DATA_DIR"]
for split_name in ("train", "val"):
    parquet_path = os.path.join(data_dir, f"{split_name}.parquet")
    df = pd.read_parquet(parquet_path, columns=["extra_info"])
    for extra_info in df["extra_info"]:
        if not isinstance(extra_info, dict):
            continue
        if extra_info.get("rlvr_source") != "replan":
            continue
        state_store = extra_info.get("state_store")
        state_id = extra_info.get("state_id")
        if not state_store or not state_id:
            print(f"[ERROR] {split_name} replan row is missing state_store/state_id", file=sys.stderr)
            sys.exit(1)
        record = load_replan_state(state_store, state_id)
        if not record:
            print(f"[ERROR] empty state record for {state_store}/{state_id}", file=sys.stderr)
            sys.exit(1)
        print(f"[PREFLIGHT] Sample replan state load ok: split={split_name} state_store={state_store} state_id={state_id}")
        sys.exit(0)

print("[PREFLIGHT] No replan rows found; skipped sample state load.")
PY
}

run_preflight() {
  if [[ "$GSI_PREFLIGHT" != "1" ]]; then
    echo "[PREFLIGHT] Skipped because GSI_PREFLIGHT=${GSI_PREFLIGHT}."
    return 0
  fi

  echo "[PREFLIGHT] Starting RLVR launch checks."
  cleanup_before_train
  check_gpu_free_memory
  check_output_disk_space
  check_state_records
  check_replan_state_load
  echo "[PREFLIGHT] Checks passed."
}

if [ ! -d "${RLVR_MODEL_PATH}" ]; then
  echo "RLVR_MODEL_PATH does not exist: ${RLVR_MODEL_PATH}" >&2
  exit 1
fi
require_file "${RLVR_DATA_DIR}/train.parquet" "RLVR train parquet"
require_file "${RLVR_DATA_DIR}/val.parquet" "RLVR val parquet"
require_file "${RLVR_DATA_DIR}/states.index.json" "RLVR states.index.json"
if [ ! -f "${RLVR_DATA_DIR}/states.jsonl" ] && [ ! -d "${RLVR_DATA_DIR}/states" ]; then
  echo "RLVR state records not found: expected ${RLVR_DATA_DIR}/states.jsonl or ${RLVR_DATA_DIR}/states/" >&2
  exit 1
fi

mkdir -p "${RLVR_OUTPUT_DIR}"
run_preflight

export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
if [[ "${TRAIN_CUDA_VISIBLE_DEVICES}" != "all" ]]; then
  export CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES}"
fi
export GSI_TANGO_SOLVER_BACKEND="${GSI_TANGO_SOLVER_BACKEND:-scip}"
export GSI_DISABLE_TOKEN_STATS="${GSI_DISABLE_TOKEN_STATS:-1}"
export VERL_N_GPUS_PER_NODE="${VERL_N_GPUS_PER_NODE:-4}"
export VERL_TRAIN_BATCH_SIZE="${VERL_TRAIN_BATCH_SIZE:-32}"
export VERL_PPO_MINI_BATCH_SIZE="${VERL_PPO_MINI_BATCH_SIZE:-8}"
export VERL_PPO_MICRO_BATCH_SIZE_PER_GPU="${VERL_PPO_MICRO_BATCH_SIZE_PER_GPU:-2}"
export VERL_ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${VERL_ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-2}"
export VERL_REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${VERL_REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-2}"
export VERL_ROLLOUT_N="${VERL_ROLLOUT_N:-8}"
export VERL_MAX_PROMPT_LENGTH="${VERL_MAX_PROMPT_LENGTH:-4096}"
export VERL_MAX_RESPONSE_LENGTH="${VERL_MAX_RESPONSE_LENGTH:-2048}"
export VERL_ROLLOUT_GPU_MEMORY_UTILIZATION="${VERL_ROLLOUT_GPU_MEMORY_UTILIZATION:-0.45}"
export VERL_LORA_RANK="${VERL_LORA_RANK:-32}"
export VERL_LORA_ALPHA="${VERL_LORA_ALPHA:-64}"
export VERL_LR="${VERL_LR:-5e-6}"
export VERL_TOTAL_EPOCHS="${VERL_TOTAL_EPOCHS:-1}"
export VERL_SAVE_FREQ="${VERL_SAVE_FREQ:-10}"
export VERL_TEST_FREQ="${VERL_TEST_FREQ:-10}"
export PROJECT_NAME="${PROJECT_NAME:-cybertown_rlvr}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3_0_6b_cybertown_rlvr}"

started_validator=0
if [ "${GSI_START_VALIDATOR:-1}" = "1" ]; then
  echo "[INFO] Starting GSI validator at http://${GSI_VALIDATOR_HOST}:${GSI_VALIDATOR_PORT}"
  "${ROOT_DIR}/llm_finetune/scripts/runtime/serve_validator.sh" start
  echo "[INFO] GSI validator is ready."
  started_validator=1
fi

echo "[INFO] Starting RLVR trainer: ${RUN_RLVR_SCRIPT}"
set +e
"${RUN_RLVR_SCRIPT}" \
  actor_rollout_ref.rollout.max_model_len="${VLLM_MAX_MODEL_LEN}" \
  actor_rollout_ref.rollout.max_num_batched_tokens="${VLLM_MAX_NUM_BATCHED_TOKENS}" \
  "$@"
status=$?
set -e

if [ "${started_validator}" = "1" ] && [ "${GSI_STOP_VALIDATOR_ON_EXIT:-0}" = "1" ]; then
  "${ROOT_DIR}/llm_finetune/scripts/runtime/serve_validator.sh" stop || true
fi

exit "${status}"
