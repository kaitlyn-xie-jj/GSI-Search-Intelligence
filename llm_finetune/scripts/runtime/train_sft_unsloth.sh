#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/GSI}"
DEFAULT_SFT_MODEL_REPO_ID="Qwen/Qwen3-0.6B"
DEFAULT_SFT_DATA_REPO_ID="WindyLab/Qwen3-0.6B-cybertown-SFT-data"
DEFAULT_SFT_DATA_FILENAME="train.jsonl"
RUN_SFT_SCRIPT="${RUN_SFT_SCRIPT:-${ROOT_DIR}/llm_finetune/scripts/runtime/internal/run_sft.sh}"
source "${ROOT_DIR}/llm_finetune/scripts/runtime/hf_cache_utils.sh"

cd "${ROOT_DIR}"

export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/llm_finetune/verl_scripts/verl:${PYTHONPATH:-}"
export SFT_BACKEND=unsloth

if [[ -n "${SFT_CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${SFT_CUDA_VISIBLE_DEVICES}"
elif [[ -z "${CUDA_VISIBLE_DEVICES:-}" && "${SFT_UNSLOTH_ALLOW_MULTI_GPU:-0}" != "1" ]]; then
  export CUDA_VISIBLE_DEVICES="${SFT_CUDA_VISIBLE_DEVICES:-0}"
fi

if [[ "${SFT_UNSLOTH_ALLOW_MULTI_GPU:-0}" != "1" ]]; then
  if [[ "${CUDA_VISIBLE_DEVICES:-}" == *","* || "${CUDA_VISIBLE_DEVICES:-}" == "all" ]]; then
    echo "[ERROR] Unsloth SFT defaults to one visible GPU. Set SFT_CUDA_VISIBLE_DEVICES=<id> or SFT_UNSLOTH_ALLOW_MULTI_GPU=1 to override." >&2
    exit 1
  fi
fi

MODEL_PATH_INPUT="${MODEL_PATH:-${SFT_MODEL_PATH:-}}"
SFT_MODEL_REPO_ID="${SFT_MODEL_REPO_ID:-}"
if [[ -z "${MODEL_PATH_INPUT}" && -z "${SFT_MODEL_REPO_ID}" ]]; then
  SFT_MODEL_REPO_ID="${DEFAULT_SFT_MODEL_REPO_ID}"
fi
MODEL_PATH="$(resolve_hf_model_path "${MODEL_PATH_INPUT}" "${SFT_MODEL_REPO_ID}" "SFT base model")"

DATA_PATH_INPUT="${DATA_PATH:-${SFT_DATA_PATH:-}}"
SFT_DATA_REPO_ID="${SFT_DATA_REPO_ID:-}"
if [[ -z "${DATA_PATH_INPUT}" && -z "${SFT_DATA_REPO_ID}" ]]; then
  SFT_DATA_REPO_ID="${DEFAULT_SFT_DATA_REPO_ID}"
fi
SFT_DATA_FILENAME="${SFT_DATA_FILENAME:-${DEFAULT_SFT_DATA_FILENAME}}"
DATA_PATH="$(resolve_sft_data_path "${DATA_PATH_INPUT}" "${SFT_DATA_REPO_ID}" "${SFT_DATA_FILENAME}")"
OUTPUT_DIR="${OUTPUT_DIR:-${SFT_OUTPUT_DIR:-${ROOT_DIR}/outputs/sft/qwen3_0_6b_cybertown_sft_unsloth}}"
TEMPLATE_PATH="${TEMPLATE_PATH:-${SFT_TEMPLATE_PATH:-${ROOT_DIR}/llm_finetune/chat_template/qwen3_nonthinking.jinja}}"

if [ ! -d "${MODEL_PATH}" ]; then
  echo "MODEL_PATH does not exist: ${MODEL_PATH}" >&2
  exit 1
fi
if [ ! -f "${DATA_PATH}" ]; then
  echo "DATA_PATH does not exist: ${DATA_PATH}" >&2
  exit 1
fi

args=(
  --model-path "${MODEL_PATH}"
  --data-path "${DATA_PATH}"
  --output-dir "${OUTPUT_DIR}"
  --template-path "${TEMPLATE_PATH}"
  --max-seq-length "${SFT_MAX_SEQ_LENGTH:-4096}"
  --per-device-batch-size "${SFT_PER_DEVICE_BATCH_SIZE:-1}"
  --gradient-accumulation-steps "${SFT_GRAD_ACCUM:-8}"
  --num-train-epochs "${SFT_EPOCHS:-1}"
  --learning-rate "${SFT_LR:-2e-4}"
  --logging-steps "${SFT_LOGGING_STEPS:-1}"
  --save-total-limit "${SFT_SAVE_TOTAL_LIMIT:-2}"
  --lora-r "${SFT_LORA_R:-16}"
  --lora-alpha "${SFT_LORA_ALPHA:-32}"
  --report-to "${SFT_REPORT_TO:-none}"
)

if [ "${SFT_LOAD_IN_4BIT:-1}" = "1" ]; then
  args+=(--load-in-4bit)
fi

exec "${RUN_SFT_SCRIPT}" "${args[@]}" "$@"
