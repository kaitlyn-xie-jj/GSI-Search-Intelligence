#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

SFT_BACKEND="${SFT_BACKEND:-transformers}"
if [[ "${SFT_BACKEND}" == "unsloth" ]]; then
  python -m llm_finetune.sft.sft_train_unsloth "$@"
elif [[ "${SFT_BACKEND}" == "transformers" ]]; then
  python -m llm_finetune.sft.sft_train "$@"
else
  echo "Unknown SFT_BACKEND=${SFT_BACKEND}. Expected unsloth or transformers." >&2
  exit 2
fi
