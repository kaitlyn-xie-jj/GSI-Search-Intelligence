#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/GSI}"
cd "${ROOT_DIR}"

if [ -z "${HF_HOME:-}" ]; then HF_HOME=/root/.cache/huggingface; fi
if [ -z "${PIP_DISABLE_PIP_VERSION_CHECK:-}" ]; then PIP_DISABLE_PIP_VERSION_CHECK=1; fi
if [ -z "${UNSLOTH_COMPILE_LOCATION:-}" ]; then UNSLOTH_COMPILE_LOCATION=/tmp/unsloth_compiled_cache; fi

export HF_HOME PIP_DISABLE_PIP_VERSION_CHECK UNSLOTH_COMPILE_LOCATION
export PYTHONPATH="${ROOT_DIR}:${ROOT_DIR}/llm_finetune/verl_scripts/verl:${PYTHONPATH:-}"
export PATH="${ROOT_DIR}/llm_finetune/scripts/runtime:${PATH}"
export NO_PROXY="127.0.0.1,localhost,${NO_PROXY:-}"
export no_proxy="127.0.0.1,localhost,${no_proxy:-}"

if [ "$#" -eq 0 ]; then
  set -- bash
fi

exec "$@"
