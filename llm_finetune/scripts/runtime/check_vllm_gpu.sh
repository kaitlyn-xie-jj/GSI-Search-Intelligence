#!/usr/bin/env bash
set -euo pipefail

VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.75}"
VLLM_CUDA_VISIBLE_DEVICES="${VLLM_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-0}}"
VLLM_PREFLIGHT_HEADROOM_MB="${VLLM_PREFLIGHT_HEADROOM_MB:-1024}"

trim() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "$value"
}

is_valid_utilization() {
  awk -v util="$VLLM_GPU_MEMORY_UTILIZATION" '
    BEGIN {
      if (util ~ /^[0-9]*\.?[0-9]+$/ && util > 0 && util <= 1) exit 0
      exit 1
    }
  '
}

gpu_is_selected() {
  local gpu_idx="$1"
  local visible="$2"
  local item

  if [ -z "$visible" ]; then
    visible="0"
  fi
  if [ "$visible" = "all" ]; then
    return 0
  fi

  IFS=',' read -r -a visible_items <<< "$visible"
  for item in "${visible_items[@]}"; do
    item="$(trim "$item")"
    if [ "$item" = "$gpu_idx" ]; then
      return 0
    fi
  done

  return 1
}

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[ERROR] nvidia-smi not found; cannot check vLLM GPU memory." >&2
  exit 1
fi

if ! is_valid_utilization; then
  echo "[ERROR] VLLM_GPU_MEMORY_UTILIZATION must be > 0 and <= 1: ${VLLM_GPU_MEMORY_UTILIZATION}" >&2
  exit 1
fi

if [ -n "${VLLM_PREFLIGHT_MIN_FREE_MB:-}" ] && ! [[ "$VLLM_PREFLIGHT_MIN_FREE_MB" =~ ^[0-9]+$ ]]; then
  echo "[ERROR] VLLM_PREFLIGHT_MIN_FREE_MB must be an integer MiB value: ${VLLM_PREFLIGHT_MIN_FREE_MB}" >&2
  exit 1
fi

if ! [[ "$VLLM_PREFLIGHT_HEADROOM_MB" =~ ^[0-9]+$ ]]; then
  echo "[ERROR] VLLM_PREFLIGHT_HEADROOM_MB must be an integer MiB value: ${VLLM_PREFLIGHT_HEADROOM_MB}" >&2
  exit 1
fi

gpu_rows="$(nvidia-smi --query-gpu=index,memory.free,memory.total --format=csv,noheader,nounits)"
found=0
bad=0

echo "[PREFLIGHT] vLLM GPU memory check"
while IFS=',' read -r gpu_idx free_mb total_mb; do
  gpu_idx="$(trim "$gpu_idx")"
  free_mb="$(trim "$free_mb")"
  total_mb="$(trim "$total_mb")"

  if ! gpu_is_selected "$gpu_idx" "$VLLM_CUDA_VISIBLE_DEVICES"; then
    continue
  fi

  found=1
  required_mb="$(awk -v total="$total_mb" -v util="$VLLM_GPU_MEMORY_UTILIZATION" -v headroom="$VLLM_PREFLIGHT_HEADROOM_MB" 'BEGIN { printf "%d", int(total * util + headroom + 0.999999) }')"
  if [ -n "${VLLM_PREFLIGHT_MIN_FREE_MB:-}" ] && [ "$VLLM_PREFLIGHT_MIN_FREE_MB" -gt "$required_mb" ]; then
    required_mb="$VLLM_PREFLIGHT_MIN_FREE_MB"
  fi

  echo "  gpu=${gpu_idx} free=${free_mb}MiB total=${total_mb}MiB utilization=${VLLM_GPU_MEMORY_UTILIZATION} headroom=${VLLM_PREFLIGHT_HEADROOM_MB}MiB required=${required_mb}MiB"

  if [ "$free_mb" -lt "$required_mb" ]; then
    bad=1
  fi
done <<< "$gpu_rows"

if [ "$found" -eq 0 ]; then
  echo "[ERROR] No selected GPU found in nvidia-smi output for VLLM_CUDA_VISIBLE_DEVICES=${VLLM_CUDA_VISIBLE_DEVICES}" >&2
  exit 1
fi

if [ "$bad" -ne 0 ]; then
  echo "[ERROR] At least one selected GPU has less free memory than vLLM requires. Decrease VLLM_GPU_MEMORY_UTILIZATION, choose another GPU, or stop stale GPU processes." >&2
  exit 1
fi

echo "[PREFLIGHT] vLLM GPU memory OK."
