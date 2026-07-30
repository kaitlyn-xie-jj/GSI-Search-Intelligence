# Quick Start

This page runs a minimal SGI benchmark with Docker. The goal is to verify the dataset, model service, solver, and output pipeline. For full reproduction, see [Reproduce Results](reproduce-results.md).

## Prerequisites

The minimal runtime chain includes:

```text
GSI Docker image
Hugging Face cache
OpenAI-compatible LLM endpoint
TANGO solver backend
```

Docker details are in `docker/README.runtime-train.md`. A local Python setup can use the same commands if Python, CUDA, and vLLM are already stable.

## 1. Prepare Hugging Face Cache

Run on the host from the repository root:

```bash
cd /path/to/GSI

python -m pip install -U huggingface_hub

export HF_HOME=$PWD/hf_cache
export HF_HUB_CACHE=$HF_HOME/hub
mkdir -p "$HF_HOME"

hf download --repo-type dataset --revision small WindyLab/GSI
hf download WindyLab/Qwen3-0.6B-cybertown-RLVR
```

Download training resources only when needed:

```bash
hf download --repo-type dataset WindyLab/Qwen3-0.6B-cybertown-SFT-data
hf download --repo-type dataset WindyLab/Qwen3-0.6B-cybertown-RLVR-data
hf download WindyLab/Qwen3-0.6B-cybertown-SFT
hf download Qwen/Qwen3-0.6B
```

`WindyLab/GSI` is the benchmark dataset. `WindyLab/Qwen3-0.6B-cybertown-RLVR` is the full model used by the quick start.

## 2. Enter the Runtime Container

```bash
cd /path/to/GSI
docker compose -f docker/docker-compose.runtime-train.yml build gsi-train
docker compose -f docker/docker-compose.runtime-train.yml run --rm gsi-train bash
```

Inside the container:

```bash
cd /GSI
check_env.sh
```

## 3. Resolve the Dataset Path

```bash
export GSI_DATASET_ROOT=$(
python - <<'PY'
from huggingface_hub import snapshot_download
print(snapshot_download(
    repo_id="WindyLab/GSI",
    repo_type="dataset",
    revision="small",
))
PY
)

find "$GSI_DATASET_ROOT" -maxdepth 5 \( -name tasks.jsonl -o -name goals.jsonl \) -print
```

No output usually means the container-visible cache is incomplete or the mount path is wrong.

## 4. Start vLLM

In the first container shell:

```bash
export VLLM_HOST=127.0.0.1
export VLLM_PORT=8001
export VLLM_GPU_MEMORY_UTILIZATION=0.75
export VLLM_MAX_NUM_SEQS=64
export SERVED_MODEL_NAME=qwen3_0_6b_cybertown_rlvr
export MODEL_PATH=WindyLab/Qwen3-0.6B-cybertown-RLVR
```

Check free GPU memory:

```bash
CUDA_VISIBLE_DEVICES=0 ./llm_finetune/scripts/runtime/check_vllm_gpu.sh
```

Start the service:

```bash
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --host "$VLLM_HOST" \
  --port "$VLLM_PORT" \
  --max-model-len 4096 \
  --max-num-seqs "$VLLM_MAX_NUM_SEQS" \
  --dtype bfloat16 \
  --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
  --trust-remote-code \
  --no-enable-log-requests
```

## 5. Check the Service and Configure GSI

Open another container shell:

```bash
cd /path/to/GSI
docker compose -f docker/docker-compose.runtime-train.yml run --rm gsi-train bash
cd /GSI
```

Check the model list:

```bash
curl -fsS http://127.0.0.1:8001/v1/models | python -m json.tool
```

Set GSI variables:

```bash
export GSI_LLM_API_BASE=http://127.0.0.1:8001/v1
export GSI_LLM_API_KEY=EMPTY
export GSI_LLM_MODEL=qwen3_0_6b_cybertown_rlvr
export GSI_DISABLE_TOKEN_STATS=1

export GSI_TANGO_SOLVER_BACKEND=scip
export GSI_TANGO_SOLVER_MAX_TIME=120
```

## 6. Run the Minimal Benchmark

```bash
python run/run_exp_multi_method.py \
  --methods sgi \
  --task-mix transport=1.0 \
  --max-count 2 \
  --max-workers 1 \
  --dataset-root "$GSI_DATASET_ROOT" \
  --output-root outputs/quickstart
```

Use `--max-workers 1` for the first run to verify the pipeline before increasing concurrency.

## 7. Inspect Results

Typical output:

```text
outputs/quickstart/<timestamp>/
  batch_sgi/
    summary.jsonl
    aggregate_full.json
    0001_<task_id>/
    0002_<task_id>/
```

View the latest aggregate:

```bash
python - <<'PY'
import json
from pathlib import Path

root = Path("outputs/quickstart")
latest = max(root.iterdir(), key=lambda p: p.stat().st_mtime)
agg = latest / "batch_sgi" / "aggregate_full.json"
print(agg)
print(json.dumps(json.loads(agg.read_text()).get("overall", {}), ensure_ascii=False, indent=2))
PY
```

Check:

- `overall.success_rate`
- `overall.replans_total`
- `overall.total_energy`
- `input_params`

See [Output Directory](../training/outputs.md) for details.

## Local Python Run

Install dependencies:

```bash
cd /path/to/GSI
python -m pip install -r requirements.txt
```

Dataset preparation, vLLM startup, environment variables, and benchmark commands are the same as in Docker. Paths are local instead of `/GSI`, and outputs are written to the current workspace.

## Common Issues

- `check_env.sh` fails: rebuild the image and check GPU visibility and mounts.
- Dataset directory is empty: re-run `snapshot_download(...)` or check the cache mount.
- `/v1/models` is unreachable: check the vLLM process, port, GPU memory, and startup logs.
- `model not found`: set `GSI_LLM_MODEL` to the id returned by `/v1/models`.
- `No tasks found. Exiting.`: check the dataset path and task filters.
- Allocator errors: use `GSI_TANGO_SOLVER_BACKEND=scip` first, then see [Solver](../troubleshooting/solver.md).

## Next Steps

- Reproduce experiments: [Reproduce Results](reproduce-results.md).
- Understand the system: [Architecture](../concepts/architecture.md).
- Evaluate another model: [vLLM Startup and GSI Evaluation](../training/vllm-eval.md).
