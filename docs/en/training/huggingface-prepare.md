# Hugging Face Preparation

This page explains how to prepare Hugging Face cache before training and evaluation. Commands are assumed to run from the GSI repository root on the host.

## Set Cache Directory

Place the cache under the repository so Docker compose can mount it into the container:

```bash
export HF_HOME=$PWD/hf_cache
export HF_HUB_CACHE=$HF_HOME/hub
mkdir -p "$HF_HOME"
```

If a root container previously wrote this directory, the host user may see `Permission denied`. Fix ownership or remove the incomplete cache and download again.

## Download Resources

Base model:

```bash
hf download Qwen/Qwen3-0.6B
```

Released checkpoints:

```bash
hf download WindyLab/Qwen3-0.6B-cybertown-SFT
hf download WindyLab/Qwen3-0.6B-cybertown-RLVR
```

Training data:

```bash
hf download --repo-type dataset WindyLab/Qwen3-0.6B-cybertown-SFT-data
hf download --repo-type dataset WindyLab/Qwen3-0.6B-cybertown-RLVR-data
```

Benchmark data:

```bash
hf download --repo-type dataset --revision small WindyLab/GSI
```

Resource list:

```text
Model:   Qwen/Qwen3-0.6B
Model:   WindyLab/Qwen3-0.6B-cybertown-SFT
Model:   WindyLab/Qwen3-0.6B-cybertown-RLVR
Dataset: WindyLab/Qwen3-0.6B-cybertown-SFT-data
Dataset: WindyLab/Qwen3-0.6B-cybertown-RLVR-data
Dataset: WindyLab/GSI (revision: small)
```

`WindyLab/GSI` is used for benchmark, replan collection, and multi-method evaluation. It is not SFT/RLVR training data.

## Check Inside the Container

Enter the training container:

```bash
cd /GSI/llm_finetune
./scripts/runtime/check_env.sh
```

For offline runs:

```bash
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline
```

Offline mode only resolves existing local snapshots. Missing files cause immediate failure.

## Verify Benchmark Dataset

```bash
export GSI_DATASET_ROOT=$(python - <<'PY'
from huggingface_hub import snapshot_download

print(snapshot_download(
    repo_id="WindyLab/GSI",
    repo_type="dataset",
    revision="small",
))
PY
)
```

Check task and goal files:

```bash
find "$GSI_DATASET_ROOT" -maxdepth 5 \( -name tasks.jsonl -o -name goals.jsonl \) -print
```

No output usually means the cache is incomplete or the downloaded revision does not contain benchmark tasks.

## Common Issues

- `401 Unauthorized`: check repository permissions and Hugging Face token.
- `Permission denied`: check cache directory ownership.
- Offline mode cannot find a model: confirm that host and container use the same cache.
- Download shows `0.00B`: if files already exist in cache, this is normal reuse.
