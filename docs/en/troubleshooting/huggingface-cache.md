# Hugging Face Cache

Common Hugging Face cache issues include missing files, offline-mode failures, permission errors, and different cache visibility between host and container.

## Online Download

```bash
python - <<'PY'
from huggingface_hub import snapshot_download
print(snapshot_download(repo_id="WindyLab/GSI", repo_type="dataset"))
PY
```

## Offline Mode

```bash
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

Offline mode requires complete local model or dataset files. Disable offline mode and download again when files are missing.

## Container Path

Common cache path inside the training container:

```text
/root/.cache/huggingface
```

If the host has downloaded files but the container cannot see them, check whether the Docker volume mounts the same cache directory.

## Permissions

If a root container wrote the cache, the host user may see `Permission denied`. Fix directory ownership or remove the incomplete cache and prepare it again.

## Training Data Structure

RLVR data requires parquet files plus state index and shards:

```text
train.parquet
val.parquet
states.index.json
states/*.jsonl
```

Preparation is covered in [Hugging Face Preparation](../training/huggingface-prepare.md).
