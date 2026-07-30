# Use Hugging Face Models and Data

GSI public task data, SFT/RLVR models, and training datasets are managed through Hugging Face. Runtime commands should set explicit local paths to avoid inconsistent cache usage.

## Download the Benchmark Dataset

```bash
python - <<'PY'
from huggingface_hub import snapshot_download
path = snapshot_download(
    repo_id="WindyLab/GSI",
    repo_type="dataset",
    revision="small",
)
print(path)
PY
```

Set the dataset path:

```bash
export GSI_DATASET_ROOT=/path/printed/by/snapshot_download
```

Pass it to benchmark:

```bash
python run/run_exp_multi_method.py \
  --methods sgi \
  --dataset-root "$GSI_DATASET_ROOT"
```

## Training Resources

Common models:

```text
Qwen/Qwen3-0.6B
WindyLab/Qwen3-0.6B-cybertown-SFT
WindyLab/Qwen3-0.6B-cybertown-RLVR
```

Common datasets:

```text
WindyLab/GSI
WindyLab/Qwen3-0.6B-cybertown-SFT-data
WindyLab/Qwen3-0.6B-cybertown-RLVR-data
```

Preparation inside the training container is covered in [Hugging Face Preparation](../training/huggingface-prepare.md).

## Offline Mode

To use only local cache:

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

Offline mode requires complete local model and dataset files. Missing files cause immediate failure. See [Hugging Face Cache](../troubleshooting/huggingface-cache.md).
