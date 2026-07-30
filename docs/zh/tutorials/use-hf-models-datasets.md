# 使用 Hugging Face 模型和数据

GSI 的公开任务数据、SFT/RLVR 模型和训练数据通过 Hugging Face 管理。运行时建议显式设置本地路径，避免不同脚本使用不同 cache。

## 下载 Benchmark 数据集

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

设置数据集路径：

```bash
export GSI_DATASET_ROOT=/path/printed/by/snapshot_download
```

运行 benchmark 时传入：

```bash
python run/run_exp_multi_method.py \
  --methods sgi \
  --dataset-root "$GSI_DATASET_ROOT"
```

## 训练相关资源

常用模型：

```text
Qwen/Qwen3-0.6B
WindyLab/Qwen3-0.6B-cybertown-SFT
WindyLab/Qwen3-0.6B-cybertown-RLVR
```

常用数据集：

```text
WindyLab/GSI
WindyLab/Qwen3-0.6B-cybertown-SFT-data
WindyLab/Qwen3-0.6B-cybertown-RLVR-data
```

训练容器中的准备流程见 [Hugging Face 准备](../training/huggingface-prepare.md)。

## 离线模式

如果只允许使用本地 cache：

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

离线模式要求模型和数据已经完整存在于当前 cache。缺文件时会直接失败。排查方式见 [Hugging Face Cache](../troubleshooting/huggingface-cache.md)。
