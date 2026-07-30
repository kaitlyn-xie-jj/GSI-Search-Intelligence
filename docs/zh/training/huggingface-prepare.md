# Hugging Face 准备

本页说明训练和评估前如何准备 Hugging Face cache。命令默认在宿主机的 GSI 仓库根目录执行。

## 设置缓存目录

建议将 cache 放在仓库目录下，便于 Docker compose 挂载到容器内：

```bash
export HF_HOME=$PWD/hf_cache
export HF_HUB_CACHE=$HF_HOME/hub
mkdir -p "$HF_HOME"
```

如果该目录曾由 root 容器写入，宿主机用户可能遇到 `Permission denied`。应先修正目录所有权，或删除不完整 cache 后重新下载。

## 下载资源

基础模型：

```bash
hf download Qwen/Qwen3-0.6B
```

已发布 checkpoint：

```bash
hf download WindyLab/Qwen3-0.6B-cybertown-SFT
hf download WindyLab/Qwen3-0.6B-cybertown-RLVR
```

训练数据：

```bash
hf download --repo-type dataset WindyLab/Qwen3-0.6B-cybertown-SFT-data
hf download --repo-type dataset WindyLab/Qwen3-0.6B-cybertown-RLVR-data
```

Benchmark 数据：

```bash
hf download --repo-type dataset --revision small WindyLab/GSI
```

资源清单：

```text
Model:   Qwen/Qwen3-0.6B
Model:   WindyLab/Qwen3-0.6B-cybertown-SFT
Model:   WindyLab/Qwen3-0.6B-cybertown-RLVR
Dataset: WindyLab/Qwen3-0.6B-cybertown-SFT-data
Dataset: WindyLab/Qwen3-0.6B-cybertown-RLVR-data
Dataset: WindyLab/GSI (revision: small)
```

`WindyLab/GSI` 用于 benchmark、replan 采集和 multi-method 评估，不是 SFT/RLVR 训练数据。

## 容器内检查

进入训练容器：

```bash
cd /GSI/llm_finetune
./scripts/runtime/check_env.sh
```

离线运行时设置：

```bash
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_MODE=offline
```

离线模式只解析已存在的本地 snapshot。缺文件时会直接失败。

## 验证 Benchmark 数据集

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

检查 task 和 goal 文件：

```bash
find "$GSI_DATASET_ROOT" -maxdepth 5 \( -name tasks.jsonl -o -name goals.jsonl \) -print
```

无输出通常表示 cache 不完整，或下载的 revision 不包含 benchmark task。

## 常见问题

- `401 Unauthorized`：检查 repo 权限和 Hugging Face token。
- `Permission denied`：检查 cache 目录所有权。
- 离线模式找不到模型：确认宿主机和容器使用同一 cache。
- 下载显示 `0.00B`：如果文件已在 cache 中，这是正常复用。
