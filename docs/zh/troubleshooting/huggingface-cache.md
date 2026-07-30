# Hugging Face Cache

Hugging Face cache 常见问题包括缺文件、离线模式失败、权限错误，以及宿主机与容器看到的 cache 不一致。

## 在线下载

```bash
python - <<'PY'
from huggingface_hub import snapshot_download
print(snapshot_download(repo_id="WindyLab/GSI", repo_type="dataset"))
PY
```

## 离线模式

```bash
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

离线模式要求 cache 中已经存在完整模型或数据集。缺文件时，应关闭离线模式并重新下载。

## 容器路径

训练容器内常见 cache 路径：

```text
/root/.cache/huggingface
```

如果宿主机已下载但容器不可见，检查 Docker volume 是否挂载了同一 cache 目录。

## 权限问题

如果 cache 曾由 root 容器写入，宿主机用户可能遇到 `Permission denied`。修正目录所有权后再下载，或清理不完整 cache 后重新准备。

## 训练数据结构

RLVR 数据不仅需要 parquet，还需要 state index 和 shard：

```text
train.parquet
val.parquet
states.index.json
states/*.jsonl
```

准备流程见 [Hugging Face 准备](../training/huggingface-prepare.md)。
