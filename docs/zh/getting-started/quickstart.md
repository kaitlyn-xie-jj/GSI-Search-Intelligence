# 快速跑通

本页说明如何用 Docker 运行一个最小 SGI benchmark。目标是验证数据集、模型服务、solver 和输出链路是否可用；完整复现实验见 [复现实验结果](reproduce-results.md)。

## 前置条件

最小运行链路包括：

```text
GSI Docker 镜像
Hugging Face cache
OpenAI-compatible LLM endpoint
TANGO solver 后端
```

Docker 细节见 `docker/README.runtime-train.md`。如果已有稳定的 Python、CUDA 和 vLLM 环境，也可按同样命令在本地运行。

## 1. 准备 Hugging Face Cache

在宿主机仓库根目录执行：

```bash
cd /path/to/GSI

python -m pip install -U huggingface_hub

export HF_HOME=$PWD/hf_cache
export HF_HUB_CACHE=$HF_HOME/hub
mkdir -p "$HF_HOME"

hf download --repo-type dataset --revision small WindyLab/GSI
hf download WindyLab/Qwen3-0.6B-cybertown-RLVR
```

如需后续训练，再下载训练资源：

```bash
hf download --repo-type dataset WindyLab/Qwen3-0.6B-cybertown-SFT-data
hf download --repo-type dataset WindyLab/Qwen3-0.6B-cybertown-RLVR-data
hf download WindyLab/Qwen3-0.6B-cybertown-SFT
hf download Qwen/Qwen3-0.6B
```

`WindyLab/GSI` 是 benchmark 数据集；`WindyLab/Qwen3-0.6B-cybertown-RLVR` 是 quickstart 使用的完整模型。

## 2. 进入运行容器

```bash
cd /path/to/GSI
docker compose -f docker/docker-compose.runtime-train.yml build gsi-train
docker compose -f docker/docker-compose.runtime-train.yml run --rm gsi-train bash
```

容器内执行：

```bash
cd /GSI
check_env.sh
```

## 3. 解析数据集路径

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

如果没有输出，说明容器可见的 Hugging Face cache 不完整或挂载路径不正确。

## 4. 启动 vLLM

在第一个容器 shell 中设置：

```bash
export VLLM_HOST=127.0.0.1
export VLLM_PORT=8001
export VLLM_GPU_MEMORY_UTILIZATION=0.75
export VLLM_MAX_NUM_SEQS=64
export SERVED_MODEL_NAME=qwen3_0_6b_cybertown_rlvr
export MODEL_PATH=WindyLab/Qwen3-0.6B-cybertown-RLVR
```

启动前检查显存：

```bash
CUDA_VISIBLE_DEVICES=0 ./llm_finetune/scripts/runtime/check_vllm_gpu.sh
```

启动服务：

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

## 5. 检查服务并配置 GSI

另开一个容器 shell：

```bash
cd /path/to/GSI
docker compose -f docker/docker-compose.runtime-train.yml run --rm gsi-train bash
cd /GSI
```

检查模型列表：

```bash
curl -fsS http://127.0.0.1:8001/v1/models | python -m json.tool
```

设置 GSI 环境变量：

```bash
export GSI_LLM_API_BASE=http://127.0.0.1:8001/v1
export GSI_LLM_API_KEY=EMPTY
export GSI_LLM_MODEL=qwen3_0_6b_cybertown_rlvr
export GSI_DISABLE_TOKEN_STATS=1

export GSI_TANGO_SOLVER_BACKEND=scip
export GSI_TANGO_SOLVER_MAX_TIME=120
```

## 6. 运行最小 Benchmark

```bash
python run/run_exp_multi_method.py \
  --methods sgi \
  --task-mix transport=1.0 \
  --max-count 2 \
  --max-workers 1 \
  --dataset-root "$GSI_DATASET_ROOT" \
  --output-root outputs/quickstart
```

首次运行使用 `--max-workers 1`，以便优先验证链路稳定性。

## 7. 查看结果

输出目录通常为：

```text
outputs/quickstart/<timestamp>/
  batch_sgi/
    summary.jsonl
    aggregate_full.json
    0001_<task_id>/
    0002_<task_id>/
```

查看最新聚合结果：

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

重点检查：

- `overall.success_rate`
- `overall.replans_total`
- `overall.total_energy`
- `input_params`

完整说明见 [输出目录](../training/outputs.md)。

## 本地 Python 运行

本地运行时先安装依赖：

```bash
cd /path/to/GSI
python -m pip install -r requirements.txt
```

数据集准备、vLLM 启动、环境变量和 benchmark 命令与 Docker 路径一致。区别是仓库路径不再是 `/GSI`，输出会写入当前工作区。

## 常见问题

- `check_env.sh` 失败：确认镜像已重新构建、GPU 可见、仓库挂载正确。
- 数据集目录为空：重新解析 `snapshot_download(...)`，或检查 Hugging Face cache 挂载。
- `/v1/models` 不可访问：检查 vLLM 进程、端口、显存和启动日志。
- `model not found`：以 `/v1/models` 返回的 id 为准设置 `GSI_LLM_MODEL`。
- `No tasks found. Exiting.`：检查数据集路径和任务筛选条件。
- allocator 报错：先使用 `GSI_TANGO_SOLVER_BACKEND=scip`，再按 [Solver](../troubleshooting/solver.md) 排查。

## 下一步

- 复现实验：阅读 [复现实验结果](reproduce-results.md)。
- 理解系统：阅读 [系统架构](../concepts/architecture.md)。
- 评估其他模型：阅读 [vLLM 启动与 GSI 评估](../training/vllm-eval.md)。
