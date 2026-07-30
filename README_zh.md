# GSI

TODO

---

## 目录

- [配置与运行](#配置与运行)
- [数据集与模型](#数据集与模型)
- [训练](#训练)
- [项目结构](#项目结构)
- [详细文档](#详细文档)

---

## 配置与运行

### 0. 创建宿主机 `gsi` 环境

无论使用 Docker 还是手动运行，宿主机上都建议先创建 `gsi` 虚拟环境，避免改动系统级 Python。

```bash
conda create -n gsi python=3.10
conda activate gsi

python -m pip install -U huggingface_hub mkdocs-material
```

### 方式 A：Docker 配置

#### 1. 准备 Hugging Face Cache

在宿主机仓库根目录执行。以下命令使用前面创建的 `gsi` 环境：

```bash
cd /path/to/GSI
conda activate gsi

export HF_HOME=$PWD/hf_cache
export HF_HUB_CACHE=$HF_HOME/hub
mkdir -p "$HF_HOME"

# 下载 benchmark 数据集
hf download --repo-type dataset --revision small WindyLab/GSI

# 下载 rlvr 微调 llm
hf download WindyLab/Qwen3-0.6B-cybertown-RLVR
```

如需训练，再下载：

```bash
# sft 微调数据集
hf download --repo-type dataset WindyLab/Qwen3-0.6B-cybertown-SFT-data

# rlvr 微调数据集
hf download --repo-type dataset WindyLab/Qwen3-0.6B-cybertown-RLVR-data

# 经过 sft 微调后的模型
hf download WindyLab/Qwen3-0.6B-cybertown-SFT

# 基础模型
hf download Qwen/Qwen3-0.6B
```

#### 2. 构建并进入容器

```bash
docker compose -f docker/docker-compose.runtime-train.yml build gsi-train
docker compose -f docker/docker-compose.runtime-train.yml run --rm gsi-train bash
```

容器内默认工作目录是 `/GSI`：

```bash
cd /GSI
check_env.sh
```

期望最后看到：

```text
GSI runtime/training environment OK
```

Docker 挂载、代理、离线模式和路径映射见 [`docker/README.runtime-train.md`](docker/README.runtime-train.md)。

#### 3. 在容器内解析数据集路径

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

没有输出时，说明容器可见的 Hugging Face cache 不完整，或挂载路径不正确。

#### 4. 启动 vLLM

在第一个容器 shell 中启动模型服务：

```bash
export VLLM_HOST=127.0.0.1
export VLLM_PORT=8001
export VLLM_GPU_MEMORY_UTILIZATION=0.75
export VLLM_MAX_NUM_SEQS=64
export SERVED_MODEL_NAME=qwen3_0_6b_cybertown_rlvr
export MODEL_PATH=WindyLab/Qwen3-0.6B-cybertown-RLVR
```

如果运行环境不能稳定访问 Hugging Face，但 `hf_cache` 已完整挂载，先改用本地 cache 路径：

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export MODEL_PATH=$(
python - <<'PY'
from huggingface_hub import snapshot_download
print(snapshot_download(
    "WindyLab/Qwen3-0.6B-cybertown-RLVR",
    local_files_only=True,
))
PY
)
```

启动服务：

```bash
CUDA_VISIBLE_DEVICES=0 ./llm_finetune/scripts/runtime/check_vllm_gpu.sh

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

#### 5. 配置运行环境

另开一个容器 shell：

```bash
docker compose -f docker/docker-compose.runtime-train.yml run --rm gsi-train bash
cd /GSI
```

检查模型服务：

```bash
curl -fsS http://127.0.0.1:8001/v1/models | python -m json.tool
```

设置运行环境：

```bash
export GSI_LLM_API_BASE=http://127.0.0.1:8001/v1
export GSI_LLM_API_KEY=EMPTY
export GSI_LLM_MODEL=qwen3_0_6b_cybertown_rlvr
export GSI_DISABLE_TOKEN_STATS=1
export GSI_TANGO_SOLVER_BACKEND=scip
export GSI_TANGO_SOLVER_MAX_TIME=120
```

其中，`GSI_TANGO_SOLVER_BACKEND`是求解器后端，`scip`是开源求解器，`gurobi`是闭源求解器，`gurobi`的求解速度约为`scip`的两倍。如果机器有有效 Gurobi license，可以切换：
```bash
export GSI_TANGO_SOLVER_BACKEND=gurobi
```
Gurobi 安装和 license 说明见 [Solver 故障排查](docs/zh/troubleshooting/solver.md)。另外注意要在该新容器内解析数据集路径
```bash
export GSI_DATASET_ROOT=xxx
```

然后转到 [运行命令](#运行命令)，开始测试。

---

### 方式 B：手动本地运行

#### 1. 安装 GSI 本地依赖

```bash
conda activate gsi
```

先安装与 CUDA 版本匹配的 PyTorch。示例：

```bash
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
```

安装项目依赖：

```bash
pip install -r requirements.txt
```

#### 2. 创建本地 vLLM 环境（可选）

如果使用云端 API，可以跳过本步骤。如果在本机启动模型服务，建议单独创建 `vllm` 环境，避免 vLLM 依赖影响 GSI 主运行环境。

```bash
conda create -n vllm python=3.10
conda activate vllm

pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install vllm==0.13.0
```

`vllm` 环境中的 CUDA / PyTorch 版本可与 `gsi` 环境保持一致。在 `vllm` 环境中启动本地模型服务：

```bash
export SERVED_MODEL_NAME=qwen3_0_6b_cybertown_rlvr
export MODEL_PATH=WindyLab/Qwen3-0.6B-cybertown-RLVR

CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --host 127.0.0.1 \
  --port 8001 \
  --max-model-len 4096 \
  --max-num-seqs 64 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.75 \
  --trust-remote-code \
  --no-enable-log-requests
```

该终端会被 vLLM 服务占用。后续 GSI 命令应在另一个终端中切回 `gsi` 环境执行。

#### 3. 配置数据、模型和 solver

手动方式仍需设置：

```bash
conda activate gsi

export GSI_DATASET_ROOT=/path/to/gsi/dataset
export GSI_LLM_API_BASE=http://127.0.0.1:8001/v1
export GSI_LLM_API_KEY=EMPTY
export GSI_LLM_MODEL=qwen3_0_6b_cybertown_rlvr
export GSI_TANGO_SOLVER_BACKEND=scip
export GSI_TANGO_SOLVER_MAX_TIME=120
```

如果使用云端 API，将 `GSI_LLM_API_BASE`、`GSI_LLM_API_KEY` 和 `GSI_LLM_MODEL` 改为对应服务配置，如
```bash
export GSI_LLM_API_BASE=https://openrouter.ai/api/v1
export GSI_LLM_API_KEY=your_api_key
export GSI_LLM_MODEL=google/gemini-3-flash-preview
```

然后转到 [运行命令](#运行命令)，开始测试。

---

## 运行命令

适用于 Docker 和手动本地环境。区别只在于：Docker 中工作目录是 `/GSI`；手动环境中工作目录是本地仓库根目录。

### 单任务调试

```bash
python run/run_exp.py
```
可通过`--task-id`参数选择任务

### 批量 Benchmark

```bash
python run/run_exp_multi_method.py \
  --methods sgi \
  --max-count 100 \
  --max-workers 10 \
  --dataset-root "$GSI_DATASET_ROOT" \
  --output-root results/batch_runs
```

### 多方法对比

```bash
python run/run_exp_multi_method.py \
  --methods sgi spine smartllm lipllm \
  --max-count 200 \
  --max-workers 10 \
  --dataset-root "$GSI_DATASET_ROOT" \
  --output-root results/batch_runs
```
注意，注意该命令需要设置调用云端 api，因为微调模型只适用于 sgi。

### New Case / Replan 评估

```bash
python run/run_exp_multi_method.py \
  --methods sgi \
  --enable-newcase \
  --newcase-counts 1 2 3 4 \
  --dataset-root "$GSI_DATASET_ROOT" \
  --output-root results/newcase_runs
```

输出文件说明见 [输出目录](docs/zh/training/outputs.md)。

### Unreal 平台

Unreal 平台需要先启动 [MultiAgent-Unreal](https://github.com/WindyLab/MultiAgent-Unreal) 仿真环境。

1. 将 `config/default.json` 中 `platform_type` 设为 `"unreal"`。
2. 运行：

```bash
python run/run_exp.py
```

---

## 训练

训练推荐使用 Docker 方式。训练前进入容器：

```bash
docker compose -f docker/docker-compose.runtime-train.yml run --rm gsi-train bash
cd /GSI/llm_finetune
export ROOT_DIR=/GSI
./scripts/runtime/check_env.sh
```

### SFT

```bash
SFT_MODEL_REPO_ID=Qwen/Qwen3-0.6B \
SFT_DATA_REPO_ID=WindyLab/Qwen3-0.6B-cybertown-SFT-data \
SFT_OUTPUT_DIR=/GSI/outputs/sft/qwen3_0_6b_cybertown_sft \
SFT_MAX_SEQ_LENGTH=4096 \
SFT_PER_DEVICE_BATCH_SIZE=1 \
SFT_GRAD_ACCUM=8 \
SFT_EPOCHS=1 \
./scripts/runtime/train_sft_unsloth.sh
```

合并 LoRA：

```bash
python -m llm_finetune.sft.merge_lora \
  --base-model Qwen/Qwen3-0.6B \
  --adapter-path /GSI/outputs/sft/qwen3_0_6b_cybertown_sft/lora_adapters \
  --output-dir /GSI/outputs/sft/qwen3_0_6b_cybertown_sft_merged
```

### RLVR

RLVR 通常从 SFT 后模型开始：

```bash
export RLVR_OUTPUT_DIR=/GSI/outputs/rlvr_gsi/qwen3_0_6b_cybertown_rlvr
export GSI_TANGO_SOLVER_BACKEND=scip

TRAIN_CUDA_VISIBLE_DEVICES=0 \
RLVR_MODEL_REPO_ID=WindyLab/Qwen3-0.6B-cybertown-SFT \
RLVR_DATA_REPO_ID=WindyLab/Qwen3-0.6B-cybertown-RLVR-data \
RLVR_STATE_STORE_NAME=train6_val10_initial30_replan70_val15_20260528 \
RLVR_OUTPUT_DIR=$RLVR_OUTPUT_DIR \
GSI_PREFLIGHT=1 \
GSI_PREFLIGHT_MIN_GPU_FREE_MB=18000 \
VERL_N_GPUS_PER_NODE=1 \
VERL_TRAIN_BATCH_SIZE=4 \
VERL_PPO_MINI_BATCH_SIZE=2 \
VERL_ROLLOUT_N=2 \
VERL_MAX_PROMPT_LENGTH=4096 \
VERL_MAX_RESPONSE_LENGTH=2048 \
VERL_ROLLOUT_GPU_MEMORY_UTILIZATION=0.35 \
VERL_SAVE_FREQ=200 \
VERL_TEST_FREQ=200 \
VLLM_MAX_MODEL_LEN=6144 \
VLLM_MAX_NUM_BATCHED_TOKENS=6144 \
./scripts/runtime/train_rlvr.sh
```

`RLVR_OUTPUT_DIR` 中已有 `global_step_*` 时，VeRL 默认自动恢复训练。新实验建议使用新的输出目录。

完整训练说明：

- [SFT 训练](docs/zh/training/sft.md)
- [RLVR 训练](docs/zh/training/rlvr.md)
- [GPU 与 vLLM](docs/zh/troubleshooting/gpu-and-vllm.md)
- [Ray 与 Checkpoint](docs/zh/troubleshooting/ray-and-checkpoint.md)

---


## 数据集与模型

### 公开资源

| 类型 | 资源 | 用途 |
|------|------|------|
| Benchmark 数据集 | `WindyLab/GSI` | 语义任务、场景、目标和 prompt 数据，用于 benchmark 和复现。 |
| Base model | `Qwen/Qwen3-0.6B` | SFT 训练起点。 |
| SFT 模型 | `WindyLab/Qwen3-0.6B-cybertown-SFT` | RLVR 基础模型，也可直接评估。 |
| RLVR 模型 | `WindyLab/Qwen3-0.6B-cybertown-RLVR` | 训练后完整模型，适合直接用 vLLM 评估。 |
| SFT 数据 | `WindyLab/Qwen3-0.6B-cybertown-SFT-data` | SFT 训练数据。 |
| RLVR 数据 | `WindyLab/Qwen3-0.6B-cybertown-RLVR-data` | RLVR/GRPO 训练数据。 |

### 手动构建测试数据集

该部分为可选步骤，不影响测试与训练。

生成场景：

```bash
python modules/dataset_builder/generate_scenarios.py
```

生成目标：

```bash
python modules/dataset_builder/generate_goals.py
```

生成任务：

```bash
python modules/dataset_builder/generate_tasks.py
```

采集 replan 数据：

```bash
python run/run_collect_replan_dataset.py --help
```

更多说明：

- [Hugging Face 准备](docs/zh/training/huggingface-prepare.md)
- [SFT 训练](docs/zh/training/sft.md)
- [RLVR 训练](docs/zh/training/rlvr.md)
- [采集 Replan 数据](docs/zh/tutorials/collect-replan-data.md)

## 详细文档

文档入口：

- 中文文档：[docs/zh/index.md](docs/zh/index.md)
- 英文文档：[docs/en/index.md](docs/en/index.md)
- 快速跑通：[docs/zh/getting-started/quickstart.md](docs/zh/getting-started/quickstart.md)
- 复现实验：[docs/zh/getting-started/reproduce-results.md](docs/zh/getting-started/reproduce-results.md)
- 系统架构：[docs/zh/concepts/architecture.md](docs/zh/concepts/architecture.md)
- 配置参考：[docs/zh/reference/config.md](docs/zh/reference/config.md)
- 训练概览：[docs/zh/training/overview.md](docs/zh/training/overview.md)

本地网页预览：

```bash
conda activate gsi
python -m mkdocs serve -a 127.0.0.1:8002
```

访问：

```text
http://127.0.0.1:8002/
http://127.0.0.1:8002/zh/
http://127.0.0.1:8002/en/
```

构建静态网站：

```bash
conda run -n gsi python -m mkdocs build --strict
```

---

## Citation

论文引用信息会在正式发布前补齐。
