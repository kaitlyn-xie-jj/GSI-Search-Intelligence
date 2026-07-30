# vLLM 启动与 GSI 评估

本页说明如何将模型通过 vLLM 暴露为 OpenAI-compatible endpoint，并让 GSI benchmark 或 replan 采集脚本调用该服务。

适用对象：

- base/fresh 模型
- SFT LoRA adapter
- RLVR LoRA adapter
- 已合并的完整模型

## 通用环境

命令默认在训练容器内执行：

```bash
cd /GSI
export VLLM_HOST=127.0.0.1
export VLLM_PORT=8001
export VLLM_GPU_MEMORY_UTILIZATION=0.75
export VLLM_MAX_NUM_SEQS=64
export VLLM_PREFLIGHT_HEADROOM_MB=1024
```

启动前检查显存：

```bash
CUDA_VISIBLE_DEVICES=0 ./llm_finetune/scripts/runtime/check_vllm_gpu.sh
```

如果当前目录是 `/GSI/llm_finetune`：

```bash
CUDA_VISIBLE_DEVICES=0 ./scripts/runtime/check_vllm_gpu.sh
```

## Hugging Face 与离线模式

当 `--model` 使用 repo id 时，vLLM 会优先复用容器内 Hugging Face cache；缺文件且有网络时会继续下载。

只允许本地 cache 时设置：

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

如果 `--model` 是本地完整模型目录，通常不需要设置离线变量。

## 启动完整模型

适用于基础模型、SFT merged 模型、RLVR merged 模型或已发布完整 checkpoint。

已发布 RLVR 模型：

```bash
export SERVED_MODEL_NAME=qwen3_0_6b_cybertown_rlvr
export MODEL_PATH=WindyLab/Qwen3-0.6B-cybertown-RLVR
```

本地 merged 模型：

```bash
export SERVED_MODEL_NAME=qwen3_0_6b_cybertown_sft
export MODEL_PATH=/GSI/outputs/sft/qwen3_0_6b_cybertown_sft_merged
```

启动：

```bash
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --host "${VLLM_HOST:-127.0.0.1}" \
  --port "${VLLM_PORT:-8001}" \
  --max-model-len 4096 \
  --max-num-seqs "$VLLM_MAX_NUM_SEQS" \
  --dtype bfloat16 \
  --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
  --trust-remote-code \
  --no-enable-log-requests
```

## 启动 SFT LoRA Adapter

适用于 `train_sft_unsloth.sh` 输出的 `lora_adapters/`：

```bash
export BASE_MODEL=Qwen/Qwen3-0.6B
export SFT_ADAPTER_DIR=/GSI/outputs/sft/qwen3_0_6b_cybertown_sft/lora_adapters
export SERVED_MODEL_NAME=qwen3_0_6b_cybertown_sft
```

```bash
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
  --model "$BASE_MODEL" \
  --served-model-name qwen3_0_6b_base \
  --enable-lora \
  --lora-modules "${SERVED_MODEL_NAME}=${SFT_ADAPTER_DIR}" \
  --max-lora-rank 32 \
  --host "${VLLM_HOST:-127.0.0.1}" \
  --port "${VLLM_PORT:-8001}" \
  --max-model-len 4096 \
  --max-num-seqs "$VLLM_MAX_NUM_SEQS" \
  --dtype bfloat16 \
  --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
  --trust-remote-code \
  --no-enable-log-requests
```

## 启动 RLVR LoRA Adapter

适用于 VeRL/RLVR 输出的 actor LoRA checkpoint：

```bash
export BASE_MODEL=WindyLab/Qwen3-0.6B-cybertown-SFT
export RLVR_ADAPTER_DIR=/GSI/outputs/rlvr_gsi/qwen3_0_6b_cybertown_rlvr/global_step_350/actor/lora_adapter
export SERVED_MODEL_NAME=qwen3_0_6b_cybertown_rlvr
```

```bash
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
  --model "$BASE_MODEL" \
  --served-model-name qwen3_0_6b_cybertown_sft_base \
  --enable-lora \
  --lora-modules "${SERVED_MODEL_NAME}=${RLVR_ADAPTER_DIR}" \
  --max-lora-rank 32 \
  --host "${VLLM_HOST:-127.0.0.1}" \
  --port "${VLLM_PORT:-8001}" \
  --max-model-len 4096 \
  --max-num-seqs "$VLLM_MAX_NUM_SEQS" \
  --dtype bfloat16 \
  --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
  --trust-remote-code \
  --no-enable-log-requests
```

## 合并 LoRA

```bash
python -m llm_finetune.sft.merge_lora \
  --base-model "$BASE_MODEL" \
  --adapter-path "$SFT_ADAPTER_DIR" \
  --output-dir /GSI/outputs/sft/qwen3_0_6b_cybertown_sft_merged
```

RLVR adapter 也可使用同一命令，前提是 `--base-model` 指向训练该 adapter 时的 base。

## 检查服务

vLLM 启动后，原终端会被服务占用。另开容器 shell：

```bash
cd /path/to/GSI
docker compose -f docker/docker-compose.runtime-train.yml run --rm gsi-train bash
```

检查模型列表：

```bash
curl -fsS http://127.0.0.1:8001/v1/models | python -m json.tool
```

常见启动问题：

- 端口 `8001` 已被占用。
- `MODEL_PATH`、`BASE_MODEL` 或 adapter 目录不存在。
- LoRA rank 大于 `--max-lora-rank`。
- `--max-model-len`、`--max-num-seqs` 或显存比例过高。

## 配置 GSI

```bash
export GSI_LLM_API_BASE=http://127.0.0.1:8001/v1
export GSI_LLM_API_KEY=EMPTY
export GSI_LLM_MODEL=qwen3_0_6b_cybertown_sft
export GSI_DISABLE_TOKEN_STATS=1
export NO_PROXY=127.0.0.1,localhost,${NO_PROXY:-}
export no_proxy=127.0.0.1,localhost,${no_proxy:-}
```

`GSI_LLM_MODEL` 必须等于 vLLM 暴露的模型名。LoRA 模式下对应 `--lora-modules` 左侧名称；完整模型模式下对应 `--served-model-name`。

配置任务分配 solver：

```bash
export GSI_TANGO_SOLVER_BACKEND=scip
export GSI_TANGO_SOLVER_MAX_TIME=120
```

## 准备 Benchmark 数据

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
```

检查：

```bash
find "$GSI_DATASET_ROOT" -maxdepth 5 \( -name tasks.jsonl -o -name goals.jsonl \) -print
```

## 采集 Replan 数据

```bash
python run/run_collect_replan_dataset.py \
  --batch-name eval_qwen3_0_6b_cybertown_nc4 \
  --output-root outputs/replan_collect \
  --dataset-root "$GSI_DATASET_ROOT" \
  --max-count 100 \
  --max-workers 10 \
  --max-newcases-per-run 4 \
  --capture-save-llm-io \
  --capture-max-records-per-run 4 \
  --capture-data-tag eval_qwen3_0_6b_cybertown_nc4
```

## 运行 Benchmark

默认评估 SGI：

```bash
python run/run_exp_multi_method.py \
  --methods sgi \
  --enable-newcase \
  --newcase-counts 4 \
  --max-count 100 \
  --max-workers 10 \
  --dataset-root "$GSI_DATASET_ROOT" \
  --output-root outputs/eval_multi_method/qwen3_0_6b_cybertown
```

基础模型 baseline 对比：

```bash
python run/run_exp_multi_method.py \
  --methods sgi spine smartllm lipllm \
  --enable-newcase \
  --newcase-counts 4 \
  --max-count 100 \
  --max-workers 10 \
  --dataset-root "$GSI_DATASET_ROOT" \
  --output-root outputs/eval_multi_method/qwen3_0_6b_cybertown_baselines
```

## 查看结果

输出通常位于：

```text
outputs/eval_multi_method/qwen3_0_6b_cybertown/<timestamp>/newcase_*/batch_<method>/
```

优先查看 `aggregate_full.json`，再用 `summary.jsonl` 和单任务目录定位失败样本。字段说明见 [输出目录](outputs.md)。
