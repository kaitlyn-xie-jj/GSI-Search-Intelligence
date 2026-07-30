# RLVR 训练

本页说明如何在训练容器内启动 RLVR。RLVR 通常应从 SFT 后模型开始，而不是直接从 base model 开始。

## 前置检查

```bash
cd /GSI/llm_finetune
export ROOT_DIR=/GSI
./scripts/runtime/check_env.sh
```

训练前先准备模型、数据和 Hugging Face cache，见 [Hugging Face 准备](huggingface-prepare.md)。

## 模型与数据要求

推荐基础模型来源：

- 本地 SFT merged 模型，例如 `/GSI/outputs/sft/qwen3_0_6b_cybertown_sft_merged`。
- Hugging Face 上的 SFT checkpoint，例如 `WindyLab/Qwen3-0.6B-cybertown-SFT`。

如果 SFT 只保存了 `lora_adapters/`，建议先合并成完整模型，再作为 `RLVR_MODEL_PATH`。

`RLVR_DATA_DIR` 必须包含：

```text
train.parquet
val.parquet
states.index.json
states/*.jsonl
```

使用 Hugging Face 数据集时，wrapper 会将 `GSI_REPLAN_STATE_ROOT` 指向解析后的数据 snapshot。

## 常规 RLVR

以下配置用于单卡 24GB GPU 的 smoke 或初次验证：

```bash
export RLVR_OUTPUT_DIR=/GSI/outputs/rlvr_gsi/qwen3_0_6b_cybertown_rlvr
export GSI_TANGO_SOLVER_BACKEND=scip
export TENSORBOARD_DIR=/GSI/outputs/rlvr_gsi/qwen3_0_6b_cybertown_rlvr/tensorboard

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
VERL_PPO_MICRO_BATCH_SIZE_PER_GPU=1 \
VERL_ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=1 \
VERL_REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=1 \
VERL_ROLLOUT_N=2 \
VERL_MAX_PROMPT_LENGTH=4096 \
VERL_MAX_RESPONSE_LENGTH=2048 \
VERL_ROLLOUT_GPU_MEMORY_UTILIZATION=0.35 \
VERL_SAVE_FREQ=200 \
VERL_TEST_FREQ=200 \
VLLM_MAX_MODEL_LEN=6144 \
VLLM_MAX_NUM_BATCHED_TOKENS=6144 \
./scripts/runtime/train_rlvr.sh \
  trainer.logger='["console","tensorboard"]' \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.free_cache_engine=False \
  2>&1 | tee /GSI/outputs/rlvr_gsi/qwen3_0_6b_cybertown_rlvr/train.log
```

默认值包括：

- `RLVR_MODEL_REPO_ID=WindyLab/Qwen3-0.6B-cybertown-SFT`
- `RLVR_DATA_REPO_ID=WindyLab/Qwen3-0.6B-cybertown-RLVR-data`
- `RLVR_STATE_STORE_NAME=train6_val10_initial30_replan70_val15_20260528`
- `RLVR_OUTPUT_DIR=/GSI/outputs/rlvr_gsi/qwen3_0_6b_cybertown_rlvr`
- `GSI_TANGO_SOLVER_BACKEND=scip`
- `GSI_DISABLE_TOKEN_STATS=1`

## 4090 / 4090D 推荐配置

如果 smoke 配置稳定，可提高 batch 和 rollout：

```bash
export RLVR_OUTPUT_DIR=/GSI/outputs/rlvr_gsi/qwen3_0_6b_cybertown_rlvr_bsz8
export GSI_TANGO_SOLVER_BACKEND=scip
export TENSORBOARD_DIR=/GSI/outputs/rlvr_gsi/qwen3_0_6b_cybertown_rlvr_bsz8/tensorboard

CUDA_VISIBLE_DEVICES=1 \
TRAIN_CUDA_VISIBLE_DEVICES=1 \
RLVR_MODEL_REPO_ID=WindyLab/Qwen3-0.6B-cybertown-SFT \
RLVR_DATA_REPO_ID=WindyLab/Qwen3-0.6B-cybertown-RLVR-data \
RLVR_STATE_STORE_NAME=train6_val10_initial30_replan70_val15_20260528 \
RLVR_OUTPUT_DIR=$RLVR_OUTPUT_DIR \
GSI_PREFLIGHT=1 \
GSI_PREFLIGHT_STOP_RAY=1 \
GSI_PREFLIGHT_MIN_GPU_FREE_MB=18000 \
VERL_N_GPUS_PER_NODE=1 \
VERL_TRAIN_BATCH_SIZE=8 \
VERL_PPO_MINI_BATCH_SIZE=4 \
VERL_PPO_MICRO_BATCH_SIZE_PER_GPU=1 \
VERL_ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=1 \
VERL_REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=1 \
VERL_ROLLOUT_N=4 \
VERL_MAX_PROMPT_LENGTH=4096 \
VERL_MAX_RESPONSE_LENGTH=2048 \
VERL_ROLLOUT_GPU_MEMORY_UTILIZATION=0.45 \
VERL_SAVE_FREQ=200 \
VERL_TEST_FREQ=200 \
VLLM_MAX_MODEL_LEN=6144 \
VLLM_MAX_NUM_BATCHED_TOKENS=12288 \
./scripts/runtime/train_rlvr.sh \
  trainer.logger='["console","tensorboard"]' \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.free_cache_engine=False \
  2>&1 | tee /GSI/outputs/rlvr_gsi/qwen3_0_6b_cybertown_rlvr_bsz8/train.log
```

该配置每步生成 `8 * 4 = 32` 条 response。显存不足时，优先降低 `VLLM_MAX_NUM_BATCHED_TOKENS`、`VERL_ROLLOUT_GPU_MEMORY_UTILIZATION` 或 `VERL_ROLLOUT_N`。

## 使用本地路径

```bash
TRAIN_CUDA_VISIBLE_DEVICES=0 \
RLVR_MODEL_PATH=/GSI/outputs/sft/qwen3_0_6b_cybertown_sft_merged \
RLVR_DATA_DIR=/GSI/data/rlvr_gsi/replan_snapshot \
GSI_REPLAN_STATE_ROOT=/GSI/data/rlvr_gsi \
RLVR_OUTPUT_DIR=/GSI/outputs/rlvr_gsi/qwen3_0_6b_local_sft_rlvr \
VERL_N_GPUS_PER_NODE=1 \
VERL_TRAIN_BATCH_SIZE=4 \
VERL_PPO_MINI_BATCH_SIZE=2 \
VERL_ROLLOUT_N=2 \
VERL_ROLLOUT_GPU_MEMORY_UTILIZATION=0.35 \
./scripts/runtime/train_rlvr.sh
```

优先级：

- `RLVR_MODEL_PATH` 优先于 `RLVR_MODEL_REPO_ID`。
- `RLVR_DATA_DIR` 优先于 `RLVR_DATA_REPO_ID`。
- `RLVR_OUTPUT_DIR` 控制输出目录。

## Validator

`train_rlvr.sh` 默认先启动本地 plan validator，再启动 VeRL：

```bash
./scripts/runtime/serve_validator.sh start
./scripts/runtime/serve_validator.sh status
./scripts/runtime/serve_validator.sh logs
./scripts/runtime/serve_validator.sh stop
```

默认地址：

```bash
export GSI_VALIDATOR_HOST=127.0.0.1
export GSI_VALIDATOR_PORT=8000
export GSI_VALIDATOR_URL=http://127.0.0.1:8000/validate
export GSI_VALIDATOR_BATCH_URL=http://127.0.0.1:8000/validate_batch
```

如果已有 validator 服务，可设置：

```bash
export GSI_START_VALIDATOR=0
```

## Preflight

建议正式训练前打开：

```bash
TRAIN_CUDA_VISIBLE_DEVICES=0 \
GSI_PREFLIGHT=1 \
GSI_PREFLIGHT_MIN_GPU_FREE_MB=18000 \
GSI_PREFLIGHT_MIN_OUTPUT_FREE_GB=100 \
VERL_N_GPUS_PER_NODE=1 \
RLVR_MODEL_REPO_ID=WindyLab/Qwen3-0.6B-cybertown-SFT \
RLVR_DATA_REPO_ID=WindyLab/Qwen3-0.6B-cybertown-RLVR-data \
RLVR_OUTPUT_DIR=/GSI/outputs/rlvr_gsi/qwen3_0_6b_cybertown_rlvr \
./scripts/runtime/train_rlvr.sh
```

Preflight 会检查 GPU、磁盘空间、旧进程、validator、state shards 和样本 state 加载。清理旧 Ray runtime 可设置 `GSI_PREFLIGHT_STOP_RAY=1`。

## Checkpoint 恢复

`RLVR_OUTPUT_DIR` 会传给 VeRL 的 `trainer.default_local_dir`。VeRL 默认 `trainer.resume_mode=auto`，目录中已有 `global_step_*` 时会自动恢复。

恢复日志示例：

```text
Found checkpoint: .../global_step_1800
Load from checkpoint folder: .../global_step_1800
Setting global step to 1800
Resuming from .../global_step_1800
```

修改 batch、rollout、epoch 或数据集后，建议使用新的 `RLVR_OUTPUT_DIR`。强制忽略旧 checkpoint：

```bash
trainer.resume_mode=disable
```

## 常用参数

- `TRAIN_CUDA_VISIBLE_DEVICES`：训练可见 GPU。
- `VERL_N_GPUS_PER_NODE`：每节点 GPU 数。
- `VERL_TRAIN_BATCH_SIZE`：每 step prompt 数。
- `VERL_PPO_MINI_BATCH_SIZE`：PPO mini batch。
- `VERL_PPO_MICRO_BATCH_SIZE_PER_GPU`：actor update micro batch。
- `VERL_ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU`：old log prob micro batch。
- `VERL_REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU`：reference log prob micro batch。
- `VERL_ROLLOUT_N`：每条 prompt 的 response 数。
- `VERL_MAX_PROMPT_LENGTH`：最大 prompt 长度。
- `VERL_MAX_RESPONSE_LENGTH`：最大 response 长度。
- `VERL_ROLLOUT_GPU_MEMORY_UTILIZATION`：rollout vLLM 显存比例。
- `VLLM_MAX_MODEL_LEN`：rollout vLLM 最大上下文长度。
- `VLLM_MAX_NUM_BATCHED_TOKENS`：rollout vLLM batch token 上限。
- `VERL_SAVE_FREQ`：checkpoint 保存频率。
- `VERL_TEST_FREQ`：validation/test 频率。

## Smoke Test

```bash
VERL_TOTAL_EPOCHS=1 \
VERL_SAVE_FREQ=1 \
VERL_TEST_FREQ=1 \
VERL_TRAIN_BATCH_SIZE=2 \
VERL_PPO_MINI_BATCH_SIZE=1 \
VERL_ROLLOUT_N=1 \
RLVR_MODEL_REPO_ID=WindyLab/Qwen3-0.6B-cybertown-SFT \
RLVR_DATA_REPO_ID=WindyLab/Qwen3-0.6B-cybertown-RLVR-data \
RLVR_OUTPUT_DIR=/GSI/outputs/rlvr_gsi/qwen3_0_6b_cybertown_rlvr_smoke \
./scripts/runtime/train_rlvr.sh
```

## 日志与输出

TensorBoard：

```bash
tensorboard --logdir "$TENSORBOARD_DIR" \
  --host 0.0.0.0 \
  --port 6006
```

训练日志：

```bash
tail -f /GSI/outputs/rlvr_gsi/qwen3_0_6b_cybertown_rlvr/train.log
```

Validator 日志：

```bash
./scripts/runtime/serve_validator.sh logs
tail -f /GSI/outputs/logs/gsi_validator_8000.log
```

训练完成后，可用 actor LoRA adapter 启动 vLLM，或先合并再启动。评估流程见 [vLLM 启动与 GSI 评估](vllm-eval.md)。
