# RLVR Training

This page explains how to start RLVR inside the training container. RLVR should usually start from an SFT model rather than a base model.

## Preflight

```bash
cd /GSI/llm_finetune
export ROOT_DIR=/GSI
./scripts/runtime/check_env.sh
```

Prepare models, data, and Hugging Face cache first. See [Hugging Face Preparation](huggingface-prepare.md).

## Model and Data Requirements

Recommended model sources:

- Local SFT merged model, such as `/GSI/outputs/sft/qwen3_0_6b_cybertown_sft_merged`.
- SFT checkpoint on Hugging Face, such as `WindyLab/Qwen3-0.6B-cybertown-SFT`.

If SFT only produced `lora_adapters/`, merge it into a full model before using it as `RLVR_MODEL_PATH`.

`RLVR_DATA_DIR` must contain:

```text
train.parquet
val.parquet
states.index.json
states/*.jsonl
```

When using a Hugging Face dataset, the wrapper points `GSI_REPLAN_STATE_ROOT` to the resolved data snapshot.

## Standard RLVR

This configuration is a single-GPU 24GB smoke or first-run setup:

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

Defaults include:

- `RLVR_MODEL_REPO_ID=WindyLab/Qwen3-0.6B-cybertown-SFT`
- `RLVR_DATA_REPO_ID=WindyLab/Qwen3-0.6B-cybertown-RLVR-data`
- `RLVR_STATE_STORE_NAME=train6_val10_initial30_replan70_val15_20260528`
- `RLVR_OUTPUT_DIR=/GSI/outputs/rlvr_gsi/qwen3_0_6b_cybertown_rlvr`
- `GSI_TANGO_SOLVER_BACKEND=scip`
- `GSI_DISABLE_TOKEN_STATS=1`

## Recommended 4090 / 4090D Configuration

If the smoke configuration is stable, increase batch and rollout:

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

This configuration generates `8 * 4 = 32` responses per step. If memory is insufficient, first lower `VLLM_MAX_NUM_BATCHED_TOKENS`, `VERL_ROLLOUT_GPU_MEMORY_UTILIZATION`, or `VERL_ROLLOUT_N`.

## Use Local Paths

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

Priority:

- `RLVR_MODEL_PATH` overrides `RLVR_MODEL_REPO_ID`.
- `RLVR_DATA_DIR` overrides `RLVR_DATA_REPO_ID`.
- `RLVR_OUTPUT_DIR` controls the output directory.

## Validator

`train_rlvr.sh` starts the local plan validator before VeRL by default:

```bash
./scripts/runtime/serve_validator.sh start
./scripts/runtime/serve_validator.sh status
./scripts/runtime/serve_validator.sh logs
./scripts/runtime/serve_validator.sh stop
```

Default addresses:

```bash
export GSI_VALIDATOR_HOST=127.0.0.1
export GSI_VALIDATOR_PORT=8000
export GSI_VALIDATOR_URL=http://127.0.0.1:8000/validate
export GSI_VALIDATOR_BATCH_URL=http://127.0.0.1:8000/validate_batch
```

To reuse an existing validator:

```bash
export GSI_START_VALIDATOR=0
```

## Preflight

Enable before formal training:

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

Preflight checks GPU, disk space, old processes, validator, state shards, and sample state loading. Set `GSI_PREFLIGHT_STOP_RAY=1` to stop old Ray runtime.

## Checkpoint Resume

`RLVR_OUTPUT_DIR` is passed to VeRL as `trainer.default_local_dir`. VeRL uses `trainer.resume_mode=auto` by default, so directories with `global_step_*` resume automatically.

Resume log example:

```text
Found checkpoint: .../global_step_1800
Load from checkpoint folder: .../global_step_1800
Setting global step to 1800
Resuming from .../global_step_1800
```

After changing batch, rollout, epoch, or dataset settings, use a new `RLVR_OUTPUT_DIR`. To ignore old checkpoints:

```bash
trainer.resume_mode=disable
```

## Common Parameters

- `TRAIN_CUDA_VISIBLE_DEVICES`: visible GPU for training.
- `VERL_N_GPUS_PER_NODE`: GPUs per node.
- `VERL_TRAIN_BATCH_SIZE`: prompts per step.
- `VERL_PPO_MINI_BATCH_SIZE`: PPO mini batch.
- `VERL_PPO_MICRO_BATCH_SIZE_PER_GPU`: actor update micro batch.
- `VERL_ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU`: old log prob micro batch.
- `VERL_REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU`: reference log prob micro batch.
- `VERL_ROLLOUT_N`: responses per prompt.
- `VERL_MAX_PROMPT_LENGTH`: maximum prompt length.
- `VERL_MAX_RESPONSE_LENGTH`: maximum response length.
- `VERL_ROLLOUT_GPU_MEMORY_UTILIZATION`: rollout vLLM memory ratio.
- `VLLM_MAX_MODEL_LEN`: rollout vLLM maximum context length.
- `VLLM_MAX_NUM_BATCHED_TOKENS`: rollout vLLM batch token limit.
- `VERL_SAVE_FREQ`: checkpoint save frequency.
- `VERL_TEST_FREQ`: validation/test frequency.

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

## Logs and Outputs

TensorBoard:

```bash
tensorboard --logdir "$TENSORBOARD_DIR" \
  --host 0.0.0.0 \
  --port 6006
```

Training log:

```bash
tail -f /GSI/outputs/rlvr_gsi/qwen3_0_6b_cybertown_rlvr/train.log
```

Validator log:

```bash
./scripts/runtime/serve_validator.sh logs
tail -f /GSI/outputs/logs/gsi_validator_8000.log
```

After training, serve the actor LoRA adapter with vLLM or merge it first. Evaluation is covered in [vLLM Startup and GSI Evaluation](vllm-eval.md).
