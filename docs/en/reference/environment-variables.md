# Environment Variables

GSI runtime configuration comes from CLI arguments, JSON config, and environment variables. Model services, dataset paths, solver backends, and training parameters are usually set explicitly through environment variables.

## LLM Endpoint

```bash
export GSI_LLM_API_BASE=http://127.0.0.1:8001/v1
export GSI_LLM_API_KEY=EMPTY
export GSI_LLM_MODEL=qwen3_0_6b_cybertown_rlvr
export GSI_DISABLE_TOKEN_STATS=1
```

`GSI_LLM_MODEL` must match the id returned by vLLM `/v1/models`.

## Solver

```bash
export GSI_TANGO_SOLVER_BACKEND=scip
export GSI_TANGO_SOLVER_MAX_TIME=120
```

Available backends include `scip` and `gurobi`. `gurobi` requires a valid license.

## Data and Cache

```bash
export GSI_DATASET_ROOT=/path/to/gsi/dataset
export HF_HOME=$PWD/hf_cache
export HF_HUB_CACHE=$HF_HOME/hub
```

Offline mode:

```bash
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

## SFT

```bash
export SFT_MODEL_REPO_ID=Qwen/Qwen3-0.6B
export SFT_DATA_REPO_ID=WindyLab/Qwen3-0.6B-cybertown-SFT-data
export SFT_OUTPUT_DIR=/GSI/outputs/sft/qwen3_0_6b_cybertown_sft
```

Common parameters include `SFT_MAX_SEQ_LENGTH`, `SFT_PER_DEVICE_BATCH_SIZE`, `SFT_GRAD_ACCUM`, `SFT_EPOCHS`, and `SFT_LR`.

## RLVR

```bash
export RLVR_MODEL_REPO_ID=WindyLab/Qwen3-0.6B-cybertown-SFT
export RLVR_DATA_REPO_ID=WindyLab/Qwen3-0.6B-cybertown-RLVR-data
export RLVR_OUTPUT_DIR=/GSI/outputs/rlvr_gsi/qwen3_0_6b_cybertown_rlvr
export TRAIN_CUDA_VISIBLE_DEVICES=0
```

Common parameters include `VERL_TRAIN_BATCH_SIZE`, `VERL_ROLLOUT_N`, `VERL_MAX_PROMPT_LENGTH`, `VERL_MAX_RESPONSE_LENGTH`, and `VERL_ROLLOUT_GPU_MEMORY_UTILIZATION`.

## vLLM

```bash
export VLLM_HOST=127.0.0.1
export VLLM_PORT=8001
export VLLM_GPU_MEMORY_UTILIZATION=0.75
export VLLM_MAX_NUM_SEQS=64
```

Startup and evaluation are covered in [vLLM Startup and GSI Evaluation](../training/vllm-eval.md).
