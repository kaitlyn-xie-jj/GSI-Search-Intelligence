# 环境变量

GSI 的运行配置由 CLI 参数、JSON 配置和环境变量共同决定。模型服务、数据集路径、solver 后端和训练参数通常通过环境变量显式设置。

## LLM Endpoint

```bash
export GSI_LLM_API_BASE=http://127.0.0.1:8001/v1
export GSI_LLM_API_KEY=EMPTY
export GSI_LLM_MODEL=qwen3_0_6b_cybertown_rlvr
export GSI_DISABLE_TOKEN_STATS=1
```

`GSI_LLM_MODEL` 必须与 vLLM `/v1/models` 返回的 id 一致。

## Solver

```bash
export GSI_TANGO_SOLVER_BACKEND=scip
export GSI_TANGO_SOLVER_MAX_TIME=120
```

可选后端包括 `scip` 和 `gurobi`。使用 `gurobi` 需要有效 license。

## 数据与缓存

```bash
export GSI_DATASET_ROOT=/path/to/gsi/dataset
export HF_HOME=$PWD/hf_cache
export HF_HUB_CACHE=$HF_HOME/hub
```

离线运行：

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

常用参数包括 `SFT_MAX_SEQ_LENGTH`、`SFT_PER_DEVICE_BATCH_SIZE`、`SFT_GRAD_ACCUM`、`SFT_EPOCHS` 和 `SFT_LR`。

## RLVR

```bash
export RLVR_MODEL_REPO_ID=WindyLab/Qwen3-0.6B-cybertown-SFT
export RLVR_DATA_REPO_ID=WindyLab/Qwen3-0.6B-cybertown-RLVR-data
export RLVR_OUTPUT_DIR=/GSI/outputs/rlvr_gsi/qwen3_0_6b_cybertown_rlvr
export TRAIN_CUDA_VISIBLE_DEVICES=0
```

常用参数包括 `VERL_TRAIN_BATCH_SIZE`、`VERL_ROLLOUT_N`、`VERL_MAX_PROMPT_LENGTH`、`VERL_MAX_RESPONSE_LENGTH` 和 `VERL_ROLLOUT_GPU_MEMORY_UTILIZATION`。

## vLLM

```bash
export VLLM_HOST=127.0.0.1
export VLLM_PORT=8001
export VLLM_GPU_MEMORY_UTILIZATION=0.75
export VLLM_MAX_NUM_SEQS=64
```

启动和评估流程见 [vLLM 启动与 GSI 评估](../training/vllm-eval.md)。
