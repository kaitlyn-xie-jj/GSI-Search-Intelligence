# 评估模型

训练后的模型通常先通过 vLLM 暴露为 OpenAI-compatible endpoint，再由 GSI benchmark 调用。

## 1. 启动 vLLM

完整说明见 [vLLM 启动与 GSI 评估](../training/vllm-eval.md)。示例：

```bash
export SERVED_MODEL_NAME=qwen3_0_6b_cybertown_rlvr
export MODEL_PATH=WindyLab/Qwen3-0.6B-cybertown-RLVR

CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --host 127.0.0.1 \
  --port 8001 \
  --max-model-len 4096 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.75 \
  --trust-remote-code \
  --no-enable-log-requests
```

## 2. 配置 GSI

```bash
export GSI_LLM_API_BASE=http://127.0.0.1:8001/v1
export GSI_LLM_API_KEY=EMPTY
export GSI_LLM_MODEL=qwen3_0_6b_cybertown_rlvr
export GSI_DISABLE_TOKEN_STATS=1
export GSI_TANGO_SOLVER_BACKEND=scip
```

`GSI_LLM_MODEL` 必须等于 vLLM 暴露的模型名。

## 3. 运行 Benchmark

```bash
python run/run_exp_multi_method.py \
  --methods sgi \
  --max-count 100 \
  --dataset-root "$GSI_DATASET_ROOT" \
  --output-root results/eval_model
```

## 4. 查看结果

输出文件解释见 [输出目录](../training/outputs.md)。GPU 或 vLLM 问题见 [GPU 与 vLLM](../troubleshooting/gpu-and-vllm.md)。
