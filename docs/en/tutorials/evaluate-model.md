# Evaluate a Model

Trained models are usually exposed through vLLM as an OpenAI-compatible endpoint and then called by GSI benchmark.

## 1. Start vLLM

Full instructions are in [vLLM Startup and GSI Evaluation](../training/vllm-eval.md). Example:

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

## 2. Configure GSI

```bash
export GSI_LLM_API_BASE=http://127.0.0.1:8001/v1
export GSI_LLM_API_KEY=EMPTY
export GSI_LLM_MODEL=qwen3_0_6b_cybertown_rlvr
export GSI_DISABLE_TOKEN_STATS=1
export GSI_TANGO_SOLVER_BACKEND=scip
```

`GSI_LLM_MODEL` must match the model name exposed by vLLM.

## 3. Run Benchmark

```bash
python run/run_exp_multi_method.py \
  --methods sgi \
  --max-count 100 \
  --dataset-root "$GSI_DATASET_ROOT" \
  --output-root results/eval_model
```

## 4. Inspect Results

Output files are described in [Output Directory](../training/outputs.md). GPU and vLLM issues are covered in [GPU and vLLM](../troubleshooting/gpu-and-vllm.md).
