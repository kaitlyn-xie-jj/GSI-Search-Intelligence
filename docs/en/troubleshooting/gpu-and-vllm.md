# GPU and vLLM

This page covers GPU selection, insufficient memory, vLLM startup failures, and common `EngineDeadError` cases.

## GPU Selection

Set both variables in training scripts:

```bash
CUDA_VISIBLE_DEVICES=1 \
TRAIN_CUDA_VISIBLE_DEVICES=1 \
./scripts/runtime/train_rlvr.sh
```

Ray, VeRL, and vLLM may read different variables. Setting only one variable can make the actual GPU differ from the intended GPU.

## Check Memory

```bash
nvidia-smi
```

During RLVR training, memory usage changes across rollout, old log prob, actor update, and weight sync stages. Use the host `nvidia-smi` process list to confirm which physical GPU is active.

## vLLM EngineDeadError

Common symptoms:

```text
vllm.v1.engine.exceptions.EngineDeadError
SystemError: attempting to create PyCFunction with class but no METH_METHOD flag
```

This usually means the vLLM engine subprocess crashed. Handle it in this order:

1. Lower `VLLM_MAX_NUM_BATCHED_TOKENS`.
2. Lower `VERL_ROLLOUT_GPU_MEMORY_UTILIZATION`.
3. Lower `VERL_ROLLOUT_N`.
4. Keep `actor_rollout_ref.rollout.enforce_eager=True`.
5. Stop old Ray/vLLM processes and restart.

## vLLM Preflight

Repository root:

```bash
CUDA_VISIBLE_DEVICES=0 ./llm_finetune/scripts/runtime/check_vllm_gpu.sh
```

From `/GSI/llm_finetune`:

```bash
CUDA_VISIBLE_DEVICES=0 ./scripts/runtime/check_vllm_gpu.sh
```

Startup commands are in [vLLM Startup and GSI Evaluation](../training/vllm-eval.md).
