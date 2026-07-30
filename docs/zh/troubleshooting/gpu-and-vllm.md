# GPU 与 vLLM

本页用于排查 GPU 选择、显存不足、vLLM 启动失败和常见 `EngineDeadError`。

## GPU 选择

训练脚本中建议同时设置：

```bash
CUDA_VISIBLE_DEVICES=1 \
TRAIN_CUDA_VISIBLE_DEVICES=1 \
./scripts/runtime/train_rlvr.sh
```

Ray、VeRL 和 vLLM 可能读取不同变量。只设置一个变量时，实际使用的 GPU 可能与预期不一致。

## 查看显存

```bash
nvidia-smi
```

RLVR 训练期间，rollout、old log prob、actor update 和 weight sync 阶段的显存占用会波动。判断是否跑在指定 GPU 上，应以宿主机 `nvidia-smi` 的进程列表为准。

## vLLM EngineDeadError

常见表现：

```text
vllm.v1.engine.exceptions.EngineDeadError
SystemError: attempting to create PyCFunction with class but no METH_METHOD flag
```

这通常表示 vLLM engine 子进程崩溃。处理顺序：

1. 降低 `VLLM_MAX_NUM_BATCHED_TOKENS`。
2. 降低 `VERL_ROLLOUT_GPU_MEMORY_UTILIZATION`。
3. 降低 `VERL_ROLLOUT_N`。
4. 保持 `actor_rollout_ref.rollout.enforce_eager=True`。
5. 清理旧 Ray/vLLM 进程后重启。

## vLLM 预检查

仓库根目录：

```bash
CUDA_VISIBLE_DEVICES=0 ./llm_finetune/scripts/runtime/check_vllm_gpu.sh
```

`/GSI/llm_finetune` 目录：

```bash
CUDA_VISIBLE_DEVICES=0 ./scripts/runtime/check_vllm_gpu.sh
```

启动命令见 [vLLM 启动与 GSI 评估](../training/vllm-eval.md)。
