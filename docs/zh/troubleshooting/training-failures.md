# 训练失败

训练失败通常来自环境、数据、显存、validator 或 checkpoint。排查时应先确认失败阶段，再调整参数。

## 环境检查

```bash
cd /GSI/llm_finetune
./scripts/runtime/check_env.sh
```

确认依赖、GPU、脚本路径和 Hugging Face cache 均可用。

## 数据检查

SFT 需要训练 JSONL；RLVR 需要 parquet、state index 和 state shards。缺少 state 文件会导致 validator 或 reward 失败。

RLVR 结构：

```text
train.parquet
val.parquet
states.index.json
states/*.jsonl
```

## 显存问题

优先降低：

- `VLLM_MAX_NUM_BATCHED_TOKENS`
- `VERL_ROLLOUT_GPU_MEMORY_UTILIZATION`
- `VERL_ROLLOUT_N`
- micro batch 相关参数

GPU 排查见 [GPU 与 vLLM](gpu-and-vllm.md)。

## Validator 问题

检查服务状态：

```bash
./scripts/runtime/serve_validator.sh status
./scripts/runtime/serve_validator.sh logs
```

如果 validator 频繁 timeout，降低 rollout 并发，或先抽样验证单条 state。

## Checkpoint 问题

如果旧目录中已有 `global_step_*`，VeRL 默认会自动恢复。新实验应使用新的 `RLVR_OUTPUT_DIR`，或显式设置：

```bash
trainer.resume_mode=disable
```

更多说明见 [Ray 与 Checkpoint](ray-and-checkpoint.md)。
