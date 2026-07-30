# Training Failures

Training failures usually come from environment, data, memory, validator, or checkpoint issues. Identify the failing stage before adjusting parameters.

## Environment Check

```bash
cd /GSI/llm_finetune
./scripts/runtime/check_env.sh
```

Confirm dependencies, GPU visibility, script paths, and Hugging Face cache.

## Data Check

SFT requires training JSONL. RLVR requires parquet files, state index, and state shards. Missing state files can cause validator or reward failures.

RLVR structure:

```text
train.parquet
val.parquet
states.index.json
states/*.jsonl
```

## Memory Issues

Lower these first:

- `VLLM_MAX_NUM_BATCHED_TOKENS`
- `VERL_ROLLOUT_GPU_MEMORY_UTILIZATION`
- `VERL_ROLLOUT_N`
- micro batch parameters

GPU troubleshooting is in [GPU and vLLM](gpu-and-vllm.md).

## Validator Issues

Check service status:

```bash
./scripts/runtime/serve_validator.sh status
./scripts/runtime/serve_validator.sh logs
```

If validator timeout is frequent, lower rollout concurrency or validate a sampled state first.

## Checkpoint Issues

If the output directory already contains `global_step_*`, VeRL resumes automatically by default. Use a new `RLVR_OUTPUT_DIR` for new experiments, or set:

```bash
trainer.resume_mode=disable
```

More details are in [Ray and Checkpoint](ray-and-checkpoint.md).
