# SFT Training

This page explains how to start SFT inside the training container.

## Preflight

```bash
cd /GSI/llm_finetune
export ROOT_DIR=/GSI
./scripts/runtime/check_env.sh
```

Prepare Hugging Face cache and data first. See [Hugging Face Preparation](huggingface-prepare.md).

## Standard Unsloth SFT

```bash
SFT_MODEL_REPO_ID=Qwen/Qwen3-0.6B \
SFT_DATA_REPO_ID=WindyLab/Qwen3-0.6B-cybertown-SFT-data \
SFT_OUTPUT_DIR=/GSI/outputs/sft/qwen3_0_6b_cybertown_sft \
SFT_MAX_SEQ_LENGTH=4096 \
SFT_PER_DEVICE_BATCH_SIZE=1 \
SFT_GRAD_ACCUM=8 \
SFT_EPOCHS=1 \
./scripts/runtime/train_sft_unsloth.sh
```

Defaults:

- `SFT_MODEL_REPO_ID=Qwen/Qwen3-0.6B`
- `SFT_DATA_REPO_ID=WindyLab/Qwen3-0.6B-cybertown-SFT-data`
- `SFT_DATA_FILENAME=train.jsonl`
- `SFT_TEMPLATE_PATH=/GSI/llm_finetune/chat_template/qwen3_nonthinking.jinja`
- `SFT_LOAD_IN_4BIT=1`

## GPU Selection

The Unsloth wrapper exposes one GPU by default. Use the second GPU:

```bash
SFT_CUDA_VISIBLE_DEVICES=1 \
SFT_MODEL_REPO_ID=Qwen/Qwen3-0.6B \
SFT_DATA_REPO_ID=WindyLab/Qwen3-0.6B-cybertown-SFT-data \
SFT_OUTPUT_DIR=/GSI/outputs/sft/qwen3_0_6b_cybertown_sft_gpu1 \
./scripts/runtime/train_sft_unsloth.sh
```

Only set this when debugging Unsloth/HF Trainer multi-GPU behavior:

```bash
SFT_UNSLOTH_ALLOW_MULTI_GPU=1
```

## Use Local Paths

```bash
SFT_MODEL_PATH=/models/Qwen3-4B \
SFT_DATA_PATH=/GSI/data/sft/replan_curated/train.jsonl \
SFT_OUTPUT_DIR=/GSI/outputs/sft/qwen3_4b_replan_curated \
SFT_MAX_SEQ_LENGTH=4096 \
SFT_PER_DEVICE_BATCH_SIZE=1 \
SFT_GRAD_ACCUM=8 \
SFT_EPOCHS=1 \
./scripts/runtime/train_sft_unsloth.sh
```

Priority:

- `SFT_MODEL_PATH` or `MODEL_PATH` overrides `SFT_MODEL_REPO_ID`.
- `SFT_DATA_PATH` or `DATA_PATH` overrides `SFT_DATA_REPO_ID`.
- `SFT_OUTPUT_DIR` or `OUTPUT_DIR` controls the output directory.

## Common Parameters

- `SFT_MAX_SEQ_LENGTH`: maximum sequence length, commonly `4096`.
- `SFT_PER_DEVICE_BATCH_SIZE`: per-GPU batch size.
- `SFT_GRAD_ACCUM`: gradient accumulation steps.
- `SFT_EPOCHS`: training epochs.
- `SFT_LR`: learning rate, default `2e-4`.
- `SFT_LORA_R`: LoRA rank, default `16`.
- `SFT_LORA_ALPHA`: LoRA alpha, default `32`.
- `SFT_LOGGING_STEPS`: logging interval.
- `SFT_REPORT_TO`: Trainer report target, default `none`.

## Output Directory

Common contents of `SFT_OUTPUT_DIR`:

- `lora_adapters/`: LoRA adapter and tokenizer.
- `train_summary.json`: training summary.
- `merged_model_16bit/`: full model generated when merge is enabled.

After `adapter_dir` in `train_summary.json` is non-empty, the LoRA adapter can be served with vLLM.

## Merge LoRA

```bash
python -m llm_finetune.sft.merge_lora \
  --base-model Qwen/Qwen3-0.6B \
  --adapter-path /GSI/outputs/sft/qwen3_0_6b_cybertown_sft/lora_adapters \
  --output-dir /GSI/outputs/sft/qwen3_0_6b_cybertown_sft_merged
```

Serve the merged model as a full model, or serve the adapter in LoRA mode. See [vLLM Startup and GSI Evaluation](vllm-eval.md).

## Next Steps

- Continue with RLVR: [RLVR Training](rlvr.md).
- Start vLLM and evaluate: [vLLM Startup and GSI Evaluation](vllm-eval.md).
- Inspect evaluation outputs: [Output Directory](outputs.md).
