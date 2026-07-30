# SFT 训练

本页说明如何在训练容器内启动 SFT。命令默认在容器内执行。

## 前置检查

```bash
cd /GSI/llm_finetune
export ROOT_DIR=/GSI
./scripts/runtime/check_env.sh
```

训练前先准备 Hugging Face cache 和数据，见 [Hugging Face 准备](huggingface-prepare.md)。

## 常规 Unsloth SFT

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

默认值：

- `SFT_MODEL_REPO_ID=Qwen/Qwen3-0.6B`
- `SFT_DATA_REPO_ID=WindyLab/Qwen3-0.6B-cybertown-SFT-data`
- `SFT_DATA_FILENAME=train.jsonl`
- `SFT_TEMPLATE_PATH=/GSI/llm_finetune/chat_template/qwen3_nonthinking.jinja`
- `SFT_LOAD_IN_4BIT=1`

## GPU 选择

Unsloth wrapper 默认只暴露一张 GPU。指定第 2 张卡：

```bash
SFT_CUDA_VISIBLE_DEVICES=1 \
SFT_MODEL_REPO_ID=Qwen/Qwen3-0.6B \
SFT_DATA_REPO_ID=WindyLab/Qwen3-0.6B-cybertown-SFT-data \
SFT_OUTPUT_DIR=/GSI/outputs/sft/qwen3_0_6b_cybertown_sft_gpu1 \
./scripts/runtime/train_sft_unsloth.sh
```

只有调试 Unsloth/HF Trainer 多卡路径时才设置：

```bash
SFT_UNSLOTH_ALLOW_MULTI_GPU=1
```

## 使用本地路径

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

优先级：

- `SFT_MODEL_PATH` 或 `MODEL_PATH` 优先于 `SFT_MODEL_REPO_ID`。
- `SFT_DATA_PATH` 或 `DATA_PATH` 优先于 `SFT_DATA_REPO_ID`。
- `SFT_OUTPUT_DIR` 或 `OUTPUT_DIR` 控制输出目录。

## 常用参数

- `SFT_MAX_SEQ_LENGTH`：最大序列长度，常用 `4096`。
- `SFT_PER_DEVICE_BATCH_SIZE`：单卡 batch size。
- `SFT_GRAD_ACCUM`：梯度累积步数。
- `SFT_EPOCHS`：训练 epoch 数。
- `SFT_LR`：学习率，默认 `2e-4`。
- `SFT_LORA_R`：LoRA rank，默认 `16`。
- `SFT_LORA_ALPHA`：LoRA alpha，默认 `32`。
- `SFT_LOGGING_STEPS`：日志间隔。
- `SFT_REPORT_TO`：Trainer report target，默认 `none`。

## 输出目录

`SFT_OUTPUT_DIR` 常见内容：

- `lora_adapters/`：LoRA adapter 和 tokenizer。
- `train_summary.json`：训练摘要。
- `merged_model_16bit/`：启用 merge 选项后生成的完整模型。

确认 `train_summary.json` 中的 `adapter_dir` 非空后，可用 LoRA adapter 启动 vLLM。

## 合并 LoRA

```bash
python -m llm_finetune.sft.merge_lora \
  --base-model Qwen/Qwen3-0.6B \
  --adapter-path /GSI/outputs/sft/qwen3_0_6b_cybertown_sft/lora_adapters \
  --output-dir /GSI/outputs/sft/qwen3_0_6b_cybertown_sft_merged
```

合并后的模型按完整模型方式启动；未合并 adapter 按 LoRA 方式启动。见 [vLLM 启动与 GSI 评估](vllm-eval.md)。

## 后续步骤

- 在 SFT 模型上继续 RLVR：见 [RLVR 训练](rlvr.md)。
- 启动 vLLM 并评估：见 [vLLM 启动与 GSI 评估](vllm-eval.md)。
- 查看评估输出：见 [输出目录](outputs.md)。
