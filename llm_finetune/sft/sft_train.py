#!/usr/bin/env python3
"""
本地 JSONL SFT 训练入口。

业务语义:
- 输入是仓库内的 prompt_response / messages / prompt-completion JSONL。
- 输出是 HuggingFace Trainer 生成的 LoRA adapter，可选合并为完整 causal LM。

关键规则:
- 不访问网络；模型、tokenizer、chat template 都必须来自本地路径。
- prompt token 的 label 固定为 -100，只训练 assistant/response 部分。
- 使用 Transformers + PEFT，避免依赖 base image 中不存在的 Unsloth。
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

from llm_finetune.sft.paths import (
    DEFAULT_DATA_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_TEMPLATE_PATH,
)


@dataclass(frozen=True)
class DistributedEnv:
    enabled: bool
    world_size: int
    rank: int
    local_rank: int

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0


@dataclass
class ResponseOnlyCollator:
    tokenizer: Any
    pad_to_multiple_of: int = 8

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        batch = self.tokenizer.pad(
            {
                "input_ids": [item["input_ids"] for item in features],
                "attention_mask": [item["attention_mask"] for item in features],
            },
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        max_len = batch["input_ids"].shape[1]
        labels = []
        for item in features:
            pad_len = max_len - len(item["labels"])
            labels.append(item["labels"] + [-100] * pad_len)
        batch["labels"] = torch.tensor(labels, dtype=torch.long)
        return batch


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline LoRA SFT for multi-robot planner JSONL.")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--data-path", nargs="+", default=[str(DEFAULT_DATA_PATH)])
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--template-path", default=str(DEFAULT_TEMPLATE_PATH))
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--num-train-epochs", type=int, default=1)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--ddp-find-unused-parameters", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--merge-model", action="store_true")
    parser.add_argument("--report-to", default="none")
    return parser.parse_args(argv)


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return int(value)


def distributed_env() -> DistributedEnv:
    world_size = _env_int("WORLD_SIZE", 1)
    rank = _env_int("RANK", 0)
    local_rank = _env_int("LOCAL_RANK", 0)
    return DistributedEnv(enabled=world_size > 1, world_size=world_size, rank=rank, local_rank=local_rank)


def configure_distributed_device(dist: DistributedEnv) -> None:
    if dist.enabled and torch.cuda.is_available():
        torch.cuda.set_device(dist.local_rank)


def distributed_barrier(dist: DistributedEnv) -> None:
    if (
        dist.enabled
        and hasattr(torch, "distributed")
        and torch.distributed.is_available()
        and torch.distributed.is_initialized()
    ):
        torch.distributed.barrier()


def model_device_map(load_in_4bit: bool, dist: DistributedEnv) -> Any:
    if not torch.cuda.is_available():
        return None
    if not dist.enabled:
        return "auto"
    if load_in_4bit:
        return {"": dist.local_rank}
    return None


def training_argument_kwargs(args: argparse.Namespace, output_dir: Path, dist: DistributedEnv) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "num_train_epochs": args.num_train_epochs,
        "per_device_train_batch_size": args.per_device_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "logging_steps": args.logging_steps,
        "save_strategy": "steps",
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "bf16": torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        "fp16": torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        "report_to": args.report_to,
        "remove_unused_columns": False,
        "seed": args.seed,
        "gradient_checkpointing": args.gradient_checkpointing,
    }
    if dist.enabled:
        kwargs["ddp_find_unused_parameters"] = (
            args.ddp_find_unused_parameters if args.ddp_find_unused_parameters is not None else False
        )
    elif args.ddp_find_unused_parameters is not None:
        kwargs["ddp_find_unused_parameters"] = args.ddp_find_unused_parameters
    return kwargs


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_jsonl_paths(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        for row in read_jsonl(path):
            row.setdefault("_source_path", str(path))
            rows.append(row)
    return rows


def is_message_list(value: Any) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(item, dict) and "role" in item and "content" in item for item in value)
    )


def infer_row_kind(row: dict[str, Any]) -> str:
    if is_message_list(row.get("messages")):
        return "messages"
    if "prompt" in row and "completion" in row:
        return "split_messages" if is_message_list(row["prompt"]) and is_message_list(row["completion"]) else "prompt_completion"
    if "prompt" in row and "response" in row:
        return "prompt_response"
    raise ValueError("Unsupported JSONL row. Expected messages, prompt/completion, or prompt/response.")


def render_response(value: Any) -> str:
    if isinstance(value, str):
        return value
    return "```json\n" + json.dumps(value, ensure_ascii=False, indent=2) + "\n```"


def row_record_id(row: dict[str, Any]) -> Any:
    return row.get("record_id") or row.get("task_id") or row.get("id")


def normalize_message_list(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [{"role": str(item["role"]), "content": str(item["content"])} for item in messages]


def row_to_messages(row: dict[str, Any], kind: str | None = None) -> list[dict[str, str]]:
    if kind is None:
        kind = infer_row_kind(row)
    if kind == "messages":
        return normalize_message_list(row["messages"])
    if kind == "split_messages":
        return [*normalize_message_list(row["prompt"]), *normalize_message_list(row["completion"])]
    if kind == "prompt_completion":
        completion = row["completion"]
        if is_message_list(completion):
            completion_text = "\n\n".join(str(item["content"]) for item in completion)
        else:
            completion_text = render_response(completion)
        return [
            {"role": "user", "content": str(row["prompt"])},
            {"role": "assistant", "content": completion_text},
        ]
    return [
        {"role": "user", "content": str(row["prompt"])},
        {"role": "assistant", "content": render_response(row["response"])},
    ]


def load_tokenizer(model_path: str, template_path: str) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False, local_files_only=True)
    if template_path:
        tokenizer.chat_template = Path(template_path).read_text(encoding="utf-8").strip()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def render_messages(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return "\n\n".join(f"{item['role'].upper()}:\n{item['content']}" for item in messages)


def render_prompt_messages(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt = "\n\n".join(f"{item['role'].upper()}:\n{item['content']}" for item in messages)
    return f"{prompt}\n\nASSISTANT:\n"


def tokenize_response_only(tokenizer: Any, prompt_text: str, full_text: str, max_seq_length: int) -> dict[str, Any]:
    prompt = tokenizer(prompt_text, add_special_tokens=False, truncation=True, max_length=max_seq_length)
    full = tokenizer(full_text, add_special_tokens=False, truncation=True, max_length=max_seq_length)
    prompt_len = min(len(prompt["input_ids"]), len(full["input_ids"]))
    labels = [-100] * prompt_len + full["input_ids"][prompt_len:]
    if all(label == -100 for label in labels):
        raise ValueError("No trainable response tokens remain after truncation.")
    return {"input_ids": full["input_ids"], "attention_mask": full["attention_mask"], "labels": labels}


def build_dataset(tokenizer: Any, rows: list[dict[str, Any]], max_seq_length: int) -> tuple[Dataset, list[dict[str, Any]]]:
    encoded = []
    debug = []
    for row in rows:
        kind = infer_row_kind(row)
        messages = row_to_messages(row, kind)
        if len(messages) < 2 or messages[-1]["role"] != "assistant":
            continue
        prompt_text = render_prompt_messages(tokenizer, messages[:-1])
        full_text = render_messages(tokenizer, messages)
        try:
            item = tokenize_response_only(tokenizer, prompt_text, full_text, max_seq_length)
        except ValueError:
            continue
        encoded.append(item)
        if len(debug) < 3:
            active_ids = [tid for tid, label in zip(item["input_ids"], item["labels"]) if label != -100]
            debug.append(
                {
                    "record_id": row_record_id(row),
                    "dataset_kind": kind,
                    "source_path": row.get("_source_path", ""),
                    "active_label_tokens": len(active_ids),
                    "total_tokens": len(item["input_ids"]),
                    "target_preview": tokenizer.decode(active_ids[:160], skip_special_tokens=False),
                }
            )
    if not encoded:
        raise ValueError("No valid training rows produced.")
    return Dataset.from_list(encoded), debug


def load_model(args: argparse.Namespace, dist: DistributedEnv | None = None) -> Any:
    if dist is None:
        dist = distributed_env()
    quantization_config = None
    if args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
        quantization_config=quantization_config,
        device_map=model_device_map(args.load_in_4bit, dist),
    )
    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    if args.gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.config.use_cache = False
    return model


def main() -> int:
    args = parse_args()
    dist = distributed_env()
    configure_distributed_device(dist)
    if args.repeat < 1:
        raise ValueError("--repeat must be >= 1")
    data_paths = [Path(path) for path in args.data_path]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl_paths(data_paths)
    if args.max_samples > 0:
        rows = rows[: args.max_samples]
    rows = rows * args.repeat
    if not rows:
        raise ValueError(f"No rows loaded from {data_paths}")

    dataset_kinds = Counter(infer_row_kind(row) for row in rows)
    tokenizer = load_tokenizer(args.model_path, args.template_path)
    dataset, debug_rows = build_dataset(tokenizer, rows, args.max_seq_length)
    model = load_model(args, dist)

    training_args = TrainingArguments(**training_argument_kwargs(args, output_dir, dist))
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=ResponseOnlyCollator(tokenizer),
        processing_class=tokenizer,
    )
    result = trainer.train()

    adapter_dir = output_dir / "lora_adapters"
    if dist.is_main_process:
        model.save_pretrained(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))
    distributed_barrier(dist)

    merged_dir = None
    if args.merge_model:
        if dist.enabled:
            raise ValueError("Do not use --merge-model under torchrun/DDP. Merge the LoRA adapter in a separate single-process step.")
        merged_model = model.merge_and_unload()
        merged_dir = output_dir / "merged_model"
        merged_model.save_pretrained(str(merged_dir), safe_serialization=True)
        tokenizer.save_pretrained(str(merged_dir))

    summary = {
        "model_path": args.model_path,
        "data_path": str(data_paths[0]),
        "data_paths": [str(path) for path in data_paths],
        "dataset_kinds": dict(dataset_kinds),
        "train_rows": len(dataset),
        "output_dir": str(output_dir),
        "adapter_dir": str(adapter_dir),
        "merged_dir": str(merged_dir) if merged_dir else "",
        "train_loss": result.training_loss,
        "distributed": {
            "enabled": dist.enabled,
            "world_size": dist.world_size,
            "rank": dist.rank,
            "local_rank": dist.local_rank,
        },
        "debug_rows": debug_rows,
    }
    if dist.is_main_process:
        (output_dir / "train_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    distributed_barrier(dist)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
