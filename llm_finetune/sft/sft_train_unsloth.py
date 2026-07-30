#!/usr/bin/env python3
"""
Unsloth-based offline JSONL SFT entrypoint.

This mirrors the local benchmark Web Console training path while keeping the
workbench JSONL conveniences and the corrected response-only mask boundary.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from unsloth import FastLanguageModel
from transformers import AutoTokenizer, Trainer, TrainingArguments

from llm_finetune.sft.paths import (
    DEFAULT_DATA_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_TEMPLATE_PATH,
    DEFAULT_UNSLOTH_OUTPUT_DIR,
)


DEFAULT_OUTPUT_DIR = DEFAULT_UNSLOTH_OUTPUT_DIR


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
    parser = argparse.ArgumentParser(description="Offline Unsloth LoRA SFT for multi-robot planner JSONL.")
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
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--save-strategy", choices=["no", "epoch", "steps"], default="epoch")
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--merge-16bit", dest="merge_16bit", action="store_true")
    parser.add_argument("--merge-model", dest="merge_16bit", action="store_true")
    parser.add_argument("--no-save-adapters", action="store_true")
    parser.add_argument("--report-to", default="none")
    parser.add_argument("--dataloader-num-workers", type=int, default=0)
    parser.add_argument("--dataloader-pin-memory", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args(argv)


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


def load_model(args: argparse.Namespace) -> tuple[Any, Any]:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_path,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=args.load_in_4bit,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )
    model.config.use_cache = False
    return model, tokenizer


def main() -> int:
    args = parse_args()
    if args.repeat < 1:
        raise ValueError("--repeat must be >= 1")
    if args.dataloader_num_workers < 0:
        raise ValueError("--dataloader-num-workers must be >= 0")
    if args.no_save_adapters and not args.merge_16bit:
        raise ValueError("Refusing to train with no export target. Enable --merge-16bit or allow adapter saving.")

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
    template_tokenizer = load_tokenizer(args.model_path, args.template_path)
    model, tokenizer = load_model(args)
    if args.template_path:
        tokenizer.chat_template = template_tokenizer.chat_template
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset, debug_rows = build_dataset(tokenizer, rows, args.max_seq_length)
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        report_to=args.report_to,
        remove_unused_columns=False,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=bool(args.dataloader_pin_memory),
        seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=ResponseOnlyCollator(tokenizer),
        processing_class=tokenizer,
    )
    result = trainer.train()

    adapter_dir = None
    if not args.no_save_adapters:
        adapter_dir = output_dir / "lora_adapters"
        model.save_pretrained(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))

    merged_dir = None
    if args.merge_16bit:
        merged_dir = output_dir / "merged_model_16bit"
        model.save_pretrained_merged(
            save_directory=str(merged_dir),
            tokenizer=tokenizer,
            save_method="merged_16bit",
        )

    summary = {
        "backend": "unsloth",
        "model_path": args.model_path,
        "data_path": str(data_paths[0]),
        "data_paths": [str(path) for path in data_paths],
        "template_path": args.template_path,
        "dataset_kinds": dict(dataset_kinds),
        "repeat": args.repeat,
        "train_rows": len(dataset),
        "output_dir": str(output_dir),
        "adapter_dir": str(adapter_dir) if adapter_dir else "",
        "merged_dir": str(merged_dir) if merged_dir else "",
        "train_loss": result.training_loss,
        "save_adapters": not args.no_save_adapters,
        "merge_16bit": args.merge_16bit,
        "dataloader_num_workers": args.dataloader_num_workers,
        "dataloader_pin_memory": bool(args.dataloader_pin_memory),
        "debug_rows": debug_rows,
    }
    (output_dir / "train_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
