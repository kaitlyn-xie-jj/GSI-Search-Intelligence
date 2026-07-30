#!/usr/bin/env python3
"""Merge a PEFT LoRA adapter into a local base causal LM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline single-process LoRA merge.")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--safe-serialization", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args(argv)


def default_torch_dtype() -> Any:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def merge_lora(base_model: str, adapter_path: str, output_dir: str, safe_serialization: bool = True) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True, use_fast=False, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=default_torch_dtype(),
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model = PeftModel.from_pretrained(model, adapter_path, local_files_only=True)
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(str(output), safe_serialization=safe_serialization)
    tokenizer.save_pretrained(str(output))

    summary = {
        "base_model": base_model,
        "adapter_path": adapter_path,
        "output_dir": output_dir,
        "safe_serialization": safe_serialization,
    }
    (output / "merge_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main() -> int:
    args = parse_args()
    merge_lora(
        base_model=args.base_model,
        adapter_path=args.adapter_path,
        output_dir=args.output_dir,
        safe_serialization=args.safe_serialization,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
