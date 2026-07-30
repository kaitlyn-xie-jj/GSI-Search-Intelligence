"""Shared repo-relative defaults for SFT scripts."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = REPO_ROOT / "models" / "Qwen3" / "Qwen3-0.6B"
DEFAULT_DATA_PATH = REPO_ROOT / "data" / "sft" / "quality_v2_full_1500_gsi_executable_prompt_response.jsonl"
DEFAULT_TEMPLATE_PATH = REPO_ROOT / "llm_finetune" / "chat_template" / "qwen3_nonthinking.jinja"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "sft" / "qwen3_lora"
DEFAULT_UNSLOTH_OUTPUT_DIR = REPO_ROOT / "outputs" / "sft" / "qwen3_lora_unsloth"

