# run/utils/log_token_parser.py
"""
LLM call log parser - extract prompts and responses from log.md and count tokens.

Design principles:
- Offline parsing: extract from log.md after experiments without affecting runtime performance.
- Structured markers: rely on <!-- LLM_PROMPT_START --> / <!-- LLM_RESPONSE_END --> marker pairs.
- Token counting: use tiktoken, the OpenAI tokenizer, with character-count fallback.

log.md format:
  Marker line format:
    #### <span ...>info: </span>\n[timestamp]:<!-- LLM_PROMPT_START -->\n
  Prompt and response content between marker lines:
    #### <span ...>debug: </span>\n[timestamp]:Prompt:\n <content>\n
    #### <span ...>info: </span>\n[timestamp]:Response:\n <content>\n
"""
import re
from pathlib import Path
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Marker detection (match the marker text within log lines)
# ---------------------------------------------------------------------------

_MARKER_PROMPT_START = "<!-- LLM_PROMPT_START -->"
_MARKER_PROMPT_END = "<!-- LLM_PROMPT_END -->"
_MARKER_RESPONSE_START = "<!-- LLM_RESPONSE_START -->"
_MARKER_RESPONSE_END = "<!-- LLM_RESPONSE_END -->"

# Pattern to strip log metadata: timestamp prefix like "[2026-03-06 20:12:22:317666]:"
# and the "Prompt:\n " or "Response:\n " prefix from dlog
_TIMESTAMP_PREFIX = re.compile(r"^\[[\d\-: ]+\]:")
_PROMPT_LABEL = re.compile(r"^Prompt:\n ?", re.MULTILINE)
_RESPONSE_LABEL = re.compile(r"^Response:\n ?", re.MULTILINE)
# HTML span headers injected by Logger
_HTML_HEADER = re.compile(r"^#{1,6}\s*<span[^>]*>.*?</span>\s*$", re.MULTILINE)


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------

_encoder = None


def _get_encoder():
    """Lazy-load tiktoken encoder. Falls back to None if unavailable."""
    global _encoder
    if _encoder is not None:
        return _encoder
    try:
        import tiktoken
        _encoder = tiktoken.encoding_for_model("gpt-4o")
    except Exception:
        _encoder = None
    return _encoder


def count_tokens(text: str) -> int:
    """Count tokens using tiktoken; fallback to len/4 estimate."""
    enc = _get_encoder()
    if enc is not None:
        return len(enc.encode(text))
    # Rough estimate: ~4 chars per token for English/mixed text
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Content extraction helpers
# ---------------------------------------------------------------------------

def _clean_block(raw: str) -> str:
    """Remove log metadata (HTML headers, timestamps) to get pure content."""
    # Remove HTML span header lines
    text = _HTML_HEADER.sub("", raw)
    # Remove timestamp prefixes on each line
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        line = _TIMESTAMP_PREFIX.sub("", line)
        cleaned.append(line)
    text = "\n".join(cleaned)
    # Remove "Prompt:\n " / "Response:\n " labels
    text = _PROMPT_LABEL.sub("", text)
    text = _RESPONSE_LABEL.sub("", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------

def parse_llm_calls(log_path: Path) -> List[Dict[str, int]]:
    """Parse log.md and extract per-call token counts.

    Strategy: scan line-by-line for marker lines, extract content between
    PROMPT_START..PROMPT_END and RESPONSE_START..RESPONSE_END pairs.

    Args:
        log_path: Path to log.md file.

    Returns:
        List of dicts, one per LLM call:
        [{"prompt_tokens": int, "response_tokens": int}, ...]
    """
    if not log_path.exists():
        return []

    content = log_path.read_text(encoding="utf-8", errors="replace")
    lines = content.split("\n")

    calls: List[Dict[str, int]] = []
    i = 0
    n = len(lines)

    while i < n:
        # Look for PROMPT_START marker
        if _MARKER_PROMPT_START in lines[i]:
            # Collect prompt content until PROMPT_END
            i += 1
            prompt_lines = []
            while i < n and _MARKER_PROMPT_END not in lines[i]:
                prompt_lines.append(lines[i])
                i += 1
            # Skip PROMPT_END line
            if i < n:
                i += 1

            # Now look for RESPONSE_START marker
            while i < n and _MARKER_RESPONSE_START not in lines[i]:
                i += 1
            if i < n:
                i += 1  # skip RESPONSE_START line

            # Collect response content until RESPONSE_END
            response_lines = []
            while i < n and _MARKER_RESPONSE_END not in lines[i]:
                response_lines.append(lines[i])
                i += 1
            # Skip RESPONSE_END line
            if i < n:
                i += 1

            prompt_text = _clean_block("\n".join(prompt_lines))
            response_text = _clean_block("\n".join(response_lines))

            if prompt_text or response_text:
                calls.append({
                    "prompt_tokens": count_tokens(prompt_text) if prompt_text else 0,
                    "response_tokens": count_tokens(response_text) if response_text else 0,
                })
        else:
            i += 1

    return calls


def compute_token_stats(log_path: Path) -> Dict:
    """Compute token statistics from a log.md file.

    Returns:
        {
            "llm_call_count_from_log": int,
            "prompt_tokens": [int, ...],       # per-call prompt token counts
            "response_tokens": [int, ...],     # per-call response token counts
            "prompt_tokens_total": int,
            "response_tokens_total": int,
            "prompt_tokens_mean": float,
            "response_tokens_mean": float,
        }
        Returns empty dict if no LLM calls found.
    """
    calls = parse_llm_calls(log_path)
    if not calls:
        return {}

    prompt_tokens = [c["prompt_tokens"] for c in calls]
    response_tokens = [c["response_tokens"] for c in calls]

    n = len(calls)
    return {
        "llm_call_count_from_log": n,
        "prompt_tokens": prompt_tokens,
        "response_tokens": response_tokens,
        "prompt_tokens_total": sum(prompt_tokens),
        "response_tokens_total": sum(response_tokens),
        "prompt_tokens_mean": round(sum(prompt_tokens) / n, 2),
        "response_tokens_mean": round(sum(response_tokens) / n, 2),
    }
