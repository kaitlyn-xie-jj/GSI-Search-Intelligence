#!/usr/bin/env python3
"""Replay fixed benchmark suites against the GSI validator service."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import requests


def _read_suite(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            records.append(json.loads(line))
            if limit is not None and len(records) >= limit:
                break
    return records


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    idx = max(0, min(len(values) - 1, int(len(values) * pct / 100.0) - 1))
    return sorted(values)[idx]


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"avg": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    return {
        "avg": round(statistics.mean(values), 4),
        "p50": round(statistics.median(values), 4),
        "p95": round(_percentile(values, 95), 4),
        "p99": round(_percentile(values, 99), 4),
        "max": round(max(values), 4),
    }


def _plan_payload(record: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "plan": str(record.get("response") or ""),
        "task_id": str(record.get("task_id") or ""),
    }
    if record.get("state_store") and record.get("state_id"):
        payload["state_store"] = record["state_store"]
        payload["state_id"] = record["state_id"]
    return payload


def _extract_timing(result: dict[str, Any]) -> dict[str, float]:
    details = result.get("validation_details") or {}
    inner = details.get("details") or {}
    timing = inner.get("timing") or {}
    return {str(key): float(value) for key, value in timing.items() if isinstance(value, (int, float))}


def _result_row(record: dict[str, Any], result: dict[str, Any], elapsed_ms: float) -> dict[str, Any]:
    return {
        "record": record,
        "valid": bool(result.get("valid")),
        "overall_reward": float(result.get("overall_reward", 0.0)),
        "elapsed_ms": elapsed_ms,
        "timing": _extract_timing(result),
        "validator_response": result,
    }


def replay_single(records: list[dict[str, Any]], base_url: str, timeout: float) -> tuple[list[dict[str, Any]], float]:
    session = requests.Session()
    session.trust_env = False
    rows: list[dict[str, Any]] = []
    start = time.perf_counter()
    for record in records:
        item_start = time.perf_counter()
        response = session.post(f"{base_url}/validate", json=_plan_payload(record), timeout=timeout)
        response.raise_for_status()
        rows.append(_result_row(record, response.json(), (time.perf_counter() - item_start) * 1000.0))
    return rows, time.perf_counter() - start


def replay_batch(
    records: list[dict[str, Any]],
    base_url: str,
    timeout: float,
    batch_size: int,
) -> tuple[list[dict[str, Any]], float]:
    session = requests.Session()
    session.trust_env = False
    rows: list[dict[str, Any]] = []
    start = time.perf_counter()
    for idx in range(0, len(records), batch_size):
        chunk = records[idx:idx + batch_size]
        item_start = time.perf_counter()
        response = session.post(
            f"{base_url}/validate_batch",
            json={"plans": [_plan_payload(record) for record in chunk]},
            timeout=timeout * max(1, len(chunk)),
        )
        response.raise_for_status()
        elapsed_ms = (time.perf_counter() - item_start) * 1000.0
        results = response.json().get("results", [])
        if len(results) != len(chunk):
            raise RuntimeError(f"batch result length mismatch: requested={len(chunk)} returned={len(results)}")
        per_item_ms = elapsed_ms / max(1, len(chunk))
        for record, result in zip(chunk, results, strict=True):
            rows.append(_result_row(record, result, per_item_ms))
    return rows, time.perf_counter() - start


def summarize_results(rows: list[dict[str, Any]], *, wall_s: float, mode: str) -> dict[str, Any]:
    elapsed = [float(row["elapsed_ms"]) for row in rows]
    rewards = [float(row["overall_reward"]) for row in rows]
    first_proxy_matches = [
        row for row in rows
        if row["record"].get("batch_first_plan_reward_proxy") == row["overall_reward"]
    ]
    final_proxy_matches = [
        row for row in rows
        if row["record"].get("batch_final_reward_proxy") == row["overall_reward"]
    ]
    timing_keys = sorted({key for row in rows for key in row.get("timing", {})})
    timing_stats = {
        key: _stats([float(row["timing"][key]) for row in rows if key in row.get("timing", {})])
        for key in timing_keys
    }
    return {
        "mode": mode,
        "count": len(rows),
        "wall_s": round(wall_s, 4),
        "plans_per_sec": round(len(rows) / wall_s, 4) if wall_s else 0.0,
        "elapsed_ms": _stats(elapsed),
        "reward": _stats(rewards),
        "valid_count": sum(1 for row in rows if row["valid"]),
        "match_first_plan_proxy_count": len(first_proxy_matches),
        "match_first_plan_proxy_ratio": round(len(first_proxy_matches) / len(rows), 4) if rows else 0.0,
        "match_final_proxy_count": len(final_proxy_matches),
        "match_final_proxy_ratio": round(len(final_proxy_matches) / len(rows), 4) if rows else 0.0,
        "timing_ms": timing_stats,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark the GSI validator with fixed plan fixtures.")
    parser.add_argument("--suite", required=True, help="Benchmark JSONL.")
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="Validator base URL.")
    parser.add_argument("--mode", choices=["single", "batch"], default="batch")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default="", help="Optional JSON output path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = _read_suite(Path(args.suite), limit=args.limit)
    base_url = args.url.rstrip("/")
    if args.mode == "single":
        rows, wall_s = replay_single(records, base_url, args.timeout)
    else:
        rows, wall_s = replay_batch(records, base_url, args.timeout, max(1, args.batch_size))
    summary = summarize_results(rows, wall_s=wall_s, mode=args.mode)
    output = {"summary": summary, "results": rows}
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
