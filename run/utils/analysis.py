# run/utils/analysis.py
"""
Experiment results aggregation analysis module

Design principles:
- Common metrics (shared by all methods): explicitly defined, ensuring statistical completeness
- Algorithm-specific metrics (e.g., SGI's batch_count, LLaMAR's verifier_calls): auto-discovered, auto-aggregated
- Unified statistics: numeric metrics use mean±std, boolean metrics use ratios
"""
import json
from math import sqrt
from pathlib import Path
from typing import Any, Dict, List, Optional


# ============================================================================
# Statistics helper functions
# ============================================================================

def _mean(vals: List[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0

def _var(vals: List[float]) -> float:
    if not vals:
        return 0.0
    m = _mean(vals)
    return sum((x - m) ** 2 for x in vals) / len(vals)

def _std(vals: List[float]) -> float:
    return sqrt(_var(vals))

def _r6(x: float) -> float:
    return round(float(x), 6)

def _pack(vals: List[float]) -> str:
    """Format as mean±std string."""
    return f"{_r6(_mean(vals))}±{_r6(_std(vals))}"


# ============================================================================
# Common metric definitions
# ============================================================================

# Common metrics: produced by all methods and handled explicitly during aggregation.
# key -> extraction description, only for documentation; extraction is implemented in code.
COMMON_METRICS = {
    # Success rate, boolean to ratio.
    "success_rate",
    # Numeric scalar metrics, read directly from metrics and aggregated as mean±std.
    "llm_calls",
    "replans_full",
    "replans_partial",
    "replans_total",
    "total_energy",
    "newcase_total",
    "newcase_total_orig",
    # List metrics, flattened and aggregated as mean±std.
    "planning_durations",
    "allocation_durations",
    "prompt_tokens",
    "response_tokens",
    # Token statistic scalars, read from summary.
    "prompt_tokens_mean",
    "response_tokens_mean",
    "prompt_tokens_total",
    "response_tokens_total",
    "llm_call_count_from_log",
}

# Metrics skipped during aggregation because they are non-numeric or handled elsewhere.
_SKIP_KEYS = {
    "success",           # Handled through success_rate.
    "newcase_by_type",   # Dict type, handled separately.
}


# ============================================================================
# Core aggregation functions
# ============================================================================

def _collect_scalar_series(metrics_list: List[Dict], key: str) -> List[float]:
    """Collect scalar metric series from metrics list."""
    vals = []
    for m in metrics_list:
        v = m.get(key)
        if v is not None:
            try:
                vals.append(float(v))
            except (TypeError, ValueError):
                pass
    return vals


def _collect_list_series(metrics_list: List[Dict], key: str) -> List[float]:
    """Collect list-type metrics from metrics list, flatten to 1D series."""
    vals = []
    for m in metrics_list:
        v = m.get(key)
        if isinstance(v, list):
            for item in v:
                try:
                    vals.append(float(item))
                except (TypeError, ValueError):
                    pass
    return vals


def _aggregate_bucket(metrics_list: List[Dict], topk_newcase: int = 10) -> Dict[str, Any]:
    """Aggregate statistics for a group of metrics.
    
    Processing logic:
    1. Success rate: bool -> ratio
    2. Common numeric scalar metrics: mean±std
    3. Common list metrics (planning_durations): flatten then mean±std
    4. newcase_by_type: aggregate by type
    5. Algorithm-specific metrics: auto-discovered, numeric uses mean±std, list flattened
    """
    if not metrics_list:
        return {"count_runs": 0}
    
    result: Dict[str, Any] = {"count_runs": len(metrics_list)}
    
    # --- 1. Success rate ---
    succ_vals = [1.0 if m.get("success") else 0.0 for m in metrics_list]
    result["success_rate"] = _pack(succ_vals)
    
    # --- 2. Common numeric scalar metrics ---
    scalar_keys = [
        "llm_calls", "replans_full", "replans_partial", "replans_total",
        "total_energy", "newcase_total", "newcase_total_orig",
        "prompt_tokens_mean", "response_tokens_mean",
        "prompt_tokens_total", "response_tokens_total",
        "llm_call_count_from_log",
    ]
    for key in scalar_keys:
        vals = _collect_scalar_series(metrics_list, key)
        if vals:
            result[key] = _pack(vals)
    
    # --- 3. Common list metrics ---
    planning_durs = _collect_list_series(metrics_list, "planning_durations")
    if planning_durs:
        result["planning_duration"] = _pack(planning_durs)
    
    allocation_durs = _collect_list_series(metrics_list, "allocation_durations")
    if allocation_durs:
        result["allocation_duration"] = _pack(allocation_durs)
    
    # Token list metrics, flattened and aggregated as mean±std.
    prompt_toks = _collect_list_series(metrics_list, "prompt_tokens")
    if prompt_toks:
        result["prompt_tokens"] = _pack(prompt_toks)
    
    response_toks = _collect_list_series(metrics_list, "response_tokens")
    if response_toks:
        result["response_tokens"] = _pack(response_toks)
    
    # --- 4. newcase_by_type ---
    all_type_keys = set()
    for m in metrics_list:
        all_type_keys.update((m.get("newcase_by_type", {}) or {}).keys())
    if all_type_keys:
        type_stats = []
        for k in sorted(all_type_keys):
            vals = [float((m.get("newcase_by_type", {}) or {}).get(k, 0)) for m in metrics_list]
            type_stats.append((k, _pack(vals)))
        type_stats.sort(key=lambda x: float(x[1].split("±")[0]), reverse=True)
        result["newcase_top_types"] = type_stats[:topk_newcase]
    
    # --- 5. Algorithm-specific metrics, auto-discovered ---
    # Collect all observed keys, excluding handled common metrics and skipped items.
    handled_keys = COMMON_METRICS | _SKIP_KEYS | {"newcase_by_type"}
    all_keys = set()
    for m in metrics_list:
        all_keys.update(m.keys())
    extra_keys = sorted(all_keys - handled_keys)
    
    if extra_keys:
        algo_specific = {}
        for key in extra_keys:
            # Infer value type from the first non-None value.
            sample = None
            for m in metrics_list:
                v = m.get(key)
                if v is not None:
                    sample = v
                    break
            
            if sample is None:
                continue
            
            if isinstance(sample, list):
                # List type: flatten for statistics.
                vals = _collect_list_series(metrics_list, key)
                if vals:
                    algo_specific[key] = _pack(vals)
            elif isinstance(sample, (int, float, bool)):
                # Numeric type: scalar statistics.
                vals = _collect_scalar_series(metrics_list, key)
                if vals:
                    algo_specific[key] = _pack(vals)
            # Skip other types, such as dict and str.
        
        if algo_specific:
            result["algorithm_specific"] = algo_specific
    
    return result


def aggregate_experiment_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate batch experiment results.
    
    Args:
        results: List of single experiment results, each containing metrics, goal_type, elapsed_sec, token_stats fields
        
    Returns:
        {
            "overall": {...},        # Overall statistics
            "by_goal_type": {        # Statistics bucketed by goal type
                "transport": {...},
                "search": {...},
                ...
            }
        }
    """
    # Merge token_stats into metrics for unified aggregation.
    for r in results:
        m = r.get("metrics")
        ts = r.get("token_stats")
        if isinstance(m, dict) and isinstance(ts, dict) and ts:
            m.update(ts)
    
    # Collect overall metrics.
    ms_all = [
        r.get("metrics", {}) for r in results if isinstance(r.get("metrics"), dict)
    ]
    elapsed_all = [
        float(r.get("elapsed_sec", 0)) for r in results if r.get("elapsed_sec") is not None
    ]
    
    overall = _aggregate_bucket(ms_all)
    if elapsed_all:
        overall["elapsed_sec"] = _pack(elapsed_all)
    
    # Bucket by goal_type.
    buckets: Dict[str, List[Dict]] = {}
    elapsed_buckets: Dict[str, List[float]] = {}
    for r in results:
        gt = r.get("goal_type") or "UNKNOWN"
        m = r.get("metrics", {})
        if isinstance(m, dict):
            buckets.setdefault(gt, []).append(m)
            elapsed_buckets.setdefault(gt, []).append(float(r.get("elapsed_sec", 0)))
    
    by_goal_type = {}
    for gt, ms in buckets.items():
        by_goal_type[gt] = _aggregate_bucket(ms, topk_newcase=5)
        el = elapsed_buckets.get(gt, [])
        if el:
            by_goal_type[gt]["elapsed_sec"] = _pack(el)
    
    return {"overall": overall, "by_goal_type": by_goal_type}


# ============================================================================
# Helper functions
# ============================================================================

def merge_replan_records(batch_root: Path, results, output_name: str = "replan_dataset.jsonl") -> Path:
    output_path = batch_root / output_name
    with output_path.open("w", encoding="utf-8") as fout:
        for r in results:
            run_dir = Path(r.get("run_dir"))
            rec_file = run_dir / "replan_records.jsonl"
            if not rec_file.exists():
                continue
            with rec_file.open("r", encoding="utf-8") as fin:
                for line in fin:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    metrics = r.get("metrics", {}) or {}
                    obj.setdefault("case", {
                        "type_name": r.get("type_name"),
                        "scenario_id": r.get("scenario_id"),
                        "goal_id": r.get("goal_id"),
                        "goal_type": r.get("goal_type"),
                        "newcase_total_orig": metrics.get("newcase_total_orig", 0),
                    })
                    fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    print(f"[collect] Replanning dataset saved to: {output_path}")
    return output_path
