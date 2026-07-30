"""
Visualisation script for the multi-robot task benchmark.

Supports:
- Single benchmark: plot diversity, difficulty, and consistency for one JSON/JSONL file.
- Multiple benchmarks: analyze metric trends as benchmark size changes.

Data structure assumption:
{
    "instruction": str,
    "goal_details": {...},
    "meta": {
        "language_level": "L0" | "L1" | "L2",
        "plan_level": ["L0", "L1", ...],
        "coor_level": ["L0", "L1", ... "L4"],
        "use_scene_features": true | false,

        # Two consistency indicators written directly by the generator (0/1)
        "source_consistency": 0 | 1,
        "binding_consistency": 0 | 1,
    }
}
"""

import argparse
import json
import random
import re
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple, Set

import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# Light color palette
# -----------------------------
LIGHT_COLORS = [
    "#aec6cf",  # light blue
    "#ffb347",  # light orange
    "#cfcfc4",  # light grey
    "#b39eb5",  # light purple
    "#77dd77",  # light green
    "#fdfd96",  # light yellow
    "#ff6961",  # light red
    "#fcbad3",  # light pink
]

_TOKEN_RE = re.compile(r"\w+")


# ============================================================
# Basic tools: load data, read meta info
# ============================================================

def load_dataset(path: Path) -> List[Dict[str, Any]]:
    """
    Load dataset from a JSONL / JSON file.
    - .jsonl: parse line by line, one sample per line
    - .json: parse as a single list
    """
    if path.suffix == ".jsonl":
        data = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data.append(json.loads(line))
        return data
    elif path.suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}, expected .json or .jsonl")


def get_goal_type(sample: Dict[str, Any]) -> str:
    """Read goal_type from a sample; returns 'unknown' if missing."""
    gd = json.loads(sample.get("goal_details", {}))
    return gd.get("goal_type", "unknown")


def get_meta(sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    Retrieve the meta field with backward-compatible handling:
    - language_level at the top level is synced into meta
    - plan_level / coor_level stored as strings are converted to lists
    """
    meta = json.loads(sample.get("meta", {}))

    # Backward compatible: sync language_level from top level into meta
    if "language_level" not in meta and "language_level" in sample:
        meta["language_level"] = sample["language_level"]

    # Normalize plan_level / coor_level to list
    for key in ("plan_level", "coor_level"):
        if key in meta and isinstance(meta[key], str):
            meta[key] = [meta[key]]

    return meta


def normalize_text(text: str) -> str:
    """Lowercase text, keep only alphanumeric chars, and normalize whitespace."""
    return " ".join(_TOKEN_RE.findall(text.lower()))


def tokenize(text: str) -> List[str]:
    """Simple tokenizer: extract tokens via \\w+ regex, lowercased."""
    return _TOKEN_RE.findall(text.lower())


# ============================================================
# Near-duplicate rate & TTR tools
# ============================================================

def build_ngram_set(tokens: List[str], n: int = 3) -> Set[str]:
    """
    Build a token-level n-gram set for Jaccard similarity computation.
    Falls back to 1-gram if token count < n.
    """
    if not tokens:
        return set()

    if len(tokens) < n:
        return set(tokens)

    grams = set()
    for i in range(len(tokens) - n + 1):
        grams.add(" ".join(tokens[i:i + n]))
    return grams


def estimate_near_duplicate_rate(
    texts: List[str],
    max_samples: int = 1000,
    comparisons_per_sample: int = 50,
    ngram_n: int = 3,
    jaccard_threshold: float = 0.8,
) -> float:
    """
    Estimate near-duplicate rate using n-gram Jaccard similarity + subsampling.
    """
    if len(texts) <= 1:
        return 0.0

    # Preprocess: normalize + tokenize + build n-gram sets
    ngram_sets: List[Set[str]] = []
    for t in texts:
        if not t:
            ngram_sets.append(set())
            continue
        norm = normalize_text(t)
        tokens = tokenize(norm)
        ngram_sets.append(build_ngram_set(tokens, n=ngram_n))

    valid_indices = [i for i, g in enumerate(ngram_sets) if g]
    if len(valid_indices) <= 1:
        return 0.0

    n = min(len(valid_indices), max_samples)
    sampled_idxs = random.sample(valid_indices, n)
    near_dup_count = 0

    for idx in sampled_idxs:
        base_set = ngram_sets[idx]

        others = [i for i in valid_indices if i != idx]
        if not others:
            continue

        k = min(len(others), comparisons_per_sample)
        comp_idxs = random.sample(others, k)

        max_sim = 0.0
        for j in comp_idxs:
            other_set = ngram_sets[j]
            if not other_set:
                continue

            inter = base_set.intersection(other_set)
            union = base_set.union(other_set)
            if not union:
                continue
            sim = len(inter) / float(len(union))
            if sim > max_sim:
                max_sim = sim
            if max_sim >= jaccard_threshold:
                break

        if max_sim >= jaccard_threshold:
            near_dup_count += 1

    return near_dup_count / float(n) if n > 0 else 0.0


# ============================================================
# Part 1: Single benchmark diversity visualization (single plot)
# ============================================================

def compute_diversity_and_redundancy(
    samples: List[Dict[str, Any]]
) -> Tuple[List[str], List[float], List[float], List[float]]:
    """
    For each goal_type, compute:
    - near_duplicate_rate
    - Non-duplicate fraction = 1 - near_duplicate_rate
    - Lexical diversity (TTR)
    """
    by_type: Dict[str, List[str]] = defaultdict(list)
    for s in samples:
        gt = get_goal_type(s)
        instr = s.get("instruction", "")
        if instr:
            by_type[gt].append(instr)

    goal_types = sorted(by_type.keys())
    non_dup_scores: List[float] = []
    ttr_values: List[float] = []
    near_dup_rates: List[float] = []

    for gt in goal_types:
        texts = by_type[gt]

        # 1) Near-duplicate rate
        near_dup = estimate_near_duplicate_rate(texts)
        near_dup_rates.append(near_dup)
        non_dup_scores.append(1.0 - near_dup)

        # 2) TTR
        tokens: List[str] = []
        for instr in texts:
            tokens.extend(tokenize(instr))
        if not tokens:
            ttr = 0.0
        else:
            vocab = set(tokens)
            ttr = len(vocab) / float(len(tokens))
        ttr_values.append(ttr)

    return goal_types, non_dup_scores, ttr_values, near_dup_rates


def plot_diversity_summary(samples: List[Dict[str, Any]], out_path: Path) -> None:
    """
    Diversity single plot:
    - Two bars per goal_type:
      - Non-duplicate fraction (1 - near_duplicate_rate)
      - Lexical diversity (TTR)
    """
    goal_types, non_dup_scores, ttr_values, _ = compute_diversity_and_redundancy(samples)

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(goal_types))
    width = 0.35

    colors_nondup = [LIGHT_COLORS[0]] * len(goal_types)
    colors_ttr = [LIGHT_COLORS[1]] * len(goal_types)

    ax.bar(x - width / 2, non_dup_scores, width, color=colors_nondup,
           label="Non-duplicate fraction (1 - near-dup)")
    ax.bar(x + width / 2, ttr_values, width, color=colors_ttr,
           label="Lexical diversity (TTR)")

    ax.set_xticks(x)
    ax.set_xticklabels(goal_types, rotation=45, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.0)
    ax.set_title("Instruction Diversity & Non-Redundancy per Goal Type")
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# ============================================================
# Part 2: Single benchmark difficulty distribution visualization
# ============================================================

def count_levels(
    samples: List[Dict[str, Any]]
) -> Tuple[Counter, Counter, Counter]:
    """
    Count frequency of different complexity labels:
    - plan_counts: total occurrences of each plan_level
    - lang_counts: sample count per language_level
    - coor_counts: total occurrences of each coor_level
    Note: when plan_level/coor_level is a list, one sample can increment multiple levels.
    """
    plan_counts = Counter()
    lang_counts = Counter()
    coor_counts = Counter()

    for s in samples:
        meta = get_meta(s)

        # Language complexity: assume one language_level per sample
        lang = meta.get("language_level")
        if lang:
            lang_counts[lang] += 1

        # Planning / coordination complexity: allow multiple levels per sample
        plan = meta.get("plan_level", [])
        coor = meta.get("coor_level", [])
        if isinstance(plan, str):
            plan = [plan]
        if isinstance(coor, str):
            coor = [coor]

        for p in plan:
            plan_counts[p] += 1
        for c in coor:
            coor_counts[c] += 1

    return plan_counts, lang_counts, coor_counts


def plot_level_distribution(
    counts: Counter,
    title: str,
    xlabel: str,
    out_path: Path
) -> None:
    """
    Generic level distribution plotting function:
    - counts: Counter(level -> frequency)
    - Output: light-colored bar chart showing sample count per level.
    """
    levels = sorted(counts.keys())
    values = [counts[lv] for lv in levels]

    fig, ax = plt.subplots(figsize=(5, 4))
    x = np.arange(len(levels))
    colors = [LIGHT_COLORS[i % len(LIGHT_COLORS)] for i in range(len(levels))]

    ax.bar(x, values, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(levels)
    ax.set_ylabel("Number of samples")
    ax.set_xlabel(xlabel)
    ax.set_title(title)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_difficulty_distributions(samples: List[Dict[str, Any]], out_dir: Path) -> None:
    """
    Generate three difficulty distribution plots:
    1) Planning complexity distribution (plan_level)
    2) Language complexity distribution (language_level)
    3) Coordination complexity distribution (coor_level)
    """
    plan_counts, lang_counts, coor_counts = count_levels(samples)

    plot_level_distribution(
        plan_counts,
        title="Planning Complexity Distribution (plan_level)",
        xlabel="plan_level",
        out_path=out_dir / "difficulty_plan_level.png",
    )

    plot_level_distribution(
        lang_counts,
        title="Language Complexity Distribution (language_level)",
        xlabel="language_level",
        out_path=out_dir / "difficulty_language_level.png",
    )

    plot_level_distribution(
        coor_counts,
        title="Coordination Complexity Distribution (coor_level)",
        xlabel="coor_level",
        out_path=out_dir / "difficulty_coor_level.png",
    )


# ============================================================
# Part 3: Single benchmark consistency visualization
# ============================================================

def compute_scene_consistency_single(
    samples: List[Dict[str, Any]]
) -> Tuple[List[str], List[float], List[float]]:
    """
    For each goal_type, compute:
    - source_ratio: mean source consistency
    - binding_ratio: mean binding consistency

    Missing fields are treated as 0.
    """
    by_type_samples: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for s in samples:
        gt = get_goal_type(s)
        by_type_samples[gt].append(s)

    goal_types = sorted(by_type_samples.keys())
    source_ratios: List[float] = []
    binding_ratios: List[float] = []

    for gt in goal_types:
        ss = by_type_samples[gt]
        if not ss:
            source_ratios.append(0.0)
            binding_ratios.append(0.0)
            continue

        src_sum = 0.0
        bind_sum = 0.0

        for s in ss:
            meta = get_meta(s)
            src_sum += float(meta.get("source_consistency", 0))
            bind_sum += float(meta.get("binding_consistency", 0))

        n = float(len(ss))
        source_ratios.append(src_sum / n)
        binding_ratios.append(bind_sum / n)

    return goal_types, source_ratios, binding_ratios


def plot_scene_consistency_single(samples: List[Dict[str, Any]], out_path: Path) -> None:
    """
    Single benchmark: plot source/binding consistency per goal_type.
    """
    goal_types, source_ratios, binding_ratios = compute_scene_consistency_single(samples)

    fig, ax = plt.subplots(figsize=(8.5, 4))
    x = np.arange(len(goal_types))
    width = 0.3

    ax.bar(x - width / 2, source_ratios, width,
           color=LIGHT_COLORS[0], label="Source consistency")
    ax.bar(x + width / 2, binding_ratios, width,
           color=LIGHT_COLORS[3], label="Binding consistency")

    ax.set_xticks(x)
    ax.set_xticklabels(goal_types, rotation=45, ha="right")
    ax.set_ylabel("Fraction")
    ax.set_ylim(0, 1.0)
    ax.set_title("Scene Consistency per Goal Type")
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# ============================================================
# Part 4: Multi-benchmark statistics (trends over scale)
# ============================================================

def compute_dataset_stats(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute statistics for one benchmark (one json/jsonl file):
    - total_size: total sample count
    - source_consistency_global: global mean source consistency
    - binding_consistency_global: global mean binding consistency
    - per_type: fine-grained stats per goal_type, including:
        * num_samples
        * unique_instrs
        * near_dup_rate
        * ttr
        * source_consistency
        * binding_consistency
    """
    stats: Dict[str, Any] = {}
    stats["total_size"] = len(samples)

    # Global consistency
    src_vals_all: List[float] = []
    bind_vals_all: List[float] = []
    for s in samples:
        meta = get_meta(s)
        src_vals_all.append(float(meta.get("source_consistency", 0)))
        bind_vals_all.append(float(meta.get("binding_consistency", 0)))
    if src_vals_all:
        stats["source_consistency_global"] = sum(src_vals_all) / len(src_vals_all)
        stats["binding_consistency_global"] = sum(bind_vals_all) / len(bind_vals_all)
    else:
        stats["source_consistency_global"] = 0.0
        stats["binding_consistency_global"] = 0.0

    # Stats per goal_type
    by_type_samples: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for s in samples:
        gt = get_goal_type(s)
        by_type_samples[gt].append(s)

    per_type: Dict[str, Dict[str, Any]] = {}
    for gt, ss in by_type_samples.items():
        instrs = [s.get("instruction", "") for s in ss if s.get("instruction")]
        unique_instrs = len(set(instrs))

        near_dup = estimate_near_duplicate_rate(instrs) if instrs else 0.0

        tokens: List[str] = []
        for t in instrs:
            tokens.extend(tokenize(t))
        if tokens:
            ttr = len(set(tokens)) / float(len(tokens))
        else:
            ttr = 0.0

        src_vals: List[float] = []
        bind_vals: List[float] = []
        for s in ss:
            meta = get_meta(s)
            src_vals.append(float(meta.get("source_consistency", 0)))
            bind_vals.append(float(meta.get("binding_consistency", 0)))
        src_mean = sum(src_vals) / len(src_vals) if src_vals else 0.0
        bind_mean = sum(bind_vals) / len(bind_vals) if bind_vals else 0.0

        per_type[gt] = {
            "num_samples": len(ss),
            "unique_instrs": unique_instrs,
            "near_dup_rate": near_dup,
            "ttr": ttr,
            "source_consistency": src_mean,
            "binding_consistency": bind_mean,
        }

    stats["per_type"] = per_type
    return stats


def plot_unique_instructions_vs_size_per_type(
    benchmarks: List[Dict[str, Any]],
    out_path: Path,
) -> None:
    """
    Plot 1: one subplot per goal_type,
    x-axis: benchmark total sample count,
    y-axis: unique instruction count for that goal_type.
    """
    # Collect all observed goal_types
    all_goal_types = sorted({gt for b in benchmarks for gt in b["per_type"].keys()})
    if not all_goal_types:
        return

    n_types = len(all_goal_types)
    cols = min(3, n_types)
    rows = math.ceil(n_types / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3.5 * rows), sharex=True)
    axes = np.array(axes).reshape(-1)

    for idx, gt in enumerate(all_goal_types):
        ax = axes[idx]
        xs = []
        ys = []
        for b in benchmarks:
            if gt in b["per_type"]:
                xs.append(b["total_size"])
                ys.append(b["per_type"][gt]["unique_instrs"])
        if not xs:
            ax.set_visible(False)
            continue

        order = np.argsort(xs)
        xs = np.array(xs)[order]
        ys = np.array(ys)[order]

        ax.plot(xs, ys, marker="o", linestyle="-",
                color=LIGHT_COLORS[idx % len(LIGHT_COLORS)])
        ax.set_title(gt)
        ax.set_ylabel("# unique instructions")

    # Unify x-axis labels
    for ax in axes:
        if ax.get_visible():
            ax.set_xlabel("Benchmark size (#samples)")

    fig.suptitle("Unique Instructions vs Benchmark Size (per goal_type)", y=0.95)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_metric_vs_size_multi(
    benchmarks: List[Dict[str, Any]],
    metric_key: str,
    title: str,
    ylabel: str,
    out_path: Path,
) -> None:
    """
    Plot 2/3: for a given metric_key, draw one curve per goal_type on the same axes:
    - metric_key in ["near_dup_rate", "ttr"]
    """
    all_goal_types = sorted({gt for b in benchmarks for gt in b["per_type"].keys()})
    if not all_goal_types:
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    for idx, gt in enumerate(all_goal_types):
        xs = []
        ys = []
        for b in benchmarks:
            pt = b["per_type"]
            if gt in pt:
                xs.append(b["total_size"])
                ys.append(pt[gt][metric_key])
        if not xs:
            continue
        order = np.argsort(xs)
        xs = np.array(xs)[order]
        ys = np.array(ys)[order]

        ax.plot(
            xs,
            ys,
            marker="o",
            linestyle="-",
            color=LIGHT_COLORS[idx % len(LIGHT_COLORS)],
            label=gt,
        )

    ax.set_xlabel("Benchmark size (#samples)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_global_consistency_vs_size(
    benchmarks: List[Dict[str, Any]],
    out_path: Path,
) -> None:
    """
    Plot 4: global (all samples) mean source/binding consistency vs benchmark size.
    """
    if not benchmarks:
        return

    xs = [b["total_size"] for b in benchmarks]
    src = [b["source_consistency_global"] for b in benchmarks]
    bind = [b["binding_consistency_global"] for b in benchmarks]

    order = np.argsort(xs)
    xs = np.array(xs)[order]
    src = np.array(src)[order]
    bind = np.array(bind)[order]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(xs, src, marker="o", linestyle="-",
            color=LIGHT_COLORS[0], label="Source consistency")
    ax.plot(xs, bind, marker="o", linestyle="-",
            color=LIGHT_COLORS[3], label="Binding consistency")

    ax.set_xlabel("Benchmark size (#samples)")
    ax.set_ylabel("Average consistency")
    ax.set_ylim(0, 1.0)
    ax.set_title("Global Scene Consistency vs Benchmark Size")
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# ============================================================
# Main entry
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Visualise multi-robot task benchmark.")
    # Compatible with single dataset
    parser.add_argument(
        "--dataset",
        type=str,
        help="Path to a single goals.json or goals.jsonl (for single-benchmark plots)",
    )
    # Multi-dataset mode
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        help="Paths to multiple JSON/JSONL benchmark files (for cross-benchmark plots)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./dataset/semantic/",
        help="Directory to save figures",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for subsampling in near-duplicate estimation",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    # Assemble dataset path list
    dataset_paths: List[str] = []
    if args.datasets:
        dataset_paths.extend(args.datasets)
    if args.dataset:
        dataset_paths.append(args.dataset)

    # dataset_paths = [
    #     "./dataset/semantic/goals/cybertown/bench1000.jsonl",
    #     "./dataset/semantic/goals/cybertown/bench91000.jsonl",
    #     "./dataset/semantic/goals/cybertown/bench181000.jsonl",
    #     "./dataset/semantic/goals/cybertown/bench271000.jsonl",
    #     "./dataset/semantic/goals/cybertown/bench361000.jsonl",
    #     "./dataset/semantic/goals/cybertown/bench541000.jsonl",
    # ]

    # If nothing provided, use old default
    if not dataset_paths:
        dataset_paths = ["./dataset/semantic/goals/cybertown/goals.jsonl"]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load all benchmarks
    benchmarks_info: List[Dict[str, Any]] = []
    for p in dataset_paths:
        path = Path(p)
        samples = load_dataset(path)
        stats = compute_dataset_stats(samples)
        benchmarks_info.append(
            {
                "name": path.stem,
                "path": str(path),
                "samples": samples,
                "stats": stats,
            }
        )
        print(f"Loaded {len(samples)} samples from {path}")

    # If only one benchmark, plot the original per-benchmark figures
    if len(benchmarks_info) == 1:
        info = benchmarks_info[0]
        samples = info["samples"]
        print("Single benchmark mode: plotting per-benchmark figures...")

        # Diversity
        print("Plotting diversity figure (non-dup + lexical diversity)...")
        plot_diversity_summary(samples, out_dir / f"{info['name']}_diversity_instruction_nondup_ttr.png")

        # Difficulty
        print("Plotting difficulty distributions...")
        plot_difficulty_distributions(samples, out_dir)

        # Consistency
        print("Plotting scene consistency figure...")
        plot_scene_consistency_single(samples, out_dir / f"{info['name']}_consistency_scene_metrics.png")

    # Multi-benchmark: analyze trends over scale
    if len(benchmarks_info) >= 2:
        print("Multi-benchmark mode: plotting metrics vs benchmark size...")

        # Collect stats
        bench_stats = [b["stats"] for b in benchmarks_info]

        # Plot 1: unique instruction count vs size per goal_type
        print("Plotting unique instructions vs size per goal_type (subplots)...")
        plot_unique_instructions_vs_size_per_type(
            bench_stats,
            out_dir / "multi_unique_instructions_vs_size_per_type.png",
        )

        # Plot 2: near-duplicate rate vs size (one line per goal_type)
        print("Plotting near-duplicate rate vs size...")
        plot_metric_vs_size_multi(
            bench_stats,
            metric_key="near_dup_rate",
            title="Near-duplicate Rate vs Benchmark Size (per goal_type)",
            ylabel="Near-duplicate rate",
            out_path=out_dir / "multi_near_duplicate_rate_vs_size.png",
        )

        # Plot 3: TTR vs size (one line per goal_type)
        print("Plotting lexical diversity (TTR) vs size...")
        plot_metric_vs_size_multi(
            bench_stats,
            metric_key="ttr",
            title="Lexical Diversity (TTR) vs Benchmark Size (per goal_type)",
            ylabel="TTR",
            out_path=out_dir / "multi_ttr_vs_size.png",
        )

        # Plot 4: global consistency vs size
        print("Plotting global scene consistency vs size...")
        plot_global_consistency_vs_size(
            bench_stats,
            out_dir / "multi_global_consistency_vs_size.png",
        )

    print(f"All figures saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
