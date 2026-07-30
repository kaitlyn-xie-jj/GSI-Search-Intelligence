#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Detailed Method Comparison Plotting Script

Loads raw data from single_run_summary.json for each method in batch_runs session directories,
and generates comparison plots.

Modes:
  general_planning  — Uses only newcase_0 data, generates 5 plots (success rate/time/counts/energy/radar)
  dynamic_replanning — Across all newcase levels, generates 5 plots:
      1. Success Rate (bar chart + delta bar chart + error bars + mean line)
      2. Completion Time (violin plot with box + delta bar chart + mean line)
      3. LLM Call Count (violin plot with box + delta bar chart + mean line)
      4. Average Input Tokens (violin plot with box + delta bar chart + mean line)
      5. Success Rate by Goal Type (2x5 subplot line chart)

Usage:
    python tools/plot_task_planning_comparison.py <session_dir> [--nc 0] [--output-dir figures] [--mode general_planning]
    Example:
    python tools/plot_task_planning_comparison.py results/batch_runs/2026-03-14_23-00-36 --nc 0
    python tools/plot_task_planning_comparison.py results/batch_runs/2026-03-14_23-00-36 --mode dynamic_replanning
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import matplotlib.ticker as mticker

warnings.filterwarnings("ignore", category=UserWarning)

# 全局字体设置为 Arial
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]

# ═══════════════════════════════════════════════════════════════════════════════
# 全局可自定义配置
# ═══════════════════════════════════════════════════════════════════════════════

# 10 个浅色系备用颜色
PALETTE_10 = [
    "#F4A7A3",  # 浅红/珊瑚
    "#7EC8E3",  # 浅蓝
    "#A8D5BA",  # 浅绿
    "#F7C59F",  # 浅橙
    "#C3AED6",  # 浅紫
    "#F9E79F",  # 浅黄
    "#A2D2FF",  # 天蓝
    "#FFB7C5",  # 浅粉
    "#B5EAD7",  # 薄荷绿
    "#D4A5A5",  # 玫瑰灰
]

# 方法显示名称映射（可自定义）
METHOD_DISPLAY_NAMES = {
    "sgi_ft": "SGI-FT",
    "sgi": "SGI",
    "spine": "SPINE",
    "smartllm": "SmartLLM",
    "llamar": "LLaMAR",
    "lipllm": "LipLLM",
}

# 方法排序
METHOD_ORDER = ["sgi_ft", "sgi", "spine", "smartllm", "lipllm", "llamar"]

# 任务类型显示名称
GOAL_TYPE_DISPLAY = {
    "area_search": "Area\nSearch",
    "assembly": "Assembly",
    "emergency_response": "Emergency\nResponse",
    "evidence_collection": "Evidence\nCollection",
    "guidance": "Guidance",
    "patrol": "Patrol",
    "target_following": "Target\nFollowing",
    "traffic_enforcement": "Traffic\nEnforcement",
    "transport": "Transport",
    "verbal_broadcast": "Verbal\nBroadcast",
}

# 任务类型排序
GOAL_TYPE_ORDER = [
    "area_search", "assembly", "emergency_response", "evidence_collection",
    "guidance", "patrol", "target_following", "traffic_enforcement",
    "transport", "verbal_broadcast",
]

# ── 字体与尺寸 ──
FONT_CONFIG = {
    "family": "Arial",
    "title_size": 14,
    "axis_label_size": 14,
    "tick_label_size": 14,
    "legend_size": 14,
    "annotation_size": 10,
    "annotation_size_small": 9,
}

# ── y 轴刻度朝内配置 (每张图可单独覆盖) ──
# direction: "in" 刻度线朝内, pad: 负值让标签也移到图内
YTICK_INWARD = {
    "direction": "in",
    "pad": -22,          # 默认内偏量, 各图可按需覆盖
}

# ── 图形尺寸 ──
FIG_SIZE = {
    "violin": (20, 6),
    "time": (20, 8),
    "count": (20, 6),
    "heatmap": (18, 4),
    "radar": (8, 8),
}

# ── 网格 ──
GRID_CONFIG = {
    "color": "#CCCCCC",
    "alpha": 0.5,
    "linestyle": "--",
    "linewidth": 0.5,
}

# ── 输出 ──
OUTPUT_FORMAT = "png"
TRANSPARENT_BG = False
DPI = 150


# ═══════════════════════════════════════════════════════════════════════════════
# 数据加载
# ═══════════════════════════════════════════════════════════════════════════════

def _wilson_ci(successes: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score interval for binomial proportion (95% CI by default).
    Returns (ci_low, ci_high), both in [0, 1]."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    spread = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - spread), min(1.0, centre + spread))


def _load_json(path: Path) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _load_jsonl_goal_map(jsonl_path: Path) -> Dict[str, str]:
    """Load run_dir -> goal_type mapping from summary.jsonl."""
    mapping = {}
    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                run_dir = Path(rec.get("run_dir", "")).name
                goal_type = rec.get("goal_type", "unknown")
                mapping[run_dir] = goal_type
    except Exception:
        pass
    return mapping


def load_all_raw_data(session_dir: Path, nc_level: int) -> Dict[str, List[dict]]:
    """
    Load raw single_run_summary data for all methods at the specified newcase level.

    Returns: {method_name: [{"goal_type": ..., "data": single_run_summary_dict}, ...]}
    """
    nc_dir = session_dir / f"newcase_{nc_level}"
    if not nc_dir.exists():
        print(f"[ERROR] Directory does not exist: {nc_dir}")
        sys.exit(1)

    all_data = {}
    for batch_dir in sorted(nc_dir.iterdir()):
        if not batch_dir.is_dir() or not batch_dir.name.startswith("batch_"):
            continue
        method = batch_dir.name.replace("batch_", "")

        # 加载 goal_type 映射
        goal_map = _load_jsonl_goal_map(batch_dir / "summary.jsonl")

        runs = []
        for run_dir in sorted(batch_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            summary_file = run_dir / "single_run_summary.json"
            data = _load_json(summary_file)
            if data is None:
                continue
            goal_type = goal_map.get(run_dir.name, "unknown")
            runs.append({"goal_type": goal_type, "data": data})

        if runs:
            all_data[method] = runs

    return all_data


def load_aggregate_data(session_dir: Path, nc_level: int) -> Dict[str, dict]:
    """Load aggregate data for heatmaps, etc."""
    nc_dir = session_dir / f"newcase_{nc_level}"
    agg = {}
    for batch_dir in sorted(nc_dir.iterdir()):
        if not batch_dir.is_dir() or not batch_dir.name.startswith("batch_"):
            continue
        method = batch_dir.name.replace("batch_", "")
        agg_file = batch_dir / "aggregate_full.json"
        data = _load_json(agg_file)
        if data:
            agg[method] = data
    return agg


def discover_nc_levels(session_dir: Path) -> List[int]:
    """Discover all newcase_N levels under the session directory, return a sorted list."""
    levels = []
    for entry in sorted(session_dir.iterdir()):
        if entry.is_dir() and entry.name.startswith("newcase_"):
            try:
                nc = int(entry.name.split("_", 1)[1])
                levels.append(nc)
            except (ValueError, IndexError):
                continue
    levels.sort()
    return levels


def load_all_nc_raw_data(session_dir: Path, nc_levels: List[int]) -> Dict[int, Dict[str, List[dict]]]:
    """
    Load raw data for all newcase levels.

    Returns: {nc_level: {method_name: [{"goal_type": ..., "data": ...}, ...]}}
    """
    all_nc_data = {}
    for nc in nc_levels:
        all_nc_data[nc] = load_all_raw_data(session_dir, nc)
    return all_nc_data


def load_all_nc_aggregate_data(session_dir: Path, nc_levels: List[int]) -> Dict[int, Dict[str, dict]]:
    """Load aggregate data for all newcase levels."""
    all_nc_agg = {}
    for nc in nc_levels:
        all_nc_agg[nc] = load_aggregate_data(session_dir, nc)
    return all_nc_agg


# ═══════════════════════════════════════════════════════════════════════════════
# 数据提取辅助
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_mean_std(s: str) -> Tuple[float, float]:
    """Parse '0.804±0.396969' format."""
    parts = s.split("±")
    return float(parts[0]), float(parts[1]) if len(parts) > 1 else 0.0


def _get_methods_sorted(all_data: dict) -> List[str]:
    """Sort available methods by METHOD_ORDER."""
    available = set(all_data.keys())
    return [m for m in METHOD_ORDER if m in available] + \
           sorted(available - set(METHOD_ORDER))


def _method_color(method: str, methods: List[str]) -> str:
    idx = methods.index(method) if method in methods else 0
    return PALETTE_10[idx % len(PALETTE_10)]


def _darker(hex_color: str, factor: float = 0.7) -> str:
    """Generate a darker color for borders."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    r, g, b = int(r * factor), int(g * factor), int(b * factor)
    return f"#{r:02x}{g:02x}{b:02x}"


def _extract_per_goal_type(all_data: Dict[str, List[dict]], metric_key: str,
                           from_nested: str = None) -> Dict[str, Dict[str, List[float]]]:
    """
    Extract the raw metric value list for each method under each goal_type.

    Returns: {goal_type: {method: [values...]}}
    """
    result = {}
    for method, runs in all_data.items():
        for run in runs:
            gt = run["goal_type"]
            d = run["data"]
            if from_nested:
                val = d.get(from_nested, {}).get(metric_key)
            else:
                val = d.get(metric_key)
            if val is None:
                continue
            result.setdefault(gt, {}).setdefault(method, []).append(float(val))
    return result


def _extract_overall(all_data: Dict[str, List[dict]], metric_key: str,
                     from_nested: str = None) -> Dict[str, List[float]]:
    """Extract the overall metric value list for each method."""
    result = {}
    for method, runs in all_data.items():
        vals = []
        for run in runs:
            d = run["data"]
            if from_nested:
                val = d.get(from_nested, {}).get(metric_key)
            else:
                val = d.get(metric_key)
            if val is not None:
                vals.append(float(val))
        if vals:
            result[method] = vals
    return result


def _build_x_groups(all_data, metric_key, from_nested=None):
    """Build 11 groups of data: Overall + 10 goal types."""
    overall = _extract_overall(all_data, metric_key, from_nested)
    per_gt = _extract_per_goal_type(all_data, metric_key, from_nested)
    groups = [("Overall", overall)]
    for gt in GOAL_TYPE_ORDER:
        if gt in per_gt:
            groups.append((gt, per_gt[gt]))
    return groups


def _apply_grid(ax):
    ax.grid(
        True,
        color=GRID_CONFIG["color"],
        alpha=GRID_CONFIG["alpha"],
        linestyle=GRID_CONFIG["linestyle"],
        linewidth=GRID_CONFIG["linewidth"],
    )
    ax.set_axisbelow(True)


def _apply_font(ax, title="", xlabel="", ylabel="", ytick_pad=None):
    if title:
        ax.set_title(title, fontsize=FONT_CONFIG["title_size"], fontweight="bold", pad=12)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=FONT_CONFIG["axis_label_size"])
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=FONT_CONFIG["axis_label_size"])
    ax.tick_params(labelsize=FONT_CONFIG["tick_label_size"])
    # y 轴刻度朝内
    pad = ytick_pad if ytick_pad is not None else YTICK_INWARD["pad"]
    ax.tick_params(axis="y", direction=YTICK_INWARD["direction"], pad=pad)
    for label in ax.yaxis.get_ticklabels():
        label.set_ha("left")


def _save_fig(fig, output_dir: Path, filename: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{filename}.{OUTPUT_FORMAT}"
    fig.savefig(
        path,
        format=OUTPUT_FORMAT,
        dpi=DPI,
        bbox_inches="tight",
        transparent=TRANSPARENT_BG,
    )
    plt.close(fig)
    print(f"  -> Saved: {path}")


def _get_x_label(gt_key: str) -> str:
    if gt_key == "Overall":
        return "Overall"
    return GOAL_TYPE_DISPLAY.get(gt_key, gt_key)



# ═══════════════════════════════════════════════════════════════════════════════
# Origin 风格带箱体小提琴图配置 (用于图2的完成时间等连续数据)
# ═══════════════════════════════════════════════════════════════════════════════

VIOLIN_CONFIG = {
    "kde_points": 200,         # KDE 评估点数
    "violin_alpha": 0.35,      # 小提琴填充透明度
    "violin_edge_lw": 1.0,     # 小提琴轮廓线宽
    "tail_shrink": 0.5,        # 小提琴尾巴收缩系数 (0-1, 越小尾巴越短)
    "box_width_ratio": 0.25,   # 箱体宽度占小提琴宽度的比例
    "box_alpha": 0.85,         # 箱体填充透明度
    "box_edge_lw": 1.0,        # 箱体边框线宽
    "whisker_iqr_factor": 1.0, # 须线长度 = Q1/Q3 ± factor*IQR (默认1.0, 比经典1.5更短)
    "whisker_lw": 1.0,         # 须线线宽
    "median_lw": 1.8,          # 中位线线宽
    "median_color": "white",   # 中位线颜色
    "mean_marker": "D",        # 均值标记形状 (菱形)
    "mean_size": 18,           # 均值标记大小
    "cap_lw": 1.0,             # 须帽线宽
    "cap_size_ratio": 0.4,     # 须帽宽度占箱体宽度的比例
}

# 下方堆叠直方图 Y 轴放大倍数 (使规划/分配时间在视觉上更清晰)
TIME_LOWER_SCALE = 5

# 图3 下方突发事件次数 Y 轴放大倍数
INCIDENT_LOWER_SCALE = 5


def _kde_density(data: np.ndarray, n_points: int = 200,
                 clip_lo: float = None, clip_hi: float = None):
    """Gaussian KDE with automatic bandwidth, returns (grid, density).
    By default clips KDE to data range [min, max] to avoid overly long tails."""
    from scipy.stats import gaussian_kde
    data = np.asarray(data, dtype=float)
    if clip_lo is None:
        clip_lo = data.min()
    if clip_hi is None:
        clip_hi = data.max()
    # 不再额外留 margin，严格裁剪到数据范围
    lo, hi = clip_lo, clip_hi
    if lo == hi:
        # 所有值相同，给一点微小范围
        lo, hi = lo - 1.0, hi + 1.0
    if len(set(data)) <= 1:
        grid = np.linspace(lo, hi, n_points)
        density = np.zeros(n_points)
        idx = np.argmin(np.abs(grid - data[0]))
        density[max(0, idx - 2):idx + 3] = 1.0
        return grid, density
    kde = gaussian_kde(data)
    grid = np.linspace(lo, hi, n_points)
    density = kde(grid)
    return grid, density


def _draw_origin_violin(ax, pos: float, vals: np.ndarray, half_w: float,
                        color: str, edge_color: str, vc: dict,
                        annotate_median: bool = True):
    """
    Draw an Origin-style violin with embedded box plot on ax.

    Outer layer: Symmetric KDE contour (filled + edge line)
    Inner: Rectangle box (Q1-Q3) + white median line + whiskers + caps + mean diamond
    """
    vals = np.asarray(vals, dtype=float)
    if len(vals) == 0:
        return

    # ── KDE 小提琴轮廓 (使用 tail_shrink 收缩尾巴) ──
    q1, med, q3 = np.percentile(vals, [25, 50, 75])
    iqr = q3 - q1
    shrink = vc.get("tail_shrink", 1.0)
    # 将数据范围向 IQR 中心收缩: 完整范围 * shrink
    data_lo, data_hi = vals.min(), vals.max()
    range_lo = med - (med - data_lo) * shrink
    range_hi = med + (data_hi - med) * shrink
    grid, density = _kde_density(vals, n_points=vc["kde_points"],
                                 clip_lo=range_lo, clip_hi=range_hi)
    if density.max() > 0:
        density = density / density.max() * half_w

    ax.fill_betweenx(grid, pos - density, pos + density,
                     facecolor=color, alpha=vc["violin_alpha"],
                     edgecolor="none", zorder=2)
    ax.plot(pos - density, grid, color=edge_color,
            linewidth=vc["violin_edge_lw"], alpha=0.8, zorder=3)
    ax.plot(pos + density, grid, color=edge_color,
            linewidth=vc["violin_edge_lw"], alpha=0.8, zorder=3)

    # ── 内嵌箱线图 ──
    wf = vc.get("whisker_iqr_factor", 1.5)
    whisker_lo = np.min(vals[vals >= q1 - wf * iqr]) if iqr > 0 else vals.min()
    whisker_hi = np.max(vals[vals <= q3 + wf * iqr]) if iqr > 0 else vals.max()
    mean_v = vals.mean()

    box_half_w = half_w * vc["box_width_ratio"]
    cap_half_w = box_half_w * vc["cap_size_ratio"]

    # 箱体矩形
    box_rect = plt.Rectangle(
        (pos - box_half_w, q1), 2 * box_half_w, q3 - q1,
        facecolor=color, edgecolor=edge_color,
        alpha=vc["box_alpha"], linewidth=vc["box_edge_lw"], zorder=4)
    ax.add_patch(box_rect)

    # 中位线
    ax.hlines(med, pos - box_half_w, pos + box_half_w,
              color=vc["median_color"], linewidth=vc["median_lw"], zorder=5)

    # 须线
    ax.vlines(pos, whisker_lo, q1, color=edge_color,
              linewidth=vc["whisker_lw"], zorder=4)
    ax.vlines(pos, q3, whisker_hi, color=edge_color,
              linewidth=vc["whisker_lw"], zorder=4)

    # 须帽
    ax.hlines(whisker_lo, pos - cap_half_w, pos + cap_half_w,
              color=edge_color, linewidth=vc["cap_lw"], zorder=4)
    ax.hlines(whisker_hi, pos - cap_half_w, pos + cap_half_w,
              color=edge_color, linewidth=vc["cap_lw"], zorder=4)

    # 均值菱形
    ax.scatter([pos], [mean_v], marker=vc["mean_marker"],
               s=vc["mean_size"], color=edge_color,
               edgecolor="white", linewidth=0.5, zorder=6)

    # 标注中位数
    if annotate_median:
        ax.text(pos, whisker_hi + (whisker_hi - whisker_lo) * 0.03,
                f"{med:.1f}", ha="center", va="bottom",
                fontsize=FONT_CONFIG["annotation_size_small"],
                color=_darker(edge_color, 0.6), fontweight="light")


# ═══════════════════════════════════════════════════════════════════════════════
# 图1: 成功率 — 柱状图 + 误差棒 + 标注均值
# ═══════════════════════════════════════════════════════════════════════════════

def plot_success_rate_bar(all_data, methods, output_dir):
    """Success rate bar chart: N bars per group, with short error bars, mean annotated on top."""
    print("[Fig 1] Success Rate Bar Chart")
    groups = _build_x_groups(all_data, "success")
    for gname, gdata in groups:
        for m in gdata:
            gdata[m] = [1.0 if v else 0.0 for v in gdata[m]]

    n_groups = len(groups)
    n_methods = len(methods)
    fig, ax = plt.subplots(figsize=FIG_SIZE["violin"])

    group_width = 0.8
    bar_width = group_width / n_methods

    for gi, (gname, gdata) in enumerate(groups):
        base_x = gi
        for mi, method in enumerate(methods):
            vals = gdata.get(method, [0.0])
            n = len(vals)
            mean_v = np.mean(vals)
            successes = int(round(sum(vals)))
            ci_lo, ci_hi = _wilson_ci(successes, n)
            err_lo = mean_v - ci_lo   # 下方误差
            err_hi = ci_hi - mean_v   # 上方误差
            pos = base_x - group_width / 2 + bar_width * (mi + 0.5)
            color = _method_color(method, methods)
            edge_color = _darker(color)

            ax.bar(pos, mean_v, width=bar_width * 0.75,
                   color=color, edgecolor=edge_color, alpha=0.75,
                   linewidth=0.8, zorder=2)
            ax.errorbar(pos, mean_v, yerr=[[err_lo], [err_hi]], fmt="none",
                        ecolor=edge_color, capsize=2.5, capthick=0.8,
                        elinewidth=0.8, zorder=3)
            # 标注均值
            ax.text(pos, ci_hi + 0.015, f"{mean_v:.2f}",
                    ha="center", va="bottom",
                    fontsize=FONT_CONFIG["annotation_size_small"],
                    color=_darker(color, 0.5))

    ax.set_xticks(range(n_groups))
    ax.set_xticklabels([_get_x_label(g[0]) for g in groups],
                       fontsize=FONT_CONFIG["tick_label_size"])
    ax.set_xlim(-0.6, n_groups - 0.4)
    ax.set_ylim(0, 1.15)

    _apply_font(ax, title="Success Rate by Task Type", ylabel="Success Rate")
    _apply_grid(ax)

    handles = [mpatches.Patch(facecolor=_method_color(m, methods), alpha=0.75,
                              edgecolor=_darker(_method_color(m, methods)),
                              label=METHOD_DISPLAY_NAMES.get(m, m))
               for m in methods]
    ax.legend(handles=handles, loc="lower left", fontsize=FONT_CONFIG["legend_size"],
              framealpha=0.8, ncol=n_methods)

    fig.tight_layout()
    _save_fig(fig, output_dir, "fig1_success_rate_bar")


# ═══════════════════════════════════════════════════════════════════════════════
# 图2: 时间指标 — 上方带箱体小提琴图(完成时间) + 下方堆叠直方图(规划+分配)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_time_metrics(all_data, methods, agg_data, output_dir):
    """
    Upper: Total completion time (elapsed_sec) Origin-style violin plot with box
    Lower: Average planning time + average allocation time stacked bar chart (downward)
    """
    print("[Fig 2] Time Metrics (Violin + Stacked Bar)")
    groups_elapsed = _build_x_groups(all_data, "elapsed_sec")
    groups_plan = _build_x_groups(all_data, "planning_duration_mean")
    groups_alloc = _build_x_groups(all_data, "allocation_duration_mean")

    vc = VIOLIN_CONFIG
    n_groups = len(groups_elapsed)
    n_methods = len(methods)
    fig, ax = plt.subplots(figsize=FIG_SIZE["time"])

    group_width = 0.8
    item_width = group_width / n_methods

    # ── 上方: 带箱体小提琴图 (elapsed_sec) ──
    for gi, (gname, gdata) in enumerate(groups_elapsed):
        base_x = gi
        for mi, method in enumerate(methods):
            vals = gdata.get(method, [])
            if not vals:
                continue
            pos = base_x - group_width / 2 + item_width * (mi + 0.5)
            half_w = item_width * 0.42
            color = _method_color(method, methods)
            edge_color = _darker(color)
            _draw_origin_violin(ax, pos, np.array(vals), half_w,
                                color, edge_color, vc, annotate_median=True)

    # ── 下方: 堆叠直方图 (planning + allocation, 向下, 放大 TIME_LOWER_SCALE 倍) ──
    scale = TIME_LOWER_SCALE
    for gi in range(n_groups):
        gdata_plan = groups_plan[gi][1]
        gdata_alloc = groups_alloc[gi][1]
        base_x = gi
        for mi, method in enumerate(methods):
            plan_vals = gdata_plan.get(method, [])
            alloc_vals = gdata_alloc.get(method, [])
            if not plan_vals:
                continue
            plan_mean = np.mean(plan_vals)
            alloc_mean = np.mean(alloc_vals) if alloc_vals else 0.0
            pos = base_x - group_width / 2 + item_width * (mi + 0.5)
            color = _method_color(method, methods)
            edge_color = _darker(color)

            # 规划时间（向下, 放大 scale 倍显示）
            ax.bar(pos, -plan_mean * scale, width=item_width * 0.7, bottom=0,
                   color=color, edgecolor=edge_color, alpha=0.7, linewidth=0.8)
            # 分配时间堆叠（继续向下），加斜线图案
            ax.bar(pos, -alloc_mean * scale, width=item_width * 0.7,
                   bottom=-plan_mean * scale,
                   color=color, edgecolor=edge_color, alpha=0.7, linewidth=0.8,
                   hatch="///")

            # 规划时间标注在其柱状图内部中心
            ax.text(pos, -plan_mean * scale / 2, f"{plan_mean:.1f}",
                    ha="center", va="center",
                    fontsize=FONT_CONFIG["annotation_size_small"],
                    color=_darker(color, 0.4), fontweight="light")
            # 分配时间标注在堆叠柱底部（下方）
            if alloc_mean > 0.01:
                total_h = (plan_mean + alloc_mean) * scale
                ax.text(pos, -total_h - 20, f"{alloc_mean:.2f}",
                        ha="center", va="top",
                        fontsize=FONT_CONFIG["annotation_size_small"] - 1,
                        color=_darker(color, 0.4), fontweight="light")

    # 零线
    ax.axhline(y=0, color="#666666", linewidth=0.8, zorder=2)

    # 下方坐标轴刻度显示为正值，且还原放大倍数（时间不可能为负）
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f"{abs(v):.0f}" if v >= 0 else f"{abs(v) / TIME_LOWER_SCALE:.1f}"))

    ax.set_xticks(range(n_groups))
    ax.set_xticklabels([_get_x_label(g[0]) for g in groups_elapsed],
                       fontsize=FONT_CONFIG["tick_label_size"])
    ax.set_xlim(-0.6, n_groups - 0.4)

    ax.set_ylabel(f"↓ Planning / Allocation Time (s, ×{TIME_LOWER_SCALE})          ↑ Total Completion Time (s)",
                  fontsize=FONT_CONFIG["axis_label_size"])
    _apply_font(ax, title="Time Metrics by Task Type")
    _apply_grid(ax)

    handles = []
    for m in methods:
        c = _method_color(m, methods)
        handles.append(mpatches.Patch(facecolor=c, alpha=0.7,
                                      edgecolor=_darker(c),
                                      label=METHOD_DISPLAY_NAMES.get(m, m)))
    handles.append(mpatches.Patch(facecolor="#CCCCCC", edgecolor="#999999",
                                  label="Planning Time (solid)"))
    handles.append(mpatches.Patch(facecolor="#CCCCCC", edgecolor="#999999",
                                  hatch="//", label="Allocation Time (hatched)"))
    ax.legend(handles=handles, loc="upper left", fontsize=FONT_CONFIG["legend_size"],
              framealpha=0.8, ncol=min(n_methods + 2, 4))

    fig.tight_layout()
    _save_fig(fig, output_dir, "fig2_time_metrics")



# ═══════════════════════════════════════════════════════════════════════════════
# 图3: 次数指标 — LLM调用次数(直方图) + 突发事件次数(折线图)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_count_metrics(all_data, methods, output_dir):
    """
    Upper (↑): LLM call count (bar chart + SEM error bars + annotations)
    Lower (↓): Incident count (line chart + SEM error bars + annotations), scaled by INCIDENT_LOWER_SCALE
    """
    print("[Fig 3] Count Metrics (LLM Calls + Incidents)")
    groups_llm = _build_x_groups(all_data, "llm_calls")
    groups_nc = _build_x_groups(all_data, "newcase_total")

    n_groups = len(groups_llm)
    n_methods = len(methods)
    fig, ax = plt.subplots(figsize=FIG_SIZE["count"])

    group_width = 0.8
    bar_width = group_width / n_methods
    scale = INCIDENT_LOWER_SCALE

    # ── 上方: LLM 调用次数柱状图 ──
    for gi, (gname, gdata) in enumerate(groups_llm):
        base_x = gi
        for mi, method in enumerate(methods):
            vals = gdata.get(method, [])
            if not vals:
                continue
            mean_v = np.mean(vals)
            sem_v = np.std(vals) / np.sqrt(len(vals))
            pos = base_x - group_width / 2 + bar_width * (mi + 0.5)
            color = _method_color(method, methods)
            edge_color = _darker(color)

            ax.bar(pos, mean_v, width=bar_width * 0.7,
                   color=color, edgecolor=edge_color, alpha=0.7, linewidth=0.8,
                   zorder=2)
            ax.errorbar(pos, mean_v, yerr=sem_v, fmt="none",
                        ecolor=edge_color, capsize=2, capthick=0.8,
                        elinewidth=0.8, zorder=3)
            ax.text(pos, mean_v + sem_v + 1, f"{mean_v:.1f}",
                    ha="center", va="bottom",
                    fontsize=FONT_CONFIG["annotation_size_small"],
                    color=_darker(color, 0.5), fontweight="light")

    # ── 下方: 突发事件次数折线图 (向下, 放大 scale 倍) ──
    for mi, method in enumerate(methods):
        color = _method_color(method, methods)
        edge_color = _darker(color)
        xs, ys_raw, ys_plot, errs_plot = [], [], [], []
        for gi, (gname, gdata) in enumerate(groups_nc):
            vals = gdata.get(method, [])
            if not vals:
                continue
            mean_v = np.mean(vals)
            sem_v = np.std(vals) / np.sqrt(len(vals))
            pos = gi - group_width / 2 + bar_width * (mi + 0.5)
            xs.append(pos)
            ys_raw.append(mean_v)
            ys_plot.append(-mean_v * scale)
            errs_plot.append(sem_v * scale)

        if xs:
            ax.errorbar(xs, ys_plot, yerr=errs_plot, fmt="-o", color=edge_color,
                        markersize=4, capsize=2, capthick=0.8,
                        elinewidth=1.8, linewidth=2.0, alpha=0.8, zorder=4,
                        markerfacecolor=color, markeredgecolor=edge_color)
            for x, yr, yp, ep in zip(xs, ys_raw, ys_plot, errs_plot):
                ax.text(x, yp - ep - 1, f"{yr:.2f}",
                        ha="center", va="top",
                        fontsize=FONT_CONFIG["annotation_size_small"],
                        color=edge_color)

    # 零线
    ax.axhline(y=0, color="#666666", linewidth=0.8, zorder=2)

    # Y 轴刻度: 上方原值, 下方还原放大倍数并显示正值
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda v, _: f"{abs(v):.0f}" if v >= 0 else f"{abs(v) / scale:.1f}"))

    # X 轴
    ax.set_xticks(range(n_groups))
    ax.set_xticklabels([_get_x_label(g[0]) for g in groups_llm],
                       fontsize=FONT_CONFIG["tick_label_size"])
    ax.set_xlim(-0.6, n_groups - 0.4)

    ax.set_ylabel(f"↓ Incident Count (×{scale})          ↑ LLM Calls",
                  fontsize=FONT_CONFIG["axis_label_size"])
    _apply_font(ax, title="Count Metrics by Task Type")
    _apply_grid(ax)

    # 图例
    bar_handles = [mpatches.Patch(facecolor=_method_color(m, methods), alpha=0.7,
                                  edgecolor=_darker(_method_color(m, methods)),
                                  label=f"{METHOD_DISPLAY_NAMES.get(m, m)} (LLM)")
                   for m in methods]
    line_handles = [Line2D([0], [0], color=_darker(_method_color(m, methods)),
                           marker="o", markersize=4,
                           markerfacecolor=_method_color(m, methods),
                           label=f"{METHOD_DISPLAY_NAMES.get(m, m)} (Incidents)")
                    for m in methods]
    # ax.legend(handles=bar_handles + line_handles, loc="upper left",
    #           fontsize=FONT_CONFIG["legend_size"] - 1, framealpha=0.8,
    #           ncol=2)

    fig.tight_layout()
    _save_fig(fig, output_dir, "fig3_count_metrics")


# ═══════════════════════════════════════════════════════════════════════════════
# 图4: 能量消耗热力图
# ═══════════════════════════════════════════════════════════════════════════════

def plot_energy_heatmap(all_data, methods, output_dir):
    """
    Energy consumption boxplot: origin at top-left, x-axis downward (task types), y-axis rightward (energy values).
    Each group has N horizontal boxplots (one per method).
    Uses whis=1.0 IQR to shorten whiskers, no outliers shown, mean annotated on the right.
    """
    print("[Fig 4] Energy Boxplot")
    groups = _build_x_groups(all_data, "total_energy")

    n_groups = len(groups)
    n_methods = len(methods)
    group_width = 0.8
    item_width = group_width / n_methods

    fig, ax = plt.subplots(figsize=(14, n_groups * 1.2 + 1.5))

    for gi, (gname, gdata) in enumerate(groups):
        for mi, method in enumerate(methods):
            vals = gdata.get(method, [])
            if not vals:
                continue
            pos = gi + (mi - (n_methods - 1) / 2) * item_width
            color = _method_color(method, methods)
            edge_color = _darker(color)

            bp = ax.boxplot(vals, positions=[pos], widths=item_width * 0.7,
                            vert=False, patch_artist=True,
                            whis=0.5,       # 缩短须线: 1.0×IQR (比默认 1.5 更紧凑)
                            showfliers=False,  # 不显示离群点
                            boxprops=dict(facecolor=color, edgecolor=edge_color,
                                          alpha=0.7, linewidth=0.8),
                            whiskerprops=dict(color=edge_color, linewidth=0.8),
                            capprops=dict(color=edge_color, linewidth=0.8),
                            medianprops=dict(color="white", linewidth=1.5),
                            zorder=2)
            # 均值菱形
            mean_v = np.mean(vals)
            ax.scatter([mean_v], [pos], marker="D", s=14, color=edge_color,
                       edgecolor="white", linewidth=0.3, zorder=5)

            # 右侧标注均值 (须线右端 + 小偏移)
            q3 = np.percentile(vals, 75)
            iqr_v = q3 - np.percentile(vals, 25)
            cap_hi = np.max([v for v in vals if v <= q3 + 0.5 * iqr_v]) if iqr_v > 0 else q3
            ax.text(cap_hi, pos, f"  {mean_v:.0f}",
                    ha="left", va="center",
                    fontsize=FONT_CONFIG["annotation_size_small"],
                    color=_darker(color, 0.5), fontweight="bold")

    # 坐标原点在左上角: x 轴向下 → 反转 y 轴
    ax.invert_yaxis()

    # Y 轴 (任务类型, 竖向排列) — 刻度标签斜 45 度
    ax.set_yticks(range(n_groups))
    ax.set_yticklabels([_get_x_label(g[0]) for g in groups],
                       fontsize=FONT_CONFIG["tick_label_size"],
                       rotation=45, ha="right")

    # X 轴 (能量值) 放在顶部
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", labelsize=FONT_CONFIG["tick_label_size"])

    ax.set_xlabel("Total Energy", fontsize=FONT_CONFIG["axis_label_size"])
    ax.set_xlim(-200, 4600)
    ax.set_title("Energy Consumption by Task Type",
                 fontsize=FONT_CONFIG["title_size"], pad=30)
    _apply_grid(ax)

    # 图例
    handles = [mpatches.Patch(facecolor=_method_color(m, methods), alpha=0.7,
                              edgecolor=_darker(_method_color(m, methods)),
                              label=METHOD_DISPLAY_NAMES.get(m, m))
               for m in methods]
    ax.legend(handles=handles, loc="lower right",
              fontsize=FONT_CONFIG["legend_size"], framealpha=0.8)

    fig.tight_layout()
    _save_fig(fig, output_dir, "fig4_energy_boxplot")



# ═══════════════════════════════════════════════════════════════════════════════
# 图5: 总体雷达图
# ═══════════════════════════════════════════════════════════════════════════════

def plot_radar(agg_data, methods, output_dir):
    """
    Radar chart with 7 metrics, normalized to 0-1, closer to 1 is better.

    Metrics:
      1. success_rate (higher is better, used directly)
      2. elapsed_sec (lower is better, inverse nonlinear normalization)
      3. planning_duration (lower is better, inverse nonlinear normalization)
      4. allocation_duration (lower is better, inverse nonlinear normalization)
      5. llm_calls (lower is better, linear inverse normalization)
      6. newcase_total (lower is better, linear inverse normalization)
      7. total_energy (lower is better, linear inverse normalization)
    """
    print("[Fig 5] Radar Chart")

    metric_keys = [
        "success_rate", "elapsed_sec", "planning_duration",
        "allocation_duration", "llm_calls", "newcase_total", "total_energy",
    ]
    metric_labels = [
        "Success Rate", "Completion\nTime", "Planning\nTime",
        "Allocation\nTime", "LLM Calls", "Incidents", "Energy",
    ]
    # 越小越好的指标
    smaller_better = {"elapsed_sec", "planning_duration", "allocation_duration",
                      "llm_calls", "newcase_total", "total_energy"}
    # 非线性归一化的指标（时间类）
    nonlinear_keys = {"elapsed_sec", "planning_duration", "allocation_duration"}

    # 收集所有方法的原始均值
    raw_values = {}  # {metric: {method: mean_value}}
    for mk in metric_keys:
        raw_values[mk] = {}
        for method in methods:
            agg = agg_data.get(method, {}).get("overall", {})
            val_str = agg.get(mk, "0±0")
            mean_val, _ = _parse_mean_std(val_str)
            raw_values[mk][method] = mean_val

    # 归一化
    normalized = {}  # {metric: {method: normalized_value}}
    for mk in metric_keys:
        vals = list(raw_values[mk].values())
        vmin, vmax = min(vals), max(vals)
        normalized[mk] = {}

        if mk == "success_rate":
            # 直接用，已经在 0-1
            for m in methods:
                normalized[mk][m] = raw_values[mk][m]
        elif mk in smaller_better:
            if vmax == vmin:
                for m in methods:
                    normalized[mk][m] = 1.0
            elif mk in nonlinear_keys:
                # 非线性反向归一化: 1 - ((v - vmin) / (vmax - vmin))^0.5
                for m in methods:
                    v = raw_values[mk][m]
                    ratio = (v - vmin) / (vmax - vmin)
                    normalized[mk][m] = 1.0 - ratio ** 0.5
            else:
                # 线性反向归一化: 1 - (v - vmin) / (vmax - vmin)
                for m in methods:
                    v = raw_values[mk][m]
                    normalized[mk][m] = 1.0 - (v - vmin) / (vmax - vmin)

    # 绘制雷达图
    n_metrics = len(metric_keys)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # 闭合

    fig, ax = plt.subplots(figsize=FIG_SIZE["radar"], subplot_kw=dict(polar=True))

    for mi, method in enumerate(methods):
        values = [normalized[mk][method] for mk in metric_keys]
        values += values[:1]  # 闭合
        color = _method_color(method, methods)
        edge_color = _darker(color)

        ax.plot(angles, values, "o-", linewidth=2.0, color=edge_color,
                markersize=4, label=METHOD_DISPLAY_NAMES.get(method, method))
        ax.fill(angles, values, alpha=0.15, color=color)

    # 设置刻度
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontsize=FONT_CONFIG["tick_label_size"])
    ax.set_ylim(0, 1)
    ax.set_yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_yticklabels(["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"],
                       fontsize=FONT_CONFIG["tick_label_size"] - 1)
    ax.yaxis.grid(True, color=GRID_CONFIG["color"], alpha=GRID_CONFIG["alpha"],
                  linestyle=GRID_CONFIG["linestyle"])
    ax.xaxis.grid(True, color=GRID_CONFIG["color"], alpha=GRID_CONFIG["alpha"],
                  linestyle=GRID_CONFIG["linestyle"])

    ax.set_title("Overall Performance Radar (closer to 1 = better)",
                 fontsize=FONT_CONFIG["title_size"], fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1),
              fontsize=FONT_CONFIG["legend_size"], framealpha=0.8)

    fig.tight_layout()
    _save_fig(fig, output_dir, "fig5_radar")


# ═══════════════════════════════════════════════════════════════════════════════
# Dynamic Replanning 模式: 跨 newcase 等级对比图
# ═══════════════════════════════════════════════════════════════════════════════

# 增量柱状图的斜线图案
DELTA_HATCH = "///"

# Dynamic Replanning 图形尺寸
DR_FIG_SIZE = {
    "bar": (14, 6),
    "violin": (14, 6),
    "goal_grid": (22, 8),
}

# 图5 子图网格精细布局 (可自定义)
GOAL_GRID_LAYOUT = {
    "fig_width": 22,          # 总图宽 (英寸)
    "fig_height": 8,          # 总图高 (英寸)
    "rows": 2,
    "cols": 5,
    "wspace": 0.01,           # 列间距 (占子图宽度的比例)
    "hspace": 0.05,           # 行间距 (占子图高度的比例)
    "left": 0.01,             # 左边距
    "right": 0.99,            # 右边距
    "top": 0.95,              # 上边距 (给标题和图例留空间)
    "bottom": 0.10,           # 下边距 (给共用x轴标签留空间)
}


def _compute_nc_overall_metric(all_nc_data: Dict[int, Dict[str, List[dict]]],
                                nc_levels: List[int], methods: List[str],
                                metric_key: str, from_nested: str = None,
                                as_bool: bool = False):
    """
    Compute the overall metric for each method at each NC level.

    Returns:
        means: {method: [mean_nc0, mean_nc1, ...]}
        stds:  {method: [std_nc0, std_nc1, ...]}
        raw:   {method: [[vals_nc0], [vals_nc1], ...]}
    """
    means = {m: [] for m in methods}
    stds = {m: [] for m in methods}
    raw = {m: [] for m in methods}

    for nc in nc_levels:
        nc_data = all_nc_data.get(nc, {})
        for m in methods:
            runs = nc_data.get(m, [])
            vals = []
            for run in runs:
                d = run["data"]
                if from_nested:
                    v = d.get(from_nested, {}).get(metric_key)
                else:
                    v = d.get(metric_key)
                if v is not None:
                    if as_bool:
                        vals.append(1.0 if v else 0.0)
                    else:
                        vals.append(float(v))
            raw[m].append(vals)
            if vals:
                means[m].append(np.mean(vals))
                stds[m].append(np.std(vals))
            else:
                means[m].append(0.0)
                stds[m].append(0.0)

    return means, stds, raw


def _compute_nc_per_goal_type_metric(all_nc_data: Dict[int, Dict[str, List[dict]]],
                                      nc_levels: List[int], methods: List[str],
                                      metric_key: str, as_bool: bool = False):
    """
    Compute the metric for each method at each NC level and each goal_type.

    Returns: {goal_type: {method: {"means": [...], "stds": [...]}}}
    """
    result = {}
    for nc_idx, nc in enumerate(nc_levels):
        nc_data = all_nc_data.get(nc, {})
        for m in methods:
            runs = nc_data.get(m, [])
            gt_vals = {}
            for run in runs:
                gt = run["goal_type"]
                d = run["data"]
                v = d.get(metric_key)
                if v is not None:
                    if as_bool:
                        gt_vals.setdefault(gt, []).append(1.0 if v else 0.0)
                    else:
                        gt_vals.setdefault(gt, []).append(float(v))
            for gt, vals in gt_vals.items():
                if gt not in result:
                    result[gt] = {}
                if m not in result[gt]:
                    result[gt][m] = {"means": [0.0] * len(nc_levels),
                                     "stds": [0.0] * len(nc_levels)}
                result[gt][m]["means"][nc_idx] = np.mean(vals)
                result[gt][m]["stds"][nc_idx] = np.std(vals) / max(1, np.sqrt(len(vals)))

    return result


def plot_dr_success_rate(all_nc_data, nc_levels, methods, output_dir):
    """
    Dynamic Replanning Fig 1: Success rate bar chart + delta bar chart + error bars + mean line.

    X-axis: newcase difficulty level (0, 1, 2, ...)
    Y-axis: success rate
    Delta: increment relative to the previous level (not for the first level)
    """
    print("[DR Fig 1] Success Rate (Bar + Delta)")
    means, stds, raw = _compute_nc_overall_metric(
        all_nc_data, nc_levels, methods, "success", as_bool=True)

    n_groups = len(nc_levels)
    n_methods = len(methods)
    fig, ax = plt.subplots(figsize=DR_FIG_SIZE["bar"])

    group_width = 0.8
    bar_width = group_width / n_methods

    for mi, method in enumerate(methods):
        color = _method_color(method, methods)
        edge_color = _darker(color)
        base_means = means[method]
        base_raw = raw[method]

        # 计算增量 (相对于上一等级)
        deltas = [0.0]  # gi=0 无增量
        for gi in range(1, n_groups):
            deltas.append(base_means[gi] - base_means[gi - 1])

        xs_for_line = []
        ys_for_line = []

        for gi in range(n_groups):
            pos = gi - group_width / 2 + bar_width * (mi + 0.5)
            mean_v = base_means[gi]
            vals = base_raw[gi]
            n = len(vals)

            # Wilson CI for error bars
            successes = int(round(sum(vals))) if vals else 0
            ci_lo, ci_hi = _wilson_ci(successes, n)
            err_lo = mean_v - ci_lo
            err_hi = ci_hi - mean_v

            # 全量成功率柱状图
            ax.bar(pos, mean_v, width=bar_width * 0.75,
                   color=color, edgecolor=edge_color, alpha=0.75,
                   linewidth=0.8, zorder=2)
            # 误差棒
            ax.errorbar(pos, mean_v, yerr=[[err_lo], [err_hi]], fmt="none",
                        ecolor=edge_color, capsize=2.5, capthick=0.8,
                        elinewidth=0.8, zorder=3)

            # 标注成功率
            label_y = ci_hi + 0.02
            ax.text(pos, label_y, f"{mean_v:.2f}",
                    ha="center", va="bottom",
                    fontsize=FONT_CONFIG["annotation_size_small"],
                    color=_darker(color, 0.5))

            # 增量柱状图 (从 gi>=1 开始, 相对于上一等级)
            if gi > 0:
                delta = deltas[gi]
                ax.bar(pos, delta, width=bar_width * 0.75, bottom=0,
                       color=color, edgecolor=edge_color, alpha=0.5,
                       linewidth=0.8, hatch=DELTA_HATCH, zorder=2)
                # 标注增量
                if delta >= 0:
                    ax.text(pos, delta + 0.01, f"{delta:+.2f}",
                            ha="center", va="bottom",
                            fontsize=FONT_CONFIG["annotation_size_small"] - 1,
                            color=_darker(color, 0.4), fontstyle="italic")
                else:
                    ax.text(pos, delta - 0.01, f"{delta:+.2f}",
                            ha="center", va="top",
                            fontsize=FONT_CONFIG["annotation_size_small"] - 1,
                            color=_darker(color, 0.4), fontstyle="italic")

            xs_for_line.append(pos)
            ys_for_line.append(mean_v)

        # 均值连线
        ax.plot(xs_for_line, ys_for_line, "-", color=edge_color,
                linewidth=1.8, alpha=0.7, zorder=5, marker="o", markersize=3)

    # 零线
    ax.axhline(y=0, color="#666666", linewidth=0.6, zorder=1)

    ax.set_xticks(range(n_groups))
    ax.set_xticklabels([f"L{nc}" for nc in nc_levels],
                       fontsize=FONT_CONFIG["tick_label_size"])
    ax.set_xlim(-0.6, n_groups - 0.4)
    # y_lo: 考虑最大负增量
    all_deltas = [means[m][gi] - means[m][gi - 1]
                  for m in methods for gi in range(1, n_groups)]
    y_lo = min(0, min(all_deltas) if all_deltas else 0) - 0.1
    ax.set_ylim(y_lo, 1.25)

    _apply_font(ax, title="Success Rate across New Case Difficulty Levels",
                ylabel="Success Rate")
    _apply_grid(ax)

    # 图例
    handles = []
    for m in methods:
        c = _method_color(m, methods)
        handles.append(mpatches.Patch(facecolor=c, alpha=0.75,
                                      edgecolor=_darker(c),
                                      label=METHOD_DISPLAY_NAMES.get(m, m)))
    handles.append(mpatches.Patch(facecolor="#CCCCCC", edgecolor="#999999",
                                  hatch=DELTA_HATCH, alpha=0.5,
                                  label="Delta (vs prev level)"))
    ax.legend(handles=handles, loc="upper right",
              fontsize=FONT_CONFIG["legend_size"], framealpha=0.8)

    fig.tight_layout()
    _save_fig(fig, output_dir, "dr_fig1_success_rate")


def _plot_dr_violin_delta(all_nc_data, nc_levels, methods, output_dir,
                          metric_key, from_nested, title, ylabel, filename):
    """
    Generic function: Draw violin plot with box (full data) + delta bar chart + mean line.

    Violin plots are shifted upward to avoid overlapping with delta bar charts.
    Delta is the increment relative to the previous level.
    Used for Fig 2 (completion time), Fig 3 (LLM call count), Fig 4 (average input tokens).
    """
    means, stds, raw = _compute_nc_overall_metric(
        all_nc_data, nc_levels, methods, metric_key, from_nested=from_nested)

    vc = VIOLIN_CONFIG
    n_groups = len(nc_levels)
    n_methods = len(methods)

    # 计算所有正增量的最大值，用于确定上移偏移量（负增量在下方，不影响小提琴重叠）
    all_deltas = []
    for method in methods:
        for gi in range(1, n_groups):
            all_deltas.append(means[method][gi] - means[method][gi - 1])
    max_pos_delta = max((d for d in all_deltas if d > 0), default=0)
    # 小提琴图的 y 偏移量: 正增量区域高度 + 间距
    violin_y_offset = max_pos_delta * 1.3 + max_pos_delta * 0.2

    fig, ax = plt.subplots(figsize=DR_FIG_SIZE["violin"])

    group_width = 0.8
    item_width = group_width / n_methods

    for mi, method in enumerate(methods):
        color = _method_color(method, methods)
        edge_color = _darker(color)
        base_means = means[method]
        base_raw = raw[method]

        # 计算增量 (相对于上一等级)
        deltas = [0.0]
        for gi in range(1, n_groups):
            deltas.append(base_means[gi] - base_means[gi - 1])

        xs_for_line = []
        ys_for_line = []

        for gi in range(n_groups):
            pos = gi - group_width / 2 + item_width * (mi + 0.5)
            half_w = item_width * 0.42
            vals = base_raw[gi]
            mean_v = base_means[gi]

            # 全量: 带箱体小提琴图 (上移 violin_y_offset)
            shifted_vals = np.array(vals, dtype=float) + violin_y_offset if vals else np.array([])
            if vals and len(vals) >= 2:
                _draw_origin_violin(ax, pos, shifted_vals, half_w,
                                    color, edge_color, vc, annotate_median=False)
                # 手动标注均值 (上移后的位置)
                shifted_mean = mean_v + violin_y_offset
                med_v = np.median(vals)
                shifted_med = med_v + violin_y_offset
                whisker_hi = np.percentile(vals, 75)
                shifted_whisker_hi = whisker_hi + violin_y_offset
                ax.text(pos, shifted_whisker_hi + (max(vals) - min(vals)) * 0.05 + violin_y_offset * 0.02,
                        f"{med_v:.1f}", ha="center", va="bottom",
                        fontsize=FONT_CONFIG["annotation_size_small"],
                        color=_darker(edge_color, 0.6), fontweight="light")
            elif vals:
                shifted_v = vals[0] + violin_y_offset
                ax.scatter([pos], [shifted_v], marker="D", s=30, color=edge_color,
                           edgecolor="white", linewidth=0.5, zorder=6)
                ax.text(pos, shifted_v + 0.5, f"{vals[0]:.1f}",
                        ha="center", va="bottom",
                        fontsize=FONT_CONFIG["annotation_size_small"],
                        color=_darker(edge_color, 0.6))

            # 增量柱状图 (从 gi>=1 开始, 相对于上一等级, 在 y=0 附近)
            if gi > 0:
                delta = deltas[gi]
                ax.bar(pos, delta, width=item_width * 0.65, bottom=0,
                       color=color, edgecolor=edge_color, alpha=0.5,
                       linewidth=0.8, hatch=DELTA_HATCH, zorder=2)
                # 标注增量
                if delta >= 0:
                    ax.text(pos, delta + max_pos_delta * 0.05, f"{delta:+.1f}",
                            ha="center", va="bottom",
                            fontsize=FONT_CONFIG["annotation_size_small"] - 1,
                            color=_darker(color, 0.4), fontstyle="italic")
                else:
                    ax.text(pos, delta - max_pos_delta * 0.05, f"{delta:+.1f}",
                            ha="center", va="top",
                            fontsize=FONT_CONFIG["annotation_size_small"] - 1,
                            color=_darker(color, 0.4), fontstyle="italic")

            xs_for_line.append(pos)
            ys_for_line.append(mean_v + violin_y_offset)

        # 均值连线 (上移后)
        ax.plot(xs_for_line, ys_for_line, "-", color=edge_color,
                linewidth=1.8, alpha=0.7, zorder=7, marker="o", markersize=3)

    # 零线 (增量区域的基准)
    ax.axhline(y=0, color="#666666", linewidth=0.6, zorder=1)
    # 分隔线 (增量区域与小提琴区域之间)
    ax.axhline(y=violin_y_offset * 0.5, color="#AAAAAA", linewidth=0.4,
               linestyle=":", zorder=1)

    ax.set_xticks(range(n_groups))
    ax.set_xticklabels([f"L{nc}" for nc in nc_levels],
                       fontsize=FONT_CONFIG["tick_label_size"])
    ax.set_xlim(-0.6, n_groups - 0.4)

    # y轴: 下方显示增量原始值, 上方显示小提琴原始值 (减去偏移)
    def _dual_formatter(v, _):
        if v < violin_y_offset * 0.3:
            # 增量区域: 直接显示
            return f"{v:.0f}"
        else:
            # 小提琴区域: 还原偏移
            return f"{v - violin_y_offset:.0f}"
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_dual_formatter))

    _apply_font(ax, title=title,
                ylabel=f"↓ Delta (vs prev)          ↑ {ylabel}")
    _apply_grid(ax)

    # 图例
    handles = []
    for m in methods:
        c = _method_color(m, methods)
        handles.append(mpatches.Patch(facecolor=c, alpha=0.75,
                                      edgecolor=_darker(c),
                                      label=METHOD_DISPLAY_NAMES.get(m, m)))
    handles.append(mpatches.Patch(facecolor="#CCCCCC", edgecolor="#999999",
                                  hatch=DELTA_HATCH, alpha=0.5,
                                  label="Delta (vs prev level)"))
    ax.legend(handles=handles, loc="upper left",
              fontsize=FONT_CONFIG["legend_size"], framealpha=0.8)

    fig.tight_layout()
    _save_fig(fig, output_dir, filename)


def plot_dr_completion_time(all_nc_data, nc_levels, methods, output_dir):
    """Dynamic Replanning Fig 2: Completion Time (violin plot + delta bar chart + mean line)."""
    print("[DR Fig 2] Completion Time (Violin + Delta)")
    _plot_dr_violin_delta(
        all_nc_data, nc_levels, methods, output_dir,
        metric_key="elapsed_sec", from_nested=None,
        title="Completion Time across New Case Difficulty Levels",
        ylabel="Completion Time (s)",
        filename="dr_fig2_completion_time")


def plot_dr_llm_calls(all_nc_data, nc_levels, methods, output_dir):
    """Dynamic Replanning Fig 3: LLM Call Count (violin plot + delta bar chart + mean line)."""
    print("[DR Fig 3] LLM Calls (Violin + Delta)")
    _plot_dr_violin_delta(
        all_nc_data, nc_levels, methods, output_dir,
        metric_key="llm_calls", from_nested=None,
        title="LLM Calls across New Case Difficulty Levels",
        ylabel="LLM Calls",
        filename="dr_fig3_llm_calls")


def plot_dr_input_tokens(all_nc_data, nc_levels, methods, output_dir):
    """Dynamic Replanning Fig 4: Average Input Tokens (violin plot + delta bar chart + mean line)."""
    print("[DR Fig 4] Avg Input Tokens (Violin + Delta)")
    _plot_dr_violin_delta(
        all_nc_data, nc_levels, methods, output_dir,
        metric_key="prompt_tokens_mean", from_nested=None,
        title="Avg Input Tokens across New Case Difficulty Levels",
        ylabel="Avg Input Tokens",
        filename="dr_fig4_input_tokens")


def _annotate_with_leader_lines(ax, points, fontsize, y_range=(0, 1.15)):
    """
    Annotate data points on a subplot with leader lines, automatically avoiding overlap.

    points: [(x, y, text, color), ...]
    Strategy:
      1. Group by x, sort points at the same x position by y
      2. If y-value spacing at the same x position < min_gap, fan out annotation positions upward/downward
      3. Use ax.annotate + arrowprops to draw leader lines
    """
    if not points:
        return

    from collections import defaultdict

    # 按 x 分组
    groups = defaultdict(list)
    for x, y, text, color in points:
        groups[x].append((y, text, color))

    y_lo, y_hi = y_range
    total_range = y_hi - y_lo
    min_gap = total_range * 0.06  # 标注之间最小间距

    for x, items in groups.items():
        items.sort(key=lambda t: t[0])  # 按 y 排序
        n = len(items)

        if n == 1:
            y, text, color = items[0]
            # 单个点: 直接标注在右上方
            offset_x = 8
            offset_y = 10
            ax.annotate(text, xy=(x, y), xytext=(offset_x, offset_y),
                        textcoords="offset points",
                        fontsize=fontsize, color=color, fontweight="light",
                        arrowprops=dict(arrowstyle="-", color=color,
                                        lw=0.6, alpha=0.6),
                        zorder=8)
        else:
            # 多个点: 计算展开后的标注 y 位置
            ys_data = [item[0] for item in items]
            center = np.mean(ys_data)

            # 计算需要的总高度
            needed_height = (n - 1) * min_gap
            # 标注位置从 center 向两侧展开
            label_ys = []
            start_y = center - needed_height / 2
            for i in range(n):
                label_ys.append(start_y + i * min_gap)

            # 确保标注不超出 y 范围
            if label_ys[0] < y_lo + min_gap * 0.5:
                shift = (y_lo + min_gap * 0.5) - label_ys[0]
                label_ys = [ly + shift for ly in label_ys]
            if label_ys[-1] > y_hi - min_gap * 0.5:
                shift = label_ys[-1] - (y_hi - min_gap * 0.5)
                label_ys = [ly - shift for ly in label_ys]

            # 交替左右偏移避免水平重叠
            for i, (item, label_y) in enumerate(zip(items, label_ys)):
                y_data, text, color = item
                # 奇偶交替: 偶数右偏, 奇数左偏
                if i % 2 == 0:
                    offset_x = 10
                    ha = "left"
                else:
                    offset_x = -10
                    ha = "right"
                ax.annotate(text, xy=(x, y_data),
                            xytext=(offset_x, (label_y - y_data) * 80),
                            textcoords="offset points",
                            fontsize=fontsize, color=color, fontweight="light",
                            ha=ha, va="center",
                            arrowprops=dict(arrowstyle="-", color=color,
                                            lw=0.5, alpha=0.5,
                                            connectionstyle="arc3,rad=0.15"),
                            zorder=8)


def plot_dr_success_by_goal_type(all_nc_data, nc_levels, methods, output_dir):
    """
    Dynamic Replanning Fig 5: Success rate by goal type (2x5 subplot line chart).

    Uses GridSpec for precise control of subplot width, height, and spacing.
    Layout parameters are centralized in the GOAL_GRID_LAYOUT dictionary for easy customization.
    """
    print("[DR Fig 5] Success Rate by Goal Type (2x5 Grid)")
    gt_data = _compute_nc_per_goal_type_metric(
        all_nc_data, nc_levels, methods, "success", as_bool=True)

    # 按 GOAL_TYPE_ORDER 排序, 取前10个
    goal_types = [gt for gt in GOAL_TYPE_ORDER if gt in gt_data]
    for gt in sorted(gt_data.keys()):
        if gt not in goal_types:
            goal_types.append(gt)
    goal_types = goal_types[:10]

    n_types = len(goal_types)
    gl = GOAL_GRID_LAYOUT
    rows, cols = gl["rows"], gl["cols"]

    fig = plt.figure(figsize=(gl["fig_width"], gl["fig_height"]))
    gs = fig.add_gridspec(
        rows, cols,
        wspace=gl["wspace"],
        hspace=gl["hspace"],
        left=gl["left"],
        right=gl["right"],
        top=gl["top"],
        bottom=gl["bottom"],
    )

    axes = [[fig.add_subplot(gs[r, c]) for c in range(cols)] for r in range(rows)]

    for ti, gt in enumerate(goal_types):
        r, c = divmod(ti, cols)
        ax = axes[r][c]
        gt_info = gt_data.get(gt, {})

        annotation_points = []

        for mi, method in enumerate(methods):
            color = _method_color(method, methods)
            edge_color = _darker(color)
            m_info = gt_info.get(method, {"means": [0.0] * len(nc_levels),
                                          "stds": [0.0] * len(nc_levels)})
            ys = m_info["means"]
            errs = m_info["stds"]

            ax.errorbar(range(len(nc_levels)), ys, yerr=errs,
                        fmt="-o", color=edge_color, markersize=4,
                        markerfacecolor=color, markeredgecolor=edge_color,
                        capsize=2, capthick=0.8, elinewidth=0.8,
                        linewidth=1.8, alpha=0.85, zorder=3,
                        label=METHOD_DISPLAY_NAMES.get(method, method) if ti == 0 else None)

            for xi, y_val in enumerate(ys):
                annotation_points.append(
                    (xi, y_val, f"{y_val:.2f}", edge_color))

        _annotate_with_leader_lines(
            ax, annotation_points,
            fontsize=FONT_CONFIG["annotation_size_small"],
            y_range=(-0.05, 1.15))

        ax.set_title(GOAL_TYPE_DISPLAY.get(gt, gt),
                     fontsize=FONT_CONFIG["tick_label_size"], pad=4)
        ax.set_xticks(range(len(nc_levels)))
        ax.set_ylim(-0.05, 1.15)
        ax.set_xlim(-0.3, len(nc_levels) - 0.7)
        _apply_grid(ax)

        # x轴刻度标签: 只有下面一行显示
        if r == rows - 1:
            ax.set_xticklabels([f"L{nc}" for nc in nc_levels],
                               fontsize=FONT_CONFIG["tick_label_size"] - 1)
        else:
            ax.tick_params(axis="x", labelbottom=False)

        # y轴刻度标签: 只有最左边一列显示; 刻度朝内
        ax.tick_params(axis="y", direction=YTICK_INWARD["direction"],
                       pad=YTICK_INWARD["pad"])
        for lbl in ax.yaxis.get_ticklabels():
            lbl.set_ha("left")
        if c == 0:
            ax.set_ylabel("Success Rate", fontsize=FONT_CONFIG["axis_label_size"] - 1)
            ax.tick_params(axis="y", labelsize=FONT_CONFIG["tick_label_size"] - 1)
        else:
            ax.tick_params(axis="y", labelleft=False)

    # 关闭多余子图
    for ti in range(n_types, rows * cols):
        r, c = divmod(ti, cols)
        axes[r][c].axis("off")

    # 共用 x 轴标签
    fig.text(0.5, gl["bottom"] * 0.3, "Contingency Difficulty Level",
             ha="center", fontsize=FONT_CONFIG["axis_label_size"])

    # 图例 (放在 top 边距区域)
    handles = [Line2D([0], [0], color=_darker(_method_color(m, methods)),
                      marker="o", markersize=4,
                      markerfacecolor=_method_color(m, methods),
                      linewidth=1.8,
                      label=METHOD_DISPLAY_NAMES.get(m, m))
               for m in methods]
    fig.legend(handles=handles, loc="upper center",
               bbox_to_anchor=(0.5, gl["top"] + (1.0 - gl["top"]) * 0.85),
               ncol=len(methods), fontsize=FONT_CONFIG["legend_size"],
               framealpha=0.8)

    _save_fig(fig, output_dir, "dr_fig5_success_by_goal_type")


# ═══════════════════════════════════════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Detailed Method Comparison Plotting")
    parser.add_argument("--session-dir", type=str, default="results/batch_runs/2026-03-14_23-00-36", help="Batch experiment session directory")
    parser.add_argument("--nc", type=int, default=0, help="newcase level (default 0, only for general_planning mode)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: <session_dir>/figures/general_planning)")
    parser.add_argument("--mode", type=str, default="general_planning",
                        choices=["general_planning", "dynamic_replanning"],
                        help="Plotting mode: general_planning (single NC level) or dynamic_replanning (across NC levels)")
    args = parser.parse_args()

    session_dir = Path(args.session_dir)
    if not session_dir.exists():
        print(f"[ERROR] Session directory does not exist: {session_dir}")
        sys.exit(1)

    if args.mode == "general_planning":
        output_dir = Path(args.output_dir) if args.output_dir else \
            session_dir / "figures" / "general_planning"

        print(f"Session: {session_dir}")
        print(f"Mode: general_planning")
        print(f"NC Level: {args.nc}")
        print(f"Output: {output_dir}")
        print()

        # 加载数据
        print("Loading raw data...")
        all_data = load_all_raw_data(session_dir, args.nc)
        print("Loading aggregate data...")
        agg_data = load_aggregate_data(session_dir, args.nc)

        methods = _get_methods_sorted(all_data)
        print(f"Methods found: {methods}")
        print(f"Runs per method: {', '.join(f'{m}={len(all_data[m])}' for m in methods)}")
        print()

        # 绘图
        plot_success_rate_bar(all_data, methods, output_dir)
        plot_time_metrics(all_data, methods, agg_data, output_dir)
        plot_count_metrics(all_data, methods, output_dir)
        plot_energy_heatmap(all_data, methods, output_dir)
        plot_radar(agg_data, methods, output_dir)

    elif args.mode == "dynamic_replanning":
        output_dir = Path(args.output_dir) if args.output_dir else \
            session_dir / "figures" / "dynamic_replanning"

        print(f"Session: {session_dir}")
        print(f"Mode: dynamic_replanning")

        # 发现所有 NC 等级
        nc_levels = discover_nc_levels(session_dir)
        if not nc_levels:
            print("[ERROR] No newcase_N directories found")
            sys.exit(1)

        print(f"NC Levels: {nc_levels}")
        print(f"Output: {output_dir}")
        print()

        # 加载所有 NC 等级数据
        print("Loading all NC level raw data...")
        all_nc_data = load_all_nc_raw_data(session_dir, nc_levels)

        # 获取方法列表 (从任意一个 NC 等级)
        methods = set()
        for nc in nc_levels:
            methods.update(all_nc_data.get(nc, {}).keys())
        methods = [m for m in METHOD_ORDER if m in methods] + \
                  sorted(methods - set(METHOD_ORDER))
        print(f"Methods found: {methods}")
        for nc in nc_levels:
            nc_data = all_nc_data.get(nc, {})
            counts = ', '.join(f'{m}={len(nc_data.get(m, []))}' for m in methods)
            print(f"  NC={nc}: {counts}")
        print()

        # 绘图
        plot_dr_success_rate(all_nc_data, nc_levels, methods, output_dir)
        plot_dr_completion_time(all_nc_data, nc_levels, methods, output_dir)
        plot_dr_llm_calls(all_nc_data, nc_levels, methods, output_dir)
        plot_dr_input_tokens(all_nc_data, nc_levels, methods, output_dir)
        plot_dr_success_by_goal_type(all_nc_data, nc_levels, methods, output_dir)

    print()
    print("All figures generated.")


if __name__ == "__main__":
    main()
