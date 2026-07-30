#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-method comparison visualization script

Usage:
    # Without new cases mode
    python tools/plot_method_comparison.py

    # New case mode: specify session directory containing newcase_0/, newcase_1/, ...
    python tools/plot_method_comparison.py --session-dir results/batch_runs/2026-03-10_12-12-46

Configuration:
    Without new cases mode: modify METHOD_DIRS dictionary below.
    New case mode: auto-discovered via --session-dir.
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ============================================================================
# 配色 & 全局样式
# ============================================================================

_METHOD_PALETTE = [
    "#7EB8DA", "#F4A582", "#B2DF8A", "#CAB2D6",
    "#FDBF6F", "#FB9A99", "#A6CEE3", "#D9D9D9",
]

plt.rcParams.update({
    "figure.facecolor": "#FAFAFA",
    "axes.facecolor": "#FAFAFA",
    "axes.edgecolor": "#CCCCCC",
    "axes.grid": True,
    "grid.color": "#E0E0E0",
    "grid.linestyle": "--",
    "grid.alpha": 0.6,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "axes.unicode_minus": False,
})

try:
    from matplotlib import font_manager
    _cjk_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    if Path(_cjk_path).exists():
        font_manager.fontManager.addfont(_cjk_path)
        _fp = font_manager.FontProperties(fname=_cjk_path)
        plt.rcParams["font.family"] = _fp.get_name()
except Exception:
    pass

# ============================================================================
# 无新情况模式的默认配置（手动填写路径）
# ============================================================================

METHOD_DIRS: Dict[str, str] = {
    "LipLLM":   "results/batch_runs/batch_2026-03-10_12-12-46_lipllm/aggregate_full.json",
    "SmartLLM": "results/batch_runs/batch_2026-03-10_12-12-46_smartllm/aggregate_full.json",
    "SPINE":    "results/batch_runs/batch_2026-03-10_12-12-46_spine/aggregate_full.json",
    "LLaMAR":   "results/batch_runs/batch_2026-03-10_12-12-46_llamar/aggregate_full.json",
    "SGI":      "results/batch_runs/batch_2026-03-10_12-12-46_sgi/aggregate_full.json",
}

# ============================================================================
# 数据工具
# ============================================================================

def parse_mean_std(s: str) -> Tuple[float, float]:
    if "±" in s:
        parts = s.split("±")
        return float(parts[0]), float(parts[1])
    return float(s), 0.0


def load_json(path) -> Optional[Dict]:
    p = Path(path)
    if not p.exists():
        print(f"[WARN] File not found, skipping: {p}")
        return None
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_metric(data: Dict, metric_key: str, bucket: str = "overall") -> Tuple[float, float]:
    section = data.get("overall", {}) if bucket == "overall" else data.get("by_goal_type", {}).get(bucket, {})
    val = section.get(metric_key)
    if val is None:
        return (np.nan, 0.0)
    if isinstance(val, str):
        return parse_mean_std(val)
    if isinstance(val, (int, float)):
        return (float(val), 0.0)
    return (np.nan, 0.0)


def _get_all_goal_types(all_data: Dict[str, Dict]) -> List[str]:
    types = set()
    for d in all_data.values():
        if d:
            types.update(d.get("by_goal_type", {}).keys())
    return sorted(types)


def _method_color(idx: int) -> str:
    return _METHOD_PALETTE[idx % len(_METHOD_PALETTE)]


def _darker(hex_color: str, factor: float = 0.7) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"#{int(r*factor):02x}{int(g*factor):02x}{int(b*factor):02x}"


# ============================================================================
# 新情况模式：自动发现数据
# ============================================================================

def discover_session_data(session_dir: Path) -> Tuple[List[int], List[str], Dict]:
    """
    Automatically discover data from session directory.

    Returns:
        nc_levels: Sorted list of new case counts [0, 1, 2, ...]
        methods: List of method names
        data: {nc_level: {method_name: aggregate_dict}}
    """
    nc_levels = []
    data = {}
    methods_set = set()

    for entry in sorted(session_dir.iterdir()):
        if not entry.is_dir() or not entry.name.startswith("newcase_"):
            continue
        try:
            nc = int(entry.name.split("_", 1)[1])
        except (ValueError, IndexError):
            continue
        nc_levels.append(nc)
        data[nc] = {}

        for batch_dir in sorted(entry.iterdir()):
            if not batch_dir.is_dir() or not batch_dir.name.startswith("batch_"):
                continue
            # 提取方法名：batch_<method> → method
            method_name = batch_dir.name.split("_", 1)[1]
            agg_path = batch_dir / "aggregate_full.json"
            agg = load_json(agg_path)
            if agg is not None:
                data[nc][method_name] = agg
                methods_set.add(method_name)

    nc_levels.sort()
    methods = sorted(methods_set)
    return nc_levels, methods, data


def aggregate_newcase_counts_from_batch(batch_dir: Path) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Traverse all subdirectories under batch directory's metrics.json, aggregate new case type and skill distributions."""
    total_by_type = defaultdict(int)
    total_by_skill = defaultdict(int)

    for name in os.listdir(batch_dir):
        subdir = batch_dir / name
        if not subdir.is_dir():
            continue
        metrics_path = subdir / "metrics.json"
        if not metrics_path.exists():
            continue
        try:
            metrics = load_json(metrics_path)
        except Exception:
            continue
        if not metrics:
            continue
        for t, cnt in (metrics.get("new_case_cnt_by_type") or {}).items():
            total_by_type[t] += cnt
        for s, cnt in (metrics.get("new_case_cnt_by_skill") or {}).items():
            total_by_skill[s] += cnt

    return dict(total_by_type), dict(total_by_skill)


# ============================================================================
# 绘图函数：（无新情况模式）
# ============================================================================

def plot_grouped_bar(all_data, metric_key, title, ylabel, filename, output_dir, fmt_func=None, ylim_bottom=None):
    methods = list(all_data.keys())
    goal_types = _get_all_goal_types(all_data)
    categories = ["Overall"] + goal_types
    n_methods, n_cats = len(methods), len(categories)
    x = np.arange(n_cats)
    bar_w = 0.75 / max(n_methods, 1)

    fig, ax = plt.subplots(figsize=(max(10, n_cats * 1.2), 5.5))
    for mi, method in enumerate(methods):
        d = all_data[method]
        means = [extract_metric(d, metric_key, "overall" if c == "Overall" else c)[0] if d else np.nan for c in categories]
        stds = [extract_metric(d, metric_key, "overall" if c == "Overall" else c)[1] if d else 0.0 for c in categories]
        offset = (mi - (n_methods - 1) / 2) * bar_w
        color = _method_color(mi)
        ax.bar(x + offset, means, bar_w, yerr=stds, capsize=3, label=method,
               color=color, edgecolor=_darker(color), linewidth=0.6,
               error_kw={"elinewidth": 0.8, "capthick": 0.8, "ecolor": "#888888"}, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=35, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=12)
    ax.legend(loc="upper right", framealpha=0.85)
    if fmt_func:
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_func))
    if ylim_bottom is not None:
        ax.set_ylim(bottom=ylim_bottom)
    ax.axvline(x=0.5, color="#BBBBBB", linewidth=0.8, linestyle=":", zorder=1)
    fig.tight_layout()
    fig.savefig(output_dir / filename)
    plt.close(fig)
    print(f"  ✓ {filename}")


def plot_success_rate(all_data, output_dir):
    methods = list(all_data.keys())
    goal_types = _get_all_goal_types(all_data)
    categories = ["Overall"] + goal_types
    n_methods, n_cats = len(methods), len(categories)
    x = np.arange(n_cats)
    bar_w = 0.75 / max(n_methods, 1)

    fig, ax = plt.subplots(figsize=(max(10, n_cats * 1.2), 5.5))
    for mi, method in enumerate(methods):
        d = all_data[method]
        means = [(extract_metric(d, "success_rate", "overall" if c == "Overall" else c)[0] * 100) if d else np.nan for c in categories]
        stds = [(extract_metric(d, "success_rate", "overall" if c == "Overall" else c)[1] * 100) if d else 0.0 for c in categories]
        offset = (mi - (n_methods - 1) / 2) * bar_w
        color = _method_color(mi)
        bars = ax.bar(x + offset, means, bar_w, yerr=stds, capsize=3, label=method,
                      color=color, edgecolor=_darker(color), linewidth=0.6,
                      error_kw={"elinewidth": 0.8, "capthick": 0.8, "ecolor": "#888888"}, zorder=3)
        for bar, m_val in zip(bars, means):
            if not np.isnan(m_val):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                        f"{m_val:.0f}%", ha="center", va="bottom", fontsize=7,
                        color=_darker(color), fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=35, ha="right")
    ax.set_ylabel("Success Rate (%)")
    ax.set_title("Success Rate by Method & Goal Type", pad=12)
    ax.set_ylim(0, 115)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
    ax.legend(loc="upper right", framealpha=0.85)
    ax.axvline(x=0.5, color="#BBBBBB", linewidth=0.8, linestyle=":", zorder=1)
    fig.tight_layout()
    fig.savefig(output_dir / "fig1_success_rate.png")
    plt.close(fig)
    print("  ✓ fig1_success_rate.png")


def plot_token_lengths(all_data, output_dir):
    methods = list(all_data.keys())
    goal_types = _get_all_goal_types(all_data)
    categories = ["Overall"] + goal_types
    n_methods, n_cats = len(methods), len(categories)
    x = np.arange(n_cats)
    bar_w = 0.75 / max(n_methods, 1)

    fig, ax1 = plt.subplots(figsize=(max(10, n_cats * 1.2), 5.5))
    ax2 = ax1.twinx()
    line_handles = []
    for mi, method in enumerate(methods):
        d = all_data[method]
        prompt_means = [extract_metric(d, "prompt_tokens_mean", "overall" if c == "Overall" else c)[0] if d else np.nan for c in categories]
        prompt_stds = [extract_metric(d, "prompt_tokens_mean", "overall" if c == "Overall" else c)[1] if d else 0.0 for c in categories]
        resp_means = [extract_metric(d, "response_tokens_mean", "overall" if c == "Overall" else c)[0] if d else np.nan for c in categories]
        offset = (mi - (n_methods - 1) / 2) * bar_w
        color = _method_color(mi)
        edge = _darker(color)
        ax1.bar(x + offset, prompt_means, bar_w, yerr=prompt_stds, capsize=2,
                color=color, edgecolor=edge, linewidth=0.6, alpha=0.85,
                error_kw={"elinewidth": 0.7, "capthick": 0.7, "ecolor": "#999999"},
                zorder=3, label=f"{method} (prompt)")
        line, = ax2.plot(x + offset, resp_means, marker="o", markersize=4, linewidth=1.5,
                         color=edge, linestyle="-", zorder=4)
        line_handles.append((line, f"{method} (response)"))

    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, rotation=35, ha="right")
    ax1.set_ylabel("Avg Prompt Tokens (bars)")
    ax2.set_ylabel("Avg Response Tokens (lines)")
    ax1.set_title("Prompt & Response Token Lengths", pad=12)
    h1, l1 = ax1.get_legend_handles_labels()
    h2 = [h for h, _ in line_handles]
    l2 = [l for _, l in line_handles]
    ax1.legend(h1 + h2, l1 + l2, loc="upper right", framealpha=0.85)
    ax1.axvline(x=0.5, color="#BBBBBB", linewidth=0.8, linestyle=":", zorder=1)
    fig.tight_layout()
    fig.savefig(output_dir / "fig5_token_lengths.png")
    plt.close(fig)
    print("  ✓ fig5_token_lengths.png")


def plot_radar(all_data, output_dir):
    methods = list(all_data.keys())
    if not methods:
        return
    radar_metrics = [
        ("success_rate", "Success Rate", True),
        ("llm_calls", "LLM Calls", False),
        ("planning_duration", "Planning Time", False),
        ("allocation_duration", "Allocation Time", False),
        ("total_energy", "Energy", False),
        ("replans_total", "Replans", False),
    ]
    n_metrics = len(radar_metrics)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.set_facecolor("#FAFAFA")
    raw_values = {m: [] for m in methods}
    for key, _, _ in radar_metrics:
        for m in methods:
            d = all_data[m]
            v, _ = extract_metric(d, key, "overall") if d else (np.nan, 0.0)
            raw_values[m].append(v)

    for i, (_, _, hb) in enumerate(radar_metrics):
        col = [raw_values[m][i] for m in methods if not np.isnan(raw_values[m][i])]
        if not col:
            continue
        vmin, vmax = min(col), max(col)
        rng = vmax - vmin if vmax != vmin else 1.0
        for m in methods:
            v = raw_values[m][i]
            if np.isnan(v):
                raw_values[m][i] = 0.0
            else:
                n = (v - vmin) / rng
                raw_values[m][i] = n if hb else (1.0 - n)

    for mi, m in enumerate(methods):
        vals = raw_values[m] + raw_values[m][:1]
        color = _method_color(mi)
        ax.plot(angles, vals, linewidth=2, color=_darker(color), label=m)
        ax.fill(angles, vals, alpha=0.15, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([l for _, l, _ in radar_metrics], fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_title("Overall Performance Radar", pad=20, fontsize=14)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), framealpha=0.85)
    fig.tight_layout()
    fig.savefig(output_dir / "fig_radar_overall.png")
    plt.close(fig)
    print("  ✓ fig_radar_overall.png")


def generate_baseline_figures(all_data, output_dir):
    """Generate 9 charts + radar chart (without new cases / newcase_0 baseline)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Generating baseline figures → {output_dir}")

    plot_success_rate(all_data, output_dir)
    for key, title, ylabel, fname in [
        ("elapsed_sec", "Completion Time", "Elapsed Time (s)", "fig2_elapsed_sec.png"),
        ("planning_duration", "Avg Planning Duration", "Planning Duration (s)", "fig3_planning_duration.png"),
        ("llm_calls", "LLM Call Count", "LLM Calls", "fig4_llm_calls.png"),
    ]:
        plot_grouped_bar(all_data, key, f"{title} by Method & Goal Type", ylabel, fname, output_dir, ylim_bottom=0)
    plot_token_lengths(all_data, output_dir)
    for key, title, ylabel, fname in [
        ("total_energy", "Total Energy Consumption", "Energy", "fig6_total_energy.png"),
        ("replans_total", "Replan Count", "Replans", "fig7_replans_total.png"),
        ("newcase_total", "Incident Count", "Incidents", "fig8_newcase_total.png"),
        ("allocation_duration", "Avg Allocation Duration", "Allocation Duration (s)", "fig9_allocation_duration.png"),
    ]:
        plot_grouped_bar(all_data, key, f"{title} by Method & Goal Type", ylabel, fname, output_dir, ylim_bottom=0)
    plot_radar(all_data, output_dir)


# ============================================================================
# 新情况模式绘图
# ============================================================================

# 9 个指标定义：(metric_key, title, ylabel)
_NC_METRICS = [
    ("success_rate",       "Success Rate",           "Success Rate"),
    ("elapsed_sec",        "Completion Time",        "Elapsed Time (s)"),
    ("planning_duration",  "Avg Planning Duration",  "Planning Duration (s)"),
    ("allocation_duration","Avg Allocation Duration", "Allocation Duration (s)"),
    ("llm_calls",          "LLM Call Count",         "LLM Calls"),
    ("prompt_tokens_mean", "Avg Prompt Tokens",      "Prompt Tokens"),
    ("response_tokens_mean","Avg Response Tokens",   "Response Tokens"),
    ("total_energy",       "Total Energy",           "Energy"),
    ("replans_total",      "Replan Count",           "Replans"),
]


def _plot_nc_metric_line(nc_levels, methods, session_data, metric_key, title, ylabel, filename, output_dir):
    """
    Fig NC-*: X-axis=new case count, Y-axis=metric value, each line=one method (Overall).
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    for mi, method in enumerate(methods):
        means, stds = [], []
        for nc in nc_levels:
            d = session_data.get(nc, {}).get(method)
            if d:
                m, s = extract_metric(d, metric_key, "overall")
                if metric_key == "success_rate":
                    m, s = m * 100, s * 100
            else:
                m, s = np.nan, 0.0
            means.append(m)
            stds.append(s)

        color = _method_color(mi)
        edge = _darker(color)
        ax.errorbar(nc_levels, means, yerr=stds, marker="o", markersize=5,
                     linewidth=2, capsize=3, color=edge, markerfacecolor=color,
                     markeredgecolor=edge, label=method)

    ax.set_xlabel("Max New Cases per Run")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title} vs New Case Count (Overall)", pad=12)
    ax.set_xticks(nc_levels)
    ax.legend(loc="best", framealpha=0.85)
    if metric_key == "success_rate":
        ax.set_ylim(bottom=0, top=115)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
    else:
        ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(output_dir / filename)
    plt.close(fig)
    print(f"  ✓ {filename}")


def _plot_nc_success_by_goal_type(nc_levels, methods, session_data, goal_types, output_dir):
    """
    Fig NC-9: One subplot per task type, X-axis=new case count, each line=one method.
    """
    n_types = len(goal_types)
    cols = min(5, n_types)
    rows = int(np.ceil(n_types / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4 * rows), squeeze=False)

    for ti, gt in enumerate(goal_types):
        r, c = divmod(ti, cols)
        ax = axes[r][c]
        for mi, method in enumerate(methods):
            vals = []
            for nc in nc_levels:
                d = session_data.get(nc, {}).get(method)
                if d:
                    m, _ = extract_metric(d, "success_rate", gt)
                    vals.append(m * 100)
                else:
                    vals.append(np.nan)
            color = _method_color(mi)
            ax.plot(nc_levels, vals, marker="o", markersize=4, linewidth=1.5,
                    color=_darker(color), markerfacecolor=color, label=method)

        ax.set_title(gt, fontsize=10)
        ax.set_xlabel("Max NC")
        ax.set_ylabel("SR (%)")
        ax.set_xticks(nc_levels)
        ax.set_ylim(0, 115)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
        if ti == 0:
            ax.legend(fontsize=7, loc="best")

    # 关闭多余子图
    for ti in range(n_types, rows * cols):
        r, c = divmod(ti, cols)
        axes[r][c].axis("off")

    fig.suptitle("Success Rate by Goal Type vs New Case Count", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(output_dir / "fig_nc9_success_by_goal_type.png")
    plt.close(fig)
    print("  ✓ fig_nc9_success_by_goal_type.png")


def _plot_nc_distribution_pies(nc, methods, session_dir, output_dir):
    """
    Fig NC-10+: For a given nc value, draw 2×N subplots (N=number of methods).
    Top row: new case type distribution pie charts; Bottom row: skill distribution pie charts.
    """
    n_methods = len(methods)
    if n_methods == 0:
        return

    colormap = plt.get_cmap("tab20")

    # 收集所有标签用于统一配色
    all_type_labels = set()
    all_skill_labels = set()
    method_data = {}

    nc_dir = session_dir / f"newcase_{nc}"
    for method in methods:
        batch_dir = nc_dir / f"batch_{method}"
        if batch_dir.exists():
            by_type, by_skill = aggregate_newcase_counts_from_batch(batch_dir)
        else:
            by_type, by_skill = {}, {}
        # 过滤掉 0 值
        by_type = {k: v for k, v in by_type.items() if v > 0}
        by_skill = {k: v for k, v in by_skill.items() if v > 0}
        method_data[method] = (by_type, by_skill)
        all_type_labels.update(by_type.keys())
        all_skill_labels.update(by_skill.keys())

    def _build_color_map(labels):
        sl = sorted(labels)
        return {lab: colormap(i % 20) for i, lab in enumerate(sl)}

    type_cmap = _build_color_map(all_type_labels)
    skill_cmap = _build_color_map(all_skill_labels)

    fig, axes = plt.subplots(2, n_methods, figsize=(5 * n_methods, 9), squeeze=False)

    for mi, method in enumerate(methods):
        by_type, by_skill = method_data[method]

        # 上排：类型分布
        ax_t = axes[0][mi]
        if by_type:
            labels = list(by_type.keys())
            sizes = list(by_type.values())
            colors = [type_cmap[l] for l in labels]
            ax_t.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90,
                     colors=colors, textprops={"fontsize": 8})
        else:
            ax_t.text(0.5, 0.5, "No data", ha="center", va="center")
            ax_t.axis("off")
        ax_t.set_title(f"{method}\n(type dist)", fontsize=10)

        # 下排：技能分布
        ax_s = axes[1][mi]
        if by_skill:
            labels = list(by_skill.keys())
            sizes = list(by_skill.values())
            colors = [skill_cmap[l] for l in labels]
            ax_s.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90,
                     colors=colors, textprops={"fontsize": 8})
        else:
            ax_s.text(0.5, 0.5, "No data", ha="center", va="center")
            ax_s.axis("off")
        ax_s.set_title(f"{method}\n(skill dist)", fontsize=10)

    fig.suptitle(f"New Case Distribution (max_newcases={nc})", fontsize=14, y=1.02)
    fig.tight_layout()
    fname = f"fig_nc_dist_newcase_{nc}.png"
    fig.savefig(output_dir / fname)
    plt.close(fig)
    print(f"  ✓ {fname}")


def generate_newcase_figures(nc_levels, methods, session_data, session_dir, output_dir):
    """Generate all charts for the new case dimension."""
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nGenerating new-case figures → {output_dir}")

    # 收集所有 goal_types
    all_gt = set()
    for nc in nc_levels:
        for m in methods:
            d = session_data.get(nc, {}).get(m)
            if d:
                all_gt.update(d.get("by_goal_type", {}).keys())
    goal_types = sorted(all_gt)

    # Fig NC-1 ~ NC-8: 8 个指标的折线图
    for i, (key, title, ylabel) in enumerate(_NC_METRICS, 1):
        _plot_nc_metric_line(nc_levels, methods, session_data, key, title, ylabel,
                            f"fig_nc{i}_{key}.png", output_dir)

    # Fig NC-9: 按任务类型的成功率子图
    if goal_types:
        _plot_nc_success_by_goal_type(nc_levels, methods, session_data, goal_types, output_dir)

    # Fig NC-10, 11, 12, ...: 每个 nc>0 的分布饼图
    for nc in nc_levels:
        if nc == 0:
            continue
        _plot_nc_distribution_pies(nc, methods, session_dir, output_dir)


# ============================================================================
# main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Multi-method comparison visualization")
    parser.add_argument("--session-dir", type=str, default="results/batch_runs/2026-03-14_23-00-36",
                        help="New case mode: session directory containing newcase_0/, newcase_1/, ...")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (auto-selected by default)")
    args = parser.parse_args()

    if args.session_dir:
        # ---- 新情况模式 ----
        session_dir = Path(args.session_dir)
        if not session_dir.exists():
            print(f"[ERROR] Session directory does not exist: {session_dir}")
            sys.exit(1)

        nc_levels, methods, session_data = discover_session_data(session_dir)
        if not nc_levels or not methods:
            print("[ERROR] No valid data found, please check directory structure.")
            sys.exit(1)

        print(f"Found {len(nc_levels)} new case levels: {nc_levels}")
        print(f"Found {len(methods)} methods: {methods}")

        output_dir = Path(args.output_dir) if args.output_dir else session_dir / "figures"

        # 1) 基线图（newcase_0 的数据）
        baseline_data = session_data.get(0, {})
        if baseline_data:
            generate_baseline_figures(baseline_data, output_dir / "baseline")

        # 2) 新情况维度图
        generate_newcase_figures(nc_levels, methods, session_data, session_dir, output_dir / "newcase")

    else:
        # ---- 无新情况模式 ----
        output_dir = Path(args.output_dir) if args.output_dir else Path("results/figures")

        all_data: Dict[str, Dict] = {}
        for method, path in METHOD_DIRS.items():
            d = load_json(path)
            if d is not None:
                all_data[method] = d

        if not all_data:
            print("[ERROR] No method data loaded, please check METHOD_DIRS configuration.")
            sys.exit(1)

        print(f"Loaded {len(all_data)} methods: {', '.join(all_data.keys())}")
        generate_baseline_figures(all_data, output_dir)

    print(f"\nAll charts saved.")


if __name__ == "__main__":
    main()
