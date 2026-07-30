import os
import json
from collections import defaultdict
from matplotlib.colors import ListedColormap

import numpy as np
from matplotlib import colormaps
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ==================== 字体设置 ====================

# 1. 用你刚刚从 fc-list 查到的路径替换这里
font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"

# 2. 注册这个字体
font_manager.fontManager.addfont(font_path)
my_font = font_manager.FontProperties(fname=font_path)

# 3. 全局设置：以后不写 fontproperties 也尽量用这个中文字体
plt.rcParams['font.family'] = my_font.get_name()
plt.rcParams['axes.unicode_minus'] = False

# ==================== 配色配置 ====================

# 使用 tab20 colormap（20种颜色，足够多，所有图公用）
colormap = plt.get_cmap("tab20")


def build_label_color_map(labels, colors):
    """
    Build a global color mapping for all labels:
    Same label always gets the same color.
    """
    label_list = sorted(labels)  # 排序保证稳定
    mapping = {}
    n_colors = len(colors)
    for idx, lab in enumerate(label_list):
        mapping[lab] = colors[idx % n_colors]
    return mapping

# ==================== 路径配置 ====================

# batch_runs 根目录
BATCH_ROOT = "./results/batch_runs"

# 这里填写你要对比的几个批量实验的文件夹名
batch_dir_names = [
    "new_0",
    "new_1",
    "new_2",
    "new_3",
    # "newcase_4",
]

# =================================================


def parse_newcase_max_from_name(name: str) -> int:
    """
    Parse the maximum newcase count from a directory name.
    Returns 2 for e.g. "new_2"
    """
    try:
        parts = name.split("_")
        return int(parts[1])
    except Exception as e:
        raise ValueError(f"Cannot parse newcase max count from directory name {name}") from e


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_success_rate(sr_str: str) -> float:
    """
    Input a string like '0.85±0.357071', extract only the mean part 0.85
    """
    if "±" in sr_str:
        return float(sr_str.split("±")[0])
    else:
        return float(sr_str)


def aggregate_new_case_counts(batch_path: str):
    """
    Traverse all subdirectories under a batch directory and aggregate:
      - new_case_cnt_by_type
      - new_case_cnt_by_skill
    Some metrics.json may lack these fields, which are skipped.
    Returns:
      (total_by_type: dict[str, int], total_by_skill: dict[str, int])
    """
    total_by_type = defaultdict(int)
    total_by_skill = defaultdict(int)

    for name in os.listdir(batch_path):
        subdir = os.path.join(batch_path, name)
        if not os.path.isdir(subdir):
            continue

        metrics_path = os.path.join(subdir, "metrics.json")
        if not os.path.exists(metrics_path):
            continue

        try:
            metrics = load_json(metrics_path)
        except Exception as e:
            print(f"Failed to read {metrics_path}: {e}")
            continue

        # 统计 new_case_cnt_by_type
        for t, cnt in metrics.get("new_case_cnt_by_type", {}).items():
            total_by_type[t] += cnt

        # 统计 new_case_cnt_by_skill
        for skill, cnt in metrics.get("new_case_cnt_by_skill", {}).items():
            total_by_skill[skill] += cnt

    return dict(total_by_type), dict(total_by_skill)


def create_pie_chart(ax, data, title, label_color_map):
    """
    Create a pie chart, ensuring the same type/skill uses the same color across all pie charts.

    ax: subplot axis
    data: {label: value}
    title: pie chart title
    label_color_map: global label -> color mapping
    """
    labels = list(data.keys())
    sizes = list(data.values())

    # 使用全局颜色映射来确保相同标签的颜色一致
    colors = [label_color_map[label] for label in labels]

    ax.pie(
        sizes,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90,
        colors=colors,
        textprops={"fontsize": 10, "fontproperties": my_font},
    )
    ax.set_title(title, fontproperties=my_font)
    ax.axis("equal")  # 保证饼图是圆形


# ==================== 读取各批次信息 ====================

batches = []
all_newcase_types = set()
all_skills = set()

for dname in batch_dir_names:
    batch_path = os.path.join(BATCH_ROOT, dname)
    aggregate_path = os.path.join(batch_path, "aggregate_full.json")

    if not os.path.exists(aggregate_path):
        raise FileNotFoundError(f"{aggregate_path} does not exist, please check the path")

    aggregate = load_json(aggregate_path)

    newcase_max = parse_newcase_max_from_name(dname)
    overall_sr = parse_success_rate(aggregate["overall"]["success_rate"])

    # 汇总该批次下所有 metrics.json 中的新情况统计
    total_by_type, total_by_skill = aggregate_new_case_counts(batch_path)

    # 累积所有出现过的新情况类型和技能，用于构建“全局颜色映射”
    all_newcase_types.update(total_by_type.keys())
    all_skills.update(total_by_skill.keys())

    batches.append(
        {
            "name": dname,
            "path": batch_path,
            "newcase_max": newcase_max,
            "aggregate": aggregate,
            "overall_success_rate": overall_sr,
            "total_new_case_by_type": total_by_type,
            "total_new_case_by_skill": total_by_skill,
        }
    )

# 按 newcase_max 排序，确保横坐标顺序是 0,1,2,3,4 ...
batches.sort(key=lambda x: x["newcase_max"])

# ======== 构建：新情况类型 / 技能 的全局颜色映射（关键） ========

newcase_type_color_map = build_label_color_map(all_newcase_types, colormap.colors)
skill_color_map = build_label_color_map(all_skills, colormap.colors)

# ==================== 图 1：总体成功率 vs 新情况最大次数 ====================

x_vals = [b["newcase_max"] for b in batches]
y_vals = [b["overall_success_rate"] for b in batches]

plt.figure(figsize=(6, 4))

# 使用浅色线 + 浅色点
line_color = colormap(0)  # 使用 colormap 中的第一个颜色
marker_face = colormap(1)  # 使用 colormap 中的第二个颜色

plt.plot(
    x_vals,
    y_vals,
    marker="o",
    color=line_color,
    markerfacecolor=marker_face,
    markeredgecolor=line_color,
    linewidth=2,
)

plt.xlabel("Max New Case Count", fontproperties=my_font)
plt.ylabel("Overall Success Rate (overall.success_rate)", fontproperties=my_font)
plt.title("Overall Success Rate vs Max New Case Count", fontproperties=my_font)
plt.grid(True, linestyle="--", alpha=0.4)
plt.xticks(x_vals)

plt.tight_layout()
plt.savefig("./results/fig1_overall_success_vs_newcase_max.png", dpi=200)
plt.show()


# ==================== 图 2：按任务类型的成功率对比 ====================

# 收集所有 batch 中出现过的任务类型
# 获取所有任务类型
all_goal_types = set()
for b in batches:
    by_goal = b["aggregate"].get("by_goal_type", {})
    all_goal_types.update(by_goal.keys())

goal_types = sorted(all_goal_types)
x = np.arange(len(goal_types))

fig2, ax2 = plt.subplots(figsize=(10, 6))

num_batches = len(batches)
if num_batches == 0:
    raise RuntimeError("No batches found, please check if batch_dir_names is configured correctly")

# 每个任务类型一簇柱状图，每个簇里有 num_batches 条柱子
total_width = 0.8
bar_width = total_width / num_batches

# 颜色映射
bar_colors = [colormap(i / num_batches) for i in range(num_batches)]  # 为每个批次生成不同的颜色

for idx, b in enumerate(batches):
    heights = []
    by_goal = b["aggregate"].get("by_goal_type", {})
    for gt in goal_types:
        if gt in by_goal:
            sr_str = by_goal[gt]["success_rate"]
            heights.append(parse_success_rate(sr_str))
        else:
            heights.append(np.nan)  # 该批次中没有这个任务类型

    offset = (idx - (num_batches - 1) / 2) * bar_width

    # 选取对应的 bar color
    bar_color = bar_colors[idx]  # 使用colormap生成的颜色

    ax2.bar(
        x + offset,
        heights,
        width=bar_width,
        label=f"newcase_max={b['newcase_max']}",
        color=bar_color,
        edgecolor="white",
        alpha=0.9,
    )

ax2.set_xlabel("Task Type (goal_type)", fontproperties=my_font)
ax2.set_ylabel("Success Rate (success_rate)", fontproperties=my_font)
ax2.set_title("Success Rate Comparison by Task Type", fontproperties=my_font)
ax2.set_xticks(x)
ax2.set_xticklabels(goal_types, rotation=45, ha="right", fontproperties=my_font)
ax2.legend(prop=my_font)
ax2.grid(True, axis="y", linestyle="--", alpha=0.3)

plt.tight_layout()
plt.savefig("./results/fig2_success_by_goal_type.png", dpi=200)
plt.show()


# ==================== 图 3：每个批次下的新情况类型分布 (饼图) ====================

num_batches = len(batches)
# 简单安排子图网格：最多两列
cols = 2 if num_batches > 1 else 1
rows = int(np.ceil(num_batches / cols))

fig3, axes3 = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
if num_batches == 1:
    axes3 = np.array([[axes3]])  # 统一用二维数组处理
elif rows == 1:
    axes3 = np.array([axes3])

axes3_flat = axes3.flatten()

for i, b in enumerate(batches):
    ax = axes3_flat[i]
    counts = b["total_new_case_by_type"]

    # 去掉计数为 0 的项
    counts = {k: v for k, v in counts.items() if v > 0}

    if not counts:
        ax.text(0.5, 0.5, "No new case type data", ha="center", va="center", fontproperties=my_font)
        ax.axis("off")
        continue

    # 画饼图
    create_pie_chart(
        ax,
        counts,
        f"newcase_max={b['newcase_max']} New Case Type Distribution",
        newcase_type_color_map,
    )

# 多余的子图关掉
for j in range(i + 1, len(axes3_flat)):
    axes3_flat[j].axis("off")

plt.tight_layout()
plt.savefig("./results/fig3_new_case_type_pies.png", dpi=200)
plt.show()


# ==================== 图 4：每个批次下产生新情况的技能分布 (饼图) ====================

num_batches = len(batches)
cols = 2 if num_batches > 1 else 1
rows = int(np.ceil(num_batches / cols))

fig4, axes4 = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
if num_batches == 1:
    axes4 = np.array([[axes4]])
elif rows == 1:
    axes4 = np.array([axes4])

axes4_flat = axes4.flatten()

for i, b in enumerate(batches):
    ax = axes4_flat[i]
    counts = b["total_new_case_by_skill"]

    counts = {k: v for k, v in counts.items() if v > 0}

    if not counts:
        ax.text(0.5, 0.5, "No skill new case data", ha="center", va="center", fontproperties=my_font)
        ax.axis("off")
        continue

    # 画饼图
    create_pie_chart(
        ax,
        counts,
        f"newcase_max={b['newcase_max']} Skill New Case Distribution",
        skill_color_map,
    )

for j in range(i + 1, len(axes4_flat)):
    axes4_flat[j].axis("off")

plt.tight_layout()
plt.savefig("./results/fig4_new_case_skill_pies.png", dpi=200)
plt.show()
