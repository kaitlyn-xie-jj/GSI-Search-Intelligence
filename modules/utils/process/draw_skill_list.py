import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import re

# Provided JSON data.
data = {
  "robot_view": {
    "UAV-1": {
      "T1": [
        "take_off",
        "search<[[500.0, 745.0], [700.0, 895.0]]>_for<equipment_failures>"
      ]
    },
    "UAV-2": {
      "T3": [
        "take_off",
        "search<[[155.0, 165.0], [245.0, 235.0]]>_for<equipment_failures>"
      ],
      "T4": [
        "take_off",
        "search<[[175.0, 485.0], [325.0, 635.0]]>_for<equipment_failures>"
      ]
    },
    "UAV-3": {
      "T2": [
        "take_off",
        "search<[[510.0, 280.0], [690.0, 460.0]]>_for<equipment_failures>"
      ]
    }
  },
  "task_view": {
    "T1": [
      "UAV-1"
    ],
    "T3": [
      "UAV-2"
    ],
    "T4": [
      "UAV-2"
    ],
    "T2": [
      "UAV-3"
    ]
  }
}


def plot_robot_schedule_v4(data):
    """
    Draw a task Gantt chart with colors distinguishing tasks and 50% opacity.
    """
    fig, ax = plt.subplots(figsize=(18, 6))

    # Automatically create a color mapping for each task ID.
    all_tasks = sorted(list(set(task for robot_tasks in data["robot_view"].values() for task in robot_tasks.keys())))
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    task_color_map = {task_id: colors[i % len(colors)] for i, task_id in enumerate(all_tasks)}

    robot_labels = sorted(data["robot_view"].keys())
    max_time = 0

    # Iterate through each robot and its tasks.
    for y_pos, robot_name in enumerate(robot_labels):
        current_time_step = 0

        # sorted ensures tasks are ordered by name, such as T1, T2, ...
        for task_id, skills in sorted(data["robot_view"][robot_name].items()):
            task_color = task_color_map.get(task_id, "#B2B2B2")

            for skill_str in skills:
                match = re.match(r'([^<]+)(<.*>)?', skill_str)
                skill_name = match.group(1)
                skill_target = match.group(2) if match.group(2) else ''
                label = f"{skill_name}\n{skill_target}"

                # --- Main change 1: add alpha=0.5 when drawing bars. ---
                ax.barh(y_pos, width=1, left=current_time_step, height=0.7, color=task_color,
                        edgecolor='black', linewidth=0.5, align='center', alpha=0.5)

                # Add a light outline effect to make text clearer.
                text_element = ax.text(current_time_step + 0.5, y_pos, label,
                                       ha='center', va='center', color='black', fontsize=8,
                                       weight='bold', linespacing=0.9)

                current_time_step += 1

        if current_time_step > max_time:
            max_time = current_time_step

    # --- Chart styling ---
    ax.set_yticks(range(len(robot_labels)))
    ax.set_yticklabels(robot_labels, fontsize=12)
    ax.invert_yaxis()

    ax.set_xlim(0, max_time)
    ax.set_xticks(range(int(max_time) + 1))
    ax.set_xlabel("Time Step", fontsize=12)

    ax.set_title("Robot Task Schedule", fontsize=16, weight='bold')
    ax.grid(axis='x', linestyle='--', alpha=0.6)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # --- Main change 2: also add alpha=0.5 when creating the legend. ---
    legend_patches = [mpatches.Patch(color=color, label=task_id, alpha=0.2)
                      for task_id, color in sorted(task_color_map.items())]
    ax.legend(handles=legend_patches, bbox_to_anchor=(1.02, 1), loc='upper left', title='Tasks')

    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.show()


# Run the new visualization function.
plot_robot_schedule_v4(data)
