# -*- coding: utf-8 -*-
"""
LipLLM Prompt Template

Builds prompts for LipLLM's two agents (SkillListAgent, DependencyGraphAgent).
"""
from typing import Dict, List, Optional

from modules.task_solver.baseline_planners.common.prompt_components import (
    PARAMETERIZATION_RULES,
    build_notes_section,
    build_skill_library_section,
)
from modules.task_solver.sgi_planner.prompt.observation import format_observation


# ============================================================================
# Shared: Robot Type Utilities
# ============================================================================

def build_robot_type_map(robot_labels: List[str]) -> Dict[str, str]:
    """Build label -> type mapping from robot label list.

    Example: ["UAV-1", "UGV-1", "Quadruped-1"] -> {"UAV-1": "UAV", "UGV-1": "UGV", "Quadruped-1": "Quadruped"}

    Args:
        robot_labels: List of robot labels.

    Returns:
        {label: robot_type} mapping dictionary.
    """
    return {
        label: (label.rsplit("-", 1)[0] if "-" in label else label)
        for label in robot_labels
    }


# ============================================================================
# Skill List Agent Prompts
# ============================================================================

def build_skill_list_system_prompt(
    robot_labels: List[str],
    goal_type: Optional[str] = None,
) -> str:
    """Build system prompt for SkillListAgent.

    Includes: role description, skill library, parameterization rules, output format, notes.

    Args:
        robot_labels: List of available robot labels.
        goal_type: Goal type, used to select goal-specific notes.

    Returns:
        System prompt string.
    """
    skill_library = build_skill_library_section(robot_labels)
    robot_list_str = ", ".join(robot_labels) if robot_labels else "available robots"
    notes_section = build_notes_section(goal_type)

    return f"""You are a multi-robot task planner. Your job is to generate a list of skills needed to complete a given task, one skill at a time. Each skill must specify which robot type should execute it.

Available robots: [{robot_list_str}]

### Skill Library ###
Each robot type has a specific set of skills. You MUST only use skills from this library, and assign each skill to a robot type that owns it.

{skill_library}

{PARAMETERIZATION_RULES}

### Output Format ###
Each output must follow the format: robot_type:skill_str
- robot_type is the type name (e.g., UAV), NOT the instance label.
- skill_str is the exact skill from the library.
- If you think the necessary skills have been generated, output "done" to indicate completion.

### Instructions ###
- You will be given a task instruction, environment observation, the current skill list and execution feedback (if replanning).
- At each step, output exactly ONE typed skill string that should be added next.
- Avoid output the same skill consecutively.
- Use the EXACT skill format from the library with the correct robot type prefix.
- Do NOT output explanations or reasoning — only the typed skill string or "done".
- Consider all available robot types when generating skills.
- A skill can ONLY be assigned to a robot type that has that skill in its library.
- If a search is required and the task does not explicitly define a search area, there are several approaches you could consider. One option is to use "cybertown" as a general search area, though alternatively you might also consider searching individual known locations from the scene graph sequentially, or selecting a subset of areas that seem most relevant to the task context. The best approach may vary depending on the situation.

### Additional Notes ###
{notes_section}

### Replanning ###
- If execution feedback is provided, you are replanning based on previous results.
- Replan or adjust the skill list accordingly. Do NOT include skills that have already been completed successfully.
"""


def build_skill_list_user_prompt(
    instruction: str,
    robot_labels: List[str],
    known_nodes: list,
    known_edges: list,
    current_skills: List[str],
    feedback_text: Optional[str] = None,
) -> str:
    """Build user prompt for SkillListAgent.

    Includes: task instruction, environment observation, current skill list, feedback (during replanning).

    Args:
        instruction: User task instruction.
        robot_labels: List of available robot labels.
        known_nodes: World model node list.
        known_edges: World model edge list.
        current_skills: Currently generated skill list.
        feedback_text: Feedback text (provided by FeedbackProcessor during replanning).

    Returns:
        User prompt string.
    """
    robot_list_str = ", ".join(robot_labels) if robot_labels else "available robots"
    skills_str = ", ".join(current_skills) if current_skills else ""

    parts = [f"Task: {instruction}"]
    parts.append(f"\nAvailable robots: [{robot_list_str}]")

    if feedback_text:
        parts.append(f"\n{feedback_text}")

    parts.append(f"\nCurrent skill list: [{skills_str}]")
    parts.append(
        "\nGenerate the next typed skill (robot_type:skill_str) needed for this task, or output 'done' if the skill list is complete."
    )

    return "\n".join(parts)


# ============================================================================
# Dependency Graph Agent Prompts
# ============================================================================

def build_dependency_graph_system_prompt() -> str:
    """Build system prompt for DependencyGraphAgent.

    Includes: role description, output format.
    Each skill in the list already has a robot_type prefix (e.g. "UAV:take_off").

    Returns:
        System prompt string.
    """
    return """You are a task dependency analyzer for a multi-robot system. Your job is to determine the execution order dependencies between a list of typed skills.

### Instructions ###
- You will receive a task instruction and a numbered list of typed skills (format: robot_type:skill_str).
- Determine which skills must be completed before others can begin.
- Output dependency edges in the format: robot_type:skill_A → robot_type:skill_B
  This means skill_A must be completed before skill_B can start.
- Use the EXACT typed skill strings from the input list (including the robot_type prefix).
- Output one edge per line.
- If there are no dependencies (all skills are independent), output "none".
- Do NOT output explanations or reasoning — only the dependency edges or "none".
- The resulting dependency graph MUST be acyclic (no circular dependencies).
"""


def build_dependency_graph_user_prompt(
    instruction: str,
    skill_list: List[str],
) -> str:
    """Build user prompt for DependencyGraphAgent.

    Includes: task instruction, numbered skill list, output format hint.

    Args:
        instruction: User task instruction.
        skill_list: Skill list generated by SkillListAgent.

    Returns:
        User prompt string.
    """
    numbered_skills = "\n".join(
        f"{i + 1}. {skill}" for i, skill in enumerate(skill_list)
    )

    return (
        f"Task: {instruction}\n\n"
        f"Typed skill list:\n{numbered_skills}\n\n"
        "Generate the dependency edges between these typed skills. "
        "Use the format 'robot_type:skill_A → robot_type:skill_B' to indicate that skill_A must be completed before skill_B. "
        "Use the EXACT typed skill strings from the list above. "
        "Output one edge per line. If there are no dependencies, output 'none'."
    )
