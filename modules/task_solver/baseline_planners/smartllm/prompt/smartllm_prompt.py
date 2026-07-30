# -*- coding: utf-8 -*-
"""
SmartLLM Prompt Templates
"""
from typing import Dict, List, Optional

from modules.task_solver.baseline_planners.common.prompt_components import (
    PARAMETERIZATION_RULES,
    build_notes_section,
    build_skill_library_section,
)


# ============================================================================
# SmartLLM Observation Formatting
# ============================================================================

def _display_name(node: Dict) -> str:
    """Pick the most informative display name for a node."""
    props = node.get("properties", {}) or {}
    if props.get("type") == "assembly_component":
        return props.get("subtype") or props.get("label") or str(node.get("id", "unknown"))
    return props.get("label", str(node.get("id", "unknown")))


def format_smartllm_observation(
    known_nodes: List[Dict],
    robot_labels: List[str],
) -> str:
    """Format WorldModelManager's known_nodes into a SmartLLM observation string.

    Args:
        known_nodes: World model node list, each containing 'id' and 'properties'.
        robot_labels: List of currently available robot labels.

    Returns:
        Formatted observation string with nodes only.
    """
    parts = ["Environment observation:"]

    node_strs = []
    for node in known_nodes:
        props = node.get("properties", {}) or {}
        name = _display_name(node)
        if props.get("category") == "robot":
            node_strs.append(f"{name}({props.get('status', 'active')})")
        else:
            node_strs.append(name)
    parts.append(f"Known Nodes: [{', '.join(node_strs)}]" if node_strs else "Nodes: []")

    return "\n".join(parts)


# ============================================================================
# Task Decomposition Agent Prompts
# ============================================================================

def build_decomposition_system_prompt(
    robot_labels: List[str],
    goal_type: Optional[str] = None,
    use_few_shot: bool = False,
) -> str:
    """Build system prompt for TaskDecompositionAgent.

    Includes: role description, GSI skill library, parameterization rules, output format, notes.

    Args:
        robot_labels: List of available robot labels.
        goal_type: Goal type, used to select goal-specific notes.
        use_few_shot: Whether to use few-shot (examples concatenated at agent level).

    Returns:
        System prompt string.
    """
    skill_library = build_skill_library_section(robot_labels)
    robot_list_str = ", ".join(robot_labels) if robot_labels else "available robots"
    notes_section = build_notes_section(goal_type)

    return f"""Agent Role: You are a multi-robot task decomposition expert. Your job is to decompose a given task into sub-tasks, each with a list of parameterized skills required to complete it.

Available robots: [{robot_list_str}]

### Skill Library ###
Each robot type has a specific set of skills. You MUST only reference skills from this library.

{skill_library}

{PARAMETERIZATION_RULES}

### Instructions ###
- You will receive a task instruction and environment observation.
- Decompose the task into logical sub-tasks.
- For each sub-task, list the parameterized skills required.
- Determine whether sub-tasks should be executed sequentially, in parallel, or mixed.
- If a search is required and the task does not explicitly define a search area, there are several approaches you could consider. One option is to use "cybertown" as a general search area, though alternatively you might also consider searching individual known locations from the scene graph sequentially, or selecting a subset of areas that seem most relevant to the task context. The best approach may vary depending on the situation.

### Output Format ###
Provide your decomposition as a valid JSON string:
```
{{{{
  "sub_tasks": [
    {{{{
      "name": "description of sub-task",
      "skills_required": ["skill_1<param>", "skill_2<param>_skill3<param>", ...]
    }}}},
    ...
  ],
  "execution_order": "sequential" | "parallel" | "mixed"
}}}}
```

### Additional Notes ###
{notes_section}

### Replanning ###
- If execution feedback is provided, you are replanning based on previous results.
- Replan or adjust the decomposition accordingly. Do NOT include sub-tasks whose skills have all been completed successfully.

NOTE: 
- Output ONLY the JSON format above. Do NOT include any extra text.
"""


def build_decomposition_user_prompt(
    instruction: str,
    known_nodes: list,
    known_edges: list,
    robot_labels: List[str],
    feedback_text: Optional[str] = None,
) -> str:
    """Build user prompt for TaskDecompositionAgent.

    Includes: task instruction, environment observation, feedback (during replanning).

    Args:
        instruction: User task instruction.
        known_nodes: World model node list.
        known_edges: World model edge list.
        robot_labels: List of available robot labels.
        feedback_text: Feedback text (provided by FeedbackProcessor during replanning).

    Returns:
        User prompt string.
    """
    robot_list_str = ", ".join(robot_labels) if robot_labels else "available robots"
    observation = format_smartllm_observation(known_nodes, robot_labels)

    parts = [f"Task: {instruction}"]
    parts.append(f"\nAvailable robots: [{robot_list_str}]")
    parts.append(f"\n{observation}")

    if feedback_text:
        parts.append(f"\nExecution feedback from previous attempt:\n{feedback_text}")

    parts.append(
        "\nDecompose this task into sub-tasks with parameterized skills. "
        "Output ONLY the JSON format specified."
    )

    return "\n".join(parts)


# ============================================================================
# Coalition Formation Agent Prompts
# ============================================================================

def build_coalition_system_prompt(
    robot_labels: List[str],
    use_few_shot: bool = False,
) -> str:
    """Build system prompt for CoalitionFormationAgent.

    Includes: role description, skill library, coalition formation rules, output format.

    Args:
        robot_labels: List of available robot labels.
        use_few_shot: Whether to use few-shot (examples concatenated at agent level).

    Returns:
        System prompt string.
    """
    skill_library = build_skill_library_section(robot_labels)
    robot_list_str = ", ".join(robot_labels) if robot_labels else "available robots"

    return f"""Agent Role: You are a multi-robot coalition formation expert. Your job is to assign robots to sub-tasks based on their capabilities and the skills required.

Available robots: [{robot_list_str}]

### Skill Library ###
{skill_library}

### Coalition Formation Rules ###
- Each sub-task should be assigned to a robot or team of robots that collectively possess all required skills.
- Use the minimum number of robots necessary for each sub-task.

### Output Format ###
Provide your reasoning and coalition assignments in the following format:
```
### Reasoning: 
(Your step-by-step analysis)

### Result:
The solution is:
For sub-task "<name>": ...
```

NOTE: Output your reasoning and result in the format above.
"""


def build_coalition_user_prompt(
    known_nodes: list,
    known_edges: list,
    robot_labels: List[str],
    decomposition_result: str,
) -> str:
    """Build user prompt for CoalitionFormationAgent.

    Includes: environment observation, decomposition result.

    Args:
        known_nodes: World model node list.
        known_edges: World model edge list.
        robot_labels: List of available robot labels.
        decomposition_result: Decomposition result from the previous stage (JSON string or text).

    Returns:
        User prompt string.
    """
    observation = format_smartllm_observation(known_nodes, robot_labels)

    return (
        f"{observation}\n\n"
        f"Task decomposition:\n{decomposition_result}\n\n"
        "Based on the decomposition above and the available robots, "
        "determine the robot coalition for each sub-task."
    )


# ============================================================================
# Task Allocation Agent Prompts
# ============================================================================

def build_allocation_system_prompt(
    robot_labels: List[str],
    goal_type: Optional[str] = None,
    use_few_shot: bool = False,
) -> str:
    """Build system prompt for TaskAllocationAgent.

    Includes: role description, skill library, parameterization rules, output format, notes.

    Args:
        robot_labels: List of available robot labels.
        goal_type: Goal type, used to select goal-specific notes.
        use_few_shot: Whether to use few-shot (examples concatenated at agent level).

    Returns:
        System prompt string.
    """
    skill_library = build_skill_library_section(robot_labels)
    robot_list_str = ", ".join(robot_labels) if robot_labels else "available robots"
    notes_section = build_notes_section(goal_type)

    return f"""Agent Role: You are a multi-robot task allocation expert. Your job is to produce a temporally-ordered execution plan based on the task decomposition and coalition formation results.

Available robots: [{robot_list_str}]

### Skill Library ###
Each robot type has a specific set of skills. You MUST only assign skills from this library.

{skill_library}

{PARAMETERIZATION_RULES}

### Instructions ###
- You will receive a task decomposition and a coalition formation solution.
- Produce a step-by-step execution plan as a list of timesteps.
- You only need to associate a specific robot label with the skills in the task decomposition. The output skill<param> must be exactly the skills (including parameters) already present in the task decomposition — they must not be altered in any way (including the parameters).

### Output Format ###
Provide your allocation as a valid JSON string:
```
{{{{
  "reasoning": "Your step-by-step reasoning for the allocation.",
  "plan": [
    ["<robot_label>:<skill_1<param>>"],
    ...
  ]
}}}}
```

### Additional Notes ###
{notes_section}

NOTE: Output ONLY the JSON format above. Do NOT include any extra text.
"""


def build_allocation_user_prompt(
    decomposition_result: str,
    coalition_result: str,
) -> str:
    """Build user prompt for TaskAllocationAgent.

    Includes: decomposition result, coalition result.

    Args:
        decomposition_result: Task decomposition result (JSON string or text).
        coalition_result: Coalition formation result (text).

    Returns:
        User prompt string.
    """
    return (
        f"Task decomposition:\n{decomposition_result}\n\n"
        f"Coalition formation solution:\n{coalition_result}\n\n"
        "Based on the decomposition and coalition above, "
        "produce the final per-robot skill list. "
        "Output ONLY the JSON format specified."
    )
