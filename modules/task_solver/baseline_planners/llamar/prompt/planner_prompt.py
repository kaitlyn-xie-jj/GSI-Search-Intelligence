# -*- coding: utf-8 -*-
"""
LLaMAR Planner Agent Prompt Template
"""
from typing import List, Dict, Optional
import json

from modules.task_solver.baseline_planners.common.prompt_components import (
    PARAMETERIZATION_RULES,
    build_notes_section,
)


# ============================================================================
# Few-shot Examples for Planner (standard role/content format)
# ============================================================================

PLANNER_FEW_SHOT_EXAMPLES = [
    {
        "role": "user",
        "content": (
            "Task: Search for a Blue_SUV in cybertown and take a photo of it\n"
            "Environment observation: UAV-1 and Quadruped-1 are at base. Cybertown area contains multiple street segments.\n"
            "Robots' open subtasks: None\n"
            "Robots' completed subtasks: None\n"
            "Robots' combined memory: None"
        ),
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "reason": "The subtask list is empty, so we need to create an initial plan. The task requires searching for Blue_SUV in cybertown and taking a photo. UAV should take off first, then search the area, and finally take a photo of the target.",
            "plan": [
                "UAV take off",
                "search cybertown for Blue_SUV",
                "navigate to Blue_SUV",
                "take photo of Blue_SUV",
            ]
        }, indent=2, ensure_ascii=False),
    },
    {
        "role": "user",
        "content": (
            "Task: Guide Suspicious Person to Library-3\n"
            "Environment observation: Quadruped-1 is at base. Suspicious Person is located at Street Segment-10. Library-3 is in the public area.\n"
            "Robots' open subtasks: ['navigate to Suspicious Person', 'guide Suspicious Person to Library-3']\n"
            "Robots' completed subtasks: None\n"
            "Robots' combined memory: Suspicious Person was found at Street Segment-10"
        ),
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "reason": "The current plan is still valid. Quadruped needs to first navigate to Suspicious Person's location, then guide them to the destination.",
            "plan": [
                "navigate to Suspicious Person",
                "guide Suspicious Person to Library-3"
            ]
        }, indent=2, ensure_ascii=False),
    },
]
"""Few-shot examples for Planner Agent. Standard (user, assistant) message pair list."""


# ============================================================================
# Prompt Building Functions
# ============================================================================

def build_planner_system_prompt(
    goal_type: Optional[str] = None,
) -> str:
    """Build system prompt for Planner Agent.

    Args:
        goal_type: Goal type, used to select goal-specific notes.

    Returns:
        Full system prompt string.
    """
    # Standard output format example
    output_format_section = """Your output should be in the form of a JSON dictionary as shown below.
Example output: {
"reason": "The subtask list is empty, so we need to create an initial plan. Based on the task instruction and current observations, the robots need to do ...",
"plan": ["search area_A for target_X", ...]
}"""
    
    # Build notes section (general + goal-specific)
    notes_section = build_notes_section(goal_type)
    
    return f"""You are an excellent planner who is tasked with helping a multi-robot system complete a task. The robots operate in a shared environment represented as a scene graph with nodes (entities) and edges (relationships).

The scene graph observation consists of:
- Nodes: entities in the environment (locations, objects, targets, robots, etc.) with their properties
- Edges: relationships between entities (e.g., located_in, connected_to)

{PARAMETERIZATION_RULES}

### INPUT FORMAT ###
{{Task: description of the task the robots are supposed to do,
Environment observation: a unified scene graph observation describing the current state of the world,
Robots' open subtasks: list of subtasks the robots are supposed to carry out to finish the task. If no plan has been already created, this will be None.
Robots' completed subtasks: list of subtasks the robots have already completed. If no subtasks have been completed, this will be None.
Robots' combined memory: description of robots' combined memory}}

Reason over the robots' task, observations, open subtasks, completed subtasks and memory, and then output the following:
* Reason: The reason for why new subtasks need to be added or the current plan needs to be updated.
* Subtasks: A list of open subtasks the robots are supposed to take to complete the task. As you get new information about the environment, you can modify this list. You can keep the same plan if you think it is still valid. Do not include the subtasks that have already been completed. The "Plan" should be in a list format where the actions are listed sequentially.

{output_format_section}

* NOTE: DO NOT OUTPUT ANYTHING EXTRA OTHER THAN WHAT HAS BEEN SPECIFIED
- Let's work this out in a step by step way to be sure we have the right answer.
- Ensure that the subtasks are not generic statements like "explore the environment" or "do the task". They should be specific to the task at hand. Do not assign subtasks to any particular robot.
- If a search is required and the task does not explicitly define a search area, there are several approaches you could consider. One option is to use "cybertown" as a general search area, though alternatively you might also consider searching individual known locations from the scene graph sequentially, or selecting a subset of areas that seem most relevant to the task context. The best approach may vary depending on the situation.

### Additional Notes ###
{notes_section}
"""


def build_planner_user_prompt(
    instruction: str,
    observation: str,
    subtasks_text: str,
    memory: str,
) -> str:
    """Build user prompt for Planner Agent (per-step dynamic content).

    Args:
        instruction: Task instruction text.
        observation: Observation string generated by format_observation().
        subtasks_text: Subtask text generated by format_subtasks().
        memory: Current accumulated memory string.

    Returns:
        User prompt string.
    """
    return (
        f"Task: {instruction}\n"
        f"{observation}\n"
        f"{subtasks_text}\n"
        f"Robots' combined memory: {memory if memory else 'None'}"
    )
