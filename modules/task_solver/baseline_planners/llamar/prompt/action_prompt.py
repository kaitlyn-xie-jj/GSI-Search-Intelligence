# -*- coding: utf-8 -*-
"""
LLaMAR Action Agent Prompt Template
"""
from typing import List, Dict, Optional
import json

from modules.task_solver.baseline_planners.common.prompt_components import (
    PARAMETERIZATION_RULES,
    build_notes_section,
    build_skill_library_section,
)


# ============================================================================
# Few-shot Examples for Action Agent
# ============================================================================

ACTION_FEW_SHOT_EXAMPLES = [
    {
        "role": "user",
        "content": (
            "Task: Search for Blue_SUV in cybertown and take a photo\n"
            "Environment observation: UAV-1 is at base. Cybertown contains Street Segment-10, Street Segment-15, etc.\n"
            "Robots's previous action: None\n"
            "Robots' open subtasks: ['UAV take off', 'search cybertown for Blue_SUV', 'navigate to Blue_SUV', 'take photo of Blue_SUV']\n"
            "Robots' completed subtasks: None\n"
            "Robots' subtask: UAV take off\n"
            "Robots' combined memory: None\n"
            "Previous feedback: None"
        ),
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "failure reason": "None",
            "memory": "UAV-1 is ready to start the search mission.",
            "reason": "UAV-1 needs to take off first before it can fly to search for the target.",
            "subtask": "UAV take off",
            "UAV-1's action": "take_off"
        }, indent=2, ensure_ascii=False),
    },
    {
        "role": "user",
        "content": (
            "Task: Guide Suspicious Person to Library-3\n"
            "Environment observation: Quadruped-1 is at Street Segment-10, Suspicious_Person detected at Street Segment-15.\n"
            "Robots's previous action: Quadruped-1: navigate<Suspicious_Person> (successful)\n"
            "Robots' open subtasks: ['guide Suspicious_Person to Library-3']\n"
            "Robots' completed subtasks: ['search cybertown for Suspicious_Person', 'navigate to Suspicious_Person']\n"
            "Robots' subtask: guide Suspicious_Person to Library-3\n"
            "Robots' combined memory: Suspicious_Person is at Street Segment-15.\n"
            "Previous feedback: Quadruped-1 successfully navigated to Suspicious_Person."
        ),
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "failure reason": "None",
            "memory": "Quadruped-1 is at Suspicious_Person's location. Quadruped-1 should guide Suspicious_Person.",
            "reason": "Quadruped-1 should guide Suspicious_Person to Library-3 since it's already at Suspicious_Person's location.",
            "subtask": "guide Suspicious_Person to Library-3",
            "Quadruped-1's action": "guide<Suspicious_Person>_to<Library-3>",
        }, indent=2, ensure_ascii=False),
    },
]
"""Few-shot examples for Action Agent. Standard (user, assistant) message pair list."""


def _build_zero_shot_section(robot_labels: List[str]) -> str:
    """Build zero-shot example section."""
    
    return f"""Your output should be in the form of a JSON dictionary as shown below.
Example output:
{{
"failure reason": "None",
"memory": "Robot observations and important state information for future reference.",
"reason": "Reasons for why the dispatched robots are taking the actions they are taking.",
"subtask": "Description of what the dispatched robots are currently working on.",
"UAV-1's action": "skill_str<skill_parameter>",
"Quadruped-1's action": "skill_str<skill_parameter>",
}}"""


# ============================================================================
# Constants
# ============================================================================

FAILURE_REASON_SECTION = """
If any robot's previous action failed, you need to think and rationalize about why the previous action failed. Output the reason for failure and how to fix this in the next timestep. If the previous action was successful, output "None".
"""


# ============================================================================
# Prompt Building Functions
# ============================================================================

def build_action_system_prompt(
    robot_labels: List[str],
    active_robot_labels: List[str] = None,
    goal_type: Optional[str] = None,
) -> str:
    """Build system prompt for Action Agent.

    Uses unified global observation, previous action describes only active robots,
    and includes Parameterization Rules & Ontology.
    Instructs LLM to output actions only for dispatched robot subset, not all robots.

    Args:
        robot_labels: All available robot labels in the current scene.
        active_robot_labels: Subset of robots dispatched in the previous step.
        goal_type: Goal type, used to select goal-specific notes.

    Returns:
        Full system prompt string.
    """
    # Build skill library description (using shared interface)
    skill_library = build_skill_library_section(robot_labels)

    # List available robots for LLM selection
    robot_list_str = ", ".join(robot_labels) if robot_labels else "available robots"

    # Build notes section (general + goal-specific)
    notes_section = build_notes_section(goal_type)

    # Standard output format example
    output_format_section = _build_zero_shot_section(robot_labels)

    return f"""You are an excellent planner and robot controller who is tasked with helping a multi-robot system complete a task. The robots operate in a shared environment represented as a scene graph with nodes and edges.

Available robots in the scene: [{robot_list_str}]

### Skill Library ###
Each robot type has a specific set of skills it can perform. You MUST only assign skills that belong to the robot's type.

{skill_library}

Your output MUST be grounded in this library. 

{PARAMETERIZATION_RULES}

You need to dispatch one or more suitable robots for the current subtask and output only their actions. You do NOT need to output actions for every robot. You will receive a textual description of the environment as a unified scene graph observation, including nodes (entities), edges (relationships), and robot states.

IMPORTANT: When outputting actions, use the EXACT format shown in the skill list above. For example:
- "search<area_name>_for<target_name>" (NOT "search area_name for target_name")

### INPUT FORMAT ###
{{Task: description of the task the robots are supposed to do,
Environment observation: a unified scene graph observation describing the current state of the world,
Robots's previous action: the action robot took in the previous step and whether it was successful
Robots' open subtasks: list of subtasks the robots are supposed to carry out to finish the task. If no plan has been already created, this will be None.
Robots' completed subtasks: list of subtasks the robots have already completed. If no subtasks have been completed, this will be None.
Robots' subtask: description of the subtasks the robots were trying to complete in the previous step,
Robots' combined memory: description of robots' combined memory,
Previous feedback: feedback from the previous step's execution results}}

You are supposed to reason over the robots' observations, previous actions, previous failures, previous memory, subtasks and the available actions the robots can perform, and think step by step and then output the following things:
* Failure reason: {FAILURE_REASON_SECTION}
* Memory: Whatever important information about the scene you think you should remember for the future as a memory. Remember that this memory will be used in future steps to carry out the task. You should not include information that is not relevant to the task. You can also include information that is already present in its memory if you think it might be useful in the future.
* Reason: The reasoning for what each dispatched robot is supposed to do next.
* Subtask: The subtask the dispatched robots should currently try to solve, choose this from the list of open subtasks.
* Actions for dispatched robots: The actions for only the robots you are dispatching. Only include robots that have meaningful work to do.

{output_format_section}

Note that the output should just be a dictionary. Only include "<robot_label>'s action" keys for the robots you are dispatching — do not include idle robots.

Important details:
* When finished with all subtasks, output "Done" for all robots.
* Use specific entity names when specifying action parameters.
* Only dispatch robots that have useful work to do for the current subtask.

### Additional Notes ###
{notes_section}

* NOTE: DO NOT OUTPUT ANYTHING EXTRA OTHER THAN WHAT HAS BEEN SPECIFIED
"""


def build_action_user_prompt(
    instruction: str,
    observation: str,
    subtasks_text: str,
    memory: str,
    current_subtask: str,
    feedback: str,
    previous_actions: str = "",
) -> str:
    """Build user prompt for Action Agent (per-step dynamic content).

    Args:
        instruction: Task instruction text.
        observation: Observation string generated by format_observation().
        subtasks_text: Subtask text generated by format_subtasks().
        memory: Current accumulated memory string.
        current_subtask: Subtask being executed in the previous step.
        feedback: Feedback text generated by format_feedback().
        previous_actions: Active robots' previous action description
            generated by format_previous_actions().

    Returns:
        User prompt string.
    """
    parts = [
        f"Task: {instruction}",
        observation,
    ]
    if previous_actions:
        parts.append(previous_actions)
    parts.extend([
        subtasks_text,
        f"Robots' subtask: {current_subtask if current_subtask else 'None'}",
        f"Robots' combined memory: {memory if memory else 'None'}",
        f"Previous feedback: {feedback}",
    ])
    return "\n".join(parts)
