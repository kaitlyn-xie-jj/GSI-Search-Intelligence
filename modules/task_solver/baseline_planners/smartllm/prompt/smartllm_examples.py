# -*- coding: utf-8 -*-
"""
SmartLLM Few-Shot Examples
"""
import json
from typing import Dict, List


# ============================================================================
# Task Decomposition Agent Few-Shot Examples
# ============================================================================

_DECOMPOSITION_EXAMPLE_1_USER = (
    "Task: Search for a Suspicious_Person in cybertown and take a photo.\n\n"
    "Available robots: [UAV-1, Quadruped-1]\n\n"
    "Environment observation:\n"
    "Nodes: [Street Segment-10, Street Segment-15, Street Segment-20, campus-1, Hotel-1]\n"
    "Edges: [Street Segment-10 -> Street Segment-15, "
    "Street Segment-15 -> Street Segment-20, "
    "Street Segment-20 -> campus-1]\n"
    "Decompose this task into sub-tasks with parameterized skills. "
    "Output ONLY the JSON format specified."
)

_DECOMPOSITION_EXAMPLE_1_ASSISTANT = json.dumps(
    {
        "sub_tasks": [
            {
                "name": "Search for Suspicious_Person from the air",
                "skills_required": [
                    "take_off",
                    "search<cybertown>_for<Suspicious_Person>",
                ],
            },
            {
                "name": "Navigate to and photograph the target",
                "skills_required": [
                    "navigate<Suspicious_Person>",
                    "take_photo<Suspicious_Person>",
                ],
            },
        ],
        "execution_order": "sequential",
    },
    indent=2,
    ensure_ascii=False,
)

_DECOMPOSITION_EXAMPLE_2_USER = (
    "Task: Transport wall_panel from Hotel-1 to Street Segment-10.\n\n"
    "Available robots: [UGV-1, Humanoid-1]\n\n"
    "Environment observation:\n"
    "Nodes: [Hotel-1, Street Segment-10, Street Segment-5, wall_panel]\n"
    "Edges: [wall_panel -> Hotel-1, "
    "Hotel-1 -> Street Segment-5, "
    "Street Segment-5 -> Street Segment-10]\n"
    "Decompose this task into sub-tasks with parameterized skills. "
    "Output ONLY the JSON format specified."
)

_DECOMPOSITION_EXAMPLE_2_ASSISTANT = json.dumps(
    {
        "sub_tasks": [
            {
                "name": "Load wall_panel onto transport vehicle",
                "skills_required": [
                    "navigate<wall_panel>",
                    "place<wall_panel>_on<UGV-1>",
                ],
            },
            {
                "name": "Transport to destination and unload",
                "skills_required": [
                    "navigate<Street Segment-10>",
                    "place<wall_panel>_on<ground>",
                ],
            },
        ],
        "execution_order": "sequential",
    },
    indent=2,
    ensure_ascii=False,
)


# ============================================================================
# Coalition Formation Agent Few-Shot Examples
# ============================================================================

_COALITION_EXAMPLE_1_USER = (
    "Environment observation:\n"
    "Nodes: [Street Segment-10, Street Segment-15, campus-1, Hotel-1]\n"
    "Edges: [Street Segment-10 -> Street Segment-15, "
    "Street Segment-15 -> campus-1]\n"
    "Task decomposition:\n"
    '{"sub_tasks": [{"name": "Search for Suspicious_Person from the air", '
    '"skills_required": ["take_off", "search<cybertown>_for<Suspicious_Person>"]}, '
    '{"name": "Navigate to and photograph the target", '
    '"skills_required": ["navigate<Suspicious_Person>", "take_photo<Suspicious_Person>"]}], '
    '"execution_order": "sequential"}\n\n'
    "Based on the decomposition above and the available robots, "
    "determine the robot coalition for each sub-task."
)

_COALITION_EXAMPLE_1_ASSISTANT = (
    "### Reasoning:\n"
    "Sub-task 1 requires take_off and search skills. UAV-1 has both take_off and search capabilities, "
    "making it ideal for aerial search. Quadruped-1 cannot take_off.\n"
    "Sub-task 2 requires navigate and take_photo. Both UAV-1 and Quadruped-1 have these skills. "
    "Quadruped-1 can approach on the ground for a closer photo.\n\n"
    "### Result:\n"
    "The solution is:\n"
    'For sub-task "Search for Suspicious_Person from the air": UAV-1 (has take_off, search skills)\n'
    'For sub-task "Navigate to and photograph the target": Quadruped-1 (has navigate, take_photo skills)'
)

_COALITION_EXAMPLE_2_USER = (
    "Environment observation:\n"
    "Nodes: [Hotel-1, Street Segment-10, Street Segment-5, wall_panel]\n"
    "Edges: [wall_panel -> Hotel-1, "
    "Hotel-1 -> Street Segment-5, "
    "Street Segment-5 -> Street Segment-10]\n"
    "Task decomposition:\n"
    '{"sub_tasks": [{"name": "Load wall_panel onto transport vehicle", '
    '"skills_required": ["navigate<wall_panel>", "place<wall_panel>_on<UGV-1>"]}, '
    '{"name": "Transport to destination and unload", '
    '"skills_required": ["navigate<Street Segment-10>", "place<wall_panel>_on<ground>"]}], '
    '"execution_order": "sequential"}\n\n'
    "Based on the decomposition above and the available robots, "
    "determine the robot coalition for each sub-task."
)

_COALITION_EXAMPLE_2_ASSISTANT = (
    "### Reasoning:\n"
    "Sub-task 1 requires navigate and place skills. Humanoid-1 has the place skill to load "
    "wall_panel onto UGV-1. Humanoid-1 needs to navigate to the wall_panel first.\n"
    "Sub-task 2 requires navigate and place skills. UGV-1 handles transport via navigate. "
    "Humanoid-1 performs the unloading with place.\n\n"
    "### Result:\n"
    "The solution is:\n"
    'For sub-task "Load wall_panel onto transport vehicle": Humanoid-1 (has navigate, place skills)\n'
    'For sub-task "Transport to destination and unload": UGV-1 (has navigate), '
    "Humanoid-1 (has place skill for unloading)"
)


# ============================================================================
# Task Allocation Agent Few-Shot Examples
# ============================================================================

_ALLOCATION_EXAMPLE_1_USER = (
    "Task decomposition:\n"
    '{"sub_tasks": [{"name": "Search for Suspicious_Person from the air", '
    '"skills_required": ["take_off", "search<cybertown>_for<Suspicious_Person>"]}, '
    '{"name": "Navigate to and photograph the target", '
    '"skills_required": ["navigate<Suspicious_Person>", "take_photo<Suspicious_Person>"]}], '
    '"execution_order": "sequential"}\n\n'
    "Coalition formation solution:\n"
    'For sub-task "Search for Suspicious_Person from the air": UAV-1 (has take_off, search skills)\n'
    'For sub-task "Navigate to and photograph the target": Quadruped-1 (has navigate, take_photo skills)\n\n'
    "Based on the decomposition and coalition above, "
    "produce the final per-robot skill list. "
    "Output ONLY the JSON format specified."
)

_ALLOCATION_EXAMPLE_1_ASSISTANT = json.dumps(
    {
        "reasoning": (
            "UAV-1 is assigned to the search sub-task: take_off then search. "
            "Quadruped-1 handles the photo sub-task: navigate to target then take_photo. "
            "UAV-1 take_off must be first, then UAV-1 search. After UAV-1 finds the target, Quadruped-1 can navigate and take_photo. "
            "This respects the sequential execution order and skill requirements."
        ),
        "plan": [
            ["UAV-1:take_off"],
            ["UAV-1:search<cybertown>_for<Suspicious_Person>"],
            ["Quadruped-1:navigate<Suspicious_Person>"],
            ["Quadruped-1:take_photo<Suspicious_Person>"]
        ],
    },
    indent=2,
    ensure_ascii=False,
)

_ALLOCATION_EXAMPLE_2_USER = (
    "Task decomposition:\n"
    '{"sub_tasks": [{"name": "Load wall_panel onto transport vehicle", '
    '"skills_required": ["navigate<wall_panel>", "place<wall_panel>_on<UGV-1>"]}, '
    '{"name": "Transport to destination and unload", '
    '"skills_required": ["navigate<Street Segment-10>", "place<wall_panel>_on<ground>"]}], '
    '"execution_order": "sequential"}\n\n'
    "Coalition formation solution:\n"
    'For sub-task "Load wall_panel onto transport vehicle": Humanoid-1 (has navigate, place skills)\n'
    'For sub-task "Transport to destination and unload": UGV-1 (has navigate), '
    "Humanoid-1 (has place skill for unloading)\n\n"
    "Based on the decomposition and coalition above, "
    "produce the final per-robot skill list. "
    "Output ONLY the JSON format specified."
)

_ALLOCATION_EXAMPLE_2_ASSISTANT = json.dumps(
    {
        "reasoning": (
            "Humanoid-1 and UGV-1 must coordinate for the loading sub-task. Humanoid-1 navigates to wall_panel and places it on UGV-1. "
            "Then UGV-1 and Humanoid-1 navigate to Street Segment-10 together."
            "Finally, Humanoid-1 unloads the wall_panel with place<wall_panel>_on<ground>. "
            "This respects the sequential execution order and skill requirements, with necessary coordination."
        ),
        "plan": [
            ["UGV-1:navigate<wall_panel>", "Humanoid-1:navigate<wall_panel>"],
            ["Humanoid-1:place<wall_panel>_on<UGV-1>"],
            ["UGV-1:navigate<Street Segment-10>", "Humanoid-1:navigate<Street Segment-10>"],
            ["Humanoid-1:place<wall_panel>_on<ground>"],
        ],
    },
    indent=2,
    ensure_ascii=False,
)


# ============================================================================
# Public API
# ============================================================================

DECOMPOSITION_FEW_SHOT_EXAMPLES: List[Dict[str, str]] = [
    {"role": "user", "content": _DECOMPOSITION_EXAMPLE_1_USER},
    {"role": "assistant", "content": _DECOMPOSITION_EXAMPLE_1_ASSISTANT},
    {"role": "user", "content": _DECOMPOSITION_EXAMPLE_2_USER},
    {"role": "assistant", "content": _DECOMPOSITION_EXAMPLE_2_ASSISTANT},
]
"""Few-shot examples for the task decomposition agent. Standard (user, assistant) message pair list."""

COALITION_FEW_SHOT_EXAMPLES: List[Dict[str, str]] = [
    {"role": "user", "content": _COALITION_EXAMPLE_1_USER},
    {"role": "assistant", "content": _COALITION_EXAMPLE_1_ASSISTANT},
    {"role": "user", "content": _COALITION_EXAMPLE_2_USER},
    {"role": "assistant", "content": _COALITION_EXAMPLE_2_ASSISTANT},
]
"""Few-shot examples for the coalition formation agent. Standard (user, assistant) message pair list."""

ALLOCATION_FEW_SHOT_EXAMPLES: List[Dict[str, str]] = [
    {"role": "user", "content": _ALLOCATION_EXAMPLE_1_USER},
    {"role": "assistant", "content": _ALLOCATION_EXAMPLE_1_ASSISTANT},
    {"role": "user", "content": _ALLOCATION_EXAMPLE_2_USER},
    {"role": "assistant", "content": _ALLOCATION_EXAMPLE_2_ASSISTANT},
]
"""Few-shot examples for the task allocation agent. Standard (user, assistant) message pair list."""
