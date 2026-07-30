# -*- coding: utf-8 -*-
"""
SPINE Few-Shot Examples
"""
import json
from typing import Dict, List


# ============================================================================
# Example 1: Multi-robot search + photo task
# ============================================================================

_EXAMPLE_1_USER = (
    "task: Search for a Suspicious_Person in cybertown and take a photo.\n\n"
    "Scene graph:\n"
    "objects: []\n"
    "regions: [Street Segment-10, Street Segment-15, Street Segment-20, campus-1, Hotel-1, Robot Base-1]\n"
    "object_connections: []\n"
    "region_connections: [[Street Segment-10, Street Segment-15], "
    "[Street Segment-15, Street Segment-20], [Street Segment-20, campus-1]]\n"
    "robot_locations: {UAV-1: Robot Base-1, Quadruped-1: Robot Base-1}"
)

_EXAMPLE_1_ASSISTANT = json.dumps(
    {
        "primary_goal": "Search for Suspicious_Person in cybertown and take a photo once found.",
        "relevant_graph": "Street Segment-10, Street Segment-15, Street Segment-20, campus-1",
        "reasoning": (
            "The task requires searching for a Suspicious_Person and taking a photo. "
            "The target location is currently unknown, so the first step is to search. "
            "UAV-1 can cover large areas from the air efficiently, so it should take off "
            "and search the main cybertown area. Quadruped-1 can search ground-level areas "
            "in parallel."
        ),
        "plan": [
            "UAV-1:take_off",
            "Quadruped-1:search<campus-1>_for<Suspicious_Person>",
        ],
    },
    indent=2,
    ensure_ascii=False,
)


# ============================================================================
# Example 2: Replanning scenario -- adjust plan after execution feedback
# ============================================================================

_EXAMPLE_2_USER = (
    "updates: UAV-1 searched Street Segment-10 but did not find Suspicious_Person. "
    "Quadruped-1 detected Suspicious_Person at campus-1.\n\n"
    "Scene graph:\n"
    "objects: [Suspicious_Person]\n"
    "regions: [Street Segment-10, Street Segment-15, campus-1]\n"
    "object_connections: [[Suspicious_Person, campus-1]]\n"
    "region_connections: [[Street Segment-10, Street Segment-15], "
    "[Street Segment-15, campus-1]]\n"
    "robot_locations: {UAV-1: Street Segment-10, Quadruped-1: campus-1}"
)

_EXAMPLE_2_ASSISTANT = json.dumps(
    {
        "primary_goal": "Take a photo of Suspicious_Person who was found at campus-1.",
        "relevant_graph": "campus-1, Suspicious_Person",
        "reasoning": (
            "Quadruped-1 found Suspicious_Person at campus-1. "
            "Quadruped-1 is already at campus-1, so it can take the photo directly. "
            "UAV-1 can navigate to campus-1 for aerial photo as backup."
        ),
        "plan": [
            "Quadruped-1:navigate<Suspicious_Person>",
            "UAV-1:navigate<Suspicious_Person>",
        ],
    },
    indent=2,
    ensure_ascii=False,
)


# ============================================================================
# Example 3: Transport task -- UGV + Humanoid collaboration
# ============================================================================

_EXAMPLE_3_USER = (
    "task: Transport wall_panel from Hotel-1 to Street Segment-10.\n\n"
    "Scene graph:\n"
    "objects: [wall_panel]\n"
    "regions: [Hotel-1, Street Segment-10, Street Segment-5]\n"
    "object_connections: [[wall_panel, Hotel-1]]\n"
    "region_connections: [[Hotel-1, Street Segment-5], "
    "[Street Segment-5, Street Segment-10]]\n"
    "robot_locations: {UGV-1: Hotel-1, Humanoid-1: Hotel-1}"
)

_EXAMPLE_3_ASSISTANT = json.dumps(
    {
        "primary_goal": "Transport wall_panel from Hotel-1 to Street Segment-10.",
        "relevant_graph": "Hotel-1, Street Segment-10, Street Segment-5, wall_panel",
        "reasoning": (
            "The wall_panel is at Hotel-1. Humanoid-1 can place it onto UGV-1 for transport. "
            "UGV-1 then navigates to Street Segment-10. Humanoid-1 follows to unload. "
            "The sequence is clear, so I output multiple steps: first both navigate to the panel, "
            "then Humanoid-1 loads it onto UGV-1, then UGV-1 drives to the destination."
        ),
        "plan": [
            ["Humanoid-1:navigate<wall_panel>", "UGV-1:navigate<wall_panel>"],
            ["Humanoid-1:place<wall_panel>_on<UGV-1>"],
            ["UGV-1:navigate<Street Segment-10>"],
        ],
    },
    indent=2,
    ensure_ascii=False,
)


# ============================================================================
# Public API
# ============================================================================

SPINE_FEW_SHOT_EXAMPLES: List[Dict[str, str]] = [
    {"role": "user", "content": _EXAMPLE_1_USER},
    {"role": "assistant", "content": _EXAMPLE_1_ASSISTANT},
    {"role": "user", "content": _EXAMPLE_2_USER},
    {"role": "assistant", "content": _EXAMPLE_2_ASSISTANT},
    {"role": "user", "content": _EXAMPLE_3_USER},
    {"role": "assistant", "content": _EXAMPLE_3_ASSISTANT},
]
