# -*- coding: utf-8 -*-
"""
LipLLM Few-Shot Examples
"""
from typing import Dict, List


# ============================================================================
# Skill List Agent Few-Shot Examples
# ============================================================================
# LipLLM skill list generation is iterative: each LLM call outputs one typed skill until "done".
# Few-shot examples show multi-turn (user, assistant) dialog simulating the step-by-step process.

_SKILL_LIST_EXAMPLE_1 = [
    # Round 1: initial request
    {
        "role": "user",
        "content": (
            "Task: Search for a Suspicious_Person in cybertown and take a photo.\n\n"
            "Available robots: [UAV-1, Quadruped-1]\n\n"
            "Environment observation:\n"
            "Nodes: [Street Segment-10, Street Segment-15, campus-1, Hotel-1]\n"
            "Edges: [Street Segment-10 -> Street Segment-15, "
            "Street Segment-15 -> campus-1]\n"
            "Current skill list: []\n\n"
            "Generate the next typed skill (robot_type:skill_str) needed for this task, "
            "or output 'done' if the skill list is complete."
        ),
    },
    {"role": "assistant", "content": "UAV:take_off"},
    # Round 2
    {
        "role": "user",
        "content": (
            "Current skill list: [UAV:take_off]\n\n"
            "Generate the next typed skill (robot_type:skill_str) needed, or 'done'."
        ),
    },
    {"role": "assistant", "content": "UAV:search<cybertown>_for<Suspicious_Person>"},
    # Round 3
    {
        "role": "user",
        "content": (
            "Current skill list: [UAV:take_off, UAV:search<cybertown>_for<Suspicious_Person>]\n\n"
            "Generate the next typed skill (robot_type:skill_str) needed, or 'done'."
        ),
    },
    {"role": "assistant", "content": "Quadruped:navigate<Suspicious_Person>"},
    # Round 4
    {
        "role": "user",
        "content": (
            "Current skill list: [UAV:take_off, UAV:search<cybertown>_for<Suspicious_Person>, "
            "Quadruped:navigate<Suspicious_Person>]\n\n"
            "Generate the next typed skill (robot_type:skill_str) needed, or 'done'."
        ),
    },
    {"role": "assistant", "content": "Quadruped:take_photo<Suspicious_Person>"},
    # Round 5: done
    {
        "role": "user",
        "content": (
            "Current skill list: [UAV:take_off, UAV:search<cybertown>_for<Suspicious_Person>, "
            "Quadruped:navigate<Suspicious_Person>, Quadruped:take_photo<Suspicious_Person>]\n\n"
            "Generate the next typed skill (robot_type:skill_str) needed, or 'done'."
        ),
    },
    {"role": "assistant", "content": "done"},
]


# ============================================================================
# Dependency Graph Agent Few-Shot Examples
# ============================================================================
# The dependency graph agent receives the full typed skill list and outputs all dependency edges at once.

_DEPENDENCY_EXAMPLE_1_USER = (
    "Task: Search for a Suspicious_Person in cybertown and take a photo.\n\n"
    "Typed skill list:\n"
    "1. UAV:take_off\n"
    "2. UAV:search<cybertown>_for<Suspicious_Person>\n"
    "3. Quadruped:navigate<Suspicious_Person>\n"
    "4. Quadruped:take_photo<Suspicious_Person>\n\n"
    "Generate the dependency edges between these typed skills. "
    "Use the format 'robot_type:skill_A → robot_type:skill_B' to indicate that skill_A "
    "must be completed before skill_B. "
    "Use the EXACT typed skill strings from the list above. "
    "Output one edge per line. If there are no dependencies, output 'none'."
)

_DEPENDENCY_EXAMPLE_1_ASSISTANT = (
    "UAV:take_off → UAV:search<cybertown>_for<Suspicious_Person>\n"
    "UAV:search<cybertown>_for<Suspicious_Person> → Quadruped:navigate<Suspicious_Person>\n"
    "UAV:search<cybertown>_for<Suspicious_Person> → Quadruped:take_photo<Suspicious_Person>"
)


# ============================================================================
# Public API
# ============================================================================

SKILL_LIST_FEW_SHOT_EXAMPLES: List[List[Dict[str, str]]] = [
    _SKILL_LIST_EXAMPLE_1,
]
"""Few-shot examples for the skill list agent. Each element is a multi-turn dialog list."""

DEPENDENCY_FEW_SHOT_EXAMPLES: List[Dict[str, str]] = [
    {"role": "user", "content": _DEPENDENCY_EXAMPLE_1_USER},
    {"role": "assistant", "content": _DEPENDENCY_EXAMPLE_1_ASSISTANT},
]
"""Few-shot examples for the dependency graph agent. Standard (user, assistant) message pair list."""
