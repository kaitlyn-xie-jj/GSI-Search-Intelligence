# -*- coding: utf-8 -*-
"""
Common Prompt Components for Baseline Methods
"""
import json
import re
from typing import Any, Dict, List, Optional, Tuple

from modules.task_solver.sgi_planner.prompt.master_context import (
    PARAMETERIZATION_RULES,
    ADDITIONAL_NOTES,
)
from modules.task_solver.sgi_planner.prompt.goal_type_notes import GOAL_TYPE_NOTES


def select_goal_notes(goal_type: Optional[str]) -> Tuple[str, str]:
    """Select corresponding notes based on goal_type."""
    if not goal_type:
        return ("default", GOAL_TYPE_NOTES.get("__default__", ""))

    key = str(goal_type).strip().lower()
    if key in GOAL_TYPE_NOTES:
        return (key, GOAL_TYPE_NOTES[key])

    if (
        "parking" in key
        or "violation" in key
        or "crowd" in key
        or key.startswith("event.")
    ):
        return ("event", GOAL_TYPE_NOTES.get("event", ""))
    
    return (
        "object",
        GOAL_TYPE_NOTES.get("object", GOAL_TYPE_NOTES.get("__default__", "")),
    )


# ============================================================================
# Utility Functions
# ============================================================================

def build_notes_section(goal_type: Optional[str] = None) -> str:
    """Build the complete notes section, including general notes and goal-specific notes."""
    _, goal_specific_notes = select_goal_notes(goal_type)
    
    notes_parts = [ADDITIONAL_NOTES]
    if goal_specific_notes:
        notes_parts.append(goal_specific_notes)
    
    return "\n".join(notes_parts)


# ============================================================================
# LLM Response JSON Extraction
# ============================================================================

def build_skill_library_section(robot_labels: List[str]) -> str:
    """Dynamically generate skill library description based on robot labels."""
    from modules.task_solver.sgi_planner.prompt.atomic_skills import robot_skill_source
    from modules.task_solver.sgi_planner.utils.process_skills import (
        skill_library_to_markdown,
    )

    seen_types = []
    for label in robot_labels:
        robot_type = label.rsplit("-", 1)[0] if "-" in label else label
        if robot_type not in seen_types and robot_type in robot_skill_source:
            seen_types.append(robot_type)

    filtered_library = {rt: robot_skill_source[rt] for rt in seen_types}
    return skill_library_to_markdown(filtered_library, include_details=False)


# ============================================================================
# LLM Response JSON Extraction
# ============================================================================

def extract_json(text: str) -> Dict[str, Any]:
    """Extract JSON dictionary from LLM response text."""
    # Try code block
    m = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try first { ... }
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    # Try raw text
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        raise ValueError(f"Failed to extract JSON from LLM response:\n{text[:500]}")
