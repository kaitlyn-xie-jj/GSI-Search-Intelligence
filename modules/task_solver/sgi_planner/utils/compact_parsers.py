# -*- coding: utf-8 -*-
"""
Compact format parsers.

All downstream modules use these two parse functions consistently.

Format rules:
- required_skills: "robot_type:skill_str:count"
  Example: "Humanoid:navigate<Hotel-1>:1"
  Multiple types: "UAV|UGV:navigate<loc>:2"
- edges (full mode): "T1->T2" or "T1->T2:condition_expression"
"""
import re
from typing import Any, Dict, List, Optional, Tuple, Union


def parse_compact_skill(skill: Union[str, Dict]) -> Dict[str, Any]:
    """Parse a compact skill string or existing dict into a unified skill dictionary.

    Format: robot_type:skill_str:count
    Or:     robot_type|robot_type2:skill_str:count

    Returns:
        {"skill_name": str, "assigned_robot_type": list, "assigned_robot_count": int|str}
    """
    if isinstance(skill, dict):
        return skill

    if not isinstance(skill, str):
        return {"skill_name": str(skill), "assigned_robot_type": [], "assigned_robot_count": 1}

    parts = skill.split(":")
    if len(parts) >= 3:
        robot_types_str = parts[0]
        # count may be in "tbd:xxx" form.
        if len(parts) >= 4 and parts[-2] == "tbd":
            count_str = f"tbd:{parts[-1]}"
            skill_name = ":".join(parts[1:-2])
        else:
            count_str = parts[-1]
            skill_name = ":".join(parts[1:-1])

        robot_types = [rt.strip() for rt in robot_types_str.split("|") if rt.strip()]

        count: Any
        if count_str.startswith("tbd:"):
            count = count_str
        else:
            try:
                count = int(count_str)
            except ValueError:
                count = 1

        return {
            "skill_name": skill_name.strip(),
            "assigned_robot_type": robot_types,
            "assigned_robot_count": count,
        }

    # If it does not match compact format, treat it as a plain skill_name.
    return {"skill_name": skill.strip(), "assigned_robot_type": [], "assigned_robot_count": 1}


def parse_compact_edge(edge: Union[str, Dict]) -> Dict[str, Any]:
    """Parse a compact edge string or existing dict into a unified edge dictionary.

    Format:
    - Normal edge: "T1->T2"
    - Conditional edge: "T1->T2:condition_expression"

    Returns:
        {"from": str, "to": str, "type": str, "condition"?: str}
    """
    if isinstance(edge, dict):
        return edge

    if not isinstance(edge, str):
        return {"from": "", "to": "", "type": "normal"}

    m = re.match(r"^\s*(T\w+)\s*->\s*(T\w+)\s*(?::(.+))?\s*$", edge)
    if m:
        from_node, to_node, condition = m.group(1), m.group(2), m.group(3)
        if condition and condition.strip():
            return {"from": from_node, "to": to_node, "type": "conditional", "condition": condition.strip()}
        return {"from": from_node, "to": to_node, "type": "normal"}

    return {"from": "", "to": "", "type": "normal", "_raw": edge}
