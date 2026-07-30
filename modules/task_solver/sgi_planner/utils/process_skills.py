import json


def _format_skill_block(
    skills_to_format: list,
    include_details: bool = False,
    include_sync_wait: bool = False,
) -> list:
    """Format a group of skill dictionaries as Markdown list items.

    Returns the list of generated Markdown lines.
    """
    lines: list = []
    # Sort by skill name for stable output; missing names go last.
    skills_sorted = sorted(
        skills_to_format,
        key=lambda s: (s.get("name") is None, s.get("name", ""))
    )
    for skill in skills_sorted:
        name = skill.get("name", "N/A")

        # Control whether sync_wait is output.
        if not include_sync_wait and name == "sync_wait":
            continue

        desc = skill.get("description", "No description.")
        lines.append(f"- **{name}:** {desc}")

        if include_details:
            if "precondition" in skill:
                lines.append(f"  - *Precondition:* {skill['precondition']}")
            if "effect" in skill:
                lines.append(f"  - *Effect:* {skill['effect']}")
    return lines


def get_robot_skills(robot_types, robot_skill_source):
    """
    Get the skill library for specified robot types.
    
    Args:
        robot_types: Robot type as a string or list, such as "UAV" or ["UAV", "UGV"].
        robot_skill_source: Complete robot skill library dictionary.
    
    Returns:
        dict: Dictionary containing skills for the specified robot types.
    """
    # Ensure input is a list.
    if isinstance(robot_types, str):
        robot_types = [robot_types]
    
    # Filter skills for the specified types.
    filtered_skills = {
        robot_type: robot_skill_source[robot_type] 
        for robot_type in robot_types 
        if robot_type in robot_skill_source
    }
    
    return filtered_skills


def skill_library_to_markdown(skill_library: dict, include_details: bool = False, include_sync_wait: bool = False,) -> str:
    """
    Convert a flat robot skill dictionary to Markdown.

    Expected input example where each robot type directly maps skill_x to skill data:
    {
        "UAV": {
            "skill_1": {...},
            "skill_2": {...}
        },
        "Quadruped": {
            "skill_1": {...}
        }
    }

    Args:
        skill_library: Robot skill library dictionary in flat structure.
        include_details: If True, append Precondition and Effect after each skill.

    Returns:
        Markdown string.
    """
    markdown_parts: list[str] = []

    for robot_type, skills_map in skill_library.items():
        if robot_type.upper() in {"UAV", "UGV", "FW_UAV"}:
            header = robot_type.upper()
        else:
            header = robot_type.replace("_", " ").title()

        markdown_parts.append(f"## {header}")

        # skills_map is expected to be { "skill_1": {...}, ... }.
        if isinstance(skills_map, dict) and skills_map:
            markdown_parts.extend(
                _format_skill_block(
                    list(skills_map.values()),
                    include_details=include_details,
                    include_sync_wait=include_sync_wait,
                )
            )
        else:
            markdown_parts.append("_No skills available._")

        markdown_parts.append("")  # Separator blank line.

    return "\n".join(markdown_parts)
