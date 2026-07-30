import re
from typing import Dict, Any
from modules.task_solver.sgi_planner.utils.compact_parsers import parse_compact_skill

def _iterate_all_skills(skills_data: dict):
    """
    A generator that iterates over all skill dictionaries for a robot,
    regardless of whether the library structure is flat or hierarchical.

    Args:
        skills_data: The dictionary for a single robot, e.g., robot_skill_library["UAV"].

    Yields:
        A dictionary for each individual skill.
    """
    # Check if the structure is hierarchical (contains 'basic_skills' or 'integrated_skills')
    if "basic_skills" in skills_data or "integrated_skills" in skills_data:
        if "basic_skills" in skills_data:
            yield from skills_data["basic_skills"].values()
        if "integrated_skills" in skills_data:
            yield from skills_data["integrated_skills"].values()
    else:
        # Otherwise, the structure is flat
        yield from skills_data.values()

def _create_regex(template: str) -> re.Pattern:
    """
    Correctly converts a skill template string into a compiled regular expression
    by replacing placeholders with patterns that match the actual instantiated content.
    """
    pattern_str = re.sub(r'<[^>]+>', r'<.*?>', template)
    escaped_parts = []
    i = 0
    while i < len(pattern_str):
        if pattern_str[i:i+5] == '<.*?>':
            escaped_parts.append('<.*?>')
            i += 5
        else:
            escaped_parts.append(re.escape(pattern_str[i]))
            i += 1
    pattern_str = ''.join(escaped_parts)
    return re.compile(f"^{pattern_str}$")


def transform_for_allocator(plan_list: list, robot_skill_library: dict, robot_counts: dict) -> dict:
    """
    Transforms the planner's output into a mathematical model for the allocation algorithm.
    This robust version correctly handles hierarchical and flat skill library structures.

    Args:
        plan_list: The JSON dictionary produced by the Planner Agent.
        robot_skill_library: The skill library (can be hierarchical or flat).
        robot_counts: A dictionary specifying the number of available robots of each type.

    Returns:
        A dictionary containing the structured mathematical descriptions.
    """
    
    skill_template_to_capid = {}
    template_matchers = []
    cap_id_counter = 0

    for robot_type, skills_data in robot_skill_library.items():
        for skill_info in _iterate_all_skills(skills_data):
            skill_name = skill_info['name']
            key = (skill_name, robot_type)
            if key not in skill_template_to_capid:
                skill_template_to_capid[key] = cap_id_counter
                template_matchers.append((_create_regex(skill_name), skill_name, robot_type))
                cap_id_counter += 1

    num_total_skills = len(skill_template_to_capid)
    atomic_task_description = {}
    dependency_description = []
    robot_capability_description = {}
    atomic_task_locations = {}
    skill_locator_map = {}
    tasks = plan_list

    for task in tasks:
        task_id = task["task_id"]
        task_skills = {}
        for original_skill_index, skill_assignment in enumerate(task.get("required_skills", [])):
            # Support compact format strings.
            if isinstance(skill_assignment, str):
                skill_assignment = parse_compact_skill(skill_assignment)

            instantiated_skill_name = skill_assignment["skill_name"]
            assigned_count = skill_assignment.get("assigned_robot_count", 1)
            assigned_robot_types = skill_assignment.get("assigned_robot_type", [])

            # Match both the regex and the assigned robot type
            for pattern, template, robot_type in template_matchers:
                if robot_type not in assigned_robot_types:
                    continue
                if not pattern.match(instantiated_skill_name):
                    continue
                skill_key = f"{template}@{robot_type}"
                if skill_key in task_skills:
                    continue

                matched_template = (template, robot_type)
                cap_id = skill_template_to_capid[matched_template]
                task_skills[skill_key] = {
                    "capId": cap_id,
                    "capReq": assigned_count
                }

                # Record the mapping from original position to final skill_key; it may be one-to-many.
                locator_key = (task_id, original_skill_index)
                if locator_key not in skill_locator_map:
                    skill_locator_map[locator_key] = []
                if skill_key not in skill_locator_map[locator_key]:
                    skill_locator_map[locator_key].append(skill_key)
        
        if not task_skills:
            raise ValueError(f"Task {task_id} has no matched skills in the skill library.")
        atomic_task_description[task_id] = {
            "skills": task_skills,
            "time_cost": 1
        }

        for dep_id in task["dependencies"]:
            dependency_description.append((dep_id, task_id))
            
        atomic_task_locations[task_id] = {
            "location": task.get("location", "UNKNOWN"),
            "initiation_location": task.get("initiation_location", "UNKNOWN")
        }

    # Set robot capabilities using (template, robot_type) key
    for robot_type, skills_data in robot_skill_library.items():
        capability_vector = [0] * num_total_skills
        for skill_info in _iterate_all_skills(skills_data):
            template = skill_info['name']
            key = (template, robot_type)
            if key in skill_template_to_capid:
                capability_vector[skill_template_to_capid[key]] = 1

        robot_capability_description[robot_type] = {
            "robot_num": robot_counts.get(robot_type, {}).get('num',0),
            "capability": capability_vector
        }

    return {
        "atomic_task_description": atomic_task_description,
        "dependency_description": dependency_description,
        "robot_capability_description": robot_capability_description,
        "atomic_task_locations": atomic_task_locations,
        "skill_index_mapping": {
            v: f"{k[0]}@{k[1]}" for k, v in skill_template_to_capid.items()
        },
        "skill_locator_map": skill_locator_map
    }
