import re
import numpy as np
from typing import Dict, List, Optional
from collections import defaultdict
import random

from modules.utils.geom_utils import area_centroid
from modules.task_solver.world_model.utils.plan_translate_utils import resolve_area_feature


def _is_current_location(label: Optional[str]) -> bool:
    """Determine whether a location means current_location."""
    if not label or not isinstance(label, str):
        return False
    return label.strip().lower() in ("current_location", "current location")


def _robot_type_from_skill_key(key: str) -> str:
    """Extract the robot type suffix from a skill key like 'task.idx@RobotType'."""
    return key.rsplit("@", 1)[-1] if "@" in key else ""


def _mean_robot_position(
    pos_map: Optional[Dict],
    available_robots: Optional[Dict],
) -> Optional[List[float]]:
    """Calculate the mean current position of all available robots."""
    if not pos_map or "robot" not in pos_map or not available_robots:
        return None
    all_pos: List[List[float]] = []
    for rtype, rtype_positions in pos_map.get("robot", {}).items():
        labels = (available_robots.get(rtype) or {}).get("labels", [])
        for rlabel in labels:
            p = rtype_positions.get(rlabel)
            if p:
                all_pos.append(p)
    if not all_pos:
        return None
    avg_x = sum(p[0] for p in all_pos) / len(all_pos)
    avg_y = sum(p[1] for p in all_pos) / len(all_pos)
    return [avg_x, avg_y]


def find_position(
    pos_map: Dict[str, Dict[str, Dict[str, List[float]]]], label: str
) -> Optional[List[float]]:
    """
    Find and return the position of an entity by its label in a multi-level, category-based dictionary.

    Args:
        pos_map (Dict): A dictionary containing positions structured as
                        {category: {subcategory: {label: [x, y]}}}.
        label (str): The label of the entity to find.

    Returns:
        Optional[List[float]]: The position coordinates [x, y] of the found entity,
                               or None if not found.
    """
    for category in pos_map.values():
        for subcategory in category.values():
            if label in subcategory:
                return subcategory[label]
    return None


def get_cybertown_center(
    building_data: Dict[str, Dict[str, List[float]]]
) -> List[float]:
    """
    Calculate the average position of all buildings in cybertown.
    """
    all_positions: List[List[float]] = []
    for type_group in building_data.values():
        for position in type_group.values():
            all_positions.append(position)

    if all_positions:
        avg_x = sum(pos[0] for pos in all_positions) / len(all_positions)
        avg_y = sum(pos[1] for pos in all_positions) / len(all_positions)
        return [avg_x, avg_y]
    else:
        return [0.0, 0.0]


def convert_config_for_tango(
    input_config,
    pos_map=None,
    available_robots=None,
    shared_skill_groups: Optional[List[List[str]]] = None,
    plan_list: Optional[List[Dict]] = None,
    goal_cfg: Optional[Dict] = None,
    area_boundaries: Optional[Dict] = None,
    category_map: Optional[Dict[str, str]] = None,
) -> dict:
    """
    Convert input format to the improved TANGO algorithm format
    """

    # Extract input parameters from the input dictionary
    task_descriptions = input_config.get("atomic_task_description", {})
    task_dependencies = input_config.get("dependency_description", [])
    task_locations = input_config.get("atomic_task_locations", {})
    robot_capabilities = input_config.get("robot_capability_description", {})
    skill_mapping = input_config.get("skill_index_mapping", {})
    skill_locator_map = input_config.get("skill_locator_map", {})

    goal_cfg = goal_cfg or {}
    area_boundaries = area_boundaries or {}
    category_map = category_map or {}

    # 1. Convert tasks to list format
    tasks: List[Dict] = []
    task_name_to_id: Dict[str, int] = {}

    for i, (task_name, task_info) in enumerate(task_descriptions.items()):
        task_name_to_id[task_name] = i

        # Get task position
        location_info = task_locations.get(task_name, {})
        location_label = location_info.get("initiation_location")
        if location_label == "UNKNOWN":
            location_label = location_info.get("location")

        pos: Optional[List[float]] = None

        # 0) current_location: mean current position of all robots.
        if _is_current_location(location_label):
            pos = _mean_robot_position(pos_map, available_robots)

        # 1) First try resolving from real_time_pos_map for rule nodes.
        if pos is None and location_label and pos_map:
            if (
                isinstance(location_label, str)
                and location_label.lower() == "cybertown"
                and "building" in pos_map
            ):
                pos = get_cybertown_center(pos_map["building"])
            else:
                pos = find_position(pos_map, location_label)

        # 2) If not found in pos_map and the label looks like an area token, resolve it as an area.
        if (
            pos is None
            and isinstance(location_label, str)
            and goal_cfg
            and area_boundaries
        ):
            if re.search(r"(BoundarySelection|PointRadius)", location_label, flags=re.I):
                area_feat = resolve_area_feature(
                    area_token=location_label,
                    goal_cfg=goal_cfg,
                    area_boundaries=area_boundaries,
                    category_map=category_map,
                )
                if area_feat:
                    if area_feat.get("kind") == "circle" and "center" in area_feat:
                        pos = [
                            float(area_feat["center"][0]),
                            float(area_feat["center"][1]),
                        ]
                    else:
                        center = area_centroid(area_feat)
                        if center:
                            pos = [float(center[0]), float(center[1])]

        if pos is None:
            pos = [0.0, 0.0]

        # Build requirements structure
        requirements: Dict[str, Dict[str, Dict[str, float]]] = {}
        skills = task_info["skills"]
        for j, (skill_name, skill_req) in enumerate(skills.items()):
            and_key = f"and{j}"
            requirements[and_key] = {
                "or0": {
                    "capId": skill_req["capId"],
                    "capReq": float(skill_req["capReq"]),
                }
            }

        # Create task dictionary
        task_dict = {
            "name": f"task{i}",
            "pos": pos,
            "time_cost": task_info["time_cost"],
            "requirements": requirements,
        }

        tasks.append(task_dict)

    # Check and handle duplicate positions.
    position_groups: Dict[tuple, List[int]] = defaultdict(list)
    for idx, task in enumerate(tasks):
        pos_key = tuple(task["pos"])
        position_groups[pos_key].append(idx)

    for pos_key, task_indices in position_groups.items():
        if len(task_indices) > 1:
            base_pos = list(pos_key)
            for idx in task_indices[1:]:
                perturbation_x = 100 * (2 * random.random() - 1)
                perturbation_y = 100 * (2 * random.random() - 1)
                tasks[idx]["pos"] = [
                    base_pos[0] + perturbation_x,
                    base_pos[1] + perturbation_y,
                ]

    # 2. Convert dependency relationships
    dependency_list: List[List[int]] = []
    for dep in task_dependencies:
        task1, task2 = dep
        id1 = task_name_to_id[task1]
        id2 = task_name_to_id[task2]
        dependency_list.append([id1, id2])

    shared_capability_groups: List[List[List[int]]] = []

    # Create a lookup map from the original plan for efficient validation
    task_map_for_validation: Dict[str, Dict] = {}
    if plan_list:
        task_map_for_validation = {task["task_id"]: task for task in plan_list}

    if shared_skill_groups and task_map_for_validation:
        for group in shared_skill_groups:
            common_robot_types = None
            is_valid_group = True

            # Step 1: Validate the group for robot type compatibility
            for skill_identifier in group:
                try:
                    task_name, skill_index_str = skill_identifier.split(".")
                    skill_index = int(skill_index_str)

                    # Get the required robot types from the original plan_list
                    task_data = task_map_for_validation[task_name]
                    skill_item = task_data["required_skills"][skill_index]
                    if isinstance(skill_item, str):
                        from modules.task_solver.sgi_planner.utils.compact_parsers import parse_compact_skill
                        skill_item = parse_compact_skill(skill_item)
                    required_types = set(
                        skill_item.get("assigned_robot_type", [])
                    )

                    if common_robot_types is None:
                        common_robot_types = required_types
                    else:
                        common_robot_types.intersection_update(required_types)

                    if not common_robot_types:
                        is_valid_group = False
                        break
                except (KeyError, IndexError, ValueError):
                    is_valid_group = False
                    break

            if not is_valid_group:
                continue  # Skip to the next group

            # Step 2: If valid, convert the group to the allocator format
            new_group: List[List[int]] = []
            for skill_identifier in group:
                try:
                    task_name, original_skill_index_str = skill_identifier.split(".")
                    original_skill_index = int(original_skill_index_str)
                    raw_target = skill_locator_map.get(
                        (task_name, original_skill_index)
                    )
                    if not raw_target:
                        continue
                    task_info = task_descriptions.get(task_name)
                    if not task_info:
                        continue

                    final_skill_keys = list(task_info["skills"].keys())
                    if isinstance(raw_target, list):

                        preferred = [
                            k
                            for k in raw_target
                            if k in task_info["skills"]
                            and (
                                common_robot_types is None
                                or _robot_type_from_skill_key(k) in common_robot_types
                            )
                        ]
                        candidates = preferred or [
                            k for k in raw_target if k in task_info["skills"]
                        ]
                        if not candidates:
                            new_group = []
                            break

                        target_skill_key = next(
                            k for k in final_skill_keys if k in candidates
                        )
                    else:
                        target_skill_key = raw_target

                    and_index = final_skill_keys.index(target_skill_key)
                    task_id = task_name_to_id[task_name]
                    or_index = 0
                    new_group.append([task_id, and_index, or_index])
                except (ValueError, KeyError, IndexError, StopIteration):
                    new_group = []
                    break

            if len(new_group) > 1:
                shared_capability_groups.append(new_group)

    # 3. Convert robots to list format
    robots: List[Dict] = []
    robot_id = 0

    for type_idx, (robot_type, robot_info) in enumerate(
        robot_capabilities.items()
    ):
        robot_labels: List[str] = []
        robot_positions: List[List[float]] = []

        if pos_map and "robot" in pos_map and robot_type in pos_map["robot"]:
            if available_robots:
                robot_labels = available_robots.get(robot_type, {}).get("labels", [])
            for rlabel in robot_labels:
                robot_positions.append(
                    pos_map["robot"][robot_type].get(rlabel, [0.0, 0.0])
                )

        num_robots = 0
        if available_robots:
            num_robots = available_robots.get(robot_type, {}).get("num", 0)

        for instance_idx in range(num_robots):
            robot_dict = {
                "name": robot_labels[instance_idx],
                "type": f"type{type_idx}",
                "pos": robot_positions[instance_idx],
                "eng_max": 1000000.0,
                "capabilities": [float(x) for x in robot_info["capability"]],
            }
            robots.append(robot_dict)
            robot_id += 1

    output_config = {
        "tasks": tasks,
        "task_dependencies": dependency_list,
        "shared_capability_groups": shared_capability_groups,
        "robots": robots,
    }

    return output_config
