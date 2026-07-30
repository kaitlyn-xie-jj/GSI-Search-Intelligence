# -*- coding: utf-8 -*-
"""
Multi-robot parameter coordination utilities.
"""

from typing import Dict, List, Tuple, Any
from modules.config.base.enums import SkillName
from modules.task_solver.world_model.utils.search_partition_utils import (
    split_rectangle,
    split_polygon_stripes_strict,
    split_circle_sectors,
    split_polyline_by_length,
)


def coordinate_multi_robot_params(
    skill_infos_in_step: List[Tuple[str, Dict[str, Any]]]
) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Coordinate multi-robot parameters within the same step:
    - Split geometry areas for SEARCH.
    
    Args:
        skill_infos_in_step: [(robot_label, skill_info), ...] skill list for the same time step.
        
    Returns:
        Coordinated skill list.
    """
    if len(skill_infos_in_step) <= 1:
        return skill_infos_in_step

    # Group by (skill, task_id).
    skill_groups: Dict[str, List[Tuple[str, Dict]]] = {}
    for robot_label, skill_info in skill_infos_in_step:
        skill = skill_info.get('skill')
        task_id = (skill_info.get('params') or {}).get('task_id')
        key = f"{skill}::{task_id}"
        skill_groups.setdefault(key, []).append((robot_label, skill_info))

    result: List[Tuple[str, Dict]] = []
    for key, group_items in skill_groups.items():
        if len(group_items) > 1:
            skill_name = group_items[0][1].get('skill')
            if skill_name in (SkillName.SEARCH.value,):
                result.extend(_split_area_for_robots_by_type(group_items))
                continue
        result.extend(group_items)
    return result


def _split_area_for_robots_by_type(
    robot_skill_pairs: List[Tuple[str, Dict[str, Any]]]
) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Group by robot type, then split each group's area.
    
    Args:
        robot_skill_pairs: [(robot_label, skill_info), ...]
        
    Returns:
        Skill list after splitting.
    """
    if not robot_skill_pairs:
        return robot_skill_pairs

    # Group in appearance order to keep stable output.
    grouped: Dict[str, List[Tuple[str, Dict]]] = {}
    for robot_label, skill_info in robot_skill_pairs:
        rtype = robot_label.split('-', 1)[0] if isinstance(robot_label, str) else 'unknown'
        grouped.setdefault(rtype, []).append((robot_label, skill_info))

    out: List[Tuple[str, Dict]] = []
    for _, sub_group in grouped.items():
        if len(sub_group) <= 1:
            out.extend(sub_group)
        else:
            # Split each type subgroup separately.
            out.extend(_split_area_for_robots(sub_group))
    return out


def _split_area_for_robots(
    robot_skill_pairs: List[Tuple[str, Dict[str, Any]]]
) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Split search or patrol areas for multiple robots, preserving the original geometry form.
    
    Rules:
    - rectangle: split the bbox into equal vertical stripes.
    - area (polygon): strict trimmed stripes.
    - circle: split into equal sectors.
    - line: split into sub-polylines by arc length and pass buffer through.
    - point or other: share without splitting.
    
    Args:
        robot_skill_pairs: [(robot_label, skill_info), ...]
        
    Returns:
        Skill list after splitting.
    """
    n = len(robot_skill_pairs)
    if n == 0:
        return robot_skill_pairs

    first = robot_skill_pairs[0][1]
    area = (first.get('params') or {}).get('area')

    if not isinstance(area, dict) or 'kind' not in area:
        return robot_skill_pairs

    kind = area.get('kind')
    if kind == 'rectangle':
        parts = split_rectangle(area.get('coords', []), n)
    elif kind == 'area':
        parts = split_polygon_stripes_strict(area.get('coords', []), n)
    elif kind == 'circle':
        parts = split_circle_sectors(
            area.get('center'), 
            float(area.get('radius', 0.0) or 0.0), 
            n
        )
    elif kind == 'line':
        parts = split_polyline_by_length(area.get('coords', []), n)
        for p in parts:
            p['buffer'] = area.get('buffer')
    elif kind == 'point':
        parts = [area] * n
    else:
        parts = [area] * n

    result = []
    for idx, (robot_label, skill_info) in enumerate(robot_skill_pairs):
        sub = parts[min(idx, len(parts) - 1)]
        new_info = dict(skill_info)
        new_params = dict(new_info.get('params', {}))
        new_params['area'] = sub
        new_info['params'] = new_params
        result.append((robot_label, new_info))
    return result
