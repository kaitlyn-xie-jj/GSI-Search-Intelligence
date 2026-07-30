# -*- coding: utf-8 -*-
"""
Allocation post-processing module. Converts allocator output into timestep-based skill execution lists.
"""
import yaml
from typing import Dict, Any, List, Optional
from copy import deepcopy


def process_allocation_to_timestep_skills(
    atomic_tasks: List[Dict[str, Any]], 
    yaml_file_path: str, 
    robot_inventory: Dict[str, Any],
    allocation_data: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Process allocation results and generate timestep-based skill execution lists.
    
    Args:
        atomic_tasks: Atomic task list.
        yaml_file_path: YAML file path.
        robot_inventory: Robot inventory.
        allocation_data: Allocation data, optional and preferred when provided.
        
    Returns:
        {
            "robot_view": {...},           # Robot-view skill sequence.
            "task_view": {...},            # Task-view robot allocation.
            "timestep_skills": {...}       # Timestep-based skill list.
        }
    """
    # 1. Load allocation data.
    data = _load_allocation_data(yaml_file_path, allocation_data)
    if "error" in data:
        return data
    
    # 2. Check whether allocation succeeded.
    result = data.get('result', {})
    if not result.get('flagSuccess', False):
        return {"error": "Task allocation failed. Check the log for details."}
    
    # 3. Build the robot-name-to-type map.
    robot_name_to_type = _build_robot_type_map(robot_inventory)
    
    # 4. Build task ID and skill maps.
    task_id_map = {i: task['task_id'] for i, task in enumerate(atomic_tasks)}
    skills_per_task_per_robot_type = _build_skills_per_task_map(atomic_tasks)
    
    # 5. Parse allocation paths and build robot_view and task_view.
    robot_view, task_view = _parse_vehicle_paths(
        data.get('vehicle_paths', {}),
        task_id_map,
        skills_per_task_per_robot_type,
        robot_name_to_type
    )
    
    # 6. Verify all tasks were assigned.
    unassigned = _check_unassigned_tasks(atomic_tasks, task_view)
    if unassigned:
        return {
            "warning": f"Some tasks were not assigned: {unassigned}",
            "robot_view": robot_view,
            "task_view": task_view,
            "timestep_skills": {}
        }
    
    # 7. Generate timestep-based skill lists.
    timestep_skills = _build_timestep_skills(robot_view, robot_name_to_type, atomic_tasks)
    
    return {
        "robot_view": robot_view,
        "task_view": task_view,
        "timestep_skills": timestep_skills
    }


def _load_allocation_data(yaml_file_path: str, allocation_data: Optional[Dict]) -> Dict[str, Any]:
    """Load allocation data."""
    if allocation_data:
        return allocation_data
    if yaml_file_path:
        try:
            with open(yaml_file_path, 'r', encoding='utf-8') as f:
                documents = list(yaml.safe_load_all(f))
                return documents[-1] if documents else {}
        except FileNotFoundError:
            return {"error": f"File not found at '{yaml_file_path}'"}
        except yaml.YAMLError as e:
            return {"error": f"Error parsing YAML: {e}"}
    return {"error": "No allocation data source provided."}


def _build_robot_type_map(robot_inventory: Dict[str, Any]) -> Dict[str, str]:
    """Build a robot-name-to-type map."""
    robot_name_to_type = {}
    for robot_type, info in robot_inventory.items():
        for label in info.get("labels", []):
            robot_name_to_type[label] = robot_type
    return robot_name_to_type


def _build_skills_per_task_map(atomic_tasks: List[Dict]) -> Dict[str, Dict[str, List[str]]]:
    """Build a skill map for each task and robot type."""
    from modules.task_solver.sgi_planner.utils.compact_parsers import parse_compact_skill
    skills_map = {}
    for task in atomic_tasks:
        task_id = task['task_id']
        skills_map[task_id] = {}
        for skill in task.get('required_skills', []):
            if isinstance(skill, str):
                skill = parse_compact_skill(skill)
            skill_name = skill['skill_name']
            for robot_type in skill.get('assigned_robot_type', []):
                if robot_type not in skills_map[task_id]:
                    skills_map[task_id][robot_type] = []
                skills_map[task_id][robot_type].append(skill_name)
    return skills_map


def _parse_vehicle_paths(
    vehicle_paths: Dict,
    task_id_map: Dict[int, str],
    skills_per_task: Dict[str, Dict[str, List[str]]],
    robot_name_to_type: Dict[str, str]
) -> tuple:
    """Parse vehicle paths and build robot_view and task_view."""
    robot_view = {}
    task_view = {}
    
    for path_key, path_data in vehicle_paths.items():
        nodes = path_data.get('nodes', [])
        if not nodes or len(nodes) < 3:
            continue
        
        robot_name = nodes[0]
        task_nodes = nodes[1:-1]
        
        if robot_name not in robot_view:
            robot_view[robot_name] = {}
        
        for task_idx in task_nodes:
            if isinstance(task_idx, int) and task_idx in task_id_map:
                actual_task_id = task_id_map[task_idx]
                robot_type = robot_name_to_type.get(robot_name)
                if robot_type is None:
                    continue
                
                skills_for_task = skills_per_task.get(actual_task_id, {}).get(robot_type, [])
                robot_view[robot_name][actual_task_id] = skills_for_task
                
                if actual_task_id not in task_view:
                    task_view[actual_task_id] = []
                if robot_name not in task_view[actual_task_id]:
                    task_view[actual_task_id].append(robot_name)
    
    return robot_view, task_view


def _check_unassigned_tasks(atomic_tasks: List[Dict], task_view: Dict) -> List[str]:
    """Check for unassigned tasks."""
    unassigned = []
    for task in atomic_tasks:
        task_id = task['task_id']
        if task_id not in task_view or not task_view[task_id]:
            unassigned.append(task_id)
    return unassigned


def _task_max_skills(task_id: str, robot_view: Dict[str, Dict[str, List[str]]]) -> int:
    """Maximum skill count for this task across all robots."""
    cnt = 0
    for robot_name, tasks in robot_view.items():
        if task_id in tasks:
            cnt = max(cnt, len(tasks[task_id]))
    return max(cnt, 1)


def _build_timestep_skills(
    robot_view: Dict[str, Dict[str, List[str]]],
    robot_name_to_type: Dict[str, str],
    atomic_tasks: List[Dict[str, Any]]
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Build timestep-based skill lists from task dependencies.

    Core idea: use dependencies in atomic_tasks for topological sorting and compute
    each task's level. Tasks at the same level can run in parallel. Higher-level
    tasks must wait until all lower-level tasks finish. Multiple skills in the same
    task occupy consecutive timesteps in order.

    Returns:
        {
            "0": {
                "UGV_01": {"skill_str": "navigate<xxx>", "task_id": "T3"},
                ...
            },
            ...
        }
    """
    from collections import defaultdict, deque

    # 1. Build the task dependency graph and compute each task level with topological DP.
    task_ids = [t['task_id'] for t in atomic_tasks if t.get('task_id')]
    task_deps: Dict[str, List[str]] = {}
    for t in atomic_tasks:
        tid = t.get('task_id')
        if tid:
            task_deps[tid] = [d for d in (t.get('dependencies') or []) if d in set(task_ids)]

    indeg: Dict[str, int] = {tid: 0 for tid in task_ids}
    succ: Dict[str, List[str]] = defaultdict(list)
    for tid, deps in task_deps.items():
        for dep in deps:
            succ[dep].append(tid)
            indeg[tid] += 1

    level: Dict[str, int] = {tid: 0 for tid in task_ids}
    q = deque([tid for tid in task_ids if indeg[tid] == 0])
    while q:
        u = q.popleft()
        for v in succ[u]:
            if level[v] < level[u] + 1:
                level[v] = level[u] + 1
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)

    # 2. Group tasks by level.
    max_level = max(level.values()) if level else 0
    tasks_by_level: Dict[int, List[str]] = defaultdict(list)
    for tid, lv in level.items():
        tasks_by_level[lv].append(tid)

    # 3. Compute the max skill count per level to decide how many timesteps it uses.
    #    Multiple skills in the same task occupy consecutive timesteps in order.

    # 4. Build timesteps by level and skill index.
    timestep_skills: Dict[str, Dict[str, Dict[str, Any]]] = {}
    current_ts = 0

    for lv in range(max_level + 1):
        tids_in_level = tasks_by_level.get(lv, [])
        if not tids_in_level:
            continue

        # Maximum skill steps in this level.
        max_skill_steps = max(_task_max_skills(tid, robot_view) for tid in tids_in_level)

        for skill_idx in range(max_skill_steps):
            ts_skills: Dict[str, Dict[str, Any]] = {}
            for robot_name, tasks in robot_view.items():
                for tid in tids_in_level:
                    if tid not in tasks:
                        continue
                    skill_list = tasks[tid]
                    if skill_idx < len(skill_list):
                        ts_skills[robot_name] = {
                            "skill_str": skill_list[skill_idx],
                            "task_id": tid
                        }
            if ts_skills:
                timestep_skills[str(current_ts)] = ts_skills
                current_ts += 1

    return timestep_skills
