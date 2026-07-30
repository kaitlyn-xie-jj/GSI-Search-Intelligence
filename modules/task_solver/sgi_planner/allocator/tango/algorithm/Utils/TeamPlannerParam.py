import os
import yaml
from dataclasses import dataclass, field
from typing import List, Dict, Any

from .VehicleParam import VehicleParam 
from .TaskParam import TaskParam, TaskReqGeq  
from .Enum import ModelType

@dataclass
class Location:
    name: str = ""
    pos: List[float] = field(default_factory=list)

@dataclass
class TeamPlannerParam:
    """
    Main data class for planner configuration. Supports loading from files or dicts.
    """
    # --- Core Solver Parameters ---
    flag_optimize_cost: bool = True
    flag_task_complete: bool = True
    optimize_path_decomposition: bool = True
    task_complete_reward: float = 100.0
    time_penalty: float = 1.0
    large_time: float = 1e4
    max_time: float = 1e3
    max_eng: float = 1e12
    flag_solver: ModelType = ModelType.TEAMPLANNER_CONDET
    solver_max_time: float = 200.0
    flag_not_use_unralavant: bool = True
    enforce_shared_capability: bool = False
    verbose_level: int = 0
    
    # --- File & Problem Structure ---
    scene_file: str = ""
    task_num: int = 0
    cap_num: int = 0
    veh_type_num: int = 0

    # --- Data Containers ---
    veh_param: List[VehicleParam] = field(default_factory=list)
    task_param: List[TaskParam] = field(default_factory=list)
    cap_type: List[int] = field(default_factory=list)
    veh_num_per_type: List[int] = field(default_factory=list)
    dependency_matrix: List[List[int]] = field(default_factory=list)
    veh_locations: List[List[Location]] = field(default_factory=list)
    task_locations: List[Location] = field(default_factory=list)
    shared_capability_groups: List[List[List[int]]] = field(default_factory=list)

    def clear(self):
        """Reset all container data to empty state."""
        self.veh_param.clear()
        self.task_param.clear()
        self.cap_type.clear()
        self.veh_num_per_type.clear()
        self.dependency_matrix.clear()
        self.veh_locations.clear()
        self.task_locations.clear()
        self.shared_capability_groups.clear()

    def _parse_planner_data(self, data: Dict[str, Any]):
        """Parse high-level planner parameters from a dictionary."""
        self.flag_optimize_cost = data.get('flag_optimize_cost', self.flag_optimize_cost)
        self.flag_task_complete = data.get('flag_task_complete', self.flag_task_complete)
        self.optimize_path_decomposition = data.get('optimize_path_decomposition', self.optimize_path_decomposition)
        self.flag_not_use_unralavant = data.get('flag_not_use_unralavant', self.flag_not_use_unralavant)
        self.task_complete_reward = float(data.get('task_complete_reward', self.task_complete_reward))
        self.time_penalty = float(data.get('time_penalty', self.time_penalty))
        self.large_time = float(data.get('large_time', self.large_time))
        self.max_time = float(data.get('max_time', self.max_time))
        self.max_eng = float(data.get('max_eng', self.max_eng))
        self.solver_max_time = float(data.get('solver_max_time', self.solver_max_time))
        self.task_num = int(data.get('task_num', self.task_num))
        self.cap_num = int(data.get('cap_num', self.cap_num))
        self.veh_type_num = int(data.get('veh_type_num', self.veh_type_num))
        self.verbose_level = int(data.get('verbose_level', self.verbose_level))
        self.shared_capability_groups = data.get('shared_capability_groups', [])
        self.enforce_shared_capability = data.get('enforce_shared_capability', self.enforce_shared_capability)

        solver_mapping = {"TEAMPLANNER_DET": ModelType.TEAMPLANNER_DET, "TEAMPLANNER_CONDET": ModelType.TEAMPLANNER_CONDET}
        self.flag_solver = solver_mapping.get(data.get('flag_solver', 'TEAMPLANNER_CONDET'), ModelType.TEAMPLANNER_CONDET)

        if 'cap_type' in data: self.cap_type = [int(x) for x in data['cap_type'][:self.cap_num]]
        if 'veh_num_per_type' in data: self.veh_num_per_type = [int(x) for x in data['veh_num_per_type'][:self.veh_type_num]]

    def _parse_vehicles(self, data: Dict[str, Any]):
        """Parse vehicle parameters from a dictionary."""
        idx = 0
        while f"vehicle_type{idx}" in data:
            key = f"vehicle_type{idx}"
            self.veh_param.append(VehicleParam(
                eng_max=data[key].get('engCap', 0.0),
                eng_cost=data[key].get('engCost', 0.0),
                cap_vector=data[key].get('capVector', [])
            ))
            idx += 1

    def _parse_tasks(self, data: Dict[str, Any]):
        """Parse task requirements and dependencies from a dictionary."""
        task_idx = 0
        while f"task{task_idx}" in data:
            task_key = f"task{task_idx}"
            task = TaskParam(time_cost=data[task_key].get("time_cost", 0.0))
            
            and_idx = 0
            while f"and{and_idx}" in data[task_key]:
                and_key = f"and{and_idx}"
                and_clause = []
                or_idx = 0
                while f"or{or_idx}" in data[task_key][and_key]:
                    or_key = f"or{or_idx}"
                    or_data = data[task_key][and_key][or_key]
                    and_clause.append(TaskReqGeq(
                        cap_id=or_data.get('capId', 0),
                        cap_req=or_data.get('capReq', 0.0),
                    ))
                    or_idx += 1
                task.req_fcn.append(and_clause)
                and_idx += 1
            self.task_param.append(task)
            task_idx += 1

        if "dependency" in data:
            self.dependency_matrix = data["dependency"]

    def _parse_locations(self, data: Dict[str, Any]):
        """Parse vehicle and task locations from scene data dictionary."""
        self.veh_locations = [[] for _ in range(self.veh_type_num)]
        for k in range(self.veh_type_num):
            type_key = f"vehicle_type{k}"
            if type_key in data:
                for robot_data in data[type_key]:
                    self.veh_locations[k].append(Location(name=robot_data.get('name', ''), pos=robot_data.get('pos', [])))
        
        if "tasks" in data:
            for task_data in data["tasks"]:
                self.task_locations.append(Location(name=task_data.get('name', ''), pos=task_data.get('pos', [])))

    def _load_and_parse(self, filename: str, parser_func):
        """Helper to load a YAML file and apply a parsing function."""
        with open(filename, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        parser_func(data)

    def read_from_dicts(self, configs: Dict[str, Dict[str, Any]]) -> bool:
        """Load configuration from a collection of in-memory dictionaries."""
        try:
            self.clear()
            self._parse_planner_data(configs.get("planner", {}))
            self._parse_vehicles(configs.get("vehicle", {}))
            self._parse_tasks(configs.get("task", {}))
            self._parse_locations(configs.get("scene", {}))
            return True
        except Exception as e:
            print(f"Error loading configuration from dictionaries: {e}")
            return False

    def read_from_file(self, curr_dir: str, filename: str) -> bool:
        """Load main configuration and nested files from disk."""
        try:
            self.clear()
            with open(filename, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f) or {}

            # Parse main planner file
            self._parse_planner_data(data)
            
            # Load and parse nested configuration files
            if 'vehicle_param_file' in data:
                self._load_and_parse(curr_dir + data['vehicle_param_file'], self._parse_vehicles)
            if 'task_param_file' in data:
                self._load_and_parse(curr_dir + data['task_param_file'], self._parse_tasks)
            if 'scene_file' in data:
                self.scene_file = curr_dir + data['scene_file']
                self._load_and_parse(self.scene_file, self._parse_locations)
            else:
                print("Error: Required 'scene_file' parameter not found.")
                return False

            return True
        except Exception as e:
            print(f"Error loading configuration from file: {e}")
            return False
            
    def print_config(self) -> None:
        """Print a formatted summary of the configuration."""
        print("\n=== Team Planner Configuration ===")
        print("\n[Core Settings]")
        print(f"  Optimize Cost: {self.flag_optimize_cost}")
        print(f"  Task Completion: {self.flag_task_complete}")
        print(f"  Optimize Path Decomposition: {self.optimize_path_decomposition}")
        print("\n[Algorithm Parameters]")
        print(f"  Solver: {self.flag_solver.name}")
        print(f"  Max Solver Time: {self.solver_max_time}s")
        print("\n[Task Configuration]")
        print(f"  Total Tasks: {self.task_num}")
        print(f"  Task Reward: {self.task_complete_reward:.2f}")
        print(f"  Time Penalty: {self.time_penalty:.2f}")
        print("\n[Vehicle Configuration]")
        print(f"  Total Vehicles: {sum(self.veh_num_per_type)}")
        print(f"  Vehicle Types: {self.veh_type_num}")
        print(f"  Vehicles per Type: {self.veh_num_per_type}")
        print("\n[Advanced Settings]")
        print(f"  Unrelated Filter: {self.flag_not_use_unralavant}")
        print(f"  Verbose Level: {self.verbose_level}")