import yaml
import os
import numpy as np
from collections import defaultdict
from modules.utils.system.logging_utils import dlog


def _solver_max_time() -> float:
    raw = os.environ.get("GSI_TANGO_SOLVER_MAX_TIME", "").strip()
    if not raw:
        return 120.0
    try:
        return max(0.001, float(raw))
    except ValueError:
        dlog(f"Invalid GSI_TANGO_SOLVER_MAX_TIME={raw!r}; using default 120.0")
        return 120.0

# ==============================================================================
# 1. Define raw problem data
# ==============================================================================
# RAW_DATA = {
#     "tasks": [
#         {"name": "task0", "pos": [80, 130], "time_cost": 1, "requirements": {"and0": {"or0": {"capId": 0, "capReq": 1.0}}}},
#         {"name": "task1", "pos": [120, 150], "time_cost": 1, "requirements": {"and0": {"or0": {"capId": 2, "capReq": 1.0}}}},
#         {"name": "task2", "pos": [100, 90], "time_cost": 2, "requirements": {"and0": {"or0": {"capId": 1, "capReq": 1.0}}}}
#     ],
#     "task_dependencies": [
#         [0, 1]  # Task 0 must finish before task 1.
#     ],
#     "shared_capability_groups": [
#         # Format: [ [task_id, and_id, or_id], ... ]
#         [ [1, 0, 0], [2, 0, 0] ]
#     ],
#     "robots": [
#         {"name": "robot_A1", "type": "type0", "pos": [10, 20], "eng_max": 1000.0, "capabilities": [1.0, 1.0, 0.0]},
#         {"name": "robot_B1", "type": "type1", "pos": [50, 50], "eng_max": 1200.0, "capabilities": [0.0, 1.0, 1.0]},
#         {"name": "robot_B2", "type": "type1", "pos": [52, 48], "eng_max": 1200.0, "capabilities": [0.0, 1.0, 1.0]}
#     ]
# }
RAW_DATA = {
    "tasks": [
        {
            "name": "task0_Clear_Entrance_A", 
            "pos": [30, 150], "time_cost": 5, 
            "requirements": {
                "and0": {"or0": {"capId": 0, "capReq": 2.5}}  # Requires strong heavy-lift capability.
            }
        },
        {
            "name": "task1_Scan_West_Wing", 
            "pos": [80, 130], "time_cost": 8, 
            "requirements": {
                "and0": {"or0": {"capId": 3, "capReq": 1.5}}, # Requirement 1: requires high-precision mapping.
                "and1": {"or0": {"capId": 1, "capReq": 2.0}}  # Requirement 2: requires advanced sensing.
            }
        },
        {
            "name": "task2_Deliver_Meds_to_West_Wing", 
            "pos": [90, 110], "time_cost": 3, 
            "requirements": {
                "and0": {"or0": {"capId": 2, "capReq": 1.0}}  # Requires medical supply delivery capability.
            }
        },
        {
            "name": "task3_Inspect_Collapsed_East_Wing", 
            "pos": [150, 70], "time_cost": 10, 
            "requirements": {
                "and0": {"or0": {"capId": 5, "capReq": 1.0}}, # Requirement 1: must establish a communication relay.
                "and1": { # Requirement 2: advanced sensing or light debris clearing.
                    "or0": {"capId": 1, "capReq": 2.5},
                    "or1": {"capId": 0, "capReq": 1.0}
                }
            }
        },
        {
            "name": "task4_Extinguish_Power_Substation_Fire", 
            "pos": [180, 40], "time_cost": 4, 
            "requirements": {
                "and0": {"or0": {"capId": 4, "capReq": 2.0}} # Requires firefighting capability.
            }
        }
    ],

    "task_dependencies": [
        [0, 1],  # Entrance A must be cleared before the west wing can be scanned.
        [4, 3],  # The substation fire must be extinguished before the east wing can be inspected.
        [3, 1],  # The relay established during east-wing inspection is required before scanning the west wing.
        [1, 2]   # Survivors must be found before supplies can be delivered to the west wing.
    ],

    "shared_capability_groups": [
        [ [1, 0, 0], [1, 1, 0] ],  # Group 1: west-wing mapping and advanced sensing must use the same robot for specialist reconnaissance.
        [ [1, 1, 0], [2, 0, 0] ],  # Group 2: the robot that performs west-wing advanced sensing must also deliver supplies for find-and-rescue.
        [ [0, 0, 0], [3, 1, 1] ]   # Group 3: the robot clearing Entrance A must also execute the east-wing light debris-clearing option.
    ],

    "robots": [
        # Define 6 capabilities: [0: heavy lift, 1: advanced sensing, 2: supply delivery, 3: mapping, 4: firefighting, 5: communication relay]
        {"name": "Hercules_1", "type": "type0", "pos": [10, 20], "eng_max": 2000.0, "capabilities": [3.0, 1.0, 1.0, 0.5, 0.0, 2.0]}, # Heavy engineering robot.
        {"name": "Scout_1",    "type": "type1", "pos": [20, 10], "eng_max": 1200.0, "capabilities": [0.5, 3.0, 2.0, 3.0, 0.0, 2.0]}, # Specialist reconnaissance and rescue robot.
        {"name": "Scout_2",    "type": "type1", "pos": [22, 12], "eng_max": 1200.0, "capabilities": [0.5, 3.0, 2.0, 3.0, 0.0, 2.0]}, # Specialist reconnaissance and rescue robot.
        {"name": "Firefly_1",  "type": "type2", "pos": [190, 10], "eng_max": 1500.0, "capabilities": [0.0, 1.0, 1.0, 1.0, 3.0, 2.0]}  # Rapid response and firefighting robot.
    ]
}

# ==============================================================================
# 2. Core generation functions (process raw data and create configuration dictionaries)
# ==============================================================================
def convert_numpy_types(obj):
    """Recursively convert numpy types to Python native types"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    return obj

def generate_config_files(raw_data):
    """
    Generate all required configuration dictionaries from raw data.
    """
    
    # --- Data processing and aggregation ---
    tasks = raw_data["tasks"]
    robots = raw_data["robots"]
    dependencies = raw_data["task_dependencies"]
    shared_capability_groups = raw_data.get("shared_capability_groups", [])
    
    # Group robots by type
    robots_by_type = defaultdict(list)
    for robot in robots:
        robots_by_type[robot["type"]].append(robot)
    
    # Determine robot types (and sort for consistency)
    robot_type_names = sorted(robots_by_type.keys())
    type_map = {name: i for i, name in enumerate(robot_type_names)}

    # 1. Generate scene.yaml content
    scene_data = {"tasks": []}
    for task in tasks:
        scene_data["tasks"].append({"name": task["name"], "pos": task["pos"]})
    
    for type_name in robot_type_names:
        type_key = f"vehicle_{type_name}"
        scene_data[type_key] = []
        for robot in robots_by_type[type_name]:
            scene_data[type_key].append({"name": robot["name"], "pos": robot["pos"]})

    # 2. Generate task_param.yaml content
    task_data = {}
    for i, task in enumerate(tasks):
        task_data[f"task{i}"] = {
            "time_cost": task["time_cost"],
            **task["requirements"]
        }
    
    dep_matrix = [[0] * len(tasks) for _ in range(len(tasks))]
    for dep in dependencies:
        dep_matrix[dep[0]][dep[1]] = 1
        dep_matrix[dep[1]][dep[0]] = -1
    task_data["dependency"] = dep_matrix

    # 3. Generate vehicle_param.yaml content
    vehicle_data = {}
    for type_name in robot_type_names:
        type_id = type_map[type_name]
        sample_robot = robots_by_type[type_name][0]
        vehicle_data[f"vehicle_type{type_id}"] = {
            "engCap": sample_robot["eng_max"],
            "engCost": 2.0,
            "capVector": sample_robot["capabilities"]
        }

    # 4. Generate planner_param.yaml content
    num_capabilities = len(robots[0]["capabilities"])
    planner_data = {
        "flag_optimize_cost": True, "flag_task_complete": True,
        "task_complete_reward": 100, "time_penalty": 1.0,
        "large_time": 10000.0, "max_time": 1000.0, "max_eng": 1e8,
        "flag_solver": "TEAMPLANNER_CONDET", "solver_max_time": _solver_max_time(),
        "flag_not_use_unralavant": False,
        "enforce_shared_capability": False,
        "shared_capability_groups": shared_capability_groups,
        "task_num": len(tasks),
        "cap_num": num_capabilities,
        "veh_type_num": len(robot_type_names),
        "veh_num_per_type": [len(robots_by_type[name]) for name in robot_type_names],
        "cap_type": [0] * num_capabilities,
        "vehicle_param_file": "/alloc_config/vehicle_param.yaml",
        "task_param_file": "/alloc_config/task_param.yaml",
        "scene_file": "/alloc_config/scene.yaml"
    }

    # Convert all numpy types to Python native types
    scene_data = convert_numpy_types(scene_data)
    task_data = convert_numpy_types(task_data)
    vehicle_data = convert_numpy_types(vehicle_data)
    planner_data = convert_numpy_types(planner_data)

    return scene_data, task_data, vehicle_data, planner_data

# ==============================================================================
# 3. Save functions (write configuration dictionaries to YAML files)
# ==============================================================================
def save_config_files(config_data, save_dir=None):
    """
    Save generated configuration data to corresponding YAML files.
    """
    scene_d, task_d, vehicle_d, planner_d = config_data
    
    # Ensure configuration directory exists
    if save_dir is None:
        base_dir = os.path.dirname(os.path.realpath(__file__))
        config_path = os.path.join(base_dir, "alloc_config")
    else:
        config_path = os.path.join(save_dir, "alloc_config")
    os.makedirs(config_path, exist_ok=True)

    # Save Scene file
    with open(os.path.join(config_path, "scene.yaml"), 'w') as f:
        yaml.dump(scene_d, f, default_flow_style=False, sort_keys=False)

    # Save Task Parameter file
    with open(os.path.join(config_path, "task_param.yaml"), 'w') as f:
        yaml.dump(task_d, f, default_flow_style=False, sort_keys=False)

    # Save Vehicle Parameter file
    with open(os.path.join(config_path, "vehicle_param.yaml"), 'w') as f:
        yaml.dump(vehicle_d, f, default_flow_style=False, sort_keys=False)

    # Save Planner Parameter file
    with open(os.path.join(config_path, "planner_param.yaml"), 'w') as f:
        yaml.dump(planner_d, f, default_flow_style=False, sort_keys=False)
        
    dlog(f"Config files have been successfully generated in: {config_path}.")

# ==============================================================================
# 4. Main execution logic
# ==============================================================================
if __name__ == '__main__':
    # Generate all configurations from raw data
    all_configs = generate_config_files(RAW_DATA)
    
    # Save files
    save_config_files(all_configs)
