from typing import List, Dict, Any, Optional
import time
import yaml
from pathlib import Path
from .Utils import SymbolType
from .Decomposer import PathDecomposer
from modules.utils.system.logging_utils import dlog

def get_cost_impl(obj) -> tuple[float, float, float, float, float]:
    """Calculate and return various cost components from the optimization model"""
    energy_cost = 0.0
    time_cost = 0.0
    other_cost = 0.0
    final_all_cost = 0.0

    # Accumulate energy cost from x variables
    for i in range(obj.x_var_num):
        energy_cost += obj.x_var[i].X * obj.x_var[i].Obj

    # Accumulate cost from y variables
    for i in range(obj.y_var_num):
        other_cost += obj.y_var[i].X * obj.y_var[i].Obj

    # Accumulate cost from z variables
    for i in range(obj.z_var_num):
        other_cost += obj.z_var[i].X * obj.z_var[i].Obj

    # Accumulate cost from alpha variables
    for var in obj.alpha_var:
        other_cost += var.X * var.Obj

    # Accumulate cost from w variables
    for var in obj.w_var:
        other_cost += var.X * var.Obj

    # Accumulate time cost from q variables
    for i in range(obj.q_var_num):
        time_cost += obj.q_var[i].X * obj.q_var[i].Obj

    # Add cost from xr variables if they exist
    if len(obj.xr_var) > 0:
        for i in range(obj.x_var_num):
            other_cost += obj.xr_var[i].X * obj.xr_var[i].Obj

    # Add cost from yr variables if they exist
    if len(obj.yr_var) > 0:
        for i in range(obj.y_var_num):
            other_cost += obj.yr_var[i].X * obj.yr_var[i].Obj

    # Calculate total final cost
    final_all_cost = energy_cost + time_cost + other_cost
    return energy_cost, time_cost, other_cost, final_all_cost

def get_final_cost_impl(obj, veh_flow: list, veh_sum_eng: list) -> tuple[float]:
    """Calculate the final deterministic cost components"""
    # Calculate deterministic energy cost from vehicle flows
    det_eng_cost = 0.0
    for path_id in range(len(veh_flow)):
        det_eng_cost += veh_flow[path_id] * veh_sum_eng[path_id]

    # Calculate cost from y variables
    y_var_cost = 0.0
    for i in range(obj.y_var_num):
        y_var_cost += obj.y_var[i].X * obj.y_var[i].Obj

    # Calculate time cost from q variables
    q_var_cost = 0.0
    for i in range(obj.q_var_num):
        q_var_cost += obj.q_var[i].X * obj.q_var[i].Obj

    # Sum all cost components
    final_cost = det_eng_cost + y_var_cost + q_var_cost

    return final_cost

def get_team_impl(obj, y_value: list = None) -> list:
    """Get vehicle-team assignments for each task based on y variables"""
    task_team = [[] for _ in range(obj.task_num)]
    
    # For each task and vehicle type, check if assigned (y > 0.5)
    for i in range(obj.task_num):
        for k in range(obj.veh_type_num):
            y_id = obj.sub2y_id(k, i)
            y_temp = obj.y_var[y_id].X if y_value is None else y_value[y_id]
            if y_temp > 0.5:
                task_team[i].append(k)
    return task_team

def get_team_continuous_impl(obj, veh_type: list, veh_flow: list, veh_path: list) -> tuple[list, list]:
    """
    Get continuous vehicle-team assignments by accumulating flows through task nodes.
    Returns both discrete team assignments and dense flow matrices.
    """
    # Initialize dense matrix tracking flow through each task node
    task_team_dense = [[0.0 for _ in range(obj.veh_type_num)] for _ in range(obj.task_num)]
    
    # Accumulate flows through each path's intermediate nodes
    for path_id in range(len(veh_path)):
        veh = veh_type[path_id]
        for node in veh_path[path_id]:
            # Only process the node if it's a valid task ID
            if node < obj.task_num:
                task_team_dense[node][veh] += veh_flow[path_id]
    
    # Convert dense matrix to discrete team assignments using threshold
    veh_eps = 1e-4  # Threshold for considering a vehicle assigned
    task_team = []
    for i in range(obj.task_num):
        team = [veh for veh in range(obj.veh_type_num) 
               if task_team_dense[i][veh] > veh_eps]
        task_team.append(team)
    
    return task_team, task_team_dense

def extract_individual_paths_impl(obj) -> tuple[list, list, list, list, list]:
    """
    Extract individual robot paths from multi-depot model solution.
    Returns: (vehicle_types, flows, paths, edges, energies)
    """
    final_veh_type = []
    final_veh_flow = []
    final_veh_path = []
    final_veh_edge = []
    final_veh_sum_eng = []
    
    # Get main model flow solution as capacity for path matching
    x_solution = {i: obj.x_var[i].X for i in range(obj.x_var_num)}

    # Determine from the MIP solution which robot type is responsible for which group
    groups_by_type = [[] for _ in range(obj.veh_type_num)]
    if obj.param.shared_capability_groups:
        num_groups = len(obj.param.shared_capability_groups)
        for g in range(num_groups):
            for k in range(obj.veh_type_num):
                if obj.B_var[k, g].X > 0.5:
                    tasks_to_visit = list(set([req[0] for req in obj.param.shared_capability_groups[g]]))
                    groups_by_type[k].append(tasks_to_visit)
                    break  # Each group is handled by exactly one type
    
    # Process each robot type
    for k in range(obj.veh_type_num):
        num_robots = obj.param.veh_num_per_type[k]
        
        # Create and configure path decomposer
        decomposer = PathDecomposer(
            type_id=k,
            graph=obj.graph[k],
            num_robots=num_robots,
            task_num=obj.task_num,
            optimize_decomposition=obj.param.optimize_path_decomposition,
            shared_task_groups=groups_by_type[k]
        )
        
        # Add constraints
        decomposer.add_capacity_constraints(x_solution, obj.sub2x_id_from_edge)
        decomposer.add_path_constraints(x_solution, obj.sub2x_id_from_edge)
        decomposer.add_shared_capability_constraints()
        
        # Solve
        if not decomposer.solve():
            dlog(f"Warning: Path decomposition failed for type {k}, possible numerical issues in main model.")
            continue
        
        # Extract paths
        paths = decomposer.extract_paths()
        
        # Add to results
        for path_info in paths:
            final_veh_type.append(k)
            final_veh_flow.append(1.0)
            final_veh_path.append(path_info['nodes'])
            final_veh_edge.append(path_info['edges'])
            final_veh_sum_eng.append(path_info['energy'])
    
    return final_veh_type, final_veh_flow, final_veh_path, final_veh_edge, final_veh_sum_eng


def post_process_impl(obj, save_file_name: str, param_file_name: str) -> Optional[Dict[str, Any]]:
    results_to_return = {}

    # Return early if model wasn't optimized
    if not obj.flag_optimized:
        dlog("Optimization did not run or failed. Status:", obj.model.Status)
        results_to_return = {
            "result": {
                "sharedCapabilityPreCheck": obj.pre_check_status,
                "flagSuccess": False,
                "optStatus": obj.model.Status,
                "reason": "Model was proven to be infeasible." if obj.model.Status == 3 else "Optimization failed."
            }
        }
        save_file_path = Path(save_file_name)
        save_file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_file_name, "a") as logfile:
            logfile.write("\n---\n") # YAML document separator
            logfile.write("# Team Planner Execution Log (FAILED)\n")
            logfile.write(f"# Run at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            logfile.write("---\n")
            logfile.write("result:\n")
            logfile.write(f"  sharedCapabilityPreCheck: {obj.pre_check_status}\n")
            logfile.write(f"  flagSuccess: false\n")
            logfile.write(f"  optStatus: {obj.model.Status}\n")
            if obj.model.Status == 3:
                logfile.write("  reason: Model was proven to be infeasible.\n")

        return results_to_return

    start_path_time = time.time()
    vehType, vehFlow, vehPath, vehEdge, vehSumEng = obj.get_path()
    all_paths_feasible = all(
        vehSumEng[i] <= obj.param.veh_param[vehType[i]].eng_max for i in range(len(vehType))
    )
    if not all_paths_feasible:
        obj.flag_success = False

    if obj.flag_continuous:
        taskTeam, _ = obj.get_team_continuous(vehType, vehFlow, vehPath)
        final_cost = obj.get_final_cost(vehFlow, vehSumEng)
    else:
        taskTeam = obj.get_team()
        taskTeamDense = []
        final_cost = sum(vehSumEng)

    energy_cost, time_cost, other_cost, final_all_cost = obj.get_cost()
    post_process_time = time.time() - start_path_time
    results_to_return['result'] = {
        "flagSuccess": obj.flag_success,
        "sharedCapabilityPreCheck": obj.pre_check_status,
        "energyConstraintCheck": 'passed' if all_paths_feasible else 'failed',
        "energyCost": energy_cost, 
        "timeCost": time_cost, 
        "otherCost": other_cost,
        "finalAllCost": final_all_cost, 
        "optStatus": obj.model.Status, 
        "objVal": obj.model.ObjVal,
        "MIPGap": obj.model.MIPGap, 
        "solverTime": obj.solver_time, 
        "postProcessTime": post_process_time,
        "plannerTime": obj.solver_time + post_process_time, 
        "finalCost": final_cost
    }
    results_to_return['team'] = { f"task{i}": {"assigned_robot_types": taskTeam[i]} for i in range(obj.task_num) }
    results_to_return['vehicle_paths'] = {}
    for k in range(len(vehType)):
        path = vehPath[k]
        if not path: continue
        veh_t = vehType[k]
        start_node = path[0]
        robot_idx = start_node - obj.task_num
        robot_name = "UnknownRobot"
        if veh_t < len(obj.param.veh_locations) and robot_idx < len(obj.param.veh_locations[veh_t]):
            robot_name = obj.param.veh_locations[veh_t][robot_idx].name
        results_to_return['vehicle_paths'][f'path{k}'] = {
            "robot_type": veh_t,
            "energy_consumed": vehSumEng[k],
            "energy_limit": obj.param.veh_param[veh_t].eng_max,
            "nodes": [robot_name] + path[1:-1] + [robot_name],
            "edges": vehEdge[k]
        }

    dlog("\n--- Optimization Results ---")
    dlog(f"Shared Capability Pre-check: {obj.pre_check_status}")
    dlog(f"Individual Energy Constraint Check Passed: {all_paths_feasible}")
    dlog("\nVehicle paths:")
    flagOffset = False
    for k in range(len(vehType)):
        path = vehPath[k]
        if not path: continue
        veh_type_id = vehType[k]
        start_node = path[0]
        robot_idx = start_node - obj.task_num
        robot_name = ""
        if veh_type_id < len(obj.param.veh_locations) and robot_idx < len(obj.param.veh_locations[veh_type_id]):
            robot_name = obj.param.veh_locations[veh_type_id][robot_idx].name
        start_node_str = f"start_depot_{start_node}({robot_name})"
        end_node_str = f"end_depot_{path[-1]}({robot_name})"
        mission_nodes_str = [obj.to_string(SymbolType.TEAMPLANNER_NODE, node, flagOffset) for node in path[1:-1]]
        full_path_list = [start_node_str] + mission_nodes_str + [end_node_str]
        path_str = " --> ".join(full_path_list)
        vehicle_type_str = obj.to_string(SymbolType.TEAMPLANNER_VEHC, veh_type_id, True)
        out_str_header = f"{vehicle_type_str} ({robot_name}, Path {k}):\t"
        validation_char = "Correct" if vehSumEng[k] <= obj.param.veh_param[veh_type_id].eng_max else "Error"
        dlog(f"{out_str_header}{path_str}\t\teng: {vehSumEng[k]:.2f} / {obj.param.veh_param[veh_type_id].eng_max:.2f} {validation_char}")
    dlog("\nTask teams:")
    for i in range(obj.task_num):
        out_str = f"{obj.to_string(2, i, flagOffset)}:\t"
        team_str = [obj.to_string(6, veh, flagOffset) for veh in taskTeam[i]]
        dlog(out_str + ", ".join(team_str))
    dlog("numVars:", obj.model.NumVars)
    dlog("numConstrs:", obj.model.NumConstrs)
    dlog("solverTime:", obj.solver_time)
    dlog("postProcessTime:", post_process_time)
    dlog("plannerTime:", obj.solver_time + post_process_time)
    dlog("flagSuccess:", obj.flag_success)
    
    save_file_path = Path(save_file_name)
    save_file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_file_name, "a") as logfile:
        logfile.write("\n---\n") # YAML document separator
        logfile.write("# Team Planner Execution Log\n")
        logfile.write(f"# Run at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        logfile.write("---\n")
        logfile.write("result:\n")
        logfile.write(f"  flagSuccess: {str(obj.flag_success).lower()}\n")
        logfile.write(f"  sharedCapabilityPreCheck: {obj.pre_check_status}\n")
        logfile.write(f"  energyConstraintCheck: {'passed' if all_paths_feasible else 'failed'}\n")
        logfile.write(f"  energyCost: {energy_cost:.6f}\n")
        logfile.write(f"  timeCost: {time_cost:.6f}\n")
        logfile.write(f"  otherCost: {other_cost:.6f}\n")
        logfile.write(f"  finalAllCost: {final_all_cost:.6f}\n")
        logfile.write(f"  optStatus: {obj.model.Status}\n")
        logfile.write(f"  objVal: {obj.model.ObjVal}\n")
        logfile.write(f"  objBound: {obj.model.ObjBound}\n")
        logfile.write(f"  MIPGap: {obj.model.MIPGap}\n")
        logfile.write(f"  varsNum: {obj.model.NumVars}\n")
        logfile.write(f"  constraintNum: {obj.constraint_num}\n")
        logfile.write(f"  solCount: {obj.model.SolCount}\n")
        logfile.write(f"  nodeCount: {obj.model.NodeCount}\n")
        logfile.write(f"  iterCount: {obj.model.IterCount}\n")
        logfile.write(f"  solverTime: {obj.solver_time}\n")
        logfile.write(f"  lastSolverTime: {obj.model.Runtime}\n")
        logfile.write(f"  postProcessTime: {post_process_time}\n")
        logfile.write(f"  plannerTime: {obj.solver_time + post_process_time}\n")
        logfile.write(f"  finalCost: {final_cost}\n")
        logfile.write("team:\n")
        for i in range(obj.task_num):
            id_list = [t for t in taskTeam[i]]
            logfile.write(f"  task{i}:\n")
            logfile.write(f"    assigned_robot_types: {id_list}\n")
        logfile.write("vehicle_paths:\n")
        for k in range(len(vehType)):
            path = vehPath[k]
            if not path: continue
            veh_t = vehType[k]
            logfile.write(f"  path{k}:\n")
            logfile.write(f"    robot_type: {veh_t}\n")
            logfile.write(f"    energy_consumed: {vehSumEng[k]:.4f}\n")
            logfile.write(f"    energy_limit: {obj.param.veh_param[veh_t].eng_max:.4f}\n")
            start_node = path[0]
            robot_idx = start_node - obj.task_num
            robot_name = f"Robot-{k}"
            if veh_t < len(obj.param.veh_locations) and robot_idx < len(obj.param.veh_locations[veh_t]):
                robot_name = obj.param.veh_locations[veh_t][robot_idx].name
            formatted_nodes = [robot_name] + path[1:-1] + [robot_name]
            logfile.write(f"    nodes: {str(formatted_nodes)}\n")
            logfile.write(f"    edges: {str(vehEdge[k])}\n")

        # Append parameter file contents
        yaml_param = yaml.safe_load(open(param_file_name, "r"))
        logfile.write(f"planner_param: {str(yaml_param)}\n")
        
        # Write variable solutions
        logfile.write("xVar: " + str([obj.x_var[i].X for i in range(obj.x_var_num)]) + "\n")
        logfile.write("yVar: " + str([obj.y_var[i].X for i in range(obj.y_var_num)]) + "\n")
        logfile.write("zVar: " + str([obj.z_var[i].X for i in range(obj.z_var_num)]) + "\n")
        logfile.write("q_var: " + str([obj.q_var[i].X for i in range(obj.q_var_num)]) + "\n")
        logfile.write("alphaVar: " + str([obj.alpha_var[i].X for i in range(len(obj.alpha_var))]) + "\n")
        logfile.write("wVar: " + str([obj.w_var[i].X for i in range(len(obj.w_var))]) + "\n")
        if len(obj.xr_var) > 0:
            logfile.write("xrVar: " + str([obj.xr_var[i].X for i in range(obj.x_var_num)]) + "\n")
        if len(obj.yr_var) > 0:
            logfile.write("yrVar: " + str([obj.yr_var[i].X for i in range(obj.y_var_num)]) + "\n")

    return results_to_return