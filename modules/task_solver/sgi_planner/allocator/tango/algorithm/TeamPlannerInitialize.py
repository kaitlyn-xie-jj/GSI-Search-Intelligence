import math
import os
from .solver_backend import gp
from typing import Dict, Any
from .Utils import ModelType, Graph
from modules.utils.system.logging_utils import dlog


def _calculate_edge_costs(pos1, pos2):
    """Compute (energy_cost, time_cost) between two positions based on distance."""
    dist = math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)
    # More complex models can be used here, or coefficients can be read from parameters
    eng_cost = dist * 0.1
    time_cost = dist * 0.01
    return eng_cost, time_cost


def initialize_model_impl(obj) -> None:
    obj.env = gp.Env()
    obj.env.setParam('OutputFlag', 1 if obj.param.verbose_level >= 1 else 0)
    obj.model = gp.Model(env=obj.env)
    obj.model.setParam('Threads', max(1, int(os.environ.get("GSI_TANGO_SOLVER_THREADS", "1"))))
    
    # Set solver parameters
    if obj.param.solver_max_time > 0.001:
        obj.model.setParam('TimeLimit', obj.param.solver_max_time)
    
    # Check continuous variable mode
    continuous_flags = {ModelType.TEAMPLANNER_CONDET}
    obj.flag_continuous = obj.param.flag_solver in continuous_flags
    
    # Add decision variables
    if obj.flag_continuous:
        obj.x_var = obj.model.addVars(obj.x_var_num, vtype=gp.GRB.CONTINUOUS, name="x")
        obj.y_var = obj.model.addVars(obj.y_var_num, vtype=gp.GRB.CONTINUOUS, name="y")
        obj.xr_var = obj.model.addVars(obj.x_var_num, vtype=gp.GRB.BINARY, name="xr")
        obj.yr_var = obj.model.addVars(obj.y_var_num, vtype=gp.GRB.BINARY, name="yr")
    else:
        obj.x_var = obj.model.addVars(obj.x_var_num, vtype=gp.GRB.BINARY, name="x")
        obj.y_var = obj.model.addVars(obj.y_var_num, vtype=gp.GRB.BINARY, name="y")
    
    # Add other variables
    obj.z_var = obj.model.addVars(obj.z_var_num, vtype=gp.GRB.BINARY, name="z")
    obj.q_var = obj.model.addVars(obj.q_var_num, vtype=gp.GRB.CONTINUOUS, name="q")

    if obj.param.shared_capability_groups:
        num_groups = len(obj.param.shared_capability_groups)
        obj.B_var = obj.model.addVars(obj.veh_type_num, num_groups, vtype=gp.GRB.BINARY, name="B")
        obj.X_group_var = obj.model.addVars(num_groups, obj.x_var_num, vtype=gp.GRB.BINARY, lb=0.0, name="X_group")
    
    # Initialize capacity parameters
    obj.sum_cap = [0.0] * obj.cap_num
    for c in range(obj.cap_num):
        for k in range(obj.veh_type_num):
            if obj.flag_continuous:
                obj.sum_cap[c] += obj.param.veh_param[k].cap_vector[c] * float(obj.param.veh_num_per_type[k])
            else:
                obj.sum_cap[c] += obj.param.veh_param[k].cap_vector[c]
    
def initialize_num_impl(obj) -> bool:
    obj.veh_type_num = obj.param.veh_type_num
    obj.task_num = obj.param.task_num
    obj.cap_num = obj.param.cap_num

    if obj.graph:
        # Calculate the maximum number of edges across all type graphs
        max_edge_num = 0
        for k in range(obj.veh_type_num):
            if obj.graph[k].edge_num() > max_edge_num:
                max_edge_num = obj.graph[k].edge_num()
        obj.edge_num = max_edge_num # Use maximum edge count as baseline

        # Calculate the maximum number of nodes across all type graphs
        max_node_num = 0
        for k in range(obj.veh_type_num):
            if obj.graph[k].node_num() > max_node_num:
                max_node_num = obj.graph[k].node_num()
        obj.node_num = max_node_num

    obj.edge_offset = [i * obj.edge_num for i in range(obj.veh_type_num)]
    obj.x_var_num = obj.veh_type_num * obj.edge_num
    obj.y_var_num = obj.veh_type_num * obj.task_num
    obj.z_var_num = obj.task_num
    obj.q_var_num = obj.task_num  # q_var is only associated with tasks, start nodes have default start time of 0

    return True

def get_params_impl(obj, curr_dir: str, filename: str = None, config_dicts: Dict[str, Dict] = None) -> bool:
    """YAML parameter reading implementation. Supports both file and dict collection inputs."""
    if config_dicts:
        obj.param.read_from_dicts(config_dicts)
    elif filename:
        obj.param.read_from_file(curr_dir, filename)
    else:
        raise ValueError("No configuration source provided (filename or config_dicts).")

    if obj.param.verbose_level >= 1:
        obj.param.print_config()
    return True

def get_graph_impl(obj) -> bool:
    """
    Dynamically build graph, dynamically generate three types of edges:
    1. Each robot start point -> each task
    2. Each task -> each robot end point
    3. Each task -> each other task
    """
    try:
        # Initialize graph objects for each type
        obj.graph = [Graph() for _ in range(obj.param.veh_type_num)]
        
        task_num = obj.param.task_num
        
        for v_id in range(obj.param.veh_type_num):
            num_robots = obj.param.veh_num_per_type[v_id]
            
            # 1. Add nodes
            # Add task nodes
            for i in range(task_num):
                obj.graph[v_id].add_node(i, obj.param.task_param[i].time_cost)
            
            # Add start and end points for each robot instance
            for i in range(num_robots):
                start_node_id = task_num + i
                obj.graph[v_id].add_node(start_node_id, 0.0)
            for i in range(num_robots):
                end_node_id = task_num + num_robots + i
                obj.graph[v_id].add_node(end_node_id, 0.0)
            
            # 2. Dynamically generate all edges
            # 2a. Generate "start point -> task" edges
            for i in range(num_robots):
                start_node_id = task_num + i
                robot_pos = obj.param.veh_locations[v_id][i].pos
                for j in range(task_num):
                    task_pos = obj.param.task_locations[j].pos
                    eng, time = _calculate_edge_costs(robot_pos, task_pos)
                    obj.graph[v_id].add_edge(start_node_id, j, 0, eng, time, True)
            
            # 2b. Generate "task -> end point" edges
            for j in range(task_num):
                task_pos = obj.param.task_locations[j].pos
                for i in range(num_robots):
                    end_node_id = task_num + num_robots + i
                    robot_pos = obj.param.veh_locations[v_id][i].pos
                    eng, time = _calculate_edge_costs(task_pos, robot_pos)
                    obj.graph[v_id].add_edge(j, end_node_id, 0, eng, time, True)
            
            # 2c. Generate "task -> task" edges (fully connected)
            for i in range(task_num):
                pos_i = obj.param.task_locations[i].pos
                for j in range(task_num):
                    if i == j:
                        continue
                    pos_j = obj.param.task_locations[j].pos
                    eng, time = _calculate_edge_costs(pos_i, pos_j)
                    obj.graph[v_id].add_edge(i, j, 0, eng, time, True)
        
        return True
    except IndexError as ie:
        dlog(f"Graph construction error: Index out of range. Please check if 'veh_num_per_type' matches the number of robots in 'scene.yaml'. Error: {ie}")
        return False
    except Exception as e:
        dlog(f"Unknown error occurred during graph construction: {str(e)}")
        return False
