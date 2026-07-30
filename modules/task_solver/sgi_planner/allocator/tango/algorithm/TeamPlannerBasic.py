import time
from .solver_backend import gp
from typing import List, Optional, Tuple
from .Utils import TeamPlannerParam, SymbolType, Graph, GraphEdgeParam
from .TeamPlannerPostProcess import *
from .TeamPlannerInitialize import *
from .TeamPlannerMain import *
from .TeamPlannerModel import *
from modules.utils.system.logging_utils import dlog

class TeamPlanner:
    def __init__(self):
        # Gurobi objects
        self.env: Optional[gp.Env] = None  
        self.model: Optional[gp.Model] = None
        
        # Decision variables (Gurobi Var lists)
        self.x_var: List[gp.Var] = []        # Binary edge selection
        self.y_var: List[gp.Var] = []        # Binary node visit
        self.z_var: List[gp.Var] = []        # Binary task completion
        self.q_var: List[gp.Var] = []        # Continuous task start time
        self.xr_var: List[gp.Var] = []       # Binary edge selection (redundant)
        self.yr_var: List[gp.Var] = []       # Binary node visit (redundant)
        self.alpha_var: List[gp.Var] = []    # Vector variables
        self.w_var: List[gp.Var] = []        # Vector variables
        self.B_var: List[gp.Var] = []        # Shared capability group assignment
        self.X_group_var: List[gp.Var] = []  # Shared capability group edge flow

        # Variable counts (integer types)
        self.x_var_num: int = 0
        self.y_var_num: int = 0
        self.z_var_num: int = 0
        self.q_var_num: int = 0

        # functional variables
        self.param: Optional[TeamPlannerParam] = TeamPlannerParam()
        self.graph: List[Graph] = []  # Graph object
        
        # Model parameters (integer types)
        self.veh_type_num: int = 0
        self.task_num: int = 0
        self.cap_num: int = 0
        self.edge_num: int = 0
        self.node_num: int = 0

        self.edge_offset: List[int] = []  # Edge offset
        self.sum_cap: List[float] = []
        
        # Timing control (float types)
        self.start_time: float = time.perf_counter()
        self.prev_time: float = 0.0
        self.solver_time: float = -1.0
        
        # Status flags (bool types)
        self.flag_optimized: bool = False
        self.flag_success: bool = False
        self.flag_continuous: bool = False
        
        # Counter
        self.constraint_num: int = 0
        self.pre_check_status: str = "Not Performed"

    def __del__(self):
        """Destructor calling cleanup"""
        self.clear()

    def clear(self) -> None:
        """Resource cleanup method"""
        # Release Gurobi resources
        if self.model is not None:
            self.model.dispose()
            self.model = None
        if self.env is not None:
            self.env.dispose()
            self.env = None
        
        # Reset all variables
        self.x_var.clear()
        self.y_var.clear()
        self.z_var.clear()
        self.q_var.clear()
        self.xr_var.clear()
        self.yr_var.clear()
        self.alpha_var.clear()
        self.w_var.clear()
        self.B_var.clear()
        self.X_group_var.clear()
        self.edge_offset.clear()
        self.sum_cap.clear()

        # 
        self.graph.clear()
        
        # Reinitialize numerical parameters
        self.veh_type_num = 0
        self.task_num = 0
        self.cap_num = 0
        self.edge_num = 0
        self.node_num = 0

        # Reset variable counts
        self.x_var_num = 0
        self.y_var_num = 0 
        self.z_var_num = 0
        self.q_var_num = 0
        
        # Reset timing
        self.prev_time = 0.0
        self.solver_time = -1.0
        
        # Reset flags
        self.flag_optimized = False
        self.flag_success = False
        self.flag_continuous = False

        #
        self.constraint_num = 0
        self.pre_check_status = "Not Performed"


    def get_time(self) -> float:
        """Get elapsed time since object creation (seconds)"""
        return time.perf_counter() - self.start_time
    
    def get_time_duration(self) -> float:
        """Get time interval since last call"""
        curr_time = self.get_time()
        duration = curr_time - self.prev_time
        self.prev_time = curr_time
        return duration
    
    def to_string(self, type: SymbolType, id: int, flag_offset: bool = False, flag_number_only: bool = False) -> str:
        """Convert ID to string representation based on type and flags"""
        offset = 1 if flag_offset else 0
        id_name = []
        
        if flag_number_only:
            if type == SymbolType.TEAMPLANNER_VEHC:
                id_name.append(str(id + offset))
            elif type == SymbolType.TEAMPLANNER_NODE:
                if id < self.task_num:
                    id_name.append(str(id + offset))
                else:
                    id_name.append("0")  # Non-task nodes uniformly return 0
            elif type in [SymbolType.TEAMPLANNER_TASK,
                        SymbolType.TEAMPLANNER_PATH,
                        SymbolType.TEAMPLANNER_AND,
                        SymbolType.TEAMPLANNER_OR]:
                id_name.append(str(id + offset))
            elif type == SymbolType.TEAMPLANNER_VEHCTYPE:
                id_name.append(str((id % self.veh_type_num) + offset))
        else:
            if type == SymbolType.TEAMPLANNER_VEHC:
                id_name.append(f"v{id + offset}")
            elif type == SymbolType.TEAMPLANNER_NODE:
                # Only distinguish between task nodes and non-task nodes (start/end points)
                if id < self.task_num:
                    id_name.append(f"m{id + offset}") # m for mission/task
                else:
                    # For all non-task nodes, use a common prefix since the specific start/end meaning depends on the graph it's in (graph[k])
                    id_name.append(f"depot_node_{id}")
            elif type == SymbolType.TEAMPLANNER_TASK:
                id_name.append(f"m{id + offset}")
            elif type == SymbolType.TEAMPLANNER_PATH:
                id_name.append(str(id + offset))  # Paths keep pure numeric format
            elif type in [SymbolType.TEAMPLANNER_AND, SymbolType.TEAMPLANNER_OR]:
                id_name.append(str(id + offset))
            elif type == SymbolType.TEAMPLANNER_VEHCTYPE:
                id_name.append(f"v{(id % self.veh_type_num) + offset}")

        return ''.join(id_name)

    ################# TeamPlannerPostProcess.py ##############
    # Cost calculation
    def get_cost(self) -> tuple[float, float, float, float, float]:
        return get_cost_impl(self)

    # Path and team analysis
    def get_team(self, task_team: list[list[int]], y_value: list[float] | None = None) -> None:
        return get_team_impl(self, task_team, y_value)
    
    def get_path(self) -> Tuple[list[list[int]], list[list[int]], list[float]]:
        return extract_individual_paths_impl(self)
    
    def get_team_continuous(self, veh_type: list[int], veh_flow: list[float],
                           veh_path: list[list[int]]) -> Tuple[List[float], List[float]]:
        return get_team_continuous_impl(self, veh_type, veh_flow, veh_path)
    
    def get_final_cost(self, veh_flow: list[float], veh_sum_eng: list[float]) -> float:
        return get_final_cost_impl(self, veh_flow, veh_sum_eng)

    # Solution handling
    def post_process(self, save_file_name: str, param_file_name: str) -> None:
        return post_process_impl(self, save_file_name, param_file_name)

    ################### TeamPlannerInitialize.py ##############
    # Model initialization
    def initialize_model(self) -> None:
        return initialize_model_impl(self)
    
    def initialize_num(self) -> bool:
        return initialize_num_impl(self)
    
    def get_params(self, curr_dir: str, file_name: Optional[str] = None, config_dicts: Optional[Dict[str, Any]] = None) -> bool:
        return get_params_impl(self, curr_dir, file_name, config_dicts)
    
    def get_graph(self) -> bool:
        return get_graph_impl(self)

    ###################### TeamPlannerMain.py ##############
    # Main functions
    def form_problem(self, curr_dir: str, param_file: Optional[str] = None, config_dicts: Optional[Dict[str, Any]] = None) -> bool:
        return form_problem_impl(self, curr_dir, param_file, config_dicts)
    
    def process_shared_capability_groups(self) -> bool:
        return process_shared_capability_groups_impl(self)
    
    def optimize(self) -> bool:
        return optimize_impl(self)

    ###################### TeamPlannerModel.py ##############
    # Common model
    def form_var_name_cost(self) -> None:
        return form_var_name_cost_impl(self)
    
    def form_common_model(self) -> None:
        return form_common_model_impl(self)
    
    def form_deterministic_energy(self) -> None:
        return form_deterministic_energy_impl(self)

    # Continuous model
    def form_continuous_model(self) -> None:
        return form_continuous_model_impl(self)
    
    def form_continuous_det_energy(self) -> None:
        return form_continuous_det_energy_impl(self)

    # Helper methods
    def sub2x_id(self, veh: int, node1: int, node2: int, type_: int) -> int:
        """Convert subgraph edge elements to global ID (via node pair)"""
        edge_id = self.graph[veh].edge2id(node1, node2, type_)
        return (edge_id + self.edge_offset[veh]) if edge_id >= 0 else -100

    def sub2x_id_from_edge(self, veh: int, edge_id: int) -> int:
        """Directly convert subgraph edge ID to global ID"""
        return edge_id + self.edge_offset[veh]

    def x_id2edge(self, id: int) -> tuple[int, GraphEdgeParam] | None:
        """Convert global ID back to subgraph edge parameters"""
        veh = id // self.edge_num
        edge_id = id - veh * self.edge_num
        edge_param = self.graph[veh].id2edge(edge_id)
        return (veh, edge_param) if edge_param else (None, None)

    def sub2y_id(self, veh: int, task: int) -> int:
        """Convert subgraph task ID to global task ID"""
        return task + veh * self.task_num

    def y_id2sub(self, id: int) -> tuple[int, int]:
        """Convert global task ID back to subgraph information"""
        veh = id // self.task_num
        return veh, id - veh * self.task_num

    def update_model(self) -> None:
        """Update Gurobi model"""
        self.model.update()

    def enable_lazy_constraints(self) -> None:
        """Enable lazy constraints"""
        dlog(f"Before: {self.model.getParam('LazyConstraints')}")
        if self.model.getParam('LazyConstraints') == 0:
            self.model.setParam('LazyConstraints', 1)
        dlog(f"After: {self.model.getParam('LazyConstraints')}")