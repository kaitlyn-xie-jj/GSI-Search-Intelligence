from .solver_backend import gp
import os
from typing import List

class PathDecomposer:
    """
    Encapsulates path decomposition optimization problem for a single robot type.
    Finds individual robot paths that satisfy the main model's flow solution.
    """
    def __init__(self, type_id: int, graph, num_robots: int, task_num: int, optimize_decomposition: bool = False, shared_task_groups: List[List[int]] = None):
        self.type_id = type_id
        self.graph = graph
        self.num_robots = num_robots
        self.task_num = task_num
        self.optimize_decomposition = optimize_decomposition
        self.shared_task_groups = shared_task_groups if shared_task_groups is not None else []
        
        # Create optimization model
        self.model = gp.Model(f"PathFinder_Type{type_id}")
        self.model.setParam('OutputFlag', 0)
        self.model.setParam('Threads', max(1, int(os.environ.get("GSI_TANGO_SOLVER_THREADS", "1"))))
        
        # Decision variables: p[robot_idx, edge_id] = 1 if robot uses edge
        self.p_var = self.model.addVars(num_robots, graph.edge_num(), vtype=gp.GRB.BINARY, name="p")

        # This helps in formulating the shared capability constraint cleanly.
        self.V_var = self.model.addVars(num_robots, self.task_num, vtype=gp.GRB.BINARY, name="V")
        
        # Set objective
        self._set_objective()
    
    def _set_objective(self):
        """Set optimization objective based on configuration"""
        if self.optimize_decomposition:
            # Minimize total energy cost across all robots
            objective_expr = gp.LinExpr(0)
            for r in range(self.num_robots):
                for e in range(self.graph.edge_num()):
                    edge_cost = self.graph.edge(e).eng_cost * 0.1
                    objective_expr += self.p_var[r, e] * edge_cost
            self.model.setObjective(objective_expr, gp.GRB.MINIMIZE)
        else:
            # Just find feasible solution
            self.model.setObjective(0, gp.GRB.MINIMIZE)
    
    def add_capacity_constraints(self, x_solution, sub2x_id_func):
        """Add capacity constraints: robot paths cannot exceed main model flow"""
        for edge_id in range(self.graph.edge_num()):
            x_id = sub2x_id_func(self.type_id, edge_id)
            capacity = x_solution.get(x_id, 0) + 1e-5
            self.model.addConstr(
                gp.quicksum(self.p_var[r, edge_id] for r in range(self.num_robots)) <= capacity,
                name=f"capacity_{edge_id}"
            )
    
    def add_path_constraints(self, x_solution, sub2x_id_func):
        """Add flow conservation and start-end constraints for each robot"""
        for r_idx in range(self.num_robots):
            start_node = self.task_num + r_idx
            end_node = self.task_num + self.num_robots + r_idx
            
            # Check if robot is dispatched
            flow_out = sum(x_solution.get(sub2x_id_func(self.type_id, eid), 0) 
                          for eid in self.graph.node(start_node).out_edge_ids)
            if flow_out < 0.5:
                continue  # Robot not dispatched
            
            # Flow conservation for task nodes
            for task_id in range(self.task_num):
                in_flow = gp.quicksum(self.p_var[r_idx, eid] 
                                     for eid in self.graph.node(task_id).in_edge_ids)
                out_flow = gp.quicksum(self.p_var[r_idx, eid] 
                                      for eid in self.graph.node(task_id).out_edge_ids)
                self.model.addConstr(in_flow == out_flow, name=f"flow_cons_{r_idx}_{task_id}")
            
            # Robot must leave its start node exactly once
            self.model.addConstr(
                gp.quicksum(self.p_var[r_idx, eid] 
                           for eid in self.graph.node(start_node).out_edge_ids) == 1,
                name=f"start_{r_idx}"
            )
            
            # Optional: force robot to return to its own end node
            # self.model.addConstr(
            #     gp.quicksum(self.p_var[r_idx, eid] 
            #                for eid in self.graph.node(end_node).in_edge_ids) == 1,
            #     name=f"end_{r_idx}"
            # )

        for r_idx in range(self.num_robots):
            for task_id in range(self.task_num):
                # A robot 'r' is considered to have visited task 'i' if it has a flow of 1 into that task's node.
                in_flow_to_task = gp.quicksum(self.p_var[r_idx, eid] 
                                              for eid in self.graph.node(task_id).in_edge_ids)
                # Since p_var is binary, in_flow_to_task will be 0 or 1.
                # This creates a direct link: V_var = 1 if and only if the robot enters the task node.
                self.model.addConstr(self.V_var[r_idx, task_id] == in_flow_to_task, 
                                     name=f"Link_V_p[{r_idx},{task_id}]")
                
    def add_shared_capability_constraints(self):
        """Adds constraints to ensure one robot handles all tasks within a shared group."""
        if not self.shared_task_groups:
            return

        for g_idx, task_group in enumerate(self.shared_task_groups):
            # A task group is a list of task IDs, e.g., [0, 2]
            if not task_group:
                continue
            
            # Z_var[r] = 1 if robot 'r' is chosen to handle this specific group 'g'.
            Z_var = self.model.addVars(self.num_robots, vtype=gp.GRB.BINARY, name=f"Z_group{g_idx}")

            # Constraint 1: Coverage
            # Exactly one robot from this type MUST be assigned to handle this group.
            self.model.addConstr(gp.quicksum(Z_var[r] for r in range(self.num_robots)) == 1, 
                                 name=f"GroupCoverage[{g_idx}]")

            # Constraint 2: Forcing
            # If a robot 'r' is chosen (Z_var[r] = 1), it is forced to visit ALL tasks in the group.
            for r_idx in range(self.num_robots):
                for task_id in task_group:
                    # This implies: Z_var[r] = 1 ==> V_var[r, task] = 1
                    # In linear form: V_var[r, task] >= Z_var[r]
                    self.model.addConstr(self.V_var[r_idx, task_id] >= Z_var[r_idx], 
                                         name=f"ForceVisit[{g_idx},{r_idx},{task_id}]")
    
    def solve(self):
        """Solve the path decomposition model"""
        self.model.optimize()
        return self.model.Status == gp.GRB.OPTIMAL
    
    def extract_paths(self):
        """Extract individual robot paths from solution"""
        paths = []
        for r_idx in range(self.num_robots):
            start_node = self.task_num + r_idx
            
            # Check if robot has a path
            flow_out = sum(self.p_var[r_idx, eid].X 
                          for eid in self.graph.node(start_node).out_edge_ids)
            if flow_out < 0.5:
                continue
            
            # Trace robot's path
            path_info = self._trace_path(r_idx, start_node)
            if path_info:
                paths.append(path_info)
        
        return paths
    
    def _trace_path(self, robot_idx: int, start_node: int):
        """Trace a single robot's path through the graph"""
        path_nodes = [start_node]
        path_edges = []
        path_energy = 0.0
        
        current = start_node
        # Continue until reaching an end node
        while current < self.task_num + self.num_robots:
            found_next = False
            for edge_id in self.graph.node(current).out_edge_ids:
                if self.p_var[robot_idx, edge_id].X > 0.5:
                    edge = self.graph.edge(edge_id)
                    path_edges.append(edge_id)
                    path_energy += edge.eng_cost
                    current = edge.edge.node2
                    path_nodes.append(current)
                    found_next = True
                    break
            
            if not found_next:
                break
        
        return {
            'nodes': path_nodes,
            'edges': path_edges,
            'energy': path_energy
        }
