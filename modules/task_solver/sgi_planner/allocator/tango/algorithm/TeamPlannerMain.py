import time
from .solver_backend import gp
import itertools
from typing import Dict, Any, List
from .Utils import ModelType
from modules.utils.system.logging_utils import dlog


def _is_group_feasible(group_set, param):
    """Check whether a single group can be handled by at least one robot type."""
    # Extract all unique required capability IDs in the group.
    caps_needed = {
        param.task_param[req[0]].req_fcn[req[1]][req[2]].cap_id
        for req in group_set
    }

    # Check all robot types to see whether one can satisfy every requirement.
    for k in range(param.veh_type_num):
        veh_caps = param.veh_param[k].cap_vector
        if all(veh_caps[cid] >= 1.0 for cid in caps_needed):
            return True  # Found a feasible type.

    return False  # No type can handle all requirements.


def form_problem_impl(obj, curr_dir: str = None, param_file: str = None, config_dicts: Dict[str, Dict] = None) -> bool:
    """
    Complete workflow for constructing the problem
    """
    # 1. Clear internal data
    obj.clear()

    # 2. Read all parameters
    obj.get_params(curr_dir=curr_dir, file_name=param_file, config_dicts=config_dicts)
    can_continue = obj.process_shared_capability_groups()
    if not can_continue:
        # A pre-check has already determined that we cannot continue solving,
        # and a dummy model has already been constructed inside obj
        return False
    
    # 3. Dynamically generate graph, without passing parameters
    obj.get_graph() 
    
    # 4. Based on the generated graph, initialize variable counts
    obj.initialize_num()
    
    # 5. Initialize Gurobi model and variables
    obj.initialize_model()

    # 6. Build variable names and objective function coefficients
    obj.form_var_name_cost()
    
    # 7. Build common model constraints
    obj.form_common_model()

    # 8. Based on solver type, build specific constraints
    if obj.param.flag_solver == ModelType.TEAMPLANNER_CONDET:
        obj.form_continuous_model()
        obj.form_continuous_det_energy()
    else:  # Default TEAMPLANNER_DET
        obj.form_deterministic_energy()

    return True

def optimize_impl(obj) -> bool:
    # Initialize flags
    obj.flag_success = True
    obj.flag_optimized = True
    
    # Set callback based on solver type
    start_time = time.time()  # Record start time

    obj.model.optimize()
    obj.constraint_num = obj.model.getAttr(gp.GRB.Attr.NumConstrs)
    if obj.model.getAttr(gp.GRB.Attr.SolCount) <= 0:
        obj.flag_success = False
        obj.flag_optimized = False

    obj.solver_time = time.time() - start_time

    return obj.flag_optimized

def process_shared_capability_groups_impl(obj) -> bool:
    # Case 1: no shared capability groups -> mark and allow optimization to continue
    if not obj.param.shared_capability_groups:
        obj.pre_check_status = "Not Applicable"
        return True

    # ---------- (1) Merge capability groups that have intersections ----------
    # First convert each group into a set of tuples so we can perform set operations
    groups_to_process = [set(map(tuple, g)) for g in obj.param.shared_capability_groups]

    while True:
        merged_in_pass = False
        i = 0
        while i < len(groups_to_process):
            j = i + 1
            while j < len(groups_to_process):
                # Check whether two groups share task IDs, which triggers merging.
                tasks_in_i = {req[0] for req in groups_to_process[i]}
                tasks_in_j = {req[0] for req in groups_to_process[j]}
                
                if not tasks_in_i.isdisjoint(tasks_in_j):
                    # 1. Simulate the merge by creating a temporary super group.
                    hypothetical_merged_group = groups_to_process[i].union(groups_to_process[j])
                    
                    # 2. Check whether the simulated super group is feasible.
                    if _is_group_feasible(hypothetical_merged_group, obj.param):
                        # 3. Accept the merge only when the result is feasible.
                        groups_to_process[i] = hypothetical_merged_group
                        del groups_to_process[j]
                        merged_in_pass = True
                        # Since element j was deleted, keep j fixed and compare the new element at this position.
                        continue # Skip j++.
                
                # If there is no overlap or the merge is infeasible, move to the next item.
                j += 1
            i += 1
        
        # Exit if no feasible merge occurred during the full pass.
        if not merged_in_pass:
            break

    # Convert the merged sets of tuples back into list format
    merged_groups = [list(map(list, g)) for g in groups_to_process]

    # Overwrite the original parameter with the merged result
    obj.param.shared_capability_groups = merged_groups

    # ---------- (2) Pre-check feasibility ----------
    if obj.param.shared_capability_groups:
        are_groups_feasible = pre_check_shared_capability(obj.param)
        if are_groups_feasible:
            obj.pre_check_status = "Passed"
            return True
        else:
            obj.pre_check_status = "Failed"

            if obj.param.enforce_shared_capability:
                # Hard requirement must be satisfied -> directly build an empty model and abort quickly
                dlog("[ERROR] Shared capability pre-check on MERGED groups failed. Planning aborted.")

                obj.flag_optimized = False
                obj.model = gp.Model()
                obj.model.setParam('TimeLimit', 0.01)
                obj.model.optimize()
                return False
            else:
                # We are allowed to ignore shared constraints and continue planning
                dlog("[WARNING] Shared capability pre-check on MERGED groups failed. Ignoring constraints.")
                obj.param.shared_capability_groups = []  # Clear them
                return True

    # In theory we shouldn't reach this branch (covered above), this is just a fallback
    obj.pre_check_status = "Not Applicable"
    return True

def pre_check_shared_capability(param) -> bool:
    """
    Checks whether, for each shared capability group, there exists at least one robot type
    that can satisfy all capability requirements in that group.
    """
    for group_id, group in enumerate(param.shared_capability_groups):
        is_group_feasible = False
        # Iterate over all robot types
        for k in range(param.veh_type_num):
            veh_caps = param.veh_param[k].cap_vector
            can_type_k_satisfy_group = True
            
            # Check whether this robot type can satisfy every requirement in the group
            for req in group:
                task_id, and_id, or_id = req
                cap_id = param.task_param[task_id].req_fcn[and_id][or_id].cap_id
                cap_req = param.task_param[task_id].req_fcn[and_id][or_id].cap_req
                
                if veh_caps[cap_id] < cap_req:
                    can_type_k_satisfy_group = False
                    break  # this type can't do it; move to next type
            
            if can_type_k_satisfy_group:
                is_group_feasible = True
                break  # this group is feasible; move to next group
        
        if not is_group_feasible:
            dlog(f"Pre-check failed for capability group {group_id}: {group}")
            return False  # if any group is infeasible, the entire check fails
            
    dlog("Shared capability pre-check passed.")
    return True
