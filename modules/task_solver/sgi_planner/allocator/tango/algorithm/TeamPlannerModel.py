from .solver_backend import gp

def form_var_name_cost_impl(obj) -> None:
    """Initialize variable names and costs (adapted for multi-depot model)"""
    # initialize x variable
    for x_id in range(obj.x_var_num):
        veh, edge_param = obj.x_id2edge(x_id)
        if edge_param:
            var_name = f"x[{obj.to_string(0, veh)}," \
                    f"{obj.to_string(1, edge_param.edge.node1)}," \
                    f"{obj.to_string(1, edge_param.edge.node2)}," \
                    f"{obj.to_string(3, edge_param.edge.type)}]"

            obj.x_var[x_id].setAttr("Obj", edge_param.eng_cost * 1)  # Use actual energy cost, edge_param.eng_cost, 0, 0.1
            obj.x_var[x_id].setAttr("VarName", var_name)
            if obj.flag_continuous:
                obj.x_var[x_id].setAttr("LB", 0.0)
                obj.x_var[x_id].setAttr("UB", float(obj.param.veh_num_per_type[veh]))
        else:
            # This is a variable corresponding to a "dummy edge", must be disabled, forcing its value to always be 0
            obj.x_var[x_id].setAttr("UB", 0.0) 
            obj.x_var[x_id].setAttr("VarName", f"x_dummy[{x_id}]")

    # initialize y variable
    for y_id in range(obj.y_var_num):
        veh, task = obj.y_id2sub(y_id)
        var_name = f"y[{obj.to_string(0, veh)}," \
                   f"{obj.to_string(2, task)}]"
        cost = obj.param.veh_param[veh].eng_cost
        obj.y_var[y_id].setAttr("Obj", 0)
        obj.y_var[y_id].setAttr("VarName", var_name)
        if obj.flag_continuous:
            obj.y_var[y_id].setAttr("LB", 0.0)
            obj.y_var[y_id].setAttr("UB", float(obj.param.veh_num_per_type[veh]))

    # initialize xr and yr variables
    if obj.flag_continuous:
        for x_id in range(obj.x_var_num):
            veh, edge_param = obj.x_id2edge(x_id)
            if edge_param:
                var_name = f"xr[{obj.to_string(0, veh)}," \
                        f"{obj.to_string(1, edge_param.edge.node1)}," \
                        f"{obj.to_string(1, edge_param.edge.node2)}," \
                        f"{obj.to_string(3, edge_param.edge.type)}]"
                obj.xr_var[x_id].setAttr("Obj", 0.0)
                obj.xr_var[x_id].setAttr("VarName", var_name)
            else:
                # Dummy edge, disable corresponding binary variable
                obj.xr_var[x_id].setAttr("UB", 0.0)
                obj.xr_var[x_id].setAttr("VarName", f"xr_dummy[{x_id}]")

        for y_id in range(obj.y_var_num):
            veh, task = obj.y_id2sub(y_id)
            var_name = f"yr[{obj.to_string(0, veh)}," \
                       f"{obj.to_string(2, task)}]"
            obj.yr_var[y_id].setAttr("Obj", 0.0)
            obj.yr_var[y_id].setAttr("VarName", var_name)

    # initialize z variable
    for z_id in range(obj.z_var_num):
        var_name = f"z[{obj.to_string(2, z_id)}]"
        obj.z_var[z_id].setAttr("Obj", 0.0)
        obj.z_var[z_id].setAttr("VarName", var_name)

    # initialize q variable
    for q_id in range(obj.q_var_num):
        var_name = f"q[{obj.to_string(1, q_id)}]"
        obj.q_var[q_id].setAttr("Obj", 0) # obj.param.time_penalty
        obj.q_var[q_id].setAttr("VarName", var_name)
        obj.q_var[q_id].setAttr("LB", 0.0)
        obj.q_var[q_id].setAttr("UB", obj.param.max_time)


############################################# Common Model #############################################
def form_common_model_impl(obj) -> None:
    """Build common model constraints (adapted for multi-depot)"""
    # set the objective is to minimize
    obj.model.setAttr("ModelSense", gp.GRB.MINIMIZE)

    # ---------------------------
    # Flow Constraint: incoming edges == outgoing edges (for task nodes)
    # The number of robots going to and leaving a task node is equal
    # ---------------------------
    for k in range(obj.veh_type_num):
        for i in range(obj.task_num):
            node_id = obj.graph[k].node2id(i)
            if node_id < 0:
                continue
                
            constr_expr = gp.LinExpr(0)
            # sum all incoming edges to node i
            for edgeId in obj.graph[k].node(node_id).in_edge_ids:
                xId = obj.sub2x_id_from_edge(k, edgeId)
                constr_expr += obj.x_var[xId]
            # sum all outgoing edges from node i
            for edgeId in obj.graph[k].node(node_id).out_edge_ids:
                xId = obj.sub2x_id_from_edge(k, edgeId)
                constr_expr -= obj.x_var[xId]
            constr_name = f"FlowInOut[{obj.to_string(0, k)},{obj.to_string(1, i)}]"
            obj.model.addConstr(constr_expr == 0, name=constr_name)

    # ---------------------------
    # Flow Constraint: incoming edges to task <= 1 (or <= vehNumPerType for continuous)
    # The number of robots going to a task node does not exceed the total number of robots of that type
    # ---------------------------
    for k in range(obj.veh_type_num):
        for i in range(obj.task_num):
            node_id = obj.graph[k].node2id(i)
            if node_id < 0:
                continue
                
            constr_expr = gp.LinExpr(0)
            for edgeId in obj.graph[k].node(node_id).in_edge_ids:
                xId = obj.sub2x_id_from_edge(k, edgeId)
                constr_expr += obj.x_var[xId]
            constr_name = f"FlowLess[{obj.to_string(0, k)},{obj.to_string(1, i)}]"
            if obj.flag_continuous:
                obj.model.addConstr(constr_expr <= float(obj.param.veh_num_per_type[k]), name=constr_name)
            else:
                obj.model.addConstr(constr_expr <= 1, name=constr_name)

    # ---------------------------
    # Flow Constraint: Each robot instance can be dispatched at most once
    # Each robot can only go in one direction to perform tasks, cannot split itself
    # ---------------------------
    for k in range(obj.veh_type_num):
        num_robots = obj.param.veh_num_per_type[k]
        
        for robot_idx in range(num_robots):
            # Dedicated start node for each robot instance
            start_node = obj.task_num + robot_idx
            start_node_id = obj.graph[k].node2id(start_node)
            
            if start_node_id < 0:
                continue
            
            constr_expr = gp.LinExpr(0)
            for edgeId in obj.graph[k].node(start_node_id).out_edge_ids:
                xId = obj.sub2x_id_from_edge(k, edgeId)
                constr_expr += obj.x_var[xId]
            
            # Each robot is dispatched at most once
            constr_name = f"FlowFromRobot[{obj.to_string(0, k)},{robot_idx}]"
            if obj.flag_continuous:
                obj.model.addConstr(constr_expr <= 1.0, name=constr_name)
            else:
                obj.model.addConstr(constr_expr <= 1, name=constr_name)

    # ---------------------------
    # Variable Relationship Constraint: Incoming edges == y
    # The number of robots going to a task node equals the number of robots performing the task at that node
    # ---------------------------
    for k in range(obj.veh_type_num):
        for i in range(obj.task_num):
            node_id = obj.graph[k].node2id(i)
            if node_id < 0:
                continue
                
            constr_expr = gp.LinExpr(0)
            for edgeId in obj.graph[k].node(node_id).in_edge_ids:
                xId = obj.sub2x_id_from_edge(k, edgeId)
                constr_expr += obj.x_var[xId]
            yId = obj.sub2y_id(k, i)
            constr_expr -= obj.y_var[yId]
            constr_name = f"RelationXY[{obj.to_string(0, k)},{obj.to_string(1, i)}]"
            obj.model.addConstr(constr_expr == 0, name=constr_name)

    # ---------------------------
    # Start-End Flow Matching Constraint
    # Path closure constraint, robots return to their starting point, which is more reasonable
    # ---------------------------
    for k in range(obj.veh_type_num):
        num_robots = obj.param.veh_num_per_type[k]
        
        for robot_idx in range(num_robots):
            # Get the dedicated start and end nodes for current robot instance
            start_node = obj.task_num + robot_idx
            end_node = obj.task_num + num_robots + robot_idx

            # 1. Calculate total flow out from start node
            out_flow_from_start = gp.LinExpr(0)
            # Check if node exists, just in case
            if start_node < obj.graph[k].node_num():
                node_id = obj.graph[k].node2id(start_node)
                for edgeId in obj.graph[k].node(node_id).out_edge_ids:
                    xId = obj.sub2x_id_from_edge(k, edgeId)
                    out_flow_from_start += obj.x_var[xId]

            # 2. Calculate total flow into end node
            in_flow_to_end = gp.LinExpr(0)
            # Check if node exists
            if end_node < obj.graph[k].node_num():
                node_id = obj.graph[k].node2id(end_node)
                for edgeId in obj.graph[k].node(node_id).in_edge_ids:
                    xId = obj.sub2x_id_from_edge(k, edgeId)
                    in_flow_to_end += obj.x_var[xId]

            # 3. Add constraint: flow out == flow in
            constr_name = f"StartEndMatch[{k},{robot_idx}]"
            obj.model.addConstr(out_flow_from_start == in_flow_to_end, name=constr_name)

    # ---------------------------
    # Time Constraint: Edge time
    # Relative temporal order constraints
    # ---------------------------
    for k in range(obj.veh_type_num):
        # Get the number of instances of current robot type to determine terminal nodes
        num_robots_of_this_type = obj.param.veh_num_per_type[k]
        task_num = obj.task_num
        
        for edgeId in range(obj.graph[k].edge_num()):
            an_edge = obj.graph[k].edge(edgeId)
            i, j = an_edge.edge.node1, an_edge.edge.node2
            xId = obj.sub2x_id_from_edge(k, edgeId)

            # Get binary switch variable
            if obj.flag_continuous:
                temp_binary_var = obj.xr_var[xId]
            else:
                temp_binary_var = obj.x_var[xId]
            
            # Operation time at node i (only calculated for task nodes)
            node_time_cost = obj.graph[k].node(i).time_cost if i < task_num else 0
            
            # Start time at node i (0 if it's a start node)
            time_at_i = obj.q_var[i] if i < task_num else 0

            # Determine node j's type and apply different constraints
            is_terminal_node = (j >= task_num + num_robots_of_this_type)

            if is_terminal_node:
                # Standard Big-M formulation: time_at_i + node_time_cost + an_edge.time_cost - obj.param.max_time <= M * (1-x)
                constr_expr_deadline = (time_at_i + node_time_cost + an_edge.time_cost - obj.param.max_time - obj.param.large_time * (1 - temp_binary_var))
                constr_name = (f"TimeDeadline[{k},{i},{j}]")
                obj.model.addConstr(constr_expr_deadline <= 0, name=constr_name)
            
            else:
                time_at_j = obj.q_var[j] if j < task_num else 0

                # time_at_i + node_time_cost + an_edge.time_cost <= time_at_j
                constr_expr_precedence = (time_at_i - time_at_j + node_time_cost + an_edge.time_cost - obj.param.large_time * (1 - temp_binary_var))
                constr_name = (f"TimePrecedence[{k},{i},{j}]")
                obj.model.addConstr(constr_expr_precedence <= 0, name=constr_name)

    # ---------------------------
    # Precedence Constraint
    # Mandatory ordering constraints between tasks
    # ---------------------------
    for i in range(obj.task_num):
        for j in range(obj.task_num):
            if i == j:
                continue
            if obj.param.dependency_matrix[i][j] == 1:  # task i must be executed before task j
                constr_expr = gp.LinExpr(0)
                constr_expr += obj.q_var[i]
                constr_expr -= obj.q_var[j]
                
                # Get time cost for task i
                time_cost_i = obj.param.task_param[i].time_cost

                constr_name = f"TaskPrecedence[{obj.to_string(1, i)},{obj.to_string(1, j)}]"
                obj.model.addConstr(constr_expr <= -time_cost_i, name=constr_name)

    # ---------------------------
    # Task Complete Constraint
    # ---------------------------
    if obj.param.flag_task_complete:
        for i in range(obj.task_num):
            constr_expr = obj.z_var[i]
            constr_name = f"TaskComplete[{obj.to_string(2, i)}]"
            obj.model.addConstr(constr_expr == 1, name=constr_name)
    else:
        for i in range(obj.task_num):
            obj.z_var[i].setAttr("Obj", -obj.param.task_complete_reward)

    # ---------------------------
    # Task Requirements Constraint 
    # ---------------------------
    # Completing a task means: the value of each alpha_var must be greater than the required capability value for that requirement
    alpha_var_map = {} # Use a map to easily reference alpha_vars later
    for i in range(obj.task_num):
        required_caps = []
        for andId in range(len(obj.param.task_param[i].req_fcn)):
            constr_expr = obj.z_var[i]
            for orId in range(len(obj.param.task_param[i].req_fcn[andId])):
                aReq = obj.param.task_param[i].req_fcn[andId][orId]
                required_caps.append(aReq.cap_id)
                
                # Create alpha_var (represents total capability supplied for this requirement)
                varLB = 0.0
                varUB = obj.sum_cap[aReq.cap_id]
                varObj = 0.0
                commonName = f"{obj.to_string(2, i)},{obj.to_string('AND', andId)},{obj.to_string('OR', orId)}]"
                varName = f"alpha[{commonName}"
                alpha_var = obj.model.addVar(lb=varLB, ub=varUB, obj=varObj, vtype=gp.GRB.CONTINUOUS, name=varName)
                obj.alpha_var.append(alpha_var)
                alpha_var_map[(i, andId, orId)] = alpha_var # Store for later reference

                # Create w_var (binary, indicates if this requirement is met)
                varLB = 0.0
                varUB = 1.0
                varObj = 0.0
                varName = f"w[{commonName}"
                w_var = obj.model.addVar(lb=varLB, ub=varUB, obj=varObj, vtype=gp.GRB.BINARY, name=varName)
                obj.w_var.append(w_var)
                constr_expr -= w_var

                # Link w_var to alpha_var
                wVarExpress = w_var
                tempLargeCap = obj.sum_cap[aReq.cap_id] - aReq.cap_req + 1.0
                if tempLargeCap < 1.0: tempLargeCap = 1.0

                constr1 = aReq.cap_req * wVarExpress - alpha_var
                obj.model.addConstr(constr1 <= 0, name=f"TaskReqOrL[{commonName}")

                constr2 = -tempLargeCap * wVarExpress + alpha_var - aReq.cap_req + 1
                obj.model.addConstr(constr2 <= 0, name=f"TaskReqOrG[{commonName}")

            # Link w_vars to z_var for the "AND" condition
            constr_name3 = f"TaskReqAnd[{obj.to_string(2, i)},{obj.to_string('AND', andId)}]"
            obj.model.addConstr(constr_expr <= 0, name=constr_name3)
        
        # "not use irrelevant" vehicle constraint
        if obj.param.flag_not_use_unralavant:
            for k in range(obj.veh_type_num):
                flag_unrelavant = True
                for cap_id in required_caps:
                    if obj.param.veh_param[k].cap_vector[cap_id] > 0.1:
                        flag_unrelavant = False
                        break
                if flag_unrelavant:
                    yId = obj.sub2y_id(k, i)
                    constr_name4 = f"TaskReqUnrelavant[{obj.to_string(2, i)},{obj.to_string(0, k)}]"
                    obj.model.addConstr(obj.y_var[yId] == 0, name=constr_name4)

    # Step 2: Define alpha_var constraints and add new shared capability constraints.
    # alpha_var may NOT be the sum of contributions from all robot types for that requirement.
    # If a requirement is assigned to a specific robot i, then ONLY robots of the same type as robot i
    # are allowed to contribute capability to the alpha_var associated with that requirement.
    for key, alpha_var in alpha_var_map.items():
        i, andId, orId = key
        aReq = obj.param.task_param[i].req_fcn[andId][orId]
        cap_id = aReq.cap_id
        alpha_contribution_sum = gp.LinExpr(0) # This will sum up contributions from all types
        for k in range(obj.veh_type_num):
            veh_cap_value = obj.param.veh_param[k].cap_vector[cap_id]
            if veh_cap_value < 1e-5:
                continue # This type cannot contribute

            # Create a new variable representing the contribution of *only* type k
            contribution_var = obj.model.addVar(vtype=gp.GRB.CONTINUOUS, name=f"contrib[{k},{i},{andId},{orId}]")
            alpha_contribution_sum += contribution_var
            
            # Link this specific contribution to the main y/yr variables
            yId = obj.sub2y_id(k, i)
            if obj.param.cap_type[cap_id] == 1: # Non-cumulative capability
                temp_binary_y_var = obj.yr_var[yId] if obj.flag_continuous else obj.y_var[yId]
                obj.model.addConstr(contribution_var == veh_cap_value * temp_binary_y_var, 
                                    name=f"LinkContrib_NonCumu[{k},{i},{andId},{orId}]")
            else: # Cumulative capability
                obj.model.addConstr(contribution_var == veh_cap_value * obj.y_var[yId], 
                                    name=f"LinkContrib_Cumu[{k},{i},{andId},{orId}]")

            # Apply shared capability constraint
            if obj.param.shared_capability_groups:
                current_req_id = [i, andId, orId]
                num_groups = len(obj.param.shared_capability_groups)
                for g in range(num_groups):
                    if current_req_id in obj.param.shared_capability_groups[g]:
                        # Using Big-M: contribution_var <= M * B_var[k, g]
                        M = obj.sum_cap[cap_id] 
                        obj.model.addConstr(contribution_var <= M * obj.B_var[k, g],
                                            name=f"ForceSingleTypeContrib[{k},{g},{i},{andId},{orId}]")

        obj.model.addConstr(alpha_var == alpha_contribution_sum, name=f"AlphaRedefine[{i},{andId},{orId}]")
    
    # This prevents multiple robots of the SAME type from collaborating on a task that must be done by a single robot.
    # Further restrict that the contribution to alpha_var comes from a SINGLE robot
    if obj.param.shared_capability_groups:
        for g in range(len(obj.param.shared_capability_groups)):
            group = obj.param.shared_capability_groups[g]
            for req in group:
                i, andId, orId = req
                aReq = obj.param.task_param[i].req_fcn[andId][orId]
                cap_id = aReq.cap_id
                
                # Get the corresponding alpha_var for this requirement
                alpha_var_for_req = alpha_var_map.get((i, andId, orId))
                if alpha_var_for_req is None: continue

                # The maximum possible contribution is the sum of (single robot capacity * B_var for that type)
                # Since only one B_var[k,g] can be 1, this expression will pick the capacity of the chosen type.
                max_single_robot_contribution = gp.LinExpr(0)
                for k in range(obj.veh_type_num):
                    single_robot_cap = obj.param.veh_param[k].cap_vector[cap_id]
                    max_single_robot_contribution += single_robot_cap * obj.B_var[k, g]
                
                # Add the final, decisive constraint:
                # The total capability supplied (alpha_var) cannot exceed what a single robot of the chosen type can provide.
                obj.model.addConstr(alpha_var_for_req <= max_single_robot_contribution, name=f"SingleRobotCapLimit_g{g}_t{i}_a{andId}_o{orId}")

    # Step 3: Add path connectivity constraints for the shared capability groups
    # This ensures that the tasks within a group are part of a single, contiguous tour in the aggregate flow model.
    # Ensure that the constraint in Step 2 enforces that it's the SAME robot,
    # traveling from one start depot and returning to one end depot
    if obj.param.shared_capability_groups:
        num_groups = len(obj.param.shared_capability_groups)
        for g in range(num_groups):
            tasks_in_group = list(set([req[0] for req in obj.param.shared_capability_groups[g]]))
            if not tasks_in_group:
                continue
            
            # This new flow X_group_var must be a sub-flow of the main x_var
            for x_id in range(obj.x_var_num):
                veh_k, edge_param = obj.x_id2edge(x_id)
                if edge_param is None:
                    continue # Skip dummy edges

                # The group flow can only exist on edges of the type chosen for this group (B_var[k,g]=1)
                obj.model.addConstr(obj.X_group_var[g, x_id] <= obj.param.veh_num_per_type[veh_k] * obj.B_var[veh_k, g],
                                    name=f"Xgroup_Subflow_Type[{g},{x_id}]")
                # The group flow cannot exceed the main flow on any edge
                obj.model.addConstr(obj.X_group_var[g, x_id] <= obj.x_var[x_id], 
                                    name=f"Xgroup_Subflow_Capacity[{g},{x_id}]")

            # Flow conservation for the special X_group flow at INTERNAL task nodes
            for i in range(obj.task_num):
                in_flow_g = gp.LinExpr(0)
                out_flow_g = gp.LinExpr(0)
                for k in range(obj.veh_type_num):
                    node_id = obj.graph[k].node2id(i)
                    if node_id < 0: continue
                    for edgeId in obj.graph[k].node(node_id).in_edge_ids:
                        xId = obj.sub2x_id_from_edge(k, edgeId)
                        in_flow_g += obj.X_group_var[g, xId]
                    for edgeId in obj.graph[k].node(node_id).out_edge_ids:
                        xId = obj.sub2x_id_from_edge(k, edgeId)
                        out_flow_g += obj.X_group_var[g, xId]

                if i in tasks_in_group:
                    obj.model.addConstr(in_flow_g == out_flow_g, name=f"Xgroup_FlowCons_Internal[{g},{i}]")
                    obj.model.addConstr(in_flow_g == 1, name=f"Xgroup_Visit[{g},{i}]")
                else:
                    obj.model.addConstr(in_flow_g == 0, name=f"Xgroup_FlowCons_External[{g},{i}]")

            # This ensures the flow originates from exactly one start depot and terminates at exactly one end depot.
            total_source_flow_g = gp.LinExpr(0)
            total_sink_flow_g = gp.LinExpr(0)
            
            for k in range(obj.veh_type_num):
                num_robots_of_type = obj.param.veh_num_per_type[k]
                for r_idx in range(num_robots_of_type):
                    start_node_id = obj.task_num + r_idx
                    end_node_id = obj.task_num + num_robots_of_type + r_idx

                    # Sum outflow from this robot's start node
                    if start_node_id < obj.graph[k].node_num():
                        for edgeId in obj.graph[k].node(start_node_id).out_edge_ids:
                            xId = obj.sub2x_id_from_edge(k, edgeId)
                            total_source_flow_g += obj.X_group_var[g, xId]
                    
                    # Sum inflow to this robot's end node
                    if end_node_id < obj.graph[k].node_num():
                        for edgeId in obj.graph[k].node(end_node_id).in_edge_ids:
                            xId = obj.sub2x_id_from_edge(k, edgeId)
                            total_sink_flow_g += obj.X_group_var[g, xId]

            # The total special flow originating from ALL start depots must be exactly 1.
            obj.model.addConstr(total_source_flow_g == 1, name=f"Xgroup_SingleSource[{g}]")
            # The total special flow terminating at ALL end depots must be exactly 1.
            obj.model.addConstr(total_sink_flow_g == 1, name=f"Xgroup_SingleSink[{g}]")

    # Step 4: Add the final constraint for the B_var variables themselves
    if obj.param.shared_capability_groups:
        num_groups = len(obj.param.shared_capability_groups)
        for g in range(num_groups):
            # For each group, exactly one vehicle type must be chosen to handle it.
            obj.model.addConstr(gp.quicksum(obj.B_var[k, g] for k in range(obj.veh_type_num)) == 1, name=f"SharedCap_SelectOneType[{g}]")

    # # ---------------------------
    # # Task Requirements Constraint
    # # ---------------------------
    # for i in range(obj.task_num):
    #     required_caps = []  # record the required capabilities
    #     # for every "and" condition (required capability)
    #     for andId in range(len(obj.param.task_param[i].req_fcn)):
    #         constr_expr = obj.z_var[i]
    #         # for every "or" condition (alternative capability)
    #         for orId in range(len(obj.param.task_param[i].req_fcn[andId])):
    #             aReq = obj.param.task_param[i].req_fcn[andId][orId]
    #             required_caps.append(aReq.cap_id)
    #             # add alpha variable
    #             varLB = 0.0
    #             varUB = obj.sum_cap[aReq.cap_id]
    #             varObj = 0.0
    #             commonName = f"{obj.to_string(2, i)},{obj.to_string('AND', andId)},{obj.to_string('OR', orId)}]"
    #             varName = f"alpha[{commonName}"
    #             alpha_var = obj.model.addVar(lb=varLB, ub=varUB, obj=varObj, vtype=gp.GRB.CONTINUOUS, name=varName)
    #             obj.alpha_var.append(alpha_var)

    #             # Task Requirement Constraint: alpha definition
    #             if obj.param.cap_type[aReq.cap_id] == 1:
    #                 # non-cumulative capability
    #                 constr0 = alpha_var
    #                 for k in range(obj.veh_type_num):
    #                     yId = obj.sub2y_id(k, i)
    #                     if obj.flag_continuous:
    #                         temp_binary_y_var = obj.yr_var[yId]
    #                     else:
    #                         temp_binary_y_var = obj.y_var[yId]
    #                     constr_name_alpha = f"TaskReqAlphaNonCumu[{commonName}"
    #                     obj.model.addConstr(alpha_var <= varUB * (1 - temp_binary_y_var) +
    #                                           obj.param.veh_param[k].cap_vector[aReq.cap_id] * temp_binary_y_var,
    #                                           name=constr_name_alpha)
    #                     if obj.param.veh_param[k].cap_vector[aReq.cap_id] > 0.1:
    #                         constr0 -= obj.param.veh_param[k].cap_vector[aReq.cap_id] * temp_binary_y_var
    #                 constr_name0 = f"TaskReqAlphaDef[{commonName}"
    #                 obj.model.addConstr(constr0 <= 0, name=constr_name0)
    #             else:
    #                 # cumulative capability
    #                 constr0 = alpha_var
    #                 for k in range(obj.veh_type_num):
    #                     yId = obj.sub2y_id(k, i)
    #                     if obj.param.veh_param[k].cap_vector[aReq.cap_id] > 0.1:
    #                         constr0 -= obj.param.veh_param[k].cap_vector[aReq.cap_id] * obj.y_var[yId]
    #                 constr_name0 = f"TaskReqAlphaDef[{commonName}"
    #                 obj.model.addConstr(constr0 == 0, name=constr_name0)

    #             # add w variable
    #             varLB = 0.0
    #             varUB = 1.0
    #             varObj = 0.0
    #             varName = f"w[{commonName}"
    #             w_var = obj.model.addVar(lb=varLB, ub=varUB, obj=varObj, vtype=gp.GRB.BINARY, name=varName)
    #             obj.w_var.append(w_var)
    #             constr_expr -= w_var

    #             # Task Requirement Constraint: the relationship between w and alpha
    #             wVarExpress = w_var
    #             tempLargeCap = obj.sum_cap[aReq.cap_id] - aReq.cap_req + 1.0
    #             if tempLargeCap < 1.0:
    #                 tempLargeCap = 1.0

    #             constr1 = aReq.cap_req * wVarExpress - alpha_var
    #             constr_name1 = f"TaskReqOrL[{commonName}"
    #             obj.model.addConstr(constr1 <= 0, name=constr_name1)

    #             constr2 = -tempLargeCap * wVarExpress + alpha_var - aReq.cap_req + 1
    #             constr_name2 = f"TaskReqOrG[{commonName}"
    #             obj.model.addConstr(constr2 <= 0, name=constr_name2)

    #         # Task Requirement Constraint: "and" condition
    #         constr_name3 = f"TaskReqAnd[{obj.to_string(2, i)},{obj.to_string('AND', andId)}]"
    #         obj.model.addConstr(constr_expr <= 0, name=constr_name3)

    #     # if the task is not relevant to the vehicle, set y = 0
    #     if obj.param.flag_not_use_unralavant:
    #         for k in range(obj.veh_type_num):
    #             flag_unrelavant = True
    #             for cap_id in required_caps:
    #                 if obj.param.veh_param[k].cap_vector[cap_id] > 0.1:
    #                     flag_unrelavant = False
    #                     break
    #             if flag_unrelavant:
    #                 yId = obj.sub2y_id(k, i)
    #                 constr_name4 = f"TaskReqUnrelavant[{obj.to_string(2, i)},{obj.to_string(0, k)}]"
    #                 obj.model.addConstr(obj.y_var[yId] == 0, name=constr_name4)


def form_deterministic_energy_impl(obj) -> None:
    """Energy constraint: limit total energy consumption by robot type"""
    for k in range(obj.veh_type_num):
        constr_expr = gp.LinExpr(0)

        # Accumulate energy consumption for all edges in type k graph
        for edge_id in range(obj.graph[k].edge_num()):
            an_edge = obj.graph[k].edge(edge_id)
            x_id = obj.sub2x_id_from_edge(k, edge_id)

            # Total energy = sum(edge energy cost * flow through that edge)
            constr_expr += an_edge.eng_cost * obj.x_var[x_id]

        # Total energy consumption for all robots of this type <= number of robots of this type * single robot energy limit
        total_eng_max = obj.param.veh_param[k].eng_max * obj.param.veh_num_per_type[k]

        constr_name = f"TotalEnergyType[{obj.to_string(0, k)}]"
        obj.model.addConstr(constr_expr <= total_eng_max, name=constr_name)


############################################# Continuous Model #############################################
def form_continuous_model_impl(obj) -> None:
    # adding the constraints between x and xr of each egde for each kind of vehicle
    for k in range(obj.veh_type_num):
        for edge_id in range(obj.graph[k].edge_num()):
            x_id = obj.sub2x_id_from_edge(k, edge_id)
            an_edge = obj.graph[k].edge(edge_id)
            common_name = (f"{obj.to_string(0, k)},"
                           f"{obj.to_string(1, an_edge.edge.node1)},"
                           f"{obj.to_string(1, an_edge.edge.node2)},"
                           f"{obj.to_string(3, an_edge.edge.type)}]")
            constr_name = "X_Xr1[" + common_name
            # x_var[x_id] >= xr_var[x_id]
            obj.model.addConstr(obj.x_var[x_id] >= obj.xr_var[x_id], name=constr_name)

            constr_name = "X_Xr2[" + common_name
            # x_var[x_id] <= xr_var[x_id] * veh_num_per_type[k]
            obj.model.addConstr(obj.x_var[x_id] <= obj.xr_var[x_id] * float(obj.param.veh_num_per_type[k]), name=constr_name)

    # adding the constraints between y and yr of each task for each kind of vehicle
    for k in range(obj.veh_type_num):
        for i in range(obj.task_num):
            y_id = obj.sub2y_id(k, i)
            common_name = f"{obj.to_string(0, k)},{obj.to_string(1, i)}]"
            constr_name = "Y_Yr1[" + common_name
            # y_var[y_id] >= yr_var[y_id]
            obj.model.addConstr(obj.y_var[y_id] >= obj.yr_var[y_id], name=constr_name)

            constr_name = "Y_Yr2[" + common_name
            # y_var[y_id] <= yr_var[y_id] * veh_num_per_type[k]
            obj.model.addConstr(obj.y_var[y_id] <= obj.yr_var[y_id] * float(obj.param.veh_num_per_type[k]), name=constr_name)

def form_continuous_det_energy_impl(obj) -> None:
    """
    Add total energy constraints for continuous model. To satisfy individual energy constraints, 
    a complete individual model still needs to be considered.
    """
    # Add a total energy constraint for each type of robot
    for k in range(obj.veh_type_num):
        constr_expr = gp.LinExpr(0)
        
        # Accumulate energy consumption for all edges in type k graph
        # Total energy = sum(edge energy cost * flow through that edge)
        for edge_id in range(obj.graph[k].edge_num()):
            an_edge = obj.graph[k].edge(edge_id)
            x_id = obj.sub2x_id_from_edge(k, edge_id)
            
            # Use continuous variable x_var to calculate energy consumption
            constr_expr += an_edge.eng_cost * obj.x_var[x_id]
        
        # Total energy consumption for all robots of this type <= number of robots of this type * single robot energy limit
        total_eng_max = obj.param.veh_param[k].eng_max * obj.param.veh_num_per_type[k]
        
        constr_name = f"TotalEnergyTypeContinuous[{obj.to_string(0, k)}]"
        obj.model.addConstr(constr_expr <= total_eng_max, name=constr_name)