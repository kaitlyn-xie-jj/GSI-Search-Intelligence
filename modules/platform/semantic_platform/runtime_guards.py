from __future__ import annotations
from typing import Dict, Any, Tuple, Optional, List, Callable
from modules.platform.semantic_platform.utils.situation_utils import (
    is_high_priority, 
    get_relevant_area_node,
    get_district_node,
    get_path_from_context,
    get_robot_and_dest_positions,           
    iter_areas_crossed_by_robot_to_dest,
)
from modules.utils.geom_utils import (
    point_to_segment_distance,
    point_in_area_geometry,
)
from modules.utils.location_utils import (
    get_entity_position,
    extract_object_position,
    shape_center_point,
)

Reason = str 

class GuardRegistry:
    """
    Manages reusable guard predicates for both pre-check and runtime use.
    Each guard returns (ok: bool, reason_code: Optional[str], detail: Optional[str]).
    """
    def __init__(self, scene_graph):
        self.ctx = scene_graph
        self._high_priority_types = {"fire", "hazmat"}
        self._guards: Dict[str, Callable[[Dict[str, Any], Dict[str, Any]], Tuple[bool, Optional[Reason], Optional[str]]]] = {
            "path_traversable": self._guard_path_traversable,
            "path_avoids_restricted": self._guard_path_avoids_restricted,
            "area_perception_safe": self._guard_area_perception_safe,
            "path_avoids_congestion": self._guard_path_avoids_congestion,
            "target_line_of_sight_clear": self._guard_target_line_of_sight_clear,
            "target_present_at_expected": self._guard_target_present_at_expected,
            "target_exists": self._guard_target_exists,
            "carrier_has_object": self._guard_carrier_has_object,
            "target_status_discovered": self._guard_target_status_discovered,
            "check_for_new_target": self._guard_check_for_new_target,
            "robot_fault": self._guard_robot_fault,
            "battery_above": self._guard_battery_above,
            "comm_link": self._guard_comm_link,
            "link_intact": self._guard_link_intact,
        }

    def eval(self, name: str, args: Dict[str, Any], context: Dict[str, Any]) -> Tuple[bool, Optional[Reason], Optional[str]]:
        fn = self._guards.get(name)
        if not fn:
            return True, None, None
        return fn(args, context)

    # ----------------- Guard Implementations -----------------

    def _guard_check_for_new_target(self, args: Dict[str, Any], context: Dict[str, Any]):
        """
        Implement the pre-check logic for the 'search' skill guard.
        """
        # 1. Check whether context contains planner expectations
        area_has_hp_target = context.get("area_has_hp_target", False)
        
        if area_has_hp_target:
            # 2. The planner expects a high-priority target to be found in this area.
            # Get planner-provided information from context
            hp_object_id = context.get("hp_object_id")
            hp_target_label = context.get("hp_target_label")
            
            # Try to get more information from the graph
            hp_node = None
            hp_target_type = None
            hp_target_location_label = None
            if hp_object_id and self.ctx:
                hp_node = self.ctx.get_node_by_id(hp_object_id)
            if hp_node:
                props = hp_node.get("properties", {})
                hp_target_label = hp_target_label or props.get("label")
                hp_target_type = props.get("type")
                hp_target_location_label = props.get("location", {}).get("label")

            details = {
                "message": "Planner-expected high priority target discovered.",
                "hp_object_id": hp_object_id,
                "hp_target_label": hp_target_label,
                "hp_target_type": hp_target_type,
                "hp_target_location_label": hp_target_location_label,
                "area_id": args.get("area_id") or context.get("area"),
            }
            # Return False and reason_code; upper layer (evaluate_precheck) will capture it as "info"
            return False, "NEW_HIGH_PRIORITY_TARGET_DISCOVERED", details
            
        # 3. Planner does not expect a high-priority target, so the guard passes
        return True, None, None
    
    def _guard_path_traversable(self, args: Dict[str, Any], context: Dict[str, Any]):
        """
        Check whether every segment on the "critical path" from robot to destination has a traversable edge.
        Applies only to ground robots (UGV, Quadruped, Humanoid); aircraft are not limited by ground paths.
        """
        # Aircraft are not limited by ground path traversability
        robot_type = context.get("robot_type")
        if robot_type in ("UAV", "FW_UAV"):
            return True, None, None

        detail = {
            "dest_id": args.get("dest_id") or context.get("destination_id") or context.get("object_id"),
            "dest_label": args.get("dest_label") or context.get("destination_label") or context.get("target_location_label"),
        }
        path = get_path_from_context(self.ctx, context)
        if not path or len(path) < 2:
            return True, None, None  # Cannot determine path, allow

        for a, b in zip(path, path[1:]):
            has_tr = (
                self.ctx.has_edge_of_type(a, b, 'traversable') or
                self.ctx.has_edge_of_type(b, a, 'traversable')
            )
            if not has_tr:
                a_label = (self.ctx.get_node_by_id(a) or {}).get('properties', {}).get('label', a)
                b_label = (self.ctx.get_node_by_id(b) or {}).get('properties', {}).get('label', b)
                detail["message"] = f"segment not traversable: {a_label}->{b_label}"
                return False, "CRITICAL_PATH_BROKEN", detail

        return True, None, None

    def _guard_path_avoids_restricted(self, args: Dict[str, Any], context: Dict[str, Any]):
        """
        Check whether the line from the current robot position to the destination crosses an area
        with status == 'restricted' that is not the robot's current area.

        If such an area exists:
        - Return False, reason="AREA_RESTRICTED".
        - Fill detail with area/destination information.
        """
        # Find all restricted areas crossed by the line, excluding the robot's current area
        crossed = iter_areas_crossed_by_robot_to_dest(
            self.ctx,
            context,
            only_restricted=True,
            exclude_current_area=True,
        )
        if not crossed:
            return True, None, None

        area_node = crossed[0]
        props = area_node.get("properties") or {}
        area_id = area_node.get("id")
        area_label = props.get("label", area_id)
        detail = {
            "restricted_area_id": area_id,
            "restricted_area_label": area_label,
            "dest_id": args.get("dest_id") or context.get("destination_id") or context.get("object_id"),
            "dest_label": args.get("dest_label") or context.get("destination_label") or context.get("target_location_label"),
        }

        # check whether destination lies inside this restricted area
        try:
            _, dest_pos = get_robot_and_dest_positions(self.ctx, context)
        except Exception:
            dest_pos = None

        geom = {}
        get_bf = getattr(self.ctx, "get_boundary_features", None)
        if callable(get_bf):
            boundary_map = get_bf()
            geom = boundary_map.get(props.get("label")) or {}
        if dest_pos is not None and geom and point_in_area_geometry(dest_pos, geom):
            detail["message"] = "Destination is inside the restricted area."

        return False, "AREA_RESTRICTED", detail

    def _guard_area_perception_safe(self, args: Dict[str, Any], context: Dict[str, Any]):
        """Unified environmental perception check (fog, darkness, wind)."""
        # Robot's current location (area / building / trans_facility)
        area_node = get_relevant_area_node(self.ctx, context)
        if area_node:
            props = (area_node.get("properties") or {})
            area_label = props.get("label", area_node.get("id"))
            skill = context.get("skill")

            # Fog/smoke: low visibility
            robot_type = context.get("robot_type")
            if robot_type in ("UAV", "FW_UAV") and props.get("visibility") == "low":
                return False, "AREA_LOW_VISIBILITY", {
                    "curr_area_id": area_node.get("id"),
                    "curr_area_label": area_label,
                    "skill": skill,
                }

            # Strong wind: only applies to UAV/FW_UAV
            if robot_type in ("UAV", "FW_UAV") and props.get("wind_condition") == "strong":
                return False, "AREA_STRONG_WIND", {
                    "curr_area_id": area_node.get("id"),
                    "curr_area_label": area_label,
                    "robot_type": robot_type,
                }

        # Whether the global district (cybertown) is dark
        district_node = get_district_node(self.ctx)
        if district_node:
            robot_type = context.get("robot_type")
            dprops = (district_node.get("properties") or {})
            if robot_type in ("UAV", "FW_UAV") and dprops.get("illumination") == "dark":
                label = dprops.get("label", district_node.get("id"))
                return False, "AREA_IS_DARK", {
                    "curr_area_id": district_node.get("id"),
                    "curr_area_label": label,
                    "skill": context.get("skill"),
                }

        return True, None, None

    def _guard_path_avoids_congestion(self, args: Dict[str, Any], context: Dict[str, Any]):
        """
        Temporary congestion check (vehicles / crowds):
        - Check intersection nodes and street segment nodes between intersections on the robot -> destination path.
        - Any node with congestion in {"vehicle","crowd"} triggers failure.
        Applies only to ground robots (UGV, Quadruped, Humanoid); aircraft are not affected by ground congestion.
        """
        # Aircraft are not affected by ground congestion
        robot_type = context.get("robot_type")
        if robot_type in ("UAV", "FW_UAV"):
            return True, None, None

        path = get_path_from_context(self.ctx, context)
        if not path:
            return True, None, None  # Cannot determine path, allow

        # Destination info
        dest_id = args.get("dest_id") or context.get("destination_id") or context.get("object_id")
        dest_label = args.get("dest_label") or context.get("destination_label") or context.get("target_location_label")
        base_detail = {
            "dest_id": dest_id,
            "dest_label": dest_label,
        }

        # Prebuild intersection -> {street_segment} mapping to find street segments between path intersections
        target_to_segments: Dict[str, set] = {}
        for e in self.ctx.get_edges_by_type("connects_to") or []:
            src = e.get("source")
            tgt = e.get("target")
            if src is None or tgt is None:
                continue
            tgt_str = str(tgt)
            s = target_to_segments.get(tgt_str)
            if s is None:
                s = set()
                target_to_segments[tgt_str] = s
            s.add(str(src))

        path_ids = [str(nid) for nid in path]

        # Check whether one node is congested and build the return value
        def _check_node(nid: str):
            node = self.ctx.get_node_by_id(nid)
            if not node:
                return None
            props = (node.get("properties") or {})
            congestion_type = props.get("congestion")
            if congestion_type not in ("vehicle", "crowd"):
                return None

            reason_code = "PATH_CONGESTED_VEHICLE" if congestion_type == "vehicle" else "PATH_CONGESTED_CROWD"
            detail = dict(base_detail)
            detail.update({
                "congestion_type": congestion_type,
                "congestion_node_id": nid,
                "congestion_label": props.get("label", nid),
            })
            return False, reason_code, detail

        # First check intersection nodes on the path
        for nid in path_ids:
            res = _check_node(nid)
            if res is not None:
                return res  # (False, reason_code, detail)

        # Then check street segment nodes between adjacent intersections
        for a, b in zip(path_ids, path_ids[1:]):
            seg_a = target_to_segments.get(a, set())
            seg_b = target_to_segments.get(b, set())
            for sid in (seg_a & seg_b):
                res = _check_node(sid)
                if res is not None:
                    return res

        return True, None, None
    
    def _guard_target_line_of_sight_clear(self, args: Dict[str, Any], context: Dict[str, Any]):
        """Obstruction check: whether a status=obstruction node exists on the robot->target line.
        Applies only to UAV and Quadruped, matching op.add_obstruction_near_target applicable_robots.
        """
        # Applies only to UAV and Quadruped
        robot_type = context.get("robot_type")
        if robot_type not in ("UAV", "Quadruped"):
            return True, None, None

        target_id = args.get("object_id") or context.get("object_id")
        if not target_id:
            return True, None, None

        robot_node = context.get("robot")
        target_node = context.get("target_node") or self.ctx.get_node_by_id(target_id)
        if not (robot_node and target_node):
            return True, None, None

        rpos = extract_object_position(robot_node) or get_entity_position(self.ctx, context.get("robot_id"))
        tpos = extract_object_position(target_node) or get_entity_position(self.ctx, target_id)
        if not rpos or not tpos:
            return True, None, None

        # Iterate in reverse: newer nodes are more likely to be dynamically injected obstructions
        nodes = getattr(self.ctx, "get_all_nodes", lambda: [])() or []
        for node in reversed(nodes):
            nid = node.get("id")
            if str(nid) in (str(context.get("robot_id")), str(target_id)):
                continue
            props = (node.get("properties") or {})
            if props.get("status") != "obstruction":
                continue

            cpos = extract_object_position(node) or shape_center_point(node.get("shape") or {})
            if not cpos:
                continue

            d = point_to_segment_distance(cpos, rpos, tpos)
            if d is None:
                continue
            # Threshold can be adjusted by map scale; use a relatively small distance here
            if d <= 20.0:
                return False, "TARGET_OBSTRUCTED", {
                    "target_id": target_id,
                    "target_label": context.get("target_label"),
                    "obstruction_id": nid,
                    "obstruction_label": props.get("label", nid),
                }

        return True, None, None

    def _guard_target_present_at_expected(self, args: Dict[str, Any], context: Dict[str, Any]):
        obj_id = args.get("object_id") or context.get("object_id")
        expected_loc = args.get("expected_loc") or context.get("target_location")
        if obj_id is None or expected_loc is None:
            return True, None, None
        node = self.ctx.get_node_by_id(obj_id)
        if not node:
            # If the node does not exist, target_exists should trigger TARGET_NOT_EXIST.
            # But if only this guard runs, it should still return a meaningful error.
            return False, "TARGET_NOT_EXIST", f"object {obj_id} not exists"
            
        loc_prop = node.get("properties", {}).get("location")
        if str(loc_prop) == str(expected_loc):
            return True, None, None
        edges = self.ctx.get_edges_by_source(obj_id) + self.ctx.get_edges_by_target(obj_id)
        for e in edges:
            if e.get("type") in ("stored_at", "located_at"):
                if str(e.get("target")) == str(expected_loc) or str(e.get("source")) == str(expected_loc):
                    return True, None, None

        # Return the new reason_code
        return False, "TARGET_LOCATION_MISMATCH", f"object {obj_id} not at expected {expected_loc}"
    
    def _guard_target_status_discovered(self, args: Dict[str, Any], context: Dict[str, Any]):
        obj_id = args.get("object_id") or context.get("object_id")
        if obj_id is None:
            return True, None, None
        node = self.ctx.get_node_by_id(obj_id)
        if not node:
            return True, None, None 
        status = (node.get("properties") or {}).get("status")
        if status == "undiscovered":
            return False, "TARGET_DISAPPEARED", "target exists but status=undiscovered"
        return True, None, None

    def _guard_link_intact(self, args: Dict[str, Any], context: Dict[str, Any]):
        src = args.get("source_id")
        rel = args.get("relation")
        tgt = args.get("target_id")
        if not (src and rel and tgt):
            return True, None, None

        # Try direct lookup first
        e = None
        if hasattr(self.ctx, "get_edge"):
            e = self.ctx.get_edge(src, tgt)

        # Fall back to scanning
        if not e:
            edges = self.ctx.get_edges_by_source(src) + self.ctx.get_edges_by_target(src)
            for cand in edges:
                if cand.get("type") == rel and (str(cand.get("source")) == str(src) and str(cand.get("target")) == str(tgt)):
                    e = cand
                    break

        if e and e.get("type") == rel:
            return True, None, None
        return False, args.get("reason_code", "link_broken"), f"missing edge {src}-{rel}->{tgt}"

    def _guard_battery_above(self, args: Dict[str, Any], context: Dict[str, Any]):
        """
        Check whether robot battery is above the threshold.
        """
        robot = context.get("robot")
        if not robot:
            return True, None, None
        threshold = float(args.get("min_percent", 20.0))
        props = (robot or {}).get("properties", {})
        if "battery_level" in props:
            try:
                val = float(props.get("battery_level", 100))
                if val < threshold:
                    return False, "ROBOT_BATTERY_LOW", f"battery={val}%< {threshold}%"
                return True, None, None
            except Exception:
                pass

        status = props.get("status")
        if status == "low_battery":
            return False, "ROBOT_BATTERY_LOW", "status=low_battery"
        return True, None, None

    def _guard_robot_fault(self, args: Dict[str, Any], context: Dict[str, Any]):
        status = ((context.get("robot") or {}).get("properties") or {}).get("status")
        if status in ("failed", "hardware_fault", "software_fault", "error"):
            return False, "ROBOT_FAULT", f"status={status}"
        return True, None, None
    
    def _guard_comm_link(self, args: Dict[str, Any], context: Dict[str, Any]):
        comm_status = ((context.get("robot") or {}).get("properties") or {}).get("comm")
        if comm_status in ("lost", "error", "jammed"):
            return False, "ROBOT_COMM_JAMMED", f"communication_status={comm_status}"
        return True, None, None
    
    def _guard_target_exists(self, args: Dict[str, Any], context: Dict[str, Any]):
        obj_id = args.get("object_id") or context.get("object_id")
        node = self.ctx.get_node_by_id(obj_id)
        if not node:
            return False, None, None
        return True, None, None
    
    def _guard_carrier_has_object(self, args: Dict[str, Any], context: Dict[str, Any]):
        carrier_id = args.get("carrier_id") or context.get("carrier_id")
        object_id = args.get("object_id") or context.get("object_id")
        if not (carrier_id and object_id):
            return True, None, None
        neighbors = self.ctx.get_neighbors_by_relation(carrier_id, "carrying") or []
        if object_id in neighbors:
            return True, None, None
        return False, "carrier_not_having_required_object", f"carrier {carrier_id} not carrying {object_id}"

class RuntimeGuardMonitor:
    """
    Runtime guard monitor: periodically evaluates guards (during in runtime/both).
    When a guard fails with severity=abort/soft_abort, returns the failure reason
    so the execution layer can decide whether to interrupt.
    """
    def __init__(self, registry: GuardRegistry, interval: float = 0.2):
        self.registry = registry
        self.interval = interval

    async def watch(self,
                    guards: List[Dict[str, Any]],
                    context: Dict[str, Any],
                    params: Dict[str, Any],
                    interrupt_flag_fn) -> Optional[Dict[str, Any]]:
        """
        Poll guards; when a guard fails, return {reason, guard_name, message, severity}.
        """
        import asyncio
        while not interrupt_flag_fn():
            for g in guards or []:
                during = (g.get("during") or "both").lower()
                if during not in ("runtime", "both"):
                    continue
                name = g.get("name")
                args = dict(g.get("args") or {})
                # Placeholder resolution (limited support)
                if args.get("dest_id") == "{object_id}":
                    args["dest_id"] = params.get("object_id") or context.get("object_id")
                if args.get("object_id") == "{object_id}":
                    args["object_id"] = params.get("object_id") or context.get("object_id")
                if args.get("expected_loc") == "{target_location}":
                    args["expected_loc"] = context.get("target_location")

                ok, reason, detail = self.registry.eval(name, args, context)
                if not ok:
                    return {
                        "reason": reason or "guard_failed",
                        "guard_name": name,
                        "message": g.get("message") or detail,
                        "severity": g.get("severity", "abort"),
                        "on_fail": g.get("on_fail")
                    }
            await asyncio.sleep(self.interval)
        return None
