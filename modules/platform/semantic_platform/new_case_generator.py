# -*- coding: utf-8 -*-
import asyncio
import logging
import random
import time
import copy
import re
from typing import Dict, Any, Optional, List, Tuple
from modules.platform.abstract_scene_graph import AbstractSceneGraph
from modules.utils.location_utils import (
    infer_nearest_location, 
    shape_center_point, 
    create_centered_shape,
    get_entity_position,
    extract_object_position,
    entities_same_location,
)
from modules.utils.geom_utils import (
    point_to_segment_distance,
    distance,
    area_centroid,
)
from modules.platform.semantic_platform.utils.situation_utils import (
    is_high_priority, 
    check_new_target_discovery_logic,
    iter_areas_crossed_by_robot_to_dest,
    get_district_node,
    get_path_from_context,
    get_relevant_area_node,
    suggest_new_pose,
)
from modules.config.entities.new_case_config import NEW_CASE_EVENT_TEMPLATES
from modules.utils.system.var_dump import dump_var
from modules.config.system_config import config

logger = logging.getLogger(__name__)

# ============================================================
# 1. Situation analysis and strategy manager
# ============================================================
class NewCaseStrategyManager:
    """
    Manages all new-case generation strategy implementations.
    This class analyzes the current situation (context, skill_info)
    and decides which graph operation to execute.
    """
    def __init__(self, scene_graph: AbstractSceneGraph):
        self.scene_graph = scene_graph
        
        # Cache common type mappings
        self._target_types_rule_A = {"person", "vehicle", "boat", "cargo", "assembly_component"}
        self._target_types_rule_B = {"fire", "hazmat", "equipment_failure", "vehicle", "boat"}

    def execute_strategy(self, strategy: str, params: Dict, 
                         skill_info: Dict, ctx: Dict) -> Optional[List[Dict]]:
        """
        Unified entry point for strategy execution.
        """
        try:
            # -----------------------------------------------
            # Environment-related strategies (extension interface)
            # -----------------------------------------------
            if strategy == "block_critical_path":
                return self.strategy_block_critical_path(params, skill_info, ctx)
            if strategy == "set_area_property":
                return self.strategy_set_area_property(params, skill_info, ctx)
            if strategy == "set_path_congestion":
                return self.strategy_set_path_congestion(params, skill_info, ctx)
            if strategy == "add_obstruction_node":
                return self.strategy_add_obstruction_node(params, skill_info, ctx)
                
            # -----------------------------------------------
            # Target state change strategies
            # -----------------------------------------------
            if strategy in ("move_target", "despawn_target"):
                return self.strategy_target_state_change(strategy, params, skill_info, ctx)
            
            if strategy in ("move_carrier", "despawn_carrier"):
                return self.strategy_carrier_state_change(strategy, params, skill_info, ctx)
                
            if strategy in ("move_surface_object", "despawn_surface_object"):
                return self.strategy_surface_state_change(strategy, params, skill_info, ctx)

            if strategy == "target_disappears": # Follow skill only
                return self.strategy_target_disappears(params, skill_info, ctx)
                
            # -----------------------------------------------
            # New information discovery strategies
            # -----------------------------------------------
            if strategy == "discover_new_high_priority_target":
                return self.strategy_discover_new_high_priority_target(params, skill_info, ctx)

            # -----------------------------------------------
            # Robot-related strategies
            # -----------------------------------------------
            if strategy == "degrade_robot_battery":
                return self.strategy_degrade_robot_battery(params, skill_info, ctx)
            if strategy == "set_robot_fault":
                return self.strategy_set_robot_fault(params, skill_info, ctx)
            if strategy == "jam_comm_link":
                return self.strategy_jam_comm_link(params, skill_info, ctx)

        except Exception as e:
            logger.error(f"Error executing strategy '{strategy}': {e}", exc_info=True)
            
        return None

    # --- Strategy implementation: target state ---

    def _precheck_same_location(self, ctx: Dict, *, which: str) -> bool:
        return entities_same_location(self.scene_graph, ctx, which=which, eps=100.0)

    def strategy_target_state_change(self, strategy: str, params: Dict, skill_info: Dict, ctx: Dict) -> Optional[List[Dict]]:
        if not self._precheck_same_location(ctx, which="target"):
            return None

        skill_name = (skill_info or {}).get('skill')
        target_id = ctx.get('object_id')
        target_node = ctx.get("target_node") or self.scene_graph.get_node_by_id(target_id)
        target_type = (ctx.get('object') or {}).get('type') or ((target_node or {}).get("properties") or {}).get("type")

        applies = False
        if skill_name in ("take_photo", "broadcast", "guide") and target_type in self._target_types_rule_A:
            applies = True
        elif skill_name == "place" and target_type in ("cargo", "assembly_component"):
            applies = True
        elif skill_name == "handle_hazard" and target_type in self._target_types_rule_B:
            applies = True
        if not applies:
            return None

        if strategy == "move_target":
            robot_node = ctx.get("robot")
            if not target_node or not robot_node:
                return None
            loc, shp = suggest_new_pose(self.scene_graph, target_node, robot_node) or (None, None)
            if not (loc or shp):
                return None
            props: Dict[str, Any] = {}
            if loc:
                props["location"] = loc
            op = {"action": "MODIFY_NODE_PROPERTIES", "node_id": target_id, "properties": props}
            if shp:
                op["shape"] = shp
            return [op]

        elif strategy == "despawn_target":
            return [{"action": "DELETE_NODE", "node_id": target_id}]

        return None

    def strategy_carrier_state_change(self, strategy: str, params: Dict, skill_info: Dict, ctx: Dict) -> Optional[List[Dict]]:
        if (skill_info or {}).get('skill') != 'place':
            return None

        carrier_id = ctx.get("carrier_id")
        if not carrier_id:
            return None

        if not self._precheck_same_location(ctx, which="carrier"):
            return None

        if strategy == "move_carrier":
            robot_node = ctx.get("robot")
            carrier_node = ctx.get("carrier_node") or self.scene_graph.get_node_by_id(carrier_id)
            if not carrier_node or not robot_node:
                return None
            loc, shp = suggest_new_pose(self.scene_graph, carrier_node, robot_node) or (None, None)
            if not (loc or shp):
                return None
            props: Dict[str, Any] = {}
            if loc:
                props["location"] = loc
            op = {"action": "MODIFY_NODE_PROPERTIES", "node_id": carrier_id, "properties": props}
            if shp:
                op["shape"] = shp
            return [op]

        elif strategy == "despawn_carrier":
            return [{"action": "DELETE_NODE", "node_id": carrier_id}]

        return None

    def strategy_surface_state_change(self, strategy: str, params: Dict, skill_info: Dict, ctx: Dict) -> Optional[List[Dict]]:
        if (skill_info or {}).get('skill') != 'place':
            return None

        surface_id = ctx.get("surface_id")
        if not surface_id:
            return None

        if not self._precheck_same_location(ctx, which="surface"):
            return None

        if strategy == "move_surface_object":
            robot_node = ctx.get("robot")
            surface_node = ctx.get("surface_node") or self.scene_graph.get_node_by_id(surface_id)
            if not surface_node or not robot_node:
                return None
            loc, shp = suggest_new_pose(self.scene_graph, surface_node, robot_node) or (None, None)
            if not (loc or shp):
                return None
            props: Dict[str, Any] = {}
            if loc:
                props["location"] = loc
            op = {"action": "MODIFY_NODE_PROPERTIES", "node_id": surface_id, "properties": props}
            if shp:
                op["shape"] = shp
            return [op]

        elif strategy == "despawn_surface_object":
            return [{"action": "DELETE_NODE", "node_id": surface_id}]

        return None

    def strategy_target_disappears(self, params: Dict, skill_info: Dict, ctx: Dict) -> Optional[List[Dict]]:
        """
        Make the target "disappear" (follow only):
        - If the target is already 'undiscovered', make no changes and return None.
        - Otherwise, always set the target to 'undiscovered'.
        - If the target is colocated with the robot, try to move it away (location/shape);
          if they are not colocated, leave it in place.
        """
        if (skill_info or {}).get('skill') != 'follow':
            return None

        target_id = ctx.get('object_id')
        if not target_id:
            return None

        target_node = ctx.get("target_node") or self.scene_graph.get_node_by_id(target_id)
        robot_node  = ctx.get("robot")
        if not target_node or not robot_node:
            return None

        tprops = (target_node.get("properties") or {})
        # Already undiscovered, so no need to disappear again
        if tprops.get("status") == "undiscovered":
            return None

        same_place = bool(self._precheck_same_location(ctx, which="target"))
        props_update: Dict[str, Any] = {"status": "undiscovered"}
        op: Dict[str, Any] = {"action": "MODIFY_NODE_PROPERTIES", "node_id": target_id, "properties": props_update}

        # If colocated, try to move it away
        if same_place:
            loc, shp = suggest_new_pose(self.scene_graph, target_node, robot_node) or (None, None)
            if loc:
                props_update["location"] = loc
            if shp:
                op["shape"] = shp

        return [op]
        
    # --- Strategy implementation: new information discovery ---
        
    def _check_new_target_discovery_logic(self, skill_info: Dict, ctx: Dict) -> Tuple[bool, Optional[Dict]]:
        """
        Check whether new information discovery conditions are met.
        Returns (trigger_event, details).
        """
        # 1. Get context information
        params = skill_info.get("params", {})
        area_id = ctx.get("area_id") or params.get("area")
        if not area_id:
             return False, None # Cannot determine area

        area_has_hp_target = params.get("area_has_hp_target", False)
        target_node = ctx.get("target_node") # Current target that the search skill may discover
        object_id = ctx.get("object_id")
        
        # 2. Call utility function
        return check_new_target_discovery_logic(
            area_id=area_id,
            area_has_hp_target=area_has_hp_target,
            target_node=target_node,
            object_id=object_id
        )

    def strategy_discover_new_high_priority_target(self, params: Dict, skill_info: Dict, ctx: Dict) -> Optional[List[Dict]]:
        if (skill_info or {}).get('skill') != 'search':
            return None

        trigger_event, _ = self._check_new_target_discovery_logic(skill_info, ctx)
        if not trigger_event:
            return None

        new_type = random.choice(["fire", "hazmat"])
        nodes = getattr(self.scene_graph, "get_all_nodes", lambda: [])() or []
        same_type = [n for n in nodes if (n.get("properties", {}) or {}).get("type") == new_type]
        if not same_type:
            return None

        # Position priority: area centroid > robot position/entity center
        pos = None
        params_obj = (skill_info.get("params") or {})
        area_geom = params_obj.get("area")
        if isinstance(area_geom, dict):
            pos = area_centroid(area_geom)
        if pos is None:
            rid = ctx.get("robot_id")
            pos = get_entity_position(self.scene_graph, rid) or extract_object_position(ctx.get("robot"))

        # Use the last node of the same type as the template
        def _label_num(n):
            lab = (n.get("properties", {}) or {}).get("label", "")
            m = re.search(r"-(\d+)$", str(lab))
            return int(m.group(1)) if m else -1

        base = max(same_type, key=_label_num)

        # Use the shared utility to generate a new node and label
        prefix = new_type.capitalize()
        default_label = f"{prefix}-0"
        new_node, new_label = self._spawn_node_with_new_id_and_label(
            base,
            nodes,
            default_label_prefix=prefix,
            default_label=default_label,
        )

        props = new_node.setdefault("properties", {})
        props["type"] = new_type
        props["status"] = "discovered"

        # location / shape
        if pos is not None:
            loc_obj = infer_nearest_location(self.scene_graph, pos, exclude_id=new_node["id"])
            if loc_obj:
                props["location"] = loc_obj
            new_node["shape"] = create_centered_shape(base, pos)

        skill_info.setdefault("params", {}).update({
            "area_has_hp_target": True,
            "hp_object_id": new_node["id"],
            "hp_target_label": new_label,
        })

        return [{"action": "ADD_NODE", "node_data": new_node}]

    # --- Strategy implementation: robot ---

    def strategy_degrade_robot_battery(self, params: Dict, skill_info: Dict, ctx: Dict) -> Optional[List[Dict]]:
        rid = ctx.get('robot_id')
        if not rid: return []
        node = self.scene_graph.get_node_by_id(rid)
        if node and (node.get("properties", {}).get("battery_level", 100) < 20):
            return []
        return [{"action": "MODIFY_NODE_PROPERTIES", "node_id": rid, "properties": {"battery_level": float(params.get("new_battery_level", 5))}}]
    
    def strategy_set_robot_fault(self, params: Dict, skill_info: Dict, ctx: Dict) -> Optional[List[Dict]]:
        rid = ctx.get('robot_id')
        if not rid: return []
        node = self.scene_graph.get_node_by_id(rid)
        if node and (node.get("properties", {}).get("status") == "error"):
            return []
        return [{"action": "MODIFY_NODE_PROPERTIES", "node_id": rid, "properties": {"status": params.get("status", "error")}}]
    
    def strategy_jam_comm_link(self, params: Dict, skill_info: Dict, ctx: Dict) -> Optional[List[Dict]]:
        rid = ctx.get('robot_id')
        if not rid: return []
        node = self.scene_graph.get_node_by_id(rid)
        if node and (node.get("properties", {}).get("comm") == "jammed"):
            return []
        return [{"action": "MODIFY_NODE_PROPERTIES", "node_id": rid, "properties": {"comm": params.get("comm_status", "jammed")}}]
        
    # --- Strategy implementation: environment ---

    def strategy_block_critical_path(self, params: Dict, skill_info: Dict, ctx: Dict) -> Optional[List[Dict]]:
        """
        Road disruption (block_critical_path).
        """
        path_nodes = get_path_from_context(self.scene_graph, ctx)
        if not path_nodes or len(path_nodes) < 2:
            return None

        # Convert adjacent node pairs on the path into edges
        edges: List[Dict[str, Any]] = []
        for a, b in zip(path_nodes, path_nodes[1:]):
            # Normalize source/target order, preferably in ascending numeric order
            sa, sb = str(a), str(b)
            try:
                ia, ib = int(sa), int(sb)
                src, tgt = (sa, sb) if ia <= ib else (sb, sa)
            except Exception:
                src, tgt = sorted([sa, sb])
            edges.append({"source": src, "target": tgt})

        if not edges:
            return None

        k = max(1, int(params.get("num_edges_to_remove", 1)))
        chosen = edges[:k]

        return [{"action": "DELETE_EDGE", "edge_data": e} for e in chosen]
        
    def strategy_set_area_property(self, params: Dict, skill_info: Dict, ctx: Dict) -> Optional[List[Dict]]:
        """
        Environment property new cases:
        - passability = restricted      -> restricted area (type 2)
        - visibility = low              -> fog/smoke (type 3)
        - illumination = dark           -> darkness/dimmed lighting (type 4)
        - wind_condition = strong       -> strong wind condition (type 5)
        """
        prop = params.get("property")
        val  = params.get("value")
        if not prop:
            return None

        # -------- 1) Restricted area: passability = restricted --------
        if prop == "passability" and val == "restricted":
            candidates = iter_areas_crossed_by_robot_to_dest(
                self.scene_graph,
                context=ctx,
                only_restricted=False,
                exclude_current_area=True,
            )
            if not candidates:
                return None

            # If any candidate area is already restricted, do not generate a new restricted area
            for n in candidates:
                props = (n.get("properties") or {})
                if props.get("passability") == "restricted":
                    return None

            # Otherwise, randomly choose one of these areas and set it to restricted
            chosen = random.choice(candidates)
            return [{
                "action": "MODIFY_NODE_PROPERTIES",
                "node_id": chosen.get("id"),
                "properties": {"passability": "restricted"},
            }]

        # -------- 2) Fog/smoke: visibility = low --------
        if prop == "visibility" and val == "low":
            node = get_relevant_area_node(self.scene_graph, ctx)
            if not node:
                return None
            props = (node.get("properties") or {})
            if props.get("visibility") == "low":
                return None
            return [{
                "action": "MODIFY_NODE_PROPERTIES",
                "node_id": node.get("id"),
                "properties": {"visibility": "low"},
            }]

        # -------- 3) Darkness/dimmed lighting: illumination = dark --------
        if prop == "illumination" and val == "dark":
            district = get_district_node(self.scene_graph)
            if not district:
                return None
            dprops = (district.get("properties") or {})
            if dprops.get("illumination") == "dark":
                return None
            return [{
                "action": "MODIFY_NODE_PROPERTIES",
                "node_id": district.get("id"),
                "properties": {"illumination": "dark"},
            }]

        # -------- 4) Strong wind condition: wind_condition = strong --------
        if prop == "wind_condition" and val == "strong":
            node = get_relevant_area_node(self.scene_graph, ctx)
            if not node:
                return None
            props = (node.get("properties") or {})
            if props.get("wind_condition") == "strong":
                return None
            return [{
                "action": "MODIFY_NODE_PROPERTIES",
                "node_id": node.get("id"),
                "properties": {"wind_condition": "strong"},
            }]

        return None
        
    def strategy_set_path_congestion(self, params: Dict, skill_info: Dict, ctx: Dict) -> Optional[List[Dict]]:
        """
        Randomly choose an intersection/path node on the robot->destination path and set congestion:
        - params["congestion_type"] in {"vehicle", "crowd"}
        """
        path_nodes = get_path_from_context(self.scene_graph, ctx)
        if not path_nodes:
            return None
        congestion_type = params.get("congestion_type") or "vehicle"

        # Collect "street segment" nodes on the path between adjacent intersections
        segments_on_path: set = set()
        # Scan all CONNECTS_TO edges once and build target(intersection) -> {source(street_segment)}
        # to avoid repeated O(E) traversals
        target_to_segments: Dict[str, set] = {}
        for e in self.scene_graph.get_edges_by_type("connects_to") or []:
            src = e.get("source")
            tgt = e.get("target")
            if src is None or tgt is None:
                continue
            tgt_str = str(tgt)
            s = target_to_segments.get(tgt_str)
            if s is None:
                s = set()
                target_to_segments[tgt_str] = s
            s.add(src)

        # For each adjacent intersection pair (a, b) on the path, find street segments connected to both
        path_ids = [str(nid) for nid in path_nodes]
        for a, b in zip(path_ids, path_ids[1:]):
            seg_a = target_to_segments.get(a, set())
            seg_b = target_to_segments.get(b, set())
            if not seg_a or not seg_b:
                continue
            # Segments connected to both a and b are the street_segments between those intersections
            segments_on_path.update(seg_a & seg_b)

        # If any intersection or street segment on the path already has this congestion type, skip generation
        def _has_type(node_id) -> bool:
            node = self.scene_graph.get_node_by_id(node_id)
            if not node:
                return False
            props = node.get("properties") or {}
            return props.get("congestion") == congestion_type

        # Check all intersections
        for nid in path_nodes:
            if _has_type(nid):
                return None
        # Check all street segments
        for sid in segments_on_path:
            if _has_type(sid):
                return None

        # Build candidates from intersections and street segments on the path, choosing only uncongested/unset nodes
        candidates: List[Any] = []
        def _maybe_add_candidate(node_id):
            node = self.scene_graph.get_node_by_id(node_id)
            if not node:
                return
            props = node.get("properties") or {}
            cur = props.get("congestion")
            # Only write if not already marked congested to avoid overwriting other congestion types
            if cur in (None, "", "none"):
                candidates.append(node_id)

        for nid in path_nodes:
            _maybe_add_candidate(nid)
        for sid in segments_on_path:
            _maybe_add_candidate(sid)
        if not candidates:
            return None

        chosen = random.choice(candidates)
        return [{"action": "MODIFY_NODE_PROPERTIES", "node_id": chosen, "properties": {"congestion": congestion_type}}]
        
    def strategy_add_obstruction_node(self, params: Dict, skill_info: Dict, ctx: Dict) -> Optional[List[Dict]]:
        """
        Add an "obstruction" node near the robot side of the robot->target line:
        - Position: biased toward the robot side of the robot->target segment (for example, 20% along the segment).
        - Node template: deep copy one cargo node (the last one), then modify type/status/label.
        """
        robot_node = ctx.get("robot")
        target_node = ctx.get("target_node")
        if not (robot_node and target_node):
            return None

        rpos = extract_object_position(robot_node)
        tpos = extract_object_position(target_node)
        if not rpos or not tpos:
            return None

        # 1) Determine the new node center position
        dist_rt = distance(rpos, tpos)
        if dist_rt is None or dist_rt < 1e-3:
            new_pos = rpos[:]  # Same position as the robot
        else:
            ax, ay = rpos
            bx, by = tpos
            alpha = float(params.get("alpha", 0.2))  # Near the robot side
            nx = ax + alpha * (bx - ax)
            ny = ay + alpha * (by - ay)
            new_pos = [nx, ny]

        # 2) Choose one cargo node as the template: use the last cargo node
        nodes = getattr(self.scene_graph, "get_all_nodes", lambda: [])() or []
        base = None
        for n in nodes:
            props = (n.get("properties") or {})
            if props.get("type") == "cargo":
                base = n  # Do not break; the last cargo overrides previous ones

        # If no cargo exists, fall back to the last node
        if base is None and nodes:
            base = nodes[-1]
        if base is None:
            return None

        # 3) Use the shared utility to generate a new node and label
        new_node, new_label = self._spawn_node_with_new_id_and_label(
            base,
            nodes,
            default_label_prefix="Obstruction",
            default_label="Obstruction-0",
        )

        # 4) Modify properties: type and status
        props = new_node.setdefault("properties", {})
        node_type = params.get("node_type") or "cargo"
        props["type"] = node_type
        props["status"] = params.get("status") or "obstruction"

        # 5) Geometry: rebuild shape centered at new_pos
        new_node["shape"] = create_centered_shape(base, new_pos)

        return [{
            "action": "ADD_NODE",
            "node_data": new_node
        }]

    
    def _spawn_node_with_new_id_and_label(
        self,
        base: Dict[str, Any],
        all_nodes: List[Dict[str, Any]],
        *,
        default_label_prefix: str = "Node",
        default_label: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], str]:
        """
        General utility: copy a new node from the base template and:
          - Assign new id = largest numeric id in the current graph + 1.
          - Generate a new label by incrementing the numeric suffix of the original label,
            or start from 1 if no suffix exists.
        Returns (new_node, new_label).
        """
        new_node = copy.deepcopy(base)

        # 1) Assign new ID
        try:
            last_num = max(
                int(n.get("id")) for n in all_nodes
                if str(n.get("id")).isdigit()
            )
        except ValueError:
            last_num = 0
        new_id = str(last_num + 1)
        new_node["id"] = new_id

        # 2) Handle label
        props = new_node.setdefault("properties", {})
        if default_label is None:
            default_label = f"{default_label_prefix}-0"

        old_label = str(props.get("label") or default_label)
        m = re.search(r"^(.*?)-(\d+)$", old_label)
        if m:
            prefix = m.group(1) or default_label_prefix
            nxt = int(m.group(2)) + 1
        else:
            prefix = default_label_prefix
            nxt = 1

        new_label = f"{prefix}-{nxt}"
        props["label"] = new_label

        return new_node, new_label


# ============================================================
# 2. NewCaseGenerator class
# ============================================================
class NewCaseGenerator:
    """
    Generate new-case events from the current skill context.
    This class is now a wrapper; the main logic is delegated to NewCaseStrategyManager.

    Note: per the new requirement, this no longer uses time cooldowns and only uses count limits.
    """

    def __init__(
        self,
        templates: List[Dict[str, Any]],
        scene_graph: AbstractSceneGraph,
        *,
        max_events: Optional[int] = None,
        is_replay: bool = False,  
    ):
        self.templates = templates or []
        self.scene_graph = scene_graph
        self.is_replay = is_replay
        self._replay_meta = None

        # Strategy manager
        self.strategy_manager = NewCaseStrategyManager(self.scene_graph)

        # Global count limit
        self._max_events: Optional[int] = max_events
        self._generated_count: int = 0

        # Counter used only for event_id generation
        self._counter = 0

        # op_code -> [event_keys...] mapping for stats/annotation
        self._op_to_event_keys: Dict[str, List[str]] = {}
        for k, tpl in NEW_CASE_EVENT_TEMPLATES.items():
            oc = tpl.get("op_code")
            if not oc:
                continue
            self._op_to_event_keys.setdefault(oc, []).append(k)
        
        self._op_usage_resolver = None

    # ---------------- Global Count Control ----------------

    @property
    def generated_count(self) -> int:
        return self._generated_count

    def set_limits(self, *, max_events: Optional[int] = None) -> None:
        if max_events is not None:
            self._max_events = max_events

    def set_replay_meta(self, meta):
        self._replay_meta = meta

    def reset_limits(self) -> None:
        self._generated_count = 0

    def can_generate(self) -> bool:
        if self._max_events is not None and self._generated_count >= self._max_events:
            return False
        return True
    
    def set_op_usage_resolver(self, resolver):
        self._op_usage_resolver = resolver

    def _get_op_usage(self, op_code: str) -> int:
        if not op_code or self._op_usage_resolver is None:
            return 0
        try:
            return int(self._op_usage_resolver(op_code) or 0)
        except Exception:
            return 0

    # ---------------- Public Main Interface ----------------

    def try_generate_once(
        self,
        robot_id: str,
        skill: str,
        params: Dict[str, Any],
        context: Dict[str, Any],
        selection: str = "random",
    ) -> Optional[Dict[str, Any]]:
        skill_info = {"skill": skill, "params": params or {}}
        return self.generate_event(skill_info, context, selection=selection)

    def generate_event(
        self,
        skill_info: Dict[str, Any],
        context: Dict[str, Any],
        selection: str = "random",
    ) -> Optional[Dict[str, Any]]:
        # Replay mode: reproduce directly
        if self.is_replay and self._replay_meta and self._replay_meta.get("new_case_event") is not None:
            event = self._replay_meta["new_case_event"]
            params_view = skill_info.setdefault("params", {})
            op_code = event.get("op_code") or event.get("op_template_id")
            tpl = None
            if op_code:
                for t in self.templates:
                    if t.get("code") == op_code:
                        tpl = t
                        break

            if tpl:
                cfg = (tpl.get("generator_config") or {})  # Same as normal execution
                strategy = cfg.get("strategy")
                if strategy == "discover_new_high_priority_target":
                    # This was a high-priority target discovery event, so re-enable precheck during replay
                    params_view["check_for_new_target"] = True

            dump_var("new_case_event", self._replay_meta["new_case_event"])
            return self._replay_meta["new_case_event"]
        
        # Count limit
        if not self.can_generate():
            return None

        skill_name = (skill_info or {}).get('skill')
        if not skill_name:
            return None
        
        params_view = skill_info.setdefault("params", {})

        # 1) Filter applicable templates
        base = [
            tpl for tpl in self.templates
            if (
                '*' in tpl.get('applicable_skills', []) or
                skill_name in tpl.get('applicable_skills', [])
            ) and (
                '*' in tpl.get('applicable_robots', []) or
                (context.get('robot_type') in tpl.get('applicable_robots', []))
            )
        ]
        if config.collect_replan_dataset or False:
            base = [tpl for tpl in base if tpl.get("category") != "robot"]
        if not base:
            return None

        # 2) Filter out templates with zero relevance and attach usage counts
        candidates: List[Dict[str, Any]] = []
        for tpl in base:
            base_score = self._estimate_relevance_score(tpl, skill_info, context)
            if base_score <= 0.0:
                continue
            op_code = tpl.get("code") or ""
            usage = self._get_op_usage(op_code)  # From NewCaseController
            candidates.append({
                "tpl": tpl,
                "score": base_score,
                "usage": usage,
            })
        if not candidates:
            return None

        # 3) First apply usage-balanced filtering: keep only templates with the lowest usage
        min_usage = min(item["usage"] for item in candidates)
        bucket = [item for item in candidates if item["usage"] == min_usage]
        if not bucket:
            return None

        # 4) Decide attempt order within this bucket based on selection
        sel = (selection or "weighted").lower()
        ordered_tpls: List[Dict[str, Any]] = []

        if sel == "first":
            # Sort simply by descending score
            bucket.sort(key=lambda x: x["score"], reverse=True)
            ordered_tpls = [item["tpl"] for item in bucket]
        elif sel == "random":
            random.shuffle(bucket)
            ordered_tpls = [item["tpl"] for item in bucket]
        else:  # "weighted": draw randomly by score until exhausted
            pool = bucket[:]
            while pool:
                total = sum(item["score"] for item in pool)
                if total <= 0:
                    # If all scores are non-positive, fall back to random order
                    random.shuffle(pool)
                    ordered_tpls.extend([item["tpl"] for item in pool])
                    break
                r = random.random() * total
                upto = 0.0
                chosen_idx = 0
                for i, item in enumerate(pool):
                    upto += item["score"]
                    if upto >= r:
                        chosen_idx = i
                        break
                chosen = pool.pop(chosen_idx)
                ordered_tpls.append(chosen["tpl"])

        # 5) Try templates one by one
        for chosen in ordered_tpls:
            cfg = chosen.get('generator_config', {}) or {}
            strategy = cfg.get('strategy')
            tpl_params = cfg.get('params', {}) or {}
            if not strategy:
                logger.warning(f"Template {chosen.get('code')} has no strategy.")
                continue

            has_hp_before = bool(params_view.get("hp_object_id"))

            # Call the concrete strategy to generate ops
            ops = self.strategy_manager.execute_strategy(strategy, tpl_params, skill_info, context)

            # ---------- Special case: discover_new_high_priority_target ----------
            if strategy == "discover_new_high_priority_target":
                if has_hp_before:
                    # Only enable check_for_new_target; do not require a real new node
                    params_view["check_for_new_target"] = True
                    ops = ops or []
                    event = self._build_event_from_template(
                        chosen,
                        ops,
                        mark_persist_ops=True,      # Keep permanently
                        rollback_delay_cycles=0,
                        strategy=strategy,
                    )
                    return event

                # If no high-priority target existed before, a real node must be generated
                if ops:
                    params_view["check_for_new_target"] = True
                    event = self._build_event_from_template(
                        chosen,
                        ops,
                        mark_persist_ops=True,      # Keep the new node across plans; do not roll it back
                        rollback_delay_cycles=0,
                        strategy=strategy,
                    )
                    return event
                else:
                    # Not applicable; try the next template
                    continue

            # ---------- Other strategies: use the unified rollback strategy ----------
            if not ops:
                continue

            # Default: rollback-enabled, restored next time
            mark_persist_ops = False
            rollback_delay_cycles = 0

            # Robot fault: not recoverable
            if strategy == "set_robot_fault":
                mark_persist_ops = True   # Do not record inverse operations

            # Battery degradation: recoverable, but only after the next rollback cycle
            elif strategy == "degrade_robot_battery":
                mark_persist_ops = False
                rollback_delay_cycles = int(tpl_params.get("rollback_delay_cycles", 1))

            # Communication jam: probabilistically recoverable
            elif strategy == "jam_comm_link":
                prob = float(tpl_params.get("rollback_prob", 0.5))
                if random.random() < prob:
                    # Recoverable: restore next time
                    mark_persist_ops = False
                    rollback_delay_cycles = int(tpl_params.get("rollback_delay_cycles", 0))
                else:
                    # Not recoverable: treat as a long-term jam
                    mark_persist_ops = True

            # Generate event
            event = self._build_event_from_template(
                chosen,
                ops,
                mark_persist_ops=mark_persist_ops,
                rollback_delay_cycles=rollback_delay_cycles,
                strategy=strategy,
            )
            return event

        dump_var("new_case_event", None)
        return None
    
    def _build_event_from_template(
        self,
        tpl: Dict[str, Any],
        ops: List[Dict[str, Any]],
        *,
        mark_persist_ops: bool = False,
        rollback_delay_cycles: int = 0,
        strategy: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Build an event object from a template and increment the count.
        - mark_persist_ops=True   -> do not record inverse operations (not rollbackable)
        - rollback_delay_cycles>=0 -> record inverse operations and restore after N rollback calls
        """
        mode = config.new_case_mode or "immediate"
        if mode == "aggregate":
            # Four special strategies: keep behavior unchanged
            special_strategies = {
                "degrade_robot_battery",
                "set_robot_fault",
                "jam_comm_link",
                "discover_new_high_priority_target",
            }
            if strategy not in special_strategies:
                # Non-special cases: make them uniformly non-recoverable
                mark_persist_ops = True
                rollback_delay_cycles = 0

        event_id = f"dyn-{self._counter}"
        self._counter += 1
        self._generated_count += 1

        op_code = tpl.get("code")
        event_keys = self._op_to_event_keys.get(op_code) or []
        event_key = random.choice(event_keys) if event_keys else None

        event: Dict[str, Any] = {
            "id": event_id,
            "category": tpl.get("category"),
            "type": tpl.get("type"),
            "note": tpl.get("description"),
            "op_template_id": op_code,
            "op_code": op_code,
            "event_key": event_key,
            "ops": ops,
            "success": None,
            "errors": [],
        }

        if mark_persist_ops:
            event["persist_ops_across_plans"] = True

        # Meaningful only for rollbackable events
        if not mark_persist_ops and rollback_delay_cycles > 0:
            event["rollback_delay_cycles"] = int(rollback_delay_cycles)

        dump_var("new_case_event", event)
        return event

    # -------- Relevance and Cooldown --------
    def _estimate_relevance_score(self, tpl: Dict[str, Any], skill_info: Dict[str, Any], ctx: Dict[str, Any]) -> float:
        # stg = (tpl.get('generator_config') or {}).get('strategy', '')
        # params = (tpl.get('generator_config') or {}).get('params', {}) or {}

        # # Environment
        # if stg in ("block_critical_path"):
        #     dest = skill_info.get('params', {}).get('dest_id')
        #     rid = ctx.get('robot_id')
        #     if not rid or not dest:
        #         return 0.0
        #     return 0.5
        # # Target
        # # Interactive object state change
        # if stg in ("move_target", "despawn_target"):
        #     obj = ctx.get('object_id')
        #     if not obj:
        #         return 0.0
        #     return 0.5
        # if stg in ("despawn_carrier", "move_carrier"):
        #     has_cid = bool(ctx.get("carrier_id") or (skill_info.get("params", {}).get("carrier_id")))
        #     return 0.5 if has_cid else 0.0
        
        # # Robot
        # if stg in ("degrade_robot_battery", "set_robot_fault", "jam_comm_link"):
        #     return 0.5 if ctx.get('robot_id') else 0.0

        return 0.4
