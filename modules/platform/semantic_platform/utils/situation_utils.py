# -*- coding: utf-8 -*-
import logging
import random
from typing import Dict, Any, Optional, List, Tuple, Callable
from modules.platform.semantic_platform.utils.scene_graph_utils import find_path, _nearest_intersection_id
from modules.utils.geom_utils import (
    segment_intersects_area,
    point_in_area_geometry
)
from modules.utils.location_utils import (
    infer_nearest_location, 
    shape_center_point, 
    create_centered_shape,
    get_entity_position,
    extract_object_position,
)

logger = logging.getLogger(__name__)

_HIGH_PRIORITY_TYPES = {"fire", "hazmat"}

def is_high_priority(node: Dict) -> bool:
    """
    Check if a node is high priority
    """
    if not node:
        return False
    props = node.get("properties", {})
    if props.get("type") in _HIGH_PRIORITY_TYPES:
        return True
    if props.get("type") == "person" and props.get("suspicious"):
        return True
    return False

def check_new_target_discovery_logic(
    area_id: Any,
    area_has_hp_target: bool,
    target_node: Optional[Dict],
    object_id: Any
) -> Tuple[bool, Optional[Dict]]:
    """
    Check if conditions for 'search' skill *new situation generation* are met
    Returns (trigger_event: bool, details: Optional[Dict])
    """
    
    # 1. Check if 'search' skill target is high priority
    target_is_hp = is_high_priority(target_node)
    
    # 2. Apply new generator logic
    if area_has_hp_target:
        # Case 1: Area *has* HP target (per plan)
        # Should *not* inject new high-priority node.
        trigger_event = False
    else:
        # Case 2: Area *does not have* HP target (per plan)
        if target_node is None:
            # 2a: Search target is empty (e.g. pure area search), allow injection
            trigger_event = True
        elif not target_is_hp:
            # 2b: Search target exists but not high priority, allow injection
            trigger_event = True
        else:
            # 2c: Search target exists and is high priority, do not inject
            trigger_event = False

    if trigger_event:
        return True, {"area_id": area_id, "target_id": object_id}
        
    return False, None

def get_relevant_area_node(graph, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Select nodes relevant to "environmental perception":
      - Only based on robot's current location (area / building / trans_facility)
      - illumination (dark) related handled separately by caller through district node
    """

    # 1) Prioritize using pre-parsed robot_location
    loc_ref = context.get("robot_location")
    if loc_ref is None:
        return None

    # 3) First get node by id
    node = graph.get_node_by_id(loc_ref)
    if not node:
        label_to_id = context.get("label_to_id_map") or {}
        nid = label_to_id.get(str(loc_ref))
        if nid is not None:
            node = graph.get_node_by_id(nid)
    if not node:
        return None

    props = (node.get("properties") or {})
    cat = (props.get("category") or "").lower()
    # Only accept area / building / trans_facility "location" nodes.
    if cat not in ("area", "building", "trans_facility"):
        return None

    return node

def get_path_from_context(graph, context: Dict[str, Any]) -> List[str]:
    """
    Unified path acquisition interface:
      - Input: scene_graph, context
      - Output: intersection node id sequence (List[str])
    """
    robot_loc_label = context.get("robot_location_label")
    dest_label = context.get("dest_label") or context.get("destination_label")
    if robot_loc_label is not None and dest_label is not None:
        if str(robot_loc_label) == str(dest_label):
            return []
        
    if not (context.get("dest_id") or context.get("destination_id")):
        target_loc_label = context.get("target_location_label")
        if robot_loc_label is not None and target_loc_label is not None:
            if str(robot_loc_label) == str(target_loc_label):
                return []
            
    rid = context.get("robot_id")
    dst = context.get("dest_id") or context.get("destination_id") or context.get("dest") or context.get("object_id")

    if rid is None or dst is None:
        return []
    if isinstance(dst, (str, int)) and dst == rid:
        return []
    
    s = _nearest_intersection_id(graph, rid)
    t = _nearest_intersection_id(graph, dst)
    if not s or not t:
        return []
    try:
        path = find_path(graph, s, t, require_traversable=False) or []
    except Exception:
        return []

    return list(path)

def get_robot_and_dest_positions(graph, context: Dict[str, Any]) -> Tuple[Optional[List[float]], Optional[List[float]]]:
    """
    Unified retrieval of robot and destination geometric positions (x, y).
    """
    rid = context.get("robot_id")

    # ---- robot pos ----
    rpos = get_entity_position(graph, rid) if rid else None
    if rpos is None:
        rnode = context.get("robot")
        if rnode:
            rpos = extract_object_position(rnode)

    # ---- dest pos ----
    d = context.get("dest")
    if isinstance(d, dict) and "x" in d and "y" in d:
        dpos = [float(d["x"]), float(d["y"])]
    elif isinstance(d, (list, tuple)) and len(d) == 2:
        dpos = [float(d[0]), float(d[1])]
    else:
        did = context.get("dest_id") or context.get("destination_id") or context.get("object_id")
        dpos = get_entity_position(graph, did) if did else None

    return rpos, dpos

def get_district_node(graph) -> Optional[Dict[str, Any]]:
    """Return unique district node (returns None if multiple exist)."""
    if not hasattr(graph, "get_all_nodes"):
        return None
    districts: List[Dict[str, Any]] = []
    for n in graph.get_all_nodes() or []:
        props = (n.get("properties") or {})
        if (props.get("category") or "").lower() == "district":
            districts.append(n)
            break
    if len(districts) == 1:
        return districts[0]
    return None

def iter_areas_crossed_by_robot_to_dest(
    graph,
    context: Dict[str, Any],
    *,
    only_restricted: bool = False,
    exclude_current_area: bool = True,
) -> List[Dict[str, Any]]:
    """
    Return all area nodes whose geometry is intersected by the
    straight line from robot to destination.

    Args:
        only_restricted: if True, only keep areas with passability == 'restricted'
        exclude_current_area: if True, exclude areas that currently contain the robot.
    """
    robot_pos, dest_pos = get_robot_and_dest_positions(graph, context)
    if robot_pos is None or dest_pos is None:
        return []

    # Pre-compute boundary features (label -> geom in {kind, coords/center/radius})
    boundary_map: Dict[str, Dict[str, Any]] = {}
    get_bf = getattr(graph, "get_boundary_features", None)
    if callable(get_bf):
        # district=None -> all
        boundary_map = get_bf()

    nodes = getattr(graph, "get_all_nodes", lambda: [])()
    result: List[Dict[str, Any]] = []

    for node in nodes or []:
        props = node.get("properties") or {}
        if (props.get("category") or "").lower() != "area":
            continue

        # Find the boundary feature for this area by label
        geom = boundary_map.get(props.get("label"))
        if not geom:
            continue

        # Exclude the area that currently contains the robot (by geometry)
        if exclude_current_area and point_in_area_geometry(robot_pos, geom):
            continue

        if only_restricted and props.get("passability") != "restricted":
            continue

        if segment_intersects_area(robot_pos, dest_pos, geom):
            result.append(node)

    return result

def suggest_new_pose(graph, node_to_move: Dict[str, Any], robot_node: Dict[str, Any]) -> Optional[Tuple[Dict[str, Any], Dict[str, Any]]]:
    if not (node_to_move and robot_node):
        return None

    # Get robot center
    rpos = extract_object_position(robot_node)
    all_nodes = getattr(graph, "get_all_nodes", lambda: [])()
    cands = []
    for n in all_nodes:
        props = (n or {}).get("properties") or {}
        if props.get("category") not in ("area", "trans_facility", "building"):
            continue
        ctr = shape_center_point((n or {}).get("shape") or {})
        if ctr is None:
            continue
        cands.append((n, ctr))
    if not cands:
        return None

    random.shuffle(cands)
    chosen = None
    best_far = None; best_d = -1.0
    for n, ctr in cands:
        if not rpos:
            chosen = (n, ctr); break
        dx = float(ctr[0] - rpos[0]); dy = float(ctr[1] - rpos[1])
        d = (dx*dx + dy*dy) ** 0.5
        if d >= 200.0:
            chosen = (n, ctr); break
        if d > best_d:
            best_far, best_d = (n, ctr), d
    if not chosen:
        chosen = best_far
    if not chosen:
        return None

    _, pos = chosen
    new_shape = create_centered_shape(node_to_move, pos)
    loc = infer_nearest_location(graph, pos, exclude_id=node_to_move.get("id"))
    return (loc, new_shape) if loc else (None, new_shape)
