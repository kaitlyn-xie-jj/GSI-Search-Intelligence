
from typing import Any, Dict, List, Optional, Literal
from math import hypot

def dist2(a: List[float], b: List[float]) -> float:
    return (a[0] - b[0])**2 + (a[1] - b[1])**2

def shape_center_point(shape: Dict[str, Any]) -> Optional[List[float]]:
    """Return a representative point for a shape."""
    if not shape:
        return None
    t = shape.get("type")
    if t == "rectangle" and "min_corner" in shape and "max_corner" in shape:
        x0, y0 = shape["min_corner"]; x1, y1 = shape["max_corner"]
        return [(x0 + x1) / 2.0, (y0 + y1) / 2.0]
    if t == "circle" and "center" in shape:
        return list(shape["center"])
    if t == "point" and "center" in shape:
        return list(shape["center"])
    if t == "polygon" and "vertices" in shape and shape["vertices"]:
        vs = shape["vertices"]; n = len(vs)
        sx = sum(p[0] for p in vs); sy = sum(p[1] for p in vs)
        return [sx / n, sy / n]
    if t == "linestring" and "points" in shape and shape["points"]:
        pts = shape["points"]; n = len(pts)
        sx = sum(p[0] for p in pts); sy = sum(p[1] for p in pts)
        return [sx / n, sy / n]
    return None

def extract_object_position(obj: Dict[str, Any]) -> Optional[List[float]]:
    """Extract center coordinates from an object node dictionary."""
    if not isinstance(obj, dict):
        return None
    shape = obj.get("shape") or {}
    center = shape_center_point(shape)
    if center is not None:
        return center
    return None

def get_entity_position(scene_graph, entity_id: Optional[int]) -> Optional[List[float]]:
    """Read an entity's position by ID, or return None if the ID does not exist."""
    if entity_id is None:
        return None
    node = scene_graph.get_node_by_id(entity_id)
    if not node:
        return None
    return extract_object_position(node)


def infer_nearest_location(scene_graph, pos: List[float], exclude_id: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """
    Select the nearest location node from scene_graph based on position and build a location property.
    """
    get_all = getattr(scene_graph, "get_all_nodes", None)
    if not callable(get_all):
        return None

    best = None
    best_d2 = float("inf")
    for node in get_all():
        if not isinstance(node, dict):
            continue
        nid = node.get("id")
        if exclude_id is not None and nid == exclude_id:
            continue

        props = node.get("properties", {}) or {}
        cat = props.get("category")
        if cat not in ("building", "trans_facility", "area"):
            continue

        center = shape_center_point(node.get("shape", {}) or {})
        if center is None:
            continue

        d2 = dist2(pos, center)
        if d2 < best_d2:
            best_d2 = d2
            best = node

    if not best:
        return None

    bprops = best.get("properties", {}) or {}
    return {
        "category": bprops.get("category", "unknown"),
        "type": bprops.get("type", "Unknown"),
        "label": bprops.get("label", "Unknown")
    }

def create_centered_shape(node: Dict[str, Any], position: List[float]) -> Dict[str, Any]:
    """
    Create a centered shape at the given position using the node's original shape spec.
    - rectangle: Keep width and height, and move the center to position.
    - circle:    Keep radius, and set center to position.
    - other:     Fall back to a default circle, reusing the original radius if possible, otherwise 1.0.
    """
    shape = (node or {}).get("shape", {}) or {}
    t = shape.get("type")

    if t == "rectangle" and "min_corner" in shape and "max_corner" in shape:
        x0, y0 = shape["min_corner"]; x1, y1 = shape["max_corner"]
        w, h = float(x1 - x0), float(y1 - y0)
        return {
            "type": "rectangle",
            "min_corner": [position[0] - w/2.0, position[1] - h/2.0],
            "max_corner": [position[0] + w/2.0, position[1] + h/2.0],
        }

    if t == "circle" and "radius" in shape:
        return {"type": "circle", "center": [position[0], position[1]], "radius": float(shape["radius"])}

    # Default circular fallback.
    radius = shape.get("radius", 1.0)
    try:
        r = float(radius)
    except Exception:
        r = 1.0
    return {"type": "circle", "center": [position[0], position[1]], "radius": r}

def entities_same_location(scene_graph,
                           ctx: Dict[str, Any],
                           *,
                           which: Literal["target", "carrier", "surface"],
                           eps: float = 2.0) -> bool:
    """
    Determine whether an entity is at the same location as the robot.
    Priority: ctx location marker over geometry center.

    Logic:
    1) If robot_location and the matching xxx_location both exist in ctx and are equal, return True.
    2) Otherwise, fall back to geometry center distance, treating distance <= eps as same location.
    3) Do not compare labels or infer nearest locations.
    """
    # 1) ctx location marker, used only as a fast pass and not as a rejection condition.
    rloc = ctx.get("robot_location")
    if which == "target":
        oloc = ctx.get("target_location")
    elif which == "carrier":
        oloc = ctx.get("carrier_location")
    else:  # "surface"
        oloc = ctx.get("surface_location")

    if (rloc is not None) and (oloc is not None) and (rloc == oloc):
        # Matching location markers mean same location.
        return True

    # 2) Fallback: geometry center distance.
    robot_node = ctx.get("robot")
    if which == "target":
        other_node = ctx.get("target_node") or scene_graph.get_node_by_id(ctx.get("object_id"))
    elif which == "carrier":
        other_node = ctx.get("carrier_node") or scene_graph.get_node_by_id(ctx.get("carrier_id"))
    else:
        other_node = ctx.get("surface_node") or scene_graph.get_node_by_id(ctx.get("surface_id"))

    if not (robot_node and other_node):
        return False

    rpos = extract_object_position(robot_node)
    opos = extract_object_position(other_node)
    if rpos and opos:
        return hypot(float(rpos[0]) - float(opos[0]), float(rpos[1]) - float(opos[1])) <= float(eps)
    return False
