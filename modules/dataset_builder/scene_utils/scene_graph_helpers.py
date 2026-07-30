# scene_graph_helpers.py
from typing import Dict, Tuple, Any, Optional, List
from random import Random
from shapely.geometry import box as shp_box, Point as ShpPoint, Polygon as ShpPolygon
from modules.dataset_builder.scene_utils.geometry_utils import (
    rect_from_center 
)

def bounds_from_rect(rect: Dict[str, Any]) -> Dict[str, float]:
    return {
        "x_min": rect["min_corner"][0],
        "y_min": rect["min_corner"][1],
        "x_max": rect["max_corner"][0],
        "y_max": rect["max_corner"][1],
    }

def center_of_shape(shape: Dict[str, Any]) -> Tuple[float, float]:
    t = shape.get("type")
    if t == "rectangle":
        return (
            (shape["min_corner"][0] + shape["max_corner"][0]) * 0.5,
            (shape["min_corner"][1] + shape["max_corner"][1]) * 0.5,
        )
    if t == "circle":
        return (float(shape["center"][0]), float(shape["center"][1]))
    if t == "polygon":
        xs = [p[0] for p in shape["vertices"]]
        ys = [p[1] for p in shape["vertices"]]
        return (sum(xs) / len(xs), sum(ys) / len(ys))
    if t == "point":
        return (float(shape["center"][0]), float(shape["center"][1]))
    if t == "linestring":
        pts = shape.get("points", [])
        if pts:
            return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))
    return (0.0, 0.0)

def footprint_from_size_at_center(center: Tuple[float, float], size: Dict[str, Any]) -> Tuple[Dict, Any]:
    """
    For robots/props: if size has 'radius' -> circle, else rectangle (width/length).
    Returns (shape_dict, shapely_geom)
    """
    cx, cy = float(center[0]), float(center[1])
    if "radius" in size:
        r = float(size["radius"])
        shp = {"type": "circle", "center": [cx, cy], "radius": r}
        geom = ShpPoint(cx, cy).buffer(r, resolution=16)
        return shp, geom
    w = float(size.get("width", size.get("length", 2.0)))
    l = float(size.get("length", size.get("width", 2.0)))
    shp = rect_from_center((cx, cy), {"width": w, "length": l})
    geom = shp_box(shp["min_corner"][0], shp["min_corner"][1], shp["max_corner"][0], shp["max_corner"][1])
    return shp, geom

def footprint_from_template(center: Tuple[float, float], template: Dict) -> Tuple[Dict, Any]:
    """
    For buildings and general footprints:
    - circle if template.size has 'radius' or template.shape=='circle'
    - otherwise rectangle using size.width/length
    """
    cx, cy = float(center[0]), float(center[1])
    size = template.get("size", {})
    shape_tag = (template.get("shape") or "").lower()

    if "radius" in size or shape_tag == "circle":
        r = float(size.get("radius", size.get("r", 10.0)))
        shape_dict = {"type": "circle", "center": [cx, cy], "radius": r}
        geom = ShpPoint(cx, cy).buffer(r, resolution=32)
        return shape_dict, geom

    w = float(size.get("width", 20.0))
    l = float(size.get("length", 20.0))
    shape_dict = rect_from_center((cx, cy), {"width": w, "length": l})
    geom = shp_box(shape_dict["min_corner"][0], shape_dict["min_corner"][1],
                   shape_dict["max_corner"][0], shape_dict["max_corner"][1])
    return shape_dict, geom

def half_extent_normal(template: Dict) -> float:
    """
    Half-width along the placement normal for road-clearance spacing:
    - circle -> radius
    - rectangle -> length/2
    """
    size = template.get("size", {})
    if "radius" in size or (template.get("shape") or "").lower() == "circle":
        return float(size.get("radius", 10.0))
    return float(size.get("length", 20.0)) * 0.5

def child_shape_within_parent(
    parent_node: Dict,
    child_size: Dict[str, Any],
    *,
    rng: Random,
    trials: int = 60,
    padding: float = 1.0,
) -> Optional[Tuple[Dict, Any]]:
    """
    Randomly place a child footprint inside a parent (rectangle/circle/polygon).
    Returns (shape, geom) or None.
    """
    pshape = parent_node.get("shape", {})
    if not pshape:
        return None

    # sampling bbox + proper contains() filter
    if pshape["type"] == "rectangle":
        minx, miny = pshape["min_corner"]
        maxx, maxy = pshape["max_corner"]
        minx += padding; miny += padding; maxx -= padding; maxy -= padding
        parent_geom = shp_box(pshape["min_corner"][0], pshape["min_corner"][1],
                              pshape["max_corner"][0], pshape["max_corner"][1])
    elif pshape["type"] == "circle":
        cx, cy = pshape["center"]
        r = float(pshape["radius"]) - padding
        minx, miny, maxx, maxy = cx - r, cy - r, cx + r, cy + r
        parent_geom = ShpPoint(cx, cy).buffer(float(pshape["radius"]), resolution=32)
    elif pshape["type"] == "polygon":
        parent_geom = ShpPolygon(pshape["vertices"])
        minx, miny, maxx, maxy = parent_geom.bounds
        minx += padding; miny += padding; maxx -= padding; maxy -= padding
    else:
        return None

    for _ in range(trials):
        cx = rng.uniform(minx, maxx)
        cy = rng.uniform(miny, maxy)
        shp, geom = footprint_from_size_at_center((cx, cy), child_size)
        if parent_geom.contains(geom):
            return shp, geom
    return None


def choose_attr_value(tmpl_manager, rng: Random, category: str, type_key: str, attr_key: str,
                      default_key: str = "default", options_key: str = "options"):
    tpl = tmpl_manager.get_template(category, type_key) or {}
    spec = (tpl.get("attributes") or {}).get(attr_key, {}) or {}
    opts = spec.get(options_key)
    if isinstance(opts, list) and opts:
        return rng.choice(opts)
    return spec.get(default_key)

def decorate_human(tmpl_manager, rng: Random) -> Dict[str, Any]:
    color = choose_attr_value(tmpl_manager, rng, "prop", "person", "clothing_color")
    item = choose_attr_value(tmpl_manager, rng, "prop", "person", "item")
    extras: Dict[str, Any] = {}
    if color: extras["clothing_color"] = color
    if item: extras["item"] = item
    return extras

def decorate_car(tmpl_manager, rng: Random) -> Dict[str, Any]:
    color = choose_attr_value(tmpl_manager, rng, "prop", "vehicle", "color")
    subtype = choose_attr_value(tmpl_manager, rng, "prop", "vehicle", "subtype")
    out = {"license_plate": f"F-{rng.randint(1000, 9999)}"}
    if color: out["color"] = color
    if subtype: out["subtype"] = subtype
    return out

def decorate_boat(tmpl_manager, rng: Random) -> Dict[str, Any]:
    color = choose_attr_value(tmpl_manager, rng, "prop", "boat", "color")
    subtype = choose_attr_value(tmpl_manager, rng, "prop", "boat", "subtype")
    out = {}
    if color: out["color"] = color
    if subtype: out["subtype"] = subtype
    return out

def decorate_cargo(tmpl_manager, rng: Random) -> Dict[str, Any]:
    color = choose_attr_value(tmpl_manager, rng, "prop", "cargo", "color")
    subtype = choose_attr_value(tmpl_manager, rng, "prop", "cargo", "subtype")
    out = {}
    if color: out["color"] = color
    if subtype: out["subtype"] = subtype
    return out

def decorate_assembly_component(tmpl_manager, rng: Random, subtype: Optional[str] = None) -> Dict[str, Any]:
    if subtype is None:
        subtype = choose_attr_value(tmpl_manager, rng, "prop", "assembly_component", "subtype")
    color = choose_attr_value(tmpl_manager, rng, "prop", "assembly_component", "color")
    out: Dict[str, Any] = {"serial": f"AC-{rng.randint(100000, 999999)}"}
    if color: out["color"] = color
    if subtype: out["subtype"] = subtype
    return out


def snap_on_road_center(streets: List[Dict], intersections: List[Dict],
                        rng: Random, seg_prob: float = 0.6) -> Tuple[Optional[Dict], Optional[Tuple[float, float]]]:
    """Pick a center point on a road or intersection (returns parent node and center)."""
    if streets and rng.random() < seg_prob:
        seg = rng.choice(streets)
        (x1, y1), (x2, y2) = seg["shape"]["points"]
        t = 0.15 + 0.7 * rng.random()
        return seg, (x1 + (x2 - x1) * t, y1 + (y2 - y1) * t)
    if intersections:
        inter = rng.choice(intersections)
        return inter, tuple(inter["shape"]["center"])
    return None, None

def place_in_building_any(parent_pool: List[Dict], size_dict: Dict[str, Any],
                          rng: Random, padding: float = 0.5):
    """Randomly pick a building from the pool to place a child shape. Returns (parent_node, shape, geom) or (None, None, None)."""
    if not parent_pool:
        return None, None, None
    parent = rng.choice(parent_pool)
    res = child_shape_within_parent(parent, size_dict, rng=rng, trials=60, padding=padding)
    if not res:
        return None, None, None
    shape, geom = res
    return parent, shape, geom

def sample_point_in_polygon(poly: ShpPolygon, rng: Random, margin: float, max_tries: int = 200):
    """Rejection-sample a point inside a polygon (optional inset margin)."""
    if poly is None or poly.is_empty:
        return None
    region = poly.buffer(-margin) if margin > 0 else poly
    if region.is_empty:
        return None
    minx, miny, maxx, maxy = region.bounds
    for _ in range(max_tries):
        x = rng.uniform(minx, maxx)
        y = rng.uniform(miny, maxy)
        # Use a tiny rectangle to avoid boundary point-containment issues
        tiny = ShpPolygon([(x-1e-6,y-1e-6),(x+1e-6,y-1e-6),(x+1e-6,y+1e-6),(x-1e-6,y+1e-6)])
        if region.contains(tiny):
            return (x, y)
    return None

def place_in_open_area(buildable_union: Optional[ShpPolygon], size_dict: Dict[str, Any],
                       rng: Random, margin_scale: float = 0.6):
    """Place a footprint within buildable_union. Returns (shape, center) or (None, None)."""
    if buildable_union is None or buildable_union.is_empty:
        return None, None
    margin = max(size_dict.get("width", 1.0), size_dict.get("length", 1.0)) * margin_scale
    center = sample_point_in_polygon(buildable_union, rng, margin)
    if center is None:
        return None, None
    from .scene_graph_helpers import footprint_from_size_at_center  # Avoid circular import
    shape, geom = footprint_from_size_at_center(center, size_dict)
    return shape, center

def nearest_district_id(district_nodes: Dict[str, Dict], point_xy: Tuple[float, float]) -> Optional[str]:
    """Find the nearest district_id from the district node map."""
    if not district_nodes:
        return None
    px, py = point_xy
    best = None
    best_d2 = 1e30
    for did, node in district_nodes.items():
        cx, cy = center_of_shape(node["shape"])
        d2 = (px - cx) ** 2 + (py - cy) ** 2
        if d2 < best_d2:
            best_d2, best = d2, did
    return best

def compute_boat_center(water_union: Optional[ShpPolygon], rng: Random,
                        size: Dict[str, Any], max_trials: int = 200):
    """Try to place a boat on the water union. Returns center (cx, cy) or None."""
    if not water_union:
        return None
    minx, miny, maxx, maxy = water_union.bounds
    r = float(size.get("radius", 0.0))
    w = float(size.get("width", 2.0))
    l = float(size.get("length", 6.0))
    # Lenient: check full containment using footprint
    from .scene_graph_helpers import footprint_from_size_at_center
    for _ in range(max_trials):
        cx = rng.uniform(minx, maxx)
        cy = rng.uniform(miny, maxy)
        if not water_union.contains(ShpPoint(cx, cy)):
            continue
        shp, geom = footprint_from_size_at_center((cx, cy), size)
        if water_union.contains(geom):
            return (cx, cy)
    return None


def _nearest_location_node(
    pos: Tuple[float, float],
    candidates: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Find the nearest node to pos among candidate location nodes."""
    px, py = float(pos[0]), float(pos[1])
    best = None
    best_d2 = float("inf")
    for node in candidates:
        c = center_of_shape(node.get("shape", {}))
        if c == (0.0, 0.0):
            continue
        d2 = (px - c[0]) ** 2 + (py - c[1]) ** 2
        if d2 < best_d2:
            best_d2, best = d2, node
    return best


def location_dict(
    node: Dict[str, Any],
    *,
    child_pos: Optional[Tuple[float, float]] = None,
    location_candidates: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Standardize the location field: {category, type, label}.

    Prefers child_pos + location_candidates for precise nearest-location computation;
    falls back to node (parent node) attributes if unavailable or computation fails.
    """
    if child_pos is not None and location_candidates:
        nearest = _nearest_location_node(child_pos, location_candidates)
        if nearest is not None:
            p = nearest.get("properties", {})
            return {
                "category": p.get("category", ""),
                "type": p.get("type", ""),
                "label": p.get("label", ""),
            }

    props = node.get("properties", {})
    return {
        "category": props.get("category", ""),
        "type": props.get("type", ""),
        "label": props.get("label", ""),
    }

def find_containing_or_nearest_area(area_nodes: List[Dict[str, Any]],
                                    point_xy: Tuple[float, float],
                                    type_filter: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Find the AREA polygon containing the point; if none found, return the AREA with the nearest centroid.
    Optionally filter by type (e.g. 'water_body').
    """
    if not area_nodes:
        return None
    px, py = float(point_xy[0]), float(point_xy[1])
    best = None
    best_d2 = 1e30

    for n in area_nodes:
        if n.get("shape", {}).get("type") != "polygon":
            continue
        if type_filter and n.get("properties", {}).get("type") != type_filter:
            continue
        poly = ShpPolygon(n["shape"]["vertices"])
        if poly.contains(ShpPoint(px, py)):
            return n  # Prefer containment
        cx, cy = poly.centroid.x, poly.centroid.y
        d2 = (px - cx) ** 2 + (py - cy) ** 2
        if d2 < best_d2:
            best_d2, best = d2, n
    return best