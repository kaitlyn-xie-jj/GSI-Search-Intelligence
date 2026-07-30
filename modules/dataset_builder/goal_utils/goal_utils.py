
import random
import math
from math import hypot
from typing import Dict, Any, List, Tuple, Optional
from modules.dataset_builder.goal_utils.helpers_refac import sample_scene_object, shape_to_area_geom, clamp_point_to_bounds
from modules.utils.geom_utils import point_in_area_geometry
from modules.utils.location_utils import extract_object_position

# Planning complexity: plan_level, represented as a list of possible levels
# L0: Shallow planning with 1-5 steps
# L1: Moderate planning complexity with 6-10 steps
# L2: Multi-stage planning with 11+ steps
META_PLAN_LEVEL: Dict[str, List[str]] = {
    "area_search": ["L0"],
    "patrol": ["L0"],
    "transport": ["L1", "L2"],
    "assembly": ["L1", "L2"],
    "target_following": ["L1"],
    "traffic_enforcement": ["L1"],
    "evidence_collection": ["L1"],
    "verbal_broadcast": ["L1"],
    "emergency_response": ["L1"],
    "guidance": ["L1"],
}
# Coordination complexity: coor_level, represented as a list of possible levels
# L0: No coordination
# L1: Weak homogeneous coordination
# L2: Weak heterogeneous coordination
# L3: Strong homogeneous coordination
# L4: Strong heterogeneous coordination
META_COOR_LEVEL: Dict[str, List[str]] = {
    "area_search": ["L0", "L1", "L2"],
    "patrol": ["L0", "L1", "L2"],
    "transport": ["L4"],
    "assembly": ["L4"],
    "target_following": ["L1", "L2"],
    "traffic_enforcement": ["L1", "L2"],
    "evidence_collection": ["L1", "L2"],
    "verbal_broadcast": ["L1", "L2"],
    "emergency_response": ["L1", "L2"],
    "guidance": ["L1", "L2"],
}

def generate_random_coords(CONFIG: Dict, num_points: int = 1, margin: float = 0.0) -> List[Dict[str, float]]:
    """
    Generate random coordinate points (x/y in meters, range [0,1000]) with margin support.
    When used for point_radius, pass margin=radius to ensure the entire circle stays within bounds.
    """
    points = []
    settings = CONFIG["geospatial_settings"]["COORDINATES"]
    x_min, x_max = settings["X_RANGE"]
    y_min, y_max = settings["Y_RANGE"]
    resolution = settings["RESOLUTION"]

    # Effective sampling range (clamp if radius makes range invalid)
    ax_min, ax_max = x_min + margin, x_max - margin
    ay_min, ay_max = y_min + margin, y_max - margin
    if ax_min > ax_max:
        ax_min = ax_max = max(min((x_min + x_max) / 2.0, x_max), x_min)
    if ay_min > ay_max:
        ay_min = ay_max = max(min((y_min + y_max) / 2.0, y_max), y_min)

    for _ in range(num_points):
        x = round(random.uniform(ax_min, ax_max), resolution)
        y = round(random.uniform(ay_min, ay_max), resolution)
        points.append({"x": x, "y": y})
    return points

def _project_to_bounds_along_ray(CONFIG: Dict, cx: float, cy: float, x: float, y: float) -> Tuple[float, float]:
    """
    If (x,y) is outside the global bounds, project the segment (cx,cy)->(x,y)
    onto the rectangle boundary and return the intersection point;
    otherwise return as-is.
    """
    settings = CONFIG["geospatial_settings"]["COORDINATES"]
    x_min, x_max = settings["X_RANGE"]
    y_min, y_max = settings["Y_RANGE"]
    resolution = settings["RESOLUTION"]

    # Already within bounds
    if x_min <= x <= x_max and y_min <= y <= y_max:
        return round(x, resolution), round(y, resolution)

    dx, dy = x - cx, y - cy
    ts = []

    # Compute potential intersection parameter t with four edges: (cx,cy) + t*(dx,dy)
    if dx != 0:
        t1 = (x_min - cx) / dx
        t2 = (x_max - cx) / dx
        ts.extend([t1, t2])
    if dy != 0:
        t3 = (y_min - cy) / dy
        t4 = (y_max - cy) / dy
        ts.extend([t3, t4])

    # Select the smallest positive t in (0, 1] that brings the point back within bounds
    candidates = []
    for t in ts:
        if 0 < t <= 1:
            px, py = cx + t * dx, cy + t * dy
            # Allow numerical tolerance when checking if point is on the rectangle boundary
            if (x_min - 1e-9 <= px <= x_max + 1e-9) and (y_min - 1e-9 <= py <= y_max + 1e-9):
                candidates.append((px, py, t))
    if not candidates:
        # Extreme fallback: clamp directly
        px = min(max(x, x_min), x_max)
        py = min(max(y, y_min), y_max)
        return round(px, resolution), round(py, resolution)

    px, py, _ = min(candidates, key=lambda z: z[2])
    return round(px, resolution), round(py, resolution)

def generate_points_around_center(CONFIG: Dict, center_point: Dict[str, float],
                                  num_points: int,
                                  radius_m: float) -> List[Dict[str, float]]:
    """
    Generate non-self-intersecting boundary points within a circle centered at center_point
    with the given radius_m. Out-of-bounds points are projected onto the rectangle boundary
    along the center-to-point ray.
    """
    settings = CONFIG["geospatial_settings"]["COORDINATES"]
    resolution = settings["RESOLUTION"]

    cx = float(center_point["x"])
    cy = float(center_point["y"])

    # Angle generation (approximate Poisson-disc)
    min_gap = 0.8 * (2 * math.pi / max(3, num_points))
    angles, trials, max_trials = [], 0, 2000
    while len(angles) < num_points and trials < max_trials:
        trials += 1
        theta = random.random() * 2 * math.pi
        if all(abs((theta - a + math.pi) % (2*math.pi) - math.pi) >= min_gap for a in angles):
            angles.append(theta)
    if len(angles) < num_points:
        base = [i * (2 * math.pi / num_points) for i in range(num_points)]
        angles = (angles + random.sample(base, k=min(num_points - len(angles), len(base))))[:num_points]
    angles.sort()

    # Slight radius smoothing
    r_min = 0.3 * radius_m
    raw_rs = [random.uniform(r_min, radius_m) for _ in angles]
    smooth_window = 3
    rs = []
    for i in range(len(raw_rs)):
        acc = 0.0
        for k in range(-smooth_window//2, smooth_window//2 + 1):
            acc += raw_rs[(i + k) % len(raw_rs)]
        rs.append(acc / (smooth_window))

    points = []
    for theta, r in zip(angles, rs):
        x = cx + r * math.cos(theta)
        y = cy + r * math.sin(theta)
        # Project onto boundary if out of bounds
        px, py = _project_to_bounds_along_ray(CONFIG, cx, cy, x, y)
        points.append({"x": round(px, resolution), "y": round(py, resolution)})
    
    return points

def sample_following_ai_recognition(CONFIG) -> Tuple[dict, bool]:
    pools = CONFIG["data_pools"]
    choose = random.choice
    controls = CONFIG.get("generation_controls",{})
    use_scene_features = random.random() < controls.get("TARGET_FROM_SCENE_PROBABILITY",1.0)

    if use_scene_features:
        kinds = ["vehicle","person","boat"]
        random.shuffle(kinds)
        for tgt_type in kinds:
            if tgt_type == "vehicle":
                node = sample_scene_object(CONFIG,"vehicle")
                if node:
                    p = node["properties"]
                    color = p.get("color"); subtype = p.get("subtype")
                    if color and subtype:
                        return {"type":"vehicle","features":{"subtype":subtype,"color":color}}, True
            elif tgt_type == "person":
                node = sample_scene_object(CONFIG,"person",lambda n: n["properties"].get("clothing_color") and n["properties"].get("item"))
                if node:
                    p = node["properties"]
                    return {"type":"person","features":{"clothing_color":p["clothing_color"],"item":p["item"]}}, True
            else:  # boat
                node = sample_scene_object(CONFIG,"boat")
                if node:
                    p = node["properties"]
                    subtype = p.get("subtype")
                    if subtype:
                        feats = {"subtype":subtype}
                        if p.get("color"): feats["color"] = p["color"]
                        return {"type":"boat","features":feats}, True

    # Random fallback logic
    tgt_type = choose(["vehicle","person","boat"])
    if tgt_type == "vehicle":
        color = choose(pools["VEHICLE_COLORS"]); subtype = choose(pools["VEHICLE_TYPES"])
        return {"type":"vehicle","features":{"subtype":subtype,"color":color}}, False
    if tgt_type == "person":
        cloth = choose(pools["CLOTHING_COLORS"]); item = choose(pools["PERSON_ITEMS"])
        return {"type":"person","features":{"clothing_color":cloth,"item":item}}, False
    subtype = choose(pools["BOAT_TYPES"])
    return {"type":"boat","features":{"subtype":subtype}}, False

def human_desc_from_ai(ai: dict) -> str:
    t = ai.get("type")
    f = ai.get("features", {})

    if t == "vehicle":
        color = f.get("color")
        subtype = f.get("subtype")
        if color and subtype:
            return f"the {color} {subtype}"
        if subtype:
            return f"the {subtype}"
        return "the vehicle"

    if t == "person":
        if f.get("clothing_color") and f.get("item"):
            return f"the person wearing {f['clothing_color']} clothes and carrying a {f['item']}"
        if f.get("clothing_color"):
            return f"the person wearing {f['clothing_color']} clothes"
        return "the person"

    if t == "boat":
        if f.get("subtype"):
            return f"the {f['subtype']}"
        return "the boat"

    return "the target"


def build_area_definition(CONFIG: Dict[str, Any],
                          area_params: Dict[str, Any],
                          fallback_locations: List[str]) -> Dict[str, Any]:
    """
    Reuse the area_search area construction logic: named_location / boundary_points / point_radius.
    """
    mix = area_params["MIX"]
    population, weights = list(mix.keys()), list(mix.values())
    area_type = random.choices(population, weights=weights, k=1)[0]

    if area_type == "named_location":
        if fallback_locations:
            loc = random.choice(fallback_locations)
            return {"area_type": "Named Area", "area_name": loc}
        # Fall back to point_radius when no named locations are available
        area_type = "point_radius"

    if area_type == "boundary_points":
        num_points = random.randint(*area_params["BOUNDARY_POINTS_RANGE"])
        radius = random.choice(area_params["RADIUS_METERS_CHOICES"])
        center = generate_random_coords(CONFIG, 1, margin=radius)[0]
        pts = generate_points_around_center(CONFIG, center, num_points, radius)
        return {"area_type": "Boundary Selection", "boundary_points": pts}

    # point_radius
    radius = random.choice(area_params["RADIUS_METERS_CHOICES"])
    center = generate_random_coords(CONFIG, 1, margin=radius)[0]
    return {"area_type": "Point Radius", "center_point": center, "radius_m": radius}

def area_instruction_fragment_from_json(area_json: Dict[str, Any]) -> str:
    t = area_json.get("area_type")
    if t == "Boundary Selection":
        cand = [
            "the marked area",
            "the highlighted search region",
            "the outlined area on the map",
            "the drawn boundary on the interface",
            "the marked polygonal search zone",
            "the area highlighted on the screen",
            "the selected region on the map",
            "the pre-drawn search boundary"
        ]
        return random.choice(cand)
    if t == "Named Area":
        name = area_json.get("area_name", "the designated area")
        cand = [
            f"{name}",
            f"the {name} zone",
            f"the vicinity of {name}",
            f"the area of {name}",
            f"the {name}",
            f"the {name} region",
            f"the {name} area"
        ]
        return random.choice(cand)
    if t == "Point Radius":
        r = area_json.get("radius_m")
        r_txt = f"{r} meters" if r is not None else "the specified radius"
        cand = [
            f"the area within a radius of {r_txt} from the marked point",
            f"the area within {r_txt} of the indicated point",
            f"the area within {r_txt} of the designated point",
            f"the area around the marked point (radius {r_txt})",
            f"the circular region (radius {r_txt}) around the marker",
            f"the circular area with radius {r_txt} centered on the marked point",
            f"the area within {r_txt} centered at the marker",
            f"the vicinity within {r_txt} around the reference point",
        ]
        return random.choice(cand)
    return "the designated search area"

def sample_language_level(CONFIG)->str:
    levels,weights = zip(*CONFIG["language_level_distribution"].items())
    return random.choices(levels,weights=weights,k=1)[0]

def build_meta(language_level: str, source_consistency, binding_consistency, goal_kind: str) -> Dict[str, str]:
    """
    Build the unified meta field.
    """
    return {
        "language_level": language_level,
        "plan_level": META_PLAN_LEVEL.get(goal_kind, "L1"),
        "coor_level": META_COOR_LEVEL.get(goal_kind, "L1"),
        "source_consistency": source_consistency,
        "binding_consistency": binding_consistency,
    }

def _area_json_contains_point(CONFIG: Dict[str, Any],
                              area_json: Dict[str, Any],
                              pos_xy: List[float]) -> bool:
    """
    On the generator side, check whether a point falls within an area_json
    (Named/Boundary/PointRadius).
    """
    if not isinstance(area_json, dict) or not pos_xy:
        return False

    atype = area_json.get("area_type")

    # Point Radius: use Euclidean distance directly
    if atype == "Point Radius":
        center = area_json.get("center_point") or {}
        r = float(area_json.get("radius_m", 0.0) or 0.0)
        cx = center.get("x")
        cy = center.get("y")
        if cx is None or cy is None:
            return False
        return hypot(pos_xy[0] - cx, pos_xy[1] - cy) <= r + 1e-6

    # Boundary Selection: convert boundary_points to polygon
    if atype == "Boundary Selection":
        pts = area_json.get("boundary_points") or []
        if not pts:
            return False
        poly = [[p["x"], p["y"]] for p in pts if "x" in p and "y" in p]
        if not poly:
            return False
        geom = {"kind": "area", "coords": poly}
        return point_in_area_geometry(pos_xy, geom)

    # Named Area: check via label -> shape in LOCATION_NODES
    if atype == "Named Area":
        area_name = area_json.get("area_name")
        if not area_name:
            return False
        for node in (CONFIG.get("data_pools") or {}).get("LOCATION_NODES", []):
            if node.get("label") == area_name:
                shp = node.get("shape")
                geom = shape_to_area_geom(shp)
                if geom and point_in_area_geometry(pos_xy, geom):
                    return True
        return False

    return False

def adjust_search_area_to_cover_target(CONFIG: Dict[str, Any],
                                       area_def_params: Dict[str, Any],
                                       search_area_json: Dict[str, Any],
                                       target_node: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure search_area_json covers the target_node position:
    - If already covered: return as-is;
    - Named Area:
        * Try to find a named area label that covers the target;
        * If none works, fall back to point_radius/boundary_points centered near the target;
    - Point Radius / Boundary Selection:
        * Rebuild the area near the target so it necessarily contains the target.
    """
    pos = extract_object_position(target_node)
    if not pos:
        return search_area_json

    pos_xy = [float(pos[0]), float(pos[1])]

    # Already within the current area: no change needed
    if _area_json_contains_point(CONFIG, search_area_json, pos_xy):
        return search_area_json

    atype = search_area_json.get("area_type")
    mix = area_def_params["MIX"]

    # Internal helper: rebuild point_radius / boundary_points around the target position
    def make_point_radius() -> Dict[str, Any]:
        radius = random.choice(area_def_params["RADIUS_METERS_CHOICES"])
        cx, cy = clamp_point_to_bounds(CONFIG, pos_xy[0], pos_xy[1], margin=radius)
        return {
            "area_type": "Point Radius",
            "center_point": {"x": cx, "y": cy},
            "radius_m": radius,
        }

    def make_boundary() -> Dict[str, Any]:
        num_points = random.randint(*area_def_params["BOUNDARY_POINTS_RANGE"])
        radius = random.choice(area_def_params["RADIUS_METERS_CHOICES"])
        cx, cy = clamp_point_to_bounds(CONFIG, pos_xy[0], pos_xy[1], margin=radius)
        center_point = {"x": cx, "y": cy}
        pts = generate_points_around_center(CONFIG, center_point, num_points, radius)
        return {
            "area_type": "Boundary Selection",
            "boundary_points": pts,
        }

    # If already point_radius / boundary form, rebuild directly at target position
    if atype == "Point Radius":
        return make_point_radius()

    if atype == "Boundary Selection":
        return make_boundary()

    # Named Area case
    if atype == "Named Area":
        # (1) Find a named area that covers the target
        for node in (CONFIG.get("data_pools") or {}).get("LOCATION_NODES", []):
            shp = node.get("shape")
            label = node.get("label")
            if not label or not shp:
                continue
            geom = shape_to_area_geom(shp)
            if geom and point_in_area_geometry(pos_xy, geom):
                return {"area_type": "Named Area", "area_name": label}

        # (2) If no named area contains the point, fall back to BP / PR
        candidates = [k for k in mix.keys() if k != "named_location"]
        if not candidates:
            # Fallback: return at least a point_radius
            return make_point_radius()
        weights = [mix[k] for k in candidates]
        choice = random.choices(candidates, weights=weights, k=1)[0]
        if choice == "boundary_points":
            return make_boundary()
        return make_point_radius()

    # Other unknown types: no action
    return search_area_json

def adjust_named_location_to_cover_target(
    CONFIG: Dict[str, Any],
    location_name: Optional[str],
    target_node: Dict[str, Any],
) -> Optional[str]:
    """
    For tasks like traffic_enforcement that only use named locations:
    - If the current location already contains target_node, keep it unchanged;
    - Otherwise search all LOCATION_NODES for a label that contains the target;
      replace with that label if found; otherwise keep the original location_name.
    """
    if not location_name or not isinstance(target_node, dict):
        return location_name

    pos = extract_object_position(target_node)
    if not pos:
        return location_name
    pos_xy = [float(pos[0]), float(pos[1])]

    loc_nodes = (CONFIG.get("data_pools") or {}).get("LOCATION_NODES", []) or []

    # 1) Check if the current location already contains the target
    for node in loc_nodes:
        if node.get("label") != location_name:
            continue
        shp = node.get("shape")
        geom = shape_to_area_geom(shp)
        if geom and point_in_area_geometry(pos_xy, geom):
            return location_name
        break

    # 2) Iterate all named areas, find the first one containing the target
    for node in loc_nodes:
        label = node.get("label")
        if not label:
            continue
        shp = node.get("shape")
        geom = shape_to_area_geom(shp)
        if geom and point_in_area_geometry(pos_xy, geom):
            return label

    # 3) If not found, leave unchanged
    return location_name

def adjust_search_area_to_exclude_target(
    CONFIG: Dict[str, Any],
    area_def_params: Dict[str, Any],
    search_area_json: Dict[str, Any],
    target_node: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Try to ensure search_area_json does not contain the target_node position.

    Logic:
    - If the current area does not contain the target: return as-is;
    - If it does:
        * Named Area:
            - Try to find a label in LOCATION_NODES that does not contain the target;
            - If not found, fall back to point_radius/boundary_points and generate
              an area near the target that explicitly excludes it;
        * Point Radius / Boundary Selection:
            - Regenerate the area near the target, explicitly ensuring the target falls outside.
    """
    pos = extract_object_position(target_node)
    if not pos:
        return search_area_json

    pos_xy = [float(pos[0]), float(pos[1])]

    # If the current area does not contain the target, no change needed
    if not _area_json_contains_point(CONFIG, search_area_json, pos_xy):
        return search_area_json

    atype = search_area_json.get("area_type")
    mix = area_def_params["MIX"]

    # Internal helper: rebuild point_radius / boundary_points near the target, excluding it
    def make_point_radius_exclusive() -> Dict[str, Any]:
        # Retry several times to ensure the new area does not contain the target
        for _ in range(8):
            radius = random.choice(area_def_params["RADIUS_METERS_CHOICES"])
            angle = random.random() * 2 * math.pi
            # Pick a center at radius*2 distance outside the target point
            cx = pos_xy[0] + math.cos(angle) * radius * 2.0
            cy = pos_xy[1] + math.sin(angle) * radius * 2.0
            cx, cy = clamp_point_to_bounds(CONFIG, cx, cy, margin=radius)
            center = {"x": cx, "y": cy}
            candidate = {
                "area_type": "Point Radius",
                "center_point": center,
                "radius_m": radius,
            }
            if not _area_json_contains_point(CONFIG, candidate, pos_xy):
                return candidate

        # Fallback: generate one (most likely outside the target anyway)
        radius = random.choice(area_def_params["RADIUS_METERS_CHOICES"])
        cx, cy = clamp_point_to_bounds(CONFIG, pos_xy[0] + radius * 2.0, pos_xy[1], margin=radius)
        return {
            "area_type": "Point Radius",
            "center_point": {"x": cx, "y": cy},
            "radius_m": radius,
        }

    def make_boundary_exclusive() -> Dict[str, Any]:
        num_points = random.randint(*area_def_params["BOUNDARY_POINTS_RANGE"])
        for _ in range(8):
            radius = random.choice(area_def_params["RADIUS_METERS_CHOICES"])
            angle = random.random() * 2 * math.pi
            cx = pos_xy[0] + math.cos(angle) * radius * 2.0
            cy = pos_xy[1] + math.sin(angle) * radius * 2.0
            cx, cy = clamp_point_to_bounds(CONFIG, cx, cy, margin=radius)
            center = {"x": cx, "y": cy}
            pts = generate_points_around_center(CONFIG, center, num_points, radius)
            candidate = {
                "area_type": "Boundary Selection",
                "boundary_points": pts,
            }
            if not _area_json_contains_point(CONFIG, candidate, pos_xy):
                return candidate

        # Fallback: last candidate
        radius = random.choice(area_def_params["RADIUS_METERS_CHOICES"])
        cx, cy = clamp_point_to_bounds(CONFIG, pos_xy[0] + radius * 2.0, pos_xy[1], margin=radius)
        center = {"x": cx, "y": cy}
        pts = generate_points_around_center(CONFIG, center, num_points, radius)
        return {
            "area_type": "Boundary Selection",
            "boundary_points": pts,
        }

    # If already point_radius / boundary form, rebuild with target-exclusion logic
    if atype == "Point Radius":
        return make_point_radius_exclusive()

    if atype == "Boundary Selection":
        return make_boundary_exclusive()

    # Named Area case
    if atype == "Named Area":
        loc_nodes = (CONFIG.get("data_pools") or {}).get("LOCATION_NODES", []) or []
        # (1) Find a named area label that does not contain the target
        candidates = []
        for node in loc_nodes:
            label = node.get("label")
            shp = node.get("shape")
            if not label or not shp:
                continue
            geom = shape_to_area_geom(shp)
            if geom and not point_in_area_geometry(pos_xy, geom):
                candidates.append(label)

        if candidates:
            return {
                "area_type": "Named Area",
                "area_name": random.choice(candidates),
            }

        # (2) If all named areas cover the target, fall back to BP / PR
        #     Same as the cover version, use MIX to decide the fallback type
        candidates_type = [k for k in mix.keys() if k != "named_location"]
        if not candidates_type:
            # Fallback: return at least a point_radius
            return make_point_radius_exclusive()
        weights = [mix[k] for k in candidates_type]
        choice = random.choices(candidates_type, weights=weights, k=1)[0]
        if choice == "boundary_points":
            return make_boundary_exclusive()
        return make_point_radius_exclusive()

    # Other unknown types: no action
    return search_area_json
