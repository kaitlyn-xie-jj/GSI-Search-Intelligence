# -*- coding: utf-8 -*-
"""
General utility functions
"""
import os
import json
import random
from typing import Any, Dict, List, Optional

# ===================== Random & ID =====================

_id_counter = 0  # Auto-increment ID sequence shared with main file (exposed via functions)


def maybe_seed(CONFIG: Dict[str, Any]) -> None:
    """Set random seed from config (does not introduce new randomness)."""
    seed = CONFIG.get("generation_controls", {}).get("RANDOM_SEED")
    if seed is not None:
        random.seed(seed)


def generate_random_id() -> str:
    """Generate auto-increment task ID, starting from 0."""
    global _id_counter
    cid = _id_counter
    _id_counter += 1
    return f"g_{cid}"


# ===================== Utilities =====================

def choose_or_none(seq: List[Any]) -> Any:
    """Randomly pick a value from a list (allows empty)."""
    return random.choice(seq) if seq else None


def dedup(seq: List[Any]) -> List[Any]:
    """Deduplicate while preserving order."""
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def trim_dots(s: str) -> str:
    """Remove duplicate trailing periods (does not affect other content)."""
    if not isinstance(s, str):
        return s
    return s[:-1] if s.endswith("..") else s


def append_robots_if_needed(instruction: str, assign: bool, robot_count: int) -> str:
    """
    Append robot count based on the assign boolean decided by the main flow.
    Note: no new random calls are made here to avoid altering the random sequence.
    """
    if assign:
        instruction = instruction.rstrip(".") + f". Deploy {robot_count} robots."
    return instruction

def infer_goal_determinacy(goal_type: str) -> str:
    """
    Returns 'open' (compatible with future extensions).
    """
    return "open"

# ===================== Scene Graph Loading Helpers =====================

def iter_graph_nodes(obj: Any):
    """Loosely iterate over node objects in scene_graph.json (preserving original logic)."""
    if isinstance(obj, list):
        for x in obj:
            if isinstance(x, dict) and "properties" in x:
                yield x
    elif isinstance(obj, dict):
        for key in ("nodes", "graph", "data"):
            if key in obj and isinstance(obj[key], (list, dict)):
                yield from iter_graph_nodes(obj[key])
        if "properties" in obj:
            yield obj


def load_scene(scene_path: str,
               AREA_TYPES_FOR_LOCATIONS,
               BUILDING_TYPES_FOR_LOCATIONS,
               INFRA_TYPES_FOR_LOCATIONS,
               INFRA_TYPES_FOR_ENFORCEMENT) -> Dict[str, Any]:
    """
    Load from scene_graph.json:
    - Location nodes (area, building, trans_facility) with shape/props/id/label
    - Available scene objects (person, vehicle, boat, ...)
    """

    # -------- storage --------
    locations_all = []                     
    area_labels, building_labels, infra_labels = [], [], []
    enforcement_labels = []
    location_nodes = []                    
    persons, vehicles, boats = [], [], []
    cargos, fires, hazmats = [], [], []
    equipment_failures, assembly_components = [], []

    # ------ If path not exists: return empty but consistent -------
    if not os.path.exists(scene_path):
        return {
            "LOCATION_LABELS": [],
            "AREA_LABELS": [],
            "BUILDING_LABELS": [],
            "INFRA_LABELS": [],
            "ENFORCEMENT_LABELS": [],
            "LOCATION_NODES": [],      
            "SCENE_OBJECTS": {
                "person": [], "vehicle": [], "boat": [], "cargo": [],
                "fire": [], "hazmat": [], "equipment_failure": [],
                "assembly_component": []
            },
        }

    # -------- load JSON --------
    with open(scene_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except:
            data = json.loads(f.read())

    # -------- iterate all nodes --------
    for node in iter_graph_nodes(data):
        props = node.get("properties", {})
        cat   = props.get("category", "").lower()
        typ   = props.get("type", "").lower()
        label = props.get("label")
        shape = node.get("shape")

        # ========== Location nodes (area/building/trans_facility) ==========
        if label and shape:

            # AREA
            if cat == "area" and typ in AREA_TYPES_FOR_LOCATIONS:
                area_labels.append(label)
                locations_all.append(label)
                location_nodes.append({
                    "id": node.get("id"),
                    "label": label,
                    "type": typ,
                    "category": cat,
                    "shape": shape,
                    "properties": props,
                })

            # BUILDING
            if cat == "building" and typ in BUILDING_TYPES_FOR_LOCATIONS:
                building_labels.append(label)
                locations_all.append(label)
                location_nodes.append({
                    "id": node.get("id"),
                    "label": label,
                    "type": typ,
                    "category": cat,
                    "shape": shape,
                    "properties": props,
                })

            # TRANS FACILITY
            if cat == "trans_facility" and typ in INFRA_TYPES_FOR_LOCATIONS:
                infra_labels.append(label)
                locations_all.append(label)
                location_nodes.append({
                    "id": node.get("id"),
                    "label": label,
                    "type": typ,
                    "category": cat,
                    "shape": shape,
                    "properties": props,
                })
                # enforcement subset
                if typ in INFRA_TYPES_FOR_ENFORCEMENT:
                    enforcement_labels.append(label)

        if cat == "prop":
            entry = {
                "id": node.get("id"),
                "type": typ,
                "label": label,
                "properties": props,
                "shape": shape,
            }
            if typ == "person": persons.append(entry)
            elif typ == "vehicle": vehicles.append(entry)
            elif typ == "boat": boats.append(entry)
            elif typ == "cargo": cargos.append(entry)
            elif typ == "fire": fires.append(entry)
            elif typ == "hazmat": hazmats.append(entry)
            elif typ == "equipment_failure": equipment_failures.append(entry)
            elif typ == "assembly_component": assembly_components.append(entry)

    return {
        "LOCATION_LABELS": dedup(locations_all),
        "AREA_LABELS": dedup(area_labels),
        "BUILDING_LABELS": dedup(building_labels),
        "INFRA_LABELS": dedup(infra_labels),
        "ENFORCEMENT_LABELS": dedup(enforcement_labels),
        "LOCATION_NODES": location_nodes,
        "SCENE_OBJECTS": {
            "person": persons,
            "vehicle": vehicles,
            "boat": boats,
            "cargo": cargos,
            "fire": fires,
            "hazmat": hazmats,
            "equipment_failure": equipment_failures,
            "assembly_component": assembly_components,
        },
    }

def sample_scene_object(config: Dict[str, Any],
                        kind: str,
                        predicate: Any = None) -> Optional[Dict[str, Any]]:
    """
    Randomly select an object from CONFIG.data_pools.SCENE_OBJECTS.
    kind: "person" / "vehicle" / "boat" / "cargo" / "fire" / "hazmat" / "equipment_failure" / "assembly_component"
    predicate: optional filter function lambda node: bool for constraining attributes (e.g. lambda n: n["properties"].get("injured"))
    """
    scene_objs = (config.get("data_pools") or {}).get("SCENE_OBJECTS") or {}
    pool = scene_objs.get(kind) or []
    if not pool:
        return None

    if predicate is not None:
        candidates = [n for n in pool if predicate(n)]
        if not candidates:
            return None
        return random.choice(candidates)
    return random.choice(pool)


def coord_bounds(CONFIG: Dict[str, Any]) -> tuple[float, float, float, float]:
    settings = CONFIG["geospatial_settings"]["COORDINATES"]
    x_min, x_max = settings["X_RANGE"]
    y_min, y_max = settings["Y_RANGE"]
    return float(x_min), float(x_max), float(y_min), float(y_max)


def clamp_point_to_bounds(CONFIG: Dict[str, Any],
                           x: float,
                           y: float,
                           margin: float = 0.0) -> tuple[float, float]:
    """
    Clamp (x,y) to coordinate bounds with a margin (ensures radius/polygon does not exceed global bounds).
    """
    x_min, x_max, y_min, y_max = coord_bounds(CONFIG)
    ax_min, ax_max = x_min + margin, x_max - margin
    ay_min, ay_max = y_min + margin, y_max - margin

    if ax_min > ax_max:
        ax_min = ax_max = (x_min + x_max) / 2.0
    if ay_min > ay_max:
        ay_min = ay_max = (y_min + y_max) / 2.0

    cx = min(max(x, ax_min), ax_max)
    cy = min(max(y, ay_min), ay_max)
    return cx, cy

def shape_to_area_geom(shape: Dict[str, Any],
                        line_buffer: float = 20.0,
                        point_buffer: float = 30.0) -> Optional[Dict[str, Any]]:
    """
    Convert node['shape'] from scene_graph to solver-side area_geom format for point_in_area_geometry reuse.
    """
    if not isinstance(shape, dict):
        return None
    t = shape.get("type")

    if t == "rectangle":
        mn = shape.get("min_corner")
        mx = shape.get("max_corner")
        if not (mn and mx):
            return None
        x0, y0 = mn
        x1, y1 = mx
        coords = [
            [x0, y0],
            [x1, y0],
            [x1, y1],
            [x0, y1],
        ]
        return {"kind": "rectangle", "coords": coords}

    if t == "circle":
        c = shape.get("center")
        r = shape.get("radius", 0.0)
        if not c:
            return None
        return {"kind": "circle", "center": c, "radius": float(r)}

    if t == "linestring":
        pts = shape.get("points") or []
        if not pts:
            return None
        return {"kind": "line", "coords": pts, "buffer": float(line_buffer)}

    if t == "point":
        c = shape.get("center")
        if not c:
            return None
        return {"kind": "point", "coords": [c], "buffer": float(point_buffer)}

    if t == "polygon":
        vs = shape.get("vertices") or []
        if not vs:
            return None
        return {"kind": "area", "coords": vs}

    return None
