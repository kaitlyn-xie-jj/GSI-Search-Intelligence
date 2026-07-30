# -*- coding: utf-8 -*-
from typing import Dict, Any, Optional, List, Tuple, Union, Iterable
import numpy as np
from modules.config.base.enums import (
    BuildingName, TransFacilityType,
)
from modules.utils.geom_utils import (
    polygon_centroid, point_in_area_geometry, polyline_length, polygon_area, euclid, point_to_polyline_distance
)
from modules.utils.location_utils import (
    get_entity_position,
    extract_object_position,
)
from modules.platform.platform_factory import get_default_global_boundary


def make_local_search_area(scene_graph, robot_id: int, radius: float = 80.0) -> Optional[Dict[str, Any]]:
    """Build a local circular area around the robot's current position."""
    rpos = get_entity_position(scene_graph, robot_id)
    if rpos is None:
        return None
    return {"kind": "circle", "center": [float(rpos[0]), float(rpos[1])], "radius": float(radius)}


# ---------- Target selection and inference ----------

def pick_targets_by_semantics_in_area(
    scene_graph,
    params_like: Dict[str, Any],
    area_geom: Dict[str, Any],
    object_id: Optional[int] = None,
) -> List[int]:
    """
    Select semantic target matches inside an area using normalize_target_semantics.
    params_like: params with 'target', 'target_token', or other clues understood by
    normalize_target_semantics.
    """
    tcat, target_filter = normalize_target_semantics(params_like)
    target_ids = select_targets_in_geometry(
        area_geom=area_geom,
        target_category=tcat or "prop",
        object_id=object_id,
        target_filter=target_filter,
        scene_graph=scene_graph,
    )
    return target_ids or []


def infer_object_id_from_target(
    scene_graph,
    target_dict: Optional[Dict[str, Any]],
    robot_id: int,
    area_geom: Optional[Dict[str, Any]] = None,
    fallback_local_radius: float = 80.0,
) -> Optional[int]:
    """
    Infer a usable object_id from target semantics when object_id is missing:
    - If area_geom is provided, select directly inside that area.
    - Otherwise, build a small local area around the robot and select there.
    - Return None if nothing is found.
    """
    if not isinstance(target_dict, dict) or not target_dict:
        return None

    # Build pseudo params so normalize_target_semantics can run.
    params_like = {"target": target_dict}

    # Choose the area.
    area = area_geom
    if not isinstance(area, dict):
        area = make_local_search_area(scene_graph, robot_id, radius=fallback_local_radius)
        if not isinstance(area, dict):
            return None

    ids = pick_targets_by_semantics_in_area(scene_graph, params_like, area)
    return ids[0] if ids else None


def ensure_area_and_pick_targets(
    scene_graph,
    label_to_id_map: Dict[str, int],
    raw_params: Dict[str, Any],
    robot_id: int,
) -> Tuple[Dict[str, Any], Optional[str], List[int], Optional[int]]:
    """
    Given params:
    1) Parse the area in one place.
    2) Parse target semantics.
    3) Select targets inside the area and fill object_id when missing.
    Returns: (area_geom, area_id, target_ids, object_id)
    """
    p = dict(raw_params or {})
    # area
    area_geom, area_id = ensure_area_geometry(p, scene_graph, label_to_id_map)
    # Target semantics.
    tcat, target_filter = normalize_target_semantics(p)
    # Explicit object_id, if any.
    object_id = p.get("object_id")
    # Select targets.
    target_ids = select_targets_in_geometry(
        area_geom=area_geom,
        target_category=tcat or "prop",
        object_id=object_id,
        target_filter=target_filter,
        scene_graph=scene_graph,
    ) or []
    if object_id is None and target_ids:
        object_id = target_ids[0]
    return area_geom, area_id, target_ids, object_id

# ---------- Area metrics for time and energy estimates ----------
def compute_search_metrics_geometry(geom: Dict[str, Any], robot_pos: Optional[List[float]]) -> Tuple[float, float]:
    if not geom or not robot_pos: return 10.0, 100.0
    kind = geom.get('kind')
    if kind in ('area', 'rectangle'):
        coords = geom.get('coords', [])
        center = polygon_centroid(coords)
        dist = euclid(robot_pos, center) if center else 10.0
        area = abs(polygon_area(coords)) if coords else 100.0
        return dist, area
    if kind == 'circle':
        c = geom.get('center'); r = float(geom.get('radius', 0.0) or 0.0)
        return (euclid(robot_pos, c) if c else 10.0), float(np.pi) * (r ** 2)
    if kind == 'line':
        line = geom.get('coords', []); buf = 20.0
        dmin, _ = point_to_polyline_distance(robot_pos, line)
        length = polyline_length(line)
        area = length * (2.0 * buf) if length else 100.0
        return (dmin if dmin is not None else 10.0), area
    if kind == 'point':
        pts = geom.get('coords', []); buf = 30.0
        p = pts[0] if pts else None
        return (euclid(robot_pos, p) if p else 10.0), float(np.pi) * (buf ** 2)
    return 10.0, 100.0

# ---------- Target parsing and filtering ----------
def normalize_target_semantics(
    params: Dict
) -> Tuple[Optional[str], Dict[str, Any]]:
    if not isinstance(params, dict):
        return None, {}
    target = params.get('target', {})
    if not isinstance(target, dict):
        return None, {}

    tclass = target.get('class')
    ttype = target.get('type')
    category = None

    if tclass == 'object':
        category = ttype if ttype else 'prop'
        return category, {
            'mode': 'object',
            'type': ttype,
            'features': target.get('features') or {},
            'conf_ge': target.get('conf_ge') or params.get('conf_ge'),
            'persist_ge_s': target.get('persist_ge_s') or params.get('persist_ge_s'),
        }

    if tclass == 'event':
        # Events still rely on entity node types such as vehicle or human.
        category = ttype if ttype else 'prop'
        return category, {
            'mode': 'event',
            'event_key': target.get('event_type') or ttype,
            'conf_ge': target.get('conf_ge') or params.get('conf_ge'),
            'persist_ge_s': target.get('persist_ge_s') or params.get('persist_ge_s'),
        }

    return None, {}

def select_targets_in_geometry(
    area_geom: Dict[str, Any],
    target_category: str,
    object_id: Optional[int],
    target_filter: Dict[str, Any],
    scene_graph
) -> List[int]:
    if object_id is not None:
        node = scene_graph.get_node_by_id(object_id)
        return [object_id] if node else []

    if target_category in [s.value for s in BuildingName]:
        candidates = scene_graph.get_all_buildings()
    elif target_category in [s.value for s in TransFacilityType]:
        candidates = scene_graph.get_all_trans_facilitys()
    else:
        candidates = scene_graph.get_all_props()

    mode = target_filter.get('mode')
    if mode == 'object':
        ttype = target_filter.get('type')
        features = target_filter.get('features') or {}
        if ttype:
            candidates = [n for n in candidates if n.get('properties', {}).get('type') == ttype]
        for k, v in features.items():
            candidates = [
                n for n in candidates
                if n.get('properties', {}).get(k) == v
                   or n.get('properties', {}).get('features', {}).get(k) == v
            ]
    elif mode == 'event':
        event_key = target_filter.get('event_key')
        if event_key:
            candidates = [n for n in candidates if n.get('properties', {}).get(event_key) is True]

    ids: List[int] = []
    for n in candidates:
        pos = extract_object_position(n)
        if pos and point_in_area_geometry(pos, area_geom):
            ids.append(n.get('id'))
    return ids

# ---------- Area geometry lookup ----------
def ensure_area_geometry(params: Dict, scene_graph, label_to_id_map: Dict) -> Tuple[Dict[str, Any], Optional[int]]:
    # 1) params.area is directly usable.
    if isinstance(params.get('area'), dict) and 'kind' in params['area']:
        a = params['area'].copy()
        if a.get('kind') == 'line' and 'buffer' not in a: a['buffer'] = params.get('line_buffer', 20.0)
        if a.get('kind') == 'point' and 'buffer' not in a: a['buffer'] = params.get('point_buffer', 30.0)
        return a, params.get('area_id')

    area_id = None; a = None
    if isinstance(params, dict):
        raw = params.get('area')
        if isinstance(raw, dict) and 'kind' in raw:
            a = raw.copy()
            if a.get('kind') == 'line' and 'buffer' not in a: a['buffer'] = params.get('line_buffer', 20.0)
            if a.get('kind') == 'point' and 'buffer' not in a: a['buffer'] = params.get('point_buffer', 30.0)
        if a is None:
            token = params.get('area_token')
            if isinstance(token, str):
                nid = label_to_id_map.get(token)
                if nid:
                    area_id = nid
                    node = scene_graph.get_node_by_id(nid)
                    if node:
                        if 'shape' in node:
                            shp = node['shape']; t = shp.get('type')
                            if t == 'rectangle':
                                a = {
                                    'kind': 'rectangle',
                                    'coords': [
                                        shp.get('min_corner'), shp.get('max_corner'),
                                        [shp.get('min_corner')[0], shp.get('max_corner')[1]],
                                        [shp.get('max_corner')[0], shp.get('min_corner')[1]],
                                    ],
                                }
                            elif t == 'circle':
                                a = {'kind': 'circle', 'center': shp.get('center'), 'radius': shp.get('radius', 0.0)}
                            elif t == 'linestring':
                                pts = shp.get('points', [])
                                a = {'kind': 'line', 'coords': pts, 'buffer': params.get('line_buffer', 20.0)}
    if a is None:
        a = get_default_global_boundary()
    return a, area_id

def find_high_priority_in_area(
    scene_graph,
    label_to_id_map: Dict[Any, Any],
    area_geom: Optional[Dict[str, Any]],
    area_id: Optional[Union[str, int]],
    exclude_ids: Optional[Iterable[Union[str, int]]] = None,
) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Find a high-priority target inside the given area.
    Use exclude_ids to skip given objects, such as the current object_id.
    Returns: (has_hp, hp_id, hp_label)
    """
    if not (scene_graph and (area_geom or area_id)):
        return False, None, None

    excl = {str(x) for x in (exclude_ids or [])}

    target_specs = [
        {"class": "object", "type": "fire"},
        {"class": "object", "type": "hazmat"},
        # {"class": "object", "type": "equipment_failure"},
        {"class": "object", "type": "person", "features": {"suspicious": True}},
    ]

    for spec in target_specs:
        tmp_params = {"area": area_geom, "area_id": area_id, "target": spec}
        _, _, ids, _ = ensure_area_and_pick_targets(scene_graph, label_to_id_map, tmp_params, None)
        if ids:
            for cand in ids:
                cid = str(cand)
                if cid in excl:
                    continue
                node = scene_graph.get_node_by_id(cid)
                hp_label = (node or {}).get("properties", {}).get("label")
                return True, cid, hp_label

    return False, None, None
