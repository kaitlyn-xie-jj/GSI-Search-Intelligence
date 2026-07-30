
from typing import Dict, Any, List, Optional, Tuple

from modules.task_solver.world_model.utils.param_utils import (
    ensure_area_geometry,             
    select_targets_in_geometry,        
    normalize_target_semantics,                           
)
from modules.platform.platform_factory import get_default_global_boundary

def area_spec_to_geom(area_spec: Dict[str, Any], scene_graph) -> Dict[str, Any]:
    """
    Lightly adapt the goal definition format into params that ensure_area_geometry can read.
    """
    label_to_id = scene_graph.get_node_map(map_type='label_to_id') if scene_graph else {}
    if isinstance(area_spec, dict) and 'kind' in area_spec:
        params = {'area': area_spec}
    else:
        # Goal definitions use area_type: "Named Area", "Point Radius", "Boundary Selection", or "None".
        params = {'area': _convert_goal_area_to_geom(area_spec, scene_graph)}
    geom, _ = ensure_area_geometry(params, scene_graph, label_to_id)
    return geom

def find_object_ids_by_target_geom(area_geom: Dict[str, Any], target_spec: Dict[str, Any], scene_graph) -> List[int]:
    """
    Filter objects by target definition within the given geometry and return their ids.
    This uses normalize_target_semantics followed by select_targets_in_geometry.
    """
    # Adapt target_spec into input params for normalize_target_semantics.
    params = {'target': _adapt_target_spec(target_spec)}
    target_category, target_filter = normalize_target_semantics(params)
    return select_targets_in_geometry(
        area_geom=area_geom,
        target_category=target_category or 'prop',
        object_id=None,
        target_filter=target_filter or {},
        scene_graph=scene_graph
    )

def _convert_goal_area_to_geom(area_spec: Optional[Dict[str, Any]], scene_graph) -> Dict[str, Any]:
    at = area_spec.get('area_type')
    if at == 'Point Radius':
        c = area_spec.get('center_point') or {}
        return {'kind': 'circle',
                'center': [c.get('x', 0.0), c.get('y', 0.0)],
                'radius': float(area_spec.get('radius_m') or 0.0)}
    if at == 'Boundary Selection':
        pts = area_spec.get('boundary_points') or []
        return {'kind': 'area', 'coords': [[p.get('x',0.0), p.get('y',0.0)] for p in pts]}
    if at == 'Named Area':
        name = area_spec.get('area_name')
        boundary_features = scene_graph.get_boundary_features()
        return boundary_features[name] 
        
    return get_default_global_boundary()

def _adapt_target_spec(target_spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Adapt a goal.target spec into the 'target' input format for normalize_target_semantics:
    - Object: {'class':'object','type':..., 'features': {...}}
    - Event: {'class':'event','type':..., 'event_type': ...}
    """
    if not isinstance(target_spec, dict):
        return {}
    if (target_spec.get('type') == 'event') or ('event_type' in target_spec):
        return {
            'class': 'event',
            'type': target_spec.get('type'),
            'event_type': target_spec.get('event_type') or target_spec.get('type'),
            'conf_ge': target_spec.get('conf_ge'),
            'persist_ge_s': target_spec.get('persist_ge_s'),
        }
    return {
        'class': 'object',
        'type': target_spec.get('type'),
        'features': target_spec.get('features') or {},
        'conf_ge': target_spec.get('conf_ge'),
        'persist_ge_s': target_spec.get('persist_ge_s'),
    }
