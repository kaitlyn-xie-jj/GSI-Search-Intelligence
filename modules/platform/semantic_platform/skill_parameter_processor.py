# -*- coding: utf-8 -*-
"""
Skill parameter processor.
Handles parameter preparation, calculation, and validation for all skills.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
from modules.config.base.enums import SkillName

from modules.utils.geom_utils import (
    area_centroid,
    distance,
)
from modules.utils.location_utils import (
    get_entity_position,
    extract_object_position,
)
from modules.task_solver.world_model.utils.param_utils import (
    make_local_search_area,
    pick_targets_by_semantics_in_area,
    infer_object_id_from_target,
    ensure_area_and_pick_targets,
    compute_search_metrics_geometry,
    find_high_priority_in_area,
)

logger = logging.getLogger(__name__)


class SkillParameterProcessor:
    """
    Unified skill parameter processor.
    Handles parameter preparation, calculation, and validation for all skills.
    """
    
    def __init__(self, scene_graph, label_to_id_map: Dict, id_to_label_map: Dict):
        self.scene_graph = scene_graph
        self.label_to_id_map = label_to_id_map
        self.id_to_label_map = id_to_label_map

        # Robot home base positions, used by the return_home skill
        self.robot_home_bases = {}  # robot_id -> home_position
        self._initialize_robot_home_bases()
        
        # Register processors for each skill
        self.processors = {
            SkillName.NAVIGATE.value: self._process_navigate_params,
            SkillName.TAKE_PHOTO.value: self._process_take_photo_params,
            SkillName.SEARCH.value: self._process_search_params,
            SkillName.TAKE_OFF.value: self._process_take_off_params,
            SkillName.LAND.value: self._process_land_params,
            SkillName.RETURN_HOME.value: self._process_return_home_params, 
            SkillName.FOLLOW.value: self._process_follow_params, 
            SkillName.BROADCAST.value: self._process_broadcast_params,
            SkillName.PLACE.value: self._process_place_params,
            SkillName.HANDLE_HAZARD.value: self._process_handle_hazard_params,
            SkillName.GUIDE.value: self._process_guide_params,
        }
        
        # Register execution time calculators
        self.time_calculators = {
            SkillName.NAVIGATE.value: self._calculate_navigate_time,
            SkillName.TAKE_PHOTO.value: self._calculate_take_photo_time,
            SkillName.SEARCH.value: self._calculate_search_time,
            SkillName.TAKE_OFF.value: self._calculate_take_off_time,
            SkillName.LAND.value: self._calculate_land_time,
            SkillName.RETURN_HOME.value: self._calculate_return_home_time, 
            SkillName.FOLLOW.value: self._calculate_follow_time, 
            SkillName.BROADCAST.value: self._calculate_broadcast_time,
            SkillName.PLACE.value: self._calculate_place_time,
            SkillName.HANDLE_HAZARD.value: self._calculate_handle_hazard_time,
            SkillName.GUIDE.value: self._calculate_guide_time,
        }

    # -------- Initialization --------
    def _initialize_robot_home_bases(self):
        """Initialize home base positions for all robots."""
        robots = self.scene_graph.get_all_robots()
        for robot in robots:
            robot_id = robot.get('id')
            robot_type = robot.get('properties', {}).get('type')
            if robot_type not in ('UAV', 'FW_UAV'):
                continue
            initial_location = robot.get('properties', {}).get('location')
            if initial_location:
                if isinstance(initial_location, str):
                    loc_id = self.label_to_id_map.get(initial_location)
                    if loc_id:
                        home_pos = get_entity_position(self.scene_graph, loc_id)
                    else:
                        home_pos = [0, 0]
                elif isinstance(initial_location, dict) and 'label' in initial_location:
                    loc_id = self.label_to_id_map.get(initial_location['label'])
                    if loc_id:
                        home_pos = get_entity_position(self.scene_graph, loc_id)
                    else:
                        home_pos = [0, 0]
                else:
                    home_pos = extract_object_position(robot)
            else:
                home_pos = extract_object_position(robot)
            if home_pos:
                self.robot_home_bases[robot_id] = home_pos
            else:
                self.robot_home_bases[robot_id] = [0, 0]

    # ============= Main Entry =============
    def process_skill_parameters(self, skill: str, params: Dict, robot_id: int) -> Dict[str, Any]:
        """Process parameters for one skill."""
        # Skip sync_wait directly
        if skill == "sync_wait":
            return params.copy()
        p = params.copy()
        processor = self.processors.get(skill)
        if not processor:
            logger.warning(f"No parameter processor for skill: {skill}")
            return p
        return processor(p, robot_id)

    def calculate_execution_time_and_energy(
        self, skill: str, params: Dict, base_time: float, base_energy: float
    ) -> Tuple[float, float]:
        """Calculate skill execution time and energy use."""
        calculator = self.time_calculators.get(skill)
        if not calculator:
            logger.warning(f"No time calculator for skill: {skill}, using base values")
            return base_time, base_energy
        return calculator(params, base_time, base_energy)

    # ============= NAVIGATE =============
    def _process_navigate_params(self, params: Dict, robot_id: int) -> Dict:
        p = params.copy()
        dest_xy = None

        # 1) Explicit coordinates provided
        if isinstance(p.get('dest'), dict) and isinstance(p['dest'].get('x'), (int, float)) and isinstance(p['dest'].get('y'), (int, float)):
            dest_xy = [float(p['dest']['x']), float(p['dest']['y'])]

        # 2) Derive from area
        if dest_xy is None and isinstance(p.get('area'), dict):
            c = area_centroid(p['area'])
            if c:
                dest_xy = c

        # 3) Fall back to entity center (object_id)
        if dest_xy is None and p.get('object_id') is not None:
            pos = get_entity_position(self.scene_graph, p['object_id'])
            if pos:
                dest_xy = pos

        # 4) If still missing, fall back to ensure_area_and_pick_targets
        if dest_xy is None:
            area_geom, area_id, target_ids, object_id = ensure_area_and_pick_targets(
                self.scene_graph, self.label_to_id_map, p, robot_id
            )
            if p.get('area') is None and isinstance(area_geom, dict):
                p['area'] = area_geom
            if area_id and p.get('area_id') is None:
                p['area_id'] = area_id
            if object_id is not None and p.get('object_id') is None:
                p['object_id'] = object_id
            if dest_xy is None and p.get('object_id') is not None:
                pos = get_entity_position(self.scene_graph, p['object_id'])
                if pos:
                    dest_xy = pos

        # Write back dest and distance
        if dest_xy is not None:
            p['dest'] = {'x': float(dest_xy[0]), 'y': float(dest_xy[1])}
            robot_pos = get_entity_position(self.scene_graph, robot_id)
            d = distance(dest_xy, robot_pos)
            if d is not None:
                p['distance'] = d

        return p
    
    def _calculate_navigate_time(self, params: Dict, base_time: float, base_energy: float) -> Tuple[float, float]:
        exec_time = base_time
        energy_used = base_energy
        if 'distance' in params:
            exec_time += params['distance'] * 0.005
            energy_used += params['distance'] * 0.01
        exec_time = min(exec_time, 5.0)
        return exec_time, energy_used

    # ============= TAKE_PHOTO =============
    def _process_take_photo_params(self, params: Dict, robot_id: int) -> Dict:
        p = params.copy()

        # Mark photo scope type
        if isinstance(p.get('area'), dict):
            p.setdefault('photo_scope', 'area')
        elif p.get('object_id') is not None:
            p.setdefault('photo_scope', 'object')

        # Photo type by robot type
        robot = self.scene_graph.get_robot(robot_id)
        rtype = robot.get('properties', {}).get('type') if robot else None
        if rtype == 'UAV':
            p.setdefault('photo_type', 'aerial_wide')
        elif rtype == 'Quadruped':
            p.setdefault('photo_type', 'ground_detail')
        else:
            p.setdefault('photo_type', 'generic')
        if p.get('object_id') is None:
            area_geom, area_id, target_ids, object_id = ensure_area_and_pick_targets(
                self.scene_graph, self.label_to_id_map, p, robot_id
            )
            p['area'] = area_geom
            if area_id:
                p['area_id'] = area_id
            p['target_ids'] = target_ids
            if p.get('object_id') is None and object_id is not None:
                p['object_id'] = object_id
        return p
    
    def _calculate_take_photo_time(self, params: Dict, base_time: float, base_energy: float) -> Tuple[float, float]:
        exec_time = min(base_time, 5.0)
        energy_used = base_energy
        return exec_time, energy_used
    
    # ============= RETURN_HOME =============
    def _process_return_home_params(self, params: Dict, robot_id: int) -> Dict:
        processed_params = params.copy()
        home_position = self.robot_home_bases.get(robot_id)
        if not home_position:
            home_base_id = self.label_to_id_map.get("home_base")
            if home_base_id:
                home_position = get_entity_position(self.scene_graph, home_base_id)
            else:
                home_position = [0, 0]
        processed_params['home_position'] = home_position
        return processed_params

    def _calculate_return_home_time(self, params: Dict, base_time: float, base_energy: float) -> Tuple[float, float]:
        exec_time = min(base_time + 1.0, 8.0)
        energy_used = base_energy + 0.5
        return exec_time, energy_used

    # ============= FOLLOW =============  
    def _process_follow_params(self, params: Dict, robot_id: int) -> Dict:
        processed_params = params.copy()
        target_id = processed_params.get('object_id')
        if target_id:
            target_node = self.scene_graph.get_node_by_id(target_id)
            if target_node:
                target_type = target_node.get('properties', {}).get('type')
                processed_params['target_type'] = target_type
                if target_type == 'vehicle':
                    processed_params['following_distance'] = processed_params.get('following_distance', 10.0)
                elif target_type == 'person':
                    processed_params['following_distance'] = processed_params.get('following_distance', 5.0)
                else:
                    processed_params['following_distance'] = processed_params.get('following_distance', 7.0)
            # Initial distance
            robot_pos = get_entity_position(self.scene_graph, robot_id)
            target_pos = get_entity_position(self.scene_graph, target_id)
            d = distance(robot_pos, target_pos)
            if d is not None:
                processed_params['initial_distance'] = d
        return processed_params

    def _calculate_follow_time(self, params: Dict, base_time: float, base_energy: float) -> Tuple[float, float]:
        exec_time = base_time
        energy_used = base_energy
        if 'initial_distance' in params:
            approach_time = params['initial_distance'] * 0.02
            exec_time += approach_time
            energy_used += approach_time * 0.1
        energy_used *= 1.5
        exec_time = 10.0
        return exec_time, energy_used
    
    # ============= SEARCH =============
    def _process_search_params(self, params: Dict, robot_id: int) -> Dict:
        processed = params.copy()
        area_geom, area_id, target_ids, object_id = ensure_area_and_pick_targets(
            self.scene_graph, self.label_to_id_map, processed, robot_id
        )
        processed['area'] = area_geom
        processed['area_id'] = area_id
        processed['target_ids'] = target_ids
        processed['object_id'] = object_id

        # Search metrics for time/energy
        robot_pos = get_entity_position(self.scene_graph, robot_id)
        dist, area_sz = compute_search_metrics_geometry(area_geom, robot_pos)
        processed['distance'] = dist
        processed['area_size'] = area_sz

        # High-priority target detection in the area
        exclude = [object_id] if object_id is not None else None
        has_hp, hp_id, hp_label = find_high_priority_in_area(
            self.scene_graph, self.label_to_id_map, area_geom, area_id, exclude_ids=exclude
        )
        processed['area_has_hp_target'] = bool(has_hp)
        processed['check_for_new_target'] = False
        if has_hp:
            processed['hp_object_id'] = hp_id
            if hp_label is not None:
                processed['hp_target_label'] = hp_label

        return processed
    
    def _calculate_search_time(self, params: Dict, base_time: float, base_energy: float) -> Tuple[float, float]:
        exec_time = base_time
        energy_used = base_energy
        if 'distance' in params:
            exec_time += params['distance'] * 0.01
            energy_used += params['distance'] * 0.01
        if 'area_size' in params:
            time_per_sq_meter = 5 / 10000
            energy_per_sq_meter = time_per_sq_meter * 5
            exec_time += params['area_size'] * time_per_sq_meter
            energy_used += params['area_size'] * energy_per_sq_meter
        exec_time = min(exec_time, 8.0)
        return exec_time, energy_used
    
    # ============= TAKE_OFF =============
    def _process_take_off_params(self, params: Dict, robot_id: int) -> Dict:
        processed_params = params.copy()
        processed_params['target_altitude'] = processed_params.get('target_altitude', 50.0)
        return processed_params
    
    def _calculate_take_off_time(self, params: Dict, base_time: float, base_energy: float) -> Tuple[float, float]:
        exec_time = min(base_time, 3.0)
        energy_used = base_energy
        return exec_time, energy_used
    
    # ============= LAND =============
    def _process_land_params(self, params: Dict, robot_id: int) -> Dict:
        processed_params = params.copy()
        processed_params['current_altitude'] = processed_params.get('current_altitude', 50.0)
        return processed_params
    
    def _calculate_land_time(self, params: Dict, base_time: float, base_energy: float) -> Tuple[float, float]:
        exec_time = min(base_time, 3.0)
        energy_used = base_energy
        return exec_time, energy_used
    
    # ============= BROADCAST =============
    def _process_broadcast_params(self, params: Dict, robot_id: int) -> Dict:
        p = params.copy()
        p.setdefault("message", "No parking here, please leave immediately.")
        p.setdefault("mode", p.get("mode", "broadcast"))

        # Infer object_id
        if p.get("object_id") is None:
            area = p.get("area")
            obj = infer_object_id_from_target(
                scene_graph=self.scene_graph,
                target_dict=p.get("target"),
                robot_id=robot_id,
                area_geom=area if isinstance(area, dict) else None,
                fallback_local_radius=80.0,
            )
            if obj is not None:
                p["object_id"] = obj
        return p

    def _calculate_broadcast_time(self, params: Dict, base_time: float, base_energy: float) -> Tuple[float, float]:
        exec_time = base_time
        if "duration_ge_s" in params:
            dur = float(params["duration_ge_s"])
            exec_time = min(base_time + max(0.0, dur), 5.0)
        energy_used = base_energy
        if exec_time > base_time:
            energy_used += (exec_time - base_time) * 0.05
        return exec_time, energy_used
    
    # ============= PLACE =============
    def _process_place_params(self, params: Dict, robot_id: int) -> Dict:
        p = params.copy()

        # 1. Resolve the object ID to place/operate on (object_id)
        if p.get('object_id') is None:
            obj_id = infer_object_id_from_target(
                scene_graph=self.scene_graph,
                target_dict=p.get('target'),
                robot_id=robot_id,
                fallback_local_radius=1000.0,
            )
            if obj_id is not None:
                p['object_id'] = obj_id

        # 2. Resolve the surface or carrier by operation type
        # surface_target is now structured data: {"class": "robot/ground/object", "type": xxx, "features": {...}}
        surface_target = p.get('surface_target', {})
        is_load_unload = False
        if isinstance(surface_target, dict):
            surface_class = surface_target.get('class', '')
            is_load_unload = surface_class in ('robot', 'ground')
        elif isinstance(surface_target, str):
            is_load_unload = surface_target in ('ugv', 'ground')
        
        if is_load_unload:
            self._resolve_carrier_id(p, robot_id)
        else:
            if p.get('surface_id') is None:
                # surface_target is structured data and can be used directly for inference
                surface_dict = surface_target if isinstance(surface_target, dict) else None
                surface_id = infer_object_id_from_target(
                    scene_graph=self.scene_graph,
                    target_dict=surface_dict,
                    robot_id=robot_id,
                    fallback_local_radius=1000.0,
                )
                if surface_id is not None:
                    p['surface_id'] = surface_id

        # 3. Calculate operated object weight
        obj_id = p.get('object_id')
        if obj_id is not None:
            prop = self.scene_graph.get_prop(obj_id)
            if prop:
                p['weight'] = prop.get('properties', {}).get('weight_kg', 5)
            else:
                p.setdefault('weight', 5)
        else:
            p.setdefault('weight', 5)

        return p
    
    def _calculate_place_time(self, params: Dict, base_time: float, base_energy: float) -> Tuple[float, float]:
        exec_time = base_time
        energy_used = base_energy
        weight = params.get('weight', 5)
        exec_time += weight * 0.1
        energy_used += weight * 0.05
        exec_time = min(exec_time, 3.0)
        return exec_time, energy_used
    
    def _resolve_carrier_id(self, processed_params: Dict, robot_id: int) -> None:
        """Resolve carrier ID."""
        if 'carrier_id' in processed_params and processed_params['carrier_id'] is not None:
            return
        
        # Extract carrier information from structured surface_target data
        surface_target = processed_params.get('surface_target', {})
        if isinstance(surface_target, dict):
            features = surface_target.get('features', {})
            carrier_label = features.get('label')
            if carrier_label:
                cid = self.label_to_id_map.get(carrier_label)
                if cid is not None:
                    processed_params['carrier_id'] = cid
                    return
        
        # Search other alias fields
        alias_keys = ('carrier_id', 'carrier', 'carrier_label', 'riding_on')
        for k in alias_keys:
            if k in processed_params:
                val = processed_params[k]
                if isinstance(val, int):
                    processed_params['carrier_id'] = val
                    return
                if isinstance(val, str):
                    cid = self.label_to_id_map.get(val)
                    if cid is not None:
                        processed_params['carrier_id'] = cid
                        return
        
        # Find nearest UGV
        ugvs = [
            r for r in self.scene_graph.get_all_robots()
            if r.get('properties', {}).get('type') == 'UGV'
        ]
        if not ugvs:
            logger.warning("No UGV found, cannot infer carrier_id")
            return
        
        import numpy as np
        
        def _pos(eid: Optional[int]) -> Optional[List[float]]:
            return get_entity_position(self.scene_graph, eid) if eid is not None else None
        def _near(p1: Optional[List[float]], p2: Optional[List[float]], eps: float = 1e-6) -> bool:
            return (p1 is not None and p2 is not None and
                    abs(p1[0] - p2[0]) <= eps and abs(p1[1] - p2[1]) <= eps)

        robot_pos = _pos(robot_id)
        target_pos = _pos(processed_params.get('object_id'))
        for ugv in ugvs:
            if _near(_pos(ugv['id']), robot_pos):
                processed_params['carrier_id'] = ugv['id']
                return
        if target_pos is not None:
            for ugv in ugvs:
                if _near(_pos(ugv['id']), target_pos):
                    processed_params['carrier_id'] = ugv['id']
                    return
        if robot_pos is not None:
            distances = []
            for ugv in ugvs:
                up = _pos(ugv['id'])
                if up is not None:
                    d = float(np.linalg.norm(np.array(up) - np.array(robot_pos)))
                    distances.append((d, ugv['id']))
            if distances:
                distances.sort(key=lambda x: x[0])
                processed_params['carrier_id'] = distances[0][1]
                logger.info(f"Inferred carrier_id={processed_params['carrier_id']} based on nearest distance")

    # ============= HANDLE_HAZARD =============
    def _process_handle_hazard_params(self, params: Dict, robot_id: int) -> Dict:
        p = params.copy()
        if p.get('object_id') is None:
            area_geom, area_id, target_ids, object_id = ensure_area_and_pick_targets(
                self.scene_graph, self.label_to_id_map, p, robot_id
            )
            if p.get('area') is None and isinstance(area_geom, dict):
                p['area'] = area_geom
            if area_id and p.get('area_id') is None:
                p['area_id'] = area_id
            if object_id is not None and p.get('object_id') is None:
                p['object_id'] = object_id
        return p
    
    def _calculate_handle_hazard_time(self, params: Dict, base_time: float, base_energy: float) -> Tuple[float, float]:
        exec_time = min(base_time + 1.5, 8.0)
        energy_used = base_energy + 0.8
        return exec_time, energy_used

    # ============= GUIDE =============
    def _process_guide_params(self, params: Dict, robot_id: int) -> Dict:
        p = params.copy()

        # 1. Resolve guided target ID
        if p.get('object_id') is None:
            _, _, _, object_id = ensure_area_and_pick_targets(
                self.scene_graph, self.label_to_id_map, p, robot_id
            )
            if object_id is not None:
                p['object_id'] = object_id

        # 2. Resolve destination coordinates
        dest_xy = None
        if isinstance(p.get('dest'), dict):
            dest = p['dest']
            if 'x' in dest and 'y' in dest:
                dest_xy = [float(dest['x']), float(dest['y'])]
        
        # Fallback: get from destination_id
        if dest_xy is None and p.get('destination_id') is not None:
            pos = get_entity_position(self.scene_graph, p['destination_id'])
            if pos:
                dest_xy = pos
                p['dest'] = {'x': float(dest_xy[0]), 'y': float(dest_xy[1])}

        # 3. Calculate distance
        if dest_xy is not None and p.get('object_id') is not None:
            target_pos = get_entity_position(self.scene_graph, p['object_id'])
            d = distance(target_pos, dest_xy)
            if d is not None:
                p['distance'] = d
        return p
    
    def _calculate_guide_time(self, params: Dict, base_time: float, base_energy: float) -> Tuple[float, float]:
        exec_time = base_time
        energy_used = base_energy
        if 'distance' in params:
            exec_time += params['distance'] * 0.008
            energy_used += params['distance'] * 0.012
        exec_time = min(exec_time, 12.0)
        return exec_time, energy_used
