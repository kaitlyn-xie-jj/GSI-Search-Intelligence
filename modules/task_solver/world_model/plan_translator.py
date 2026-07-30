# -*- coding: utf-8 -*-
"""
Plan translator - convert skill strings into structured skill parameters.

Responsibilities:
1. Parse skill strings and extract semantic parameters.
2. Translate semantic parameters into structured data, such as targets and areas.
3. Output normalized skill parameter dictionaries.

Output parameter format:
- target: structured target data {"class": "object/event", "type": xxx, "features": {...}}
- area: area geometry data {"kind": "circle/rectangle/area/line/point", "coords/center/radius": ...}
- dest: destination coordinates {"x": float, "y": float}
- object_id: target entity ID
"""

import re
from typing import Dict, Any, List, Optional
from collections import defaultdict

from modules.utils.geom_utils import area_centroid
from modules.utils.location_utils import get_entity_position
from modules.task_solver.world_model.utils.goal_translate_utils import (
    get_step_arg,
    get_message_from_goal,
    get_patrol_duration_from_goal,
    normalize_param,
    is_coord,
)
from modules.task_solver.world_model.utils.plan_translate_utils import (
    ALIASES,
    resolve_area_feature,
    resolve_target_for_skill,
    resolve_object_id_from_runtime,
    normalize_token_by_category_map,
    extract_detection_constraints,
    parse_target_token,
)


class PlanTranslator:
    """
    Plan translator - convert skill strings into structured skill parameters.
    """

    def __init__(self, scene_graph, label_to_id_map: Dict, id_to_label_map: Dict):
        self.scene_graph = scene_graph
        self.label_to_id_map = label_to_id_map
        self.id_to_label_map = id_to_label_map

    # =========================================================================
    # Public interface
    # =========================================================================
    
    def translate_timestep_skills(
        self,
        timestep_skills: Dict[str, Dict[str, Any]],
        skill_schemas: Dict[str, Any],
        category_map: Dict[str, str],
        goal_cfg: Dict[str, Any],
        area_boundaries: Dict[str, Any],
        runtime_params: Dict[str, Any],
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Translate skill lists organized by timestep.
        
        Args:
            timestep_skills: Skill list organized by timestep. Supports two formats:
                - New format: {"0": {"UGV_01": {"skill_str": "navigate<xxx>", "task_id": "task_1"}, ...}, ...}
                - Old format: {"0": {"UGV_01": "navigate<xxx>", ...}, ...}
            skill_schemas: Skill schema definitions.
            category_map: Category mapping.
            goal_cfg: Goal configuration.
            area_boundaries: Area boundaries.
            runtime_params: Runtime parameters.
            
        Returns:
            {"0": {"UGV_01": {"skill": "navigate", "params": {...}}, ...}, ...}
        """
        if not timestep_skills:
            return {}
        
        # Normalize the input format and extract skill strings plus task ID mappings.
        normalized_skills, task_id_map = self._normalize_timestep_skills(timestep_skills)
        
        # Collect all active robots.
        active_robots = sorted({
            robot for robots in normalized_skills.values() for robot in robots
        })
        
        # Build the translation context.
        ctx = TranslationContext(
            skill_schemas=skill_schemas,
            category_map=category_map or {},
            goal_cfg=goal_cfg or {},
            area_boundaries=area_boundaries or {},
            runtime_params=runtime_params or {},
            active_robots=active_robots,
            timestep_skills=normalized_skills,
        )
        
        # Translate skills for each timestep.
        translated: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for timestep_str, robot_skills in normalized_skills.items():
            translated[timestep_str] = {}
            for robot_label, skill_string in robot_skills.items():
                skill_info = self._translate_skill(skill_string, robot_label, ctx)
                if skill_info:
                    # Add task_id and goal_type to params.
                    task_id = task_id_map.get(timestep_str, {}).get(robot_label)
                    if task_id is not None:
                        skill_info["params"]["task_id"] = task_id
                    goal_type = ctx.goal_cfg.get("goal_type")
                    if goal_type is not None:
                        skill_info["params"]["goal_type"] = goal_type
                    translated[timestep_str][robot_label] = skill_info
        
        return translated
    
    def _normalize_timestep_skills(
        self, timestep_skills: Dict[str, Dict[str, Any]]
    ) -> tuple:
        """
        Normalize timestep skill data, supporting both new and old formats.
        
        Args:
            timestep_skills: Raw timestep skill data.
            
        Returns:
            (normalized_skills, task_id_map)
            - normalized_skills: {"0": {"UGV_01": "navigate<xxx>", ...}, ...}
            - task_id_map: {"0": {"UGV_01": "task_1", ...}, ...}
        """
        normalized: Dict[str, Dict[str, str]] = {}
        task_id_map: Dict[str, Dict[str, str]] = {}
        
        for timestep_str, robot_skills in timestep_skills.items():
            normalized[timestep_str] = {}
            task_id_map[timestep_str] = {}
            
            for robot_label, skill_data in robot_skills.items():
                if isinstance(skill_data, dict):
                    # New format: {"skill_str": "navigate<xxx>", "task_id": "task_1"}
                    normalized[timestep_str][robot_label] = skill_data.get("skill_str", "")
                    task_id = skill_data.get("task_id")
                    if task_id is not None:
                        task_id_map[timestep_str][robot_label] = task_id
                else:
                    # Old format: "navigate<xxx>"
                    normalized[timestep_str][robot_label] = str(skill_data)
        
        return normalized, task_id_map

    # =========================================================================
    # Skill translation core
    # =========================================================================
    
    def _translate_skill(
        self,
        skill_string: str,
        robot_label: str,
        ctx: 'TranslationContext',
    ) -> Optional[Dict[str, Any]]:
        """Translate a single skill string."""
        # Parse the skill name and parameters.
        skill_name = skill_string.split("<", 1)[0].strip()
        schema = ctx.skill_schemas.get(skill_name)
        if not schema:
            return None
        
        match = schema["pattern"].fullmatch(skill_string.strip())
        if not match:
            return None
        
        # Extract raw parameter values.
        raw_values = [normalize_param(v.strip()) for v in match.groups()]
        
        # Dispatch by skill type.
        skill_lower = skill_name.lower()
        handler = self._get_skill_handler(skill_lower)
        if handler:
            params = handler(raw_values, robot_label, ctx)
        else:
            params = self._translate_generic(raw_values, ctx)
        
        return {"skill": skill_name, "params": params}
    
    def _get_skill_handler(self, skill_lower: str):
        """Get the skill processor."""
        handlers = {
            "search": self._translate_search,
            "broadcast": self._translate_broadcast,
            "speak": self._translate_broadcast,
            "take_photo": self._translate_take_photo,
            "follow": self._translate_follow,
            "navigate": self._translate_navigate,
            "goto": self._translate_navigate,
            "go_to": self._translate_navigate,
            "place": self._translate_place,
            "handle_hazard": self._translate_handle_hazard,
            "guide": self._translate_guide,
        }
        return handlers.get(skill_lower)

    # =========================================================================
    # Skill translation implementations
    # =========================================================================
    
    def _translate_search(
        self, raw_values: List, robot_label: str, ctx: 'TranslationContext'
    ) -> Dict[str, Any]:
        """
        Translate a search skill.
        
        Input: search<area_token>_for<target_token>
        Output:
            - area: {"kind": xxx, "coords": xxx}
            - area_token: raw area semantic parameter
            - target: {"class": xxx, "type": xxx, "features": {...}}
            - target_token: raw target semantic parameter
            - object_id: target entity ID, optional
            - conf_ge, persist_ge_s: detection thresholds, optional
        """
        params: Dict[str, Any] = {}
        area_token = raw_values[0] if len(raw_values) > 0 else None
        target_token = raw_values[1] if len(raw_values) > 1 else None
        
        # Parse the area.
        area_feat = resolve_area_feature(
            area_token, ctx.goal_cfg, ctx.area_boundaries, ctx.category_map
        )
        params["area"] = area_feat
        params["area_token"] = area_token
        
        # Parse the target.
        target, object_id = self._resolve_target(target_token, ctx)
        params["target"] = target
        params["target_token"] = target_token
        if object_id is not None:
            params["object_id"] = object_id
        
        # Detection thresholds.
        self._add_detection_constraints(params, ctx.goal_cfg)
        
        # Patrol duration.
        goal_type = ctx.goal_cfg.get("goal_type", "").lower()
        if goal_type == "patrol":
            pat_dur = get_patrol_duration_from_goal(ctx.goal_cfg)
            if pat_dur is not None:
                params["duration_ge_s"] = pat_dur
        
        # Runtime parameter fallback.
        self._apply_runtime_object_id(params, robot_label, ctx)
        
        # Ensure target is available.
        self._ensure_target_from_object_id(params)
        
        return params
    
    def _translate_broadcast(
        self, raw_values: List, robot_label: str, ctx: 'TranslationContext'
    ) -> Dict[str, Any]:
        """
        Translate a broadcast or speak skill.
        
        Input: broadcast<target_token>
        Output:
            - target: {"class": xxx, "type": xxx, "features": {...}}
            - target_token: raw target semantic parameter
            - object_id: target entity ID, optional
            - message: broadcast message
            - duration_ge_s: duration, optional
        """
        params: Dict[str, Any] = {}
        target_token = raw_values[0] if raw_values else None
        
        # Parse the target.
        target, object_id = self._resolve_target(target_token, ctx)
        params["target"] = target
        params["target_token"] = target_token
        if object_id is not None:
            params["object_id"] = object_id
        
        # Detection thresholds.
        self._add_detection_constraints(params, ctx.goal_cfg)
        
        # Broadcast message.
        goal_msg = get_message_from_goal(ctx.goal_cfg)
        params["message"] = goal_msg or "No parking here, please leave immediately."
        
        # Duration.
        dur = get_step_arg(ctx.goal_cfg, "SPEAK_DURATION", "duration_ge_s")
        if dur is not None:
            params["duration_ge_s"] = float(dur)
        
        # Runtime parameter fallback.
        self._apply_runtime_object_id(params, robot_label, ctx)
        
        # Ensure target is available.
        self._ensure_target_from_object_id(params)
        
        return params
    
    def _translate_take_photo(
        self, raw_values: List, robot_label: str, ctx: 'TranslationContext'
    ) -> Dict[str, Any]:
        """
        Translate a take_photo skill.
        
        Input: take_photo<target_token>
        Output:
            - target: {"class": xxx, "type": xxx, "features": {...}}
            - target_token: raw target semantic parameter
            - object_id: target entity ID, optional
            - area: area geometry data, optional
            - bind_event_type: bound event type, optional
        """
        params: Dict[str, Any] = {}
        target_token = raw_values[0] if raw_values else None
        
        # Try parsing as an area.
        area_feat = resolve_area_feature(
            target_token, ctx.goal_cfg, ctx.area_boundaries, ctx.category_map
        )
        if area_feat is not None:
            params["area"] = area_feat
            params["area_token"] = target_token
        
        # Parse the target.
        target, object_id = self._resolve_target(target_token, ctx)
        params["target"] = target
        params["target_token"] = target_token
        if object_id is not None:
            params["object_id"] = object_id
        
        # Event binding.
        if isinstance(target, dict) and target.get("class") == "event":
            event_type = target.get("event_type")
            if event_type:
                params["bind_event_type"] = event_type
        
        # Runtime parameter fallback.
        self._apply_runtime_object_id(params, robot_label, ctx)
        
        # Ensure target is available.
        self._ensure_target_from_object_id(params)
        
        return params
    
    def _translate_follow(
        self, raw_values: List, robot_label: str, ctx: 'TranslationContext'
    ) -> Dict[str, Any]:
        """
        Translate a follow skill.
        
        Input: follow<target_token>
        Output:
            - target: {"class": xxx, "type": xxx, "features": {...}}
            - target_token: raw target semantic parameter
            - object_id: target entity ID, optional
            - bind_event_type: bound event type, optional
        """
        params: Dict[str, Any] = {}
        target_token = raw_values[0] if raw_values else None
        
        # Parse the target.
        target, object_id = self._resolve_target(target_token, ctx)
        params["target"] = target
        params["target_token"] = target_token
        if object_id is not None:
            params["object_id"] = object_id
        
        # Detection thresholds.
        self._add_detection_constraints(params, ctx.goal_cfg)
        
        # Event binding.
        if isinstance(target, dict) and target.get("class") == "event":
            event_type = target.get("event_type")
            if event_type:
                params["bind_event_type"] = event_type
        
        # Runtime parameter fallback.
        self._apply_runtime_object_id(params, robot_label, ctx)
        
        # Ensure target is available.
        self._ensure_target_from_object_id(params)
        
        return params
    
    def _translate_navigate(
        self, raw_values: List, robot_label: str, ctx: 'TranslationContext'
    ) -> Dict[str, Any]:
        """
        Translate a navigate skill.

        Input: navigate<dest_token>
        Output:
            - dest: {"x": float, "y": float} destination coordinates
            - area: area geometry data, optional
            - area_token: raw area semantic parameter
            - object_id: target entity ID, optional
        """
        params: Dict[str, Any] = {}
        dest_token = raw_values[0] if raw_values else None
        params["area_token"] = dest_token
        
        dest_xy = None
        
        # 1) Try parsing from an area.
        area_feat = resolve_area_feature(
            dest_token, ctx.goal_cfg, ctx.area_boundaries, ctx.category_map
        )
        if area_feat:
            params["area"] = area_feat
            centroid = area_centroid(area_feat)
            if centroid:
                dest_xy = centroid
        
        # 2) Try parsing from an entity.
        if dest_xy is None:
            norm_tok = normalize_token_by_category_map(dest_token, ctx.category_map)
            if norm_tok:
                oid = self.label_to_id_map.get(norm_tok)
                if oid is not None:
                    params["object_id"] = oid
                    pos = get_entity_position(self.scene_graph, oid)
                    if pos:
                        dest_xy = pos
        
        # 3) Runtime parameter fallback.
        self._apply_runtime_object_id(params, robot_label, ctx, alt_keys=["dest_id"])
        
        # 4) Get position from object_id.
        if dest_xy is None and params.get("object_id") is not None:
            pos = get_entity_position(self.scene_graph, params["object_id"])
            if pos:
                dest_xy = pos
        
        # Write destination coordinates.
        if dest_xy is not None:
            params["dest"] = {"x": float(dest_xy[0]), "y": float(dest_xy[1])}
        
        return params
    
    def _translate_place(
        self, raw_values: List, robot_label: str, ctx: 'TranslationContext'
    ) -> Dict[str, Any]:
        """
        Translate a place skill.

        Input: place<object_token>_on<surface_token>
        Output:
            - target: {"class": xxx, "type": xxx, "features": {...}} placed object
            - target_token: raw object semantic parameter
            - object_id: placed object ID, optional
            - surface_target: {"class": xxx, "type": xxx, "features": {...}} placement surface
            - surface_token: raw surface semantic parameter
            - surface_id: placement surface ID, optional
        """
        params: Dict[str, Any] = {}
        obj_token = raw_values[0] if len(raw_values) > 0 else None
        surface_token = raw_values[1] if len(raw_values) > 1 else None
        
        # Parse the placed object.
        target, object_id = self._resolve_target(obj_token, ctx)
        params["target"] = target
        params["target_token"] = obj_token
        if object_id is not None:
            params["object_id"] = object_id
        
        # Parse the placement surface.
        params["surface_token"] = surface_token
        surface_target = self._resolve_surface_target(surface_token, robot_label, ctx)
        params["surface_target"] = surface_target
        
        # For a regular placement, not loading or unloading, parse surface_id.
        if not self._is_load_unload_surface(surface_token):
            surface_result, surface_id = self._resolve_target(surface_token, ctx)
            if surface_id is not None:
                params["surface_id"] = surface_id
        
        # Runtime parameter fallback.
        self._apply_runtime_object_id(params, robot_label, ctx)
        
        return params
    
    def _translate_handle_hazard(
        self, raw_values: List, robot_label: str, ctx: 'TranslationContext'
    ) -> Dict[str, Any]:
        """
        Translate a handle_hazard skill.
        
        Input: handle_hazard<hazard_token>
        Output:
            - target: {"class": xxx, "type": xxx, "features": {...}}
            - target_token: raw target semantic parameter
            - object_id: target entity ID, optional
            - area: area geometry data
        """
        params: Dict[str, Any] = {}
        hazard_token = raw_values[0] if raw_values else None
        
        # Parse the target.
        target, object_id = self._resolve_target(hazard_token, ctx)
        params["target"] = target
        params["target_token"] = hazard_token
        if object_id is not None:
            params["object_id"] = object_id
        
        # Parse the area, preferring peer search or navigate skills for the same robot.
        area_feat = self._extract_area_from_peer_skills(robot_label, ctx)
        
        # If not found, get it from context.
        if area_feat is None:
            area_feat = resolve_area_feature(
                "", ctx.goal_cfg, ctx.area_boundaries, ctx.category_map
            )
        
        if area_feat:
            params["area"] = area_feat
        
        # Runtime parameter fallback.
        self._apply_runtime_object_id(params, robot_label, ctx)
        
        # Ensure target is available.
        self._ensure_target_from_object_id(params)
        
        return params
    
    def _translate_guide(
        self, raw_values: List, robot_label: str, ctx: 'TranslationContext'
    ) -> Dict[str, Any]:
        """
        Translate a guide skill.

        Input: guide<target_token>_to<location_token>
        Output:
            - target: {"class": xxx, "type": xxx, "features": {...}} guided object
            - target_token: raw target semantic parameter
            - object_id: guided object ID, optional
            - dest: {"x": float, "y": float} destination coordinates
            - dest_token: raw destination semantic parameter
            - destination_id: destination entity ID, optional
        """
        params: Dict[str, Any] = {}
        target_token = raw_values[0] if len(raw_values) > 0 else None
        location_token = raw_values[1] if len(raw_values) > 1 else None
        
        # Parse the guided target.
        target, object_id = self._resolve_target(target_token, ctx)
        params["target"] = target
        params["target_token"] = target_token
        if object_id is not None:
            params["object_id"] = object_id
        
        # Runtime parameter fallback for the target.
        self._apply_runtime_object_id(params, robot_label, ctx)
        
        # Ensure target is available.
        self._ensure_target_from_object_id(params)
        
        # Parse the destination.
        params["dest_token"] = location_token
        dest_xy = None
        
        # Parse from an area.
        area_feat = resolve_area_feature(
            location_token, ctx.goal_cfg, ctx.area_boundaries, ctx.category_map
        )
        if area_feat:
            centroid = area_centroid(area_feat)
            if centroid:
                dest_xy = centroid
        
        # Parse from an entity.
        if dest_xy is None:
            norm_tok = normalize_token_by_category_map(location_token, ctx.category_map)
            if norm_tok:
                loc_id = self.label_to_id_map.get(norm_tok)
                if loc_id is not None:
                    params["destination_id"] = loc_id
                    pos = get_entity_position(self.scene_graph, loc_id)
                    if pos:
                        dest_xy = pos
        
        # Write destination coordinates.
        if dest_xy is not None:
            params["dest"] = {"x": float(dest_xy[0]), "y": float(dest_xy[1])}
        
        return params
    
    def _translate_generic(
        self, raw_values: List, ctx: 'TranslationContext'
    ) -> Dict[str, Any]:
        """Generic skill translation fallback."""
        params: Dict[str, Any] = {}
        for value in raw_values:
            if isinstance(value, str) and re.search(r"cybertown", value, flags=re.I):
                feat = resolve_area_feature(
                    value, ctx.goal_cfg, ctx.area_boundaries, ctx.category_map
                )
                params["area"] = feat or value
            elif is_coord(value):
                params["area"] = value
            else:
                lookup_key = value if isinstance(value, str) else str(value)
                key = ctx.category_map.get(lookup_key, "unknown_entity")
                params[key] = value
        return params

    # =========================================================================
    # Helper methods
    # =========================================================================
    
    def _resolve_target(
        self, token: Optional[str], ctx: 'TranslationContext'
    ) -> tuple:
        """
        Parse a target and return (target_dict, object_id).
        """
        return resolve_target_for_skill(
            token, ctx.goal_cfg, self.scene_graph, self.label_to_id_map
        )
    
    def _resolve_surface_target(
        self, surface_token: Optional[str], robot_label: str, ctx: 'TranslationContext'
    ) -> Dict[str, Any]:
        """
        Parse a placement surface target.
        
        Returns:
            {"class": xxx, "type": xxx, "features": {...}}
        """
        if not surface_token:
            return {}
        
        surface_lower = surface_token.lower()
        
        # Load onto a UGV.
        if "ugv" in surface_lower:
            # Find the UGV in the current skill list.
            ugv_label = self._find_ugv_in_timestep_skills(ctx)
            return {
                "class": "robot",
                "type": "UGV",
                "features": {"label": ugv_label} if ugv_label else {}
            }
        
        # Unload to the ground.
        if "ground" in surface_lower:
            return {
                "class": "ground",
                "type": "",
                "features": {}
            }
        
        # Regular placement surface.
        target, _ = self._resolve_target(surface_token, ctx)
        return target if target else {}
    
    def _find_ugv_in_timestep_skills(self, ctx: 'TranslationContext') -> Optional[str]:
        """Find a UGV in the current skill list."""
        for robot_skills in ctx.timestep_skills.values():
            for robot_label in robot_skills.keys():
                if "ugv" in robot_label.lower():
                    return robot_label
        return None
    
    def _extract_area_from_peer_skills(
        self, robot_label: str, ctx: 'TranslationContext'
    ) -> Optional[Dict[str, Any]]:
        """
        Extract area parameters from other search or navigate skills for the same robot.
        
        Iterate over this robot's skills across all timesteps and look for area
        parameters in search or navigate skills.
        """
        # Skill types that can provide an area.
        area_providing_skills = {"search", "navigate", "goto", "go_to"}
        
        for timestep_str, robot_skills in ctx.timestep_skills.items():
            skill_string = robot_skills.get(robot_label)
            if not skill_string:
                continue
            
            # Parse the skill name.
            skill_name = skill_string.split("<", 1)[0].strip().lower()
            if skill_name not in area_providing_skills:
                continue
            
            # Get the skill schema.
            schema = ctx.skill_schemas.get(skill_name)
            if not schema:
                continue
            
            match = schema["pattern"].fullmatch(skill_string.strip())
            if not match:
                continue
            
            # Extract the area token, usually the first parameter.
            raw_values = [v.strip() for v in match.groups()]
            area_token = raw_values[0] if raw_values else None
            
            if area_token:
                area_feat = resolve_area_feature(
                    area_token, ctx.goal_cfg, ctx.area_boundaries, ctx.category_map
                )
                if area_feat:
                    return area_feat
        
        return None
    
    def _is_load_unload_surface(self, surface_token: Optional[str]) -> bool:
        """Determine whether this is a load or unload surface."""
        if not surface_token:
            return False
        surface_lower = surface_token.lower()
        return "ugv" in surface_lower or "ground" in surface_lower
    
    def _add_detection_constraints(self, params: Dict, goal_cfg: Dict) -> None:
        """Add detection threshold parameters."""
        det_constraints = extract_detection_constraints(goal_cfg)
        if det_constraints.get("conf_ge") is not None:
            params["conf_ge"] = det_constraints["conf_ge"]
        if det_constraints.get("persist_ge_s") is not None:
            params["persist_ge_s"] = det_constraints["persist_ge_s"]
    
    def _apply_runtime_object_id(
        self, params: Dict, robot_label: str, ctx: 'TranslationContext',
        alt_keys: Optional[List[str]] = None
    ) -> None:
        """Apply object_id from runtime parameters."""
        if ctx.runtime_params:
            oid = resolve_object_id_from_runtime(
                params, ctx.runtime_params, robot_label, ctx.active_robots, alt_keys
            )
            if oid:
                params["object_id"] = oid
                if alt_keys:
                    for key in alt_keys:
                        params[key] = oid
    
    def _ensure_target_from_object_id(self, params: Dict) -> None:
        """
        Ensure target content exists.
        If target is empty but object_id exists, build target from the scene graph node.
        """
        target = params.get("target")
        object_id = params.get("object_id")
        
        # Return directly if a valid target already exists.
        if isinstance(target, dict) and (target.get("class") or target.get("type")):
            return
        
        # Build target from object_id.
        if object_id is not None:
            node = self.scene_graph.get_node_by_id(object_id)
            if node and isinstance(node, dict):
                props = node.get("properties", {})
                node_type = props.get("type")
                label = props.get("label")
                
                params["target"] = {
                    "class": "object",
                    "type": node_type,
                    "features": {"label": label} if label else {}
                }


class TranslationContext:
    """Translation context that stores all configuration and state needed during translation."""
    
    def __init__(
        self,
        skill_schemas: Dict[str, Any],
        category_map: Dict[str, str],
        goal_cfg: Dict[str, Any],
        area_boundaries: Dict[str, Any],
        runtime_params: Dict[str, Any],
        active_robots: List[str],
        timestep_skills: Dict[str, Dict[str, str]],
    ):
        """
        Initialize the translation context.
        
        Args:
            skill_schemas: Skill schema definitions.
            category_map: Category mapping.
            goal_cfg: Goal configuration.
            area_boundaries: Area boundaries.
            runtime_params: Runtime parameters.
            active_robots: Active robot list.
            timestep_skills: Normalized timestep skills in plain string format.
                {"0": {"UGV_01": "navigate<xxx>", ...}, ...}
        """
        self.skill_schemas = skill_schemas
        self.category_map = category_map
        self.goal_cfg = goal_cfg
        self.area_boundaries = area_boundaries
        self.runtime_params = runtime_params
        self.active_robots = active_robots
        self.timestep_skills = timestep_skills
