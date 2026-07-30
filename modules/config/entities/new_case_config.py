# -*- coding: utf-8 -*-
import copy
import re
import json
import asyncio
import logging
import random
import time
import zlib
from typing import Dict, List, Optional, Any, Union, Tuple

logger = logging.getLogger(__name__)

def stable_id(key: str) -> int:
    """
    Generate a stable 32-bit integer ID based on a string key.
    """
    if not isinstance(key, str):
        key = str(key)
    return zlib.crc32(key.encode("utf-8")) & 0xFFFFFFFF


# ============================================================
# New Case Injection Operation Templates
# ============================================================
NEW_CASE_OP_TEMPLATES: List[Dict[str, Any]] = [
    # --------------------------------------------------------
    # Category 1.1: Environment Traversability
    # --------------------------------------------------------
    {
        "code": "op.block_critical_path",
        "description": "Break one critical edge along the path to the destination.",
        "applicable_skills": ["navigate", "guide"],
        "applicable_robots": ["UGV", "Quadruped", "Humanoid"],
        "category": "environment",
        "type": "traversability_changed",
        "generator_config": {
            "strategy": "block_critical_path",
            "params": {"num_edges_to_remove": 1, "prefer_bridge_like": True}
        }
    },
    {
        "code": "op.set_area_restricted",
        "description": "Set a relevant area node to be restricted (e.g., Keep-Out Zone).",
        "applicable_skills": ["navigate", "guide"],
        "applicable_robots": ["*"],
        "category": "environment",
        "type": "traversability_changed",
        "generator_config": {
            "strategy": "set_area_property", 
            "params": {"property": "passability", "value": "restricted"}
        }
    },
    # --------------------------------------------------------
    # Category 1.2: Environment Perception Properties
    # --------------------------------------------------------
    {
        "code": "op.set_area_low_visibility",
        "description": "Set a relevant area node to have low visibility (fog/smoke).",
        "applicable_skills": ["search", "take_photo"],
        "applicable_robots": ["UAV", "FW_UAV"],
        "category": "environment",
        "type": "perception_degraded",
        "generator_config": {
            "strategy": "set_area_property",
            "params": {"property": "visibility", "value": "low"}
        }
    },
    {
        "code": "op.set_area_dark",
        "description": "Set a relevant area node to be dark.",
        "applicable_skills": ["search", "take_photo"],
        "applicable_robots": ["UAV", "FW_UAV"], 
        "category": "environment",
        "type": "perception_degraded",
        "generator_config": {
            "strategy": "set_area_property",
            "params": {"property": "illumination", "value": "dark"}
        }
    },
    {
        "code": "op.set_area_strong_wind",
        "description": "Set a relevant area node to have strong wind, affecting UAVs.",
        "applicable_skills": ["search", "take_photo", "broadcast", "handle_hazard"],
        "applicable_robots": ["UAV", "FW_UAV"],
        "category": "environment",
        "type": "condition_hazard",
        "generator_config": {
            "strategy": "set_area_property",
            "params": {"property": "wind_condition", "value": "strong"}
        }
    },
    # --------------------------------------------------------
    # Category 1.3: Temporary Dynamic Constraints
    # --------------------------------------------------------
    {
        "code": "op.set_vehicle_congestion",
        "description": "Set path intersection nodes to 'congested' by vehicles.",
        "applicable_skills": ["navigate", "guide"],
        "applicable_robots": ["UGV", "Quadruped", "Humanoid"],
        "category": "environment",
        "type": "temporary_block",
        "generator_config": {
            "strategy": "set_path_congestion",
            "params": {"congestion_type": "vehicle"}
        }
    },
    {
        "code": "op.set_crowd_congestion",
        "description": "Set path intersection nodes to 'congested' by a crowd.",
        "applicable_skills": ["navigate", "guide"],
        "applicable_robots": ["UGV", "Quadruped", "Humanoid"],
        "category": "environment",
        "type": "temporary_block",
        "generator_config": {
            "strategy": "set_path_congestion",
            "params": {"congestion_type": "crowd"}
        }
    },
    {
        "code": "op.add_obstruction_near_target",
        "description": "Add a new cargo node near the target to obstruct view.",
        "applicable_skills": ["take_photo", "follow"],
        "applicable_robots": ["UAV", "Quadruped"],
        "category": "environment",
        "type": "perception_degraded",
        "generator_config": {
            "strategy": "add_obstruction_node",
            "params": {"node_type": "cargo", "status": "obstruction"}
        }
    },

    # --------------------------------------------------------
    # Category 2.1: Interactable Prop State
    # --------------------------------------------------------
    {
        "code": "op.move_target",
        "description": "Move target to another nearby location.",
        "applicable_skills": ["take_photo", "broadcast", "handle_hazard", "guide", "place"],
        "applicable_robots": ["*"],
        "category": "target",
        "type": "target_moved",
        "generator_config": {
            "strategy": "move_target",
            "params": {"prefer_parking_or_staging": True}
        }
    },
    {
        "code": "op.despawn_target",
        "description": "Target disappears (deleted) from the world.",
        "applicable_skills": ["take_photo", "broadcast", "handle_hazard", "guide", "place"], 
        "applicable_robots": ["*"],
        "category": "target",
        "type": "target_missing",
        "generator_config": {
            "strategy": "despawn_target",
            "params": {}
        }
    },
    {
        "code": "op.despawn_carrier",
        "description": "Carrier disappears or becomes unavailable (not found).",
        "applicable_skills": ["place"],
        "applicable_robots": ["Humanoid"], 
        "category": "target",
        "type": "carrier_missing",
        "generator_config": {
            "strategy": "despawn_carrier",
            "params": {"status": "offline"}
        }
    },
    {
        "code": "op.move_carrier", 
        "description": "Move carrier away so it's at a different location.",
        "applicable_skills": ["place"],
        "applicable_robots": ["Humanoid"], 
        "category": "target",
        "type": "carrier_moved",
        "generator_config": {
            "strategy": "move_carrier",
            "params": {"min_hops": 2}
        }
    },
    {
        "code": "op.despawn_surface_object",
        "description": "Placement surface object disappears or becomes unavailable.",
        "applicable_skills": ["place"],
        "applicable_robots": ["Humanoid"], 
        "category": "target",
        "type": "surface_object_missing",
        "generator_config": {
            "strategy": "despawn_surface_object",
            "params": {"status": "offline"}
        }
    },
    {
        "code": "op.move_surface_object", 
        "description": "Move placement surface object away so it's at a different location.",
        "applicable_skills": ["place"], 
        "applicable_robots": ["Humanoid"], 
        "category": "target",
        "type": "surface_object_moved",
        "generator_config": {
            "strategy": "move_surface_object",
            "params": {"min_hops": 2}
        }
    },
    {
        "code": "op.target_disappears",
        "description": "Target disappears from the current location (for follow skill).",
        "applicable_skills": ["follow"],
        "applicable_robots": ["*"],
        "category": "target",
        "type": "state_changed",
        "generator_config": {
            "strategy": "target_disappears",
        }
    },
    
    # --------------------------------------------------------
    # Category 2.2: New Information Discovery
    # --------------------------------------------------------
    {
        "code": "op.discover_new_high_priority_target",
        "description": "Discover a new high-priority target (fire, suspect) in the current area.",
        "applicable_skills": ["search"],
        "applicable_robots": ["UAV", "Quadruped"],
        "category": "target",
        "type": "new_target_discovered",
        "generator_config": {
            "strategy": "discover_new_high_priority_target", 
            "params": {"categories": ["fire", "suspect", "hazard"], "if_not_present": True}
        }
    },

    # --------------------------------------------------------
    # Category 3: Robot
    # --------------------------------------------------------
    {
        "code": "op.degrade_robot_battery",
        "description": "Degrade battery to a low level.",
        "applicable_skills": ["*"],
        "applicable_robots": ["*"],
        "category": "robot",
        "type": "capability_degradation",
        "generator_config": {
            "strategy": "degrade_robot_battery",
            "params": {"new_battery_level": 5}
        }
    },
    {
        "code": "op.set_robot_fault",
        "description": "Set robot hardware fault.",
        "applicable_skills": ["*"],
        "applicable_robots": ["*"],
        "category": "robot",
        "type": "severe_fault",
        "generator_config": {
            "strategy": "set_robot_fault",
            "params": {"status": "error"}
        }
    },
    {
        "code": "op.jam_comm_link",
        "description": "Jam communication link (temporary).",
        "applicable_skills": ["*"],
        "applicable_robots": ["*"],
        "category": "robot",
        "type": "communication_lost",
        "generator_config": {
            "strategy": "jam_comm_link",
            "params": {"comm_status": "jammed"}
        }
    },
]

# Convenience index: code -> template; and numeric id (stable hash) -> code
_OP_CODE_INDEX: Dict[str, Dict[str, Any]] = {op["code"]: op for op in NEW_CASE_OP_TEMPLATES}
_OP_NUM_INDEX: Dict[int, str] = {stable_id(code): code for code in _OP_CODE_INDEX.keys()}

# ============================================================
# New Case Event Templates
# ============================================================
NEW_CASE_EVENT_TEMPLATES: Dict[str, Dict[str, Any]] = {
    # --------------------------------------------------------
    # Category 1.1: Environment Traversability
    # --------------------------------------------------------
    "CRITICAL_PATH_BROKEN": {
        "op_code": "op.block_critical_path",
        "category": "environment", "type": "traversability_changed", "severity": "abort",
        "message": "The direct road from '{robot_location_label}' to '{dest_label}' is permanently blocked and cannot be directly accessed. Detour required.",
        "payload": {"dest_id": "{dest_id}", "dest_label": "{dest_label}", "robot_id": "{robot_id}", "robot_label": "{robot_label}", "robot_location_label": "{robot_location_label}","target_location_label": "{target_location_label}", "skill": "{skill}"}
    },
    "AREA_RESTRICTED": {
        "op_code": "op.set_area_restricted",
        "category": "environment", "type": "traversability_changed", "severity": "abort",
        "message": "The direct path from '{robot_location_label}' to '{dest_label}' crosses a restricted area. Detour required.",
        "payload": {"dest_id": "{dest_id}", "dest_label": "{dest_label}", "robot_id": "{robot_id}", "robot_label": "{robot_label}", "robot_location_label": "{robot_location_label}","target_location_label": "{target_location_label}", "restricted_area_id": "{restricted_area_id}", "restricted_area_label": "{restricted_area_label}", "skill": "{skill}"}
    },
    # --------------------------------------------------------
    # Category 1.2: Environment Perception Properties
    # --------------------------------------------------------
    "AREA_LOW_VISIBILITY": {
        "op_code": "op.set_area_low_visibility",
        "category": "environment", "type": "perception_degraded", "severity": "soft_abort",
        "message": "{robot_label}'s current area '{robot_location_label}' is covered by fog/smoke with low visibility. Task reliability cannot be guaranteed.",
        "payload": {"robot_id": "{robot_id}", "robot_label": "{robot_label}", "robot_location_label": "{robot_location_label}", "curr_area_id": "{curr_area_id}", "curr_area_label": "{curr_area_label}", "skill": "{skill}"}
    },
    "AREA_IS_DARK": {
        "op_code": "op.set_area_dark",
        "category": "environment", "type": "perception_degraded", "severity": "soft_abort",
        "message": "It is dark in the cybertown district. Aerial search or photography tasks (robot: {robot_label}, skill: {skill}) may fail to locate the target.",
        "payload": {"robot_id": "{robot_id}", "robot_label": "{robot_label}", "robot_type": "{robot_type}", "robot_location_label": "{robot_location_label}", "curr_area_id": "{curr_area_id}", "curr_area_label": "{curr_area_label}", "skill": "{skill}"}
    },
    "AREA_STRONG_WIND": {
        "op_code": "op.set_area_strong_wind",
        "category": "environment", "type": "condition_hazard", "severity": "abort",
        "message": "Strong wind detected in the {robot_label}'s current area '{robot_location_label}'. Flying operations are unsafe.",
        "payload": {"robot_id": "{robot_id}", "robot_label": "{robot_label}", "robot_type": "{robot_type}", "robot_location_label": "{robot_location_label}", "curr_area_id": "{curr_area_id}", "curr_area_label": "{curr_area_label}", "skill": "{skill}"}
    },
    # --------------------------------------------------------
    # Category 1.3: Temporary Dynamic Obstacles
    # --------------------------------------------------------
    "PATH_CONGESTED_VEHICLE": { 
        "op_code": "op.set_vehicle_congestion",
        "category": "environment", "type": "temporary_block", "severity": "soft_abort",
        "message": "The direct road from '{robot_location_label}' to '{dest_label or target_location_label}' is temporarily congested with vehicles and cannot be accessed.",
        "payload": {"dest_id": "{dest_id}","dest_label": "{dest_label}","robot_location_label": "{robot_location_label}","target_location_label": "{target_location_label}","congestion_node_id": "{congestion_node_id}","congestion_label": "{congestion_label}", "skill": "{skill}"}
    },
    "PATH_CONGESTED_CROWD": { 
        "op_code": "op.set_crowd_congestion",
        "category": "environment", "type": "temporary_block", "severity": "soft_abort",
        "message": "The direct road from '{robot_location_label}' to '{dest_label or target_location_label}' is crowded with people and temporarily inaccessible.",
        "payload": {"dest_id": "{dest_id}","dest_label": "{dest_label}","robot_location_label": "{robot_location_label}","target_location_label": "{target_location_label}","congestion_node_id": "{congestion_node_id}","congestion_label": "{congestion_label}", "skill": "{skill}"}
    },
    "TARGET_OBSTRUCTED": {
        "op_code": "op.add_obstruction_near_target",
        "category": "environment", "type": "perception_degraded", "severity": "soft_abort",
        "message": "The target '{target_label}' is temporarily blocked by an unknown object, and normal execution cannot proceed.",
        "payload": {"target_id": "{object_id}", "target_label": "{target_label}", "obstruction_id": "{obstruction_id}", "obstruction_label": "{obstruction_label}", "skill": "{skill}"}
    },
    
    # --------------------------------------------------------
    # Category 2.1: Interactable Prop State
    # --------------------------------------------------------
    "TARGET_NOT_EXIST": { # Corresponds to target_exists check
        "op_code": "op.despawn_target",
        "category": "target", "type": "target_missing", "severity": "abort",
        "message": "Target '{target_token or target1_token or target_label}' ({target_label}) could not be found.",
        "payload": {"target_id": "{object_id}", "target_label": "{target_label}", "skill": "{skill}"}
    },
    "TARGET_LOCATION_MISMATCH": { # Corresponds to same_location (for object)
        "op_code": "op.move_target",
        "category": "target", "type": "location_mismatch", "severity": "abort",
        "message": "Target '{target_token or target1_token or target_label}' ({target_label}) was not found in the required location.",
        "payload": {"target_id": "{object_id}", "target_label": "{target_label}", "required_location": "{robot_location}", "actual_location": "{target_location}", "skill": "{skill}"}
    },
    "CARRIER_NOT_FOUND": { # Corresponds to target_exists (for carrier)
        "op_code": "op.despawn_carrier",
        "category": "target", "type": "resource_missing", "severity": "abort",
        "message": "Carrier '{carrier_label}' not found.",
        "payload": {"carrier_id": "{carrier_id}", "skill": "{skill}"}
    },
    "CARRIER_LOCATION_MISMATCH": { # Corresponds to same_location (for carrier)
        "op_code": "op.move_carrier",
        "category": "robot", "type": "location_mismatch", "severity": "abort",
        "message": "Carrier '{carrier_label}' not at required location.",
        "payload": {"carrier_id": "{carrier_id}", "required_location": "{robot_location}", "actual_location": "{carrier_location}", "skill": "{skill}"}
    },
    "SURFACE_OBJECT_MISSING": { # Corresponds to target_exists (for surface)
        "op_code": "op.despawn_surface_object",
        "category": "target", "type": "resource_missing", "severity": "abort",
        "message": "Placement surface '{target2_token or surface_label}' could not be found.",
        "payload": {"surface_id": "{surface_id}", "surface_label": "{surface_label}", "skill": "{skill}"}
    },
    "SURFACE_OBJECT_LOCATION_MISMATCH": { # Corresponds to same_location (for surface)
        "op_code": "op.move_surface_object",
        "category": "target", "type": "location_mismatch", "severity": "abort",
        "message": "Placement surface '{target2_token or surface_label}' was not found in the required location.",
        "payload": {"surface_id": "{surface_id}", "surface_label": "{surface_label}", "required_location": "{robot_location}", "actual_location": "{surface_location}", "skill": "{skill}"}
    },
    "TARGET_DISAPPEARED": { # Follow skill specific
        "op_code": "op.target_disappears",
        "category": "target", "type": "target_lost", "severity": "abort",
        "message": "The followed target '{target_token or target_label}' ({target_label}) has disappeared.",
        "payload": {"target_id": "{object_id}","target_label": "{target_label}", "target_token": "{target_token}", "skill": "{skill}"}
    },
    # --------------------------------------------------------
    # Category 2.2: New Information Discovery
    # --------------------------------------------------------
    "NEW_HIGH_PRIORITY_TARGET_DISCOVERED": {
        "op_code": "op.discover_new_high_priority_target",
        "category": "target", "type": "new_target_discovered", "severity": "info", # Informational only; upper layer decides whether to replan
        "message": "New high-priority target '{hp_target_label}' (type: {hp_target_type}) discovered at '{hp_target_location_label}'.",
        "payload": {"hp_target_id": "{hp_object_id}", "hp_target_label": "{hp_target_label}", "hp_target_type": "{hp_target_type}", "hp_target_location_label": "{hp_target_location_label}", "skill": "{skill}"}
    },
    
    # --------------------------------------------------------
    # Category 3: Robot
    # --------------------------------------------------------
    "ROBOT_BATTERY_LOW": {
        "op_code": "op.degrade_robot_battery",
        "category": "robot", "type": "capability_degraded", "severity": "soft_abort",
        "message": "Robot '{robot_label}' battery is too low.",
        "payload": {"robot_id": "{robot_id}", "robot_label": "{robot_label}", "skill": "{skill}"}
    },
    "ROBOT_FAULT": {
        "op_code": "op.set_robot_fault",
        "category": "robot", "type": "critical_failure", "severity": "abort",
        "message": "Robot '{robot_label}' malfunctioned and cannot operate.",
        "payload": {"robot_id": "{robot_id}", "robot_label": "{robot_label}", "skill": "{skill}"}
    },
    "ROBOT_COMM_JAMMED": {
        "op_code": "op.jam_comm_link",
        "category": "robot", "type": "communication_lost", "severity": "soft_abort",
        "message": "Robot '{robot_label}' lost communication temporarily.",
        "payload": {"robot_id": "{robot_id}", "robot_label": "{robot_label}", "skill": "{skill}"}
    },

    # --------------------------------------------------------
    # Common precheck events
    # --------------------------------------------------------
    "TARGET_STATUS_INCOMPATIBLE": {
        "op_code": None,
        "category": "target", "type": "state_incompatible", "severity": "abort",
        "message": "Target '{target_label}' status '{actual_status}' not in {expected_status}.",
        "payload": {"target_id": "{object_id}", "target_label": "{target_label}", "actual_status": "{actual_status}", "expected_status": "{expected_status}", "skill": "{skill}"}
    },
    "TARGET_TYPE_INCOMPATIBLE": {
        "op_code": None,
        "category": "target", "type": "type_incompatible", "severity": "abort",
        "message": "Target type '{target_type}' not in {allowed_types} for skill '{skill}'.",
        "payload": {"target_id": "{object_id}", "target_type": "{target_type}", "allowed_types": "{allowed_types}", "skill": "{skill}"}
    },
    "ROBOT_NOT_APPLICABLE": {
        "op_code": None,
        "category": "robot", "type": "capability_incompatible", "severity": "abort",
        "message": "Skill '{skill}' not applicable to robot type '{robot_type}'.",
        "payload": {"robot_id": "{robot_id}", "robot_label": "{robot_label}", "robot_type": "{robot_type}", "skill": "{skill}"}
    },
    "SKILL_NOT_FOUND": {
        "op_code": None,
        "category": "system", "type": "definition_missing", "severity": "abort",
        "message": "Skill '{skill}' not found in templates.",
        "payload": {"skill": "{skill}"}
    },
    "ENVIRONMENTAL_HAZARD": {
        "op_code": None,
        "category": "environment", "type": "condition_hazard", "severity": "abort",
        "message": "Environmental hazard detected, preventing skill '{skill}'.",
        "payload": {"skill": "{skill}"}
    }
}

# Build: event string key -> stable numeric id index; and reverse index (numeric id -> key)
_EVENT_KEY_TO_NUM: Dict[str, int] = {k: stable_id(k) for k in NEW_CASE_EVENT_TEMPLATES.keys()}
_EVENT_NUM_INDEX: Dict[int, str] = {v: k for k, v in _EVENT_KEY_TO_NUM.items()}


# ============================================================
# Event Rendering
# ============================================================
def render_event(template_id: Union[str, int],
                 params: Dict[str, Any],
                 context: Dict[str, Any],
                 extras: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    key: Optional[str] = None
    if isinstance(template_id, int):
        key = _EVENT_NUM_INDEX.get(template_id)
        if not key:
            key = "SKILL_NOT_FOUND"
    else:
        key = template_id

    tpl_src = NEW_CASE_EVENT_TEMPLATES.get(key)
    if not tpl_src:
        # Fallback
        key = "SKILL_NOT_FOUND"
        tpl_src = NEW_CASE_EVENT_TEMPLATES[key]

    tpl = copy.deepcopy(tpl_src)

    data: Dict[str, Any] = {}
    data.update(params or {})
    data.update(context or {})
    if extras:
        if isinstance(extras, str):
            data.update({"note": extras})
        elif isinstance(extras, Dict):
            data.update(extras)

    def _fmt(v: Any) -> str:
        if isinstance(v, (dict, list, tuple, set)):
            return json.dumps(v, ensure_ascii=False)
        return "" if v is None else str(v)

    def _sub(x: Any) -> Any:
        if isinstance(x, str):
            def repl(m: re.Match) -> str:
                keys = [k.strip() for k in m.group(1).split(" or ")]
                for kk in keys:
                    val = data.get(kk)
                    if val is not None and val != '':
                        return _fmt(val)
                return m.group(0)
            return re.sub(r"\{([^{}]+(?: or [^{}]+)*)\}", repl, x)
        if isinstance(x, dict):
            return {k2: _sub(v2) for k2, v2 in x.items()}
        if isinstance(x, list):
            return [_sub(v2) for v2 in x]
        return x

    tpl["message"] = _sub(tpl.get("message")) + (f" ({data.get('message', '')})" if data.get('message') else "")
    tpl["payload"] = _sub(tpl.get("payload", {}))
    return tpl