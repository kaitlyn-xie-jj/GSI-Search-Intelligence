import copy
import re
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple

from modules.config.base.enums import SkillName, OutcomeType
from modules.platform.semantic_platform.runtime_guards import GuardRegistry
from modules.config.entities.new_case_config import NEW_CASE_EVENT_TEMPLATES, render_event
from modules.utils.location_utils import extract_object_position
from modules.config.system_config import config

# Robot - common checks (battery, fault, communication)
ROBOT_CHECKS = [
    {"rule": "robot_fault", "on_fail": "ROBOT_FAULT", "severity": "abort"},
    {"rule": "battery_above", "args": {"min_percent": 20.0}, "on_fail": "ROBOT_BATTERY_LOW", "severity": "soft_abort"},
    {"rule": "comm_link", "on_fail": "ROBOT_COMM_JAMMED", "severity": "abort"},
]

# Environment - unified perception checks (fog, darkness, wind)
ENV_PERCEPTION_CHECKS = [
    {"rule": "area_perception_safe", "args": {}, "on_fail": "ENVIRONMENTAL_HAZARD"}
]

# Environment - unified path checks (collapse, restricted zones, congestion)
PATH_CHECKS = [
    {"rule": "path_traversable", "args": {"dest_id": "{destination_id or object_id}", "dest_label": "{destination_label or target_location_label}"}, "on_fail": "CRITICAL_PATH_BROKEN"},
    {"rule": "path_avoids_restricted", "args": {"dest_id": "{destination_id or object_id}", "dest_label": "{destination_label or target_location_label}"}, "on_fail": "AREA_RESTRICTED"},
    {"rule": "path_avoids_congestion", "args": {"dest_id": "{destination_id or object_id}", "dest_label": "{destination_label or target_location_label}"}, "on_fail": "PATH_CONGESTED"},
]

# Environment - line-of-sight checks (obstruction)
LOS_CHECKS = [
    {"rule": "target_line_of_sight_clear", "args": {"object_id": "{object_id}"}, "on_fail": "TARGET_OBSTRUCTED"}
]

# Unified target checks
TARGET_EXISTS = [ 
    {"rule": "target_exists", "args": {"object_id": "{object_id}"}, "on_fail": "TARGET_NOT_EXIST"}
]
SURFACE_EXISTS_IF_PARAM = [
    {"rule": "target_exists", "args": {"object_id": "{surface_id}", "if_param": "surface_id"}, "on_fail": "SURFACE_OBJECT_MISSING"}
]
CARRIER_EXISTS_IF_PARAM = [
    {"rule": "target_exists", "args": {"object_id": "{carrier_id}", "if_param": "carrier_id"}, "on_fail": "CARRIER_NOT_FOUND"}
]

# Unified location checks
ROBOT_AT_TARGET_LOCATION = [ 
    {"rule": "same_location", "args": {"left": "robot_location", "right": "target_location"}, "on_fail": "TARGET_LOCATION_MISMATCH"}
]
ROBOT_AT_TARGET_LOCATION_IF_PARAM = [
    {"rule": "same_location", "args": {"left": "robot_location", "right": "target_location", "if_param": "object_id"}, "on_fail": "TARGET_LOCATION_MISMATCH"}
]
ROBOT_AT_SURFACE_LOCATION_IF_PARAM = [
    {"rule": "same_location", "args": {"left": "robot_location", "right": "surface_location", "if_param": "surface_id"}, "on_fail": "SURFACE_OBJECT_LOCATION_MISMATCH"}
]
ROBOT_AT_CARRIER_LOCATION_IF_PARAM = [
    {"rule": "same_location", "args": {"left": "robot_location", "right": "carrier_location", "if_param": "carrier_id"}, "on_fail": "CARRIER_LOCATION_MISMATCH"}
]

DYNAMIC_RULE_NAMES = {
    # ROBOT_CHECKS
    "robot_fault",
    "battery_above",
    "comm_link",
    # ENV_PERCEPTION_CHECKS
    "area_perception_safe",
    # PATH_CHECKS
    "path_traversable",
    "path_avoids_restricted",
    "path_avoids_congestion",
    # LOS_CHECKS
    "target_line_of_sight_clear",
    # Discovery
    "check_for_new_target",
}

SKILL_TEMPLATES = {
    SkillName.TAKE_OFF.value: {
        "parameters": {},
        "enable_dynamic_checks": True,
        "precheck": [
            {"rule": "robot_type_in", "args": {"allowed": ["UAV", "FW_UAV"]}, "on_fail": "ROBOT_NOT_APPLICABLE"},
            *ROBOT_CHECKS,
            *ENV_PERCEPTION_CHECKS,
        ],
        "effects": {
            "success": {"robot": {"UAV": {"node": {"status": "working", "altitude": 50}}, "FW_UAV": {"node": {"status": "working", "altitude": 50}}}},
            "failure": {}
        },
        "outcomes": {},
        "base_time": 2.0,
        "base_energy": 1.0
    },
    SkillName.RETURN_HOME.value: {
        "parameters": {},
        "enable_dynamic_checks": True,
        "precheck": [
            {"rule": "robot_type_in", "args": {"allowed": ["UAV", "FW_UAV"]}, "on_fail": "ROBOT_NOT_APPLICABLE"},
            *ROBOT_CHECKS,
            *ENV_PERCEPTION_CHECKS,
        ],
        "effects": {
            "success": {"robot": {"UAV": {"node": {"status": "idle", "location": "home_base", "altitude": 0}}, "FW_UAV": {"node": {"status": "idle", "location": "home_base", "altitude": 0}}}},
            "failure": {}
        },
        "outcomes": {},
        "base_time": 3.0,
        "base_energy": 1.5
    },
    SkillName.LAND.value: {
        "parameters": {},
        "enable_dynamic_checks": True,
        "precheck": [
            {"rule": "robot_type_in", "args": {"allowed": ["UAV", "FW_UAV"]}, "on_fail": "ROBOT_NOT_APPLICABLE"},
            *ROBOT_CHECKS,
            *ENV_PERCEPTION_CHECKS,
        ],
        "effects": {
            "success": {"robot": {"UAV": {"node": {"status": "idle", "altitude": 0}}, "FW_UAV": {"node": {"status": "idle", "altitude": 0}}}},
            "failure": {}
        },
        "outcomes": {},
        "base_time": 2.0,
        "base_energy": 0.7
    },
    SkillName.NAVIGATE.value: {
        "parameters": {"object_id": "str"},
        "enable_dynamic_checks": True,
        "precheck": [
            *ROBOT_CHECKS,
            *PATH_CHECKS,
            *ENV_PERCEPTION_CHECKS, 
        ],
        "effects": {
            "success": {
                "robot": {
                    "UAV": {"node": {"status": "working", "location": "{object_id}"}, "edge": [{"action": "update", "type": "stationed_at", "from": "robot", "to": "{object_id}"}]},
                    "UGV": {"node": {"status": "idle", "location": "{object_id}"}, "edge": [{"action": "update", "type": "stationed_at", "from": "robot", "to": "{object_id}"}]},
                    "Quadruped": {"node": {"status": "idle", "location": "{object_id}"}, "edge": [{"action": "update", "type": "stationed_at", "from": "robot", "to": "{object_id}"}]},
                    "Humanoid": {"node": {"status": "idle", "location": "{object_id}"}, "edge": [{"action": "update", "type": "stationed_at", "from": "robot", "to": "{object_id}"}]}
                },
                "carried_objects": {"node": {"location": "{object_id}"}}
            },
            "failure": {}
        },
        "outcomes": {},
        "base_time": 1.0,
        "base_energy": 1.0
    },
    SkillName.BROADCAST.value: {
        "parameters": {"message": "str", "mode": "str"},
        "enable_dynamic_checks": True,
        "precheck": [
            {"rule": "robot_type_in", "args": {"allowed": ["UAV", "UGV", "Quadruped"]}, "on_fail": "ROBOT_NOT_APPLICABLE"},
            *ROBOT_CHECKS,
            *ENV_PERCEPTION_CHECKS,
            *TARGET_EXISTS,
            *ROBOT_AT_TARGET_LOCATION_IF_PARAM,
        ],
        "effects": {
            "success": {"robot": {"UAV": {"node": {"status": "working"}}}, "UGV": {"node": {"status": "working"}}, "Quadruped": {"node": {"status": "working"}}},
            "failure": {}
        },
        "outcomes": {
            "success": [{"type": OutcomeType.KNOWLEDGE_ACQUIRED.value, "data": {"knowledge_type": "broadcast_event", "robot_id": "{robot_id}", "message_text": "{message}"}}]
        },
        "base_time": 5.0,
        "base_energy": 1
    },
    SkillName.TAKE_PHOTO.value: {
        "parameters": {"object_id": "str"},
        "enable_dynamic_checks": True,
        "precheck": [
            {"rule": "robot_type_in", "args": {"allowed": ["UAV", "Quadruped"]}, "on_fail": "ROBOT_NOT_APPLICABLE"},
            *ROBOT_CHECKS,
            *ENV_PERCEPTION_CHECKS,
            *TARGET_EXISTS,
            *ROBOT_AT_TARGET_LOCATION_IF_PARAM,
            *LOS_CHECKS, 
        ],
        "effects": {
            "success": {
                "robot": {"UAV": {"node": {"status": "working"}}, "Quadruped": {"node": {"status": "idle"}}},
                "target": {"node": {"status": "discovered"}}
            },
            "failure": {}
        },
        "outcomes": {
            "success": [
                {"type": OutcomeType.KNOWLEDGE_ACQUIRED.value, "data": {"knowledge_type": "photo", "target_id": "{object_id}", "target_type": "{target_type}", "robot_id": "{robot_id}", "photo_metadata": {"timestamp": "{current_time}", "location": "{robot_location}", "photo_type": "{photo_type}"}}},
            ]
        },
        "base_time": 1.0,
        "base_energy": 0.2
    },
    SkillName.HANDLE_HAZARD.value: {
        "parameters": {"object_id": "str"},
        "enable_dynamic_checks": True,
        "precheck": [
            {"rule": "robot_type_in", "args": {"allowed": ["UAV"]}, "on_fail": "ROBOT_NOT_APPLICABLE"},
            *ROBOT_CHECKS,
            *ENV_PERCEPTION_CHECKS,
            *TARGET_EXISTS,
            *ROBOT_AT_TARGET_LOCATION,
        ],
        "effects": {
            "success": {
                "robot": {"UAV": {"node": {"status": "idle"}}},
                "target": {"node": {"is_fire": False, "is_spill": False, "is_handled": True}}
            },
            "failure": {}
        },
        "outcomes": {
            "success": [{
                "type": OutcomeType.KNOWLEDGE_ACQUIRED.value, "data": {"knowledge_type": "handle_log", "robot_id": "{robot_id}", "hazard_id": "{object_id}", "result": "hazard_mitigated"}
            }]
        },
        "base_time": 4.0,
        "base_energy": 1.8
    },
    SkillName.FOLLOW.value: {
        "parameters": {"object_id": "str"},
        "enable_dynamic_checks": True,
        "precheck": [
            {"rule": "robot_type_in", "args": {"allowed": ["Quadruped", "UAV"]}, "on_fail": "ROBOT_NOT_APPLICABLE"},
            *ROBOT_CHECKS,
            *ENV_PERCEPTION_CHECKS,
            *TARGET_EXISTS,
            *LOS_CHECKS, 
            {"rule": "target_status_discovered", "args": {"object_id": "{object_id}"}, "on_fail": "TARGET_DISAPPEARED", "severity": "abort"},
        ],
        "effects": {
            "success": {
                "robot": {"UAV": {"node": {"status": "working"}, "edge": [{"action": "add", "type": "following", "from": "robot", "to": "{object_id}"}]},
                          "Quadruped": {"node": {"status": "working"}, "edge": [{"action": "add", "type": "following", "from": "robot", "to": "{object_id}"}]}},
                "target": {"node": {"status": "followed"}}
            },
            "failure": {}
        },
        "outcomes": {
            "success": [
                {"type": OutcomeType.KNOWLEDGE_ACQUIRED.value, "data": {"knowledge_type": "following_data", "target_id": "{object_id}", "robot_id": "{robot_id}"}},
            ]
        },
        "base_time": 7.0,
        "base_energy": 1.2
    },
    SkillName.GUIDE.value: {
        "parameters": {"object_id": "str", "destination_id": "str"},
        "enable_dynamic_checks": True,
        "precheck": [
            {"rule": "robot_type_in", "args": {"allowed": ["Humanoid", "Quadruped"]}, "on_fail": "ROBOT_NOT_APPLICABLE"},
            *ROBOT_CHECKS,
            *PATH_CHECKS,
            *TARGET_EXISTS,
            *ROBOT_AT_TARGET_LOCATION,
        ],
        "effects": {
            "success": {
                "robot": {"Quadruped": {"node": {"status": "idle", "location": "{destination_id}"}, "edge": [{"action": "update", "type": "stationed_at", "from": "robot", "to": "{destination_id}"}]}},
            },
            "failure": {}
        },
        "outcomes": {
            "success": [
                {"type": OutcomeType.KNOWLEDGE_ACQUIRED.value,"data": {"knowledge_type": "guide_log", "robot_id": "{robot_id}", "guided_entity_id": "{object_id}", "dest_id": "{destination_id}", "status": "completed"}}
            ]
        },
        "base_time": 3.5,
        "base_energy": 1.8
    },
    SkillName.SEARCH.value: {
        "parameters": {"area": "str"},
        "enable_dynamic_checks": True,
        "precheck": [
            {"rule": "robot_type_in", "args": {"allowed": ["UAV", "FW_UAV", "Quadruped"]}, "on_fail": "ROBOT_NOT_APPLICABLE"},
            *ROBOT_CHECKS,
            *ENV_PERCEPTION_CHECKS,
            {"rule": "check_for_new_target", "args": {"area_id": "{area}"}, "on_info": "NEW_HIGH_PRIORITY_TARGET_DISCOVERED"}
        ],
        "effects": {
            "success": {"robot": {"UAV": {"node": {"status": "working"}}, "FW_UAV": {"node": {"status": "working"}}, "Quadruped": {"node": {"status": "working"}}}, "target": {"node": {"status": "discovered"}}},
            "failure": {}
        },
        "outcomes": {
            "success": [{"type": OutcomeType.KNOWLEDGE_ACQUIRED.value, "data": {"knowledge_type": "entity_discovery", "robot_id": "{robot_id}", "area_searched": "{area_id}", "entities": "{discovered_entities}"}}]
        },
        "base_time": 5.0,
        "base_energy": 1.5
    },
    SkillName.PLACE.value: {
        "parameters": {"object_id": "str", "surface_id": "str", "carrier_id": "str"},
        "enable_dynamic_checks": True,
        "precheck": [
            {"rule": "robot_type_in", "args": {"allowed": ["Humanoid"]}, "on_fail": "ROBOT_NOT_APPLICABLE"},
            *ROBOT_CHECKS,
            *TARGET_EXISTS,
            *ROBOT_AT_TARGET_LOCATION,
            *SURFACE_EXISTS_IF_PARAM,
            *ROBOT_AT_SURFACE_LOCATION_IF_PARAM,
            *CARRIER_EXISTS_IF_PARAM,
            *ROBOT_AT_CARRIER_LOCATION_IF_PARAM,
        ],
        "effects": {
            "success": {
                "robot": {"Humanoid": {"node": {"status": "idle"}}},
                "options": {
                    "load": {
                        "target": {"node": {"status": "loaded"}, "edge": [{"action": "remove", "type": "stored_at", "from": "{object_id}", "to": "{old_location}"}]},
                        "carrier": {"edge": [{"action": "add", "type": "carrying", "from": "{carrier_id}", "to": "{object_id}"}]}
                    },
                    "unload": {
                        "target": {"node": {"status": "unloaded", "location": "{robot_location}"}, "edge": [{"action": "add", "type": "stored_at", "from": "{object_id}", "to": "{robot_location}"}]},
                        "carrier": {"edge": [{"action": "remove", "type": "carrying", "from": "{carrier_id}", "to": "{object_id}"}]}
                    },
                    "assembly": {
                        "target": {"node": {"is_installed": True, "parent_component": "{surface_subtype}"}}
                    }
                }
            },
            "failure": {}
        },
        "outcomes": {
            "success": [
                {"type": OutcomeType.KNOWLEDGE_ACQUIRED.value, "data": {"knowledge_type": "place_log", "robot_id": "{robot_id}", "object_placed_id": "{object_id}", "surface_id": "{surface_id}", "location": "{robot_location}"}}
            ]
        },
        "base_time": 1.5,
        "base_energy": 1.0
    },
}


class SkillTemplateManager:
    """Unified skill template manager."""

    # ----------------- Basic Access -----------------
    @staticmethod
    def get_template(skill: str) -> Optional[Dict]:
        return SKILL_TEMPLATES.get(skill)

    @staticmethod
    def get_parameters(skill: str) -> Dict[str, str]:
        template = SKILL_TEMPLATES.get(skill, {})
        return template.get("parameters", {})
    
    @staticmethod
    def get_dynamic_check_setting(skill: str) -> bool:
        """Get whether the specified skill has dynamic checks enabled (default is True)."""
        template = SKILL_TEMPLATES.get(skill, {})
        return template.get("enable_dynamic_checks", True)
    
    @staticmethod
    def get_effects(skill: str, outcome: str, robot_type: str, params: Dict, target_type: Optional[str] = None) -> Dict:
        """Get skill effects."""
        template = SKILL_TEMPLATES.get(skill, {})
        effects = copy.deepcopy(template.get("effects", {}).get(outcome, {}))
        if "options" in effects and skill == SkillName.PLACE.value:
            options = effects.pop("options")  # Extract and remove the options block
            
            # Parse surface_target: supports structured format and legacy string format
            surface_target = params.get('surface_target', {})
            if isinstance(surface_target, dict):
                surface_class = surface_target.get('class', '')
            else:
                # Backward compatible with legacy string format
                surface_class = 'robot' if surface_target == 'ugv' else ('ground' if surface_target == 'ground' else '')
            
            selected_effects = {}
            if surface_class == 'robot':
                selected_effects = options.get("load", {})
            elif surface_class == 'ground':
                selected_effects = options.get("unload", {})
            else: 
                # Default case or when surface_id is specified, treat as assembly
                selected_effects = options.get("assembly", {})
            for key, value in selected_effects.items():
                effects.setdefault(key, {}).update(value)
        
        result = {}
        
        # Robot effects
        robot_effects = effects.get("robot", {})
        if robot_type in robot_effects:
            result["robot"] = robot_effects[robot_type]
        elif "all" in robot_effects:
            result["robot"] = robot_effects["all"]
        
        # Target effects - dynamically adjusted based on target type
        if "target" in effects:
            target_effects = effects["target"].copy()
            
            # Special handling: for equipment_failure type, certain skills require special status
            if target_type == "equipment_failure" and skill == SkillName.TAKE_PHOTO.value:
                # For equipment_failure, take_photo should set status to resolved instead of discovered
                if "node" in target_effects and "status" in target_effects["node"]:
                    target_effects["node"]["status"] = "resolved"
            
            result["target"] = target_effects
        
        # Carried objects effects
        if "carried_objects" in effects:
            result["carried_objects"] = effects["carried_objects"]

        # Carrier effects
        if "carrier" in effects:
            result["carrier"] = effects["carrier"]
        
        return result

    @staticmethod
    def get_execution_info(skill: str) -> Dict[str, float]:
        template = SKILL_TEMPLATES.get(skill, {})
        return {
            "base_time": template.get("base_time", 1.0),
            "base_energy": template.get("base_energy", 1.0)
        }

    # ----------------- Unified Precheck -----------------
    @staticmethod
    def evaluate_precheck(robot: Dict, skill: str, params: Dict, context: Dict,
                        check_dynamic_conditions: bool = True) -> Tuple[str, Optional[str], Optional[Dict]]:
        """
        Evaluate precheck rules.
        Returns:
            ("pass", None, None) -> All rules passed
            ("fail", reason_code, payload) ->
                - immediate mode: payload is a single event (backward compatible)
                - aggregate mode: payload is {"primary_event": ..., "all_events": [...]} structure
            ("info", reason_code, payload) -> Same as above
        """
        template = SKILL_TEMPLATES.get(skill)
        if not template:
            event = render_event("SKILL_NOT_FOUND", params, context, {"skill": skill})
            return "fail", "skill_not_found", event

        rules: List[Dict] = list(template.get("precheck") or [])
        _augment_context_for_precheck(context, params)
        mode = config.new_case_mode or "immediate"

        # ---------- 1) Immediate mode: keep existing short-circuit behavior ----------
        if mode == "immediate":
            info_event: Optional[Tuple[str, Optional[Dict]]] = None

            for rule in rules:
                status, reason, extras = SkillTemplateManager._eval_rule(
                    rule, robot, params, context, check_dynamic_conditions
                )

                if status == "fail":
                    event_key = reason or rule.get("on_fail", "PRECHECK_GUARD_FAILED")
                    event = render_event(event_key, params, context, extras or {})
                    return "fail", event_key or "precheck_failed", event

                if status == "info" and not info_event:
                    event_key = reason or rule.get("on_info", "PRECHECK_INFO")
                    event = render_event(event_key, params, context, extras or {})
                    info_event = (event_key or "info_trigger", event)

            if info_event:
                return "info", info_event[0], info_event[1]

            return "pass", None, None

        # ---------- 2) Aggregate mode: collect all fail/info results ----------
        fail_events: List[Dict[str, Any]] = []
        info_events: List[Dict[str, Any]] = []

        for rule in rules:
            status, reason, extras = SkillTemplateManager._eval_rule(
                rule, robot, params, context, check_dynamic_conditions
            )
            rule_name = rule.get("rule")

            if status == "fail":
                event_key = reason or rule.get("on_fail", "PRECHECK_GUARD_FAILED")
                event = render_event(event_key, params, context, extras or {})
                fail_events.append({
                    "kind": "fail",
                    "event_key": event_key,
                    "event": event,
                    "rule": rule_name,
                })
            elif status == "info":
                event_key = reason or rule.get("on_info", "PRECHECK_INFO")
                event = render_event(event_key, params, context, extras or {})
                info_events.append({
                    "kind": "info",
                    "event_key": event_key,
                    "event": event,
                    "rule": rule_name,
                })

        # Check fail first, then info
        if fail_events:
            primary = fail_events[0]
            details = {
                "primary_event": primary["event"],
                "all_events": fail_events + info_events,  # Failures first, info second
            }
            return "fail", primary["event_key"] or "precheck_failed", details

        if info_events:
            primary = info_events[0]
            details = {
                "primary_event": primary["event"],
                "all_events": info_events,
            }
            return "info", primary["event_key"] or "info_trigger", details

        return "pass", None, None

    @staticmethod
    def _eval_rule(rule: Dict, robot: Dict, params: Dict, context: Dict, check_dynamic_conditions: bool = True) -> Tuple[str, Optional[str], Optional[Dict]]:
        """Evaluate a single rule, returns (status: 'pass'|'fail'|'info', reason_code, details)."""
        name = rule.get("rule")
        if name in DYNAMIC_RULE_NAMES and not check_dynamic_conditions:
            return "pass", None, None # Skip dynamic checks
        if name == "check_for_new_target" and not params.get("check_for_new_target", False):
            return "pass", None, None # Skip dynamic checks
        args = resolve_args(rule.get("args") or {}, params, context)
        robot_props = robot.get("properties", {}) if robot else {}
        registry = GuardRegistry(context.get("graph"))

        # ENV
        if name == "path_traversable":
            ok, reason, detail = registry.eval("path_traversable", args, context)
            if not ok:
                if rule.get("on_info"):
                    return "info", reason or rule.get("on_info"), detail
                return "fail", reason or rule.get("on_fail") or "CRITICAL_PATH_BROKEN", detail
            return "pass", None, None

        # Unified environment perception checks
        if name == "area_perception_safe":
            ok, reason, detail = registry.eval("area_perception_safe", args, context)
            if not ok:
                if rule.get("on_info"):
                    return "info", reason or rule.get("on_info"), detail
                return "fail", reason or rule.get("on_fail") or "ENVIRONMENTAL_HAZARD", detail
            return "pass", None, None
        
        # Unified path constraint checks
        if name == "path_avoids_restricted":
            ok, reason, detail = registry.eval("path_avoids_restricted", args, context)
            if not ok:
                if rule.get("on_info"):
                    return "info", reason or rule.get("on_info"), detail
                return "fail", reason or rule.get("on_fail") or "AREA_RESTRICTED", detail
            return "pass", None, None
            
        if name == "path_avoids_congestion":
            ok, reason, detail = registry.eval("path_avoids_congestion", args, context)
            if not ok:
                if rule.get("on_info"):
                    return "info", reason or rule.get("on_info"), detail
                return "fail", reason or rule.get("on_fail") or "PATH_CONGESTED", detail
            return "pass", None, None
            
        # Line-of-sight checks
        if name == "target_line_of_sight_clear":
            ok, reason, detail = registry.eval("target_line_of_sight_clear", args, context)
            if not ok:
                if rule.get("on_info"):
                    return "info", reason or rule.get("on_info"), detail
                return "fail", reason or rule.get("on_fail") or "TARGET_OBSTRUCTED", detail
            return "pass", None, None
        
        if name == "check_for_new_target":
            ok, reason, detail = registry.eval("check_for_new_target", args, context)
            if not ok: # Triggered
                return "info", reason or rule.get("on_info"), detail
            return "pass", None, None

        # ROBOT
        if name == "robot_type_in":
            allowed = set(args.get("allowed") or [])
            rtype = robot_props.get("type")
            if rtype not in allowed:
                detail = {"robot_type": rtype, "skill": context.get("skill") or params.get("skill")}
                if rule.get("on_info"):
                    return "info", rule.get("on_info"), detail
                return "fail", rule.get("on_fail") or "ROBOT_NOT_APPLICABLE", detail
            return "pass", None, None

        if name == "robot_fault":
            ok, reason, detail = registry.eval("robot_fault", {}, context)
            if not ok:
                if rule.get("on_info"):
                    return "info", reason or rule.get("on_info"), detail
                return "fail", reason or rule.get("on_fail") or "ROBOT_FAULT", {"message": detail}
            return "pass", None, None

        if name == "battery_above":
            ok, reason, detail = registry.eval("battery_above", {"min_percent": float(args.get("min_percent", 0))}, context)
            if not ok:
                if rule.get("on_info"):
                    return "info", reason or rule.get("on_info"), detail
                return "fail", reason or rule.get("on_fail") or "ROBOT_BATTERY_LOW", {"min_percent": args.get("min_percent"), "message": detail}
            return "pass", None, None
        
        if name == "comm_link":
            ok, reason, detail = registry.eval("comm_link", {}, context)
            if not ok:
                if rule.get("on_info"):
                    return "info", reason or rule.get("on_info"), detail
                return "fail", reason or rule.get("on_fail") or "ROBOT_COMM_JAMMED", {"message": detail}
            return "pass", None, None

        # TARGET
        if name == "target_exists":
            if args.get("if_param") and not params.get(args["if_param"]):
                return "pass", None, None
            ok, reason, detail = registry.eval("target_exists", args, context)
            if not ok:
                if rule.get("on_info"):
                    return "info", reason or rule.get("on_info"), detail
                return "fail", reason or rule.get("on_fail") or "TARGET_NOT_EXIST", detail
            return "pass", None, None

        if name == "target_status_in":
            node = context.get("target_node")
            allowed = set(args.get("allowed") or [])
            cur = (node or {}).get("properties", {}).get("status")
            if cur not in allowed:
                detail = {"object_id": context.get("object_id"), "target_label": context.get("target_label"), "actual_status": cur, "expected_status": list(allowed)}
                if rule.get("on_info"):
                    return "info", rule.get("on_info"), detail
                return "fail", rule.get("on_fail") or "TARGET_STATUS_INCOMPATIBLE", detail
            return "pass", None, None
        
        if name == "target_category_in":
            node = context.get("target_node")
            allowed = set(args.get("allowed") or [])
            cat = (node or {}).get("properties", {}).get("category")
            if cat not in allowed:
                detail = {"object_id": context.get("object_id"), "target_label": context.get("target_label"), "target_type": cat, "allowed_types": list(allowed), "skill": context.get("skill") or params.get("skill")}
                if rule.get("on_info"):
                    return "info", rule.get("on_info"), detail
                return "fail", rule.get("on_fail") or "TARGET_TYPE_INCOMPATIBLE", detail
            return "pass", None, None
        
        if name == "target_status_discovered":
            ok, reason, detail = registry.eval("target_status_discovered", {"object_id": args.get("object_id")}, context)
            if not ok:
                detail_payload = {"object_id": context.get("object_id"), "target_label": context.get("target_label"), "message": detail}
                if rule.get("on_info"):
                    return "info", reason or rule.get("on_info"), detail_payload
                return "fail", reason or rule.get("on_fail") or "TARGET_DISAPPEARED", detail_payload
            return "pass", None, None

        # Location 
        if name == "same_location":
            A = rule.get("args") or {}
            if A.get("if_param") and not params.get(A["if_param"]): 
                return "pass", None, None
                
            left  = pull_ctx(A.get("left"),  context, params)
            right = pull_ctx(A.get("right"), context, params)
            if left == right:
                return "pass", None, None
                
            g = context.get("graph")
            robot_pos = extract_object_position(context.get("robot") or {})
            def _get_node(node_key: str, id_key: str):
                node = context.get(node_key)
                if node is None and g and params.get(id_key) is not None:
                    node = g.get_node_by_id(params[id_key])
                return node
            
            def _near(node, payload, reason_code: str):
                if node is None:
                    payload["note"] = "missing_position_info"
                    if rule.get("on_info"):
                        return "info", rule.get("on_info"), payload
                    return "fail", reason_code, payload
                tgt_pos = extract_object_position(node)
                if robot_pos is not None and tgt_pos is not None:
                    dist = np.linalg.norm(np.array(robot_pos) - np.array(tgt_pos))
                    if dist <= 100:
                        return "pass", None, None
                    payload["distance"] = dist
                    if rule.get("on_info"):
                        return "info", rule.get("on_info"), payload
                    return "fail", reason_code, payload
                payload["note"] = "missing_position_info"
                if rule.get("on_info"):
                    return "info", rule.get("on_info"), payload
                return "fail", reason_code, payload
            
            if A.get("right") == "target_location":
                node = _get_node("target_node", "object_id")
                return _near(node, {"robot_location": left, "required_location": right, "actual_location": context.get("target_location"), "object_id": params.get("object_id")}, rule.get("on_fail") or "TARGET_LOCATION_MISMATCH")
            elif A.get("right") == "carrier_location":
                node = _get_node("carrier_node", "carrier_id")
                return _near(node, {"robot_location": left, "required_location": right, "actual_location": context.get("carrier_location"), "carrier_id": params.get("carrier_id")}, rule.get("on_fail") or "CARRIER_LOCATION_MISMATCH")
            elif A.get("right") == "surface_location":
                node = _get_node("surface_node", "surface_id")
                return _near(node, {"robot_location": left, "required_location": right, "actual_location": context.get("surface_location"), "surface_id": params.get("surface_id")}, rule.get("on_fail") or "SURFACE_OBJECT_LOCATION_MISMATCH")
            
            # Fallback
            detail_payload = {"robot_location": left, "required_location": right, "note": "unknown location type in rule"}
            if rule.get("on_info"):
                return "info", rule.get("on_info"), detail_payload
            return "fail", rule.get("on_fail") or "TARGET_LOCATION_MISMATCH", detail_payload

        # Default pass
        return "pass", None, None

    # ----------------- Effects and Outcomes -----------------
    @staticmethod
    def apply_effects(robot: Dict, skill: str, outcome: str, params: Dict, context: Dict) -> List[Dict]:
        """Apply skill effects."""
        robot_type = robot.get("properties", {}).get("type")
        target_type = None
        if "target_node" in context and context["target_node"]:
            target_type = context["target_node"].get("properties", {}).get("type")

        effects = SkillTemplateManager.get_effects(skill, outcome, robot_type, params, target_type)
        if not effects:
            return []

        operations = []

        # ---- Robot effects ----
        if "robot" in effects:
            robot_effects = effects["robot"]

            if "node" in robot_effects:
                updates, missing = resolve_placeholders_strict(robot_effects["node"], params, context)
                if updates is not None:
                    operations.append({
                        "type": "update_node",
                        "target_id": robot["id"],
                        "updates": updates
                    })

            if "edge" in robot_effects:
                for edge_op in robot_effects["edge"]:
                    resolved_op, missing = resolve_placeholders_strict(edge_op, params, context)
                    if resolved_op is None:
                        continue
                    operations.append({
                        "type": f"{resolved_op['action']}_edge",
                        "edge_type": resolved_op["type"],
                        "from": resolved_op["from"],
                        "to": resolved_op["to"]
                    })

        # ---- Target effects ----
        if "target" in effects:
            target_effects = effects["target"]
            target_ids = params.get("target_ids", [])
            if not target_ids:
                single_id = params.get("object_id") or context.get("object_id")
                target_ids = [single_id] if single_id else []

            for target_id in target_ids:
                if "node" in target_effects:
                    temp_params = params.copy()
                    temp_params["object_id"] = target_id
                    updates, missing = resolve_placeholders_strict(target_effects["node"], temp_params, context)
                    if updates is not None:
                        operations.append({
                            "type": "update_node",
                            "target_id": target_id,
                            "updates": updates
                        })

                if "edge" in target_effects:
                    for edge_op in target_effects["edge"]:
                        temp_params = params.copy()
                        temp_params["object_id"] = target_id
                        resolved_op, missing = resolve_placeholders_strict(edge_op, temp_params, context)
                        if resolved_op is None:
                            continue
                        operations.append({
                            "type": f"{resolved_op['action']}_edge",
                            "edge_type": resolved_op["type"],
                            "from": resolved_op["from"],
                            "to": resolved_op["to"]
                        })

        # ---- Carried objects effects ----
        if "carried_objects" in effects:
            carried_effects = effects["carried_objects"]
            carried_ids = context.get("carried_object_ids", [])
            for obj_id in carried_ids:
                if "node" in carried_effects:
                    updates, missing = resolve_placeholders_strict(carried_effects["node"], params, context)
                    if updates is not None:
                        operations.append({
                            "type": "update_node",
                            "target_id": obj_id,
                            "updates": updates
                        })

        # ---- Carrier effects ----
        if "carrier" in effects and context.get("carrier_id"):
            carrier_effects = effects["carrier"]
            carrier_id = context["carrier_id"]

            if "node" in carrier_effects:
                updates, missing = resolve_placeholders_strict(carrier_effects["node"], params, context)
                if updates is not None:
                    operations.append({
                        "type": "update_node",
                        "target_id": carrier_id,
                        "updates": updates
                    })

            if "edge" in carrier_effects:
                for edge_op in carrier_effects["edge"]:
                    resolved_op, missing = resolve_placeholders_strict(edge_op, params, context)
                    if resolved_op is None:
                        continue
                    operations.append({
                        "type": f"{resolved_op['action']}_edge",
                        "edge_type": resolved_op["type"],
                        "from": resolved_op["from"],
                        "to": resolved_op["to"]
                    })

        return operations
    
    @staticmethod
    def get_outcomes(skill: str, outcome: str, target_type: Optional[str] = None) -> List[Dict]:
        """Get skill execution outcomes (supports adjustment based on target type)."""
        template = SKILL_TEMPLATES.get(skill, {})
        outcomes = copy.deepcopy(template.get("outcomes", {}).get(outcome, []))  # Deep copy
        
        # Special rule: TAKE_PHOTO handling for equipment_failure
        if skill == SkillName.TAKE_PHOTO.value and target_type == "equipment_failure":
            for outcome_item in outcomes:
                if outcome_item.get("type") == OutcomeType.ENTITY_STATE_CHANGED.value:
                    # Change discovered to resolved
                    if outcome_item.get("data", {}).get("new_value") == "discovered":
                        outcome_item["data"]["new_value"] = "resolved"
                elif outcome_item.get("type") == OutcomeType.KNOWLEDGE_ACQUIRED.value:
                    # Change photo to inspection_report
                    outcome_item["data"]["knowledge_type"] = "inspection_report"
        
        return outcomes
    
    @staticmethod
    def resolve_outcomes(outcomes: List[Dict], params: Dict, context: Dict, 
                        skill: str = None, target_type: str = None) -> List[Dict]:
        """
        Resolve placeholders in outcomes. If a critical ID placeholder (e.g., object_id/target_id/entity_id) is missing in an outcome item, that item is skipped.
        """
        resolved_list = []
        for outcome_template in outcomes:
            # Deep copy to avoid modifying the original template
            resolved_outcome = copy.deepcopy(outcome_template)
            data_template = resolved_outcome.get("data", {})

            # Handle the special discovered_entities placeholder first
            if isinstance(data_template.get("entities"), str) and data_template["entities"] == "{discovered_entities}":
                discovered_ids = params.get('target_ids', [])
                discovered_entities_nodes = []
                graph = context.get("graph")
                if graph and discovered_ids:
                    for entity_id in discovered_ids:
                        entity_node = graph.get_node_by_id(entity_id)
                        if entity_node:
                            discovered_entities_nodes.append(entity_node)
                data_template["entities"] = discovered_entities_nodes

            # Resolve and get the set of missing placeholder keys
            resolved_data, missing_keys = resolve_placeholders_with_missing(data_template, params, context)

            # If a critical ID placeholder is missing, skip this outcome item
            if _should_skip_outcome_for_missing(missing_keys):
                continue

            resolved_outcome["data"] = resolved_data
            resolved_list.append(resolved_outcome)

        return resolved_list

# =========================
# Utility Functions
# =========================
def resolve_placeholders(x: Any, params: Dict, context: Dict) -> Any:
    """
    Non-strict placeholder resolution: ignores missing info, returns only the resolved data.
    Also supports '{a or b or c}'.
    """
    resolved, _ = resolve_placeholders_with_missing(x, params, context)
    return resolved

def resolve_placeholders_strict(x: Any,
                                params: Dict,
                                context: Dict) -> Tuple[Optional[Any], set]:
    """
    Strict placeholder resolution:
    - If any unfillable placeholder (missing) appears at any level, returns None for the whole result;
    - Also returns the set of missing keys.
    """
    resolved, missing = resolve_placeholders_with_missing(x, params, context)
    if missing:
        return None, missing
    return resolved, set()

def resolve_args(args: Dict[str, Any], params: Dict, context: Dict) -> Dict[str, Any]:
    return resolve_placeholders(copy.deepcopy(args), params, context) or {}

def _resolve_placeholder_token(
    token: str,
    params: Dict[str, Any],
    context: Dict[str, Any],
    missing: Optional[set] = None,
) -> Any:
    """
    Resolve placeholders supporting single key and 'a or b or c' forms.
    Looks up in params first, then in context.
    If missing is not None and all keys are missing, adds all keys to missing.
    """
    # e.g.: "object_id" or "object_id or area or robot_location"
    expr = token.strip()
    keys = [k.strip() for k in expr.split(" or ")]
    for k in keys:
        if k in params:
            return params[k]
        if k in context:
            return context[k]
    if missing is not None:
        for k in keys:
            missing.add(k)
    return None

def resolve_placeholders_with_missing(data: Any,
                                      params: Dict,
                                      context: Dict) -> Tuple[Any, set]:
    """
    Only resolves placeholders in the form '{...}'; supports '{a or b or c}'.
    Plain strings are kept as-is.
    Returns (resolved data, set of missing placeholder keys).
    """
    missing: set = set()

    def _rec(x: Any) -> Any:
        if isinstance(x, str):
            # Explicitly preserve these special literals
            if x in ('robot', 'object', 'target'):
                return x

            # Only process { ... } form
            if x.startswith("{") and x.endswith("}"):
                inner = x[1:-1]  # e.g. "object_id" or "object_id or area or robot_location"
                val = _resolve_placeholder_token(inner, params, context, missing)
                return val  # Returns None if not found

            # Plain string: return as-is
            return x

        if isinstance(x, dict):
            return {k: _rec(v) for k, v in x.items()}
        if isinstance(x, list):
            return [_rec(i) for i in x]
        return x

    resolved = _rec(data)
    return resolved, missing

def pull_ctx(key_or_literal: Any, context: Dict, params: Dict) -> Any:
    # Supports getting values from ctx or params; also resolves placeholders
    if isinstance(key_or_literal, str) and key_or_literal.startswith("{") and key_or_literal.endswith("}"):
        k = key_or_literal[1:-1]
        return params.get(k, context.get(k))
    # ctx key
    if isinstance(key_or_literal, str) and key_or_literal in context:
        return context[key_or_literal]
    # params key
    if isinstance(key_or_literal, str) and key_or_literal in params:
        return params[key_or_literal]
    return key_or_literal

def _infer_node_location_id(graph, node: Dict[str, Any], context: Dict[str, Any]) -> Optional[int]:
    try:
        label_to_id = context.get("label_to_id_map") or {}
        _get_node_location_label = getattr(graph, "_get_node_location_label")
        loc_label = _get_node_location_label(node)
        return label_to_id.get(loc_label)
    except Exception:
        return None

def _augment_context_for_precheck(context: Dict[str, Any], params: Dict[str, Any]) -> None:
    """Populate common fields to avoid repeated lookups in rules."""
    g = context.get("graph")
    robot = context.get("robot") or {}
    props = robot.get("properties", {}) or {}
    context.setdefault("robot_id", robot.get("id"))
    context.setdefault("robot_label", props.get("label"))
    context.setdefault("robot_type", props.get("type"))
    context.setdefault("skill", context.get("skill") or params.get("skill"))
    # Populate carrier location
    if params.get("carrier_id") and "carrier_location" not in context and g:
        node = g.get_node_by_id(params["carrier_id"])
        if node:
            context["carrier_node"] = node
            context["carrier_location"] = _infer_node_location_id(g, node, context)


# Determine whether an outcome should be skipped due to critical ID missing
_CRITICAL_ID_KEYS = {"object_id", "target_id", "entity_id", "carrier_id", "dest_id", "from", "to"}

def _should_skip_outcome_for_missing(missing_keys: set) -> bool:
    """
    Skip the outcome if the missing keys set intersects with the critical ID keys set.
    """
    return bool(missing_keys & _CRITICAL_ID_KEYS)

_RUNTIME_WATCHABLE_RULES = {
    "robot_fault",
    "battery_above",
    "path_traversable",
    "target_present_at_expected",
    "target_exists",
    "area_perception_safe",
    "path_avoids_congestion",
    "target_line_of_sight_clear",
}

def extract_runtime_watch_specs(skill: str, params: Dict, context: Dict) -> List[Dict[str, Any]]:
    """
    Filter rules from the skill template's precheck that can be used for runtime monitoring.
    """
    template = SKILL_TEMPLATES.get(skill) or {}
    checks = template.get("precheck") or []
    specs = []
    for item in checks:
        name = (item.get("rule") or "").strip()
        if name in _RUNTIME_WATCHABLE_RULES:
            args = resolve_placeholders(item.get("args") or {}, params, context)
            if name == "area_perception_safe" and args.get("area_id"):
                placeholder = args["area_id"] # e.g., "{object_id or area or robot_location}"
                keys = re.findall(r'\{([^}]+)\}', placeholder)
                if keys:
                    found = False
                    for key in keys[0].split(" or "):
                        key = key.strip()
                        if key in params:
                            args["area_id"] = params[key]
                            found = True
                            break
                        if not found and key in context:
                            args["area_id"] = context[key]
                            found = True
                            break
                    if not found:
                        args["area_id"] = None
            specs.append({
                "name": name,
                "args": args,
                "severity": item.get("severity", "abort"),
                "on_fail": item.get("on_fail") or "PRECHECK_GUARD_FAILED",
                "message": None,
            })
    return specs


# Create global instance
skill_template_manager = SkillTemplateManager()