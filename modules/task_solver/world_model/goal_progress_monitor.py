# -*- coding: utf-8 -*-
"""
Goal progress monitor - determine goal completion from world state and goal definitions.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime
import copy
import asyncio
import numpy as np

from modules.task_solver.world_model.utils.goal_eval_utils import area_spec_to_geom, find_object_ids_by_target_geom
from modules.utils.location_utils import get_entity_position
from modules.utils.geom_utils import point_in_area_geometry
from modules.platform.abstract_scene_graph import AbstractSceneGraph


# =========================================================================
# Data classes - recent event cache
# =========================================================================

@dataclass
class LastSearch:
    goal_id: str
    found: Optional[bool]
    area: Optional[str]
    target_spec: Dict[str, Any]
    ts: datetime
    object_id: Optional[int] = None
    duration_s: Optional[float] = None
    area_raw: Optional[Dict[str, Any]] = None
    conf_ge: Optional[float] = None
    persist_ge_s: Optional[float] = None


@dataclass
class PatrolSearchRecord:
    """Single patrol search event record."""
    goal_id: str
    robot_types: Set[str]       # Robot types in this search, such as {"UAV", "Quadruped"}.
    duration_s: float           # Duration of this search.
    area: Optional[str]
    ts: datetime


@dataclass
class LastTrack:
    goal_id: str
    robot_id: Optional[int]
    object_id: Optional[int]
    duration_s: Optional[float]
    target_spec: Dict[str, Any]
    ts: datetime


@dataclass
class LastPhoto:
    goal_id: str
    robot_id: Optional[int]
    object_id: Optional[int]
    target_spec: Dict[str, Any]
    ts: datetime
    distance_at_event: Optional[float] = None


@dataclass
class LastBroadcast:
    goal_id: str
    robot_id: Optional[int]
    object_id: Optional[int]
    duration_s: Optional[float]
    message: Optional[str]
    ts: datetime
    distance_at_event: Optional[float] = None


@dataclass
class LastEventDetected:
    goal_id: str
    event_type: Optional[str]
    area: Optional[str]
    ts: datetime
    conf_ge: Optional[float] = None
    persist_ge_s: Optional[float] = None


# =========================================================================
# Goal progress monitor
# =========================================================================

class GoalProgressMonitor:
    """Goal progress monitor - determine whether goals are complete."""
    
    # Distance threshold in meters.
    CLOSE_DISTANCE_THRESHOLD = 80.0

    def __init__(self, scene_graph: Optional[AbstractSceneGraph] = None):
        self.scene_graph = scene_graph
        self.active_goals: Dict[str, Dict[str, Any]] = {}
        self.completed_goals: Set[str] = set()
        
        # Event cache.
        self._last_search: Dict[str, LastSearch] = {}
        self._last_follow: Dict[str, LastTrack] = {}
        self._last_photo: Dict[str, LastPhoto] = {}
        self._last_broadcast: Dict[str, LastBroadcast] = {}
        self._last_event: Dict[str, LastEventDetected] = {}
        self._events: List[Dict[str, Any]] = []
        self._patrol_search_records: Dict[str, List[PatrolSearchRecord]] = {}
        
        self._label_to_id = scene_graph.get_node_map(map_type='label_to_id') if scene_graph else {}

    def set_scene_graph(self, scene_graph: Optional[AbstractSceneGraph]):
        self.scene_graph = scene_graph
        self._label_to_id = scene_graph.get_node_map(map_type='label_to_id') if scene_graph else {}

    def register_goal(self, goal_config: Dict[str, Any]):
        """Register a goal."""
        goal_id = goal_config.get("id")
        self.active_goals[goal_id] = {
            "definition": goal_config,
            "status": "active",
            "progress": {},
        }

    # =========================================================================
    # Outcome handling
    # =========================================================================

    def process_outcomes(self, outcomes: List[Dict[str, Any]]) -> List[str]:
        """Process outcomes and return completed goal IDs."""
        now = datetime.now()
        self._events.extend(outcomes or [])

        for outcome in outcomes or []:
            self._process_single_outcome(outcome, now)

        # Check goal completion.
        completed = []
        for goal_id, goal_data in list(self.active_goals.items()):
            if goal_data["status"] != "active":
                continue
            if self._check_goal_completion(goal_id):
                goal_data["status"] = "completed"
                self.completed_goals.add(goal_id)
                completed.append(goal_id)
        
        return completed

    def _process_single_outcome(self, outcome: Dict[str, Any], now: datetime):
        """Process a single outcome."""
        otype = outcome.get("type") or ""
        if otype != "KNOWLEDGE_ACQUIRED":
            return
            
        data = outcome.get("data") or {}
        meta = outcome.get("meta") or {}
        skill = (meta.get("skill") or "").strip().lower()
        
        # Get goal_id.
        scene_goal = self.scene_graph.get_goal() if self.scene_graph else None
        goal_id = (scene_goal.get("id") if scene_goal else None) or data.get("goal_id")
        if not goal_id or goal_id not in self.active_goals:
            return

        success = bool(meta.get("success", True))
        if not success:
            return

        # Update caches by skill type.
        if skill == "search":
            self._handle_search_outcome(goal_id, data, now)
        elif skill == "follow":
            self._handle_follow_outcome(goal_id, data, now)
        elif skill == "take_photo":
            self._handle_photo_outcome(goal_id, data, now)
        elif skill == "broadcast" or data.get("knowledge_type") == "broadcast_event":
            self._handle_broadcast_outcome(goal_id, data, now)

    @staticmethod
    def _extract_robot_type(label: str) -> str:
        """Extract robot type from a label, such as 'UAV-1' -> 'UAV'."""
        if not label:
            return ""
        # Use the part before the last '-number' as the type.
        parts = label.rsplit("-", 1)
        return parts[0].strip() if parts else label.strip()

    def _handle_search_outcome(self, goal_id: str, data: Dict, now: datetime):
        """Process a search outcome without downgrading a positive result."""
        found = data.get("found")
        found_bool = None if found is None else bool(found)
        area = data.get("area_token") or data.get("area_searched") or data.get("area")
        duration = data.get("duration_s") or data.get("persist_s")
        
        # Patrol search event: accumulate robot types regardless of found result.
        knowledge_type = (data.get("knowledge_type") or "").lower().strip()
        if knowledge_type == "patrol_log":
            robots = data.get("robots") or []
            robot_types: Set[str] = set()
            rtype = data.get("robot_type")
            if rtype:
                robot_types.add(rtype)
            else:
                for r in robots:
                    rtype = self._extract_robot_type(r.get("label") or "")
                    if rtype:
                        robot_types.add(rtype)
            dur_val = float(duration) if duration else 0.0
            record = PatrolSearchRecord(
                goal_id=goal_id,
                robot_types=robot_types,
                duration_s=dur_val,
                area=str(area) if area else None,
                ts=now,
            )
            self._patrol_search_records.setdefault(goal_id, []).append(record)

        # Do not overwrite an existing positive result with a non-positive result.
        existing = self._last_search.get(goal_id)
        if existing and existing.found is True and found_bool is not True:
            return
        
        self._last_search[goal_id] = LastSearch(
            goal_id=goal_id,
            found=found_bool,
            area=str(area) if area else None,
            target_spec=data.get("target_spec") or {},
            ts=now,
            object_id=data.get("object_id"),
            duration_s=float(duration) if duration else None,
            area_raw=area if isinstance(area, dict) else None,
            conf_ge=data.get("conf_ge"),
            persist_ge_s=data.get("persist_ge_s"),
        )
        
        # Record an event when the target is found.
        if found_bool:
            evt_type = (data.get("target_spec") or {}).get("event_type") or (data.get("target_spec") or {}).get("type")
            self._last_event[goal_id] = LastEventDetected(
                goal_id=goal_id,
                event_type=evt_type,
                area=self._last_search[goal_id].area,
                ts=now,
                conf_ge=self._last_search[goal_id].conf_ge,
                persist_ge_s=self._last_search[goal_id].persist_ge_s,
            )

    def _handle_follow_outcome(self, goal_id: str, data: Dict, now: datetime):
        """Process a follow outcome."""
        duration = data.get("duration_s") or data.get("persist_s") or data.get("persist_ge_s")
        self._last_follow[goal_id] = LastTrack(
            goal_id=goal_id,
            robot_id=data.get("robot_id"),
            object_id=data.get("target_id") or data.get("object_id"),
            duration_s=float(duration) if duration else None,
            target_spec=data.get("target_spec") or {},
            ts=now,
        )

    def _handle_photo_outcome(self, goal_id: str, data: Dict, now: datetime):
        """Process a photo outcome."""
        self._last_photo[goal_id] = LastPhoto(
            goal_id=goal_id,
            robot_id=data.get("robot_id"),
            object_id=data.get("target_id") or data.get("object_id"),
            target_spec=data.get("target_spec") or {},
            ts=now,
            distance_at_event=data.get("robot_target_distance"),
        )

    def _handle_broadcast_outcome(self, goal_id: str, data: Dict, now: datetime):
        """Process a broadcast outcome."""
        self._last_broadcast[goal_id] = LastBroadcast(
            goal_id=goal_id,
            robot_id=data.get("robot_id"),
            object_id=data.get("target_id") or data.get("object_id"),
            duration_s=data.get("duration_s"),
            message=data.get("message_text"),
            ts=now,
            distance_at_event=data.get("robot_target_distance"),
        )

    # =========================================================================
    # Goal completion checks
    # =========================================================================

    def _check_goal_completion(self, goal_id: str) -> bool:
        """Check whether a goal is complete."""
        gdef = (self.active_goals.get(goal_id) or {}).get("definition") or {}
        gtype = (gdef.get("goal_type") or "").lower().strip()

        # Dispatch by goal type.
        handlers = {
            "traffic_enforcement": self._check_enforcement,
            "area_search": self._check_area_search,
            "target_following": self._check_following,
            "transport": self._check_transport,
            "evidence_collection": self._check_evidence,
            "verbal_broadcast": self._check_broadcast,
            "patrol": self._check_patrol,
            "assembly": self._check_assembly,
            "emergency_response": self._check_emergency,
            "guidance": self._check_guidance,
        }
        
        handler = handlers.get(gtype)
        return handler(goal_id, gdef) if handler else False

    def has_search_not_found(self, goal_id: str) -> bool:
        """
        Check whether the latest search result for the given goal was not found.

        Args:
            goal_id: Goal ID.

        Returns:
            True if the latest search result has found=False.
        """
        ls = self._last_search.get(goal_id)
        return ls is not None and ls.found is False

    def get_search_context_for_decision(self, goal_id: str) -> Dict[str, Any]:
        """
        Get search context for HITL decisions.

        Args:
            goal_id: Goal ID.

        Returns:
            Dictionary containing search context.
        """
        ls = self._last_search.get(goal_id)
        gdef = (self.active_goals.get(goal_id) or {}).get("definition") or {}
        return {
            "goal_id": goal_id,
            "searched_area": ls.area if ls else None,
            "search_duration": ls.duration_s if ls else None,
            "target_spec": ls.target_spec if ls else {},
            "goal_type": gdef.get("goal_type"),
            "goal_description": gdef.get("description"),
        }

    def _check_enforcement(self, goal_id: str, gdef: Dict) -> bool:
        """Check an enforcement goal."""
        ls = self._last_search.get(goal_id)
        if ls and ls.found is False:
            return False
        
        event_ok = (goal_id in self._last_event) or (ls and ls.found is True)
        if not event_ok:
            return False
        
        if self._is_strict_enforcement(gdef):
            return self._check_photo_close(goal_id)
        else:
            return self._check_broadcast_close(goal_id)

    def _check_area_search(self, goal_id: str, gdef: Dict) -> bool:
        """Check an area search goal."""
        return goal_id in self._last_search

    def _check_following(self, goal_id: str, gdef: Dict) -> bool:
        """Check a following goal."""
        ls = self._last_search.get(goal_id)
        if ls and ls.found is False:
            return False
        
        lt = self._last_follow.get(goal_id)
        if not lt:
            return False

        # Check duration.
        required = self._get_condition_duration(gdef, "FOLLOWED", "duration_ge_s")
        if required and (lt.duration_s or 0) < required:
            return False

        # Check distance.
        return self._are_entities_close(lt.robot_id, lt.object_id)

    def _check_transport(self, goal_id: str, gdef: Dict) -> bool:
        """Check a transport goal."""
        ls = self._last_search.get(goal_id)
        if ls and ls.found is False:
            return False
        
        det_args = self._get_first_condition_args(gdef, {"DETECTED"})
        if not det_args:
            return False
        return self._is_target_in_area(goal_id, det_args.get("target") or {}, det_args.get("area"))

    def _check_evidence(self, goal_id: str, gdef: Dict) -> bool:
        """Check an evidence collection goal."""
        ls = self._last_search.get(goal_id)
        if ls and ls.found is False:
            return False
        return self._check_photo_close(goal_id)

    def _check_broadcast(self, goal_id: str, gdef: Dict) -> bool:
        """Check a broadcast goal."""
        ls = self._last_search.get(goal_id)
        if ls and ls.found is False:
            return False
        
        lb = self._last_broadcast.get(goal_id)
        if not lb:
            return False

        # Check duration.
        required = self._get_condition_duration(gdef, "SPEAK_DURATION", "duration_ge_s")
        if required and (lb.duration_s or 0) < required:
            return False

        return self._check_entity_close_by_event(lb.robot_id, lb.object_id, lb.distance_at_event)

    def _check_patrol(self, goal_id: str, gdef: Dict) -> bool:
        """Check a patrol goal.
        
        Completion rules: if found=True, pass directly. Otherwise all rules below must hold:
        1. A search event record exists.
        2. Accumulated robot types cover {UAV, Quadruped}, or there are at least
           two Quadruped search records.
        3. At least one search event reaches the required duration.
        """
        # If the search found the target, treat the goal as complete.
        if goal_id in self._last_event:
            return True
        
        ls = self._last_search.get(goal_id)
        if ls and ls.found is True:
            return True

        # Get the duration requirement.
        required = self._get_condition_duration(gdef, "PATROL_DURATION", "duration_ge_s")
        if required is None:
            return False

        # Check accumulated patrol search records.
        records = self._patrol_search_records.get(goal_id)
        if not records:
            return False

        # Collect robot types from records that satisfy the duration requirement,
        # and count qualified Quadruped records.
        accumulated_types: Set[str] = set()
        quadruped_qualified_count = 0

        for rec in records:
            if rec.duration_s >= required:
                accumulated_types.update(rec.robot_types)
                if "Quadruped" in rec.robot_types:
                    quadruped_qualified_count += 1

        # Rule 1: both UAV and Quadruped are covered.
        if "UAV" in accumulated_types and "Quadruped" in accumulated_types:
            return True

        # Rule 2: at least two Quadruped search records satisfy the duration requirement.
        if quadruped_qualified_count >= 2:
            return True

        return False

    def _check_assembly(self, goal_id: str, gdef: Dict) -> bool:
        """Check an assembly goal."""
        return self._evaluate_condition(gdef.get("success_condition") or {})

    def _check_emergency(self, goal_id: str, gdef: Dict) -> bool:
        """Check an emergency response goal."""
        ls = self._last_search.get(goal_id)
        if ls and ls.found is False:
            return False
        return self._evaluate_condition(gdef.get("success_condition") or {})

    def _check_guidance(self, goal_id: str, gdef: Dict) -> bool:
        """Check a guidance goal."""
        ls = self._last_search.get(goal_id)
        if ls and ls.found is False:
            return False
        
        det_args = self._get_first_condition_args(gdef, {"DETECTED"})
        if not det_args:
            return False
        return self._is_target_in_area(goal_id, det_args.get("target") or {}, det_args.get("area"))

    # =========================================================================
    # Helper methods
    # =========================================================================

    def _check_photo_close(self, goal_id: str) -> bool:
        """Check whether photo capture satisfies the distance condition."""
        lp = self._last_photo.get(goal_id)
        if not lp:
            return False
        if lp.object_id is None:
            return True
        return self._check_entity_close_by_event(lp.robot_id, lp.object_id, lp.distance_at_event)

    def _check_broadcast_close(self, goal_id: str) -> bool:
        """Check whether broadcast satisfies the distance condition."""
        lb = self._last_broadcast.get(goal_id)
        if not lb:
            return False
        if lb.object_id is None:
            return True
        return self._check_entity_close_by_event(lb.robot_id, lb.object_id, lb.distance_at_event)

    def _check_entity_close_by_event(
        self, 
        robot_id: Optional[int], 
        object_id: Optional[int], 
        distance_at_event: Optional[float]
    ) -> bool:
        """Check entity distance, preferring the event-recorded distance."""
        if distance_at_event is not None:
            return distance_at_event <= self.CLOSE_DISTANCE_THRESHOLD
        return self._are_entities_close(robot_id, object_id)

    def _are_entities_close(self, id1: Optional[int], id2: Optional[int]) -> bool:
        """Check whether two entities are close enough."""
        if id1 is None or id2 is None or not self.scene_graph:
            return False
        
        pos1 = get_entity_position(self.scene_graph, id1)
        pos2 = get_entity_position(self.scene_graph, id2)
        if pos1 is None or pos2 is None:
            return False
        
        distance = np.linalg.norm(np.array(pos1) - np.array(pos2))
        return distance <= self.CLOSE_DISTANCE_THRESHOLD

    def _is_strict_enforcement(self, gdef: Dict) -> bool:
        """Determine whether strict enforcement mode is enabled."""
        ctype = ((gdef.get("context") or {}).get("evidence_type") or {}).get("type", "")
        return str(ctype).strip().lower().startswith("strict")

    def _get_condition_duration(self, gdef: Dict, op_name: str, key: str) -> Optional[float]:
        """Extract a duration requirement from success conditions."""
        sc = gdef.get("success_condition") or {}
        op = (sc.get("op") or "").upper()
        
        if op == op_name:
            val = (sc.get("args") or {}).get(key)
            return float(val) if val is not None else None
        
        if op in {"AND", "OR", "SEQ"}:
            for step in sc.get("args") or []:
                if (step.get("op") or "").upper() == op_name:
                    val = (step.get("args") or {}).get(key)
                    return float(val) if val is not None else None
        return None

    def _get_first_condition_args(self, gdef: Dict, wanted: Set[str]) -> Optional[Dict]:
        """Get args for the first matching operation."""
        sc = gdef.get("success_condition") or {}
        op = (sc.get("op") or "").upper()
        
        if op in wanted:
            return sc.get("args") or {}
        
        if op in {"AND", "OR", "SEQ"}:
            for step in sc.get("args") or []:
                if (step.get("op") or "").upper() in wanted:
                    return step.get("args") or {}
        return None

    def _is_target_in_area(self, goal_id: str, target_spec: Dict, area_spec: Dict) -> bool:
        """Check whether the target is inside the specified area."""
        ls = self._last_search.get(goal_id)
        obj_id = ls.object_id if ls else None
        area_hint = ls.area_raw if ls else None

        # If object_id is available, check its position directly.
        if obj_id is not None and self.scene_graph:
            geom = area_hint if isinstance(area_hint, dict) else area_spec_to_geom(area_spec, self.scene_graph)
            pos = get_entity_position(self.scene_graph, obj_id)
            if pos is None:
                return False
            return point_in_area_geometry(pos, geom)

        # Search using area geometry.
        if isinstance(area_hint, dict):
            ids = find_object_ids_by_target_geom(area_hint, target_spec, self.scene_graph)
            return len(ids) > 0

        # Fall back to the goal definition.
        geom = area_spec_to_geom(area_spec, self.scene_graph)
        ids = find_object_ids_by_target_geom(geom, target_spec, self.scene_graph)
        return len(ids) > 0

    def _evaluate_condition(self, node: Dict) -> bool:
        """Recursively evaluate success conditions."""
        if not node or not isinstance(node, dict) or not self.scene_graph:
            return False

        op = (node.get("op") or "").upper()
        args = node.get("args")

        if op == "AND":
            return isinstance(args, list) and all(self._evaluate_condition(sub) for sub in args)
        
        if op == "OR":
            return isinstance(args, list) and any(self._evaluate_condition(sub) for sub in args)
        
        if op == "STATE":
            return self._evaluate_state_condition(args)
        
        return False

    def _evaluate_state_condition(self, args: Any) -> bool:
        """Evaluate a STATE condition."""
        if not isinstance(args, dict):
            return False
        
        target_spec = args.get("target")
        key = args.get("key")
        expected = args.get("equals")
        area_spec = args.get("area")
        
        if not all([target_spec, key is not None, expected is not None, area_spec]):
            return False

        # Remove the property key being checked.
        finding_spec = copy.deepcopy(target_spec)
        if 'features' in finding_spec and isinstance(finding_spec['features'], dict):
            finding_spec['features'].pop(key, None)

        geom = area_spec_to_geom(area_spec, self.scene_graph)
        object_ids = find_object_ids_by_target_geom(geom, finding_spec, self.scene_graph)

        for obj_id in object_ids:
            node = self.scene_graph.get_node_by_id(obj_id)
            if not node:
                continue
            
            actual = node.get("properties", {}).get(key)
            
            # Normalize types.
            if isinstance(expected, bool) and isinstance(actual, str):
                actual = actual.lower() in ("true", "1")
            elif isinstance(expected, str) and not isinstance(actual, str):
                actual = str(actual)

            if actual == expected:
                return True
        
        return False

    # =========================================================================
    # Query interface
    # =========================================================================

    def get_goal_nature(self, goal_id: str) -> str:
        """Get goal nature: deterministic or open."""
        gdef = (self.active_goals.get(goal_id) or {}).get("definition") or {}
        nature = (gdef.get("goal_determinacy") or "").lower().strip()
        return nature if nature in ("deterministic", "open") else "open"

    def generate_terminal_feedback(
        self,
        goal_id: Optional[str],
        outcomes: List[Dict[str, Any]],
        plan_completed: bool,
        achieved: Optional[bool]
    ) -> str:
        """Generate terminal feedback."""
        msgs: List[str] = []
        
        # Extract the latest message from outcomes.
        allowed_skills = {"search", "take_photo", "follow", "handle_hazard", "guide"}
        allowed_kt = {"entity_discovery", "photo", "patrol_log"}
        
        for o in reversed(outcomes or []):
            data = o.get("data") or {}
            msg = data.get("message")
            if not msg:
                continue
            
            meta = o.get("meta") or {}
            skill = (meta.get("skill") or "").lower().strip()
            ktype = (data.get("knowledge_type") or "").lower().strip()
            
            if skill in allowed_skills or ktype in allowed_kt:
                msgs.append(msg)
                break

        # Generate the conclusion.
        tail = None
        if goal_id and goal_id in self.active_goals:
            nature = self.get_goal_nature(goal_id)
            if nature == "open":
                if achieved:
                    tail = "Open-ended goal. Main goal achieved. Task finished."
                else:
                    tail = "Open-ended goal. Main goal not achieved. Replanning is required."
            else:
                if achieved is True:
                    tail = "Main goal achieved. Task finished."
                elif achieved is False:
                    tail = "Main goal not achieved. Replanning is required."
                else:
                    tail = "Goal status unknown."
        else:
            tail = "Plan concluded." if plan_completed else "Execution interrupted."

        if tail:
            msgs.append(tail)
        
        return " ".join(msgs) if msgs else (tail or "No feedback available.")
