# -*- coding: utf-8 -*-
"""
Feedback processor - processes skill execution results and newcase events.
"""
import copy
import json
import logging
from typing import Dict, List, Any, Optional, Callable

from modules.task_solver.sgi_planner.base_feedback_processor import (
    BaseFeedbackProcessor,
    ReplanningStrategy,
)
from modules.hitl.hitl_manager import get_hitl_manager
from modules.platform.platform_factory import get_scene_graph

logger = logging.getLogger(__name__)


def canonicalize_robot_type(t: Optional[str]) -> Optional[str]:
    """Normalize robot type names."""
    if not t:
        return None
    aliases = {
        "uav": "UAV", "fw_uav": "FW_UAV", "ugv": "UGV",
        "quadruped": "Quadruped", "quad": "Quadruped", "humanoid": "Humanoid",
    }
    return aliases.get(str(t).strip().lower(), str(t).strip().upper())


class FeedbackProcessor(BaseFeedbackProcessor):
    """SGI feedback processor.
    - process_outcome_event: replanning strategy decisions, including HITL and graph_is_ongoing.
    - _decide_strategy: event strategy decisions, PARTIAL or FULL.
    - prepare_feedback_data: build structured feedback data for planning, such as failed_skills and completed_skills.
    - build_compact_task_plan: build a compact task plan consistent with the LLM output format.
    - HITL logic: set_user_feedback, get_user_feedback.
    - _has_replacement_robot, should_abort_for_no_replacement: replacement robot checks.
    """

    def __init__(self, logger=None, context=None):
        super().__init__(logger=logger, context=context)
        # HITL manager reference.
        self._hitl_manager = None
        try:
            self._hitl_manager = get_hitl_manager()
        except Exception:
            pass

    def reset(self):
        """Reset processor state."""
        super().reset()
        self.user_feedback: Optional[Dict[str, Any]] = None  # HITL user feedback.

    # =========================================================================
    # Event Handling
    # =========================================================================

    def process_outcome_event(
        self,
        outcomes: List[Dict[str, Any]],
        planner_mode: str,
        status: str,
        goal_completed: bool,
        graph_is_ongoing: bool,
        world_model_manager: Any,
        goal_progress_monitor: Any = None,
        hitl_decision: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Process normal execution results."""
        if self.last_event is not None:
            return

        # Accumulate outcomes.
        if outcomes:
            self._accumulated_outcomes.extend(outcomes)

        # Process outcomes and update runtime_params.
        self._process_outcomes(outcomes, world_model_manager)

        # Handle user feedback from the HITL decision.
        if hitl_decision is not None:
            user_feedback_text = hitl_decision.get("user_feedback", "")
            if user_feedback_text:
                self.user_feedback = user_feedback_text

        # Decide replanning strategy.
        # If user_feedback is set, HITL chose continue_task, so use global replanning.
        if self.user_feedback:
            self.replanning_requested = True
            self.replanning_strategy = ReplanningStrategy.FULL
        else:
            self.replanning_requested = True
            if status == 'completed':
                if goal_completed:
                    self.replanning_strategy = ReplanningStrategy.NONE
                    self.replanning_requested = False
                elif planner_mode == "full" and graph_is_ongoing:
                    if self._has_search_target_not_found(goal_progress_monitor):
                        self.replanning_strategy = ReplanningStrategy.FULL
                    else:
                        self.replanning_strategy = ReplanningStrategy.PARTIAL
                else:
                    self.replanning_strategy = ReplanningStrategy.FULL
            else:
                self.replanning_strategy = ReplanningStrategy.FULL

        # Build evaluation event.
        reason = self._determine_reason(status, goal_completed)
        details = {
            "status": status,
            "goal_completed": goal_completed,
            "reason": reason,
            "discovery_feedback": self.last_discovery_feedback,
            "outcomes": outcomes,
        }
        if self.user_feedback:
            details["user_feedback"] = self.user_feedback

        self.last_event = {
            "type": "EVALUATION_TRIGGER",
            "message": f"Execution: status={status}, goal_completed={goal_completed}",
            "severity": "info",
            "category": "evaluation",
            "details": details,
        }

    def _determine_reason(self, status: str, goal_completed: bool) -> str:
        """Determine the replanning reason."""
        if not self.replanning_requested:
            return "goal_achieved"
        if self.replanning_strategy == ReplanningStrategy.FULL:
            return "plan_completed_but_goal_unachieved" if status == 'completed' else "execution_failed"
        return "continue_existing_plan"

    def _has_search_target_not_found(self, goal_progress_monitor: Any = None) -> bool:
        """Check through the goal progress monitor whether a search did not find the target."""
        if goal_progress_monitor is None:
            return False
        scene_graph = get_scene_graph()
        goal = scene_graph.get_goal() if scene_graph else None
        goal_id = goal.get("id") if goal else None
        if not goal_id:
            return False
        return goal_progress_monitor.has_search_not_found(goal_id)

    def _decide_strategy(self, event: Dict[str, Any]) -> ReplanningStrategy:
        """Decide the replanning strategy by event type."""
        etype = (event.get("type") or "").upper()
        # Robot-related events use partial replanning.
        if etype in ("ROBOT_FAULT", "ROBOT_BATTERY_LOW", "ROBOT_COMM_JAMMED"):
            return ReplanningStrategy.PARTIAL
        return ReplanningStrategy.FULL

    # =========================================================================
    # HITL
    # =========================================================================

    def set_user_feedback(self, feedback: Dict[str, Any]) -> None:
        """
        Manually set user feedback.

        Args:
            feedback: User feedback dictionary containing decision and optional user_feedback text.
        """
        self.user_feedback = feedback

    def get_user_feedback(self) -> Optional[Dict[str, Any]]:
        """Get current user feedback."""
        return self.user_feedback

    # =========================================================================
    # Replacement Robot Checks
    # =========================================================================

    def _has_replacement_robot(self, details: Dict[str, Any]) -> bool:
        """Check whether there is a replacement robot of the same type."""
        try:
            robot_info = details.get("robot") or {}
            failed_label = robot_info.get("label") or details.get("robot_label")
            rtype_raw = robot_info.get("type") or details.get("robot_type")
            rtype = canonicalize_robot_type(rtype_raw)

            if not self.context:
                return False

            gt = getattr(self.context, "_generated_text", {}) or {}
            avail = gt.get("available_robots")  # Format: {"UAV": {"labels": [...], "num": N}, ...}

            if isinstance(avail, dict) and rtype in avail:
                labels = list(avail[rtype].get("labels") or [])
                # Exclude itself.
                if failed_label in labels:
                    labels = [lb for lb in labels if lb != failed_label]
                return len(labels) > 0

            return False
        except Exception:
            return False

    def should_abort_for_no_replacement(self) -> bool:
        """Return True if the latest event is robot fault or low battery and no same-type replacement exists."""
        try:
            if not self.last_event:
                return False

            etype = (self.last_event.get("type") or "").upper()
            if etype not in ("ROBOT_FAULT", "ROBOT_BATTERY_LOW"):
                return False

            details = self.last_event.get("details") or {}
            robot = details.get("robot") or {}
            failed_label = robot.get("label") or details.get("robot_label")
            rtype_raw = robot.get("type") or details.get("robot_type")
            rtype = canonicalize_robot_type(rtype_raw)

            if not (self.context and rtype):
                return False

            avail = (getattr(self.context, "_generated_text", {}) or {}).get("available_robots") or {}
            bucket = avail.get(rtype) or {}
            labels = list(bucket.get("labels") or [])
            labels = [lb for lb in labels if lb != failed_label]  # Exclude itself.
            return len(labels) == 0
        except Exception:
            return False

    # =========================================================================
    # Feedback Data Preparation
    # =========================================================================

    def prepare_feedback_data(
        self,
        task_graph_manager: Optional[Any] = None,
        **kwargs,
    ) -> Optional[List[Dict[str, Any]]]:
        """Build structured feedback data for the planning layer.

        Format:
        - Newcase event: {failed_skills: [...], completed_skills: [...]}
        - Evaluation event: {type, reason, completed_skills: [...]}
        """
        consumed = self.drain_replan_signals()
        if not consumed:
            return None

        entry: Dict[str, Any] = {}
        failed_skills: List[str] = []
        outcomes = []
        user_feedback = None

        for sig in consumed:
            if sig.get("strategy") != ReplanningStrategy.FULL:
                continue

            source = sig.get("source", "")
            payload = sig.get("payload") or {}
            details = payload.get("details") or {}
            robot_label = details.get("robot_label") or (details.get("robot", {}) or {}).get("label")

            # Take outcomes only once; multiple signals share the same copy.
            if not outcomes:
                outcomes = details.get("outcomes") or []

            if details.get("user_feedback"):
                user_feedback = details["user_feedback"]

            if source == "newcase":
                # Failed skills come from the event itself and are merged into one list.
                fail_task_id = payload.get("task_id") or "T?"
                fail_skill = payload.get("skill") or "unknown"
                fail_type = payload.get("type") or ""
                fail_msg = payload.get("message") or ""
                fail_label = f"{fail_task_id}'s {fail_skill} ({robot_label} executed): {fail_type}: {fail_msg}".rstrip(": ")
                failed_skills.append(fail_label)
            else:
                # Evaluation event.
                if not (details.get("discovery_feedback") or details.get("status") == "completed"):
                    continue
                entry["type"] = "EVALUATION_TRIGGER"
                entry["reason"] = details.get("reason")

        # No valid signals.
        if not failed_skills and not entry:
            return None

        if failed_skills:
            entry["failed_skills"] = failed_skills
        if user_feedback:
            entry["user_feedback"] = user_feedback

        # Prefer accumulated outcomes and fall back to outcomes from this signal.
        all_outcomes = self._accumulated_outcomes if self._accumulated_outcomes else outcomes

        # Extract completed_skills from all_outcomes.
        completed_skills = []
        for o in all_outcomes:
            meta = o.get("meta") or {}
            data = o.get("data") or {}
            skill_name = (meta.get("skill") or "").strip()
            if not skill_name:
                continue
            success = meta.get("success")
            if success is None:
                success = data.get("success", True)
            if not success:
                continue
            task_id = meta.get("task_id") or "T?"
            msg = (data.get("message") or meta.get("message") or "").strip()
            label = f"{task_id}'s {skill_name}: {msg}" if msg else f"{task_id}'s {skill_name}"
            completed_skills.append(label)

        if completed_skills:
            entry["completed_skills"] = completed_skills

        # Clear accumulated outcomes for real replanning.
        self._accumulated_outcomes = []

        return [entry]

    def build_compact_task_plan(self) -> Optional[Dict[str, Any]]:
        """Build a compact task plan consistent with the LLM output format."""
        if not self.context:
            return None
        try:
            gt = getattr(self.context, "_generated_text", {}) or {}
            raw_plan = gt.get("task_plan")
            if not raw_plan:
                return None

            plan_data = json.loads(raw_plan) if isinstance(raw_plan, str) else raw_plan
            if not isinstance(plan_data, dict):
                return None

            compact = copy.deepcopy(plan_data)

            # Locate task list.
            if "task_graph" in compact and isinstance(compact["task_graph"], dict):
                nodes = compact["task_graph"].get("nodes") or []
            else:
                nodes = compact.get("atomic_tasks") or []

            # Compress nodes: remove description and keep compact skill format.
            for node in nodes:
                node.pop("description", None)
                # Convert dictionary-format skills to compact strings.
                skills = node.get("required_skills") or []
                compact_skills = []
                for sk in skills:
                    if isinstance(sk, str):
                        compact_skills.append(sk)
                    elif isinstance(sk, dict):
                        rt = "|".join(sk.get("assigned_robot_type") or [])
                        name = sk.get("skill_name", "")
                        cnt = sk.get("assigned_robot_count", 1)
                        compact_skills.append(f"{rt}:{name}:{cnt}" if rt else f":{name}:{cnt}")
                    else:
                        compact_skills.append(str(sk))
                node["required_skills"] = compact_skills

            # Convert dictionary-format edges to compact strings.
            if "task_graph" in compact and isinstance(compact["task_graph"], dict):
                edges = compact["task_graph"].get("edges") or []
                compact_edges = []
                for e in edges:
                    if isinstance(e, str):
                        compact_edges.append(e)
                    elif isinstance(e, dict):
                        s = f"{e.get('from', '')}->{e.get('to', '')}"
                        if e.get("type") == "conditional" and e.get("condition"):
                            s += f":{e['condition']}"
                        compact_edges.append(s)
                    else:
                        compact_edges.append(str(e))
                compact["task_graph"]["edges"] = compact_edges

            return compact
        except Exception:
            return None
