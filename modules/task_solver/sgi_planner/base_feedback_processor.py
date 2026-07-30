# -*- coding: utf-8 -*-
"""
BaseFeedbackProcessor - base class for feedback processors.
"""
import copy
import logging
from typing import Dict, List, Any, Optional
from enum import Enum
from datetime import datetime


class ReplanningStrategy(Enum):
    """Replanning strategy."""
    FULL = "full"
    PARTIAL = "partial"
    NONE = "none"


class BaseFeedbackProcessor:
    """Base class for feedback processors.

    Common methods:
    - process_newcase_event: process newcase events.
    - _process_outcomes: process outcome lists, extract parameters, and store them in runtime_params.
    - _extract_skill_event: extract a skill event structure from an outcome.
    - _stash_runtime_params: write skill events to context.runtime_params.
    - _normalize_event: normalize event descriptions.
    - reset: reset processor state.

    Common state:
    - last_event: most recent event.
    - replanning_requested: whether replanning is requested.
    - replanning_strategy: replanning strategy.
    - last_discovery_feedback: most recent discovery feedback.
    """

    # Skills that should be stored in runtime_params.
    SKILLS_TO_STASH = {"search", "take_photo", "guide", "follow"}

    # Knowledge types to skip.
    SKIP_KNOWLEDGE_TYPES = {"takeoff_log", "navigation_log", "place_log"}

    def __init__(self, logger=None, context=None):
        self.logger = logger or logging.getLogger(__name__)
        self.context = context
        self.reset()

    def reset(self):
        """Reset processor state."""
        self.last_event: Optional[Dict[str, Any]] = None
        self.replanning_requested: bool = False
        self.replanning_strategy: ReplanningStrategy = ReplanningStrategy.FULL
        self.last_discovery_feedback: Optional[str] = None
        self.replan_signals: List[Dict[str, Any]] = []
        self._accumulated_outcomes: List[Dict[str, Any]] = []

    def set_context(self, context):
        self.context = context

    # =========================================================================
    # Replanning Signal Management
    # =========================================================================

    def add_replan_signal(self, source: str, payload: Any = None) -> None:
        """Append one replanning signal.
        
        Args:
            source: Signal source, such as "newcase" or "evaluation".
            payload: Signal payload.
        """
        self.replan_signals.append({
            "source": source,
            "strategy": self.replanning_strategy,
            "payload": payload,
        })

    def drain_replan_signals(self) -> List[Dict[str, Any]]:
        """Drain and clear all replanning signals."""
        signals = self.replan_signals
        self.replan_signals = []
        return signals

    def aggregate_strategy(self, enable_replanning: bool = True) -> ReplanningStrategy:
        """Aggregate the highest-priority replanning strategy from current signals.
        
        Priority: FULL > PARTIAL > NONE.
        If enable_replanning is False, FULL is downgraded to PARTIAL or NONE.
        
        Args:
            enable_replanning: Whether FULL replanning is allowed.
            
        Returns:
            Aggregated replanning strategy.
        """
        if not self.replan_signals:
            return self.replanning_strategy

        order = {ReplanningStrategy.NONE: 0, ReplanningStrategy.PARTIAL: 1, ReplanningStrategy.FULL: 2}
        best = ReplanningStrategy.NONE
        for sig in self.replan_signals:
            s = sig.get("strategy") or ReplanningStrategy.NONE
            if order[s] > order[best]:
                best = s
                if best == ReplanningStrategy.FULL:
                    break

        if best == ReplanningStrategy.FULL and not enable_replanning:
            has_partial = any(s.get("strategy") == ReplanningStrategy.PARTIAL for s in self.replan_signals)
            best = ReplanningStrategy.PARTIAL if has_partial else ReplanningStrategy.NONE

        return best

    # =========================================================================
    # Feedback Data Preparation
    # =========================================================================

    def prepare_feedback_data(self, **kwargs) -> Any:
        """Prepare feedback data before planning; subclasses implement this.
        
        Each feedback processor implements this method based on its needs.
        UnifiedTaskSolver calls fp.prepare_feedback_data(...) uniformly.
        
        Returns:
            Feedback data in a subclass-defined format. Defaults to None.
        """
        return None

    # =========================================================================
    # Event Handling
    # =========================================================================

    def process_newcase_event(
        self,
        event_description: Dict[str, Any],
        outcomes: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Process a newcase event.
        
        Args:
            event_description: Raw event description dictionary.
            outcomes: Outcome list for the current execution cycle, stored in last_event.details for later use.
        """
        normalized = self._normalize_event(event_description)
        if outcomes is not None:
            normalized.setdefault("details", {})["outcomes"] = outcomes
            self._accumulated_outcomes.extend(outcomes)
        self.last_event = normalized
        self.replanning_requested = True
        self.replanning_strategy = self._decide_strategy(normalized)

    def _decide_strategy(self, event: Dict[str, Any]) -> ReplanningStrategy:
        """Decide the replanning strategy by event type."""
        etype = (event.get("type") or "").upper()
        if etype in ("ROBOT_FAULT", "ROBOT_BATTERY_LOW", "ROBOT_COMM_JAMMED"):
            return ReplanningStrategy.PARTIAL
        return ReplanningStrategy.FULL

    # =========================================================================
    # Outcome Handling
    # =========================================================================

    def _process_outcomes(self, outcomes: List[Dict[str, Any]], world_model_manager) -> None:
        """Process an outcome list, extract parameters, and store them in runtime_params."""
        if not outcomes:
            return

        messages: List[str] = []

        for outcome in outcomes:
            otype = (outcome.get("type") or "").upper()
            data = outcome.get("data") or {}
            meta = outcome.get("meta") or {}

            # Skip entity state changes and specific knowledge types.
            if otype == "ENTITY_STATE_CHANGED":
                continue
            if otype == "KNOWLEDGE_ACQUIRED" and data.get("knowledge_type") in self.SKIP_KNOWLEDGE_TYPES:
                continue

            # Collect messages.
            if msg := data.get("message"):
                messages.append(str(msg).strip())

            # Extract skill events and store them.
            skill = (meta.get("skill") or data.get("skill") or "").lower().strip()
            if skill in self.SKILLS_TO_STASH:
                if event := self._extract_skill_event(outcome):
                    self._stash_runtime_params(event)

        # Merge messages.
        if messages:
            unique_msgs = list(dict.fromkeys(messages))
            self.last_discovery_feedback = ". ".join(unique_msgs) + "."

    def _extract_skill_event(self, outcome: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract a skill event structure from an outcome."""
        data = outcome.get("data") or {}

        meta = outcome.get("meta") or {}

        skill = (meta.get("skill") or data.get("skill") or "").lower().strip()

        # Infer success state.
        success = data.get("success", meta.get("success", True))
        msg = (data.get("message") or "").lower()
        if " failed " in f" {msg} " or msg.startswith("failed"):
            success = False

        return {
            "skill": skill or None,
            "success": success,
            "message": data.get("message"),
            "ts": datetime.now().isoformat(timespec="seconds"),
            "robot": {
                "id": meta.get("robot_id") or data.get("robot_id"),
                "label": meta.get("robot_label") or data.get("agent_id"),
            },
            "area": data.get("area_token") or data.get("area"),
            "destination": data.get("dest_id") or data.get("dest_label"),
            "duration_s": data.get("duration_s") or data.get("persist_s"),
            "target_spec": data.get("target_spec") or data.get("target") or {},
            "found_ids": data.get("found_ids") or [],
            "followed_ids": data.get("followed_ids") or data.get("target_ids") or [],
            "target_id": data.get("target_id") or data.get("object_id"),
            "object_id": data.get("object_id"),
            "task_id": meta.get("task_id") or data.get("task_id"),
        }

    def _stash_runtime_params(self, event: Dict[str, Any]) -> None:
        """Write a skill event to context.runtime_params."""
        if not self.context:
            return

        gt = getattr(self.context, "_generated_text", None)
        if gt is None:
            return

        rp = gt.setdefault("runtime_params", {})
        rp["last"] = event

        if skill := event.get("skill"):
            rp.setdefault("by_skill", {})[skill] = event

        # Link target/object.
        ids = set()
        for key in ("target_id", "object_id"):
            if val := event.get(key):
                ids.add(str(val))
        for key in ("found_ids", "followed_ids"):
            for _id in event.get(key) or []:
                ids.add(str(_id))

        if ids:
            by_target = rp.setdefault("by_target", {})
            for _id in ids:
                by_target[_id] = event

        # Pipeline hints
        hints = rp.setdefault("pipeline_hints", {})
        if event.get("skill") == "search" and event.get("found_ids"):
            hints["candidates"] = event["found_ids"]
            hints["selected_target_id"] = event["found_ids"][0]
            hints["source_area"] = event.get("area")
            hints["target_spec"] = event.get("target_spec") or {}
        elif event.get("skill") == "follow":
            followed = event.get("followed_ids") or ([event["target_id"]] if event.get("target_id") else [])
            if followed:
                hints["followed_targets"] = followed

    # =========================================================================
    # Event Normalization
    # =========================================================================

    def _normalize_event(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize an event description."""
        reason = raw.get("reason") or "SKILL_NOT_FOUND"
        details = copy.deepcopy(raw.get("details") or {})

        if robot := raw.get("robot"):
            details.setdefault("robot", robot)
        if obj := raw.get("object"):
            details.setdefault("object", obj)

        return {
            "type": reason,
            "message": details.get("message") or f"Event {reason}",
            "skill": details.get("payload", {}).get("skill") or raw.get("skill") or "unknown",
            "task_id": raw.get("task_id") or "unknown",
            "severity": details.get("severity") or self._guess_severity(reason),
            "category": details.get("category") or self._guess_category(reason),
            "details": details,
            "event_kind": (raw.get("event_kind") or "incident").lower(),
        }

    def _guess_category(self, etype: str) -> str:
        if etype.startswith("ROBOT_"):
            return "robot"
        if etype.startswith(("TARGET_", "CARRIER_")):
            return "target"
        if etype in ("SKILL_NOT_FOUND", "ROBOT_NOT_APPLICABLE"):
            return "system"
        return "unknown"

    def _guess_severity(self, etype: str) -> str:
        if etype in ("ROBOT_FAULT", "CRITICAL_PATH_BROKEN"):
            return "abort"
        if etype in ("ROBOT_COMM_JAMMED", "ROBOT_BATTERY_LOW"):
            return "soft_abort"
        return "abort"
