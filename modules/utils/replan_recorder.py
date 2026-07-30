
import json
import copy
import logging
import os
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Set

from modules.platform.platform_factory import get_scene_graph

logger = logging.getLogger(__name__)


class ReplanSampleCollected(Exception):
    """
    Signals to the caller that a replan sample has been collected and the run can end.
    """
    pass


class ReplanDatasetRecorder:
    """
    Replan dataset recorder.
    - Global switch: when disabled, all APIs are no-op and normal logic is unaffected.
    - When enabled:
        * snapshot_newcase(...) records an event snapshot to context when NewCaseEvent occurs.
        * handle_replan_prompt(...) writes JSONL during replan prompt construction based on filters.
          `world_state` uses prompt-time state, while `event_world_state` keeps the event-time snapshot.
          When stop_after_record=True, raises ReplanSampleCollected after writing.
    """
    _enabled: bool = False
    _min_newcases: int = 1
    _min_replan_index: int = 1
    _event_types: Optional[Set[str]] = None
    _event_reasons: Optional[Set[str]] = None
    _goal_types: Optional[Set[str]] = None
    _max_records_per_run: int = 1
    _stop_after_record: bool = True
    _save_llm_io: bool = False
    _data_tag: Optional[str] = None
    _require_prompt_event_match: bool = False
    _single_event_only: bool = False

    @staticmethod
    def _normalize_filter(values: Optional[Iterable[str]]) -> Optional[Set[str]]:
        if values is None:
            return None
        normalized = {str(value).strip() for value in values if str(value).strip()}
        return normalized or None

    @classmethod
    def enable(
        cls,
        *,
        min_newcases: int = 1,
        min_replan_index: int = 1,
        event_types: Optional[Iterable[str]] = None,
        event_reasons: Optional[Iterable[str]] = None,
        goal_types: Optional[Iterable[str]] = None,
        max_records_per_run: int = 1,
        stop_after_record: bool = True,
        save_llm_io: bool = False,
        data_tag: Optional[str] = None,
        require_prompt_event_match: bool = False,
        single_event_only: bool = False,
    ) -> None:
        cls._enabled = True
        cls._min_newcases = max(1, int(min_newcases or 1))
        cls._min_replan_index = max(1, int(min_replan_index or 1))
        cls._event_types = cls._normalize_filter(event_types)
        cls._event_reasons = cls._normalize_filter(event_reasons)
        cls._goal_types = cls._normalize_filter(goal_types)
        cls._max_records_per_run = max(1, int(max_records_per_run or 1))
        cls._stop_after_record = bool(stop_after_record)
        cls._save_llm_io = bool(save_llm_io)
        cls._data_tag = str(data_tag or os.environ.get("GSI_REPLAN_DATA_TAG") or "").strip() or None
        cls._require_prompt_event_match = bool(require_prompt_event_match)
        cls._single_event_only = bool(single_event_only)
        logger.info("[ReplanDatasetRecorder] capture enabled")

    @classmethod
    def disable(cls) -> None:
        cls._enabled = False
        cls._min_newcases = 1
        cls._min_replan_index = 1
        cls._event_types = None
        cls._event_reasons = None
        cls._goal_types = None
        cls._max_records_per_run = 1
        cls._stop_after_record = True
        cls._save_llm_io = False
        cls._data_tag = None
        cls._require_prompt_event_match = False
        cls._single_event_only = False
        logger.info("[ReplanDatasetRecorder] capture disabled")

    @classmethod
    def is_enabled(cls) -> bool:
        return cls._enabled

    @classmethod
    def save_llm_io_enabled(cls) -> bool:
        return cls._enabled and cls._save_llm_io

    @staticmethod
    def _event_field_values(event: Dict[str, Any], field: str) -> Set[str]:
        values = set()
        direct = event.get(field)
        if direct is not None:
            values.add(str(direct))
        details = event.get("details")
        if isinstance(details, dict):
            nested = details.get(field)
            if nested is not None:
                values.add(str(nested))
        if field == "type":
            alias = event.get("event_type")
            if alias is not None:
                values.add(str(alias))
        return values

    @classmethod
    def _event_matches_filters(cls, event: Dict[str, Any]) -> bool:
        if cls._event_types is not None and cls._event_types.isdisjoint(cls._event_field_values(event, "type")):
            return False
        if cls._event_reasons is not None and cls._event_reasons.isdisjoint(cls._event_field_values(event, "reason")):
            return False
        return True

    @staticmethod
    def _normalize_text(value: Any) -> str:
        return str(value or "").strip().lower()

    @classmethod
    def _event_reason_values(cls, event: Dict[str, Any]) -> List[str]:
        values = sorted(cls._event_field_values(event, "reason"))
        if values:
            return values
        return sorted(cls._event_field_values(event, "type"))

    @staticmethod
    def _feedback_texts(feedback_data: Any) -> List[str]:
        if not isinstance(feedback_data, list):
            return []

        texts: List[str] = []
        for item in feedback_data:
            if not isinstance(item, dict):
                continue
            for key in ("type", "reason", "user_feedback"):
                value = item.get(key)
                if value:
                    texts.append(str(value))
            for key in ("failed_skills", "completed_skills"):
                values = item.get(key) or []
                if isinstance(values, list):
                    texts.extend(str(value) for value in values if value)
        return texts

    @classmethod
    def _select_snapshot_for_prompt(
        cls,
        snapshots: List[Dict[str, Any]],
        feedback_data: Any,
    ) -> Dict[str, Any]:
        feedback_texts = cls._feedback_texts(feedback_data)
        has_prompt_feedback = bool(feedback_texts)
        joined = "\n".join(feedback_texts).lower()

        reason_matches: List[Dict[str, Any]] = []
        type_matches: List[Dict[str, Any]] = []

        if has_prompt_feedback:
            for snapshot in snapshots:
                event = snapshot.get("event", {}) or {}
                reason_values = cls._event_field_values(event, "reason")
                type_values = cls._event_field_values(event, "type")
                if any(cls._normalize_text(value) in joined for value in reason_values if value):
                    reason_matches.append(snapshot)
                elif any(cls._normalize_text(value) in joined for value in type_values if value):
                    type_matches.append(snapshot)

        matched_snapshots = reason_matches or type_matches
        selected = matched_snapshots[-1] if matched_snapshots else snapshots[-1]

        prompt_event_reasons: List[str] = []
        seen = set()
        for snapshot in matched_snapshots:
            event = snapshot.get("event", {}) or {}
            for value in cls._event_reason_values(event):
                if value not in seen:
                    prompt_event_reasons.append(value)
                    seen.add(value)

        return {
            "snapshot": selected,
            "prompt_feedback_data": copy.deepcopy(feedback_data) if isinstance(feedback_data, list) else None,
            "prompt_event_history": [
                copy.deepcopy(item.get("event", {}) or {}) for item in matched_snapshots
            ],
            "prompt_event_reasons": prompt_event_reasons,
            "prompt_event_count": len(matched_snapshots),
            "has_prompt_feedback": has_prompt_feedback,
            "event_prompt_match": bool(matched_snapshots) if has_prompt_feedback else None,
        }

    @staticmethod
    def _get_generated_text(context: Any) -> Optional[Dict[str, Any]]:
        if context is None or not hasattr(context, "_generated_text"):
            return None
        gt = context._generated_text
        return gt if isinstance(gt, dict) else None

    @classmethod
    def _workspace_file(cls, gt: Dict[str, Any], filename: str) -> Optional[Path]:
        workspace_root = gt.get("workspace_root")
        if not workspace_root:
            logger.warning("[ReplanDatasetRecorder] workspace_root missing in context._generated_text")
            return None
        base_dir = Path(workspace_root)
        base_dir.mkdir(parents=True, exist_ok=True)
        return base_dir / filename

    @staticmethod
    def _world_state_from_scene_graph(scene_graph: Any) -> Dict[str, Any]:
        return {
            "nodes": copy.deepcopy(getattr(scene_graph, "_nodes", []) or []),
            "edges": copy.deepcopy(getattr(scene_graph, "_edges", []) or []),
            "goal": copy.deepcopy(getattr(scene_graph, "_goal", None)),
        }

    @classmethod
    def _prompt_time_world_state(cls) -> tuple[Dict[str, Any], str]:
        try:
            scene_graph = get_scene_graph()
        except Exception as exc:
            logger.warning("[ReplanDatasetRecorder] prompt-time scene graph unavailable: %s", exc)
            return {}, "unavailable"
        if scene_graph is None:
            return {}, "unavailable"
        return cls._world_state_from_scene_graph(scene_graph), "prompt_time_scene_graph"

    @classmethod
    def _build_replan_record(cls, context: Any, prompt: str) -> Optional[Dict[str, Any]]:
        gt = cls._get_generated_text(context)
        if not cls._enabled or gt is None:
            return None

        is_replanning = bool(gt.get("is_replanning", False))
        if not is_replanning:
            # Initial planning: use the normal LLM call and do not record replan samples.
            return None

        snapshots = gt.get("replan_snapshots") or []
        if not snapshots:
            logger.warning("[ReplanDatasetRecorder] is_replanning=True but no snapshots found")
            return None

        replan_index = int(gt.get("replan_capture_replan_index", 0) or 0) + 1
        gt["replan_capture_replan_index"] = replan_index
        event_count = len(snapshots)
        records_written = int(gt.get("replan_capture_records_written", 0) or 0)

        if cls._goal_types is not None and str(gt.get("goal_type", "")) not in cls._goal_types:
            return None
        if event_count < cls._min_newcases:
            return None
        if replan_index < cls._min_replan_index:
            return None
        if records_written >= cls._max_records_per_run:
            return None

        selection = cls._select_snapshot_for_prompt(snapshots, gt.get("feedback_data"))
        if cls._require_prompt_event_match and selection["event_prompt_match"] is False:
            return None
        if cls._single_event_only:
            if selection["has_prompt_feedback"]:
                if selection["prompt_event_count"] != 1:
                    return None
            elif event_count != 1:
                return None

        snapshot = selection["snapshot"]
        event = snapshot.get("event", {}) or {}
        if not cls._event_matches_filters(event):
            return None
        prompt_world_state, world_state_source = cls._prompt_time_world_state()
        event_world_state = copy.deepcopy(snapshot.get("world_state", {}) or {})
        if not prompt_world_state:
            prompt_world_state = copy.deepcopy(event_world_state)
            world_state_source = "event_snapshot_fallback"

        return {
            "timestamp": datetime.now().isoformat(),
            "data_tag": cls._data_tag,
            "prompt": prompt,
            "world_state": prompt_world_state,
            "event_world_state": event_world_state,
            "event": event,
            "event_history": [
                copy.deepcopy(item.get("event", {}) or {}) for item in snapshots
            ],
            "prompt_feedback_data": selection["prompt_feedback_data"],
            "prompt_event_history": selection["prompt_event_history"],
            "meta": {
                "goal": gt.get("goal"),
                "goal_type": gt.get("goal_type"),
                "task_plan_brief": gt.get("task_plan_brief"),
                "planner_mode": gt.get("planner_mode"),
                "event_count": event_count,
                "prompt_event_count": selection["prompt_event_count"],
                "prompt_event_reasons": selection["prompt_event_reasons"],
                "snapshot_event_reasons": cls._event_reason_values(event),
                "event_prompt_match": selection["event_prompt_match"],
                "multi_event_prompt": selection["prompt_event_count"] > 1,
                "replan_index": replan_index,
                "records_written_before": records_written,
                "world_state_source": world_state_source,
            },
        }

    @classmethod
    def _write_replan_record(cls, context: Any, record: Dict[str, Any]) -> bool:
        gt = cls._get_generated_text(context)
        if gt is None:
            return False
        try:
            file_path = cls._workspace_file(gt, "replan_records.jsonl")
            if file_path is None:
                return False

            with file_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

            records_written = int(gt.get("replan_capture_records_written", 0) or 0)
            gt["replan_capture_records_written"] = records_written + 1
            logger.info(f"[ReplanDatasetRecorder] replan sample written to {file_path}")
            return True
        except Exception as e:
            logger.error(f"[ReplanDatasetRecorder] write replan record error: {e}", exc_info=True)
            return False

    @classmethod
    def snapshot_newcase(cls,
                         context: Any,
                         scene_graph: Any,
                         event_content: Dict[str, Any]) -> None:
        """
        Capture a world state snapshot when NewCaseEvent fires and store it in
        context._generated_text['replan_snapshots'].
        """
        if not cls._enabled:
            return
        if context is None or not hasattr(context, "_generated_text"):
            return
        if scene_graph is None:
            return

        try:
            snapshot = {
                "timestamp": datetime.now().isoformat(),
                "event": copy.deepcopy(event_content) if event_content else {},
                "world_state": cls._world_state_from_scene_graph(scene_graph),
            }
            history = context._generated_text.setdefault("replan_snapshots", [])
            history.append(snapshot)
        except Exception as e:
            logger.error(f"[ReplanDatasetRecorder] snapshot_newcase error: {e}", exc_info=True)

    @classmethod
    def handle_replan_prompt(cls,
                             context: Any,
                             prompt: str) -> bool:
        """
        Call after a replan prompt is built:
        - If in the replanning stage, is_replanning=True, select the matching event snapshot from context.
        - `world_state` stores prompt-time state; `event_world_state` stores event-time state.
        - Write to workspace_root/replan_records.jsonl.
        - Return True if one record was written, or False if filters did not match or context is missing.
        - When stop_after_record=True, keep old behavior and raise ReplanSampleCollected after writing.
        """
        record = cls._build_replan_record(context, prompt)
        if record is None:
            return False

        written = cls._write_replan_record(context, record)
        if written and cls._stop_after_record:
            # Tell the caller that this run's sample has been collected and can end.
            raise ReplanSampleCollected("Replan sample collected")
        return written

    @classmethod
    def prepare_replan_prompt(cls, context: Any, prompt: str) -> bool:
        """
        Used when save_llm_io=True: temporarily store a matching replan record,
        then write it after the LLM returns.
        """
        if not cls.save_llm_io_enabled():
            return False

        gt = cls._get_generated_text(context)
        if gt is None:
            return False

        record = cls._build_replan_record(context, prompt)
        if record is None:
            return False
        gt["replan_capture_pending_record"] = record
        return True

    @classmethod
    def finalize_replan_prompt(
        cls,
        context: Any,
        *,
        response: Optional[str] = None,
        error: Optional[str] = None,
    ) -> bool:
        """
        Write the pending replan prompt and LLM output together to replan_records.jsonl.
        """
        if not cls.save_llm_io_enabled():
            return False

        gt = cls._get_generated_text(context)
        if gt is None:
            return False

        record = gt.pop("replan_capture_pending_record", None)
        if not isinstance(record, dict):
            return False

        record["response"] = response
        record["completion"] = [
            {
                "role": "assistant",
                "content": response,
            }
        ] if response is not None else []
        record["messages"] = [
            {
                "role": "user",
                "content": record.get("prompt", ""),
            },
        ]
        if response is not None:
            record["messages"].append(
                {
                    "role": "assistant",
                    "content": response,
                }
            )
        if "task_plan" in gt:
            record["parsed_task_plan"] = copy.deepcopy(gt.get("task_plan"))
        record["llm"] = {
            "node_name": "TaskPlan",
            "success": error is None,
            "error": error,
        }

        written = cls._write_replan_record(context, record)
        if written and cls._stop_after_record and error is None:
            raise ReplanSampleCollected("Replan sample collected")
        return written

    @classmethod
    def record_llm_call(
        cls,
        context: Any,
        *,
        node_name: str,
        prompt: str,
        response: Optional[str] = None,
        is_replanning: Optional[bool] = None,
        error: Optional[str] = None,
    ) -> bool:
        """
        Save each TaskPlan LLM call for reviewing multi-round planning and replanning.
        """
        if not cls.save_llm_io_enabled():
            return False

        gt = cls._get_generated_text(context)
        if gt is None:
            return False
        snapshots = gt.get("replan_snapshots") or []
        try:
            file_path = cls._workspace_file(gt, "llm_trace.jsonl")
            if file_path is None:
                return False

            record = {
                "timestamp": datetime.now().isoformat(),
                "data_tag": cls._data_tag,
                "node_name": node_name,
                "is_replanning": bool(gt.get("is_replanning", False)) if is_replanning is None else bool(is_replanning),
                "prompt": prompt,
                "response": response,
                "error": error,
                "event_history": [
                    copy.deepcopy(item.get("event", {}) or {}) for item in snapshots
                ],
                "meta": {
                    "goal": gt.get("goal"),
                    "goal_type": gt.get("goal_type"),
                    "planner_mode": gt.get("planner_mode"),
                    "event_count": len(snapshots),
                    "replan_index": gt.get("replan_capture_replan_index"),
                    "records_written": gt.get("replan_capture_records_written", 0),
                },
            }
            if "task_plan" in gt:
                record["parsed_task_plan"] = copy.deepcopy(gt.get("task_plan"))

            with file_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            return True
        except Exception as e:
            logger.error(f"[ReplanDatasetRecorder] record_llm_call error: {e}", exc_info=True)
            return False
