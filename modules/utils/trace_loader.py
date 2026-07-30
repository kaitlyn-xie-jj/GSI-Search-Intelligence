import json
from pathlib import Path
from typing import Any, Dict, List, Optional


class ReplayEpisode(dict):
    """Replay record container for one planning-execution round."""

    def __init__(
        self,
        skills_by_timestep: Dict[str, Any],
        current_plan_selection: Optional[Any],
        new_case_event: Optional[Dict[str, Any]],
        execution_result: Optional[Dict[str, Any]],
        conversations: Optional[List[Dict[str, str]]] = None,
        dispatcher_result: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self["skills_by_timestep"] = skills_by_timestep
        self["current_plan_selection"] = current_plan_selection
        self["new_case_event"] = new_case_event
        self["execution_result"] = execution_result
        self["conversations"] = conversations or []
        self["dispatcher_result"] = dispatcher_result


class ReplayTrace:
    """Replay trace for an entire run."""

    def __init__(self, run_input: Dict[str, Any], episodes: List[ReplayEpisode]):
        self.run_input = run_input
        self.episodes = episodes
        self._cursor = 0

    def next_episode(self) -> Optional[ReplayEpisode]:
        """Return the next round in order, or None when no rounds remain."""
        if self._cursor >= len(self.episodes):
            return None
        ep = self.episodes[self._cursor]
        self._cursor += 1
        return ep

    def reset(self):
        self._cursor = 0


def load_replay_trace(run_dir: Path, tag: str = "default") -> ReplayTrace:
    """
    Build ReplayTrace from dump_var output.

    Assumes variables are saved as jsonl at run_dir / var_dump.jsonl, with each
    line shaped like {"name": "...", "value": {...}, "meta": {...}}.
    Adjust the file name or field names here if needed.
    """
    dump_path = run_dir / "temp_vars.jsonl"
    run_input: Dict[str, Any] = {}
    episodes: List[ReplayEpisode] = []

    # Current round being accumulated.
    cur_dispatcher = None
    cur_skills = None
    cur_sel = None
    cur_event = None
    cur_exec = None
    cur_conversations: List[Dict[str, str]] = []
    # Temporarily store an unmatched prompt.
    _pending_prompt: Optional[str] = None

    def _flush_episode():
        nonlocal cur_dispatcher, cur_skills, cur_sel, cur_event, cur_exec
        nonlocal cur_conversations, _pending_prompt
        # If an unmatched prompt exists, store it in conversations with an empty response.
        if _pending_prompt is not None:
            cur_conversations.append({"prompt": _pending_prompt, "response": ""})
            _pending_prompt = None
        if cur_skills is None:
            return
        episodes.append(
            ReplayEpisode(
                skills_by_timestep=cur_skills,
                current_plan_selection=cur_sel,
                new_case_event=cur_event,
                execution_result=cur_exec,
                conversations=cur_conversations,
                dispatcher_result=cur_dispatcher,
            )
        )
        cur_dispatcher = None
        cur_skills = None
        cur_sel = None
        cur_event = None
        cur_exec = None
        cur_conversations = []

    with dump_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            name = rec.get("name")
            val = rec.get("value")

            # ---- Run-level input, initial configuration ----
            if name == "run_input_default":
                run_input = val or {}
                continue
            if name == "goal" and not run_input:
                # Older logs may lack run_input_default, so fill some information from goal.
                try:
                    run_input = {
                        "goal_id": val.get("id"),
                        "goal_type": val.get("goal_type"),
                    }
                except Exception:
                    pass
                continue

            # ---- Accumulate episodes by round ----
            if name == "prompt":
                # If an unmatched prompt exists, store it first with an empty response.
                if _pending_prompt is not None:
                    cur_conversations.append({"prompt": _pending_prompt, "response": ""})
                _pending_prompt = val if isinstance(val, str) else str(val) if val is not None else ""
                continue
            if name == "response":
                resp_text = val if isinstance(val, str) else str(val) if val is not None else ""
                if _pending_prompt is not None:
                    cur_conversations.append({"prompt": _pending_prompt, "response": resp_text})
                    _pending_prompt = None
                else:
                    # Response has no matching prompt.
                    cur_conversations.append({"prompt": "", "response": resp_text})
                continue
            if name == "dispatcher_result":
                cur_dispatcher = val
                continue
            if name == "skills_by_timestep":
                cur_skills = val
                continue
            if name == "current_plan_selection":
                cur_sel = val
                continue
            if name == "new_case_event":
                cur_event = val
                continue
            if name == "execution_result":
                cur_exec = val
                _flush_episode()
                continue

    _flush_episode()

    return ReplayTrace(run_input=run_input, episodes=episodes)
