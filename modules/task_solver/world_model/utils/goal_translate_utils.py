# -*- coding: utf-8 -*-
import ast
import re
from typing import Any, Dict, List, Optional, Type, TypeVar
from enum import Enum

E = TypeVar("E", bound=Enum)

# ---------------------------
# Lightweight parsing and traversal utilities
# ---------------------------

def unquote(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
        return s[1:-1]
    return s

def normalize_param(raw: str):
    v = unquote(raw)
    if v and v[0] in "[(":
        try:
            return ast.literal_eval(v)
        except Exception:
            pass
    return v

def is_coord(val: Any) -> bool:
    if isinstance(val, (list, tuple)) and len(val) == 2:
        return all(isinstance(x, (int, float)) for x in val)
    if isinstance(val, (list, tuple)) and val:
        return all(isinstance(p, (list, tuple)) and len(p) == 2 and
                   all(isinstance(x, (int, float)) for x in p) for p in val)
    return False

def parse_enum_ci(enum_cls: Type[E], s: str) -> Optional[Any]:
    if not s:
        return None
    css = s.strip().casefold()
    if not css:
        return None
    for m in enum_cls:
        if str(m.value).casefold() == css:
            return m.value
    return None

def split_token_parts(token: str) -> List[str]:
    rough = re.split(r"[\s_\-]+", token.strip())
    parts: List[str] = []
    for r in rough:
        if not r:
            continue
        segs = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z])|[0-9]+", r)
        if segs:
            parts.extend(segs)
        else:
            parts.append(r)
    return [p for p in parts if p]

# ---------------------------
# goal/success_condition utilities
# ---------------------------

def _collect_sc_nodes_by_op(node: Any, op_name: str, out: List[Dict[str, Any]]) -> None:
    """Recursively collect success_condition nodes whose 'op' matches op_name."""
    if not isinstance(node, dict):
        return
    if node.get("op") == op_name:
        out.append(node)
    args = node.get("args")
    if isinstance(args, list):
        for ch in args:
            _collect_sc_nodes_by_op(ch, op_name, out)


def find_sc_nodes_by_op(goal_cfg: Dict[str, Any], op_name: str) -> List[Dict[str, Any]]:
    sc = (goal_cfg or {}).get("success_condition") or {}
    out: List[Dict[str, Any]] = []
    _collect_sc_nodes_by_op(sc, op_name, out)
    return out


def _walk_sc_args_dicts(node: Any):
    """Recursively yield 'args' dicts found within a success_condition tree."""
    if not isinstance(node, dict):
        return
    args = node.get("args")
    if isinstance(args, dict):
        yield args
    elif isinstance(args, list):
        for ch in args:
            yield from _walk_sc_args_dicts(ch)


def iter_sc_args_dicts(goal_cfg: Dict[str, Any]):
    sc = (goal_cfg or {}).get("success_condition") or {}
    yield from _walk_sc_args_dicts(sc)

def flatten_context(goal_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    ctx = (goal_cfg or {}).get("context")

    if isinstance(ctx, dict):
        out.append(ctx)
        seq = ctx.get("sequence")
        if isinstance(seq, list):
            for step in seq:
                if isinstance(step, dict):
                    cp = step.get("core_params")
                    if isinstance(cp, dict):
                        blk = {}
                        if "search_area" in cp:
                            blk["search_area"] = cp["search_area"]
                        if "ai_recognition" in cp:
                            blk["ai_recognition"] = cp["ai_recognition"]
                        if "execution_device" in cp:
                            blk["execution_device"] = cp["execution_device"]
                        if blk:
                            out.append(blk)
    elif isinstance(ctx, list):
        out.extend([c for c in ctx if isinstance(c, dict)])

    return out

def get_step_arg(goal_cfg: Dict[str, Any], op_name: str, key: str, default=None):
    nodes = find_sc_nodes_by_op(goal_cfg or {}, op_name)
    if not nodes:
        return default
    args = nodes[0].get("args") or {}
    return args.get(key, default)

def extract_target_from_node_args(args: Dict[str, Any], aliases: Optional[Dict[str, str]] = None) -> Optional[Dict[str, Any]]:
    """
    Extract a normalized target from a success_condition node's args:
    - Event: {"type":"event","event_type": "..."} -> {"class":"event","event_type":"..."}
    - Object, infrastructure, cargo, etc.:
      {"type":"<t>","features":{...},"label": "..."} -> {"class":"object","type":"<t>",...}
    """
    target = args.get("target")
    if not target:
        return None
    if isinstance(target, dict):
        out = dict(target)
        ttype = (out.get("type") or "").strip().lower()
        if ttype == "event" and out.get("event_type"):
            ev = str(out["event_type"])
            if aliases:
                ev = aliases.get(ev.lower(), ev)
            return {"class": "event", "event_type": ev}
        if ttype:
            cand = {"class": "object", "type": ttype}
            if "label" in out:
                cand["label"] = out["label"]
            feats = out.get("features") or {}
            if feats:
                cand["features"] = dict(feats)
            return cand
    return None

def get_message_from_goal(goal_cfg: Dict[str, Any]) -> Optional[str]:
    nodes = find_sc_nodes_by_op(goal_cfg or {}, "SPEAK_DURATION")
    if nodes:
        args = nodes[0].get("args") or {}
        msg = args.get("message")
        if isinstance(msg, str) and msg.strip():
            return msg.strip()
    for ctx in flatten_context(goal_cfg or {}):
        msg = ctx.get("message")
        if isinstance(msg, str) and msg.strip():
            return msg.strip()
    return None

def get_patrol_duration_from_goal(goal_cfg: Dict[str, Any]) -> Optional[float]:
    nodes = find_sc_nodes_by_op(goal_cfg or {}, "PATROL_DURATION")
    if nodes:
        args = nodes[0].get("args") or {}
        if args.get("duration_ge_s") is not None:
            try:
                return float(args["duration_ge_s"])
            except Exception:
                pass
    for ctx in flatten_context(goal_cfg or {}):
        ai = ctx.get("ai_recognition") or {}
        val = ai.get("patrol_duration_s")
        if val is not None:
            try:
                return float(val)
            except Exception:
                pass
    return None
