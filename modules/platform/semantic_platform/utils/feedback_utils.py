# -*- coding: utf-8 -*-
from typing import Any, Dict, List, Optional

def _get(d: Optional[dict], path: str, default=None):
    """
    Safe deep get: _get(obj, "a.b.c", default)
    """
    cur = d or {}
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def _join_items(items: List[str], limit: int = 5) -> str:
    """
    Join multiple items into concise text, use ellipsis if exceeds limit.
    """
    if not items:
        return ""
    if len(items) <= limit:
        return "; ".join(items)
    return "; ".join(items[:limit]) + " ..."

def _same_text(a: Optional[str], b: Optional[str]) -> bool:
    """Case and whitespace insensitive text equality (simplified: trim whitespace only)."""
    return (a or "").strip() == (b or "").strip()

from typing import List, Any, Iterable

def _entity_location_texts(
    found_ids: Iterable[Any],
    context: Dict[str, Any],
) -> List[str]:
    """
    Get entity names and locations from graph by found_ids, returns strings like:
      - "Vehicle-14 at Intersection-13"
      - "Person-19" (omits 'at ...' when location category is area)
    Fallback: uses id string when graph unavailable or node not found.
    """
    texts: List[str] = []
    graph = context.get("graph")
    if not graph:
        return [str(eid) for eid in found_ids]
    for eid in found_ids:
        try:
            node = graph.get_node_by_id(eid)
        except Exception:
            node = None
        if not node:
            texts.append(str(eid))
            continue
        props = node.get("properties", {}) or {}
        label = props.get("label") or str(eid)
        loc = props.get("location")
        # Only show "at XXX" when location exists and its category is not 'area'
        loc_cat = (loc or {}).get("category")
        if isinstance(loc, dict) and (str(loc_cat).lower() != "area"):
            loc_label = loc.get("label") or loc.get("type")
            if loc_label:
                texts.append(f"{label} at {loc_label}")
                continue
        texts.append(label)
    return texts


def _dedup_robot_list(robots: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Deduplicate robot list (by id+label).
    """
    seen = set()
    out = []
    for r in robots:
        key = (r.get("id"), r.get("label"))
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out
