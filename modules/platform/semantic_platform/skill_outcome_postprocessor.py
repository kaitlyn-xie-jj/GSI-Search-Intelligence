
from typing import List, Dict, Any, Tuple, Optional
import copy
from modules.platform.semantic_platform.utils.feedback_utils import (
    _get, _join_items, _dedup_robot_list, _same_text
)


class OutcomePostProcessor:
    @staticmethod
    def merge_outcomes(outcomes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not outcomes:
            return []

        # Group containers
        search_groups: Dict[Tuple, List[Dict[str, Any]]] = {}
        patrol_groups: Dict[Tuple, List[Dict[str, Any]]] = {}
        passthrough: List[Dict[str, Any]] = []

        for o in outcomes:
            if not isinstance(o, dict):
                passthrough.append(o); continue

            otype = o.get("type")
            data  = o.get("data", {}) or {}
            meta  = o.get("meta", {}) or {}
            skill = (meta.get("skill") or "").lower()

            if otype == "KNOWLEDGE_ACQUIRED" and data.get("knowledge_type") == "entity_discovery": 
                timestep = meta.get("timestep")
                area_key = OutcomePostProcessor._area_key(data)
                target_sig = OutcomePostProcessor._target_signature(data.get("target_spec"))
                search_groups.setdefault((timestep, "search", area_key, target_sig), []).append(o)
                continue

            if otype == "KNOWLEDGE_ACQUIRED" and data.get("knowledge_type") == "patrol_log":
                timestep = meta.get("timestep")
                area_key = OutcomePostProcessor._area_key(data)
                patrol_groups.setdefault((timestep, area_key), []).append(o)
                continue

            # Pass through other types
            passthrough.append(o)

        merged: List[Dict[str, Any]] = []
        # Merge search entity_discovery
        for _, items in search_groups.items():
            merged.append(OutcomePostProcessor._merge_entity_discovery_group(items))
        # Merge patrol patrol_log
        for _, items in patrol_groups.items():
            merged.append(OutcomePostProcessor._merge_patrol_log_group(items))

        return merged + passthrough

    # ---------------- internal helpers ----------------

    @staticmethod
    def _area_key(data: Dict[str, Any]) -> str:
        # Unified area identification priority: token > area_searched > short string from area(dict)
        token = data.get("area_token")
        if token:
            return f"token:{token}"
        if data.get("area_searched") is not None:
            return f"searched:{data.get('area_searched')}"
        area = data.get("area")
        if isinstance(area, dict) and area.get("kind"):
            # Simplify to a short signature from kind + key geometry parameters
            kind = area.get("kind")
            if kind == "circle":
                c = area.get("center"); r = area.get("radius")
                return f"circle:{c}-{r}"
            if kind == "rectangle":
                return f"rect:{area.get('min')}-{area.get('max')}"
            if kind == "line":
                return f"line:{len(area.get('coords', []))}pts"
            if kind == "area":
                return f"polygon:{len(area.get('coords', []))}pts"
            if kind == "point":
                return f"point:{area.get('coord') or area.get('center')}"
        # Degenerate case: cannot determine
        return "area:unknown"

    @staticmethod
    def _target_signature(spec: Optional[Dict[str, Any]]) -> str:
        """Normalize target_spec into a stable grouping string; missing spec is treated as wildcard."""
        if not spec:
            return "target:*"
        klass = spec.get("class") or "object"
        ttype = spec.get("type") or "*"
        feats = spec.get("features", {}) or {}
        color = feats.get("color", "*")
        sub   = feats.get("subtype", "*")
        return f"{klass}|{ttype}|{color}|{sub}"

    @staticmethod
    def _merge_entity_discovery_group(items: List[Dict[str, Any]]) -> Dict[str, Any]:
        base = copy.deepcopy(items[0])
        bd = base.setdefault("data", {})
        bm = base.setdefault("meta", {})

        # Initialize statistics variables
        all_robots = []
        all_found_ids = set()
        negative_names: List[str] = []
        area_token = bd.get("area_token") or bd.get("area_searched") or "area"
        target_spec = bd.get("target_spec")
        ttxt = OutcomePostProcessor._target_text_from_spec(target_spec)

        # Group results by discovered entity ID signature
        groups: Dict[Tuple[str, ...], Dict[str, Any]] = {}
        for o in items:
            d = o.get("data", {}) or {}
            m = o.get("meta", {}) or {}
            rid = d.get("robot_id") or m.get("robot_id")
            rlabel = m.get("robot_label") or str(rid)
            if rid or rlabel:
                all_robots.append({"id": rid, "label": rlabel})

            found_ids = d.get("found_ids") or []
            if found_ids:
                all_found_ids.update(found_ids)
                # Use the sorted ID tuple as the unique grouping signature
                sig = tuple(sorted(str(x) for x in found_ids))
                if sig not in groups:
                    # If this discovery combination is first seen, create an item and generate description text
                    entities  = d.get("entities") or []
                    names: List[str] = []
                    for ent in entities:
                        label = _get(ent, "properties.label")
                        loc_label = _get(ent, "properties.location.label") or _get(ent, "properties.location.type")
                        if label and loc_label:
                            names.append(f"{label} at {loc_label}")
                        else:
                            names.append(label or str(_get(ent, "id")))
                    
                    groups[sig] = {
                        "robots": [], 
                        "names_text": _join_items(names) if names else ", ".join(map(str, sig))
                    }
                groups[sig]["robots"].append(rlabel)
            else:
                negative_names.append(rlabel)

        # Generate discovery detail messages from grouped results
        discovery_details: List[str] = []
        for sig, info in groups.items():
            rnames = ", ".join(info["robots"])
            names_text = info["names_text"]
            if len(info["robots"]) > 1:
                discovery_details.append(f"{rnames} found the same {ttxt}: {names_text}.")
            else:
                discovery_details.append(f"{rnames} found {ttxt}: {names_text}.")

        # Build final summary message
        robots = _dedup_robot_list(all_robots)
        names_str = ", ".join([r.get("label") or str(r.get("id")) for r in robots]) or "Robots"
        if not all_found_ids:
            summary = f"{names_str} did not find any {ttxt} in {area_token}."
        else:
            summary = f"{names_str} completed the search in {area_token}."
            if discovery_details:
                summary += " " + " ".join(discovery_details)
            if negative_names:
                 summary += f" {', '.join(negative_names)} found nothing."
        
        # Update and return the merged outcome
        bd["robots"] = robots
        bd["found_ids"] = sorted(list(all_found_ids))
        bd["found"] = bool(bd["found_ids"])
        bd["found_count"] = len(bd["found_ids"])
        bd["message"] = summary
        bm["merged_from"] = len(items)
        return base

    @staticmethod
    def _target_text_from_spec(spec: Optional[Dict[str, Any]]) -> str:
        if not spec:
            return "target"
        # Event class
        if (spec.get("class") or "").lower() == "event":
            return str(spec.get("event_type") or "event").replace("_", " ")

        # Object class
        ttype = spec.get("type") or "object"
        feats = (spec.get("features") or {}).copy()
        color          = feats.pop("color", None)
        subtype        = feats.pop("subtype", None)
        clothing_color = feats.pop("clothing_color", None)
        item           = feats.pop("item", None)
        flags: List[str] = []
        for k, v in list(feats.items()):
            if isinstance(v, bool) and v:
                flags.append(str(k).replace("_", " "))
        parts = [ttype]
        if color: parts.append(str(color))
        if subtype: parts.append(str(subtype))
        if clothing_color: parts.append(str(clothing_color) + "_clothing")
        if item: parts.append(str(item))
        for f in flags:
            parts.append(f)
        return "/".join(parts)

    @staticmethod
    def _merge_patrol_log_group(items: List[Dict[str, Any]]) -> Dict[str, Any]:
        base = copy.deepcopy(items[0])
        bd = base.setdefault("data", {})
        bm = base.setdefault("meta", {})
        robots = []
        area_token = bd.get("area_token") or bd.get("area_patrolled") or bd.get("area") or "area"
        target_spec = bd.get("target_spec")
        ttxt = OutcomePostProcessor._target_text_from_spec(target_spec)
        max_dur = 0.0

        # 1) Collect found_ids for each robot
        union_ids: List[str] = []
        seen_union = set()
        groups: Dict[Tuple[str, ...], Dict[str, Any]] = {}
        for o in items:
            d = o.get("data", {}) or {}
            m = o.get("meta", {}) or {}
            rid = d.get("robot_id") or m.get("robot_id")
            rlabel = m.get("robot_label") or str(rid)
            if rid or rlabel:
                robots.append({"id": rid, "label": rlabel})
            fids = d.get("found_ids") or []
            sig = tuple(sorted(str(x) for x in fids)) if fids else tuple()
            if sig not in groups:
                groups[sig] = {"robots": [], "ids": list(sig)}
            groups[sig]["robots"].append(rlabel)

            # Deduplicate union
            for fid in fids:
                sf = str(fid)
                if sf not in seen_union:
                    seen_union.add(sf)
                    union_ids.append(sf)
            dur = d.get("duration_s")
            try:
                max_dur = max(max_dur, float(dur))
            except Exception:
                pass

        # Deduplicate robots
        robots = _dedup_robot_list(robots)
        names_str = ", ".join([r.get("label") or str(r.get("id")) for r in robots]) or "Robots"

        # 2) Generate summary message
        total = len(union_ids)
        if total == 0:
            summary = f"{names_str} completed the patrol in {area_token}; no abnormal detections."
        else:
            details: List[str] = []
            for sig, info in groups.items():
                if not sig:
                    continue  # Do not write the not-found group
                rnames = ", ".join(info["robots"])
                n = len(info["ids"])
                if len(info["robots"]) > 1:
                    # Multiple robots found the same target set
                    details.append(f"{rnames} both found the same {n} {ttxt}.")
                else:
                    # Single robot
                    details.append(f"{rnames} found {n} {ttxt}.")
            detail_txt = " ".join(details).strip()
            summary = f"{names_str} completed the patrol in {area_token}. Detections: {total} {ttxt}."
            if detail_txt:
                summary += f" {detail_txt}"

        # 3) Write back skeleton data
        bd["robots"] = robots
        bd["entities"] = union_ids  
        bd["found"] = total > 0
        bd["found_count"] = total
        bd["found_ids"] = union_ids
        bd["message"] = summary
        if max_dur:
            bd["duration_s"] = max_dur
        bm["merged_from"] = len(items)
        return base
