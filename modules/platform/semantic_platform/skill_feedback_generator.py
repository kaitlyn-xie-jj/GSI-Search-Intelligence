import numpy as np
from typing import Dict, Any, List, Optional
from modules.config.base.enums import BuildingName, TransFacilityType
from modules.platform.semantic_platform.utils.feedback_utils import (
    _get, _join_items, _same_text, _entity_location_texts
)
from modules.utils.location_utils import extract_object_position

def _fmt_area_from_params(params: Optional[Dict]) -> str:
    if not params:
        return "unknown area"
    token = params.get("area_token")
    if token:
        return str(token)
    kind = _get(params, "area.kind")
    if kind == "line":       return "line segment"
    if kind == "area":       return "polygon area"
    if kind == "point":      return "point"
    if kind == "rectangle":  return "rectangle area"
    if kind == "circle":     return "circular area"
    return "unknown area"

def _fmt_target_from_params(params: Optional[Dict]) -> str:
    if not params:
        return "target"

    tgt = params.get("target") or {}
    tclass = tgt.get("class")

    # Event class
    if tclass == "event":
        et = tgt.get("event_type") or _get(tgt, "features.subtype") or "event"
        et = str(et).replace("_", " ")
        return f"{et} event"

    # Object class
    otype = tgt.get("type") or "object"
    feats = (tgt.get("features") or {}).copy()

    BUILDING_TYPES = {e.value for e in BuildingName}
    INFRA_TYPES = {e.value for e in TransFacilityType}
    if str(otype).lower() in (BUILDING_TYPES | INFRA_TYPES):
        label = feats.get("label")
        if label:
            return str(label)

    color          = feats.pop("color", None)
    subtype        = feats.pop("subtype", None)
    clothing_color = feats.pop("clothing_color", None)
    item           = feats.pop("item", None)

    salient_bool_keys = ("suspicious", "armed", "crowd")
    flags: List[str] = []
    for k in salient_bool_keys:
        v = feats.pop(k, None)
        if isinstance(v, bool) and v:
            flags.append(k.replace("_", " "))

    for k, v in list(feats.items()):
        if isinstance(v, bool) and v:
            flags.append(str(k).replace("_", " "))
            feats.pop(k, None)

    tok = str(params.get("target_token") or "")
    if ("suspicious" in tok.lower()) and ("suspicious" not in flags):
        flags.append("suspicious")

    parts = [otype]
    if color: parts.append(str(color))
    if subtype: parts.append(str(subtype))
    if clothing_color: parts.append(str(clothing_color) + "_clothing")
    if item: parts.append(str(item))
    for f in flags:
        parts.append(f)

    return "/".join(parts)

def _robot_name(ctx: Dict) -> str:
    return ctx.get("robot_label") or (f"Robot-{ctx.get('robot_id')}" if ctx.get("robot_id") is not None else "Robot")

def _secs_text(val: Optional[float]) -> str:
    if val is None:
        return "0s"
    try:
        v = float(val)
    except Exception:
        return f"{val}s"
    if abs(v - round(v)) < 1e-6:
        return f"{int(round(v))}s"
    return f"{v:.1f}s"

def _best_target_text_from_params(params: Optional[Dict], label: Optional[str], default_word: str = "target") -> str:
    txt = _fmt_target_from_params(params)
    if txt in ("target", "object") and label:
        return label
    return txt or (label or default_word)

def _get_entity_pos(entity_id: Optional[int], context: Dict[str, Any]) -> Optional[List[float]]:
    """Utility function: get an entity position from the graph in context."""
    if entity_id is None:
        return None
    graph = context.get("graph")
    if not graph:
        return None
    node = graph.get_node_by_id(entity_id)
    if not node:
        return None
    return extract_object_position(node)

def _calculate_distance(id1: Optional[int], id2: Optional[int], context: Dict[str, Any]) -> Optional[float]:
    """Utility function: calculate the Euclidean distance between two entities."""
    pos1 = _get_entity_pos(id1, context)
    pos2 = _get_entity_pos(id2, context)

    if pos1 is None or pos2 is None:
        return None

    try:
        distance = np.linalg.norm(np.array(pos1) - np.array(pos2))
        return float(distance)
    except Exception:
        return None

class FeedbackGenerator:
    """
    Generate readable feedback from params/context only and merge it into outcomes.
    """

    @staticmethod
    def merge_into_outcomes(skill: str,
                            success: bool,
                            params: Dict[str, Any],
                            context: Dict[str, Any],
                            skill_result: Dict[str, Any],
                            skill_info: Dict[str, Any],
                            outcomes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        s = (skill or "").lower()
        if s == "search":
            return FeedbackGenerator._merge_search(success, params, context, skill_result, outcomes)
        if s == "take_off":
            return FeedbackGenerator._merge_take_off(success, params, context, outcomes)
        if s == "return_home":
            return FeedbackGenerator._merge_return_home(success, context, outcomes)
        if s == "navigate":
            return FeedbackGenerator._merge_navigate(success, params, context, outcomes)
        if s == "take_photo":
            return FeedbackGenerator._merge_take_photo(success, params, context, outcomes)
        if s == "follow":
            return FeedbackGenerator._merge_follow(success, params, context, outcomes)
        if s == "broadcast":
            return FeedbackGenerator._merge_broadcast(success, params, context, outcomes)
        if s == "place":
            return FeedbackGenerator._merge_place(success, params, context, outcomes)
        if s == "handle_hazard":
            return FeedbackGenerator._merge_handle_hazard(success, params, context, outcomes)
        if s == "guide":
            return FeedbackGenerator._merge_guide(success, params, context, outcomes)
        return outcomes

    # ---------- helpers ----------
    @staticmethod
    def _find_or_create_entity_discovery(outcomes: List[Dict[str, Any]],
                                         context: Dict[str, Any],
                                         area_hint: Any) -> Dict[str, Any]:
        idx = next((i for i, o in enumerate(outcomes)
                    if o.get("type") == "KNOWLEDGE_ACQUIRED"
                    and (o.get("data", {}) or {}).get("knowledge_type") == "entity_discovery"), None)
        if idx is not None:
            return outcomes[idx]["data"]
        data = {
            "knowledge_type": "entity_discovery",
            "robot_id": context.get("robot_id"),
            "robot_type": context.get("robot_type"),
            "area_searched": area_hint,
            "entities": []
        }
        outcomes.append({"type": "KNOWLEDGE_ACQUIRED", "data": data})
        return data
    
    @staticmethod
    def _ensure_knowledge_outcome(outcomes: List[Dict[str, Any]],
                                  knowledge_type: str,
                                  defaults: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Ensure a KNOWLEDGE_ACQUIRED outcome exists; create a base item if missing and return its data.
        """
        idx = next((i for i, o in enumerate(outcomes)
                    if o.get("type") == "KNOWLEDGE_ACQUIRED"
                    and (o.get("data", {}) or {}).get("knowledge_type") == knowledge_type), None)
        if idx is not None:
            return outcomes[idx].setdefault("data", {})

        data = {"knowledge_type": knowledge_type}
        if defaults:
            data.update(defaults)
        outcomes.append({"type": "KNOWLEDGE_ACQUIRED", "data": data})
        return data
    
    @staticmethod
    def _get_or_create_knowledge_data(outcomes: List[Dict[str, Any]],
                                      knowledge_type: str,
                                      defaults: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        if not outcomes:
            data = {"knowledge_type": knowledge_type}
            if defaults:
                data.update(defaults)
            outcomes.append({"type": "KNOWLEDGE_ACQUIRED", "data": data})
            return data
        else:
            for o in outcomes:
                if o.get("type") == "KNOWLEDGE_ACQUIRED":
                    return o.setdefault("data", {})
        return None
    
    # ---------- skill merges ----------
    @staticmethod
    def _merge_search(success: bool,
                    params: Dict[str, Any],
                    context: Dict[str, Any],
                    skill_result: Dict[str, Any],
                    outcomes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        search_type = params.get('goal_type', 'area_search')
        robot = _robot_name(context)
        area_txt   = _fmt_area_from_params(params)          # Prefer area_token
        target_txt = _fmt_target_from_params(params)
        found_ids  = params.get("target_ids") or skill_result.get("found_ids") or []
        item_texts = _entity_location_texts(found_ids, context)
        if search_type == 'patrol':
            dur = params.get("execution_time") or params.get("duration_ge_s")
            dur_txt = _secs_text(dur)
            if success and found_ids:
                if _same_text(target_txt, area_txt) or target_txt in ("target", "object"):
                    message = f"{robot} patrolled {area_txt} for {dur_txt} and found {len(found_ids)}: {_join_items(item_texts)}."
                else:
                    tail = f": {_join_items(item_texts)}" if item_texts else ""
                    message = f"{robot} patrolled {area_txt} for {dur_txt} and found {len(found_ids)} {target_txt}{tail}."
            elif success:
                message = f"{robot} patrolled {area_txt} for {dur_txt} and found nothing suspicious."
            else:
                message = f"{robot}'s patrol failed in {area_txt}."

            outcomes = []
            d = FeedbackGenerator._ensure_knowledge_outcome(
                outcomes, knowledge_type="patrol_log", defaults={"robot_id": context.get("robot_id"), "robot_type": context.get("robot_type")}
            )
            d["message"] = message
            d["area"] = params.get("area_token") or params.get("area")
            d["duration_s"] = float(dur) if dur is not None else 0.0
            d["found"] = bool(found_ids)
            d["found_ids"] = list(found_ids)
            d["target_spec"] = params.get("target", {})
            return outcomes
        else: # search_type == 'search'
            if success and found_ids:
                if _same_text(target_txt, area_txt):
                    message = f"{robot} found {len(found_ids)} targets: {_join_items(item_texts)}."
                else:
                    tail = f": {_join_items(item_texts)}" if item_texts else ""
                    message = f"{robot} found {len(found_ids)} {target_txt} in {area_txt}{tail}."
            elif success:
                message = f"{robot} did not find any {target_txt} in {area_txt}."
            else:
                message = f"{robot}'s search failed in {area_txt}."

            area_hint = params.get("area_token") or params.get("area")
            d = FeedbackGenerator._find_or_create_entity_discovery(outcomes, context, area_hint)
            d["message"] = message
            d["found"] = bool(found_ids)
            d["found_ids"] = found_ids
            d["target_spec"] = params.get("target", {})
            if params.get("conf_ge") is not None: d["conf_ge"] = params["conf_ge"]
            if params.get("persist_ge_s") is not None: d["persist_ge_s"] = params["persist_ge_s"]
            return outcomes
        
    @staticmethod
    def _merge_take_off(success: bool,
                        params: Dict[str, Any],
                        context: Dict[str, Any],
                        outcomes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        robot = _robot_name(context)
        alt = (params.get("target_altitude") if params is not None else None)
        alt_txt = None
        alt_val: Optional[float] = None
        if alt is not None:
            alt_val = float(alt)
            alt_txt = f"{alt_val:.0f}m" if abs(alt_val - round(alt_val)) < 1e-6 else f"{alt_val:.1f}m"

        # Duration
        dur = params.get("execution_time") if params else None
        if dur is None:
            dur = params.get("duration_ge_s") if params else None
        dur_txt = _secs_text(dur)

        # Generate message
        if success:
            if alt_txt:
                msg = f"{robot} took off to {alt_txt} successfully in {dur_txt}."
            else:
                msg = f"{robot} took off successfully in {dur_txt}."
        else:
            if alt_txt:
                msg = f"{robot} failed to take off to {alt_txt}."
            else:
                msg = f"{robot} failed to take off."

        # Write/create KNOWLEDGE_ACQUIRED
        d = FeedbackGenerator._get_or_create_knowledge_data(
            outcomes, knowledge_type="takeoff_log", defaults={"robot_id": context.get("robot_id"), "robot_type": context.get("robot_type")}
        )
        if d is not None:
            d["message"] = msg
            if alt_val is not None:
                d["target_altitude_m"] = alt_val

        return outcomes

    @staticmethod
    def _merge_return_home(success: bool,
                           context: Dict[str, Any],
                           outcomes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        robot = _robot_name(context)
        msg = f"{robot} returned to base successfully." if success else f"{robot} failed to return to base."
        for o in outcomes:
            if o.get("type") == "KNOWLEDGE_ACQUIRED":
                o.setdefault("data", {})["message"] = msg
                return outcomes
        # Create if missing
        d = FeedbackGenerator._get_or_create_knowledge_data(
            outcomes, knowledge_type="return_log", defaults={"robot_id": context.get("robot_id"), "robot_type": context.get("robot_type")}
        )
        if d:
            d["message"] = msg
        return outcomes

    @staticmethod
    def _merge_navigate(success: bool,
                        params: Dict[str, Any],
                        context: Dict[str, Any],
                        outcomes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        robot = _robot_name(context)

        # Destination text prefers area_token, then object label, then target text
        area_tok = params.get("area_token")
        dest_id = params.get("object_id") or context.get("object_id")
        graph = context.get("graph")
        label = None
        if graph and dest_id is not None:
            node = graph.get_node_by_id(dest_id)
            label = _get(node, "properties.label", str(dest_id))

        if area_tok:
            dest_txt = str(area_tok)
        elif label:
            dest_txt = label
        else:
            dest_txt = _best_target_text_from_params(params, label, "destination")

        msg = f"{robot} navigated to {dest_txt} successfully." if success else f"{robot} failed to navigate to {dest_txt}."
        d = FeedbackGenerator._get_or_create_knowledge_data(
            outcomes, knowledge_type="navigation_log", defaults={"robot_id": context.get("robot_id"), "robot_type": context.get("robot_type")}
        )
        if d:
            d["message"] = msg
            d["dest_id"] = dest_id
            d["dest_label"] = label or 'destination'
            d["destination_text"] = dest_txt
        return outcomes

    @staticmethod
    def _merge_take_photo(success: bool,
                          params: Dict[str, Any],
                          context: Dict[str, Any],
                          outcomes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        robot = _robot_name(context)
        robot_id = context.get("robot_id")
        tid = params.get("object_id") or context.get("object_id")
        graph = context.get("graph")
        tlabel = None
        if graph and tid is not None:
            tlabel = _get(graph.get_node_by_id(tid), "properties.label", str(tid))
        target_txt = _best_target_text_from_params(params, tlabel, "target")

        msg = f"{robot} captured a photo of {target_txt} successfully." if success and tid else f"{robot} failed to capture a photo of {target_txt}."
        if tid is None:
            msg += " (no target)"

        d = None
        for o in outcomes:
            if o.get("type") == "KNOWLEDGE_ACQUIRED":
                d = o.setdefault("data", {})
                break
        d = FeedbackGenerator._get_or_create_knowledge_data(
            outcomes, knowledge_type="photo", defaults={"robot_id": context.get("robot_id"), "robot_type": context.get("robot_type"), "target_id": tid}
        )
        if d:
            d["message"] = msg
            d["target_id"] = d.get("target_id", tid)
            d["target_label"] = tlabel or 'target'
            d["target_text"] = target_txt
            if success and robot_id is not None and tid is not None:
                distance = _calculate_distance(robot_id, tid, context)
                if distance is not None:
                    d["robot_target_distance"] = distance
        return outcomes

    @staticmethod
    def _merge_follow(success: bool,
                     params: Dict[str, Any],
                     context: Dict[str, Any],
                     outcomes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        robot = _robot_name(context)
        tid = params.get("object_id") or context.get("object_id")
        graph = context.get("graph")
        tlabel = None
        if graph and tid is not None:
            tlabel = _get(graph.get_node_by_id(tid), "properties.label", str(tid))

        target_txt = _best_target_text_from_params(params, tlabel, "target")
        exec_time = params.get("execution_time")
        if exec_time is None:
            exec_time = params.get("duration_ge_s")
        msg = (f"{robot} followed {target_txt} for {_secs_text(exec_time)}."
               if success else
               f"{robot} failed to follow {target_txt}.")
        for o in outcomes:
            if o.get("type") == "KNOWLEDGE_ACQUIRED":
                d = o.setdefault("data", {})
                d["message"] = msg
                d["target_id"] = d.get("target_id", tid)
                d["target_label"] = tlabel or 'target'
                d["target_text"]  = target_txt
                if exec_time is not None:
                    try:
                        d["duration_s"] = float(exec_time)
                    except Exception:
                        d["duration_s"] = exec_time
                return outcomes
        return outcomes
    
    @staticmethod
    def _merge_broadcast(success: bool,
                         params: Dict[str, Any],
                         context: Dict[str, Any],
                         outcomes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        robot = _robot_name(context)
        robot_id = context.get("robot_id")
        tid = params.get("object_id") or context.get("object_id")
        msg_text = params.get("message") or ""
        msg = f"{robot} broadcasted '{msg_text}'." if success else f"{robot} failed to broadcast '{msg_text}'."

        # Reuse/create broadcast outcome
        d = None
        for o in outcomes:
            if o.get("type") == "KNOWLEDGE_ACQUIRED" and (o.get("data", {}) or {}).get("knowledge_type") in ("broadcast_event", "broadcast_log"):
                d = o.setdefault("data", {})
                break
        d = FeedbackGenerator._get_or_create_knowledge_data(
            outcomes, knowledge_type="broadcast_event", defaults={"robot_id": context.get("robot_id"), "robot_type": context.get("robot_type"), "message_text": msg_text}
        )
        if d:
            d["message"] = msg
            d["message_text"] = msg_text
            dur = params.get("execution_time", params.get("duration_ge_s"))
            if dur is not None:
                try:
                    d["duration_s"] = float(dur)
                except Exception:
                    d["duration_s"] = dur
            if success and robot_id is not None and tid is not None:
                distance = _calculate_distance(robot_id, tid, context)
                if distance is not None:
                    d["robot_target_distance"] = distance
        return outcomes

    @staticmethod
    def _merge_place(success: bool,
                     params: Dict[str, Any],
                     context: Dict[str, Any],
                     outcomes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        robot = _robot_name(context)
        graph = context.get("graph")
        obj_id = params.get("object_id") or context.get("object_id")
        obj_label = context.get("target_label")
        if not obj_label and graph and obj_id is not None:
            obj_label = _get(graph.get_node_by_id(obj_id), "properties.label", str(obj_id))
        obj_txt = _best_target_text_from_params(params, obj_label, "object")

        # Parse surface_target: supports structured format and legacy string format
        surface_target = params.get('surface_target', {})
        if isinstance(surface_target, dict):
            surface_class = surface_target.get('class', '')
        else:
            surface_class = 'robot' if surface_target == 'ugv' else ('ground' if surface_target == 'ground' else '')

        if surface_class == 'robot':
            # Load onto carrier
            car_id = params.get("carrier_id") or context.get("carrier_id")
            car_label = None
            if graph and car_id is not None:
                car_label = _get(graph.get_node_by_id(car_id), "properties.label", str(car_id))
            msg = (f"{robot} loaded {obj_txt} onto {car_label or 'carrier'}."
                   if success else
                   f"{robot} failed to load {obj_txt} onto {car_label or 'carrier'}.")
            outcomes = []
            d = FeedbackGenerator._get_or_create_knowledge_data(
                outcomes, "place_log", {"robot_id": context.get("robot_id"), "robot_type": context.get("robot_type"), "object_id": obj_id, "carrier_id": car_id}
            )
            if d:
                d["message"] = msg
                d["object_text"] = obj_txt
                d["carrier_label"] = car_label or 'carrier'
        elif surface_class == 'ground':
            # Unload to the ground
            loc_label = context.get("robot_location_label") or "the current location"
            msg = f"{robot} unloaded {obj_txt} at {loc_label}." if success else f"{robot} failed to unload {obj_txt}."
            outcomes = []
            d = FeedbackGenerator._get_or_create_knowledge_data(
                outcomes, "place_log", {"robot_id": context.get("robot_id"), "robot_type": context.get("robot_type"), "object_id": obj_id}
            )
            if d:
                d["message"] = msg
                d["object_text"] = obj_txt
                d["location_label"] = loc_label
        else:
            # Place on another object's surface
            surface_id = params.get("surface_id")
            surface_label = context.get("surface_label", "the surface")
            if not context.get("surface_label") and graph and surface_id is not None:
                surface_label = _get(graph.get_node_by_id(surface_id), "properties.label", str(surface_id))
            msg = (f"{robot} successfully placed {obj_txt} onto {surface_label}."
                   if success else
                   f"{robot} failed to place {obj_txt} onto {surface_label}.")
            d = FeedbackGenerator._get_or_create_knowledge_data(
                outcomes, "place_log", {"robot_id": context.get("robot_id"), "robot_type": context.get("robot_type"), "object_placed_id": obj_id, "surface_id": surface_id}
            )
            if d:
                d["message"] = msg

        return outcomes

    @staticmethod
    def _merge_handle_hazard(success: bool,
                             params: Dict[str, Any],
                             context: Dict[str, Any],
                             outcomes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        robot = _robot_name(context)
        hazard_id = context.get("object_id")
        hazard_label = context.get("target_label")
        hazard_txt = _best_target_text_from_params(params, hazard_label, "the hazard")
        msg = (f"{robot} successfully handled {hazard_txt}."
               if success else
               f"{robot} failed to handle {hazard_txt}.")

        d = FeedbackGenerator._get_or_create_knowledge_data(
            outcomes, knowledge_type="handle_log", defaults={"robot_id": context.get("robot_id"), "robot_type": context.get("robot_type"), "hazard_id": hazard_id}
        )
        if d:
            d["message"] = msg
        return outcomes

    @staticmethod
    def _merge_guide(success: bool,
                     params: Dict[str, Any],
                     context: Dict[str, Any],
                     outcomes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        robot = _robot_name(context)
        graph = context.get("graph")

        # Get guided entity text
        guided_id = context.get("object_id")
        guided_label = context.get("target_label")
        guided_txt = _best_target_text_from_params(params, guided_label, "the target")
        
        # Get destination text
        dest_id = params.get("destination_id")
        dest_label = params.get("destination_label")
        if graph and dest_id is not None and dest_label is None:
            node = graph.get_node_by_id(dest_id)
            dest_label = _get(node, "properties.label", str(dest_id))
        dest_txt = dest_label or "the destination"
        msg = (f"{robot} successfully guided {guided_txt} to {dest_txt}."
               if success else
               f"{robot} failed to guide {guided_txt} to {dest_txt}.")

        d = FeedbackGenerator._get_or_create_knowledge_data(
            outcomes, knowledge_type="guide_log", defaults={"robot_id": context.get("robot_id"), "robot_type": context.get("robot_type"), "guided_entity_id": guided_id, "dest_id": dest_id}
        )
        if d:
            d["message"] = msg
        return outcomes
