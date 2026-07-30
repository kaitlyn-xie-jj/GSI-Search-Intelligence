# -*- coding: utf-8 -*-
from typing import Any, Dict, Optional, List
from modules.config.base.enums import AssemblyComponentType


# --------------------------------
# Logical combinators
# --------------------------------
def _pct_to_float(pct: int) -> float:
    return max(0.0, min(1.0, pct / 100.0))


def and_(*exprs: Dict[str, Any]) -> Dict[str, Any]:
    xs = [e for e in exprs if e]
    return xs[0] if len(xs) == 1 else {"op": "AND", "args": xs}


def or_(*exprs: Dict[str, Any]) -> Dict[str, Any]:
    xs = [e for e in exprs if e]
    return xs[0] if len(xs) == 1 else {"op": "OR", "args": xs}


def not_(expr: Dict[str, Any]) -> Dict[str, Any]:
    return {"op": "NOT", "args": [expr]}


def _norm_area(a: Any) -> Any:
    return a if a is not None else {"area_type": "None"}


# --------------------------------
# Atomic predicates (args must contain area / target; use None if absent)
# --------------------------------
def DETECTED(
    area: Any, target: Dict[str, Any], conf_ge: float, persist_ge_s: float
) -> Dict[str, Any]:
    return {
        "op": "DETECTED",
        "args": {
            "area": _norm_area(area),
            "target": target or None,
            "conf_ge": conf_ge,
            "persist_ge_s": persist_ge_s,
        },
    }


def EVENT(
    area: Any,
    event_type: str,
    conf_ge: float,
    persist_ge_s: float,
    constraints: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    tgt = {"type": "event", "event_type": event_type}
    return {
        "op": "EVENT",
        "args": {
            "area": _norm_area(area),
            "target": tgt,
            "conf_ge": conf_ge,
            "persist_ge_s": persist_ge_s,
            "constraints": constraints or {},
        },
    }


def FOLLOWED(
    area: Any,
    target_ref: Dict[str, Any],
    robot_set: Dict[str, Any],
    duration_ge_s: float,
) -> Dict[str, Any]:
    return {
        "op": "FOLLOWED",
        "args": {
            "area": _norm_area(area),
            "target": target_ref or None,
            "robot_set": robot_set,
            "duration_ge_s": duration_ge_s,
        },
    }


def PHOTO_TAKEN(area: Any, target_ref: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "op": "PHOTO_TAKEN",
        "args": {"area": _norm_area(area), "target": target_ref or None},
    }


def SPEAK_DURATION(
    area: Any,
    target: Dict[str, Any],
    robot_set: Dict[str, Any],
    duration_ge_s: float,
    message: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "op": "SPEAK_DURATION",
        "args": {
            "area": _norm_area(area),
            "target": target or None,
            "robot_set": robot_set,
            "duration_ge_s": duration_ge_s,
            "message": message,
        },
    }


def PATROL_DURATION(
    area: Any,
    robot_set: Dict[str, Any],
    duration_ge_s: float,
    target: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "op": "PATROL_DURATION",
        "args": {
            "area": _norm_area(area),
            "target": target or None,
            "robot_set": robot_set,
            "duration_ge_s": duration_ge_s,
        },
    }


def STATE(area: Any, target: Dict[str, Any], key: str, equals: Any) -> Dict[str, Any]:
    return {
        "op": "STATE",
        "args": {
            "area": _norm_area(area),
            "target": target or None,
            "key": key,
            "equals": equals,
        },
    }


def SAFE(area: Any, target: Dict[str, Any]) -> Dict[str, Any]:
    return STATE(area=area, target=target, key="is_safe", equals=True)


def INSTALLED(area: Any, component_ref: Dict[str, Any]) -> Dict[str, Any]:
    return STATE(area=area, target=component_ref, key="is_installed", equals=True)


# --------------------------------
# Target type normalization
# --------------------------------
def _canonicalize_target_from_ai(spec: dict) -> dict:
    """
    Accept {type, features?, event_type?} and normalize to:
      - type in {vehicle, boat, person, cargo, fire, event, building, trans_facility}
      - For vehicle/boat, secondary type goes into features.subtype
      - For events, record in features.event_type
    """
    if not spec:
        return {}
    base = str(spec.get("type", "")).lower()
    feats = dict(spec.get("features", {}))
    if base in ("vehicle", "boat") and "type" in feats:
        feats["subtype"] = feats.pop("type")
    if base == "event" and "event_type" in spec:
        feats["event_type"] = spec["event_type"]
    return {"type": base, "features": feats}


# --------------------------------
# Success condition builders
# --------------------------------
def build_success_for_area_search(
    goal_details: Dict[str, Any], defaults: Dict[str, Any]
) -> Dict[str, Any]:
    core = goal_details["core_params"]
    area = core["search_area"]
    ai = core["ai_recognition"]
    conf = _pct_to_float(
        ai.get("confidence_threshold_percent", defaults["conf_pct_fallback"])
    )
    if ai.get("type") == "event" or ("event_type" in ai):
        return EVENT(
            area,
            ai.get("event_type", "unknown_event"),
            conf,
            defaults["event_persist_s"],
        )
    tgt = _canonicalize_target_from_ai(ai)
    return DETECTED(area, tgt, conf, defaults["detect_persist_s"])


def build_success_for_target_following(
    goal_details: Dict[str, Any], defaults: Dict[str, Any]
) -> Dict[str, Any]:
    core = goal_details["core_params"]
    ai = core["ai_recognition"]
    dev = core["robot_allocation"]
    robot_set = {"type": dev["type"], "quantity": dev["quantity"]}
    t_ref = {"by": "ai_recognition", "spec": _canonicalize_target_from_ai(ai)}
    return FOLLOWED(
        area=None,
        target_ref=t_ref,
        robot_set=robot_set,
        duration_ge_s=defaults["follow_duration_s"],
    )


def build_success_for_traffic_enforcement(
    goal_details: Dict[str, Any], defaults: Dict[str, Any]
) -> Dict[str, Any]:
    core = goal_details["core_params"]
    area = core["location"]
    dev = core["robot_allocation"]
    enf_type = str(core.get("evidence_type", {}).get("type", "")).strip()
    robot_set = {"type": dev["type"], "quantity": dev["quantity"]}

    ev = EVENT(
        area=area,
        event_type="illegal_parking",
        conf_ge=defaults["event_conf_ge"],
        persist_ge_s=defaults["event_persist_s"],
    )

    if enf_type == "Strict Road":
        ref = {"by": "event_binding", "type": "violating_vehicle"}
        photos = PHOTO_TAKEN(area=area, target_ref=ref)
        return and_(ev, photos)
    elif enf_type == "Non-Strict Road":
        say = SPEAK_DURATION(
            area=area,
            target={"type": "vehicle", "features": {"illegal_parking": True}},
            robot_set=robot_set,
            duration_ge_s=defaults["audio_duration_s"],
            message=None,
        )
        return and_(ev, say)


def build_success_for_transport(
    goal_details: Dict[str, Any], defaults: Dict[str, Any]
) -> Dict[str, Any]:
    core = goal_details["core_params"]
    dst_area = core["destination_area"]
    obj_spec = core.get("object", {})
    conf = _pct_to_float(
        obj_spec.get("confidence_threshold_percent", defaults["conf_pct_fallback"])
    )
    tgt = _canonicalize_target_from_ai(obj_spec)
    return DETECTED(
        area=dst_area,
        target=tgt,
        conf_ge=conf,
        persist_ge_s=defaults["detect_persist_s"],
    )


def build_success_for_evidence_collection(
    goal_details: Dict[str, Any], defaults: Dict[str, Any]
) -> Dict[str, Any]:
    core = goal_details["core_params"]
    obj = core["object"]
    area = core.get("area", None)
    return PHOTO_TAKEN(area=area, target_ref=obj)


def build_success_for_verbal_broadcast(
    goal_details: Dict[str, Any], defaults: Dict[str, Any]
) -> Dict[str, Any]:
    core = goal_details["core_params"]
    dev = core["robot_allocation"]
    robot_set = {"type": dev["type"], "quantity": dev["quantity"]}
    area = core.get("area")
    target = core.get("target_audience")
    msg = core.get("message")
    dur = core.get("min_audio_duration_s", defaults.get("audio_duration_s", 5.0))
    return SPEAK_DURATION(
        area=area, target=target, robot_set=robot_set, duration_ge_s=dur, message=msg
    )


def build_success_for_patrol(
    goal_details: Dict[str, Any], defaults: Dict[str, Any]
) -> Dict[str, Any]:
    core = goal_details["core_params"]
    area = core.get("area")
    ai = core.get("ai_recognition", {}) or {}

    # 1) Duration condition
    dwell_s = float(
        ai.get("patrol_duration_s", defaults.get("follow_duration_s", 10.0))
    )
    dev = core.get("robot_allocation", {"type": "UAV", "quantity": 1})
    robot_set = {"type": dev["type"], "quantity": dev["quantity"]}
    patrol_ok = PATROL_DURATION(
        area=area, robot_set=robot_set, duration_ge_s=dwell_s, target=None
    )

    # 2) Target detection condition (object or event)
    conf = _pct_to_float(
        ai.get("confidence_threshold_percent", defaults["conf_pct_fallback"])
    )
    if ai.get("type") == "event" or ("event_type" in ai):
        detect_ok = EVENT(
            area=area,
            event_type=ai.get("event_type", "unknown_event"),
            conf_ge=conf,
            persist_ge_s=defaults["event_persist_s"],
        )
    else:
        target = _canonicalize_target_from_ai(ai)
        detect_ok = DETECTED(
            area=area,
            target=target,
            conf_ge=conf,
            persist_ge_s=defaults["detect_persist_s"],
        )
    # Either condition suffices
    return or_(patrol_ok, detect_ok)


def build_success_for_assembly(
    goal_details: Dict[str, Any], defaults: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Success condition: components assembled in order with parent-child relationships.
    """
    core = goal_details["core_params"]
    area = core.get("area")
    objs = core.get("object", []) or []
    seq: List[str] = []
    for comp in objs:
        feats = comp.get("features", {}) or {}
        st = str(feats.get("subtype", "")).lower()
        if st:
            seq.append(st)
    preds = []
    prev = AssemblyComponentType.FOUNDATION_BASE.value
    for st in seq:
        if st == AssemblyComponentType.FOUNDATION_BASE.value:
            prev = st
            continue
        comp_ref = {"type": "assembly_component", "features": {"subtype": st}}
        # 1) Component is installed
        preds.append(INSTALLED(area=area, component_ref=comp_ref))
        # 2) Parent-child link correct: parent_component equals previous component subtype
        preds.append(
            STATE(area=area, target=comp_ref, key="parent_component", equals=prev)
        )
        prev = st
    return and_(*preds)


def build_success_for_emergency_response(
    goal_details: Dict[str, Any], defaults: Dict[str, Any]
) -> Dict[str, Any]:
    core = goal_details["core_params"]
    area = core.get("area")
    obj = core.get("object", {}) or {}
    otype = str(obj.get("type", "")).lower()
    feats = dict(obj.get("features", {}) or {})
    preds = []
    # Branch 1: Object has is_fire / is_spill flag (bound entity)
    is_fire = ("is_fire" in feats) and (feats["is_fire"])
    is_spill = ("is_spill" in feats) and (feats["is_spill"])
    if is_fire or is_spill:
        if is_fire:
            preds.append(STATE(area=area, target=obj, key="is_fire", equals=False))
        if is_spill:
            preds.append(STATE(area=area, target=obj, key="is_spill", equals=False))
        return (
            and_(*preds)
            if preds
            else STATE(area=area, target=obj, key="is_handled", equals=True)
        )
    # Branch 2: Event entity (type is the event type, features empty)
    if otype in ("fire", "hazmat", "equipment_failure"):
        return STATE(area=area, target=obj, key="is_handled", equals=True)

    # Branch 3: Fallback (rarely triggered): use safety as criterion
    return STATE(area=area, target=obj, key="is_safe", equals=True)


def build_success_for_guidance(
    goal_details: Dict[str, Any], defaults: Dict[str, Any]
) -> Dict[str, Any]:
    core = goal_details["core_params"]
    dst_area = core["area"]
    obj_spec = _canonicalize_target_from_ai(core.get("object", {}))
    conf = _pct_to_float(
        obj_spec.get("features", {}).get(
            "confidence_threshold_percent",
            core.get("object", {}).get(
                "confidence_threshold_percent", defaults["conf_pct_fallback"]
            ),
        )
    )
    return DETECTED(
        area=dst_area,
        target=obj_spec,
        conf_ge=conf,
        persist_ge_s=defaults["detect_persist_s"],
    )
