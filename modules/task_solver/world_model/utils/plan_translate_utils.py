import re
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict

from modules.config.base.enums import (
    ColorOption, CarSubtype, BoatSubtype, PersonItem, EventName,
    BuildingName, TransFacilityType, AssemblyComponentType, CargoSubtype
)
from modules.utils.geom_utils import area_centroid
from modules.task_solver.world_model.utils.goal_translate_utils import (
    find_sc_nodes_by_op,
    flatten_context,
    extract_target_from_node_args,
    get_step_arg,
    normalize_param,
    is_coord,
    parse_enum_ci,
    split_token_parts,
)
from modules.platform.platform_factory import get_default_global_boundary

# Target aliases for events and labels.
ALIASES: Dict[str, str] = {
    "illegally_parked_vehicle": "illegal_parking",
    "crowd_gathering": "crowd_gathering",
}

# ============================================================================
# Category keyword definitions. Use a keyword-first matching strategy consistently.
# ============================================================================

# Person keywords.
_PERSON_KEYWORDS = r"\b(person|pedestrian|man|woman|people|human|civilian|victim|injured[_\- ]?person|crowd|group)\b"

# Explicit vehicle keywords.
_VEHICLE_KEYWORDS = r"\b(vehicle|car|automobile|motor[_\- ]?vehicle)\b"

# Explicit boat keywords.
_BOAT_KEYWORDS = r"\b(boat|ship|vessel|watercraft)\b"

# Cargo keywords. These have higher priority than color or subtype inference.
_CARGO_KEYWORDS = r"\b(cargo|toolkit|package|box|crate|supplies|goods|battery|battery[_\- ]?pack|power[_\- ]?bank|parcel|container|luggage|baggage|freight|load|shipment|delivery|medical[_\- ]?supply|food[_\- ]?supply|water[_\- ]?container)\b"

# Fire keywords.
_FIRE_KEYWORDS = r"\b(fire|fire[_\- ]?spot|fire[_\- ]?source|blaze|flame|inferno|conflagration)\b"

# Equipment failure keywords.
_EQUIPMENT_FAILURE_KEYWORDS = r"\b(equipment[_\- ]?failure|equipment[_\- ]?hazard|malfunction|breakdown|defect|fault)\b"

# Hazmat keywords.
_HAZMAT_KEYWORDS = r"\b(hazmat|hazardous[_\- ]?material|dangerous[_\- ]?goods|toxic|chemical[_\- ]?spill)\b"

# Assembly component keywords.
_ASSEMBLY_KEYWORDS = r"\b(assembly[_\- ]?component|foundation[_\- ]?base|wall[_\- ]?panel|roof[_\- ]?panel|solar[_\- ]?panel|lighting[_\- ]?unit|antenna[_\- ]?module|display[_\- ]?screen|address[_\- ]?speaker|weather[_\- ]?module|surveillance[_\- ]?mast|charging[_\- ]?dock|landing[_\- ]?pad|smart[_\- ]?trash|pump[_\- ]?module|emergency[_\- ]?box)\b"

# Building keywords.
_BUILDING_KEYWORDS = r"\b(building|library|hotel|mall|hospital|school|station|hotel|robot[_\- ]?base|parking|power[_\- ]?station|factory|office|apartment|tower|complex)\b"

# Transportation facility keywords.
_TRANS_FACILITY_KEYWORDS = r"\b(intersection|bridge|tunnel|road|street|street[_\- ]?segment|sidewalk|crosswalk|highway|avenue|boulevard|lane|path|water[_\- ]?body)\b"

# Event keywords.
_EVENT_KEYWORDS = r"\b(illegal[_\- ]?parking|traffic[_\- ]?violation|crowd[_\- ]?gathering|accident|incident|emergency|alert)\b"

# Stop words ignored during attribute parsing.
_STOP_WORDS = {
    "with", "and", "holding", "carrying", "wearing",
    "clothes", "clothing", "shirt", "coat", "pants", "hat", "hoodie", "jacket",
    "of", "the", "a", "an", "on", "in", "at", "to", "for", "by", "from",
}

# Keyword sets by category for matching after token splitting.
_CARGO_KEYWORD_SET = {
    "cargo", "toolkit", "package", "box", "crate", "supplies", "goods", 
    "battery", "parcel", "container", "luggage", "baggage", "freight", 
    "load", "shipment", "delivery",
}
_FIRE_KEYWORD_SET = {"fire", "blaze", "flame", "inferno", "conflagration"}
_EQUIPMENT_FAILURE_KEYWORD_SET = {"malfunction", "breakdown", "defect", "fault"}
_HAZMAT_KEYWORD_SET = {"hazmat", "toxic"}
_PERSON_KEYWORD_SET = {"person", "pedestrian", "man", "woman", "people", "human", "civilian", "victim", "crowd", "group"}
_VEHICLE_KEYWORD_SET = {"vehicle", "car", "automobile", "sedan", "suv", "truck", "van", "bus", "motorcycle"}
_BOAT_KEYWORD_SET = {"boat", "ship", "vessel", "watercraft", "speedboat", "yacht", "sailboat", "fishing_boat", "cargo_ship"}
_BUILDING_KEYWORD_SET = {"building", "library", "hotel", "mall", "hospital", "school", "station", "hotel", "parking", "factory", "office", "apartment", "tower", "complex"}
_TRANS_FACILITY_KEYWORD_SET = {"intersection", "bridge", "tunnel", "road", "street", "sidewalk", "crosswalk", "highway", "avenue", "boulevard", "lane", "path"}


def _match_keywords(low: str, parts_low: List[str], pattern: str, keyword_set: set) -> bool:
    """
    Check whether keywords match:
    1. Use regex on the raw token first to handle underscore or hyphen separators.
    2. Then check split parts to handle camel-case tokens.
    """
    # Match the raw token with regex.
    if re.search(pattern, low, re.I):
        return True
    # Check split parts.
    for p in parts_low:
        if p in keyword_set:
            return True
    return False


def normalize_token_by_category_map(
    token: Optional[str],
    category_map: Optional[Dict[str, str]],
) -> Optional[str]:
    if not token or not category_map:
        return token
    low = str(token).casefold()
    candidates = [k for k in category_map.keys() if str(k).casefold() in low]
    if not candidates:
        return token
    hit_key = max(candidates, key=lambda k: len(str(k)))
    return str(hit_key)


def _normalize_boundary_entry(entry: Any) -> Optional[Dict[str, Any]]:
    if isinstance(entry, dict) and "kind" in entry:
        kind = str(entry["kind"]).lower()
        out: Dict[str, Any] = {"kind": kind}
        if kind == "circle":
            if "center" in entry and "radius" in entry:
                cx, cy = entry["center"]
                out["center"] = [float(cx), float(cy)]
                out["radius"] = float(entry["radius"])
                return out
        coords = entry.get("coords") or []
        if isinstance(coords, list):
            out["coords"] = [
                [float(p[0]), float(p[1])]
                for p in coords
                if p and len(p) >= 2
            ]
            if out["coords"]:
                return out
    return None


def _area_lookup_ci(area_boundaries: Dict[str, Any], key: str) -> Optional[Dict[str, Any]]:
    if not key:
        return None
    low = key.casefold()
    for k, v in area_boundaries.items():
        if str(k).casefold() == low:
            return _normalize_boundary_entry(v)
    return None


def _parse_area_from_dict(
    area_dict: Dict[str, Any],
    area_boundaries: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    a_type = (area_dict.get("area_type") or "").lower()
    if a_type == "named area":
        name = area_dict.get("area_name")
        if name:
            return _area_lookup_ci(area_boundaries, str(name))
    if a_type == "point radius":
        cp = area_dict.get("center_point") or {}
        rr = area_dict.get("radius_m")
        if {"y", "x"} <= set(cp.keys()) and rr is not None:
            return {
                "kind": "circle",
                "center": [float(cp["x"]), float(cp["y"])],
                "radius": float(rr),
            }
    if a_type == "boundary selection":
        pts = area_dict.get("boundary_points") or []
        verts = [
            [float(p["x"]), float(p["y"])]
            for p in pts
            if "y" in p and "x" in p
        ]
        if verts:
            return {"kind": "area", "coords": verts}
    return None


def _parse_area_from_any(
    area_any: Any,
    area_boundaries: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    if area_any is None:
        return None
    if isinstance(area_any, dict) and str(area_any.get("area_type", "")).lower() == "none":
        return None
    if isinstance(area_any, dict):
        return _parse_area_from_dict(area_any, area_boundaries)
    if isinstance(area_any, str):
        return _area_lookup_ci(area_boundaries, area_any)
    return None


def _infer_area_type_from_token(token: Optional[str]) -> Optional[str]:
    if not token:
        return None
    low = str(token).lower()
    if "pointradius" in low:
        return "circle"
    if "boundaryselection" in low:
        return "area"
    return None


def _extract_radius_hint_from_token(token: Optional[str]) -> Optional[float]:
    """Extract radius information from an area token, in meters."""
    if not token:
        return None
    low = str(token).lower()
    m = re.search(r"(\d+(?:\.\d+)?)\s*m\b", low)
    if not m:
        m = re.search(r"(\d+(?:\.\d+)?)[_-]m\b", low)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def _pick_area_by_kind(
    candidates: List[Optional[Dict[str, Any]]],
    desired_kind: Optional[str],
    desired_radius: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    valid = [a for a in candidates if isinstance(a, dict)]
    if not valid:
        return None
    kind_filtered = [a for a in valid if desired_kind and a.get("kind") == desired_kind] or valid
    if desired_radius is not None:
        circles = [a for a in kind_filtered if a.get("kind") == "circle" and "radius" in a]
        if circles:
            return min(circles, key=lambda a: abs(float(a.get("radius", 0.0)) - float(desired_radius)))
    return kind_filtered[0]


def _sc_area_candidates(sc_args: Any, area_boundaries: Dict[str, Any]) -> List[Optional[Dict[str, Any]]]:
    """Collect area candidates from a success_condition's args."""
    cand = None
    if isinstance(sc_args, list):
        for item in sc_args:
            if isinstance(item, dict):
                a = _parse_area_from_any((item.get("args") or {}).get("area"), area_boundaries)
                if a:
                    cand = a
                    break
    elif isinstance(sc_args, dict):
        cand = _parse_area_from_any(sc_args.get("area"), area_boundaries)
    return [cand] if cand else []


def _event_area_candidates(goal_cfg: Dict[str, Any], area_boundaries: Dict[str, Any]) -> List[Optional[Dict[str, Any]]]:
    """Collect area candidates from EVENT nodes in the goal's success_condition."""
    out: List[Optional[Dict[str, Any]]] = []
    for nd in find_sc_nodes_by_op(goal_cfg or {}, "EVENT") or []:
        a = (nd.get("args") or {}).get("area")
        parsed = _parse_area_from_any(a, area_boundaries)
        if parsed:
            out.append(parsed)
    return out


def resolve_area_feature(
    area_token: Optional[str],
    goal_cfg: Dict[str, Any],
    area_boundaries: Dict[str, Any],
    category_map: Optional[Dict[str, str]] = None,
) -> Optional[Dict[str, Any]]:
    token = (area_token or "").strip()

    # 1) Explicit named area: direct lookup.
    if token:
        hit = _area_lookup_ci(area_boundaries, token)
        if hit:
            return hit

    # 2) cybertown special case: prefer area_boundaries, otherwise use the default.
    if token and re.search(r"cybertown", token, flags=re.I):
        cybertown_boundary = _area_lookup_ci(area_boundaries, "cybertown")
        if cybertown_boundary:
            return cybertown_boundary
        
        # Fallback: use the default global area boundary, selected by platform type.
        return get_default_global_boundary()

    # 3) Decide whether context fallback is allowed based on the token.
    desired_kind = _infer_area_type_from_token(token)
    desired_radius = _extract_radius_hint_from_token(token)
    allow_fallback_by_context = (not token) or (desired_kind is not None)
    if not allow_fallback_by_context:
        return None

    goal_type = (goal_cfg or {}).get("goal_type", "") if goal_cfg else ""
    sc = (goal_cfg or {}).get("success_condition") or {}
    sc_args = sc.get("args") or {}

    ctx_map: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for ctx in flatten_context(goal_cfg or {}):
        for key in ("source_area", "destination_area", "search_area", "area"):
            if key in ctx:
                parsed = _parse_area_from_any(ctx.get(key), area_boundaries)
                if parsed is not None:
                    ctx_map[key].append(parsed)
        if "location" in ctx and isinstance(ctx["location"], str):
            parsed = _parse_area_from_any(ctx["location"], area_boundaries)
            if parsed is not None:
                ctx_map["location"].append(parsed)

    sc_candidates = _sc_area_candidates(sc_args, area_boundaries)
    event_candidates = _event_area_candidates(goal_cfg or {}, area_boundaries)

    gtype = str(goal_type).lower()
    candidates: List[Optional[Dict[str, Any]]] = []
    if gtype == "transport":
        candidates += ctx_map.get("source_area", []) + sc_candidates + ctx_map.get("destination_area", []) + ctx_map.get("search_area", []) + ctx_map.get("area", [])
    elif gtype in {"patrol", "area_search"}:
        candidates += sc_candidates + ctx_map.get("search_area", []) + ctx_map.get("area", [])
    elif gtype in {"verbal_broadcast", "evidence_collection"}:
        candidates += sc_candidates + ctx_map.get("area", [])
    elif gtype == "traffic_enforcement":
        candidates += event_candidates + ctx_map.get("location", []) + ctx_map.get("area", [])
    elif gtype in {"guidance", "assembly", "emergency_response"}:
        candidates += sc_candidates + ctx_map.get("area", []) + ctx_map.get("location", [])
    else:
        candidates += sc_candidates + ctx_map.get("area", []) + ctx_map.get("search_area", [])

    return _pick_area_by_kind(candidates, desired_kind, desired_radius)


# ============================================================================
# Attribute Parsing Helpers
# ============================================================================

def _parse_color(parts: List[str]) -> Optional[str]:
    """Parse color from token parts."""
    for p in parts:
        if p.lower() in _STOP_WORDS:
            continue
        c = parse_enum_ci(ColorOption, p)
        if c is not None:
            return c
    return None


def _parse_hazard_flags(low: str) -> Dict[str, bool]:
    """Parse hazard flags such as fire or spill."""
    flags = {}
    if re.search(r"\b(fire|burning)\b", low):
        flags["is_fire"] = True
    if re.search(r"\b(spill|leak|leaking)\b", low):
        flags["is_spill"] = True
    return flags


def _parse_person_features(token: str, parts: List[str], parts_low: List[str]) -> Dict[str, Any]:
    """Parse person features."""
    low = token.lower()
    features: Dict[str, Any] = {}
    
    # Parse carried items in with_xxx format.
    m_item = re.search(r"(?:^|[_\s\-])with[_\s\-]([A-Za-z]+)$", low)
    if m_item:
        it = parse_enum_ci(PersonItem, m_item.group(1))
        if it is not None:
            features["item"] = it
    
    # Parse all parts.
    for p in parts:
        # Clothing color.
        c = parse_enum_ci(ColorOption, p)
        if c is not None and "clothing_color" not in features:
            features["clothing_color"] = c
        # Carried item.
        if "item" not in features:
            it = parse_enum_ci(PersonItem, p)
            if it is not None:
                features["item"] = it
        # Special state.
        pl = p.lower()
        if pl in {"injured", "wounded", "hurt"}:
            features["injured"] = True
        if pl in {"crowd", "group"}:
            features["crowd"] = True
        if pl == "suspicious":
            features["suspicious"] = True
    
    return features


def _parse_vehicle_features(token: str, parts: List[str]) -> Dict[str, Any]:
    """Parse vehicle features."""
    low = token.lower()
    features: Dict[str, Any] = {}
    
    # Color.
    color = _parse_color(parts)
    if color:
        features["color"] = color
    
    # Subtype.
    for p in parts:
        if p.lower() in _STOP_WORDS:
            continue
        sub = parse_enum_ci(CarSubtype, p)
        if sub is not None:
            features["subtype"] = sub
            break
    
    # Hazard flags.
    features.update(_parse_hazard_flags(low))
    return features


def _parse_boat_features(token: str, parts: List[str]) -> Dict[str, Any]:
    """Parse boat features."""
    low = token.lower()
    features: Dict[str, Any] = {}
    
    # Subtype.
    for p in parts:
        if p.lower() in _STOP_WORDS:
            continue
        sub = parse_enum_ci(BoatSubtype, p)
        if sub is not None:
            features["subtype"] = sub
            break
    
    # Suspicious flag.
    if "suspicious" in low:
        features["suspicious"] = True
    
    # Hazard flags.
    features.update(_parse_hazard_flags(low))
    return features


def _parse_cargo_features(token: str, parts: List[str]) -> Dict[str, Any]:
    """Parse cargo features."""
    low = token.lower()
    parts_low = [p.lower() for p in parts]
    features: Dict[str, Any] = {}
    
    # Subtype mapping table with unified lowercase matching.
    subtype_map = {
        "toolkit": "toolkit",
        "package": "box",
        "parcel": "box",
        "box": "box",
        "crate": "crate",
        "supplies": "box",
        "goods": "box",
        "battery": "battery_pack",
        "container": "water_container",
        "luggage": "box",
        "baggage": "box",
        "freight": "crate",
        "load": "box",
        "shipment": "box",
        "delivery": "box",
    }
    
    # Check split parts first to handle camel-case tokens.
    for p in parts_low:
        if p in subtype_map:
            features["subtype"] = subtype_map[p]
            break
    
    # Then check compound keywords such as medical_supply and battery_pack.
    if "subtype" not in features:
        compound_patterns = [
            (r"medical[_\- ]?supply", "medical_supply"),
            (r"food[_\- ]?supply", "food_supply"),
            (r"water[_\- ]?container", "water_container"),
            (r"battery[_\- ]?pack", "battery_pack"),
            (r"power[_\- ]?bank", "battery_pack"),
        ]
        for pattern, subtype in compound_patterns:
            if re.search(pattern, low):
                features["subtype"] = subtype
                break
            break
    
    # Color.
    color = _parse_color(parts)
    if color:
        features["color"] = color
    
    return features


def _parse_assembly_features(token: str, parts: List[str]) -> Dict[str, Any]:
    """Parse assembly component features."""
    features: Dict[str, Any] = {}
    parts_low = [p.lower() for p in parts]
    
    # Subtype with multiple matching strategies.
    sub = None
    
    # 1. Try the full token.
    sub = parse_enum_ci(AssemblyComponentType, token)
    
    # 2. Try joining all split parts with underscores.
    if sub is None:
        joined = "_".join(parts_low)
        sub = parse_enum_ci(AssemblyComponentType, joined)
    
    # 3. Try adjacent two-word combinations, such as BlueSolarPanel -> solar_panel.
    if sub is None and len(parts_low) >= 2:
        for i in range(len(parts_low) - 1):
            combo = f"{parts_low[i]}_{parts_low[i+1]}"
            sub = parse_enum_ci(AssemblyComponentType, combo)
            if sub is not None:
                break
    
    # 4. Try matching single parts.
    if sub is None:
        for p in parts_low:
            sub = parse_enum_ci(AssemblyComponentType, p)
            if sub is not None:
                break
    
    if sub is not None:
        features["subtype"] = sub
    
    # Color.
    color = _parse_color(parts)
    if color:
        features["color"] = color
    
    return features


def _parse_building_features(token: str, parts: List[str], parts_low: List[str]) -> Tuple[Optional[str], Dict[str, Any]]:
    """Parse building features and return (type, features)."""
    low = token.lower()
    features: Dict[str, Any] = {}
    btype: Optional[str] = None
    
    # Try matching label formats such as Library-1 or Hospital_2.
    m_label = re.search(r"\b([A-Za-z]+[-_]\d+)\b", token)
    if m_label:
        label = m_label.group(1)
        type_part = re.sub(r"[-_]\d+$", "", label)
        btype = parse_enum_ci(BuildingName, type_part) or type_part
        features["label"] = label
    else:
        # Extract type from keywords.
        building_words = {"building", "library", "hotel", "mall", "hospital", "school", 
                         "station", "hotel", "robot_base", "parking", "power_station"}
        for p in parts_low:
            if p in building_words:
                btype = parse_enum_ci(BuildingName, p) or p
                break
    
    # Hazard flags.
    features.update(_parse_hazard_flags(low))
    return btype, features


def _parse_trans_facility_features(token: str, parts: List[str], parts_low: List[str]) -> Tuple[Optional[str], Dict[str, Any]]:
    """Parse transportation facility features and return (type, features)."""
    low = token.lower()
    features: Dict[str, Any] = {}
    itype: Optional[str] = None
    
    # Try matching label format.
    m_label = re.search(r"\b([A-Za-z]+[-_]\d+)\b", token)
    if m_label:
        label = m_label.group(1)
        type_part = re.sub(r"[-_]\d+$", "", label)
        itype = parse_enum_ci(TransFacilityType, type_part) or type_part
        features["label"] = label
    else:
        # Extract type from keywords.
        infra_words = {"intersection", "bridge", "tunnel", "road", "street", "street_segment", "sidewalk"}
        for p in parts_low:
            if p in infra_words:
                itype = parse_enum_ci(TransFacilityType, p) or p
                break
    
    # Hazard flags.
    features.update(_parse_hazard_flags(low))
    return itype, features


# ============================================================================
# Core Parsing Function
# ============================================================================

def parse_target_token(
    token: Optional[str],
    scene_graph,
    label_to_id_map: Dict[str, Any],
    aliases: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Parse a target token with a keyword-first matching strategy for accurate classification.
    
    Priority order:
    1. Direct label lookup, highest priority.
    2. Event.
    3. Cargo.
    4. Fire.
    5. Equipment failure.
    6. Hazmat.
    7. Person.
    8. Vehicle.
    9. Boat.
    10. Assembly component.
    11. Building.
    12. Transportation facility.
    13. Generic fallback.
    """
    if aliases is None:
        aliases = ALIASES

    token = (token or "").strip()
    if not token:
        return {}

    # 1) Direct label lookup, highest priority.
    if scene_graph and label_to_id_map:
        normalized_token = token.casefold()
        for label, obj_id in label_to_id_map.items():
            # Exact label match.
            if str(label).casefold() == normalized_token:
                try:
                    node = scene_graph.get_node_by_id(obj_id)
                    if node and isinstance(node, dict):
                        return {
                            "class": "object",
                            "type": node.get("properties", {}).get("type"),
                            "features": {"label": label},
                            "_id_from_label": obj_id,
                        }
                except Exception:
                    continue

            # Assembly component: subtype match.
            try:
                node = scene_graph.get_node_by_id(obj_id)
            except Exception:
                node = None
            if node:
                properties = node.get("properties", {})
                subtype = properties.get("subtype")
                if subtype and str(subtype).casefold() == normalized_token and properties.get("type") == "assembly_component":
                    return {
                        "class": "object",
                        "type": properties.get("type"),
                        "features": {"subtype": properties.get("subtype")},
                        "_id_from_label": obj_id,
                    }

    raw = token
    low = token.lower()

    # Alias handling.
    if low in aliases:
        token = aliases[low]
        low = token.lower()

    # Preprocess by splitting tokens.
    parts = split_token_parts(token)
    parts_low = [p.lower() for p in parts]

    # 2) Event, keyword first.
    if re.search(_EVENT_KEYWORDS, low, re.I):
        base = re.sub(r"_event$", "", token, flags=re.I)
        ev = parse_enum_ci(EventName, base)
        if ev is not None:
            return {"class": "event", "event_type": ev}
    # Also check standard event enum.
    base = re.sub(r"_event$", "", token, flags=re.I)
    ev = parse_enum_ci(EventName, base)
    if ev is not None:
        return {"class": "event", "event_type": ev}

    # 3) Cargo, keyword first.
    if _match_keywords(low, parts_low, _CARGO_KEYWORDS, _CARGO_KEYWORD_SET):
        features = _parse_cargo_features(token, parts)
        return {"class": "object", "type": "cargo", "features": features}

    # 4) Fire, keyword first.
    if _match_keywords(low, parts_low, _FIRE_KEYWORDS, _FIRE_KEYWORD_SET):
        return {"class": "object", "type": "fire", "features": {}}

    # 5) Equipment failure, keyword first.
    if _match_keywords(low, parts_low, _EQUIPMENT_FAILURE_KEYWORDS, _EQUIPMENT_FAILURE_KEYWORD_SET):
        return {"class": "object", "type": "equipment_failure", "features": {}}

    # 6) Hazmat, keyword first.
    if _match_keywords(low, parts_low, _HAZMAT_KEYWORDS, _HAZMAT_KEYWORD_SET):
        return {"class": "object", "type": "hazmat", "features": {}}

    # 7) Person, keyword first.
    if _match_keywords(low, parts_low, _PERSON_KEYWORDS, _PERSON_KEYWORD_SET):
        features = _parse_person_features(token, parts, parts_low)
        return {"class": "object", "type": "person", "features": features}

    # 8) Vehicle, explicit keyword first.
    if _match_keywords(low, parts_low, _VEHICLE_KEYWORDS, _VEHICLE_KEYWORD_SET):
        features = _parse_vehicle_features(token, parts)
        return {"class": "object", "type": "vehicle", "features": features}

    # 9) Boat, explicit keyword first.
    if _match_keywords(low, parts_low, _BOAT_KEYWORDS, _BOAT_KEYWORD_SET):
        features = _parse_boat_features(token, parts)
        return {"class": "object", "type": "boat", "features": features}

    # 10) Assembly component, keyword first or enum match.
    if re.search(_ASSEMBLY_KEYWORDS, low, re.I) or parse_enum_ci(AssemblyComponentType, token):
        features = _parse_assembly_features(token, parts)
        return {"class": "object", "type": "assembly_component", "features": features}

    # 11) Building, keyword first.
    if _match_keywords(low, parts_low, _BUILDING_KEYWORDS, _BUILDING_KEYWORD_SET):
        btype, features = _parse_building_features(token, parts, parts_low)
        return {"class": "object", "type": btype, "features": features}

    # 12) Transportation facility, keyword first.
    if _match_keywords(low, parts_low, _TRANS_FACILITY_KEYWORDS, _TRANS_FACILITY_KEYWORD_SET):
        itype, features = _parse_trans_facility_features(token, parts, parts_low)
        return {"class": "object", "type": itype, "features": features}

    # ============================================================================
    # Implicit Inference, Used Only When No Keyword Matches
    # ============================================================================

    # Vehicle implicit inference through color and subtype.
    veh_color = None
    veh_sub = None
    for p in parts:
        if p.lower() in _STOP_WORDS:
            continue
        if veh_color is None:
            veh_color = parse_enum_ci(ColorOption, p)
        if veh_sub is None:
            veh_sub = parse_enum_ci(CarSubtype, p)
    if veh_sub:  # Infer vehicle only when there is an explicit vehicle subtype.
        features: Dict[str, Any] = {}
        if veh_color:
            features["color"] = veh_color
        features["subtype"] = veh_sub
        features.update(_parse_hazard_flags(low))
        return {"class": "object", "type": "vehicle", "features": features}

    # Boat implicit inference through subtype.
    boat_sub = None
    for p in parts:
        if p.lower() in _STOP_WORDS:
            continue
        boat_sub = boat_sub or parse_enum_ci(BoatSubtype, p)
    if boat_sub:
        features = {"subtype": boat_sub}
        if "suspicious" in low:
            features["suspicious"] = True
        features.update(_parse_hazard_flags(low))
        return {"class": "object", "type": "boat", "features": features}

    # Building implicit inference through label format, such as Library-1.
    m_bld = re.search(
        r"\b((?:Library|Building|Mall|Hospital|Hotel|RobotBase|Parking|PowerStation|Hotel)[-_]\d+)\b",
        token, flags=re.I
    )
    if m_bld:
        label = m_bld.group(1)
        type_part = re.sub(r"[-_]\d+$", "", label)
        btype = parse_enum_ci(BuildingName, type_part) or type_part
        features = {"label": label}
        features.update(_parse_hazard_flags(low))
        return {"class": "object", "type": btype, "features": features}

    # Transportation facility implicit inference through label format.
    m_infra = re.search(
        r"\b((?:Intersection|Bridge|Tunnel|Road|Street|StreetSegment|Sidewalk|WaterBody)[-_]\d+)\b",
        token, flags=re.I
    )
    if m_infra:
        label = m_infra.group(1)
        type_part = re.sub(r"[-_]\d+$", "", label)
        itype = parse_enum_ci(TransFacilityType, type_part) or type_part
        features = {"label": label}
        features.update(_parse_hazard_flags(low))
        return {"class": "object", "type": itype, "features": features}

    # Generic label fallback.
    m_label = re.search(r"\b([A-Za-z]+[-_]\d+)\b", token)
    if m_label:
        label = m_label.group(1)
        type_part = re.sub(r"[-_]\d+$", "", label)
        btype = parse_enum_ci(BuildingName, type_part)
        itype = parse_enum_ci(TransFacilityType, type_part)
        final_type = btype if btype else (itype if itype else type_part)
        return {"class": "object", "type": final_type, "features": {"label": label}}

    # Final fallback.
    return {"token": raw}


# ============================================================================
# Target Calibration and Constraint Extraction
# ============================================================================

def _extract_goal_target_and_args(
    goal_cfg: Dict[str, Any],
    aliases: Optional[Dict[str, str]] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Extract the ground-truth target definition from goal_cfg and the args of its containing node."""
    if aliases is None:
        aliases = ALIASES

    goal_cfg = goal_cfg or {}
    extracted = None
    matched_args = None

    op_priority = ["STATE", "EVENT", "DETECTED", "PHOTO_TAKEN", "SPEAK_DURATION", "PATROL_DURATION"]
    for op in op_priority:
        nodes = find_sc_nodes_by_op(goal_cfg, op)
        if not nodes:
            continue
        for nd in nodes:
            args = nd.get("args") or {}
            cand = extract_target_from_node_args(args, aliases=aliases)
            if cand:
                extracted = cand
                matched_args = args
                break
        if extracted:
            break

    if not extracted:
        for ctx in flatten_context(goal_cfg):
            ta = ctx.get("target_audience")
            if isinstance(ta, dict):
                tmp = extract_target_from_node_args({"target": ta}, aliases=aliases)
                if tmp:
                    extracted = tmp
                    break
            obj = ctx.get("object")
            objects_to_check: List[Dict[str, Any]] = []
            if isinstance(obj, dict):
                objects_to_check.append(obj)
            elif isinstance(obj, list):
                objects_to_check.extend(obj)
            for item in objects_to_check:
                if isinstance(item, dict):
                    tmp = extract_target_from_node_args({"target": item}, aliases=aliases)
                    if tmp:
                        extracted = tmp
                        break
            if extracted:
                break

    if not extracted:
        for ctx in flatten_context(goal_cfg):
            ai = ctx.get("ai_recognition") or {}
            ai_type = (ai.get("type") or "").strip().lower()
            if ai_type:
                if ai_type == "event":
                    ev = ai.get("event_type")
                    if isinstance(ev, str) and ev:
                        extracted = {"class": "event", "event_type": aliases.get(ev.lower(), ev)}
                        break
                else:
                    cand2: Dict[str, Any] = {"class": "object", "type": ai_type}
                    feats = ai.get("features") or {}
                    if feats:
                        cand2["features"] = dict(feats)
                    extracted = cand2
                    break

    return extracted, matched_args


def extract_detection_constraints(
    goal_cfg: Dict[str, Any],
    matched_args: Optional[Dict[str, Any]] = None,
) -> Dict[str, Optional[float]]:
    """Extract detection-related thresholds from goal_cfg."""
    goal_cfg = goal_cfg or {}
    conf_ge: Optional[float] = None
    persist: Optional[float] = None

    if matched_args is None:
        _, matched_args = _extract_goal_target_and_args(goal_cfg)

    if matched_args:
        if matched_args.get("conf_ge") is not None:
            try:
                conf_ge = float(matched_args["conf_ge"])
            except Exception:
                conf_ge = None
        if matched_args.get("persist_ge_s") is not None:
            try:
                persist = float(matched_args["persist_ge_s"])
            except Exception:
                persist = None

    if conf_ge is None:
        for ctx in flatten_context(goal_cfg or {}):
            ai = ctx.get("ai_recognition") or {}
            if ai.get("confidence_threshold_percent") is not None:
                try:
                    v = float(ai["confidence_threshold_percent"])
                    conf_ge = v / 100.0 if v > 1.0 else v
                except Exception:
                    pass
                break

    return {"conf_ge": conf_ge, "persist_ge_s": persist}


def calibrate_target_with_goal(
    target_guess: Dict[str, Any],
    goal_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Normalize the target extraction and merge strategy."""
    out = dict(target_guess or {})
    if "_id_from_label" in out:
        out.pop("_id_from_label", None)

    goal_type = (goal_cfg or {}).get("goal_type", "").lower()
    if goal_type == "assembly":
        return out

    extracted, matched_args = _extract_goal_target_and_args(goal_cfg)

    if extracted:
        if extracted.get("class") == "event":
            out["class"] = "event"
            out["event_type"] = extracted.get("event_type")
            out.pop("type", None)
            out.pop("features", None)
            out.pop("label", None)
        else:
            out["class"] = "object"
            if extracted.get("type"):
                out["type"] = extracted["type"]
            if "label" in extracted and "label" not in out:
                out["label"] = extracted["label"]
            feats = extracted.get("features") or {}
            if feats:
                out.setdefault("features", {})
                for k, v in feats.items():
                    out["features"].setdefault(k, v)

    det = extract_detection_constraints(goal_cfg, matched_args)
    if det.get("conf_ge") is not None:
        out["conf_ge"] = det["conf_ge"]
    if det.get("persist_ge_s") is not None:
        out["persist_ge_s"] = det["persist_ge_s"]

    return out


def resolve_target_for_skill(
    token: Optional[str],
    goal_cfg: Dict[str, Any],
    scene_graph,
    label_to_id_map: Dict[str, Any],
) -> Tuple[Dict[str, Any], Optional[str]]:
    """
    Unified entry point:
    - Prefer parsing object_id from label_to_id_map.
    - Otherwise use calibrate_target_with_goal to complete and align based on goal_cfg.
    Return (target_dict, object_id_if_any).
    """
    target_guess = parse_target_token(token, scene_graph, label_to_id_map)
    object_id = target_guess.get("_id_from_label")
    if object_id is not None:
        target = dict(target_guess)
        target.pop("_id_from_label", None)
        return target, str(object_id)
    target = calibrate_target_with_goal(target_guess, goal_cfg)
    return target, None


def resolve_object_id_from_runtime(
    current_params: Dict[str, Any],
    runtime_params: Dict[str, Any],
    robot_label: str,
    active_robots: List[str],
    alt_keys: Optional[List[str]] = None,
) -> Optional[str]:
    """Select the runtime target from runtime_params."""
    if current_params.get("object_id") is not None:
        return str(current_params["object_id"])

    if alt_keys:
        for key in alt_keys:
            if current_params.get(key) is not None:
                return str(current_params[key])

    rp = runtime_params or {}
    se = (rp.get("by_skill") or {}).get("search") or {}
    found_ids = se.get("found_ids") or []

    if not found_ids:
        ph = rp.get("pipeline_hints") or {}
        sel = ph.get("selected_target_id")
        if sel:
            return str(sel)
        cands = ph.get("candidates") or []
        if cands:
            found_ids = cands
        else:
            return None

    num_found = len(found_ids)
    num_robots = len(active_robots)

    if num_found == 0:
        return None

    if num_found == 1:
        return str(found_ids[0])

    # Multiple IDs: current strategy is simple; all robots share the first ID.
    if num_found > 1:
        if num_robots <= 1:
            return str(found_ids[0])
        try:
            _ = active_robots.index(robot_label)
            return str(found_ids[0])
        except ValueError:
            return str(found_ids[0])

    return None
