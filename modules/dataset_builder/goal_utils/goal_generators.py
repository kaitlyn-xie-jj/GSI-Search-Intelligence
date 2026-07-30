# -*- coding: utf-8 -*-
"""
Core goal generation functions module.
"""

from typing import Dict, Any, Callable, List
import random
import re

from modules.dataset_builder.goal_utils.success_conditions import (
    build_success_for_area_search,
    build_success_for_target_following,
    build_success_for_traffic_enforcement,
    build_success_for_transport,
    build_success_for_evidence_collection,
    build_success_for_verbal_broadcast,
    build_success_for_patrol,
    build_success_for_assembly,
    build_success_for_emergency_response,
    build_success_for_guidance,
)

from modules.dataset_builder.goal_utils.goal_utils import (
    generate_random_coords,
    generate_points_around_center,
    sample_following_ai_recognition,
    human_desc_from_ai,
    build_area_definition,
    area_instruction_fragment_from_json,
    sample_language_level,
    build_meta,
    adjust_search_area_to_cover_target,
    adjust_named_location_to_cover_target,
    adjust_search_area_to_exclude_target,
)

from modules.config.base.enums import (
    CarSubtype, ColorOption, BoatSubtype, PersonItem, CargoSubtype, AssemblyComponentType,
)

from modules.dataset_builder.goal_utils.helpers_refac import (
    maybe_seed, generate_random_id, choose_or_none, trim_dots, append_robots_if_needed,
    infer_goal_determinacy, sample_scene_object
)

from modules.dataset_builder.goal_utils.instruction_templates import (
    AREA_SEARCH_TEMPLATES,
    TARGET_FOLLOWING_TEMPLATES,
    get_traffic_enforcement_templates,
    TRANSPORT_TEMPLATES_NO_SOURCE,
    TRANSPORT_TEMPLATES_WITH_SOURCE,
    EVIDENCE_EVENT_TEMPLATES,
    EVIDENCE_STRUCTURE_TEMPLATES,
    EVIDENCE_OBJECT_AREA_TEMPLATES,
    VERBAL_BROADCAST_MESSAGES,
    VERBAL_BROADCAST_TEMPLATES_WITH_MESSAGE,
    VERBAL_BROADCAST_TEMPLATES_NO_MESSAGE,
    VERBAL_BROADCAST_TEMPLATES_WITH_MESSAGE_NO_AREA,
    VERBAL_BROADCAST_TEMPLATES_NO_MESSAGE_NO_AREA,
    PATROL_TEMPLATES_WITH_TARGET,
    PATROL_TEMPLATES_PLAIN,
    ASSEMBLY_TEMPLATES,
    EMERGENCY_TEMPLATES_WITH_AREA,
    EMERGENCY_TEMPLATES_NO_AREA,
    GUIDANCE_TEMPLATES,
)


# ---------------------------------------------------------------------
# 1. 11 core generation functions
# ---------------------------------------------------------------------

def generate_area_search_goal(
    config: Dict[str, Any],
    followable_only: bool = False,
    assign_id: bool = True,
    enforce_target_in_area: bool = False,
) -> Dict[str, Any]:
    """Generate an area_search goal."""
    level = sample_language_level(config)
    params = config["goal_specific_parameters"]["area_search"]
    confidence = random.randint(*params["ai_recognition"]["CONFIDENCE_RANGE_PERCENT"])
    robot_count = random.randint(*params["robot_assignment"]["COUNT_RANGE"])
    robot_mention = random.random() < params["robot_assignment"]["ASSIGNMENT_PROBABILITY"]
    robot_type = random.choice(config["robot_settings"]["BY_CAPABILITY"]["observation"])

    # Build area definition
    area_params = params["area_definition"]
    mix = area_params["MIX"]
    population, weights = list(mix.keys()), list(mix.values())
    area_type = random.choices(population, weights=weights, k=1)[0]
    if area_type == "named_location" and not config["data_pools"]["LOCATION_LABELS"]:
        area_type = random.choice(["boundary_points", "point_radius"])

    area_def: Dict[str, Any] = {}
    if area_type == "boundary_points":
        num_points = random.randint(*area_params["BOUNDARY_POINTS_RANGE"])
        radius = random.choice(area_params["RADIUS_METERS_CHOICES"])
        center = generate_random_coords(config, 1, margin=radius)[0]
        points = generate_points_around_center(config, center, num_points, radius)
        area_def = {"area_type": "Boundary Selection", "boundary_points": points}
    elif area_type == "named_location":
        location = choose_or_none(config["data_pools"]["LOCATION_LABELS"])
        if not location:
            area_type = "point_radius"
        else:
            area_def = {"area_type": "Named Area", "area_name": location}
    if area_type == "point_radius":
        radius = random.choice(area_params["RADIUS_METERS_CHOICES"])
        center = generate_random_coords(config, 1, margin=radius)[0]
        area_def = {"area_type": "Point Radius", "center_point": center, "radius_m": radius}

    area_desc = area_instruction_fragment_from_json(area_def)

    # Select search target
    search_targets_pool = config["data_pools"]["SEARCH_TARGETS"]
    if followable_only:
        target_info = random.choice([
            t for t in search_targets_pool["object"]
            if t["category"] in ["vehicle", "person", "boat"]
        ])
    else:
        target_type = random.choice(list(search_targets_pool.keys()))
        target_info = random.choice(search_targets_pool[target_type])

    category = target_info["category"]
    pools = config["data_pools"]
    ai_recognition: Dict[str, Any] = {}
    target_desc = ""
    target_node = None

    controls = config.get("generation_controls", {})
    use_scene_features = random.random() < controls.get("TARGET_FROM_SCENE_PROBABILITY", 1.0)

    # Generate description and AI recognition params based on target category
    if category == "person":
        use_attr = (
            random.random() < params["target_details"]["PERSON_ATTRIBUTE_PROBABILITY"]
            and pools["CLOTHING_COLORS"]
            and pools["PERSON_ITEMS"]
        )
        if use_scene_features:
            if use_attr:
                target_node = sample_scene_object(
                    config, "person",
                    lambda n: n["properties"].get("clothing_color") and n["properties"].get("item")
                )
            else:
                target_node = sample_scene_object(
                    config, "person",
                    lambda n: n["properties"].get("suspicious")
                )
            if not target_node:
                return None
            props = target_node["properties"]
        else:
            props = None

        if use_attr:
            if use_scene_features:
                cloth_color, item = props["clothing_color"], props["item"]
            else:
                cloth_color = random.choice(pools["CLOTHING_COLORS"])
                item = random.choice(pools["PERSON_ITEMS"])
            desc_opts = [
                f"a person wearing {cloth_color} clothes and carrying a {item}",
                f"a person in {cloth_color} clothing with a {item}",
                f"a person dressed in {cloth_color} and holding a {item}",
                f"an individual in {cloth_color} clothes carrying a {item}",
                f"a {cloth_color}-clothed person with a {item}",
                f"a person in {cloth_color} attire, carrying a {item}",
                f"a person wearing {cloth_color} and equipped with a {item}",
            ]
            target_desc = random.choice(desc_opts)
            feats = {"clothing_color": cloth_color, "item": item, "description": target_desc}
            ai_recognition = {"type": "person", "features": feats}
        else:
            desc_opts = [
                "suspicious persons",
                "any suspicious individuals",
                "people behaving suspiciously",
                "persons with abnormal behaviour",
                "potentially suspicious persons",
                "individuals showing suspicious behaviour",
                "persons whose behaviour appears abnormal",
            ]
            target_desc = random.choice(desc_opts)
            ai_recognition = {"type": "person", "features": {"suspicious": True}}

    elif category == "vehicle":
        if use_scene_features:
            target_node = sample_scene_object(config, "vehicle")
            if not target_node:
                return None
            props = target_node["properties"]
            color, v_type = props.get("color"), props.get("subtype")
            if not (color and v_type):
                return None
        else:
            color = random.choice(pools["VEHICLE_COLORS"])
            v_type = random.choice(pools["VEHICLE_TYPES"])
        desc_opts = [
            f"a {color} {v_type}",
            f"any {color} {v_type}",
            f"a {color} {v_type} vehicle",
            f"any vehicle of type {v_type} in {color}",
            f"a {color} {v_type} that may be moving or parked",
            f"a single {color} {v_type} of interest",
        ]
        target_desc = random.choice(desc_opts)
        ai_recognition = {"type": "vehicle", "features": {"color": color, "subtype": v_type}}

    elif category == "boat":
        if use_scene_features:
            target_node = sample_scene_object(config, "boat")
            if not target_node:
                return None
            props = target_node["properties"]
            color, b_type = props.get("color"), props.get("subtype")
            if not (color and b_type):
                return None
        else:
            color = random.choice(pools["VEHICLE_COLORS"])
            b_type = random.choice(pools["BOAT_TYPES"])
        desc_opts = [
            f"a {color} {b_type}",
            f"any {color} {b_type}",
            f"a {color} {b_type} vessel",
            f"any boat of type {b_type} in {color}",
            f"a single {color} {b_type} of interest",
        ]
        target_desc = random.choice(desc_opts)
        ai_recognition = {"type": "boat", "features": {"color": color, "subtype": b_type}}

    elif category == "fire":
        if use_scene_features:
            target_node = sample_scene_object(config, "fire")
            if not target_node:
                return None
        desc_opts = [
            "any fire spots or fire signs",
            "visible fire sources or hotspots",
            "open flames or strong heat spots",
            "possible fire outbreaks",
            "localised areas showing signs of fire",
            "fire spots, flames or obvious burning regions",
            "any evidence of active fire",
        ]
        target_desc = random.choice(desc_opts)
        ai_recognition = {"type": "fire", "features": {}}

    elif category == "illegal_parking":
        desc_opts = [
            "illegal parking events and illegally parked vehicles",
            "vehicles parked in violation of parking rules",
            "any illegally parked vehicles",
            "suspected illegal parking cases",
            "parking violations and illegally parked cars",
            "cars stopping or parking where not allowed",
            "illegally parked vehicles or similar parking offences",
        ]
        target_desc = random.choice(desc_opts)
        ai_recognition = {"type": "event", "event_type": "illegal_parking"}

    elif category == "traffic_violation":
        desc_opts = [
            "traffic violations and abnormal vehicle behaviour",
            "vehicles committing traffic violations",
            "any serious traffic rule breaches",
            "abnormal or dangerous vehicle behaviours",
            "suspected traffic violation events",
            "vehicles ignoring traffic rules",
            "overt traffic rule violations",
        ]
        target_desc = random.choice(desc_opts)
        ai_recognition = {"type": "event", "event_type": "traffic_violation"}

    elif category == "crowd":
        desc_opts = [
            "abnormal or unusually dense crowd gatherings",
            "unusually large crowds",
            "abnormally dense crowd clusters",
            "suspicious crowd gatherings",
            "crowds that are denser than usual",
            "large groups forming abnormal gatherings",
            "crowd congestion or abnormal group behaviour",
        ]
        target_desc = random.choice(desc_opts)
        ai_recognition = {"type": "event", "event_type": "crowd"}

    # Consistency flags
    source_consistency = 1 if use_scene_features else 0
    binding_consistency = 0 if enforce_target_in_area else 1

    if enforce_target_in_area and use_scene_features and target_node is not None:
        area_def = adjust_search_area_to_cover_target(config, area_params, area_def, target_node)
        area_desc = area_instruction_fragment_from_json(area_def)
        binding_consistency = 1

    ai_recognition["confidence_threshold_percent"] = confidence
    base_instruction = random.choice(AREA_SEARCH_TEMPLATES[level]).format(
        area=area_desc, target=target_desc
    )
    if random.random() < params["ai_recognition"]["CONFIDENCE_IN_INSTRUCTION_PROBABILITY"]:
        base_instruction = base_instruction.rstrip('.') + f", with a confidence requirement above {confidence}%."
    final_instruction = append_robots_if_needed(base_instruction, robot_mention, robot_count)

    goal_details = {
        "goal_id": generate_random_id() if assign_id else None,
        "goal_type": "area_search",
        "goal_determinacy": infer_goal_determinacy("area_search"),
        "core_params": {
            "search_area": area_def,
            "ai_recognition": ai_recognition,
            "execution_robot": {"type": robot_type, "quantity": robot_count}
        }
    }
    goal_details["success_condition"] = build_success_for_area_search(
        goal_details=goal_details, defaults=config["success_defaults"]
    )
    meta = build_meta(level, source_consistency, binding_consistency, "area_search")
    return {"instruction": final_instruction, "goal_details": goal_details, "meta": meta}


def generate_target_following_goal(config: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a target_following goal."""
    level = sample_language_level(config)
    ai, used_scene_features = sample_following_ai_recognition(config)
    robot_type = random.choice(config["robot_settings"]["BY_CAPABILITY"]["observation"])
    robot_count = random.randint(
        *config["goal_specific_parameters"]["target_following"]["robot_assignment"]["COUNT_RANGE"]
    )

    human_desc = human_desc_from_ai(ai)
    instruction = random.choice(TARGET_FOLLOWING_TEMPLATES[level]).format(t=human_desc)
    assign = random.random() < config["goal_specific_parameters"]["target_following"]["robot_assignment"]["ASSIGNMENT_PROBABILITY"]
    instruction = append_robots_if_needed(instruction, assign, robot_count)

    goal_details = {
        "goal_id": generate_random_id(),
        "goal_type": "target_following",
        "core_params": {
            "ai_recognition": ai,
            "robot_allocation": {"type": robot_type, "quantity": robot_count}
        }
    }
    goal_details["goal_determinacy"] = infer_goal_determinacy("target_following")
    goal_details["success_condition"] = build_success_for_target_following(
        goal_details=goal_details, defaults=config["success_defaults"]
    )
    source_consistency = 1 if used_scene_features else 0
    meta = build_meta(level, source_consistency, 1, "target_following")
    return {"instruction": instruction, "goal_details": goal_details, "meta": meta}


def generate_traffic_enforcement_goal(config: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a traffic_enforcement goal."""
    level = sample_language_level(config)
    params = config["goal_specific_parameters"]["traffic_enforcement"]
    pools = config["data_pools"]
    controls = config.get("generation_controls", {})
    bind_enabled = controls.get("ENABLE_TARGET_AREA_BINDING", False)
    bind_enforce = (
        random.random() < controls.get("TARGET_AREA_BINDING_PROB_SEQ", 1.0)
    ) if bind_enabled else False

    enf_types = pools["ENFORCEMENT_TYPES"]
    enf_key = random.choice(list(enf_types.keys()))
    enf_val = enf_types[enf_key]
    use_megaphone = random.random() < params["actions"]["USE_MEGAPHONE_PROBABILITY"]
    robot_type = random.choice(config["robot_settings"]["BY_CAPABILITY"]["observation"])
    robot_count = random.randint(*params["robot_assignment"]["COUNT_RANGE"])

    source_consistency = 1
    binding_consistency = 0
    location = (
        choose_or_none(pools.get("ENFORCEMENT_LABELS", []))
        or choose_or_none(pools.get("LOCATION_LABELS", []))
    )
    if bind_enforce and location:
        target_node = sample_scene_object(
            config, "vehicle",
            lambda n: n["properties"].get("illegal_parking") or n["properties"].get("traffic_violation")
        )
        if target_node:
            location = adjust_named_location_to_cover_target(config, location, target_node)
            binding_consistency = 1

    templates = get_traffic_enforcement_templates(
        location=location, enf_key=enf_key, enf_val=enf_val, use_megaphone=use_megaphone
    )
    final_instruction = random.choice(templates[level])
    assign = random.random() < params["robot_assignment"]["ASSIGNMENT_PROBABILITY"]
    final_instruction = append_robots_if_needed(final_instruction, assign, robot_count)

    goal_details = {
        "goal_id": generate_random_id(),
        "goal_type": "traffic_enforcement",
        "goal_determinacy": "open",
        "core_params": {
            "evidence_type": {
                "type": enf_val,
                "action": "Take photos" if enf_val == "Strict Road" else "none"
            },
            "robot_allocation": {"type": robot_type, "quantity": robot_count},
            "evidence_action": {"type": "verbal" if use_megaphone else "none", "enabled": use_megaphone},
            "location": location
        }
    }
    goal_details["success_condition"] = build_success_for_traffic_enforcement(
        goal_details=goal_details, defaults=config["success_defaults"]
    )
    meta = build_meta(level, source_consistency, binding_consistency, "traffic_enforcement")
    return {"instruction": final_instruction, "goal_details": goal_details, "meta": meta}



def generate_transport_goal(config: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a transport goal."""
    level = sample_language_level(config)
    params = config["goal_specific_parameters"]["transport"]
    pools = config["data_pools"]
    controls = config.get("generation_controls", {})
    bind_enabled = controls.get("ENABLE_TARGET_AREA_BINDING", False)
    bind_enforce = (
        random.random() < controls.get("TARGET_AREA_BINDING_PROB_SEQ", 1.0)
    ) if bind_enabled else False
    use_scene_features = random.random() < controls.get("TARGET_FROM_SCENE_PROBABILITY", 1.0)

    # Destination
    dst_def = build_area_definition(
        config, params["area_definition"],
        pools["INFRA_LABELS"] + pools["BUILDING_LABELS"]
    )
    dst_desc = area_instruction_fragment_from_json(dst_def)

    # Source location (optional)
    use_src = random.random() < 0.5
    src_def, src_desc = None, None
    if use_src:
        labels = (pools.get("INFRA_LABELS") or []) + (pools.get("BUILDING_LABELS") or [])
        for _ in range(8):
            cand = build_area_definition(config, params["area_definition"], labels)
            cand_desc = area_instruction_fragment_from_json(cand)
            # Ensure source and destination differ in both description and area_type
            if cand_desc and cand_desc != dst_desc and cand.get("area_type") != dst_def.get("area_type"):
                src_def, src_desc = cand, cand_desc
                break

    # Transport object
    target_node = None
    if random.random() < 0.3:
        # Injured person
        person_color_prob = params.get("person_color_probability", 0.35)
        color_pool = pools.get("CLOTHING_COLORS") or []
        use_color = bool(color_pool) and (random.random() < person_color_prob)
        if use_scene_features:
            if use_color:
                target_node = sample_scene_object(
                    config, "person",
                    lambda n: n["properties"].get("injured") and n["properties"].get("clothing_color")
                )
            else:
                target_node = sample_scene_object(
                    config, "person",
                    lambda n: n["properties"].get("injured")
                )
            if not target_node:
                return None
            props = target_node["properties"]
            if use_color:
                col = props["clothing_color"]
                obj_desc = f"an injured person in {col}"
                obj = {"type": "person", "features": {"injured": True, "clothing_color": col}}
            else:
                obj_desc = "an injured person"
                obj = {"type": "person", "features": {"injured": True}}
        else:
            if use_color:
                col = random.choice(color_pool)
                obj_desc = f"an injured person in {col}"
                obj = {"type": "person", "features": {"injured": True, "clothing_color": col}}
            else:
                obj_desc = "an injured person"
                obj = {"type": "person", "features": {"injured": True}}
    else:
        # Cargo
        if use_scene_features:
            target_node = sample_scene_object(config, "cargo")
            if not target_node:
                return None
            props = target_node["properties"]
            sub = props.get("subtype")
            col = props.get("color")
            name = (sub or "cargo").replace("_", " ")
            options = [(f"a {name}", {"subtype": sub} if sub else {})]
            if col:
                options.append((f"a {col} cargo", {"color": col}))
                if sub:
                    options.append((f"a {col} {name}", {"subtype": sub, "color": col}))
            obj_desc, feats = random.choice(options)
            obj = {"type": "cargo", "features": feats}
        else:
            sub = random.choice(pools.get("CARGO_SUBTYPES", ["box", "crate", "battery_pack"]))
            color_pool = (pools.get("CLOTHING_COLORS") or []) or (pools.get("VEHICLE_COLORS") or [])
            options = [(f"a {sub.replace('_', ' ')}", {"subtype": sub})]
            if color_pool:
                col = random.choice(color_pool)
                options.append((f"a {col} cargo", {"color": col}))
                options.append((f"a {col} {sub.replace('_', ' ')}", {"subtype": sub, "color": col}))
            obj_desc, feats = random.choice(options)
            obj = {"type": "cargo", "features": feats}

    source_consistency = 1 if use_scene_features else 0
    binding_consistency = 0
    if bind_enforce and src_def is not None and use_scene_features and target_node is not None:
        src_def = adjust_search_area_to_cover_target(
            config, params["area_definition"], src_def, target_node
        )
        src_desc = area_instruction_fragment_from_json(src_def)
        binding_consistency = 1

    if src_def is None:
        binding_consistency = source_consistency

    if src_desc:
        instr = random.choice(TRANSPORT_TEMPLATES_WITH_SOURCE[level]).format(
            o=obj_desc, s=src_desc, d=dst_desc
        )
    else:
        instr = random.choice(TRANSPORT_TEMPLATES_NO_SOURCE[level]).format(
            o=obj_desc, d=dst_desc
        )

    core = {"object": obj, "destination_area": dst_def}
    if src_def:
        core["source_area"] = src_def

    details = {
        "goal_id": generate_random_id(),
        "goal_type": "transport",
        "goal_determinacy": "open",
        "core_params": core
    }
    details["success_condition"] = build_success_for_transport(details, config["success_defaults"])
    meta = build_meta(level, source_consistency, binding_consistency, "transport")
    return {"instruction": instr, "goal_details": details, "meta": meta}


def generate_evidence_collection_goal(config: Dict[str, Any]) -> Dict[str, Any]:
    """Generate an evidence_collection goal."""
    level = sample_language_level(config)
    params = config["goal_specific_parameters"]["evidence_collection"]
    pools = config["data_pools"]
    robot_type = random.choice(config["robot_settings"]["BY_CAPABILITY"]["observation"])
    robot_count = random.randint(*params["robot_assignment"]["COUNT_RANGE"])
    assign = random.random() < params["robot_assignment"]["ASSIGNMENT_PROBABILITY"]
    controls = config.get("generation_controls", {})
    bind_enabled = controls.get("ENABLE_TARGET_AREA_BINDING", False)
    bind_enforce = (
        random.random() < controls.get("TARGET_AREA_BINDING_PROB_SEQ", 1.0)
    ) if bind_enabled else False
    use_scene_features = random.random() < controls.get("TARGET_FROM_SCENE_PROBABILITY", 1.0)
    source_consistency = 1 if use_scene_features else 0
    binding_consistency = 0

    def pick_area():
        return build_area_definition(
            config, params["area_definition"],
            fallback_locations=pools["LOCATION_LABELS"]
        )

    is_event = random.random() < 0.35
    if is_event:
        ev = random.choice(pools["SEARCH_TARGETS"]["event"])
        area_def = pick_area()
        if bind_enforce:
            ev_cat = ev.get("category")
            if ev_cat in ["illegal_parking", "traffic_violation"]:
                target_node = sample_scene_object(
                    config, "vehicle",
                    lambda n: n["properties"].get(ev_cat) is True
                )
                if target_node:
                    area_def = adjust_search_area_to_cover_target(
                        config, params["area_definition"], area_def, target_node
                    )
                    binding_consistency = 1
            else:
                target_node = sample_scene_object(
                    config, "person",
                    lambda n: n["properties"].get("crowd") is True
                )
                if target_node:
                    area_def = adjust_search_area_to_cover_target(
                        config, params["area_definition"], area_def, target_node
                    )
                    binding_consistency = 1

        area_desc = area_instruction_fragment_from_json(area_def)
        instruction = random.choice(EVIDENCE_EVENT_TEMPLATES[level]).format(
            what=ev["name"], where=area_desc
        )
        instruction = append_robots_if_needed(instruction, assign, robot_count)
        core = {
            "object": {"type": "event", "event_type": ev["category"]},
            "area": area_def,
            "robot_allocation": {"type": robot_type, "quantity": robot_count}
        }
    else:
        is_building = random.random() < 0.5
        if is_building:
            b_labels = pools.get("BUILDING_LABELS", []) or []
            i_labels = pools.get("INFRA_LABELS", []) or []
            candidates = [("building", x) for x in b_labels] + [("trans_facility", x) for x in i_labels]
            struct_type, sname = random.choice(candidates)
            parsed_type = re.sub(r"[-_]*\d+$", "", sname).strip()
            obj = {"type": parsed_type, "features": {"label": sname}}
            instruction = random.choice(EVIDENCE_STRUCTURE_TEMPLATES[level]).format(what=sname)
            instruction = append_robots_if_needed(instruction, assign, robot_count)
            core = {"object": obj, "robot_allocation": {"type": robot_type, "quantity": robot_count}}
            binding_consistency = 1
        else:
            kind = random.choice(["person", "vehicle", "boat", "fire"])
            target_node = None
            if kind == "person":
                use_attr = random.random() < config["goal_specific_parameters"]["area_search"]["target_details"]["PERSON_ATTRIBUTE_PROBABILITY"]
                if use_scene_features:
                    if use_attr:
                        target_node = sample_scene_object(
                            config, "person",
                            lambda n: n["properties"].get("clothing_color") and n["properties"].get("item")
                        )
                        if not target_node:
                            return None
                        p = target_node["properties"]
                        cc, it = p["clothing_color"], p["item"]
                        obj = {"type": "person", "features": {"clothing_color": cc, "item": it}}
                        what = f"a person in {cc} with a {it}"
                    else:
                        target_node = sample_scene_object(
                            config, "person",
                            lambda n: n["properties"].get("suspicious")
                        )
                        if not target_node:
                            return None
                        obj = {"type": "person", "features": {"suspicious": True}}
                        what = "a suspicious person"
                else:
                    if use_attr and pools["CLOTHING_COLORS"] and pools["PERSON_ITEMS"]:
                        cc = random.choice(pools["CLOTHING_COLORS"])
                        it = random.choice(pools["PERSON_ITEMS"])
                        obj = {"type": "person", "features": {"clothing_color": cc, "item": it}}
                        what = f"a person in {cc} with a {it}"
                    else:
                        obj = {"type": "person", "features": {"suspicious": True}}
                        what = "a suspicious person"
            elif kind == "vehicle":
                if use_scene_features:
                    target_node = sample_scene_object(config, "vehicle")
                    if not target_node:
                        return None
                    p = target_node["properties"]
                    color, subtype = p.get("color"), p.get("subtype")
                    if not (color and subtype):
                        return None
                else:
                    color = random.choice(pools["VEHICLE_COLORS"])
                    subtype = random.choice(pools["VEHICLE_TYPES"])
                obj = {"type": "vehicle", "features": {"color": color, "subtype": subtype}}
                what = f"a {color} {subtype}"
            elif kind == "boat":
                if use_scene_features:
                    target_node = sample_scene_object(config, "boat")
                    if not target_node:
                        return None
                    p = target_node["properties"]
                    color, subtype = p.get("color"), p.get("subtype")
                    if not (color and subtype):
                        return None
                else:
                    color = random.choice(pools["VEHICLE_COLORS"])
                    subtype = random.choice(pools["BOAT_TYPES"])
                obj = {"type": "boat", "features": {"color": color, "subtype": subtype}}
                what = f"a {color} {subtype}"
            else:  # fire
                if use_scene_features:
                    target_node = sample_scene_object(config, "fire")
                    if not target_node:
                        return None
                obj = {"type": "fire", "features": {}}
                what = "a fire spot"

            area_def = pick_area()
            if bind_enforce and use_scene_features and target_node is not None:
                area_def = adjust_search_area_to_cover_target(
                    config, params["area_definition"], area_def, target_node
                )
                binding_consistency = 1

            area_desc = area_instruction_fragment_from_json(area_def)
            instruction = random.choice(EVIDENCE_OBJECT_AREA_TEMPLATES[level]).format(
                what=what, where=area_desc
            )
            instruction = append_robots_if_needed(instruction, assign, robot_count)
            core = {
                "object": obj,
                "area": area_def,
                "robot_allocation": {"type": robot_type, "quantity": robot_count}
            }

    goal_details = {
        "goal_id": generate_random_id(),
        "goal_type": "evidence_collection",
        "goal_determinacy": "open",
        "core_params": core
    }
    goal_details["success_condition"] = build_success_for_evidence_collection(
        goal_details=goal_details, defaults=config["success_defaults"]
    )
    meta = build_meta(level, source_consistency, binding_consistency, "evidence_collection")
    return {"instruction": instruction, "goal_details": goal_details, "meta": meta}



def generate_verbal_broadcast_goal(config: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a verbal_broadcast goal."""
    level = sample_language_level(config)
    params = config["goal_specific_parameters"]["verbal_broadcast"]
    pools = config["data_pools"]
    controls = config.get("generation_controls", {})
    bind_enabled = controls.get("ENABLE_TARGET_AREA_BINDING", False)
    bind_enforce = (
        random.random() < controls.get("TARGET_AREA_BINDING_PROB_SEQ", 1.0)
    ) if bind_enabled else False
    area_params = config["goal_specific_parameters"]["area_search"]["area_definition"]
    use_scene_features = random.random() < controls.get("TARGET_FROM_SCENE_PROBABILITY", 1.0)
    source_consistency = 1 if use_scene_features else 0
    binding_consistency = 0

    target_kind = random.choices(
        population=["person", "vehicle", "boat"],
        weights=[0.5, 0.3, 0.2], k=1
    )[0]
    target_node = None

    if target_kind == "person":
        is_crowd = random.random() < 0.35
        suspicious = (not is_crowd) and (random.random() < 0.7)
        features, desc_parts = {}, []
        if use_scene_features:
            if is_crowd:
                target_node = sample_scene_object(
                    config, "person", lambda n: n["properties"].get("crowd")
                )
            elif suspicious:
                target_node = sample_scene_object(
                    config, "person", lambda n: n["properties"].get("suspicious")
                )
            else:
                target_node = sample_scene_object(config, "person")
            if not target_node:
                return None
            p = target_node["properties"]
            if is_crowd:
                features["crowd"] = True
                desc_parts.append("a crowd person")
            else:
                if suspicious:
                    features["suspicious"] = True
                    desc_parts.append("a suspicious person")
                else:
                    desc_parts.append("a person")
            attr_bits = []
            if p.get("clothing_color"):
                features["clothing_color"] = p["clothing_color"]
                attr_bits.append(f"in {p['clothing_color']}")
            if p.get("item"):
                features["item"] = p["item"]
                attr_bits.append(f"with a {p['item']}")
            if attr_bits:
                desc_parts.append(" ".join(attr_bits))
        else:
            if is_crowd:
                features["crowd"] = True
                desc_parts.append("a crowd person")
            else:
                if suspicious:
                    features["suspicious"] = True
                    desc_parts.append("a suspicious person")
                else:
                    desc_parts.append("a person")
            force_attrs = (not suspicious) and (not is_crowd)
            attr_bits = []
            if pools["CLOTHING_COLORS"] and (force_attrs or random.random() < 0.4):
                cc = random.choice(pools["CLOTHING_COLORS"])
                features["clothing_color"] = cc
                attr_bits.append(f"in {cc}")
            if pools["PERSON_ITEMS"] and (force_attrs or random.random() < 0.35):
                it = random.choice(pools["PERSON_ITEMS"])
                features["item"] = it
                attr_bits.append(f"with a {it}")
            if attr_bits:
                desc_parts.append(" ".join(attr_bits))
        target = {"type": "person", "features": features}
        target_text = " ".join(desc_parts)

    elif target_kind == "vehicle":
        if use_scene_features:
            target_node = sample_scene_object(config, "vehicle")
            if not target_node:
                return None
            p = target_node["properties"]
            features = {}
            if "illegal_parking" in p:
                features["illegal_parking"] = bool(p["illegal_parking"])
            if "traffic_violation" in p:
                features["traffic_violation"] = bool(p["traffic_violation"])
            if p.get("color"):
                features["color"] = p["color"]
            if p.get("subtype"):
                features["subtype"] = p["subtype"]
        else:
            features = {}
            if random.random() < 0.5:
                features["illegal_parking"] = True
            if random.random() < 0.3:
                features["traffic_violation"] = True
            violating = features.get("illegal_parking") or features.get("traffic_violation")
            if violating:
                if pools["VEHICLE_COLORS"] and random.random() < 0.4:
                    features["color"] = random.choice(pools["VEHICLE_COLORS"])
                if pools["VEHICLE_TYPES"] and random.random() < 0.4:
                    features["subtype"] = random.choice(pools["VEHICLE_TYPES"])
            else:
                if pools["VEHICLE_COLORS"]:
                    features["color"] = random.choice(pools["VEHICLE_COLORS"])
                if pools["VEHICLE_TYPES"]:
                    features["subtype"] = random.choice(pools["VEHICLE_TYPES"])
        parts = []
        if features.get("illegal_parking"):
            parts.append("an illegally parked")
        elif features.get("traffic_violation"):
            parts.append("a violating")
        else:
            parts.append("a")
        if "color" in features:
            parts.append(features["color"])
        parts.append(features.get("subtype", "vehicle"))
        target_text = " ".join(parts)
        target = {"type": "vehicle", "features": features}

    else:  # boat
        if use_scene_features:
            target_node = sample_scene_object(config, "boat")
            if not target_node:
                return None
            p = target_node["properties"]
            features = {}
            if p.get("color"):
                features["color"] = p["color"]
            if p.get("subtype"):
                features["subtype"] = p["subtype"]
        else:
            features = {}
            if pools["VEHICLE_COLORS"] and random.random() < 0.8:
                features["color"] = random.choice(pools["VEHICLE_COLORS"])
            if pools["BOAT_TYPES"] and random.random() < 0.8:
                features["subtype"] = random.choice(pools["BOAT_TYPES"])
        parts = ["a"]
        if "color" in features:
            parts.append(features["color"])
        parts.append(features.get("subtype", "boat"))
        target_text = " ".join(parts)
        target = {"type": "boat", "features": features}

    message = random.choice(VERBAL_BROADCAST_MESSAGES)

    if random.random() < 0.7:
        area_def = build_area_definition(
            config, area_params, fallback_locations=pools["LOCATION_LABELS"]
        )
        if bind_enforce and use_scene_features and target_node:
            area_def = adjust_search_area_to_cover_target(
                config, area_params, area_def, target_node
            )
            binding_consistency = 1
        area_desc = " " + area_instruction_fragment_from_json(area_def)
    else:
        area_def = {"area_type": "None"}
        area_desc = ""
        binding_consistency = source_consistency

    # Select template
    use_area = area_def.get("area_type") != "None"
    if random.random() < 0.5:
        if use_area:
            tmpl_list = VERBAL_BROADCAST_TEMPLATES_WITH_MESSAGE[level]
            instruction = random.choice(tmpl_list).format(
                who=target_text, msg=message, where=area_desc
            )
        else:
            tmpl_list = VERBAL_BROADCAST_TEMPLATES_WITH_MESSAGE_NO_AREA[level]
            instruction = random.choice(tmpl_list).format(who=target_text, msg=message)
    else:
        if use_area:
            tmpl_list = VERBAL_BROADCAST_TEMPLATES_NO_MESSAGE[level]
            instruction = random.choice(tmpl_list).format(who=target_text, where=area_desc)
        else:
            tmpl_list = VERBAL_BROADCAST_TEMPLATES_NO_MESSAGE_NO_AREA[level]
            instruction = random.choice(tmpl_list).format(who=target_text)

    instruction = trim_dots(instruction)
    robot_type = random.choice(
        config["robot_settings"]["BY_CAPABILITY"].get("verbal_broadcast", ["UAV"])
    )
    robot_count = random.randint(*params["robot_assignment"]["COUNT_RANGE"])
    assign = random.random() < params["robot_assignment"]["ASSIGNMENT_PROBABILITY"]
    instruction = append_robots_if_needed(instruction, assign, robot_count)

    details = {
        "goal_id": generate_random_id(),
        "goal_type": "verbal_broadcast",
        "goal_determinacy": "open",
        "core_params": {
            "target_audience": target,
            "message": message,
            "area": area_def,
            "robot_allocation": {"type": robot_type, "quantity": robot_count}
        }
    }
    details["success_condition"] = build_success_for_verbal_broadcast(
        details, config["success_defaults"]
    )
    meta = build_meta(level, source_consistency, binding_consistency, "verbal_broadcast")
    return {"instruction": instruction, "goal_details": details, "meta": meta}


def generate_patrol_goal(config: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a patrol goal."""
    level = sample_language_level(config)
    params = config["goal_specific_parameters"]["patrol"]
    pools = config["data_pools"]
    controls = config.get("generation_controls", {})
    bind_enabled = controls.get("ENABLE_TARGET_AREA_BINDING", False)
    bind_enforce = (
        random.random() < controls.get("TARGET_AREA_BINDING_PROB_SEQ", 1.0)
    ) if bind_enabled else False
    use_scene_features = random.random() < controls.get("TARGET_FROM_SCENE_PROBABILITY", 1.0)
    source_consistency = 1 if use_scene_features else 0
    binding_consistency = 0

    target_node = None
    use_event = random.random() < 0.2
    if use_event:
        ev = random.choice(pools["SEARCH_TARGETS"]["event"])
        if ev["category"] in ("illegal_parking", "traffic_violation"):
            target_node = sample_scene_object(
                config, "vehicle",
                lambda n: n["properties"].get("illegal_parking") or n["properties"].get("traffic_violation")
            )
        elif ev["category"] == "crowd":
            target_node = sample_scene_object(
                config, "person", lambda n: n["properties"].get("crowd")
            )
        watch_desc = ev["name"]
        ai_recognition = {"type": "event", "event_type": ev["category"]}
    else:
        kind = random.choice(["person", "fire", "boat", "vehicle"])
        if kind == "person":
            if use_scene_features:
                target_node = sample_scene_object(config, "person")
                if not target_node:
                    return None
                p = target_node["properties"]
                feats: Dict[str, Any] = {}
                if p.get("suspicious"):
                    feats["suspicious"] = True
                parts = ["a suspicious person" if feats.get("suspicious") else "a person"]
                frag_bits = []
                if pools["CLOTHING_COLORS"] and (not feats.get("suspicious") or random.random() < 0.4) and p.get("clothing_color"):
                    cc = p["clothing_color"]
                    feats["clothing_color"] = cc
                    frag_bits.append(f"in {cc}")
                if pools["PERSON_ITEMS"] and (not feats.get("suspicious") or random.random() < 0.35) and p.get("item"):
                    it = p["item"]
                    feats["item"] = it
                    frag_bits.append(f"with a {it}")
                if frag_bits:
                    parts.append(" ".join(frag_bits))
                watch_desc = " ".join(parts)
                ai_recognition = {"type": "person", "features": feats}
            else:
                suspicious = random.random() < 0.7
                feats = {"suspicious": suspicious}
                parts = ["a suspicious person" if suspicious else "a person"]
                frag_bits = []
                if pools["CLOTHING_COLORS"] and (not suspicious or random.random() < 0.4):
                    cc = random.choice(pools["CLOTHING_COLORS"])
                    feats["clothing_color"] = cc
                    frag_bits.append(f"in {cc}")
                if pools["PERSON_ITEMS"] and (not suspicious or random.random() < 0.35):
                    it = random.choice(pools["PERSON_ITEMS"])
                    feats["item"] = it
                    frag_bits.append(f"with a {it}")
                if frag_bits:
                    parts.append(" ".join(frag_bits))
                watch_desc = " ".join(parts)
                ai_recognition = {"type": "person", "features": feats}
        elif kind == "fire":
            if use_scene_features:
                target_node = sample_scene_object(config, "fire")
                if not target_node:
                    return None
            watch_desc = "a fire spot"
            ai_recognition = {"type": "fire", "features": {}}
        elif kind == "boat":
            if use_scene_features:
                target_node = sample_scene_object(config, "boat")
                if not target_node:
                    return None
                p = target_node["properties"]
                feats = {}
                if p.get("color"):
                    feats["color"] = p["color"]
                if p.get("subtype"):
                    feats["subtype"] = p["subtype"]
            else:
                feats = {}
                if pools["VEHICLE_COLORS"] and random.random() < 0.7:
                    feats["color"] = random.choice(pools["VEHICLE_COLORS"])
                if pools["BOAT_TYPES"] and random.random() < 0.5:
                    feats["subtype"] = random.choice(pools["BOAT_TYPES"])
            watch_desc = "a " + " ".join(filter(None, [feats.get("color"), feats.get("subtype", "boat")]))
            ai_recognition = {"type": "boat", "features": feats}
        else:  # vehicle
            if use_scene_features:
                target_node = sample_scene_object(config, "vehicle")
                if not target_node:
                    return None
                p = target_node["properties"]
                feats = {}
                if "illegal_parking" in p:
                    feats["illegal_parking"] = bool(p["illegal_parking"])
                if "traffic_violation" in p:
                    feats["traffic_violation"] = bool(p["traffic_violation"])
                if p.get("color"):
                    feats["color"] = p["color"]
                if p.get("subtype"):
                    feats["subtype"] = p["subtype"]
            else:
                feats = {}
                if random.random() < 0.6:
                    feats["illegal_parking"] = True
                elif random.random() < 0.4:
                    feats["traffic_violation"] = True
                violating = feats.get("illegal_parking") or feats.get("traffic_violation")
                if violating:
                    if pools["VEHICLE_COLORS"] and random.random() < 0.4:
                        feats["color"] = random.choice(pools["VEHICLE_COLORS"])
                    if pools["VEHICLE_TYPES"] and random.random() < 0.4:
                        feats["subtype"] = random.choice(pools["VEHICLE_TYPES"])
                else:
                    if pools["VEHICLE_COLORS"]:
                        feats["color"] = random.choice(pools["VEHICLE_COLORS"])
                    if pools["VEHICLE_TYPES"]:
                        feats["subtype"] = random.choice(pools["VEHICLE_TYPES"])
            tokens = []
            if feats.get("illegal_parking"):
                tokens.append("an illegally parked")
            elif feats.get("traffic_violation"):
                tokens.append("a violating")
            else:
                tokens.append("a")
            if "color" in feats:
                tokens.append(feats["color"])
            tokens.append(feats.get("subtype", "vehicle"))
            watch_desc = " ".join(tokens)
            ai_recognition = {"type": "vehicle", "features": feats}

    dwell_s = random.randint(*params["dwell_time_s_range"])
    ai_recognition["patrol_duration_s"] = dwell_s
    ai_recognition["confidence_threshold_percent"] = config["success_defaults"]["conf_pct_fallback"]

    robot_type = random.choice(config["robot_settings"]["BY_CAPABILITY"]["observation"])
    robot_count = random.randint(*params["robot_assignment"]["COUNT_RANGE"])
    assign = random.random() < params["robot_assignment"]["ASSIGNMENT_PROBABILITY"]

    if random.random() < 0.8:
        area_def = build_area_definition(
            config, params["area_definition"],
            fallback_locations=pools["LOCATION_LABELS"]
        )
        if bind_enforce and use_scene_features and target_node:
            area_def = adjust_search_area_to_cover_target(
                config, params["area_definition"], area_def, target_node
            )
            binding_consistency = 1
        area_desc = area_instruction_fragment_from_json(area_def)
        instruction = random.choice(PATROL_TEMPLATES_WITH_TARGET[level]).format(
            area=area_desc, what=watch_desc
        )
    else:
        area_def = build_area_definition(
            config, params["area_definition"],
            fallback_locations=pools["LOCATION_LABELS"]
        )
        area_desc = area_instruction_fragment_from_json(area_def)
        instruction = random.choice(PATROL_TEMPLATES_PLAIN[level]).format(area=area_desc)

    instruction = append_robots_if_needed(instruction, assign, robot_count)

    goal_details = {
        "goal_id": generate_random_id(),
        "goal_type": "patrol",
        "goal_determinacy": "open",
        "core_params": {
            "area": area_def,
            "ai_recognition": ai_recognition,
            "robot_allocation": {"type": robot_type, "quantity": robot_count}
        }
    }
    goal_details["success_condition"] = build_success_for_patrol(
        goal_details=goal_details, defaults=config["success_defaults"]
    )
    meta = build_meta(level, source_consistency, binding_consistency, "patrol")
    return {"instruction": instruction, "goal_details": goal_details, "meta": meta}



def generate_assembly_goal(config: Dict[str, Any]) -> Dict[str, Any]:
    """Generate an assembly goal."""
    level = sample_language_level(config)
    params = config["goal_specific_parameters"].get("assembly", {})
    pools = config["data_pools"]
    area_params = params.get("area_definition")
    labels = (pools.get("INFRA_LABELS") or []) + (pools.get("BUILDING_LABELS") or [])
    area_def = build_area_definition(config, area_params, fallback_locations=labels)
    area_desc = area_instruction_fragment_from_json(area_def)
    source_consistency = 1
    binding_consistency = 1

    STRUCTURES = [
        {"name": "solar base station", "requires": [AssemblyComponentType.SOLAR_PANEL.value]},
        {"name": "antenna station", "requires": [AssemblyComponentType.ANTENNA_MODULE.value]},
        {"name": "pump base station", "requires": [AssemblyComponentType.PUMP_MODULE.value]},
        {"name": "charging station", "requires": [AssemblyComponentType.ROBOT_CHARGING_DOCK.value]},
        {"name": "surveillance pole", "requires": [AssemblyComponentType.SURVEILLANCE_CAMERA_MAST.value]},
        {"name": "temporary shelter", "requires": [AssemblyComponentType.WALL_PANEL.value, AssemblyComponentType.ROOF_PANEL.value]},
        {"name": "emergency call point", "requires": [AssemblyComponentType.EMERGENCY_CALL_BOX.value]},
        {"name": "public display kiosk", "requires": [AssemblyComponentType.PUBLIC_DISPLAY_SCREEN.value]},
        {"name": "public address speaker post", "requires": [AssemblyComponentType.PUBLIC_ADDRESS_SPEAKER.value]},
        {"name": "weather station", "requires": [AssemblyComponentType.WEATHER_STATION_MODULE.value]},
        {"name": "lighting post", "requires": [AssemblyComponentType.LIGHTING_UNIT.value]},
        {"name": "drone landing pad", "requires": [AssemblyComponentType.DRONE_LANDING_PAD.value]},
        {"name": "smart trash point", "requires": [AssemblyComponentType.SMART_TRASH_RECEPTACLE.value]},
    ]
    struct = random.choice(STRUCTURES)
    name = struct["name"]
    reqs = list(struct["requires"])

    def _comp(subtype_val: str) -> Dict[str, Any]:
        return {"type": "assembly_component", "features": {"subtype": subtype_val}}

    object_list: List[Dict[str, Any]] = []
    object_list.append(_comp(AssemblyComponentType.FOUNDATION_BASE.value))
    for rv in reqs:
        object_list.append(_comp(rv))

    instruction = random.choice(ASSEMBLY_TEMPLATES[level]).format(
        obj=name, where=area_desc
    ).strip()
    instruction = trim_dots(instruction)

    details = {
        "goal_id": generate_random_id(),
        "goal_type": "assembly",
        "goal_determinacy": "open",
        "core_params": {"area": area_def, "object": object_list}
    }
    details["success_condition"] = build_success_for_assembly(details, config["success_defaults"])
    meta = build_meta(level, source_consistency, binding_consistency, "assembly")
    return {"instruction": instruction, "goal_details": details, "meta": meta}


def generate_emergency_response_goal(config: Dict[str, Any]) -> Dict[str, Any]:
    """Generate an emergency_response goal."""
    level = sample_language_level(config)
    pools = config["data_pools"]
    params = config["goal_specific_parameters"]["emergency_response"]
    controls = config.get("generation_controls", {})
    bind_enabled = controls.get("ENABLE_TARGET_AREA_BINDING", False)
    bind_enforce = (
        random.random() < controls.get("TARGET_AREA_BINDING_PROB_SEQ", 1.0)
    ) if bind_enabled else False
    use_scene_features = random.random() < controls.get("TARGET_FROM_SCENE_PROBABILITY", 1.0)
    source_consistency = 1 if use_scene_features else 0
    binding_consistency = 0

    hazard_types = ["fire", "hazmat", "equipment_failure"]
    hazard = random.choice(hazard_types)

    use_area = random.random() >= params.get("omit_area_probability", 0.20) and hazard != "equipment_failure"
    area_def, area_desc = None, ""
    if use_area:
        area_def = build_area_definition(
            config, params["area_definition"],
            fallback_locations=pools["LOCATION_LABELS"]
        )
        area_desc = area_instruction_fragment_from_json(area_def)

    bind_object = (hazard in ("fire", "hazmat")) and (random.random() < 0.95)
    target_node = None
    obj, obj_text = None, None

    if bind_object:
        bind_vehicle_or_boat = random.random() < 0.9
        if bind_vehicle_or_boat:
            if use_scene_features:
                kind = random.choice(["vehicle", "boat"])
                if hazard == "fire":
                    pred = lambda n: n["properties"].get("is_fire")
                elif hazard == "hazmat":
                    pred = lambda n: n["properties"].get("is_spill")
                else:
                    pred = None
                target_node = sample_scene_object(config, kind, pred) if pred else sample_scene_object(config, kind)
                if not target_node:
                    return None
                p = target_node["properties"]
                feats = {}
                if p.get("color"):
                    feats["color"] = p["color"]
                if p.get("subtype"):
                    feats["subtype"] = p["subtype"]
                if hazard == "fire":
                    feats["is_fire"] = True
                    feats["is_spill"] = False
                elif hazard == "hazmat":
                    feats["is_fire"] = False
                    feats["is_spill"] = True
                obj = {"type": kind, "features": feats}
                parts = []
                if "color" in feats:
                    parts.append(feats["color"])
                parts.append(feats.get("subtype", kind))
                obj_text = " ".join(parts)
            else:
                kind = random.choice(["vehicle", "boat"])
                feats = {}
                if kind == "vehicle":
                    if pools["VEHICLE_COLORS"] and random.random() < 0.8:
                        feats["color"] = random.choice(pools["VEHICLE_COLORS"])
                    if pools["VEHICLE_TYPES"] and random.random() < 0.8:
                        feats["subtype"] = random.choice(pools["VEHICLE_TYPES"])
                else:
                    if pools["VEHICLE_COLORS"] and random.random() < 0.8:
                        feats["color"] = random.choice(pools["VEHICLE_COLORS"])
                    if pools["BOAT_TYPES"] and random.random() < 0.8:
                        feats["subtype"] = random.choice(pools["BOAT_TYPES"])
                if hazard == "fire":
                    feats["is_fire"] = True
                    feats["is_spill"] = False
                elif hazard == "hazmat":
                    feats["is_fire"] = False
                    feats["is_spill"] = True
                obj = {"type": kind, "features": feats}
                parts = []
                if "color" in feats:
                    parts.append(feats["color"])
                parts.append(feats.get("subtype", kind))
                obj_text = " ".join(parts)
        else:
            b_labels = pools.get("BUILDING_LABELS", []) or []
            i_labels = pools.get("INFRA_LABELS", []) or []
            if b_labels or i_labels:
                candidates = [("building", x) for x in b_labels] + [("trans_facility", x) for x in i_labels]
                _, sname = random.choice(candidates)
                parsed_type = re.sub(r"[-_]*\d+$", "", sname).strip().lower()
                feats = {"label": sname}
                if hazard == "fire":
                    feats["is_fire"] = True
                    feats["is_spill"] = False
                elif hazard == "hazmat":
                    feats["is_fire"] = False
                    feats["is_spill"] = True
                obj = {"type": parsed_type, "features": feats}
                obj_text = sname
            else:
                obj = {"type": hazard, "features": {}}
                obj_text = hazard
    else:
        obj = {"type": hazard, "features": {}}
        obj_text = hazard

    # Build event description
    if hazard == "fire":
        if obj["type"] in ("vehicle", "boat"):
            event_desc = f"a {obj_text} on fire"
        elif obj["features"] and obj["features"].get("label"):
            event_desc = f"fire at {obj['features']['label']}"
        else:
            event_desc = "a fire incident"
    elif hazard == "hazmat":
        if obj["type"] in ("vehicle", "boat"):
            event_desc = f"hazardous spill from {obj_text}"
        elif obj["features"] and obj["features"].get("label"):
            event_desc = f"hazardous leak at {obj['features']['label']}"
        else:
            event_desc = "a hazardous material spill"
    else:
        if obj["features"] and obj["features"].get("label"):
            event_desc = f"equipment hazard at {obj['features']['label']}"
        else:
            event_desc = "equipment-related hazard"

    if use_area and bind_enforce and use_scene_features and target_node:
        area_def = adjust_search_area_to_cover_target(
            config, params["area_definition"], area_def, target_node
        )
        binding_consistency = 1

    if not use_area:
        binding_consistency = source_consistency

    if use_area:
        ts = EMERGENCY_TEMPLATES_WITH_AREA[hazard][level]
        instruction = random.choice(ts).format(event=event_desc, where=area_desc)
    else:
        ts = EMERGENCY_TEMPLATES_NO_AREA[hazard][level]
        instruction = random.choice(ts).format(event=event_desc)

    instruction = trim_dots(instruction)
    core = {"object": obj}
    if use_area:
        core["area"] = area_def

    details = {
        "goal_id": generate_random_id(),
        "goal_type": "emergency_response",
        "goal_determinacy": "open",
        "core_params": core
    }
    details["success_condition"] = build_success_for_emergency_response(
        details, config["success_defaults"]
    )
    meta = build_meta(level, source_consistency, binding_consistency, "emergency_response")
    return {"instruction": instruction, "goal_details": details, "meta": meta}


def generate_guidance_goal(config: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a guidance goal."""
    level = sample_language_level(config)
    pools = config["data_pools"]
    gcfg = config["goal_specific_parameters"].get("guidance", {})
    controls = config.get("generation_controls", {})
    bind_enabled = controls.get("ENABLE_TARGET_AREA_BINDING", False)
    bind_enforce = (
        random.random() < controls.get("TARGET_AREA_BINDING_PROB_SEQ", 1.0)
    ) if bind_enabled else False
    use_scene_features = random.random() < controls.get("TARGET_FROM_SCENE_PROBABILITY", 1.0)
    source_consistency = 1 if use_scene_features else 0
    binding_consistency = 0

    area_params = gcfg.get("area_definition")
    area_def = build_area_definition(
        config, area_params,
        fallback_locations=(pools.get("INFRA_LABELS") or []) + (pools.get("BUILDING_LABELS") or [])
    )
    dst_desc = area_instruction_fragment_from_json(area_def)

    kind = random.choice(["person", "vehicle"])
    target_node = None

    if kind == "person":
        if use_scene_features:
            target_node = sample_scene_object(config, "person")
            if not target_node:
                return None
            p = target_node["properties"]
            feats = {}
            if p.get("suspicious"):
                feats["suspicious"] = True
            if p.get("clothing_color"):
                feats["clothing_color"] = p["clothing_color"]
            if p.get("item"):
                feats["item"] = p["item"]
            obj = {"type": "person", "features": feats}
            parts = ["a suspicious person" if feats.get("suspicious") else "a person"]
            if "clothing_color" in feats:
                parts.append(f"in {feats['clothing_color']}")
            if "item" in feats:
                parts.append(f"with a {feats['item']}")
            obj_text = " ".join(parts)
        else:
            feats = {}
            if random.random() < 0.40:
                feats["suspicious"] = True
            if pools.get("CLOTHING_COLORS") and random.random() < (0.40 if feats.get("suspicious") else 1.0):
                feats["clothing_color"] = random.choice(pools["CLOTHING_COLORS"])
            if pools.get("PERSON_ITEMS") and random.random() < (0.20 if feats.get("suspicious") else 1.0):
                feats["item"] = random.choice(pools["PERSON_ITEMS"])
            obj = {"type": "person", "features": feats}
            parts = ["a suspicious person" if feats.get("suspicious") else "a person"]
            if "clothing_color" in feats:
                parts.append(f"in {feats['clothing_color']}")
            if "item" in feats:
                parts.append(f"with a {feats['item']}")
            obj_text = " ".join(parts)
    else:  # vehicle
        if use_scene_features:
            target_node = sample_scene_object(config, "vehicle")
            if not target_node:
                return None
            p = target_node["properties"]
            feats = {}
            if p.get("color"):
                feats["color"] = p["color"]
            if p.get("subtype"):
                feats["subtype"] = p["subtype"]
            if p.get("illegal_parking"):
                feats["illegal_parking"] = True
            if p.get("traffic_violation"):
                feats["traffic_violation"] = True
            obj = {"type": "vehicle", "features": feats}
        else:
            feats = {}
            if pools.get("VEHICLE_COLORS") and random.random() < 0.80:
                feats["color"] = random.choice(pools["VEHICLE_COLORS"])
            if pools.get("VEHICLE_TYPES") and random.random() < 0.80:
                feats["subtype"] = random.choice(pools["VEHICLE_TYPES"])
            r = random.random()
            if r < 0.20:
                feats["illegal_parking"] = True
            elif r < 0.35:
                feats["traffic_violation"] = True
            obj = {"type": "vehicle", "features": feats}
        tokens = []
        if feats.get("illegal_parking"):
            tokens.append("an illegally parked")
        elif feats.get("traffic_violation"):
            tokens.append("a violating")
        else:
            tokens.append("a")
        if "color" in feats:
            tokens.append(feats["color"])
        tokens.append(feats.get("subtype", "vehicle"))
        obj_text = " ".join(tokens)

    if bind_enforce and use_scene_features and target_node:
        area_def = adjust_search_area_to_exclude_target(
            config, area_params, area_def, target_node
        )
        dst_desc = area_instruction_fragment_from_json(area_def)
        binding_consistency = 1

    instruction = random.choice(GUIDANCE_TEMPLATES[level]).format(obj=obj_text, dst=dst_desc)
    core = {"object": obj, "area": area_def}
    details = {
        "goal_id": generate_random_id(),
        "goal_type": "guidance",
        "goal_determinacy": "open",
        "core_params": core
    }
    details["success_condition"] = build_success_for_guidance(details, config["success_defaults"])
    meta = build_meta(level, source_consistency, binding_consistency, "guidance")
    return {"instruction": instruction, "goal_details": details, "meta": meta}


# ---------------------------------------------------------------------
# 2. Task type to generator function mapping
# ---------------------------------------------------------------------

TASK_GENERATOR_MAPPING: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {
    "area_search": generate_area_search_goal,
    "target_following": generate_target_following_goal,
    "traffic_enforcement": generate_traffic_enforcement_goal,
    "transport": generate_transport_goal,
    "evidence_collection": generate_evidence_collection_goal,
    "verbal_broadcast": generate_verbal_broadcast_goal,
    "patrol": generate_patrol_goal,
    "assembly": generate_assembly_goal,
    "emergency_response": generate_emergency_response_goal,
    "guidance": generate_guidance_goal,
}
