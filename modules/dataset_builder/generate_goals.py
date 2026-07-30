# -*- coding: utf-8 -*-
"""
Main generation script
- Configuration / parameters
- Scene loading
- Dataset generation main flow
- Save to JSON / JSONL
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Callable

# ---------------------------------------------------------------------
# 0. Project root path & Python path
# ---------------------------------------------------------------------

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from modules.dataset_builder.goal_utils.helpers_refac import maybe_seed, load_scene
from modules.config.base.enums import (
    CarSubtype, ColorOption, BoatSubtype, PersonItem, CargoSubtype, AssemblyComponentType
)

# ---------------------------------------------------------------------
# 1. Scene paths & type sets
# ---------------------------------------------------------------------

SCENE_PATH_DEFAULT = os.path.join(
    os.path.dirname(__file__),
    "../../dataset/semantic/scenarios/cybertown/scenario_1/scene_graph.json",
)

AREA_TYPES_FOR_LOCATIONS = {"campus", "industrial_park", "garden", "square", "water_body", "neighborhood"}
BUILDING_TYPES_FOR_LOCATIONS = {"hospital", "power_station", "mall", "parking", "library"}
INFRA_TYPES_FOR_LOCATIONS = {"intersection", "street_segment", "bridge"}
INFRA_TYPES_FOR_ENFORCEMENT = {"intersection", "street_segment", "bridge"}

# ---------------------------------------------------------------------
# 2. Global CONFIG
# ---------------------------------------------------------------------

CONFIG: Dict[str, Any] = {
    "generation_controls": {
        "GENERATOR_VERSION": "v2.1",
        "TOTAL_SAMPLES_TO_GENERATE": 20000,
        "RANDOM_SEED": 42,
        "TARGET_FROM_SCENE_PROBABILITY": 1.0,
        "ENABLE_TARGET_AREA_BINDING": True,
        "TARGET_AREA_BINDING_PROB_SEQ": 1.0,
    },
    "geospatial_settings": {"COORDINATES": {"X_RANGE": (0.0, 1000.0), "Y_RANGE": (0.0, 1000.0), "RESOLUTION": 3}},
    "goal_distribution": {
        "area_search": 0.1, 
        "target_following": 0.1, 
        "traffic_enforcement": 0.1,
        "transport": 0.1, 
        "evidence_collection": 0.1, 
        "verbal_broadcast": 0.1, 
        "patrol": 0.1, 
        "assembly": 0.1,
        "emergency_response": 0.1, 
        "guidance": 0.1
    },
    "language_level_distribution": {"L0": 0.34,"L1": 0.33,"L2": 0.33},
    "data_pools": {
        "LOCATION_LABELS": [],
        "ENFORCEMENT_LABELS": [],
        "BUILDING_LABELS": [],
        "INFRA_LABELS": [],
        "VEHICLE_COLORS": [c.value for c in ColorOption],
        "VEHICLE_TYPES": [s.value for s in CarSubtype],
        "BOAT_TYPES": [s.value for s in BoatSubtype],
        "CLOTHING_COLORS": [c.value for c in ColorOption],
        "PERSON_ITEMS": [s.value for s in PersonItem],
        "CARGO_SUBTYPES": [s.value for s in CargoSubtype],
        "ENFORCEMENT_TYPES": {"Strict Road Enforcement": "Strict Road", "Non-Strict Road Eviction": "Non-Strict Road"},
        "SEARCH_TARGETS": {
            "object": [
                {"category": "vehicle", "name": "Vehicle"},
                {"category": "person", "name": "Person"},
                {"category": "boat", "name": "Boat"},
                {"category": "fire", "name": "Fire Spot"}
            ],
            "event": [
                {"category": "illegal_parking", "name": "Illegal Parking"},
                {"category": "traffic_violation", "name": "Traffic Violation"},
                {"category": "crowd", "name": "Abnormal Crowd Gathering"}
            ]
        },
    },
    "robot_settings": {"BY_CAPABILITY": {"observation": ["UAV"], "manipulation": ["UGV"], "verbal_broadcast": ["UAV"]}},
    "success_defaults": {
        "conf_pct_fallback": 85, "detect_persist_s": 1.0, "event_persist_s": 2.0, "event_conf_ge": 0.85,
        "follow_duration_s": 10.0, "audio_duration_s": 5.0
    },
    "goal_specific_parameters": {
        "area_search": {
            "area_definition": {"MIX": {"boundary_points": 0.4, "named_location": 0.3, "point_radius": 0.3},
                                "BOUNDARY_POINTS_RANGE": (3, 8),
                                "RADIUS_METERS_CHOICES": [100, 150, 200, 250, 300]},
            "ai_recognition": {"CONFIDENCE_RANGE_PERCENT": (75, 95), "CONFIDENCE_IN_INSTRUCTION_PROBABILITY": 0.6},
            "target_details": {"PERSON_ATTRIBUTE_PROBABILITY": 0.5},
            "robot_assignment": {"ASSIGNMENT_PROBABILITY": 0.5, "COUNT_RANGE": (1, 5)}
        },
        "target_following": {
            "robot_assignment": {"COUNT_RANGE": (1, 1), "ASSIGNMENT_PROBABILITY": 0.3}, 
            "FOLLOWING_ACTIONS": ["Issue verbal warning", "Keep locking and following"]
        },
        "traffic_enforcement": {
            "robot_assignment": {"COUNT_RANGE": (1, 3), "ASSIGNMENT_PROBABILITY": 0.3}, 
            "actions": {"USE_MEGAPHONE_PROBABILITY": 0.5}
        },
        "transport": {
            "area_definition": {"MIX": {"named_location": 0.5, "point_radius": 0.3, "boundary_points": 0.2},
                                "BOUNDARY_POINTS_RANGE": (3, 6),
                                "RADIUS_METERS_CHOICES": [10, 20, 30, 40, 50]}
        },
        "evidence_collection": {
            "area_definition": {"MIX": {"named_location": 0.5, "point_radius": 0.3, "boundary_points": 0.2},
                                "BOUNDARY_POINTS_RANGE": (3, 6),
                                "RADIUS_METERS_CHOICES": [20, 40, 60, 80]},
            "robot_assignment": {"ASSIGNMENT_PROBABILITY": 0.5, "COUNT_RANGE": (1, 3)}
        },
        "verbal_broadcast": {
            "robot_assignment": {"ASSIGNMENT_PROBABILITY": 0.5, "COUNT_RANGE": (1, 2)}, 
            "min_audio_duration_s": 5.0
        },
        "patrol": {
            "area_definition": {"MIX": {"boundary_points": 0.5, "named_location": 0.3, "point_radius": 0.2},
                                       "BOUNDARY_POINTS_RANGE": (4, 8),
                                       "RADIUS_METERS_CHOICES": [120, 180, 240, 300]},
            "robot_assignment": {"ASSIGNMENT_PROBABILITY": 0.6, "COUNT_RANGE": (1, 3)},
            "dwell_time_s_range": (6, 8)
        },
        "assembly": {
            "area_definition": {"MIX": {"named_location": 0.7, "point_radius": 0.3},
                                         "BOUNDARY_POINTS_RANGE": (3, 5),
                                         "RADIUS_METERS_CHOICES": [2, 4, 6, 8, 10]}
        },
        "emergency_response": {
            "area_definition": {"MIX": {"named_location": 0.4, "point_radius": 0.35, "boundary_points": 0.25},
                                         "BOUNDARY_POINTS_RANGE": (3, 6),
                                         "RADIUS_METERS_CHOICES": [20, 40, 60, 80]},
            "omit_area_probability": 0.30
        },
        "guidance": {
            "area_definition": {"MIX": {"named_location": 0.4, "point_radius": 0.3, "boundary_points": 0.3},
                                         "BOUNDARY_POINTS_RANGE": (3, 6),
                                         "RADIUS_METERS_CHOICES": [10, 20, 30, 40, 50]}
        }
    },
}

# ---------------------------------------------------------------------
# 3. Scene loading: populate CONFIG["data_pools"]
# ---------------------------------------------------------------------

def populate_locations_from_scene(
    config: Dict[str, Any],
    scene_path: str = SCENE_PATH_DEFAULT,
) -> None:
    """
    Update location / building / infrastructure / scene object info in config["data_pools"] based on the scene.
    """
    loaded = load_scene(
        scene_path,
        AREA_TYPES_FOR_LOCATIONS,
        BUILDING_TYPES_FOR_LOCATIONS,
        INFRA_TYPES_FOR_LOCATIONS,
        INFRA_TYPES_FOR_ENFORCEMENT,
    )

    pools = config["data_pools"]
    pools["LOCATION_LABELS"] = loaded["LOCATION_LABELS"]
    pools["ENFORCEMENT_LABELS"] = loaded["ENFORCEMENT_LABELS"]
    pools["BUILDING_LABELS"] = loaded.get("BUILDING_LABELS", [])
    pools["INFRA_LABELS"] = loaded.get("INFRA_LABELS", [])
    pools["SCENE_OBJECTS"] = loaded.get("SCENE_OBJECTS", {})
    pools["LOCATION_NODES"] = loaded.get("LOCATION_NODES", {})

# ---------------------------------------------------------------------
# 4. Population building logic
# ---------------------------------------------------------------------

def build_population(
    goal_mix: Dict[str, float],
    total_samples: int,
    generator_mapping: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]],
) -> List[Callable[[Dict[str, Any]], Dict[str, Any]]]:
    """
    Build a function list 'population' according to goal_distribution,
    where each element is a function that generates a single goal (signature: fn(config) -> sample).
    """
    population: List[Callable[[Dict[str, Any]], Dict[str, Any]]] = []

    for goal_key, proportion in goal_mix.items():
        func = generator_mapping.get(goal_key)
        if func:
            count = int(total_samples * proportion)
            population.extend([func] * count)

    # Fill remaining slots randomly
    while len(population) < total_samples:
        random_key = __import__("random").choice(list(goal_mix.keys()))
        func = generator_mapping.get(random_key)
        if func:
            population.append(func)

    __import__("random").shuffle(population)
    return population

# ---------------------------------------------------------------------
# 5. Dataset generation main function
# ---------------------------------------------------------------------

def generate_dataset(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Use the 10 core generation functions provided in goal_generators.TASK_GENERATOR_MAPPING
    to generate the dataset according to CONFIG["goal_distribution"].
    """
    from modules.dataset_builder.goal_utils.goal_generators import TASK_GENERATOR_MAPPING

    maybe_seed(config)
    dataset: List[Dict[str, Any]] = []

    controls = config["generation_controls"]
    total_samples = controls["TOTAL_SAMPLES_TO_GENERATE"]
    goal_mix = config["goal_distribution"]
    version = controls["GENERATOR_VERSION"]

    print(f"Start generating {total_samples} advanced goal dataset samples (version: {version})...")

    population = build_population(goal_mix, total_samples, TASK_GENERATOR_MAPPING)

    for i, goal_func in enumerate(population):
        sample = goal_func(config)
        if sample:
            dataset.append(sample)

        if (i + 1) % 1000 == 0:
            print(f"  ...generated {i + 1}/{total_samples} samples")

    return dataset

# ---------------------------------------------------------------------
# 6. Save functions
# ---------------------------------------------------------------------

def save_dataset_to_jsonl(dataset: List[Dict[str, Any]], config: Dict[str, Any]) -> str:
    version = config["generation_controls"]["GENERATOR_VERSION"]
    filename = os.path.join(os.path.dirname(__file__), "../../dataset/semantic/goals/cybertown/goals.jsonl")
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, "w", encoding="utf-8") as f:
        for entry in dataset:
            # Serialize goal_details to string to avoid PyArrow type conflicts
            # Note: we copy to avoid modifying the original dataset object (in case it's used elsewhere)
            record = entry.copy()

            # Unify binding_consistency type
            if "meta" in record and isinstance(record["meta"], dict):
                record["meta"] = json.dumps(record["meta"], ensure_ascii=False)

            if "goal_details" in record and isinstance(record["goal_details"], dict):
                record["goal_details"] = json.dumps(
                    record["goal_details"], ensure_ascii=False
                )

            if not "id" in record:
                # Try to extract goal_id from goal_details as top-level id
                try:
                    details = (
                        json.loads(record["goal_details"])
                        if isinstance(record["goal_details"], str)
                        else record["goal_details"]
                    )
                    goal_id = details.get("goal_id")
                    if goal_id:
                        record["id"] = goal_id
                except:
                    pass

            f.write(json.dumps(record, ensure_ascii=False) + "\n")


    print(f"\nDataset successfully saved to: {filename}")
    print(f"Total generated {len(dataset)} valid samples.")

    # Build index file
    index_filename = filename + ".index"
    index = {}
    print(f"Building index: {index_filename}")
    with open(filename, "r", encoding="utf-8") as f:
        while True:
            offset = f.tell()
            line = f.readline()
            if not line:
                break
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                # Prefer top-level id
                record_id = record.get("id")
                if not record_id and "goal_details" in record:
                    # goal_details was serialized to string above, need to parse it back
                    try:
                        details = (
                            json.loads(record["goal_details"])
                            if isinstance(record["goal_details"], str)
                            else record["goal_details"]
                        )
                        record_id = details.get("goal_id")
                    except:
                        pass

                if record_id:
                    index[str(record_id)] = offset
            except json.JSONDecodeError:
                continue

    with open(index_filename, "w", encoding="utf-8") as f:
        json.dump(index, f)
    print(f"Index saved with {len(index)} entries.")

    return filename


def save_dataset_to_json(dataset: List[Dict[str, Any]], config: Dict[str, Any]) -> str:
    version = config["generation_controls"]["GENERATOR_VERSION"]
    filename = os.path.join(os.path.dirname(__file__), "../../dataset/semantic/goals/cybertown/goals.json")
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)

    print(f"\nDataset successfully saved to: {filename}")
    print(f"Total generated {len(dataset)} valid samples.")
    return filename

# ---------------------------------------------------------------------
# 7. CLI entry point
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate and optionally validate a goal dataset.")
    args = parser.parse_args()

    # Fix random seed
    maybe_seed(CONFIG)

    # Load scene, populate CONFIG["data_pools"]
    populate_locations_from_scene(CONFIG, SCENE_PATH_DEFAULT)

    # Generate raw dataset
    raw_dataset = generate_dataset(CONFIG)
    final_dataset = raw_dataset

    # Save
    if final_dataset:
        save_dataset_to_jsonl(final_dataset, CONFIG)


if __name__ == "__main__":
    main()
