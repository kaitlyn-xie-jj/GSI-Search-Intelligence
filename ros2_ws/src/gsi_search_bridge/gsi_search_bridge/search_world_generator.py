"""Generate a deterministic Gazebo/GSI outdoor search benchmark world."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple
import xml.etree.ElementTree as ET


Color = Tuple[float, float, float, float]
Bounds = Tuple[float, float, float, float]


ZONE_COLORS: Mapping[str, Color] = {
    "parking": (0.24, 0.27, 0.29, 1.0),
    "loading_zone": (0.34, 0.31, 0.27, 1.0),
    "park": (0.20, 0.43, 0.22, 1.0),
    "campus": (0.40, 0.43, 0.45, 1.0),
    "road": (0.13, 0.14, 0.15, 1.0),
    "building_entrance": (0.50, 0.48, 0.38, 1.0),
    "restricted_zone": (0.45, 0.16, 0.15, 1.0),
    "industrial_yard": (0.38, 0.40, 0.41, 1.0),
    "storage_yard": (0.31, 0.34, 0.36, 1.0),
    "sports_field": (0.18, 0.40, 0.20, 1.0),
    "residential": (0.43, 0.45, 0.40, 1.0),
    "commercial": (0.46, 0.43, 0.38, 1.0),
    "sidewalk": (0.48, 0.48, 0.46, 1.0),
}

V2_ARCHETYPES = {"campus", "industrial", "suburban"}


def load_config(path: Path | str) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    validate_config(config)
    return config


def validate_config(config: Mapping[str, Any]) -> None:
    required = {
        "world",
        "visionflow",
        "search",
        "complexity",
        "target",
        "semantic_prior",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"SearchWorld config missing sections: {', '.join(missing)}")

    world = config["world"]
    size = world.get("size_m") or {}
    width = float(size.get("x", 0.0))
    height = float(size.get("y", 0.0))
    if width < 60.0 or height < 50.0:
        raise ValueError("SearchWorld must be at least 60 m by 50 m")
    if not str(world.get("name", "")).strip():
        raise ValueError("world.name must not be empty")
    origin = world.get("geodetic_origin") or {}
    latitude = float(origin.get("latitude_deg", math.nan))
    longitude = float(origin.get("longitude_deg", math.nan))
    if not -90.0 <= latitude <= 90.0 or not -180.0 <= longitude <= 180.0:
        raise ValueError("world.geodetic_origin must contain valid latitude/longitude")

    visionflow = config["visionflow"]
    for name in ("profile_id", "profile_name", "px4_target", "spawn_pose"):
        if not str(visionflow.get(name, "")).strip():
            raise ValueError(f"visionflow.{name} must not be empty")

    sensor = config.get("sensor", {})
    if sensor:
        required_sensor_fields = (
            "gz_topic_root",
            "rgb_image_suffix",
            "camera_info_suffix",
            "depth_image_suffix",
            "point_cloud_suffix",
            "frame_id",
        )
        for name in required_sensor_fields:
            if not str(sensor.get(name, "")).strip():
                raise ValueError(f"sensor.{name} must not be empty")
        if float(sensor.get("maximum_range_m", 0.0)) <= 0.0:
            raise ValueError("sensor.maximum_range_m must be positive")
        extrinsics = sensor.get("extrinsics") or {}
        for name in ("x_m", "y_m", "z_m", "roll_rad", "pitch_rad", "yaw_rad"):
            if name not in extrinsics:
                raise ValueError(f"sensor.extrinsics.{name} is required")

    search = config["search"]
    resolution = float(search.get("grid_resolution_m", 0.0))
    altitude = float(search.get("flight_altitude_m", 0.0))
    if resolution <= 0.0:
        raise ValueError("search.grid_resolution_m must be positive")
    if altitude < 10.0:
        raise ValueError("search.flight_altitude_m must clear the V1 obstacles")
    if int(search.get("max_viewpoints", 0)) <= 0:
        raise ValueError("search.max_viewpoints must be positive")
    execution = config.get("execution", {})
    if int(execution.get("prestream_setpoint_count", 40)) <= 0:
        raise ValueError("execution.prestream_setpoint_count must be positive")

    complexity = config["complexity"]
    for name in ("tree_count", "parked_vehicle_count", "container_count"):
        if int(complexity.get(name, -1)) < 0:
            raise ValueError(f"complexity.{name} must not be negative")
    for name in ("building_count", "barrier_count", "utility_pole_count"):
        if name in complexity and int(complexity[name]) < 0:
            raise ValueError(f"complexity.{name} must not be negative")

    scene = config.get("scene")
    if scene:
        archetype = str(scene.get("archetype", "")).strip()
        if archetype not in V2_ARCHETYPES:
            choices = ", ".join(sorted(V2_ARCHETYPES))
            raise ValueError(f"scene.archetype must be one of: {choices}")
        if width < 90.0 or height < 70.0:
            raise ValueError("SearchWorld V2 scenes must be at least 90 m by 70 m")

    target = config["target"]
    if not str(target.get("query", "")).strip():
        raise ValueError("target.query must not be empty")
    slot_index = int(target.get("slot_index", -1))
    if slot_index < -1:
        raise ValueError("target.slot_index must be -1 (seeded) or non-negative")

    confidence = float(config["semantic_prior"].get("confidence", -1.0))
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("semantic_prior.confidence must be within [0, 1]")


def generate_artifacts(
    config: Mapping[str, Any],
    output_dir: Path | str,
) -> Dict[str, Path]:
    """Generate all public, simulator, ROS, and evaluator artifacts."""
    validate_config(config)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    layout = _build_layout(config)
    artifacts = {
        "world": output / f"{config['world']['name']}.sdf",
        "semantic_map": output / "semantic_map.json",
        "search_prior": output / "search_prior.json",
        "ground_truth": output / "ground_truth.json",
        "search_params": output / "search_params.yaml",
        "gz_bridge": output / "gz_bridge.yaml",
        "visionflow_profile": output / "visionflow_profile.conf",
        "manifest": output / "scenario_manifest.json",
    }

    world_root = _build_sdf(config, layout)
    ET.indent(world_root, space="  ")
    ET.ElementTree(world_root).write(
        artifacts["world"],
        encoding="utf-8",
        xml_declaration=True,
    )

    semantic_map = _semantic_map(config, layout)
    prior = _search_prior(config)
    ground_truth = _ground_truth(config, layout)
    _write_json(artifacts["semantic_map"], semantic_map)
    _write_json(artifacts["search_prior"], prior)
    _write_json(artifacts["ground_truth"], ground_truth)
    artifacts["search_params"].write_text(
        _search_params_yaml(config), encoding="utf-8", newline="\n"
    )
    artifacts["gz_bridge"].write_text(
        _gz_bridge_yaml(config),
        encoding="utf-8",
        newline="\n",
    )
    artifacts["visionflow_profile"].write_text(
        _visionflow_profile(config), encoding="utf-8", newline="\n"
    )

    manifest = _manifest(config, layout, artifacts)
    _write_json(artifacts["manifest"], manifest)
    return artifacts


def _build_layout(config: Mapping[str, Any]) -> Dict[str, Any]:
    if config.get("scene"):
        return _build_v2_layout(config)
    return _build_legacy_layout(config)


def _build_legacy_layout(config: Mapping[str, Any]) -> Dict[str, Any]:
    world = config["world"]
    complexity = config["complexity"]
    width = float(world["size_m"]["x"])
    height = float(world["size_m"]["y"])
    hx, hy = width / 2.0, height / 2.0
    margin = 4.0
    # Keep at least one row of regular grid centers inside the road semantics.
    road_half_width = min(5.5, height * 0.1)
    rng = random.Random(int(world.get("seed", 0)))

    zones = [
        _zone("parking-west", "parking", (-hx + margin, -hy + margin, -4.0, -road_half_width - 2.0)),
        _zone("loading-east", "loading_zone", (4.0, -hy + margin, hx - margin, -road_half_width - 2.0)),
        _zone("park-northwest", "park", (-hx + margin, road_half_width + 2.0, -4.0, hy - margin)),
        _zone("campus-northeast", "campus", (4.0, road_half_width + 2.0, hx - margin, hy - margin)),
        _zone("road-main", "road", (-hx, -road_half_width, hx, road_half_width), category="trans_facility"),
        _zone("entrance-east", "building_entrance", (4.0, road_half_width + 2.0, min(16.0, hx - margin), min(16.0, hy - margin))),
        _zone("restricted-northeast", "restricted_zone", (hx - 15.0, hy - 15.0, hx - margin, hy - margin), passability="restricted"),
    ]

    buildings = [
        {
            "id": "office-main",
            "type": "office_building",
            "center": (min(23.0, hx - 15.0), min(20.0, hy - 10.0)),
            "size": (16.0, 9.0, 7.0),
            "color": (0.52, 0.55, 0.58, 1.0),
        },
        {
            "id": "service-building",
            "type": "service_building",
            "center": (min(29.0, hx - 10.0), max(9.0, road_half_width + 5.0)),
            "size": (11.0, 7.0, 5.5),
            "color": (0.62, 0.58, 0.50, 1.0),
        },
    ]

    park_bounds = _zone_by_id(zones, "park-northwest")["bounds"]
    trees = []
    for index in range(int(complexity.get("tree_count", 0))):
        trees.append({
            "id": f"tree-{index + 1:02d}",
            "center": _sample_point(rng, park_bounds, 2.5),
            "height": rng.uniform(4.0, 6.0),
        })

    parking_bounds = _zone_by_id(zones, "parking-west")["bounds"]
    parked_vehicles = []
    vehicle_colors = (
        (0.22, 0.34, 0.62, 1.0),
        (0.66, 0.68, 0.70, 1.0),
        (0.45, 0.18, 0.16, 1.0),
        (0.15, 0.18, 0.20, 1.0),
    )
    for index in range(int(complexity.get("parked_vehicle_count", 0))):
        fraction = (index + 1) / (int(complexity["parked_vehicle_count"]) + 1)
        x = parking_bounds[0] + fraction * (parking_bounds[2] - parking_bounds[0])
        y = parking_bounds[1] + 4.0 + (index % 2) * 5.0
        parked_vehicles.append({
            "id": f"parked-vehicle-{index + 1:02d}",
            "center": (x, min(y, parking_bounds[3] - 2.0)),
            "yaw": math.pi / 2.0,
            "color": vehicle_colors[index % len(vehicle_colors)],
        })

    loading_bounds = _zone_by_id(zones, "loading-east")["bounds"]
    containers = []
    for index in range(int(complexity.get("container_count", 0))):
        columns = max(1, int(math.ceil(int(complexity["container_count"]) / 2.0)))
        row, column = divmod(index, columns)
        x = loading_bounds[0] + 7.0 + column * 7.0
        y = loading_bounds[1] + 5.0 + row * 7.0
        containers.append({
            "id": f"container-{index + 1:02d}",
            "center": (min(x, loading_bounds[2] - 4.0), min(y, loading_bounds[3] - 3.0)),
            "color": (0.16, 0.35 + 0.08 * (index % 2), 0.42, 1.0),
        })

    target_slots = [
        {"id": "parking-west-1", "region_id": "parking-west", "pose": _pose(-hx * 0.57, -hy * 0.55, 0.5, 0.0)},
        {"id": "loading-east-1", "region_id": "loading-east", "pose": _pose(hx * 0.52, -hy * 0.55, 0.5, math.pi)},
        {"id": "entrance-east-1", "region_id": "entrance-east", "pose": _pose(8.0, road_half_width + 5.0, 0.5, math.pi / 2.0)},
        {"id": "road-main-1", "region_id": "road-main", "pose": _pose(-hx * 0.22, 0.0, 0.5, 0.0)},
    ]
    configured_slot = int(config["target"].get("slot_index", -1))
    slot_index = rng.randrange(len(target_slots)) if configured_slot == -1 else configured_slot
    if slot_index >= len(target_slots):
        raise ValueError(f"target.slot_index must be smaller than {len(target_slots)}")

    return {
        "bounds": (-hx, -hy, hx, hy),
        "zones": zones,
        "buildings": buildings,
        "trees": trees,
        "parked_vehicles": parked_vehicles,
        "containers": containers,
        "target_slots": target_slots,
        "selected_target_slot": target_slots[slot_index],
        "barriers": [],
        "utility_poles": [],
        "layout_archetype": "legacy_quadrants",
    }


def _build_v2_layout(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Build one deterministic member of the V2 realistic scene family."""
    world = config["world"]
    complexity = config["complexity"]
    width = float(world["size_m"]["x"])
    height = float(world["size_m"]["y"])
    hx, hy = width / 2.0, height / 2.0
    archetype = str(config["scene"]["archetype"])
    blueprint = _v2_blueprint(archetype, hx, hy)
    rng = random.Random(int(world.get("seed", 0)))

    building_count = int(complexity.get("building_count", len(blueprint["buildings"])))
    if building_count > len(blueprint["buildings"]):
        raise ValueError(
            f"complexity.building_count for {archetype} must be at most "
            f"{len(blueprint['buildings'])}"
        )
    buildings = list(blueprint["buildings"][:building_count])

    tree_zone = _zone_by_id(blueprint["zones"], blueprint["tree_zone"])["bounds"]
    trees = [
        {
            "id": f"tree-{index + 1:02d}",
            "center": _sample_point(rng, tree_zone, 2.0),
            "height": rng.uniform(4.5, 7.5),
        }
        for index in range(int(complexity.get("tree_count", 0)))
    ]

    vehicle_zone = _zone_by_id(
        blueprint["zones"], blueprint["vehicle_zone"]
    )["bounds"]
    vehicle_colors = (
        (0.16, 0.24, 0.46, 1.0),
        (0.70, 0.72, 0.73, 1.0),
        (0.42, 0.15, 0.13, 1.0),
        (0.12, 0.14, 0.16, 1.0),
        (0.55, 0.55, 0.50, 1.0),
    )
    parked_vehicles = []
    vehicle_count = int(complexity.get("parked_vehicle_count", 0))
    for index, center in enumerate(_parking_points(vehicle_zone, vehicle_count)):
        parked_vehicles.append({
            "id": f"parked-vehicle-{index + 1:02d}",
            "center": center,
            "yaw": math.pi / 2.0 if archetype != "suburban" else 0.0,
            "color": vehicle_colors[index % len(vehicle_colors)],
        })

    container_zone = _zone_by_id(
        blueprint["zones"], blueprint["container_zone"]
    )["bounds"]
    containers = []
    container_count = int(complexity.get("container_count", 0))
    for index, center in enumerate(_yard_points(container_zone, container_count)):
        containers.append({
            "id": f"container-{index + 1:02d}",
            "center": center,
            "color": (0.17, 0.32 + 0.07 * (index % 3), 0.40, 1.0),
        })

    barriers = list(
        blueprint["barriers"][: int(complexity.get("barrier_count", len(blueprint["barriers"])))]
    )
    pole_count = int(complexity.get("utility_pole_count", 0))
    utility_poles = _utility_poles(blueprint["pole_axis"], pole_count)

    target_slots = list(blueprint["target_slots"])
    configured_slot = int(config["target"].get("slot_index", -1))
    slot_index = rng.randrange(len(target_slots)) if configured_slot == -1 else configured_slot
    if slot_index >= len(target_slots):
        raise ValueError(f"target.slot_index must be smaller than {len(target_slots)}")

    return {
        "bounds": (-hx, -hy, hx, hy),
        "zones": list(blueprint["zones"]),
        "buildings": buildings,
        "trees": trees,
        "parked_vehicles": parked_vehicles,
        "containers": containers,
        "barriers": barriers,
        "utility_poles": utility_poles,
        "target_slots": target_slots,
        "selected_target_slot": target_slots[slot_index],
        "layout_archetype": archetype,
    }


def _v2_blueprint(archetype: str, hx: float, hy: float) -> Dict[str, Any]:
    margin = 4.0
    road = 5.0
    if archetype == "campus":
        zones = [
            _zone("parking-southwest", "parking", (-hx + margin, -hy + margin, -8.0, -road - 3.0)),
            _zone("sports-northwest", "sports_field", (-hx + margin, road + 3.0, -10.0, hy - margin)),
            _zone("academic-core", "campus", (8.0, road + 3.0, hx - margin, hy - margin)),
            _zone("road-east-west", "road", (-hx, -road, hx, road), category="trans_facility"),
            _zone("road-campus-spur", "road", (-5.0, -hy, 5.0, hy), category="trans_facility"),
            _zone("entrance-south", "building_entrance", (-5.0, -hy + margin, 8.0, -road - 3.0)),
            _zone("restricted-utilities", "restricted_zone", (hx - 16.0, hy - 16.0, hx - margin, hy - margin), passability="restricted"),
        ]
        buildings = [
            _building("library", "academic_building", (18.0, 18.0), (18.0, 11.0, 9.0), (0.58, 0.57, 0.53, 1.0)),
            _building("engineering", "academic_building", (35.0, 10.5), (16.0, 9.0, 12.0), (0.46, 0.50, 0.52, 1.0)),
            _building("student-center", "public_building", (14.0, 31.0), (13.0, 8.0, 6.5), (0.64, 0.60, 0.52, 1.0)),
            _building("laboratory", "laboratory", (34.0, 28.0), (15.0, 10.0, 8.0), (0.50, 0.53, 0.55, 1.0)),
        ]
        target_slots = [
            _target_slot("campus-parking-1", "parking-southwest", -28.0, -23.0, 0.0),
            _target_slot("campus-entrance-1", "entrance-south", 2.0, -21.0, math.pi / 2.0),
            _target_slot("campus-road-1", "road-east-west", -18.0, 0.0, 0.0),
            _target_slot("campus-core-1", "academic-core", 10.0, 11.0, math.pi / 2.0),
        ]
        barriers = [
            _barrier("campus-gate-west", (-7.5, -12.0), (0.3, 8.0, 1.2)),
            _barrier("campus-service-wall", (42.0, 23.0), (0.3, 12.0, 2.2)),
            _barrier("campus-bike-rack", (9.0, 8.5), (7.0, 0.25, 1.0)),
        ]
        return _blueprint(zones, buildings, target_slots, barriers, "sports-northwest", "parking-southwest", "restricted-utilities", (-hx + 5.0, -road - 2.0, hx - 5.0, -road - 2.0))

    if archetype == "industrial":
        zones = [
            _zone("employee-parking", "parking", (-hx + margin, -hy + margin, -12.0, -road - 3.0)),
            _zone("loading-apron", "loading_zone", (8.0, -hy + margin, hx - margin, -road - 3.0)),
            _zone("warehouse-yard", "industrial_yard", (-hx + margin, road + 3.0, 2.0, hy - margin)),
            _zone("container-storage", "storage_yard", (8.0, road + 3.0, hx - margin, hy - margin)),
            _zone("haul-road", "road", (-hx, -road, hx, road), category="trans_facility"),
            _zone("security-gate", "building_entrance", (-7.0, -hy + margin, 7.0, -road - 3.0)),
            _zone("restricted-tank-farm", "restricted_zone", (-hx + margin, hy - 18.0, -hx + 20.0, hy - margin), passability="restricted"),
        ]
        buildings = [
            _building("warehouse-a", "warehouse", (-28.0, 17.0), (22.0, 13.0, 9.0), (0.48, 0.50, 0.51, 1.0)),
            _building("warehouse-b", "warehouse", (-5.0, 28.0), (20.0, 12.0, 8.0), (0.54, 0.52, 0.48, 1.0)),
            _building("dispatch-office", "office_building", (20.0, 12.0), (13.0, 8.0, 6.0), (0.58, 0.56, 0.50, 1.0)),
            _building("maintenance-shop", "service_building", (37.0, 28.0), (16.0, 10.0, 7.0), (0.45, 0.48, 0.49, 1.0)),
            _building("security-office", "security_building", (3.0, -17.0), (7.0, 5.0, 3.5), (0.62, 0.60, 0.54, 1.0)),
        ]
        target_slots = [
            _target_slot("industrial-parking-1", "employee-parking", -30.0, -23.0, 0.0),
            _target_slot("industrial-loading-1", "loading-apron", 25.0, -22.0, math.pi),
            _target_slot("industrial-gate-1", "security-gate", 0.0, -15.0, math.pi / 2.0),
            _target_slot("industrial-road-1", "haul-road", -14.0, 0.0, 0.0),
        ]
        barriers = [
            _barrier("loading-divider", (8.0, -18.0), (0.3, 18.0, 1.5)),
            _barrier("yard-wall-west", (-42.0, 17.0), (0.3, 20.0, 2.5)),
            _barrier("tank-farm-wall", (-35.0, 33.0), (18.0, 0.3, 2.5)),
            _barrier("gate-arm", (0.0, -9.0), (9.0, 0.25, 1.0)),
        ]
        return _blueprint(zones, buildings, target_slots, barriers, "warehouse-yard", "employee-parking", "container-storage", (-hx + 5.0, road + 2.0, hx - 5.0, road + 2.0))

    zones = [
        _zone("roadside-parking", "parking", (-hx + margin, -hy + margin, hx - margin, -road - 3.0)),
        _zone("community-park", "park", (-hx + margin, road + 3.0, -12.0, hy - margin)),
        _zone("residential-east", "residential", (8.0, road + 3.0, hx - margin, hy - margin)),
        _zone("local-shops", "commercial", (-8.0, road + 3.0, 8.0, hy - margin)),
        _zone("main-street", "road", (-hx, -road, hx, road), category="trans_facility"),
        _zone("side-street", "road", (18.0, -hy, 28.0, hy), category="trans_facility"),
        _zone("school-entrance", "building_entrance", (-8.0, road + 3.0, 8.0, 18.0)),
        _zone("restricted-substation", "restricted_zone", (hx - 15.0, hy - 15.0, hx - margin, hy - margin), passability="restricted"),
    ]
    buildings = [
        _building("school", "school_building", (-1.0, 28.0), (18.0, 11.0, 7.0), (0.62, 0.58, 0.50, 1.0)),
        _building("shop-row", "commercial_building", (10.0, 13.0), (14.0, 7.0, 4.5), (0.56, 0.52, 0.45, 1.0)),
        _building("house-a", "residential_building", (35.0, 12.0), (10.0, 8.0, 5.0), (0.66, 0.61, 0.55, 1.0)),
        _building("house-b", "residential_building", (36.0, 27.0), (11.0, 8.0, 5.5), (0.58, 0.62, 0.60, 1.0)),
        _building("house-c", "residential_building", (14.0, 29.0), (9.0, 7.0, 4.8), (0.64, 0.56, 0.50, 1.0)),
        _building("clinic", "public_building", (-16.0, 13.0), (12.0, 8.0, 5.0), (0.70, 0.70, 0.66, 1.0)),
    ]
    target_slots = [
        _target_slot("suburban-parking-1", "roadside-parking", -30.0, -20.0, 0.0),
        _target_slot("suburban-shops-1", "local-shops", 0.0, 11.0, math.pi / 2.0),
        _target_slot("suburban-school-1", "school-entrance", -2.0, 13.0, math.pi / 2.0),
        _target_slot("suburban-road-1", "main-street", 35.0, 0.0, math.pi),
    ]
    barriers = [
        _barrier("school-fence", (-10.0, 22.0), (0.3, 20.0, 1.6)),
        _barrier("park-fence", (-14.0, 8.0), (20.0, 0.3, 1.3)),
        _barrier("substation-wall", (42.0, 28.0), (0.3, 14.0, 2.2)),
    ]
    return _blueprint(zones, buildings, target_slots, barriers, "community-park", "roadside-parking", "restricted-substation", (-hx + 5.0, road + 2.0, hx - 5.0, road + 2.0))


def _blueprint(zones, buildings, target_slots, barriers, tree_zone, vehicle_zone, container_zone, pole_axis):
    return {
        "zones": zones,
        "buildings": buildings,
        "target_slots": target_slots,
        "barriers": barriers,
        "tree_zone": tree_zone,
        "vehicle_zone": vehicle_zone,
        "container_zone": container_zone,
        "pole_axis": pole_axis,
    }


def _building(name, kind, center, size, color):
    return {"id": name, "type": kind, "center": center, "size": size, "color": color}


def _barrier(name, center, size):
    return {"id": name, "center": center, "size": size, "color": (0.48, 0.47, 0.43, 1.0)}


def _target_slot(name, region_id, x, y, yaw):
    return {"id": name, "region_id": region_id, "pose": _pose(x, y, 0.5, yaw)}


def _parking_points(bounds: Bounds, count: int) -> list[Tuple[float, float]]:
    if count <= 0:
        return []
    x_min, y_min, x_max, y_max = bounds
    columns = max(1, math.ceil(count / 2.0))
    dx = (x_max - x_min - 6.0) / max(columns, 1)
    return [
        (
            min(x_max - 2.0, x_min + 3.0 + (index // 2 + 0.5) * dx),
            min(y_max - 2.0, y_min + 4.0 + (index % 2) * 5.0),
        )
        for index in range(count)
    ]


def _yard_points(bounds: Bounds, count: int) -> list[Tuple[float, float]]:
    if count <= 0:
        return []
    x_min, y_min, x_max, y_max = bounds
    columns = max(1, math.ceil(count / 2.0))
    return [
        (
            min(x_max - 3.0, x_min + 4.0 + (index % columns) * 6.5),
            min(y_max - 2.0, y_min + 4.0 + (index // columns) * 6.0),
        )
        for index in range(count)
    ]


def _utility_poles(axis: Sequence[float], count: int) -> list[Dict[str, Any]]:
    if count <= 0:
        return []
    x1, y1, x2, y2 = (float(value) for value in axis)
    return [
        {
            "id": f"utility-pole-{index + 1:02d}",
            "center": (
                x1 + (index + 1) * (x2 - x1) / (count + 1),
                y1 + (index + 1) * (y2 - y1) / (count + 1),
            ),
            "height": 6.0,
        }
        for index in range(count)
    ]


def _zone(
    feature_id: str,
    feature_type: str,
    bounds: Bounds,
    *,
    category: str = "area",
    passability: str = "open",
) -> Dict[str, Any]:
    return {
        "id": feature_id,
        "type": feature_type,
        "category": category,
        "bounds": tuple(float(value) for value in bounds),
        "passability": passability,
    }


def _zone_by_id(zones: Iterable[Mapping[str, Any]], feature_id: str) -> Mapping[str, Any]:
    return next(zone for zone in zones if zone["id"] == feature_id)


def _sample_point(rng: random.Random, bounds: Bounds, inset: float) -> Tuple[float, float]:
    return (
        rng.uniform(bounds[0] + inset, bounds[2] - inset),
        rng.uniform(bounds[1] + inset, bounds[3] - inset),
    )


def _pose(x: float, y: float, z: float, yaw: float) -> Tuple[float, float, float, float]:
    return tuple(round(float(value), 6) for value in (x, y, z, yaw))


def _build_sdf(config: Mapping[str, Any], layout: Mapping[str, Any]) -> ET.Element:
    world_config = config["world"]
    root = ET.Element("sdf", {"version": "1.9"})
    world = ET.SubElement(root, "world", {"name": str(world_config["name"])})
    physics = ET.SubElement(world, "physics", {"type": "ode"})
    ET.SubElement(physics, "max_step_size").text = "0.004"
    ET.SubElement(physics, "real_time_factor").text = "1.0"
    ET.SubElement(physics, "real_time_update_rate").text = "250"
    ET.SubElement(world, "gravity").text = "0 0 -9.8"
    ET.SubElement(world, "magnetic_field").text = "6e-06 2.3e-05 -4.2e-05"
    ET.SubElement(world, "atmosphere", {"type": "adiabatic"})
    scene = ET.SubElement(world, "scene")
    ET.SubElement(scene, "grid").text = "false"
    ET.SubElement(scene, "ambient").text = "0.62 0.62 0.60 1"
    ET.SubElement(scene, "background").text = "0.70 0.78 0.84 1"
    ET.SubElement(scene, "shadows").text = "true"
    _add_sun(world)
    _add_spherical_coordinates(world, world_config["geodetic_origin"])

    width = float(world_config["size_m"]["x"])
    height = float(world_config["size_m"]["y"])
    _add_ground(world, width + 30.0, height + 30.0)

    for zone in layout["zones"]:
        _add_zone_visual(world, zone)
    if layout["layout_archetype"] == "legacy_quadrants":
        _add_road_markings(world, layout["bounds"])
    else:
        for zone in layout["zones"]:
            if zone["type"] == "road":
                _add_road_zone_markings(world, zone)

    for building in layout["buildings"]:
        _add_box_model(
            world,
            building["id"],
            (*building["center"], building["size"][2] / 2.0),
            building["size"],
            building["color"],
        )
    for tree in layout["trees"]:
        _add_tree(world, tree)
    for vehicle in layout["parked_vehicles"]:
        _add_vehicle(world, vehicle, yellow=False)
    for container in layout["containers"]:
        _add_box_model(
            world,
            container["id"],
            (*container["center"], 1.3),
            (5.5, 2.4, 2.6),
            container["color"],
        )
    for barrier in layout["barriers"]:
        _add_box_model(
            world,
            barrier["id"],
            (*barrier["center"], barrier["size"][2] / 2.0),
            barrier["size"],
            barrier["color"],
        )
    for pole in layout["utility_poles"]:
        _add_utility_pole(world, pole)
    restricted = [zone for zone in layout["zones"] if zone["type"] == "restricted_zone"]
    for zone in restricted:
        prefix = "restricted" if zone["id"] == "restricted-northeast" else zone["id"]
        _add_restricted_fence(world, zone["bounds"], prefix=prefix)

    target = config["target"]
    slot = layout["selected_target_slot"]
    _add_vehicle(
        world,
        {
            "id": str(target["model_name"]),
            "center": slot["pose"][:2],
            "yaw": slot["pose"][3],
            "color": (1.0, 0.72, 0.02, 1.0),
        },
        yellow=True,
    )
    return root


def _add_sun(world: ET.Element) -> None:
    light = ET.SubElement(world, "light", {"name": "sun", "type": "directional"})
    ET.SubElement(light, "pose").text = "0 0 100 0 0 0"
    ET.SubElement(light, "cast_shadows").text = "true"
    ET.SubElement(light, "intensity").text = "1.0"
    ET.SubElement(light, "direction").text = "-0.3 0.2 -0.9"
    ET.SubElement(light, "diffuse").text = "0.95 0.93 0.88 1"
    ET.SubElement(light, "specular").text = "0.25 0.25 0.25 1"


def _add_spherical_coordinates(
    world: ET.Element,
    origin: Mapping[str, Any],
) -> None:
    coordinates = ET.SubElement(world, "spherical_coordinates")
    ET.SubElement(coordinates, "surface_model").text = "EARTH_WGS84"
    ET.SubElement(coordinates, "world_frame_orientation").text = "ENU"
    ET.SubElement(coordinates, "latitude_deg").text = f"{float(origin['latitude_deg']):.8f}"
    ET.SubElement(coordinates, "longitude_deg").text = f"{float(origin['longitude_deg']):.8f}"
    ET.SubElement(coordinates, "elevation").text = f"{float(origin.get('elevation_m', 0.0)):g}"


def _add_ground(world: ET.Element, width: float, height: float) -> None:
    model = ET.SubElement(world, "model", {"name": "ground_plane"})
    ET.SubElement(model, "static").text = "true"
    link = ET.SubElement(model, "link", {"name": "link"})
    collision = ET.SubElement(link, "collision", {"name": "collision"})
    plane = ET.SubElement(ET.SubElement(collision, "geometry"), "plane")
    ET.SubElement(plane, "normal").text = "0 0 1"
    ET.SubElement(plane, "size").text = f"{width:g} {height:g}"
    visual = ET.SubElement(link, "visual", {"name": "visual"})
    visual_plane = ET.SubElement(ET.SubElement(visual, "geometry"), "plane")
    ET.SubElement(visual_plane, "normal").text = "0 0 1"
    ET.SubElement(visual_plane, "size").text = f"{width:g} {height:g}"
    _add_material(visual, (0.30, 0.34, 0.28, 1.0))


def _add_zone_visual(world: ET.Element, zone: Mapping[str, Any]) -> None:
    x_min, y_min, x_max, y_max = zone["bounds"]
    _add_box_model(
        world,
        f"semantic-{zone['id']}",
        ((x_min + x_max) / 2.0, (y_min + y_max) / 2.0, 0.015),
        (x_max - x_min, y_max - y_min, 0.02),
        ZONE_COLORS[str(zone["type"])],
        collidable=False,
        cast_shadows=False,
    )


def _add_road_markings(world: ET.Element, bounds: Bounds) -> None:
    x_min, _, x_max, _ = bounds
    x = x_min + 3.0
    index = 0
    while x < x_max - 3.0:
        index += 1
        _add_box_model(
            world,
            f"road-marking-{index:02d}",
            (x, 0.0, 0.035),
            (3.0, 0.18, 0.025),
            (0.92, 0.90, 0.68, 1.0),
            collidable=False,
            cast_shadows=False,
        )
        x += 6.0


def _add_road_zone_markings(world: ET.Element, zone: Mapping[str, Any]) -> None:
    x_min, y_min, x_max, y_max = zone["bounds"]
    horizontal = (x_max - x_min) >= (y_max - y_min)
    length_min, length_max = (x_min, x_max) if horizontal else (y_min, y_max)
    fixed = (y_min + y_max) / 2.0 if horizontal else (x_min + x_max) / 2.0
    position = length_min + 3.0
    index = 0
    while position < length_max - 3.0:
        index += 1
        center = (position, fixed, 0.04) if horizontal else (fixed, position, 0.04)
        size = (3.0, 0.18, 0.025) if horizontal else (0.18, 3.0, 0.025)
        _add_box_model(
            world,
            f"{zone['id']}-marking-{index:02d}",
            center,
            size,
            (0.92, 0.90, 0.68, 1.0),
            collidable=False,
            cast_shadows=False,
        )
        position += 6.0


def _add_box_model(
    world: ET.Element,
    name: str,
    center: Sequence[float],
    size: Sequence[float],
    color: Color,
    *,
    yaw: float = 0.0,
    collidable: bool = True,
    cast_shadows: bool = True,
) -> None:
    model = ET.SubElement(world, "model", {"name": str(name)})
    ET.SubElement(model, "static").text = "true"
    ET.SubElement(model, "pose").text = _numbers((*center, 0.0, 0.0, yaw))
    link = ET.SubElement(model, "link", {"name": "link"})
    if collidable:
        collision = ET.SubElement(link, "collision", {"name": "collision"})
        box = ET.SubElement(ET.SubElement(collision, "geometry"), "box")
        ET.SubElement(box, "size").text = _numbers(size)
    visual = ET.SubElement(link, "visual", {"name": "visual"})
    box = ET.SubElement(ET.SubElement(visual, "geometry"), "box")
    ET.SubElement(box, "size").text = _numbers(size)
    ET.SubElement(visual, "cast_shadows").text = str(cast_shadows).lower()
    _add_material(visual, color)


def _add_tree(world: ET.Element, tree: Mapping[str, Any]) -> None:
    height = float(tree["height"])
    model = ET.SubElement(world, "model", {"name": str(tree["id"])})
    ET.SubElement(model, "static").text = "true"
    ET.SubElement(model, "pose").text = _numbers((*tree["center"], 0.0, 0.0, 0.0, 0.0))
    link = ET.SubElement(model, "link", {"name": "tree"})
    collision = ET.SubElement(link, "collision", {"name": "trunk_collision"})
    ET.SubElement(collision, "pose").text = f"0 0 {height * 0.35:g} 0 0 0"
    cylinder = ET.SubElement(ET.SubElement(collision, "geometry"), "cylinder")
    ET.SubElement(cylinder, "radius").text = "0.32"
    ET.SubElement(cylinder, "length").text = f"{height * 0.7:g}"
    trunk = ET.SubElement(link, "visual", {"name": "trunk"})
    ET.SubElement(trunk, "pose").text = f"0 0 {height * 0.35:g} 0 0 0"
    cylinder = ET.SubElement(ET.SubElement(trunk, "geometry"), "cylinder")
    ET.SubElement(cylinder, "radius").text = "0.28"
    ET.SubElement(cylinder, "length").text = f"{height * 0.7:g}"
    _add_material(trunk, (0.28, 0.16, 0.08, 1.0))
    canopy = ET.SubElement(link, "visual", {"name": "canopy"})
    ET.SubElement(canopy, "pose").text = f"0 0 {height * 0.78:g} 0 0 0"
    sphere = ET.SubElement(ET.SubElement(canopy, "geometry"), "sphere")
    ET.SubElement(sphere, "radius").text = f"{height * 0.25:g}"
    _add_material(canopy, (0.10, 0.36, 0.12, 1.0))


def _add_vehicle(world: ET.Element, vehicle: Mapping[str, Any], *, yellow: bool) -> None:
    model = ET.SubElement(world, "model", {"name": str(vehicle["id"])})
    ET.SubElement(model, "static").text = "true"
    ET.SubElement(model, "pose").text = _numbers((*vehicle["center"], 0.0, 0.0, 0.0, vehicle["yaw"]))
    link = ET.SubElement(model, "link", {"name": "body"})
    collision = ET.SubElement(link, "collision", {"name": "body_collision"})
    ET.SubElement(collision, "pose").text = "0 0 0.5 0 0 0"
    box = ET.SubElement(ET.SubElement(collision, "geometry"), "box")
    ET.SubElement(box, "size").text = "2.2 1.05 1.0"
    visual = ET.SubElement(link, "visual", {"name": "yellow_body" if yellow else "body_visual"})
    ET.SubElement(visual, "pose").text = "0 0 0.5 0 0 0"
    box = ET.SubElement(ET.SubElement(visual, "geometry"), "box")
    ET.SubElement(box, "size").text = "2.2 1.05 1.0"
    _add_material(visual, vehicle["color"])
    roof = ET.SubElement(link, "visual", {"name": "roof"})
    ET.SubElement(roof, "pose").text = "-0.15 0 1.05 0 0 0"
    roof_box = ET.SubElement(ET.SubElement(roof, "geometry"), "box")
    ET.SubElement(roof_box, "size").text = "1.15 0.94 0.35"
    _add_material(roof, vehicle["color"])


def _add_utility_pole(world: ET.Element, pole: Mapping[str, Any]) -> None:
    x, y = pole["center"]
    height = float(pole["height"])
    _add_box_model(
        world,
        str(pole["id"]),
        (x, y, height / 2.0),
        (0.18, 0.18, height),
        (0.24, 0.25, 0.24, 1.0),
    )
    _add_box_model(
        world,
        f"{pole['id']}-lamp",
        (x + 0.38, y, height - 0.15),
        (0.8, 0.28, 0.22),
        (0.66, 0.65, 0.57, 1.0),
        collidable=False,
    )


def _add_restricted_fence(world: ET.Element, bounds: Bounds, *, prefix: str = "restricted") -> None:
    x_min, y_min, x_max, y_max = bounds
    color = (0.62, 0.12, 0.10, 1.0)
    for name, center, size in (
        ("south", ((x_min + x_max) / 2.0, y_min, 1.2), (x_max - x_min, 0.22, 2.4)),
        ("north", ((x_min + x_max) / 2.0, y_max, 1.2), (x_max - x_min, 0.22, 2.4)),
        ("west", (x_min, (y_min + y_max) / 2.0, 1.2), (0.22, y_max - y_min, 2.4)),
        ("east", (x_max, (y_min + y_max) / 2.0, 1.2), (0.22, y_max - y_min, 2.4)),
    ):
        _add_box_model(world, f"{prefix}-fence-{name}", center, size, color)


def _add_material(visual: ET.Element, color: Color) -> None:
    material = ET.SubElement(visual, "material")
    value = _numbers(color)
    ET.SubElement(material, "ambient").text = value
    ET.SubElement(material, "diffuse").text = value
    ET.SubElement(material, "specular").text = "0.12 0.12 0.12 1"


def _semantic_map(config: Mapping[str, Any], layout: Mapping[str, Any]) -> Dict[str, Any]:
    nodes = []
    for zone in layout["zones"]:
        nodes.append({
            "id": zone["id"],
            "properties": {
                "category": zone["category"],
                "type": zone["type"],
                "label": zone["id"],
                "passability": zone["passability"],
                "description": f"SearchWorld {zone['type'].replace('_', ' ')} region",
                "visibility": "public",
            },
            "shape": _rectangle_shape(zone["bounds"]),
        })
    for building in layout["buildings"]:
        x, y = building["center"]
        sx, sy, _ = building["size"]
        nodes.append({
            "id": building["id"],
            "properties": {
                "category": "building",
                "type": building["type"],
                "label": building["id"],
                "passability": "restricted",
                "description": "Physical building footprint",
                "visibility": "public",
            },
            "shape": _rectangle_shape((x - sx / 2.0, y - sy / 2.0, x + sx / 2.0, y + sy / 2.0)),
        })
    return {
        "schema_version": "1.0",
        "world_name": config["world"]["name"],
        "coordinate_frame": "ENU",
        "units": "meters",
        "search_area": _rectangle_geometry(layout["bounds"]),
        "nodes": nodes,
        "metadata": {
            "seed": int(config["world"].get("seed", 0)),
            "layout_archetype": layout["layout_archetype"],
            "ground_truth_excluded": True,
            "generator": "gsi_search_bridge.search_world_generator",
        },
    }


def _search_prior(config: Mapping[str, Any]) -> Dict[str, Any]:
    prior = config["semantic_prior"]
    return {
        "semantic_weights": dict(prior["semantic_weights"]),
        "confidence": float(prior["confidence"]),
        "default_weight": float(prior.get("default_weight", 0.05)),
        "excluded_labels": list(prior.get("excluded_labels", ())),
        "projection_mode": str(prior.get("projection_mode", "cell_affinity")),
        "metadata": {
            "source": "searchworld_task_conditioned_prior_fixture",
            "target_query": config["target"]["query"],
        },
    }


def _ground_truth(config: Mapping[str, Any], layout: Mapping[str, Any]) -> Dict[str, Any]:
    target = config["target"]
    slot = layout["selected_target_slot"]
    x, y, z, yaw = slot["pose"]
    return {
        "schema_version": "1.0",
        "evaluator_only": True,
        "world_name": config["world"]["name"],
        "seed": int(config["world"].get("seed", 0)),
        "targets": [{
            "entity_id": target["entity_id"],
            "model_name": target["model_name"],
            "query": target["query"],
            "semantic_region_id": slot["region_id"],
            "slot_id": slot["id"],
            "pose_enu_m": {"x": x, "y": y, "z": z, "yaw_rad": yaw},
        }],
    }


def _manifest(
    config: Mapping[str, Any],
    layout: Mapping[str, Any],
    artifacts: Mapping[str, Path],
) -> Dict[str, Any]:
    public_keys = (
        "world",
        "semantic_map",
        "search_prior",
        "search_params",
        "gz_bridge",
        "visionflow_profile",
    )
    return {
        "schema_version": "1.0",
        "scenario_id": config["world"]["name"],
        "seed": int(config["world"].get("seed", 0)),
        "coordinate_frame": "ENU",
        "units": "meters",
        "search_area": _rectangle_geometry(layout["bounds"]),
        "target_query": config["target"]["query"],
        "target_slot_count": len(layout["target_slots"]),
        "ground_truth_policy": "evaluator-only; never load into search policy",
        "complexity": {
            "layout_archetype": layout["layout_archetype"],
            "semantic_region_count": len(layout["zones"]),
            "building_count": len(layout["buildings"]),
            "tree_count": len(layout["trees"]),
            "parked_vehicle_count": len(layout["parked_vehicles"]),
            "container_count": len(layout["containers"]),
            "barrier_count": len(layout["barriers"]),
            "utility_pole_count": len(layout["utility_poles"]),
        },
        "artifacts": {
            key: {
                "filename": artifacts[key].name,
                "sha256": _sha256(artifacts[key]),
            }
            for key in public_keys
        },
        "private_artifacts": {
            "ground_truth": {
                "filename": artifacts["ground_truth"].name,
                "sha256": _sha256(artifacts["ground_truth"]),
            }
        },
    }


def _search_params_yaml(config: Mapping[str, Any]) -> str:
    world = config["world"]
    search = config["search"]
    target = config["target"]
    hx = float(world["size_m"]["x"]) / 2.0
    hy = float(world["size_m"]["y"]) / 2.0
    scenario_name = str(world["name"])
    scenario_directory = str(config.get("artifact_directory", "search_world_v1"))
    semantic_path = f"simulation/{scenario_directory}/generated/semantic_map.json"
    prior_path = f"simulation/{scenario_directory}/generated/search_prior.json"
    sensor = config.get("sensor") or {
        "frame_id": "camera_link",
        "maximum_range_m": 19.1,
        "extrinsics": {
            "x_m": 0.121998,
            "y_m": -0.002,
            "z_m": 0.064561,
            "roll_rad": 0.0,
            "pitch_rad": 0.785398,
            "yaw_rad": 0.0,
        },
    }
    extrinsics = sensor["extrinsics"]
    prestream_setpoint_count = int(
        config.get("execution", {}).get("prestream_setpoint_count", 40)
    )
    return f"""gsi_search_node:
  ros__parameters:
    use_sim_time: true
    target_query: {target['query']}
    area_id: {world['name']}
    area_min_x_m: {_yaml_float(-hx)}
    area_min_y_m: {_yaml_float(-hy)}
    area_max_x_m: {_yaml_float(hx)}
    area_max_y_m: {_yaml_float(hy)}
    grid_resolution_m: {_yaml_float(search['grid_resolution_m'])}
    flight_altitude_m: {_yaml_float(search['flight_altitude_m'])}
    sensor_footprint_radius_m: {_yaml_float(search['sensor_footprint_radius_m'])}
    max_viewpoints: {int(search['max_viewpoints'])}
    min_confirmations: 2
    max_localization_error_m: 5.0
    verification_followup_limit: 0
    semantic_map_path: {semantic_path}
    search_prior_path: {prior_path}
    sensor_detection_probability: 0.85
    sensor_false_positive_probability: 0.01
    position_tolerance_m: 1.0
    velocity_tolerance_mps: 0.35
    settle_time_s: 1.0
    goal_republish_interval_s: 0.5
    sensor_timeout_s: 1.0
    maximum_sensor_skew_s: 0.5
    require_detections: true
    require_point_cloud: true
    detections_in_map_frame: true
    camera_translation_x_m: {_yaml_float(extrinsics['x_m'])}
    camera_translation_y_m: {_yaml_float(extrinsics['y_m'])}
    camera_translation_z_m: {_yaml_float(extrinsics['z_m'])}
    camera_roll_rad: {_yaml_float(extrinsics['roll_rad'])}
    camera_pitch_rad: {_yaml_float(extrinsics['pitch_rad'])}
    camera_yaw_rad: {_yaml_float(extrinsics['yaw_rad'])}
    ground_plane_z_m: {_yaml_float(world.get('ground_z_m', 0.0))}
    ground_tolerance_m: 0.35
    visibility_point_resolution_m: 0.5
    pointcloud_sample_limit: 10000
    pointcloud_maximum_range_m: {_yaml_float(sensor['maximum_range_m'])}
    pointcloud_frame_id: {sensor['frame_id']}
    odom_topic: /mavros/local_position/odom
    imu_topic: /mavros/imu/data
    rgb_topic: /oakd1/rgb/image
    camera_info_topic: /oakd1/rgb/camera_info
    depth_topic: /oakd1/depth/image
    point_cloud_topic: /oakd1/depth/points
    detections_topic: /gsi/detections
    battery_topic: /mavros/battery
    goal_pose_topic: /gsi/uav/goal_pose
    outcome_topic: /gsi/search/outcome
    trace_output_path: /tmp/GSI/results/gazebo_sensor_validation/{scenario_name}_trace.jsonl

gsi_mavros_offboard_controller:
  ros__parameters:
    use_sim_time: true
    state_topic: /mavros/state
    odom_topic: /mavros/local_position/odom
    goal_pose_topic: /gsi/uav/goal_pose
    setpoint_topic: /mavros/setpoint_position/local
    arming_service: /mavros/cmd/arming
    set_mode_service: /mavros/set_mode
    map_frame: map
    setpoint_rate_hz: 20.0
    prestream_setpoint_count: {prestream_setpoint_count}
    request_interval_s: 2.0
    auto_offboard: true
    auto_arm: true
    staged_takeoff: true
    takeoff_altitude_tolerance_m: 0.5
    horizontal_setpoint_speed_mps: 1.5

gsi_color_target_detector:
  ros__parameters:
    use_sim_time: true
    rgb_topic: /oakd1/rgb/image
    rgb_camera_info_topic: /oakd1/rgb/camera_info
    depth_topic: /oakd1/depth/image
    depth_camera_info_topic: /oakd1/depth/camera_info
    odom_topic: /mavros/local_position/odom
    detections_topic: /gsi/detections
    target_label: {target['query']}
    target_entity_id: {target['entity_id']}
    detection_rate_hz: 5.0
    sensor_timeout_s: 0.5
    minimum_red: 150
    minimum_green: 120
    maximum_blue: 120
    minimum_yellow_margin: 60
    maximum_red_green_difference: 100
    minimum_pixels: 250
    depth_window_radius_px: 6
    minimum_depth_m: 0.2
    maximum_depth_m: {_yaml_float(sensor['maximum_range_m'])}
    camera_translation_x_m: {_yaml_float(extrinsics['x_m'])}
    camera_translation_y_m: {_yaml_float(extrinsics['y_m'])}
    camera_translation_z_m: {_yaml_float(extrinsics['z_m'])}
    camera_roll_rad: {_yaml_float(extrinsics['roll_rad'])}
    camera_pitch_rad: {_yaml_float(extrinsics['pitch_rad'])}
    camera_yaw_rad: {_yaml_float(extrinsics['yaw_rad'])}
"""


def _gz_bridge_yaml(config: Mapping[str, Any]) -> str:
    world_name = str(config["world"]["name"])
    sensor = config.get("sensor")
    if sensor:
        prefix = str(sensor["gz_topic_root"]).rstrip("/")
        rgb_suffix = str(sensor["rgb_image_suffix"])
        camera_info_suffix = str(sensor["camera_info_suffix"])
        depth_suffix = str(sensor["depth_image_suffix"])
        points_suffix = str(sensor["point_cloud_suffix"])
    else:
        prefix = (
            f"/world/{world_name}/model/q940_ti_gripper4_0/model/"
            "oakd_lite_one/link/camera_link/sensor"
        )
        rgb_suffix = "/IMX214/image"
        camera_info_suffix = "/IMX214/camera_info"
        depth_suffix = "/StereoOV7251/depth_image"
        points_suffix = "/StereoOV7251/depth_image/points"
    pairs = (
        ("/clock", "/clock", "rosgraph_msgs/msg/Clock", "gz.msgs.Clock"),
        ("/oakd1/rgb/image", f"{prefix}{rgb_suffix}", "sensor_msgs/msg/Image", "gz.msgs.Image"),
        ("/oakd1/rgb/camera_info", f"{prefix}{camera_info_suffix}", "sensor_msgs/msg/CameraInfo", "gz.msgs.CameraInfo"),
        ("/oakd1/depth/image", f"{prefix}{depth_suffix}", "sensor_msgs/msg/Image", "gz.msgs.Image"),
        ("/oakd1/depth/camera_info", f"{prefix}{camera_info_suffix}", "sensor_msgs/msg/CameraInfo", "gz.msgs.CameraInfo"),
        ("/oakd1/depth/points", f"{prefix}{points_suffix}", "sensor_msgs/msg/PointCloud2", "gz.msgs.PointCloudPacked"),
    )
    sections = []
    for ros_topic, gz_topic, ros_type, gz_type in pairs:
        sections.append(
            "\n".join((
                f"- ros_topic_name: {ros_topic}",
                f"  gz_topic_name: {gz_topic}",
                f"  ros_type_name: {ros_type}",
                f"  gz_type_name: {gz_type}",
                "  direction: GZ_TO_ROS",
            ))
        )
    return "\n\n".join(sections) + "\n"


def _visionflow_profile(config: Mapping[str, Any]) -> str:
    visionflow = config["visionflow"]
    world_name = str(config["world"]["name"])
    return f"""

# BEGIN {visionflow['profile_id']} profile (managed).
add_sitl_profile \\
    --id \"{visionflow['profile_id']}\" \\
    --name \"{visionflow['profile_name']}\" \\
    --world \"{world_name}\" \\
    --target \"{visionflow['px4_target']}\" \\
    --pose \"{visionflow['spawn_pose']}\"
# END {visionflow['profile_id']} profile (managed).
"""


def _rectangle_shape(bounds: Bounds) -> Dict[str, Any]:
    return {
        "type": "rectangle",
        "min_corner": [bounds[0], bounds[1]],
        "max_corner": [bounds[2], bounds[3]],
    }


def _rectangle_geometry(bounds: Bounds) -> Dict[str, Any]:
    return {
        "kind": "rectangle",
        "coords": [
            [bounds[0], bounds[1]],
            [bounds[2], bounds[1]],
            [bounds[2], bounds[3]],
            [bounds[0], bounds[3]],
        ],
    }


def _numbers(values: Sequence[float]) -> str:
    return " ".join(f"{float(value):.6g}" for value in values)


def _yaml_float(value: Any) -> str:
    formatted = f"{float(value):.6f}".rstrip("0").rstrip(".")
    return formatted if "." in formatted else formatted + ".0"


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def main(args: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    options = parser.parse_args(args)
    artifacts = generate_artifacts(load_config(options.config), options.output)
    for name, path in artifacts.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
