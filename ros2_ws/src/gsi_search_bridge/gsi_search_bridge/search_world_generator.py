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
}


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

    search = config["search"]
    resolution = float(search.get("grid_resolution_m", 0.0))
    altitude = float(search.get("flight_altitude_m", 0.0))
    if resolution <= 0.0:
        raise ValueError("search.grid_resolution_m must be positive")
    if altitude < 10.0:
        raise ValueError("search.flight_altitude_m must clear the V1 obstacles")
    if int(search.get("max_viewpoints", 0)) <= 0:
        raise ValueError("search.max_viewpoints must be positive")

    complexity = config["complexity"]
    for name in ("tree_count", "parked_vehicle_count", "container_count"):
        if int(complexity.get(name, -1)) < 0:
            raise ValueError(f"complexity.{name} must not be negative")

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
        _gz_bridge_yaml(str(config["world"]["name"])),
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
    }


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
    _add_road_markings(world, layout["bounds"])

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
    _add_restricted_fence(world, _zone_by_id(layout["zones"], "restricted-northeast")["bounds"])

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


def _add_restricted_fence(world: ET.Element, bounds: Bounds) -> None:
    x_min, y_min, x_max, y_max = bounds
    color = (0.62, 0.12, 0.10, 1.0)
    for name, center, size in (
        ("south", ((x_min + x_max) / 2.0, y_min, 1.2), (x_max - x_min, 0.22, 2.4)),
        ("north", ((x_min + x_max) / 2.0, y_max, 1.2), (x_max - x_min, 0.22, 2.4)),
        ("west", (x_min, (y_min + y_max) / 2.0, 1.2), (0.22, y_max - y_min, 2.4)),
        ("east", (x_max, (y_min + y_max) / 2.0, 1.2), (0.22, y_max - y_min, 2.4)),
    ):
        _add_box_model(world, f"restricted-fence-{name}", center, size, color)


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
                "description": f"SearchWorld V1 {zone['type'].replace('_', ' ')} region",
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
        "metadata": {
            "source": "searchworld_v1_task_conditioned_prior",
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
            "semantic_region_count": len(layout["zones"]),
            "building_count": len(layout["buildings"]),
            "tree_count": len(layout["trees"]),
            "parked_vehicle_count": len(layout["parked_vehicles"]),
            "container_count": len(layout["containers"]),
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
    semantic_path = "simulation/search_world_v1/generated/semantic_map.json"
    prior_path = "simulation/search_world_v1/generated/search_prior.json"
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
    camera_translation_x_m: 0.121998
    camera_translation_y_m: -0.002
    camera_translation_z_m: 0.064561
    camera_roll_rad: 0.0
    camera_pitch_rad: 0.785398
    camera_yaw_rad: 0.0
    ground_plane_z_m: {_yaml_float(world.get('ground_z_m', 0.0))}
    ground_tolerance_m: 0.35
    visibility_point_resolution_m: 0.5
    pointcloud_sample_limit: 10000
    pointcloud_maximum_range_m: 19.1
    pointcloud_frame_id: camera_link
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
    prestream_setpoint_count: 40
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
    maximum_depth_m: 19.1
    camera_translation_x_m: 0.121998
    camera_translation_y_m: -0.002
    camera_translation_z_m: 0.064561
    camera_roll_rad: 0.0
    camera_pitch_rad: 0.785398
    camera_yaw_rad: 0.0
"""


def _gz_bridge_yaml(world_name: str) -> str:
    prefix = (
        f"/world/{world_name}/model/q940_ti_gripper4_0/model/"
        "oakd_lite_one/link/camera_link/sensor"
    )
    pairs = (
        ("/clock", "/clock", "rosgraph_msgs/msg/Clock", "gz.msgs.Clock"),
        ("/oakd1/rgb/image", f"{prefix}/IMX214/image", "sensor_msgs/msg/Image", "gz.msgs.Image"),
        ("/oakd1/rgb/camera_info", f"{prefix}/IMX214/camera_info", "sensor_msgs/msg/CameraInfo", "gz.msgs.CameraInfo"),
        ("/oakd1/depth/image", f"{prefix}/StereoOV7251/depth_image", "sensor_msgs/msg/Image", "gz.msgs.Image"),
        ("/oakd1/depth/camera_info", f"{prefix}/StereoOV7251/camera_info", "sensor_msgs/msg/CameraInfo", "gz.msgs.CameraInfo"),
        ("/oakd1/depth/points", f"{prefix}/StereoOV7251/depth_image/points", "sensor_msgs/msg/PointCloud2", "gz.msgs.PointCloudPacked"),
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

# BEGIN GSI SearchWorld V1 profile (managed).
add_sitl_profile \\
    --id \"{visionflow['profile_id']}\" \\
    --name \"{visionflow['profile_name']}\" \\
    --world \"{world_name}\" \\
    --target \"{visionflow['px4_target']}\" \\
    --pose \"{visionflow['spawn_pose']}\"
# END GSI SearchWorld V1 profile (managed).
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
