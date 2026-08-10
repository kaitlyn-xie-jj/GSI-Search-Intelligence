"""Offline geometry and ideal-visibility validation for Yungu coverage search."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Dict, Iterable, Tuple

import yaml

from modules.search_intelligence import (
    CoveragePolicy,
    SearchGrid,
    SearchState,
    SearchTask,
    Viewpoint,
)
from ros2_ws.src.gsi_search_bridge.gsi_search_bridge.building_route_planner import (
    load_building_obstacles,
    plan_building_avoiding_route,
    point_has_building_clearance,
)
from ros2_ws.src.gsi_search_bridge.gsi_search_bridge.scenario_context import (
    load_search_scenario_context,
)


ROOT = Path(__file__).resolve().parents[1]


def validate(config_path: Path) -> Dict[str, object]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    params = config["gsi_search_node"]["ros__parameters"]
    search_bounds = _bounds(params, "area")
    navigation_bounds = (
        _bounds(params, "navigation")
        if params.get("navigation_bounds_enabled")
        else search_bounds
    )
    task = SearchTask.from_skill_params({
        "task_id": "yungu-coverage-offline-validation",
        "area_token": params["area_id"],
        "area": {
            "kind": "rectangle",
            "coords": _rectangle(search_bounds),
        },
        "target_token": params["target_query"],
        "max_viewpoints": int(params["max_viewpoints"]),
    })
    grid = SearchGrid.from_task(task, float(params["grid_resolution_m"]))
    semantic_path = ROOT / "data/yungu2030_v1/semantic_map.json"
    prior_path = ROOT / "ros2_ws/simulation/yungu2030_v1/yungu_search_prior.json"
    grid = load_search_scenario_context(
        task,
        grid,
        semantic_map_path=str(semantic_path),
        search_prior_path=str(prior_path),
    ).grid
    obstacles = load_building_obstacles(str(semantic_path))
    clearance = float(params["building_horizontal_clearance_m"])
    vertical_clearance = float(params["building_vertical_clearance_m"])

    def safe(viewpoint: Viewpoint) -> bool:
        return (
            navigation_bounds[0] <= viewpoint.x <= navigation_bounds[2]
            and navigation_bounds[1] <= viewpoint.y <= navigation_bounds[3]
            and point_has_building_clearance(
            (viewpoint.x, viewpoint.y, viewpoint.z),
            obstacles,
            horizontal_clearance_m=clearance,
            vertical_clearance_m=vertical_clearance,
            )
        )

    start = Viewpoint(
        float(params["map_origin_x_m"]),
        float(params["map_origin_y_m"]),
        float(params["flight_altitude_m"]),
        0.0,
    )
    policy = CoveragePolicy(
        pass_spacing_m=float(params["coverage_pass_spacing_m"]),
        altitude_m=float(params["flight_altitude_m"]),
        camera_pitch_rad=float(params["coverage_camera_pitch_rad"]),
        observation_spacing_m=float(params["coverage_observation_spacing_m"]),
        start_from_nearest_endpoint=bool(
            params["coverage_start_from_nearest_endpoint"]
        ),
        route_start_hint=start,
        viewpoint_filter=safe,
        search_grid=grid,
        recovery_enabled=bool(params["coverage_recovery_enabled"]),
        recovery_min_quality=float(params["coverage_recovery_min_quality"]),
        recovery_offset_m=float(params["coverage_recovery_offset_m"]),
    )
    state = SearchState.initial(task, {}, current_viewpoint=start)
    primary = policy.plan(state)
    route_result = _validate_routes(
        start,
        primary,
        obstacles,
        navigation_bounds,
        clearance,
        vertical_clearance,
        float(params["building_corner_offset_m"]),
    )
    if route_result["unreachable_segments"]:
        raise RuntimeError(
            f"primary route has {route_result['unreachable_segments']} unreachable segments"
        )

    half_width, half_height = _footprint(params)
    coverage = _observe(grid, primary, half_width, half_height)
    state = replace(
        state,
        current_viewpoint=primary[-1] if primary else start,
        visited_viewpoint_keys=tuple(item.key for item in primary),
        observed_cell_quality=coverage,
        step_index=len(primary),
    )
    recovery_count = 0
    recovery_route_distance = 0.0
    recovery_detours = 0
    while True:
        remaining = policy.plan(state)
        if not remaining:
            break
        viewpoint = remaining[0]
        recovery_route = _validate_routes(
            state.current_viewpoint or start,
            (viewpoint,),
            obstacles,
            navigation_bounds,
            clearance,
            vertical_clearance,
            float(params["building_corner_offset_m"]),
        )
        if recovery_route["unreachable_segments"]:
            raise RuntimeError(f"recovery viewpoint is unreachable: {viewpoint.key}")
        recovery_route_distance += float(recovery_route["distance_m"])
        recovery_detours += int(recovery_route["detour_waypoints"])
        coverage.update(_observe(grid, (viewpoint,), half_width, half_height))
        state = replace(
            state,
            current_viewpoint=viewpoint,
            visited_viewpoint_keys=state.visited_viewpoint_keys + (viewpoint.key,),
            observed_cell_quality=coverage,
            step_index=state.step_index + 1,
        )
        recovery_count += 1
        if state.step_index >= int(params["max_viewpoints"]):
            break

    statuses = policy.coverage_cell_states(state)
    deferred = sorted(
        cell_id for cell_id, status in statuses.items() if status == "DEFERRED"
    )
    if deferred:
        raise RuntimeError(
            f"ideal-visibility validation left {len(deferred)} deferred cells: "
            + ", ".join(deferred[:10])
        )
    return {
        "search_bounds": search_bounds,
        "navigation_bounds": navigation_bounds,
        "grid_cells": len(grid.cells),
        "searchable_cells": len(grid.searchable_cells),
        "excluded_cells": len(grid.cells) - len(grid.searchable_cells),
        "primary_viewpoints": len(primary),
        "primary_route_distance_m": round(float(route_result["distance_m"]), 3),
        "primary_detour_waypoints": route_result["detour_waypoints"],
        "recovery_viewpoints": recovery_count,
        "recovery_route_distance_m": round(recovery_route_distance, 3),
        "recovery_detour_waypoints": recovery_detours,
        "covered_cells": sum(status == "COVERED" for status in statuses.values()),
        "deferred_cells": len(deferred),
        "excluded_status_cells": sum(
            status == "EXCLUDED" for status in statuses.values()
        ),
        "validation": "passed",
    }


def _bounds(params: Dict[str, object], prefix: str) -> Tuple[float, float, float, float]:
    return tuple(
        float(params[f"{prefix}_{name}_{axis}_m"])
        for name, axis in (("min", "x"), ("min", "y"), ("max", "x"), ("max", "y"))
    )  # type: ignore[return-value]


def _rectangle(bounds: Tuple[float, float, float, float]):
    x_min, y_min, x_max, y_max = bounds
    return [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]


def _footprint(params: Dict[str, object]) -> Tuple[float, float]:
    altitude = float(params["flight_altitude_m"])
    half_width = altitude * math.tan(float(params["camera_horizontal_fov_rad"]) / 2.0)
    half_height = half_width * float(params["camera_image_height_px"]) / float(
        params["camera_image_width_px"]
    )
    scale = float(params["planning_footprint_scale"])
    return half_width * scale, half_height * scale


def _observe(
    grid: SearchGrid,
    viewpoints: Iterable[Viewpoint],
    half_width: float,
    half_height: float,
) -> Dict[str, float]:
    coverage: Dict[str, float] = {}
    for viewpoint in viewpoints:
        visible = grid.cells_within_oriented_rectangle(
            viewpoint.x,
            viewpoint.y,
            half_width,
            half_height,
            yaw_rad=viewpoint.yaw,
        )
        coverage.update({cell.cell_id: 1.0 for cell in visible})
    return coverage


def _validate_routes(
    start: Viewpoint,
    goals: Iterable[Viewpoint],
    obstacles,
    bounds,
    clearance: float,
    vertical_clearance: float,
    corner_offset: float,
) -> Dict[str, object]:
    current = start
    distance = 0.0
    detours = 0
    unreachable = 0
    for goal in goals:
        route = plan_building_avoiding_route(
            (current.x, current.y, current.z),
            (goal.x, goal.y, goal.z),
            obstacles,
            horizontal_clearance_m=clearance,
            vertical_clearance_m=vertical_clearance,
            corner_offset_m=corner_offset,
            route_bounds=bounds,
        )
        if route is None:
            unreachable += 1
            current = goal
            continue
        points = ((current.x, current.y, current.z),) + tuple(route)
        distance += sum(math.dist(first, second) for first, second in zip(points, points[1:]))
        detours += max(0, len(route) - 1)
        current = goal
    return {
        "distance_m": distance,
        "detour_waypoints": detours,
        "unreachable_segments": unreachable,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "ros2_ws/simulation/yungu2030_v1/yungu_search_params.yaml",
    )
    parser.add_argument("--output", type=Path)
    options = parser.parse_args()
    result = validate(options.config.resolve())
    payload = json.dumps(result, indent=2, sort_keys=True)
    print(payload)
    if options.output:
        options.output.parent.mkdir(parents=True, exist_ok=True)
        options.output.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
