"""Deterministic 2.5D routing around static rectangular buildings."""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence, Tuple


Point3 = Tuple[float, float, float]


@dataclass(frozen=True)
class BuildingObstacle:
    obstacle_id: str
    min_x: float
    min_y: float
    max_x: float
    max_y: float
    min_z: float
    max_z: float

    def inflated(self, margin_m: float) -> "BuildingObstacle":
        return BuildingObstacle(
            obstacle_id=self.obstacle_id,
            min_x=self.min_x - margin_m,
            min_y=self.min_y - margin_m,
            max_x=self.max_x + margin_m,
            max_y=self.max_y + margin_m,
            min_z=self.min_z,
            max_z=self.max_z,
        )


def load_building_obstacles(path_value: str) -> Tuple[BuildingObstacle, ...]:
    """Read restricted rectangular buildings from a semantic-map document."""
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    with path.open("r", encoding="utf-8") as stream:
        document = json.load(stream)
    nodes = document.get("nodes")
    if nodes is None and isinstance(document.get("scene_graph"), Mapping):
        nodes = document["scene_graph"].get("nodes")
    if not isinstance(nodes, list):
        raise TypeError("semantic map must provide a nodes array")

    obstacles = []
    for node in nodes:
        if not isinstance(node, Mapping):
            continue
        properties = node.get("properties")
        shape = node.get("shape")
        if not isinstance(properties, Mapping) or not isinstance(shape, Mapping):
            continue
        if str(properties.get("category", "")).strip().lower() != "building":
            continue
        if str(shape.get("type", "")).strip().lower() != "rectangle":
            continue
        minimum = shape.get("min_corner")
        maximum = shape.get("max_corner")
        if not isinstance(minimum, Sequence) or not isinstance(maximum, Sequence):
            continue
        if len(minimum) < 2 or len(maximum) < 2:
            continue
        obstacles.append(BuildingObstacle(
            obstacle_id=str(node.get("id") or properties.get("label") or "building"),
            min_x=min(float(minimum[0]), float(maximum[0])),
            min_y=min(float(minimum[1]), float(maximum[1])),
            max_x=max(float(minimum[0]), float(maximum[0])),
            max_y=max(float(minimum[1]), float(maximum[1])),
            min_z=float(properties.get("elevation_min_m", float("-inf"))),
            max_z=float(properties.get("elevation_max_m", float("inf"))),
        ))
    return tuple(obstacles)


def plan_building_avoiding_route(
    start: Point3,
    goal: Point3,
    obstacles: Iterable[BuildingObstacle],
    *,
    horizontal_clearance_m: float,
    vertical_clearance_m: float,
    corner_offset_m: float = 0.05,
    route_bounds: Optional[Tuple[float, float, float, float]] = None,
) -> Optional[Tuple[Point3, ...]]:
    """Return waypoints ending at goal, or None when no safe route exists."""
    if horizontal_clearance_m < 0 or vertical_clearance_m < 0:
        raise ValueError("building clearances must not be negative")
    if corner_offset_m <= 0:
        raise ValueError("corner_offset_m must be positive")
    if route_bounds is not None and (
        not _point_in_bounds(start, route_bounds)
        or not _point_in_bounds(goal, route_bounds)
    ):
        return None

    flight_min_z = min(start[2], goal[2])
    active = tuple(
        obstacle.inflated(horizontal_clearance_m)
        for obstacle in obstacles
        if flight_min_z <= obstacle.max_z + vertical_clearance_m
    )
    if not active or _segment_is_clear(start, goal, active):
        return (goal,)
    if _point_in_any_obstacle(start, active) or _point_in_any_obstacle(goal, active):
        return None

    nodes = [start, goal]
    route_z = goal[2]
    for obstacle in active:
        corners = (
            (obstacle.min_x - corner_offset_m, obstacle.min_y - corner_offset_m, route_z),
            (obstacle.min_x - corner_offset_m, obstacle.max_y + corner_offset_m, route_z),
            (obstacle.max_x + corner_offset_m, obstacle.min_y - corner_offset_m, route_z),
            (obstacle.max_x + corner_offset_m, obstacle.max_y + corner_offset_m, route_z),
        )
        nodes.extend(
            point for point in corners
            if route_bounds is None or _point_in_bounds(point, route_bounds)
        )

    adjacency = [[] for _ in nodes]
    for left in range(len(nodes)):
        for right in range(left + 1, len(nodes)):
            if not _segment_is_clear(nodes[left], nodes[right], active):
                continue
            distance = math.dist(nodes[left], nodes[right])
            adjacency[left].append((right, distance))
            adjacency[right].append((left, distance))

    distances = [float("inf")] * len(nodes)
    previous = [-1] * len(nodes)
    distances[0] = 0.0
    queue = [(0.0, 0)]
    while queue:
        distance, node_index = heapq.heappop(queue)
        if distance != distances[node_index]:
            continue
        if node_index == 1:
            break
        for neighbor, edge_length in adjacency[node_index]:
            candidate = distance + edge_length
            if candidate < distances[neighbor]:
                distances[neighbor] = candidate
                previous[neighbor] = node_index
                heapq.heappush(queue, (candidate, neighbor))
    if not math.isfinite(distances[1]):
        return None

    indices = []
    cursor = 1
    while cursor != 0:
        indices.append(cursor)
        cursor = previous[cursor]
        if cursor < 0:
            return None
    indices.reverse()
    return tuple(nodes[index] for index in indices)


def point_has_building_clearance(
    point: Point3,
    obstacles: Iterable[BuildingObstacle],
    *,
    horizontal_clearance_m: float,
    vertical_clearance_m: float,
) -> bool:
    """Return whether a viewpoint is outside every relevant inflated building."""
    active = (
        obstacle.inflated(horizontal_clearance_m)
        for obstacle in obstacles
        if point[2] <= obstacle.max_z + vertical_clearance_m
    )
    return not _point_in_any_obstacle(point, active)


def segment_intersects_obstacle(
    start: Point3,
    end: Point3,
    obstacle: BuildingObstacle,
) -> bool:
    """Conservatively treat touching an obstacle boundary as intersection."""
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    t_min = 0.0
    t_max = 1.0
    for origin, delta, lower, upper in (
        (start[0], dx, obstacle.min_x, obstacle.max_x),
        (start[1], dy, obstacle.min_y, obstacle.max_y),
    ):
        if abs(delta) < 1e-12:
            if origin < lower or origin > upper:
                return False
            continue
        entry = (lower - origin) / delta
        exit_ = (upper - origin) / delta
        if entry > exit_:
            entry, exit_ = exit_, entry
        t_min = max(t_min, entry)
        t_max = min(t_max, exit_)
        if t_min > t_max:
            return False
    return True


def _segment_is_clear(
    start: Point3,
    end: Point3,
    obstacles: Iterable[BuildingObstacle],
) -> bool:
    return not any(segment_intersects_obstacle(start, end, item) for item in obstacles)


def _point_in_any_obstacle(
    point: Point3,
    obstacles: Iterable[BuildingObstacle],
) -> bool:
    return any(
        obstacle.min_x <= point[0] <= obstacle.max_x
        and obstacle.min_y <= point[1] <= obstacle.max_y
        for obstacle in obstacles
    )


def _point_in_bounds(
    point: Point3,
    bounds: Tuple[float, float, float, float],
) -> bool:
    min_x, min_y, max_x, max_y = bounds
    return min_x <= point[0] <= max_x and min_y <= point[1] <= max_y
