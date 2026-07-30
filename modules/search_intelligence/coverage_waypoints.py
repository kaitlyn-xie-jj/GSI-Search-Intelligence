"""Geometry-only waypoint generation shared by coverage search implementations."""

import math
from typing import Any, Dict, List, Sequence, Tuple


def _linspace(start: float, stop: float, count: int) -> List[float]:
    if count <= 1:
        return [float(start)]
    step = (stop - start) / (count - 1)
    return [start + index * step for index in range(count)]


def plan_search_waypoints(
    area_geom: Dict[str, Any],
    pass_spacing: float = 40.0,
    zigzag: bool = True,
    max_points: int = 3000,
) -> List[List[float]]:
    """Generate the legacy GSI coverage route for a supported area geometry."""
    kind = area_geom.get("kind")
    if kind in ("rectangle", "area"):
        coords = area_geom.get("coords") or []
        return _waypoints_polygon(coords, pass_spacing, zigzag, max_points) if coords else []
    if kind == "circle":
        center = area_geom.get("center")
        radius = float(area_geom.get("radius", 0.0) or 0.0)
        return _waypoints_circle_chords(center, radius, pass_spacing, zigzag, max_points) \
            if center and radius > 0 else []
    if kind == "line":
        line = area_geom.get("coords") or []
        if len(line) == 2:
            return _waypoints_line_backforth(line, int(area_geom.get("num_passes", 3)))
        width = float(area_geom.get("buffer", 20.0) or 20.0)
        return _waypoints_polyline_band(line, width, pass_spacing, zigzag, max_points)
    if kind == "point":
        points = area_geom.get("coords") or []
        radius = float(area_geom.get("buffer", 30.0) or 30.0)
        return _waypoints_spiral(points[0], radius, pass_spacing, max_points) if points else []
    return []


def _waypoints_polygon(
    polygon: Sequence[Sequence[float]],
    pass_spacing: float,
    zigzag: bool,
    max_points: int,
) -> List[List[float]]:
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    y_min, y_max = min(ys), max(ys)
    if y_max <= y_min:
        return []

    pass_count = max(2, int(math.ceil((y_max - y_min) / pass_spacing)))
    waypoints: List[List[float]] = []
    for row_index, y in enumerate(_linspace(y_min, y_max, pass_count)):
        row: List[List[float]] = []
        for x_left, x_right in _scanline_intersections(polygon, y):
            if x_left > x_right:
                x_left, x_right = x_right, x_left
            endpoints = [[x_left, y], [x_right, y]]
            row.extend(endpoints if not zigzag or row_index % 2 == 0 else reversed(endpoints))
        if row:
            waypoints.extend(row[1:] if waypoints and waypoints[-1] == row[0] else row)
        if len(waypoints) >= max_points:
            break
    return waypoints[:max_points]


def _scanline_intersections(
    polygon: Sequence[Sequence[float]], y: float
) -> List[Tuple[float, float]]:
    intersections: List[float] = []
    for index, first in enumerate(polygon):
        second = polygon[(index + 1) % len(polygon)]
        x1, y1 = first
        x2, y2 = second
        if min(y1, y2) <= y < max(y1, y2) and y2 != y1:
            ratio = (y - y1) / (y2 - y1)
            intersections.append(x1 + ratio * (x2 - x1))
        if y == y2 and y > y1:
            intersections.append(x2)
    intersections.sort()
    return [
        (intersections[index], intersections[index + 1])
        for index in range(0, len(intersections) - 1, 2)
    ]


def _waypoints_circle_chords(
    center: Sequence[float],
    radius: float,
    pass_spacing: float,
    zigzag: bool,
    max_points: int,
) -> List[List[float]]:
    center_x, center_y = center
    pass_count = max(2, int(math.ceil((2 * radius) / pass_spacing)))
    waypoints: List[List[float]] = []
    for row_index, y in enumerate(_linspace(center_y - radius, center_y + radius, pass_count)):
        half_width = math.sqrt(max(radius * radius - (y - center_y) ** 2, 0.0))
        endpoints = [[center_x - half_width, y], [center_x + half_width, y]]
        waypoints.extend(endpoints if not zigzag or row_index % 2 == 0 else reversed(endpoints))
        if len(waypoints) >= max_points:
            break
    return waypoints[:max_points]


def _waypoints_polyline_band(
    line: Sequence[Sequence[float]],
    width: float,
    pass_spacing: float,
    zigzag: bool,
    max_points: int,
) -> List[List[float]]:
    if len(line) < 2:
        return []
    half_count = max(0, int(math.floor((width / 2.0) / pass_spacing)))
    offsets = [index * pass_spacing for index in range(-half_count, half_count + 1)]
    if 0.0 not in offsets:
        offsets = sorted(offsets + [0.0])

    normals = _polyline_vertex_normals(line)
    offset_lines: List[List[List[float]]] = []
    for offset in offsets:
        offset_lines.append([
            [point[0] + offset * normal[0], point[1] + offset * normal[1]]
            for point, normal in zip(line, normals)
        ])

    waypoints: List[List[float]] = []
    for index, offset_line in enumerate(offset_lines):
        sequence = offset_line if not zigzag or index % 2 == 0 else list(reversed(offset_line))
        if waypoints and waypoints[-1] != sequence[0]:
            waypoints.append(sequence[0])
        else:
            waypoints.extend(sequence[:1])
        waypoints.extend(sequence[1:])
        if len(waypoints) >= max_points:
            break
    return waypoints[:max_points]


def _polyline_vertex_normals(line: Sequence[Sequence[float]]) -> List[List[float]]:
    directions: List[Tuple[float, float]] = []
    for first, second in zip(line, line[1:]):
        delta_x, delta_y = second[0] - first[0], second[1] - first[1]
        length = math.hypot(delta_x, delta_y)
        directions.append((delta_x / length, delta_y / length) if length > 1e-9 else (0.0, 0.0))

    normals: List[List[float]] = []
    for index in range(len(line)):
        previous = directions[index - 1] if index > 0 else directions[0]
        following = directions[index] if index < len(directions) else directions[-1]
        direction = (previous[0] + following[0], previous[1] + following[1])
        if math.hypot(*direction) < 1e-9:
            direction = following if math.hypot(*following) > 0 else previous
        normal = (-direction[1], direction[0])
        length = math.hypot(*normal)
        normals.append([normal[0] / length, normal[1] / length] if length > 1e-9 else [0.0, 0.0])
    return normals


def _waypoints_spiral(
    center: Sequence[float], radius: float, pitch: float, max_points: int
) -> List[List[float]]:
    center_x, center_y = center
    turns = max(1.0, radius / max(1e-6, pitch))
    theta_max = 2.0 * math.pi * turns
    radial_scale = pitch / (2.0 * math.pi)
    theta_step = 2.0 * math.pi / 60.0
    waypoints: List[List[float]] = []
    theta = 0.0
    while theta <= theta_max and len(waypoints) < max_points:
        current_radius = radial_scale * theta
        waypoints.append([
            center_x + current_radius * math.cos(theta),
            center_y + current_radius * math.sin(theta),
        ])
        theta += theta_step
    return waypoints


def _waypoints_line_backforth(
    line: Sequence[Sequence[float]], num_passes: int = 3
) -> List[List[float]]:
    if len(line) < 2:
        return []
    first, second = list(line[0]), list(line[1])
    waypoints: List[List[float]] = []
    for index in range(max(0, num_passes)):
        waypoints.extend((first, second) if index % 2 == 0 else (second, first))
    return waypoints
