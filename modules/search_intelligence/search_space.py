"""Discrete search space and candidate viewpoints for active search."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from .contracts import SearchArea, SearchTask, Viewpoint


Point2D = Tuple[float, float]
Bounds2D = Tuple[float, float, float, float]


@dataclass(frozen=True)
class SearchCell:
    """One regular cell in a task's search-area bounding grid."""

    cell_id: str
    row: int
    column: int
    center: Point2D
    bounds: Bounds2D
    searchable: bool = True
    semantic_labels: Tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.cell_id.strip():
            raise ValueError("SearchCell.cell_id must not be empty")
        if self.row < 0 or self.column < 0:
            raise ValueError("SearchCell row and column must not be negative")
        if len(self.center) != 2 or len(self.bounds) != 4:
            raise ValueError("SearchCell center and bounds must be 2D")
        x_min, y_min, x_max, y_max = (float(value) for value in self.bounds)
        if x_max <= x_min or y_max <= y_min:
            raise ValueError("SearchCell bounds must have positive area")
        center = (float(self.center[0]), float(self.center[1]))
        if not (x_min <= center[0] <= x_max and y_min <= center[1] <= y_max):
            raise ValueError("SearchCell center must lie inside its bounds")
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "bounds", (x_min, y_min, x_max, y_max))
        object.__setattr__(
            self,
            "semantic_labels",
            tuple(str(label) for label in self.semantic_labels),
        )
        object.__setattr__(self, "metadata", dict(self.metadata))

    def contains(self, x: float, y: float) -> bool:
        """Return whether a world-coordinate point lies within the cell."""
        x_min, y_min, x_max, y_max = self.bounds
        return x_min <= x <= x_max and y_min <= y <= y_max

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SearchGrid:
    """Regular 2D grid covering a SearchArea's axis-aligned bounds."""

    area_id: str
    resolution_m: float
    origin: Point2D
    width: int
    height: int
    cells: Tuple[SearchCell, ...]
    source_geometry: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.area_id.strip():
            raise ValueError("SearchGrid.area_id must not be empty")
        if self.resolution_m <= 0:
            raise ValueError("SearchGrid.resolution_m must be positive")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("SearchGrid dimensions must be positive")
        cells = tuple(self.cells)
        if len(cells) != self.width * self.height:
            raise ValueError("SearchGrid must contain width * height cells")
        if len({cell.cell_id for cell in cells}) != len(cells):
            raise ValueError("SearchGrid cell IDs must be unique")
        object.__setattr__(self, "origin", (float(self.origin[0]), float(self.origin[1])))
        object.__setattr__(self, "cells", cells)
        object.__setattr__(self, "source_geometry", dict(self.source_geometry))

    @classmethod
    def from_task(
        cls,
        task: SearchTask,
        resolution_m: float,
        *,
        excluded_geometries: Iterable[Mapping[str, Any]] = (),
    ) -> "SearchGrid":
        return cls.from_area(
            task.search_area,
            resolution_m,
            excluded_geometries=excluded_geometries,
        )

    @classmethod
    def from_area(
        cls,
        area: SearchArea,
        resolution_m: float,
        *,
        excluded_geometries: Iterable[Mapping[str, Any]] = (),
    ) -> "SearchGrid":
        """Discretize an area using cell-center inclusion."""
        if resolution_m <= 0:
            raise ValueError("resolution_m must be positive")
        geometry = dict(area.geometry)
        bounds = _geometry_bounds(geometry)
        if bounds is None:
            raise ValueError("search area geometry is unsupported or empty")
        x_min, y_min, x_max, y_max = bounds
        width = max(1, int(math.ceil((x_max - x_min) / resolution_m)))
        height = max(1, int(math.ceil((y_max - y_min) / resolution_m)))
        exclusions = tuple(dict(item) for item in excluded_geometries)
        geometry_exclusions = geometry.get("excluded_geometries") or ()
        if isinstance(geometry_exclusions, Iterable) and not isinstance(
            geometry_exclusions, (str, bytes, Mapping)
        ):
            exclusions += tuple(
                dict(item) for item in geometry_exclusions if isinstance(item, Mapping)
            )

        cells = []
        for row in range(height):
            cell_y_min = y_min + row * resolution_m
            cell_y_max = min(y_max, cell_y_min + resolution_m)
            for column in range(width):
                cell_x_min = x_min + column * resolution_m
                cell_x_max = min(x_max, cell_x_min + resolution_m)
                center = (
                    (cell_x_min + cell_x_max) / 2.0,
                    (cell_y_min + cell_y_max) / 2.0,
                )
                within_area = point_in_search_geometry(center, geometry)
                excluded = any(
                    point_in_search_geometry(center, excluded_geometry)
                    for excluded_geometry in exclusions
                )
                cells.append(SearchCell(
                    cell_id=f"{area.area_id}:r{row}:c{column}",
                    row=row,
                    column=column,
                    center=center,
                    bounds=(cell_x_min, cell_y_min, cell_x_max, cell_y_max),
                    searchable=within_area and not excluded,
                    metadata={
                        "within_search_area": within_area,
                        "excluded": excluded,
                    },
                ))

        return cls(
            area_id=area.area_id,
            resolution_m=float(resolution_m),
            origin=(x_min, y_min),
            width=width,
            height=height,
            cells=tuple(cells),
            source_geometry=geometry,
        )

    @property
    def searchable_cells(self) -> Tuple[SearchCell, ...]:
        return tuple(cell for cell in self.cells if cell.searchable)

    @property
    def bounds(self) -> Bounds2D:
        return (
            self.origin[0],
            self.origin[1],
            max(cell.bounds[2] for cell in self.cells),
            max(cell.bounds[3] for cell in self.cells),
        )

    def cell(self, row: int, column: int) -> Optional[SearchCell]:
        if row < 0 or row >= self.height or column < 0 or column >= self.width:
            return None
        return self.cells[row * self.width + column]

    def cell_at(self, x: float, y: float) -> Optional[SearchCell]:
        """Return the grid cell containing a world-coordinate point."""
        x_min, y_min, x_max, y_max = self.bounds
        if x < x_min or y < y_min or x > x_max or y > y_max:
            return None
        column = min(self.width - 1, int((x - x_min) // self.resolution_m))
        row = min(self.height - 1, int((y - y_min) // self.resolution_m))
        return self.cell(row, column)

    def nearest_searchable_cell(self, x: float, y: float) -> Optional[SearchCell]:
        return min(
            self.searchable_cells,
            key=lambda cell: math.hypot(cell.center[0] - x, cell.center[1] - y),
            default=None,
        )

    def cells_within_radius(
        self,
        x: float,
        y: float,
        radius_m: float,
        *,
        searchable_only: bool = True,
    ) -> Tuple[SearchCell, ...]:
        if radius_m < 0:
            raise ValueError("radius_m must not be negative")
        candidates = self.searchable_cells if searchable_only else self.cells
        return tuple(
            cell
            for cell in candidates
            if math.hypot(cell.center[0] - x, cell.center[1] - y) <= radius_m + 1e-9
        )

    def uniform_belief(self) -> Dict[str, float]:
        """Return a normalized prior over searchable cells."""
        searchable = self.searchable_cells
        if not searchable:
            return {}
        probability = 1.0 / len(searchable)
        return {cell.cell_id: probability for cell in searchable}

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ViewpointCandidate:
    """A policy candidate and the grid cells observable from it."""

    candidate_id: str
    viewpoint: Viewpoint
    anchor_cell_id: str
    visible_cell_ids: Tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.candidate_id.strip() or not self.anchor_cell_id.strip():
            raise ValueError("candidate and anchor cell IDs must not be empty")
        object.__setattr__(
            self,
            "visible_cell_ids",
            tuple(dict.fromkeys(str(cell_id) for cell_id in self.visible_cell_ids)),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CandidateViewpointGenerator:
    """Generate nadir-camera candidates over searchable grid cells."""

    altitude_m: float = 30.0
    horizontal_fov_rad: float = math.pi / 2.0
    footprint_radius_m: Optional[float] = None
    stride_cells: int = 1
    yaw_rad: float = 0.0
    pitch_rad: float = -math.pi / 2.0
    max_candidates: Optional[int] = None

    def __post_init__(self) -> None:
        if self.altitude_m <= 0:
            raise ValueError("altitude_m must be positive")
        if not 0 < self.horizontal_fov_rad < math.pi:
            raise ValueError("horizontal_fov_rad must be within (0, pi)")
        if self.footprint_radius_m is not None and self.footprint_radius_m < 0:
            raise ValueError("footprint_radius_m must not be negative")
        if self.stride_cells <= 0:
            raise ValueError("stride_cells must be positive")
        if self.max_candidates is not None and self.max_candidates <= 0:
            raise ValueError("max_candidates must be positive")

    @property
    def resolved_footprint_radius_m(self) -> float:
        if self.footprint_radius_m is not None:
            return float(self.footprint_radius_m)
        return self.altitude_m * math.tan(self.horizontal_fov_rad / 2.0)

    def generate(self, grid: SearchGrid) -> Tuple[ViewpointCandidate, ...]:
        candidates = []
        for cell in grid.searchable_cells:
            if cell.row % self.stride_cells or cell.column % self.stride_cells:
                continue
            viewpoint = Viewpoint(
                x=cell.center[0],
                y=cell.center[1],
                z=float(self.altitude_m),
                yaw=float(self.yaw_rad),
                pitch=float(self.pitch_rad),
            )
            visible = grid.cells_within_radius(
                viewpoint.x,
                viewpoint.y,
                self.resolved_footprint_radius_m,
            )
            candidates.append(ViewpointCandidate(
                candidate_id=f"candidate:{cell.cell_id}",
                viewpoint=viewpoint,
                anchor_cell_id=cell.cell_id,
                visible_cell_ids=tuple(item.cell_id for item in visible),
            ))
            if self.max_candidates is not None and len(candidates) >= self.max_candidates:
                break
        return tuple(candidates)


def viewpoint_distance_matrix(
    candidates: Sequence[ViewpointCandidate],
) -> Tuple[Tuple[float, ...], ...]:
    """Return pairwise 3D Euclidean travel distances between candidates."""
    rows = []
    for first in candidates:
        rows.append(tuple(
            math.sqrt(
                (first.viewpoint.x - second.viewpoint.x) ** 2
                + (first.viewpoint.y - second.viewpoint.y) ** 2
                + (first.viewpoint.z - second.viewpoint.z) ** 2
            )
            for second in candidates
        ))
    return tuple(rows)


def _geometry_bounds(geometry: Mapping[str, Any]) -> Optional[Bounds2D]:
    kind = geometry.get("kind")
    if kind in ("rectangle", "area"):
        points = _points(geometry.get("coords"))
        bounds = _bounds_for_points(points)
        if bounds is None or bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
            return None
        return bounds
    if kind == "circle":
        center = geometry.get("center")
        radius = float(geometry.get("radius", 0.0) or 0.0)
        if not _is_point(center) or radius <= 0:
            return None
        return (
            float(center[0]) - radius,
            float(center[1]) - radius,
            float(center[0]) + radius,
            float(center[1]) + radius,
        )
    if kind in ("line", "point"):
        points = _points(geometry.get("coords"))
        radius = float(geometry.get("buffer", 20.0) or 20.0)
        bounds = _bounds_for_points(points)
        if bounds is None or radius <= 0:
            return None
        return (
            bounds[0] - radius,
            bounds[1] - radius,
            bounds[2] + radius,
            bounds[3] + radius,
        )
    return None


def point_in_search_geometry(point: Point2D, geometry: Mapping[str, Any]) -> bool:
    """Return whether a 2D point lies in a supported search geometry."""
    kind = geometry.get("kind")
    if kind in ("rectangle", "area"):
        return _point_in_polygon(point, _points(geometry.get("coords")))
    if kind == "circle":
        center = geometry.get("center")
        radius = float(geometry.get("radius", 0.0) or 0.0)
        return bool(
            _is_point(center)
            and math.hypot(point[0] - float(center[0]), point[1] - float(center[1]))
            <= radius + 1e-9
        )
    if kind == "line":
        line = _points(geometry.get("coords"))
        radius = float(geometry.get("buffer", 20.0) or 20.0)
        return _point_to_polyline_distance(point, line) <= radius + 1e-9
    if kind == "point":
        points = _points(geometry.get("coords"))
        radius = float(geometry.get("buffer", 20.0) or 20.0)
        return bool(
            points
            and math.hypot(point[0] - points[0][0], point[1] - points[0][1])
            <= radius + 1e-9
        )
    return False


def _point_in_polygon(point: Point2D, polygon: Sequence[Point2D]) -> bool:
    if len(polygon) < 3:
        return False
    x, y = point
    inside = False
    for index, first in enumerate(polygon):
        second = polygon[(index + 1) % len(polygon)]
        if _point_on_segment(point, first, second):
            return True
        x1, y1 = first
        x2, y2 = second
        if (y1 <= y < y2) or (y2 <= y < y1):
            intersection_x = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if intersection_x >= x:
                inside = not inside
    return inside


def _point_on_segment(point: Point2D, first: Point2D, second: Point2D) -> bool:
    ax, ay = first
    bx, by = second
    px, py = point
    cross = abs((bx - ax) * (py - ay) - (by - ay) * (px - ax))
    if cross > 1e-9:
        return False
    dot = (px - ax) * (bx - ax) + (py - ay) * (by - ay)
    length_squared = (bx - ax) ** 2 + (by - ay) ** 2
    return -1e-9 <= dot <= length_squared + 1e-9


def _point_to_polyline_distance(point: Point2D, line: Sequence[Point2D]) -> float:
    if not line:
        return math.inf
    if len(line) == 1:
        return math.hypot(point[0] - line[0][0], point[1] - line[0][1])
    minimum = math.inf
    for first, second in zip(line, line[1:]):
        dx = second[0] - first[0]
        dy = second[1] - first[1]
        length_squared = dx * dx + dy * dy
        if length_squared <= 1e-12:
            projection = first
        else:
            ratio = max(0.0, min(
                1.0,
                ((point[0] - first[0]) * dx + (point[1] - first[1]) * dy)
                / length_squared,
            ))
            projection = (first[0] + ratio * dx, first[1] + ratio * dy)
        minimum = min(
            minimum,
            math.hypot(point[0] - projection[0], point[1] - projection[1]),
        )
    return minimum


def _points(values: Any) -> Tuple[Point2D, ...]:
    if not isinstance(values, Iterable) or isinstance(values, (str, bytes, Mapping)):
        return ()
    return tuple(
        (float(value[0]), float(value[1]))
        for value in values
        if _is_point(value)
    )


def _is_point(value: Any) -> bool:
    return (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and len(value) >= 2
        and isinstance(value[0], (int, float))
        and isinstance(value[1], (int, float))
    )


def _bounds_for_points(points: Sequence[Point2D]) -> Optional[Bounds2D]:
    if not points:
        return None
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return min(xs), min(ys), max(xs), max(ys)
