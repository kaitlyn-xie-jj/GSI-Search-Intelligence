"""Deterministic zigzag coverage baseline."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from ..contracts import SearchState, Viewpoint
from ..coverage_waypoints import plan_search_waypoints
from ..search_space import SearchGrid
from .base import SearchPolicy


@dataclass(frozen=True)
class CoveragePolicy(SearchPolicy):
    """Wrap the existing GSI lawnmower planner as a SearchPolicy baseline."""

    pass_spacing_m: float = 40.0
    altitude_m: Optional[float] = None
    camera_pitch_rad: float = -math.pi / 2.0
    max_points: int = 3000
    observation_spacing_m: Optional[float] = None
    start_from_nearest_endpoint: bool = True
    route_start_hint: Optional[Viewpoint] = None
    viewpoint_filter: Optional[Callable[[Viewpoint], bool]] = None
    search_grid: Optional[SearchGrid] = None
    recovery_enabled: bool = False
    recovery_min_quality: float = 0.5
    recovery_offset_m: Optional[float] = None

    def __post_init__(self) -> None:
        if self.pass_spacing_m <= 0:
            raise ValueError("pass_spacing_m must be positive")
        if self.altitude_m is not None and self.altitude_m < 0:
            raise ValueError("altitude_m must not be negative")
        if self.max_points <= 0:
            raise ValueError("max_points must be positive")
        if self.observation_spacing_m is not None and self.observation_spacing_m <= 0:
            raise ValueError("observation_spacing_m must be positive")
        if not 0.0 <= self.recovery_min_quality <= 1.0:
            raise ValueError("recovery_min_quality must be within [0, 1]")
        if self.recovery_offset_m is not None and self.recovery_offset_m <= 0:
            raise ValueError("recovery_offset_m must be positive")
        if self.recovery_enabled and self.search_grid is None:
            raise ValueError("recovery_enabled requires search_grid")

    def plan(self, state: SearchState) -> Tuple[Viewpoint, ...]:
        """Generate the unvisited portion of the deterministic coverage route."""
        viewpoints = self._primary_viewpoints(state)
        visited = set(state.visited_viewpoint_keys)
        remaining_primary = tuple(
            viewpoint for viewpoint in viewpoints if viewpoint.key not in visited
        )
        if remaining_primary:
            remaining = remaining_primary
        elif self.recovery_enabled:
            remaining = self._recovery_viewpoints(state, visited)
        else:
            remaining = ()
        viewpoint_limit = state.task.budget.max_viewpoints
        if viewpoint_limit is not None:
            available = max(0, viewpoint_limit - state.step_index)
            remaining = remaining[:available]
        return remaining

    def decision_metadata(
        self,
        state: SearchState,
        viewpoint: Viewpoint,
    ) -> Mapping[str, Any]:
        metadata: Dict[str, Any] = {
            "policy_name": type(self).__name__,
            "step_index": state.step_index,
            "selected_viewpoint_key": viewpoint.key,
            "coverage_phase": (
                "recovery"
                if self.recovery_enabled and not self._primary_remaining(state)
                else "primary"
            ),
        }
        metadata.update(self.coverage_status_counts(state))
        return metadata

    def coverage_status_counts(self, state: SearchState) -> Dict[str, int]:
        if self.search_grid is None:
            return {}
        covered = sum(
            state.observed_cell_quality.get(cell.cell_id, 0.0)
            >= self.recovery_min_quality
            for cell in self.search_grid.searchable_cells
        )
        searchable = len(self.search_grid.searchable_cells)
        primary_complete = not self._primary_remaining(state)
        return {
            "coverage_searchable_cells": searchable,
            "coverage_covered_cells": covered,
            "coverage_unseen_cells": 0 if primary_complete else searchable - covered,
            "coverage_deferred_cells": searchable - covered if primary_complete else 0,
            "coverage_excluded_cells": len(self.search_grid.cells) - searchable,
        }

    def _primary_remaining(self, state: SearchState) -> Tuple[Viewpoint, ...]:
        primary = self._primary_viewpoints(state)
        visited = set(state.visited_viewpoint_keys)
        return tuple(item for item in primary if item.key not in visited)

    def _primary_viewpoints(self, state: SearchState) -> Tuple[Viewpoint, ...]:
        geometry = dict(state.task.search_area.geometry)
        points = plan_search_waypoints(
            geometry,
            pass_spacing=self.pass_spacing_m,
            zigzag=True,
            max_points=self.max_points,
        )
        points = self._deduplicate_consecutive(points)
        if self.observation_spacing_m is not None:
            points = self._sample_segments(points, self.observation_spacing_m)
        altitude = self._resolve_altitude(state, geometry)
        viewpoints = self._to_viewpoints(points, altitude)
        if self.viewpoint_filter is not None:
            viewpoints = tuple(item for item in viewpoints if self.viewpoint_filter(item))
            viewpoints = self._to_viewpoints(
                tuple((item.x, item.y) for item in viewpoints), altitude
            )
        if (
            self.start_from_nearest_endpoint
            and self.route_start_hint is not None
            and len(viewpoints) > 1
            and self._distance(self.route_start_hint, viewpoints[-1])
            < self._distance(self.route_start_hint, viewpoints[0])
        ):
            viewpoints = self._reverse_with_headings(viewpoints)
        return viewpoints

    def _recovery_viewpoints(
        self,
        state: SearchState,
        visited: set[str],
    ) -> Tuple[Viewpoint, ...]:
        assert self.search_grid is not None
        altitude = self._resolve_altitude(state, dict(state.task.search_area.geometry))
        offset = self.recovery_offset_m or self.pass_spacing_m / 2.0
        uncovered = tuple(
            cell for cell in self.search_grid.searchable_cells
            if state.observed_cell_quality.get(cell.cell_id, 0.0)
            < self.recovery_min_quality
        )
        # Stable lawn-mower order keeps recovery auditable while the alternate
        # offsets let an occluded cell be revisited from several sides.
        ordered = sorted(
            uncovered,
            key=lambda cell: (
                cell.row,
                cell.column if cell.row % 2 == 0 else -cell.column,
            ),
        )
        points: List[Tuple[float, float]] = []
        for cell in ordered:
            x, y = cell.center
            points.extend(((x, y), (x - offset, y), (x + offset, y), (x, y - offset), (x, y + offset)))
        candidates = self._to_viewpoints(tuple(dict.fromkeys(points)), altitude)
        if self.viewpoint_filter is not None:
            candidates = tuple(item for item in candidates if self.viewpoint_filter(item))
        candidates = tuple(item for item in candidates if item.key not in visited)
        if state.current_viewpoint is not None and candidates:
            first = min(
                range(len(candidates)),
                key=lambda index: self._distance(state.current_viewpoint, candidates[index]),
            )
            candidates = candidates[first:] + candidates[:first]
        return candidates

    @staticmethod
    def _distance(first: Viewpoint, second: Viewpoint) -> float:
        return math.hypot(first.x - second.x, first.y - second.y)

    def _reverse_with_headings(
        self, viewpoints: Sequence[Viewpoint]
    ) -> Tuple[Viewpoint, ...]:
        return self._to_viewpoints(
            tuple((item.x, item.y) for item in reversed(viewpoints)),
            float(viewpoints[0].z),
        )

    def _resolve_altitude(self, state: SearchState, geometry: dict) -> float:
        if self.altitude_m is not None:
            return float(self.altitude_m)
        if state.current_viewpoint is not None:
            return float(state.current_viewpoint.z)
        for key in ("flight_altitude", "altitude", "altitude_m"):
            if geometry.get(key) is not None:
                return float(geometry[key])
        return 30.0

    def _to_viewpoints(
        self, xy_points: Sequence[Sequence[float]], altitude: float
    ) -> Tuple[Viewpoint, ...]:
        if not xy_points:
            return ()

        headings: List[float] = []
        for index, point in enumerate(xy_points):
            if index + 1 < len(xy_points):
                next_point = xy_points[index + 1]
                headings.append(
                    math.atan2(float(next_point[1]) - float(point[1]),
                               float(next_point[0]) - float(point[0]))
                )
            elif headings:
                headings.append(headings[-1])
            else:
                headings.append(0.0)

        return tuple(
            Viewpoint(
                x=float(point[0]),
                y=float(point[1]),
                z=altitude,
                yaw=headings[index],
                pitch=self.camera_pitch_rad,
            )
            for index, point in enumerate(xy_points)
        )

    @staticmethod
    def _deduplicate_consecutive(
        xy_points: Sequence[Sequence[float]],
    ) -> Tuple[Tuple[float, float], ...]:
        result: List[Tuple[float, float]] = []
        for point in xy_points:
            if len(point) < 2:
                continue
            normalized = (float(point[0]), float(point[1]))
            if not result or normalized != result[-1]:
                result.append(normalized)
        return tuple(result)

    @staticmethod
    def _sample_segments(
        xy_points: Sequence[Sequence[float]],
        spacing_m: float,
    ) -> Tuple[Tuple[float, float], ...]:
        """Insert observation poses so no route segment exceeds the spacing."""
        if not xy_points:
            return ()
        sampled = [(float(xy_points[0][0]), float(xy_points[0][1]))]
        for endpoint in xy_points[1:]:
            start_x, start_y = sampled[-1]
            end_x, end_y = float(endpoint[0]), float(endpoint[1])
            distance = math.hypot(end_x - start_x, end_y - start_y)
            segment_count = max(1, int(math.ceil(distance / spacing_m)))
            sampled.extend(
                (
                    start_x + (end_x - start_x) * index / segment_count,
                    start_y + (end_y - start_y) * index / segment_count,
                )
                for index in range(1, segment_count + 1)
            )
        return tuple(sampled)
