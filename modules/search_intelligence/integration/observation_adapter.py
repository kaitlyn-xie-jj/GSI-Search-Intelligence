"""Convert synchronized robotics sensor frames into search observations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

from ..contracts import SearchObservation, Viewpoint
from ..search_space import SearchGrid
from .sensor_frame import SearchSensorFrame


@dataclass(frozen=True)
class SearchObservationAdapter:
    """Project processed sensor coverage and actual pose onto a SearchGrid."""

    grid: SearchGrid
    target_query: Optional[str] = None
    fallback_footprint_radius_m: Optional[float] = None
    maximum_sensor_skew_s: Optional[float] = 0.25

    def __post_init__(self) -> None:
        if (
            self.fallback_footprint_radius_m is not None
            and self.fallback_footprint_radius_m < 0
        ):
            raise ValueError("fallback_footprint_radius_m must not be negative")
        if self.maximum_sensor_skew_s is not None and self.maximum_sensor_skew_s <= 0:
            raise ValueError("maximum_sensor_skew_s must be positive")

    def adapt(
        self,
        frame: SearchSensorFrame,
        commanded_viewpoint: Viewpoint,
    ) -> SearchObservation:
        visible_cell_ids = self._visible_cell_ids(frame)
        quality = frame.observation_quality
        if self.maximum_sensor_skew_s is not None:
            skew_ratio = min(
                1.0,
                frame.maximum_sensor_skew_s / self.maximum_sensor_skew_s,
            )
            quality *= 1.0 - skew_ratio
        position_error = math.sqrt(
            (frame.viewpoint.x - commanded_viewpoint.x) ** 2
            + (frame.viewpoint.y - commanded_viewpoint.y) ** 2
            + (frame.viewpoint.z - commanded_viewpoint.z) ** 2
        )
        yaw_error = _wrapped_angle_difference(
            frame.viewpoint.yaw,
            commanded_viewpoint.yaw,
        )
        metadata = {
            **dict(frame.metadata),
            "source": "robotics_sensor_frame",
            "frame_id": frame.frame_id,
            "commanded_viewpoint_key": commanded_viewpoint.key,
            "position_error_m": position_error,
            "yaw_error_rad": yaw_error,
            "maximum_sensor_skew_s": frame.maximum_sensor_skew_s,
            "sensor_timestamps_s": dict(frame.sensor_timestamps_s),
            "frame_references": dict(frame.frame_references),
        }
        return SearchObservation(
            viewpoint=frame.viewpoint,
            timestamp_s=frame.timestamp_s,
            action_viewpoint_key=commanded_viewpoint.key,
            detections=tuple(
                detection for detection in frame.detections
                if self.target_query is None
                or _normalize_label(detection.label)
                == _normalize_label(self.target_query)
            ),
            visible_cell_ids=visible_cell_ids,
            observation_quality=quality,
            travel_time_s=frame.travel_time_s,
            travel_distance_m=frame.travel_distance_m,
            energy_used=frame.energy_used,
            sensor_metadata=metadata,
        )

    def _visible_cell_ids(self, frame: SearchSensorFrame) -> Tuple[str, ...]:
        searchable_ids = {cell.cell_id for cell in self.grid.searchable_cells}
        visible = [
            cell_id for cell_id in frame.visible_cell_ids
            if cell_id in searchable_ids
        ]
        for point in frame.visible_ground_points_xy:
            cell = self.grid.cell_at(point[0], point[1])
            if cell is not None and cell.searchable:
                visible.append(cell.cell_id)
        if not visible and self.fallback_footprint_radius_m is not None:
            visible.extend(
                cell.cell_id
                for cell in self.grid.cells_within_radius(
                    frame.viewpoint.x,
                    frame.viewpoint.y,
                    self.fallback_footprint_radius_m,
                )
            )
        return tuple(dict.fromkeys(visible))


def _wrapped_angle_difference(first: float, second: float) -> float:
    return math.atan2(math.sin(first - second), math.cos(first - second))


def _normalize_label(value: str) -> str:
    return "".join(character for character in value.lower() if character.isalnum())
