"""Platform-neutral sensor snapshot produced at one commanded viewpoint."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Mapping, Tuple

from ..contracts import TargetDetection, Viewpoint


Point2D = Tuple[float, float]


@dataclass(frozen=True)
class SearchSensorFrame:
    """Synchronized, post-perception sensor data for one search observation.

    Raw RGB, depth, and point-cloud payloads remain in the robotics middleware.
    This contract carries their references and structured results into the
    platform-neutral search layer.
    """

    timestamp_s: float
    viewpoint: Viewpoint
    frame_id: str = "map"
    detections: Tuple[TargetDetection, ...] = ()
    visible_cell_ids: Tuple[str, ...] = ()
    visible_ground_points_xy: Tuple[Point2D, ...] = ()
    observation_quality: float = 1.0
    travel_time_s: float = 0.0
    travel_distance_m: float = 0.0
    energy_used: float = 0.0
    sensor_timestamps_s: Mapping[str, float] = field(default_factory=dict)
    frame_references: Mapping[str, str] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not math.isfinite(self.timestamp_s) or self.timestamp_s < 0:
            raise ValueError("timestamp_s must be finite and non-negative")
        if not self.frame_id.strip():
            raise ValueError("frame_id must not be empty")
        if not 0 <= self.observation_quality <= 1:
            raise ValueError("observation_quality must be within [0, 1]")
        for name in ("travel_time_s", "travel_distance_m", "energy_used"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        points = tuple(
            (float(point[0]), float(point[1]))
            for point in self.visible_ground_points_xy
        )
        if any(
            not math.isfinite(coordinate)
            for point in points
            for coordinate in point
        ):
            raise ValueError("visible ground points must be finite")
        timestamps = {
            str(name): float(value)
            for name, value in self.sensor_timestamps_s.items()
        }
        if any(not math.isfinite(value) or value < 0 for value in timestamps.values()):
            raise ValueError("sensor timestamps must be finite and non-negative")
        object.__setattr__(self, "detections", tuple(self.detections))
        object.__setattr__(
            self,
            "visible_cell_ids",
            tuple(dict.fromkeys(str(item) for item in self.visible_cell_ids)),
        )
        object.__setattr__(self, "visible_ground_points_xy", points)
        object.__setattr__(self, "sensor_timestamps_s", timestamps)
        object.__setattr__(
            self,
            "frame_references",
            {str(name): str(value) for name, value in self.frame_references.items()},
        )
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def maximum_sensor_skew_s(self) -> float:
        if not self.sensor_timestamps_s:
            return 0.0
        values = tuple(self.sensor_timestamps_s.values())
        return max(values) - min(values)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
