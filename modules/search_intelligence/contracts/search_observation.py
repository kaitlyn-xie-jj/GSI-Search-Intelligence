"""Observation contract returned after one search viewpoint."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple


@dataclass(frozen=True)
class Viewpoint:
    """Platform-neutral UAV camera pose."""

    x: float
    y: float
    z: float
    yaw: float
    pitch: float = 0.0

    @property
    def key(self) -> str:
        """Stable key suitable for visited-viewpoint tracking."""
        return f"{self.x:.3f}:{self.y:.3f}:{self.z:.3f}:{self.yaw:.3f}:{self.pitch:.3f}"


@dataclass(frozen=True)
class TargetDetection:
    """A detector result; entity identity is optional and observation-derived."""

    label: str
    confidence: float
    attributes: Mapping[str, Any] = field(default_factory=dict)
    bbox_xyxy: Optional[Tuple[float, float, float, float]] = None
    estimated_position: Optional[Tuple[float, float, float]] = None
    entity_id: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.label.strip():
            raise ValueError("TargetDetection.label must not be empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("TargetDetection.confidence must be within [0, 1]")
        if self.bbox_xyxy is not None and len(self.bbox_xyxy) != 4:
            raise ValueError("bbox_xyxy must contain four values")
        if self.estimated_position is not None and len(self.estimated_position) != 3:
            raise ValueError("estimated_position must contain three values")
        object.__setattr__(self, "attributes", dict(self.attributes))


@dataclass(frozen=True)
class SearchObservation:
    """Sensor evidence and execution cost produced by one viewpoint action."""

    viewpoint: Viewpoint
    timestamp_s: float
    detections: Tuple[TargetDetection, ...] = ()
    visible_cell_ids: Tuple[str, ...] = ()
    observed_area: Mapping[str, Any] = field(default_factory=dict)
    observation_quality: float = 1.0
    visibility_probability: float = 1.0
    negative_update_strength: float = 1.0
    negative_update_rejection_reason: Optional[str] = None
    travel_time_s: float = 0.0
    travel_distance_m: float = 0.0
    energy_used: float = 0.0
    sensor_metadata: Mapping[str, Any] = field(default_factory=dict)
    action_viewpoint_key: Optional[str] = None

    def __post_init__(self) -> None:
        if self.timestamp_s < 0:
            raise ValueError("timestamp_s must not be negative")
        if self.action_viewpoint_key is not None and not self.action_viewpoint_key.strip():
            raise ValueError("action_viewpoint_key must not be empty")
        for name in (
            "observation_quality",
            "visibility_probability",
            "negative_update_strength",
        ):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be within [0, 1]")
        if (
            self.negative_update_rejection_reason is not None
            and not self.negative_update_rejection_reason.strip()
        ):
            raise ValueError("negative_update_rejection_reason must not be empty")
        for name in ("travel_time_s", "travel_distance_m", "energy_used"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must not be negative")
        object.__setattr__(self, "detections", tuple(self.detections))
        object.__setattr__(self, "visible_cell_ids", tuple(str(v) for v in self.visible_cell_ids))
        object.__setattr__(self, "observed_area", dict(self.observed_area))
        object.__setattr__(self, "sensor_metadata", dict(self.sensor_metadata))

    def matching_detections(self, min_confidence: float) -> Tuple[TargetDetection, ...]:
        """Return detections that satisfy a caller-provided confidence threshold."""
        return tuple(d for d in self.detections if d.confidence >= min_confidence)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)
