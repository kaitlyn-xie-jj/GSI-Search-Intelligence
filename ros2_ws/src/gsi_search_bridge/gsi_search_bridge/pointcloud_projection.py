"""Project an organized depth point cloud into map-frame ground observations."""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


Point3D = Tuple[float, float, float]
Point2D = Tuple[float, float]


@dataclass(frozen=True)
class PointCloudGroundProjector:
    """Apply a fixed camera extrinsic and a live body pose to depth points."""

    camera_translation: Point3D = (0.0, 0.0, 0.0)
    camera_rpy: Point3D = (0.0, 0.0, 0.0)
    ground_plane_z_m: float = 0.0
    ground_tolerance_m: float = 0.3
    point_resolution_m: float = 0.5
    sample_limit: int = 8000
    maximum_range_m: Optional[float] = None
    expected_frame_id: Optional[str] = None

    def __post_init__(self) -> None:
        if self.ground_tolerance_m <= 0:
            raise ValueError("ground_tolerance_m must be positive")
        if self.point_resolution_m <= 0:
            raise ValueError("point_resolution_m must be positive")
        if self.sample_limit <= 0:
            raise ValueError("sample_limit must be positive")
        if self.maximum_range_m is not None and self.maximum_range_m <= 0:
            raise ValueError("maximum_range_m must be positive")
        values = (*self.camera_translation, *self.camera_rpy, self.ground_plane_z_m)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("projector transform and ground plane must be finite")

    def project(self, cloud: object, body_pose: object) -> Tuple[Point2D, ...]:
        """Return deduplicated XY points whose transformed Z lies on the ground."""

        frame_id = str(getattr(getattr(cloud, "header", None), "frame_id", ""))
        if self.expected_frame_id and frame_id != self.expected_frame_id:
            raise ValueError(
                f"point cloud frame '{frame_id}' does not match "
                f"'{self.expected_frame_id}'"
            )
        fields = _xyz_fields(cloud)
        width = int(cloud.width)
        height = int(cloud.height)
        total = width * height
        if width <= 0 or height <= 0 or total <= 0:
            return ()
        sample_stride = max(1, math.ceil(total / self.sample_limit))
        camera_rotation = _rotation_matrix_from_rpy(*self.camera_rpy)
        visible: Dict[Tuple[int, int], Point2D] = {}

        for flat_index in range(0, total, sample_stride):
            row, column = divmod(flat_index, width)
            point_offset = row * int(cloud.row_step) + column * int(cloud.point_step)
            point_camera = tuple(
                _read_field(cloud, point_offset, fields[name])
                for name in ("x", "y", "z")
            )
            if not all(math.isfinite(value) for value in point_camera):
                continue
            if self.maximum_range_m is not None and math.sqrt(
                sum(value * value for value in point_camera)
            ) > self.maximum_range_m:
                continue

            point_map = self.transform_point(
                point_camera,
                body_pose,
                camera_rotation=camera_rotation,
            )
            if abs(point_map[2] - self.ground_plane_z_m) > self.ground_tolerance_m:
                continue
            key = (
                round(point_map[0] / self.point_resolution_m),
                round(point_map[1] / self.point_resolution_m),
            )
            visible.setdefault(key, (point_map[0], point_map[1]))
        return tuple(visible.values())

    def transform_point(
        self,
        point_camera: Point3D,
        body_pose: object,
        *,
        camera_rotation: Optional[object] = None,
    ) -> Point3D:
        """Transform one Gazebo camera-frame point into the MAVROS map frame."""

        rotation = camera_rotation or _rotation_matrix_from_rpy(*self.camera_rpy)
        point_body = _add(
            _matrix_vector(rotation, point_camera),
            self.camera_translation,
        )
        return _add(
            _rotate_by_quaternion(point_body, body_pose.orientation),
            (
                float(body_pose.position.x),
                float(body_pose.position.y),
                float(body_pose.position.z),
            ),
        )


@dataclass(frozen=True)
class _FieldReader:
    offset: int
    format_character: str


def _xyz_fields(cloud: object) -> Dict[str, _FieldReader]:
    formats = {7: "f", 8: "d"}  # sensor_msgs/PointField FLOAT32 and FLOAT64
    result = {}
    for field in cloud.fields:
        if field.name in {"x", "y", "z"} and field.datatype in formats:
            result[field.name] = _FieldReader(
                int(field.offset),
                formats[int(field.datatype)],
            )
    missing = {"x", "y", "z"} - set(result)
    if missing:
        raise ValueError(f"point cloud is missing floating fields: {sorted(missing)}")
    return result


def _read_field(cloud: object, point_offset: int, field: _FieldReader) -> float:
    endian = ">" if bool(cloud.is_bigendian) else "<"
    return float(struct.unpack_from(
        endian + field.format_character,
        cloud.data,
        point_offset + field.offset,
    )[0])


def _rotation_matrix_from_rpy(roll: float, pitch: float, yaw: float):
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return (
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp, cp * sr, cp * cr),
    )


def _matrix_vector(matrix: object, point: Point3D) -> Point3D:
    return tuple(
        sum(float(matrix[row][column]) * point[column] for column in range(3))
        for row in range(3)
    )


def _rotate_by_quaternion(point: Point3D, quaternion: object) -> Point3D:
    x, y, z, w = (
        float(quaternion.x),
        float(quaternion.y),
        float(quaternion.z),
        float(quaternion.w),
    )
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 1e-9:
        raise ValueError("body pose quaternion must be non-zero")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    tx = 2.0 * (y * point[2] - z * point[1])
    ty = 2.0 * (z * point[0] - x * point[2])
    tz = 2.0 * (x * point[1] - y * point[0])
    return (
        point[0] + w * tx + y * tz - z * ty,
        point[1] + w * ty + z * tx - x * tz,
        point[2] + w * tz + x * ty - y * tx,
    )


def _add(first: Point3D, second: Point3D) -> Point3D:
    return tuple(first[index] + second[index] for index in range(3))
