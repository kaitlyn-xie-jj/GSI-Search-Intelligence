#!/usr/bin/env python3
"""Print raw and map-frame statistics for one VisionFlow OakD point cloud."""

import math
import struct

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2


class PointCloudProbe(Node):
    def __init__(self):
        super().__init__("gsi_pointcloud_probe")
        self.odom = None
        self.create_subscription(
            Odometry,
            "/mavros/local_position/odom",
            self._on_odom,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            PointCloud2,
            "/oakd1/depth/points",
            self._on_cloud,
            qos_profile_sensor_data,
        )

    def _on_odom(self, message):
        self.odom = message

    def _on_cloud(self, message):
        if self.odom is None:
            return
        offsets = {field.name: field.offset for field in message.fields}
        if not {"x", "y", "z"}.issubset(offsets):
            raise RuntimeError(f"Point cloud fields are {tuple(offsets)}")
        prefix = ">" if message.is_bigendian else "<"
        count = message.width * message.height
        stride = max(1, count // 5000)
        raw = []
        map_direct = []
        map_optical = []
        for index in range(0, count, stride):
            base = index * message.point_step
            point = tuple(
                struct.unpack_from(prefix + "f", message.data, base + offsets[name])[0]
                for name in ("x", "y", "z")
            )
            if not all(math.isfinite(value) for value in point):
                continue
            raw.append(point)
            map_direct.append(self._to_map(point))
            map_optical.append(self._to_map((point[2], -point[0], -point[1])))

        print(f"frame={message.header.frame_id} sampled={len(raw)}")
        _print_stats("raw", raw)
        _print_stats("map_direct", map_direct)
        _print_stats("map_optical", map_optical)
        print(
            "ground counts: "
            f"direct={sum(-0.25 <= p[2] <= 0.35 for p in map_direct)}, "
            f"optical={sum(-0.25 <= p[2] <= 0.35 for p in map_optical)}"
        )
        rclpy.shutdown()

    def _to_map(self, camera_point):
        pitch = math.pi / 4.0
        cp, sp = math.cos(pitch), math.sin(pitch)
        sensor_offset = (0.01233, -0.03, 0.01878)
        rotated_offset = (
            cp * sensor_offset[0] + sp * sensor_offset[2],
            sensor_offset[1],
            -sp * sensor_offset[0] + cp * sensor_offset[2],
        )
        camera_translation = (
            0.10 + rotated_offset[0],
            0.028 + rotated_offset[1],
            0.06 + rotated_offset[2],
        )
        point_base = (
            cp * camera_point[0] + sp * camera_point[2] + camera_translation[0],
            camera_point[1] + camera_translation[1],
            -sp * camera_point[0] + cp * camera_point[2] + camera_translation[2],
        )
        pose = self.odom.pose.pose
        rotated = _rotate_by_quaternion(point_base, pose.orientation)
        return (
            rotated[0] + pose.position.x,
            rotated[1] + pose.position.y,
            rotated[2] + pose.position.z,
        )


def _rotate_by_quaternion(point, quaternion):
    x, y, z, w = quaternion.x, quaternion.y, quaternion.z, quaternion.w
    tx = 2.0 * (y * point[2] - z * point[1])
    ty = 2.0 * (z * point[0] - x * point[2])
    tz = 2.0 * (x * point[1] - y * point[0])
    return (
        point[0] + w * tx + y * tz - z * ty,
        point[1] + w * ty + z * tx - x * tz,
        point[2] + w * tz + x * ty - y * tx,
    )


def _print_stats(label, points):
    if not points:
        print(f"{label}: no finite points")
        return
    axes = list(zip(*points))
    print(
        f"{label}: "
        + ", ".join(
            f"{name}=[{min(values):.2f}, {max(values):.2f}]"
            for name, values in zip("xyz", axes)
        )
    )


def main():
    rclpy.init()
    node = PointCloudProbe()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
