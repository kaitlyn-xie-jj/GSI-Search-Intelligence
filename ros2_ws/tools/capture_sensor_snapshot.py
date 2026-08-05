#!/usr/bin/env python3
"""Persist one exact bridged depth image and point-cloud message."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, PointCloud2


def _header_metadata(message: Image | PointCloud2) -> dict[str, object]:
    stamp = message.header.stamp
    return {
        "frame_id": message.header.frame_id,
        "stamp_sec": stamp.sec,
        "stamp_nanosec": stamp.nanosec,
    }


class SensorSnapshot(Node):
    def __init__(self, output: Path) -> None:
        super().__init__("gsi_yungu_sensor_snapshot")
        self._output = output
        self._depth_saved = False
        self._points_saved = False
        self.create_subscription(
            Image, "/oakd1/depth/image", self._save_depth, qos_profile_sensor_data
        )
        self.create_subscription(
            PointCloud2,
            "/oakd1/depth/points",
            self._save_points,
            qos_profile_sensor_data,
        )

    @property
    def complete(self) -> bool:
        return self._depth_saved and self._points_saved

    def _save_depth(self, message: Image) -> None:
        if self._depth_saved:
            return
        (self._output / "depth_image.raw").write_bytes(bytes(message.data))
        metadata = _header_metadata(message)
        metadata.update({
            "width": message.width,
            "height": message.height,
            "encoding": message.encoding,
            "is_bigendian": message.is_bigendian,
            "step": message.step,
            "byte_length": len(message.data),
        })
        (self._output / "depth_image.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self._depth_saved = True

    def _save_points(self, message: PointCloud2) -> None:
        if self._points_saved:
            return
        (self._output / "point_cloud.bin").write_bytes(bytes(message.data))
        metadata = _header_metadata(message)
        metadata.update({
            "width": message.width,
            "height": message.height,
            "is_bigendian": message.is_bigendian,
            "point_step": message.point_step,
            "row_step": message.row_step,
            "is_dense": message.is_dense,
            "byte_length": len(message.data),
            "fields": [
                {
                    "name": field.name,
                    "offset": field.offset,
                    "datatype": field.datatype,
                    "count": field.count,
                }
                for field in message.fields
            ],
        })
        (self._output / "point_cloud.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self._points_saved = True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--timeout-s", type=float, default=15.0)
    args = parser.parse_args()

    if args.timeout_s <= 0:
        parser.error("--timeout-s must be positive")
    args.output.mkdir(parents=True, exist_ok=True)

    rclpy.init()
    node = SensorSnapshot(args.output)
    deadline = time.monotonic() + args.timeout_s
    try:
        while not node.complete and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.25)
    finally:
        complete = node.complete
        node.destroy_node()
        rclpy.shutdown()
    if not complete:
        raise SystemExit("timed out before receiving both depth and point-cloud messages")


if __name__ == "__main__":
    main()
