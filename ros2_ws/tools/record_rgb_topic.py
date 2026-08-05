#!/usr/bin/env python3
"""Stream a ROS 2 RGB image topic as tightly packed RGB24 frames on stdout."""

from __future__ import annotations

import argparse
import sys
import time

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


class RgbTopicRecorder(Node):
    def __init__(self, topic: str, width: int, height: int, fps: float) -> None:
        super().__init__("gsi_rgb_topic_recorder")
        self._width = width
        self._height = height
        self._period = 1.0 / fps
        self._next_frame_at = 0.0
        self._frames = 0
        self._subscription = self.create_subscription(
            Image,
            topic,
            self._on_image,
            qos_profile_sensor_data,
        )

    def _on_image(self, message: Image) -> None:
        now = time.monotonic()
        if now < self._next_frame_at:
            return
        self._next_frame_at = now + self._period

        if (message.width, message.height, message.encoding.lower()) != (
            self._width,
            self._height,
            "rgb8",
        ):
            raise RuntimeError(
                "expected "
                f"{self._width}x{self._height} rgb8, received "
                f"{message.width}x{message.height} {message.encoding}"
            )

        packed_step = self._width * 3
        data = memoryview(message.data)
        if message.step == packed_step:
            sys.stdout.buffer.write(data)
        else:
            for row in range(self._height):
                start = row * message.step
                sys.stdout.buffer.write(data[start : start + packed_step])
        sys.stdout.buffer.flush()
        self._frames += 1

    @property
    def frame_count(self) -> int:
        return self._frames


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", default="/oakd1/rgb/image")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=float, default=10.0)
    args = parser.parse_args()

    rclpy.init()
    node = RgbTopicRecorder(args.topic, args.width, args.height, args.fps)
    try:
        rclpy.spin(node)
    except (BrokenPipeError, KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        print(f"recorded_frames={node.frame_count}", file=sys.stderr)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
