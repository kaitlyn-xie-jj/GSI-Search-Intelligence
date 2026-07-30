"""ROS 2 yellow-target detector baseline with depth localization."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose

from .color_detection import (
    YellowThresholds,
    camera_point_from_pixel,
    find_yellow_region,
    median_depth,
    remap_pixel,
)
from .pointcloud_projection import PointCloudGroundProjector


class ColorTargetDetector(Node):
    """Publish a standard detection stream for the yellow-van simulation fixture."""

    def __init__(self) -> None:
        super().__init__("gsi_color_target_detector")
        defaults = {
            "rgb_topic": "/oakd1/rgb/image",
            "rgb_camera_info_topic": "/oakd1/rgb/camera_info",
            "depth_topic": "/oakd1/depth/image",
            "depth_camera_info_topic": "/oakd1/depth/camera_info",
            "odom_topic": "/mavros/local_position/odom",
            "detections_topic": "/gsi/detections",
            "target_label": "yellow-van",
            "target_entity_id": "yellow-search-van",
            "detection_rate_hz": 5.0,
            "sensor_timeout_s": 0.5,
            "minimum_red": 150,
            "minimum_green": 120,
            "maximum_blue": 120,
            "minimum_yellow_margin": 60,
            "maximum_red_green_difference": 100,
            "minimum_pixels": 400,
            "depth_window_radius_px": 6,
            "minimum_depth_m": 0.2,
            "maximum_depth_m": 19.1,
            "camera_translation_x_m": 0.121998,
            "camera_translation_y_m": -0.002,
            "camera_translation_z_m": 0.064561,
            "camera_roll_rad": 0.0,
            "camera_pitch_rad": 0.785398,
            "camera_yaw_rad": 0.0,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

        self._latest: Dict[str, Tuple[object, float]] = {}
        self._thresholds = YellowThresholds(
            minimum_red=int(self._parameter("minimum_red")),
            minimum_green=int(self._parameter("minimum_green")),
            maximum_blue=int(self._parameter("maximum_blue")),
            minimum_yellow_margin=int(self._parameter("minimum_yellow_margin")),
            maximum_red_green_difference=int(
                self._parameter("maximum_red_green_difference")
            ),
            minimum_pixels=int(self._parameter("minimum_pixels")),
        )
        self._projector = PointCloudGroundProjector(
            camera_translation=(
                float(self._parameter("camera_translation_x_m")),
                float(self._parameter("camera_translation_y_m")),
                float(self._parameter("camera_translation_z_m")),
            ),
            camera_rpy=(
                float(self._parameter("camera_roll_rad")),
                float(self._parameter("camera_pitch_rad")),
                float(self._parameter("camera_yaw_rad")),
            ),
        )
        self._publisher = self.create_publisher(
            Detection3DArray,
            self._parameter("detections_topic"),
            10,
        )
        subscriptions = (
            (Image, "rgb", "rgb_topic"),
            (CameraInfo, "rgb_info", "rgb_camera_info_topic"),
            (Image, "depth", "depth_topic"),
            (CameraInfo, "depth_info", "depth_camera_info_topic"),
            (Odometry, "odom", "odom_topic"),
        )
        for message_type, name, topic_parameter in subscriptions:
            self.create_subscription(
                message_type,
                self._parameter(topic_parameter),
                lambda message, sensor=name: self._receive(sensor, message),
                qos_profile_sensor_data,
            )
        self.create_timer(
            1.0 / float(self._parameter("detection_rate_hz")),
            self._detect,
        )
        self._last_detection_state: Optional[bool] = None
        self.get_logger().info("Yellow target simulation baseline is active")

    def _receive(self, name: str, message: object) -> None:
        self._latest[name] = (message, self._now_s())

    def _detect(self) -> None:
        required = ("rgb", "rgb_info", "depth", "depth_info", "odom")
        now = self._now_s()
        timeout = float(self._parameter("sensor_timeout_s"))
        if not all(
            name in self._latest and now - self._latest[name][1] <= timeout
            for name in required
        ):
            return

        output = Detection3DArray()
        output.header.stamp = self.get_clock().now().to_msg()
        output.header.frame_id = "map"
        try:
            region = find_yellow_region(self._latest["rgb"][0], self._thresholds)
            if region is not None:
                detection = self._localize(region, output)
                if detection is not None:
                    output.detections.append(detection)
        except (ValueError, TypeError) as error:
            self.get_logger().warning(f"Color detection skipped: {error}")
            return
        self._publisher.publish(output)

        detected = bool(output.detections)
        if detected != self._last_detection_state:
            if detected:
                point = output.detections[0].results[0].pose.pose.position
                self.get_logger().info(
                    f"Detected {self._parameter('target_label')} at map "
                    f"({point.x:.2f}, {point.y:.2f}, {point.z:.2f})"
                )
            else:
                self.get_logger().info("No yellow target in current RGB frame")
            self._last_detection_state = detected

    def _localize(self, region: object, output: Detection3DArray) -> Optional[Detection3D]:
        rgb_info = self._latest["rgb_info"][0]
        depth_info = self._latest["depth_info"][0]
        depth_image = self._latest["depth"][0]
        depth_pixel = remap_pixel(region.centroid, rgb_info, depth_info)
        depth_m = median_depth(
            depth_image,
            *depth_pixel,
            window_radius_px=int(self._parameter("depth_window_radius_px")),
            minimum_depth_m=float(self._parameter("minimum_depth_m")),
            maximum_depth_m=float(self._parameter("maximum_depth_m")),
        )
        if depth_m is None:
            return None
        camera_point = camera_point_from_pixel(depth_pixel, depth_m, depth_info)
        map_point = self._projector.transform_point(
            camera_point,
            self._latest["odom"][0].pose.pose,
        )

        detection = Detection3D()
        detection.header = output.header
        detection.id = str(self._parameter("target_entity_id"))
        result = ObjectHypothesisWithPose()
        result.hypothesis.class_id = str(self._parameter("target_label"))
        result.hypothesis.score = float(region.confidence)
        result.pose.pose.position.x = map_point[0]
        result.pose.pose.position.y = map_point[1]
        result.pose.pose.position.z = map_point[2]
        result.pose.pose.orientation.w = 1.0
        detection.results.append(result)
        detection.bbox.center.position.x = map_point[0]
        detection.bbox.center.position.y = map_point[1]
        detection.bbox.center.position.z = map_point[2]
        detection.bbox.center.orientation.w = 1.0
        detection.bbox.size.x = 0.8
        detection.bbox.size.y = max(
            0.1,
            (region.x_max - region.x_min + 1) / float(rgb_info.k[0]) * depth_m,
        )
        detection.bbox.size.z = max(
            0.1,
            (region.y_max - region.y_min + 1) / float(rgb_info.k[4]) * depth_m,
        )
        return detection

    def _parameter(self, name: str):
        return self.get_parameter(name).value

    def _now_s(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ColorTargetDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except RuntimeError:
        if rclpy.ok():
            raise
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
