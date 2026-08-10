"""ROS 2 Ultralytics vehicle detector with RGB-D map localization."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose

from .color_detection import camera_point_from_pixel, median_depth, remap_pixel
from .pointcloud_projection import PointCloudGroundProjector
from .yolo_detection import best_vehicle_region, image_as_bgr


class YoloTargetDetector(Node):
    """Detect COCO vehicles and preserve the existing /gsi/detections contract."""

    def __init__(self) -> None:
        super().__init__("gsi_yolo_target_detector")
        defaults = {
            "rgb_topic": "/oakd1/rgb/image",
            "rgb_camera_info_topic": "/oakd1/rgb/camera_info",
            "depth_topic": "/oakd1/depth/image",
            "depth_camera_info_topic": "/oakd1/depth/camera_info",
            "odom_topic": "/mavros/local_position/odom",
            "detections_topic": "/gsi/detections",
            "fallback_detections_topic": "",
            "model_path": "yolo11n.pt",
            "device": "",
            "image_size": 640,
            "confidence_threshold": 0.20,
            "iou_threshold": 0.45,
            "vehicle_class_ids": [2, 5, 7],
            "target_label": "car",
            "target_entity_id": "yellow-search-van",
            "detection_rate_hz": 5.0,
            "sensor_timeout_s": 3.0,
            "depth_window_radius_px": 8,
            "minimum_depth_m": 0.2,
            "maximum_depth_m": 90.0,
            "map_origin_x_m": 0.0,
            "map_origin_y_m": 0.0,
            "map_origin_z_m": 0.0,
            "camera_translation_x_m": 0.0,
            "camera_translation_y_m": 0.0,
            "camera_translation_z_m": 0.0,
            "camera_roll_rad": 0.0,
            "camera_pitch_rad": 0.0,
            "camera_yaw_rad": 0.0,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise RuntimeError(
                "Ultralytics is required by yolo_target_detector; install the "
                "vision requirements or disable start_yolo_detector"
            ) from error
        self._model = YOLO(str(self._parameter("model_path")))
        self._latest: Dict[str, Tuple[object, float]] = {}
        self._latest_fallback: Optional[Detection3DArray] = None
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
            Detection3DArray, str(self._parameter("detections_topic")), 10
        )
        for message_type, name, parameter in (
            (Image, "rgb", "rgb_topic"),
            (CameraInfo, "rgb_info", "rgb_camera_info_topic"),
            (Image, "depth", "depth_topic"),
            (CameraInfo, "depth_info", "depth_camera_info_topic"),
            (Odometry, "odom", "odom_topic"),
        ):
            self.create_subscription(
                message_type,
                str(self._parameter(parameter)),
                lambda message, sensor=name: self._receive(sensor, message),
                qos_profile_sensor_data,
            )
        fallback_topic = str(self._parameter("fallback_detections_topic")).strip()
        if fallback_topic:
            self.create_subscription(
                Detection3DArray,
                fallback_topic,
                self._receive_fallback,
                10,
            )
        self.create_timer(
            1.0 / float(self._parameter("detection_rate_hz")), self._detect
        )
        self.get_logger().info(
            f"YOLO vehicle detector active: {self._parameter('model_path')}"
        )

    def _receive(self, name: str, message: object) -> None:
        self._latest[name] = (message, self._now_s())

    def _receive_fallback(self, message: Detection3DArray) -> None:
        self._latest_fallback = message

    def _detect(self) -> None:
        now = self._now_s()
        timeout = float(self._parameter("sensor_timeout_s"))
        if not all(name in self._latest for name in (
            "rgb", "rgb_info", "depth", "depth_info", "odom"
        )):
            return
        if any(now - self._latest[name][1] > timeout for name in (
            "rgb", "depth", "odom"
        )):
            return

        image = image_as_bgr(self._latest["rgb"][0])
        predict_args = {
            "source": image,
            "imgsz": int(self._parameter("image_size")),
            "conf": float(self._parameter("confidence_threshold")),
            "iou": float(self._parameter("iou_threshold")),
            "classes": [int(item) for item in self._parameter("vehicle_class_ids")],
            "verbose": False,
        }
        device = str(self._parameter("device")).strip()
        if device:
            predict_args["device"] = device
        results = self._model.predict(**predict_args)
        region = best_vehicle_region(
            results,
            predict_args["classes"],
            getattr(self._model, "names", None),
        )
        output = Detection3DArray()
        output.header.stamp = self.get_clock().now().to_msg()
        output.header.frame_id = "map"
        if region is not None:
            detection = self._localize(region, output)
            if detection is not None:
                output.detections.append(detection)
        elif self._latest_fallback is not None:
            output.detections.extend(self._latest_fallback.detections)
        self._publisher.publish(output)

    def _localize(self, region: object, output: Detection3DArray) -> Optional[Detection3D]:
        rgb_info = self._latest["rgb_info"][0]
        depth_info = self._latest["depth_info"][0]
        depth_pixel = remap_pixel(region.centroid, rgb_info, depth_info)
        depth_m = median_depth(
            self._latest["depth"][0],
            *depth_pixel,
            window_radius_px=int(self._parameter("depth_window_radius_px")),
            minimum_depth_m=float(self._parameter("minimum_depth_m")),
            maximum_depth_m=float(self._parameter("maximum_depth_m")),
        )
        if depth_m is None:
            return None
        camera_point = camera_point_from_pixel(depth_pixel, depth_m, depth_info)
        map_point = self._projector.transform_point(
            camera_point, self._latest["odom"][0].pose.pose
        )
        map_point = tuple(map_point[index] + float(self._parameter(name)) for index, name in enumerate(
            ("map_origin_x_m", "map_origin_y_m", "map_origin_z_m")
        ))
        detection = Detection3D()
        detection.header = output.header
        detection.id = str(self._parameter("target_entity_id"))
        result = ObjectHypothesisWithPose()
        result.hypothesis.class_id = str(self._parameter("target_label"))
        result.hypothesis.score = float(region.confidence)
        result.pose.pose.position.x, result.pose.pose.position.y, result.pose.pose.position.z = map_point
        result.pose.pose.orientation.w = 1.0
        detection.results.append(result)
        detection.bbox.center = result.pose.pose
        detection.bbox.size.x = max(0.1, (region.x_max - region.x_min) / float(rgb_info.k[0]) * depth_m)
        detection.bbox.size.y = max(0.1, (region.y_max - region.y_min) / float(rgb_info.k[4]) * depth_m)
        detection.bbox.size.z = 1.5
        return detection

    def _parameter(self, name: str):
        return self.get_parameter(name).value

    def _now_s(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9


def main(args=None) -> None:
    rclpy.init(args=args)
    node = YoloTargetDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
