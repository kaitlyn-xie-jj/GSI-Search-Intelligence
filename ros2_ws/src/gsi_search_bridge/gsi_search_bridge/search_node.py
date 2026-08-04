"""Event-driven ROS 2 node running GSI active search over Gazebo sensors."""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path
from typing import Dict, Optional, Tuple

import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import BatteryState, CameraInfo, Image, Imu, PointCloud2
from std_msgs.msg import String
from vision_msgs.msg import Detection3DArray

from .pointcloud_projection import PointCloudGroundProjector
from .scenario_context import load_search_scenario_context

from modules.search_intelligence import (
    AdaptiveActiveSearchPolicy,
    ActiveSearchPolicy,
    BayesianBeliefUpdater,
    BinarySensorModel,
    CandidateViewpointGenerator,
    SearchGrid,
    SearchObservationAdapter,
    SearchSensorFrame,
    SearchSession,
    SearchTask,
    TargetDetection,
    Viewpoint,
)


class GsiSearchNode(Node):
    """Select, execute, observe, and update one viewpoint at a time."""

    def __init__(self) -> None:
        super().__init__("gsi_search_node")
        self._declare_parameters()
        self._latest: Dict[str, Tuple[object, float, float]] = {}
        self._odom: Optional[Odometry] = None
        self._odometry_distance_m = 0.0
        self._last_odom_position: Optional[Tuple[float, float, float]] = None
        self._battery_percentage: Optional[float] = None
        self._session: Optional[SearchSession] = None
        self._adapter: Optional[SearchObservationAdapter] = None
        self._grid: Optional[SearchGrid] = None
        self._commanded_viewpoint: Optional[Viewpoint] = None
        self._command_start_time_s = 0.0
        self._command_start_distance_m = 0.0
        self._command_start_battery: Optional[float] = None
        self._settled_since_s: Optional[float] = None
        self._last_goal_publish_s = float("-inf")
        self._outcome_published = False
        self._pointcloud_projector = PointCloudGroundProjector(
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
            ground_plane_z_m=float(self._parameter("ground_plane_z_m")),
            ground_tolerance_m=float(self._parameter("ground_tolerance_m")),
            point_resolution_m=float(
                self._parameter("visibility_point_resolution_m")
            ),
            sample_limit=int(self._parameter("pointcloud_sample_limit")),
            maximum_range_m=float(self._parameter("pointcloud_maximum_range_m")),
            expected_frame_id=str(self._parameter("pointcloud_frame_id")),
        )

        self._goal_publisher = self.create_publisher(
            PoseStamped,
            self._parameter("goal_pose_topic"),
            10,
        )
        self._outcome_publisher = self.create_publisher(
            String,
            self._parameter("outcome_topic"),
            10,
        )
        self._create_sensor_subscriptions()
        self.create_timer(0.1, self._tick)
        self.get_logger().info("GSI search node waiting for odometry and sensors")

    def _declare_parameters(self) -> None:
        defaults = {
            "target_query": "yellow-van",
            "area_id": "gazebo-search-area",
            "area_min_x_m": -50.0,
            "area_min_y_m": -40.0,
            "area_max_x_m": 50.0,
            "area_max_y_m": 40.0,
            "grid_resolution_m": 10.0,
            "flight_altitude_m": 20.0,
            "sensor_footprint_radius_m": 15.0,
            "max_viewpoints": 30,
            "min_confirmations": 2,
            "max_localization_error_m": 5.0,
            "verification_followup_limit": 0,
            "semantic_map_path": "",
            "search_prior_path": "",
            "sensor_detection_probability": 0.85,
            "sensor_false_positive_probability": 0.01,
            "active_detection_weight": 1.0,
            "active_information_gain_weight": 1.0,
            "active_novelty_weight": 0.25,
            "active_travel_weight": 0.1,
            "active_adaptive_weights_enabled": True,
            "active_distance_scale_mode": "fixed",
            "active_distance_scale_m": 100.0,
            "position_tolerance_m": 0.75,
            "velocity_tolerance_mps": 0.25,
            "settle_time_s": 1.0,
            "goal_republish_interval_s": 0.5,
            "sensor_timeout_s": 0.5,
            "maximum_sensor_skew_s": 0.25,
            "require_detections": True,
            "require_point_cloud": False,
            "camera_translation_x_m": 0.0,
            "camera_translation_y_m": 0.0,
            "camera_translation_z_m": 0.0,
            "camera_roll_rad": 0.0,
            "camera_pitch_rad": 0.0,
            "camera_yaw_rad": 0.0,
            "ground_plane_z_m": 0.0,
            "ground_tolerance_m": 0.3,
            "visibility_point_resolution_m": 0.5,
            "pointcloud_sample_limit": 8000,
            "pointcloud_maximum_range_m": 19.1,
            "pointcloud_frame_id": "camera_link",
            "battery_capacity_wh": 0.0,
            "detections_in_map_frame": True,
            "odom_topic": "/uav/odom",
            "imu_topic": "/uav/imu",
            "rgb_topic": "/uav/camera/image_raw",
            "camera_info_topic": "/uav/camera/camera_info",
            "depth_topic": "/uav/depth/image_raw",
            "point_cloud_topic": "/uav/lidar/points",
            "detections_topic": "/uav/detections",
            "battery_topic": "/uav/battery_state",
            "goal_pose_topic": "/gsi/uav/goal_pose",
            "outcome_topic": "/gsi/search/outcome",
            "trace_output_path": "",
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

    def _create_sensor_subscriptions(self) -> None:
        subscriptions = (
            (Image, "rgb", "rgb_topic"),
            (CameraInfo, "camera_info", "camera_info_topic"),
            (Image, "depth", "depth_topic"),
            (PointCloud2, "point_cloud", "point_cloud_topic"),
            (Imu, "imu", "imu_topic"),
            (Detection3DArray, "detections", "detections_topic"),
            (BatteryState, "battery", "battery_topic"),
        )
        self.create_subscription(
            Odometry,
            self._parameter("odom_topic"),
            self._on_odometry,
            qos_profile_sensor_data,
        )
        for message_type, sensor_name, topic_parameter in subscriptions:
            self.create_subscription(
                message_type,
                self._parameter(topic_parameter),
                lambda message, name=sensor_name: self._on_sensor(name, message),
                qos_profile_sensor_data,
            )

    def _on_odometry(self, message: Odometry) -> None:
        position = message.pose.pose.position
        current = (float(position.x), float(position.y), float(position.z))
        if self._last_odom_position is not None:
            self._odometry_distance_m += math.dist(self._last_odom_position, current)
        self._last_odom_position = current
        self._odom = message
        self._on_sensor("odom", message)

    def _on_sensor(self, name: str, message: object) -> None:
        received = self._now_s()
        timestamp = _message_timestamp_s(message) or received
        self._latest[name] = (message, timestamp, received)
        if name == "battery":
            percentage = float(getattr(message, "percentage", float("nan")))
            if math.isfinite(percentage) and percentage >= 0:
                self._battery_percentage = percentage

    def _tick(self) -> None:
        if self._odom is None:
            return
        if self._session is None:
            self._initialize_session()
        assert self._session is not None
        if self._session.completed:
            self._publish_outcome()
            return
        if self._commanded_viewpoint is None:
            viewpoint = self._session.next_viewpoint()
            if viewpoint is None:
                self._publish_outcome()
                return
            self._start_viewpoint_action(viewpoint)
            return
        self._republish_goal_if_due()
        if not self._is_settled_at(self._commanded_viewpoint):
            self._settled_since_s = None
            return
        now = self._now_s()
        if self._settled_since_s is None:
            self._settled_since_s = now
            return
        if now - self._settled_since_s < float(self._parameter("settle_time_s")):
            return
        if not self._required_sensors_are_fresh():
            return
        self._record_sensor_observation()

    def _initialize_session(self) -> None:
        actual = self._actual_viewpoint()
        task = SearchTask.from_skill_params({
            "task_id": "gazebo-active-search",
            "area_token": self._parameter("area_id"),
            "area": {
                "kind": "rectangle",
                "coords": [
                    [self._parameter("area_min_x_m"), self._parameter("area_min_y_m")],
                    [self._parameter("area_max_x_m"), self._parameter("area_min_y_m")],
                    [self._parameter("area_max_x_m"), self._parameter("area_max_y_m")],
                    [self._parameter("area_min_x_m"), self._parameter("area_max_y_m")],
                ],
            },
            "target_token": self._parameter("target_query"),
            "max_viewpoints": int(self._parameter("max_viewpoints")),
            "min_confirmations": int(self._parameter("min_confirmations")),
            "max_localization_error_m": float(
                self._parameter("max_localization_error_m")
            ),
        })
        grid = SearchGrid.from_task(
            task,
            resolution_m=float(self._parameter("grid_resolution_m")),
        )
        scenario = load_search_scenario_context(
            task,
            grid,
            semantic_map_path=str(self._parameter("semantic_map_path")),
            search_prior_path=str(self._parameter("search_prior_path")),
        )
        grid = scenario.grid
        self._grid = grid
        footprint = float(self._parameter("sensor_footprint_radius_m"))
        candidates = CandidateViewpointGenerator(
            altitude_m=float(self._parameter("flight_altitude_m")),
            footprint_radius_m=footprint,
        ).generate(grid)
        sensor_model = BinarySensorModel(
            detection_probability=float(self._parameter("sensor_detection_probability")),
            false_positive_probability=float(
                self._parameter("sensor_false_positive_probability")
            ),
        )
        verification_limit = int(self._parameter("verification_followup_limit"))
        distance_scale_mode = str(
            self._parameter("active_distance_scale_mode")
        ).strip().lower()
        if distance_scale_mode == "map_diagonal":
            x_min, y_min, x_max, y_max = grid.bounds
            distance_scale_m = math.hypot(x_max - x_min, y_max - y_min)
        elif distance_scale_mode == "fixed":
            distance_scale_m = float(self._parameter("active_distance_scale_m"))
        else:
            raise ValueError(
                "active_distance_scale_mode must be fixed or map_diagonal"
            )
        adaptive_weights_enabled = bool(
            self._parameter("active_adaptive_weights_enabled")
        )
        policy_type = (
            AdaptiveActiveSearchPolicy
            if adaptive_weights_enabled
            else ActiveSearchPolicy
        )
        policy = policy_type(
            candidates,
            sensor_model=sensor_model,
            detection_weight=float(self._parameter("active_detection_weight")),
            information_gain_weight=float(
                self._parameter("active_information_gain_weight")
            ),
            novelty_weight=float(self._parameter("active_novelty_weight")),
            travel_weight=float(self._parameter("active_travel_weight")),
            distance_scale_m=distance_scale_m,
            verification_followup_limit=(
                verification_limit if verification_limit > 0 else None
            ),
        )
        self._adapter = SearchObservationAdapter(
            grid,
            target_query=str(self._parameter("target_query")),
            fallback_footprint_radius_m=footprint,
            maximum_sensor_skew_s=float(self._parameter("maximum_sensor_skew_s")),
        )
        self._session = SearchSession(
            task,
            policy,
            initial_belief=scenario.initial_belief,
            current_viewpoint=actual,
            initial_policy_metadata={
                "source": "ros2_gazebo_harmonic",
                "active_utility_weights": {
                    "detection": float(self._parameter("active_detection_weight")),
                    "information_gain": float(
                        self._parameter("active_information_gain_weight")
                    ),
                    "novelty": float(self._parameter("active_novelty_weight")),
                    "travel": float(self._parameter("active_travel_weight")),
                },
                "active_distance_scale_mode": distance_scale_mode,
                "active_distance_scale_m": distance_scale_m,
                "active_adaptive_weights_enabled": adaptive_weights_enabled,
                **scenario.policy_metadata,
            },
            search_grid=grid,
            belief_updater=BayesianBeliefUpdater(sensor_model),
        )
        self.get_logger().info(
            f"Initialized {'adaptive' if adaptive_weights_enabled else 'fixed'} "
            f"active search with {len(candidates)} candidate viewpoints; "
            f"semantics={scenario.policy_metadata['semantic_map_loaded']}, "
            f"prior={scenario.policy_metadata['prior_loaded']}"
        )

    def _start_viewpoint_action(self, viewpoint: Viewpoint) -> None:
        self._commanded_viewpoint = viewpoint
        self._command_start_time_s = self._now_s()
        self._command_start_distance_m = self._odometry_distance_m
        self._command_start_battery = self._battery_percentage
        self._settled_since_s = None
        self._publish_goal(viewpoint)
        self._write_trace("command", {
            "step_index": self._session.state.step_index if self._session else 0,
            "commanded_viewpoint": _viewpoint_dict(viewpoint),
        })

    def _republish_goal_if_due(self) -> None:
        assert self._commanded_viewpoint is not None
        if self._now_s() - self._last_goal_publish_s >= float(
            self._parameter("goal_republish_interval_s")
        ):
            self._publish_goal(self._commanded_viewpoint)

    def _publish_goal(self, viewpoint: Viewpoint) -> None:
        message = PoseStamped()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = "map"
        message.pose.position.x = viewpoint.x
        message.pose.position.y = viewpoint.y
        message.pose.position.z = viewpoint.z
        message.pose.orientation.z = math.sin(viewpoint.yaw / 2.0)
        message.pose.orientation.w = math.cos(viewpoint.yaw / 2.0)
        self._goal_publisher.publish(message)
        self._last_goal_publish_s = self._now_s()

    def _record_sensor_observation(self) -> None:
        assert self._session is not None
        assert self._adapter is not None
        assert self._commanded_viewpoint is not None
        now = self._now_s()
        timeout = float(self._parameter("sensor_timeout_s"))
        fresh = {
            name: item for name, item in self._latest.items()
            if now - item[2] <= timeout
        }
        visible_ground_points = self._visible_ground_points(fresh)
        frame = SearchSensorFrame(
            timestamp_s=now,
            viewpoint=self._actual_viewpoint(),
            frame_id="map",
            detections=self._target_detections(),
            visible_ground_points_xy=visible_ground_points,
            observation_quality=self._observation_quality(),
            travel_time_s=max(0.0, now - self._command_start_time_s),
            travel_distance_m=max(
                0.0, self._odometry_distance_m - self._command_start_distance_m
            ),
            energy_used=self._energy_used(),
            sensor_timestamps_s={
                # Receipt times share the node's ROS clock. Gazebo camera headers
                # use simulation time while MAVROS headers use synchronized Unix
                # time, so header stamps cannot be compared directly.
                name: item[2] for name, item in fresh.items()
                if name in {"odom", "rgb", "camera_info", "depth"}
            },
            frame_references={
                name: f"{name}@{item[1]:.9f}"
                for name, item in fresh.items()
            },
            metadata={
                "rgb_available": "rgb" in fresh,
                "depth_available": "depth" in fresh,
                "point_cloud_available": "point_cloud" in fresh,
                "camera_info_available": "camera_info" in fresh,
                "detections_available": "detections" in fresh,
                "sensor_header_timestamps_s": {
                    name: item[1] for name, item in fresh.items()
                },
                "projected_ground_point_count": len(visible_ground_points),
            },
        )
        observation = self._adapter.adapt(frame, self._commanded_viewpoint)
        state = self._session.record_observation(observation)
        self._write_trace("observation", {
            "step_index": state.step_index,
            "observation": observation.to_dict(),
            "belief": dict(state.belief),
            "policy_decision": (
                self._session.policy_decisions[-1]
                if self._session.policy_decisions else None
            ),
        })
        self.get_logger().info(
            f"Recorded viewpoint {state.step_index}; "
            f"detections={len(observation.detections)}, "
            f"visible_cells={len(observation.visible_cell_ids)}, "
            f"ground_points={len(visible_ground_points)}, "
            f"quality={observation.observation_quality:.3f}"
        )
        self._commanded_viewpoint = None
        self._settled_since_s = None
        if self._session.completed:
            self._publish_outcome()

    def _required_sensors_are_fresh(self) -> bool:
        now = self._now_s()
        timeout = float(self._parameter("sensor_timeout_s"))
        required = ["odom", "rgb", "camera_info", "depth"]
        if bool(self._parameter("require_detections")):
            required.append("detections")
        if bool(self._parameter("require_point_cloud")):
            required.append("point_cloud")
        return all(
            name in self._latest and now - self._latest[name][2] <= timeout
            for name in required
        )

    def _visible_ground_points(
        self,
        fresh: Dict[str, Tuple[object, float, float]],
    ) -> Tuple[Tuple[float, float], ...]:
        item = fresh.get("point_cloud")
        if item is None or self._odom is None:
            return ()
        try:
            return self._pointcloud_projector.project(
                item[0],
                self._odom.pose.pose,
            )
        except (ValueError, struct.error) as error:
            self.get_logger().warning(f"Point-cloud projection skipped: {error}")
            return ()

    def _observation_quality(self) -> float:
        quality = 1.0
        imu_item = self._latest.get("imu")
        if imu_item is not None:
            angular = imu_item[0].angular_velocity
            angular_speed = math.sqrt(angular.x ** 2 + angular.y ** 2 + angular.z ** 2)
            quality *= max(0.0, 1.0 - angular_speed / 1.5)
        return quality

    def _target_detections(self) -> Tuple[TargetDetection, ...]:
        item = self._latest.get("detections")
        if item is None:
            return ()
        message = item[0]
        target = _normalize_label(str(self._parameter("target_query")))
        in_map = bool(self._parameter("detections_in_map_frame"))
        detections = []
        for detection in message.detections:
            if not detection.results:
                continue
            result = max(
                detection.results,
                key=lambda candidate: float(candidate.hypothesis.score),
            )
            label = str(result.hypothesis.class_id)
            if _normalize_label(label) != target:
                continue
            position = result.pose.pose.position
            estimated_position = (
                (float(position.x), float(position.y), float(position.z))
                if in_map else None
            )
            localized_cell_id = None
            if estimated_position is not None and self._grid is not None:
                cell = self._grid.cell_at(
                    estimated_position[0],
                    estimated_position[1],
                )
                if cell is not None and cell.searchable:
                    localized_cell_id = cell.cell_id
            localization_uncertainty = _horizontal_position_uncertainty_m(
                result.pose.covariance
            )
            detections.append(TargetDetection(
                label=label,
                confidence=float(result.hypothesis.score),
                estimated_position=estimated_position,
                entity_id=str(detection.id) if detection.id else None,
                attributes={
                    "source": "vision_msgs/Detection3DArray",
                    "detection_frame_id": message.header.frame_id,
                    "localized_cell_id": localized_cell_id,
                    "localization_error_m": localization_uncertainty,
                    "localization_metric": "reported_horizontal_covariance",
                },
            ))
        return tuple(detections)

    def _is_settled_at(self, viewpoint: Viewpoint) -> bool:
        actual = self._actual_viewpoint()
        position_error = math.sqrt(
            (actual.x - viewpoint.x) ** 2
            + (actual.y - viewpoint.y) ** 2
            + (actual.z - viewpoint.z) ** 2
        )
        velocity = self._odom.twist.twist.linear
        speed = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
        return (
            position_error <= float(self._parameter("position_tolerance_m"))
            and speed <= float(self._parameter("velocity_tolerance_mps"))
        )

    def _actual_viewpoint(self) -> Viewpoint:
        pose = self._odom.pose.pose
        return Viewpoint(
            float(pose.position.x),
            float(pose.position.y),
            float(pose.position.z),
            _yaw_from_quaternion(pose.orientation),
            pitch=0.0,
        )

    def _energy_used(self) -> float:
        if self._command_start_battery is None or self._battery_percentage is None:
            return 0.0
        fraction = max(0.0, self._command_start_battery - self._battery_percentage)
        capacity = float(self._parameter("battery_capacity_wh"))
        return fraction * capacity if capacity > 0 else fraction

    def _publish_outcome(self) -> None:
        if self._outcome_published or self._session is None:
            return
        if self._session.outcome is None:
            self._session.next_viewpoint()
        if self._session.outcome is None:
            return
        message = String()
        message.data = json.dumps(
            self._session.outcome.to_platform_result(),
            sort_keys=True,
        )
        self._outcome_publisher.publish(message)
        self._write_trace("outcome", {
            "outcome": self._session.outcome.to_dict(),
        })
        self._outcome_published = True
        self.get_logger().info(f"Search completed: {message.data}")

    def _parameter(self, name: str):
        return self.get_parameter(name).value

    def _now_s(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    def _write_trace(self, event: str, payload: Dict[str, object]) -> None:
        output_path = str(self._parameter("trace_output_path")).strip()
        if not output_path:
            return
        path = Path(output_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "schema_version": "gsi-gazebo-sensor-trace-v1",
            "event": event,
            "timestamp_s": self._now_s(),
            **payload,
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def _message_timestamp_s(message: object) -> float:
    header = getattr(message, "header", None)
    stamp = getattr(header, "stamp", None)
    if stamp is None:
        return 0.0
    return float(stamp.sec) + float(stamp.nanosec) / 1e9


def _yaw_from_quaternion(quaternion: object) -> float:
    x, y, z, w = (
        float(quaternion.x),
        float(quaternion.y),
        float(quaternion.z),
        float(quaternion.w),
    )
    return math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def _normalize_label(value: str) -> str:
    return "".join(character for character in value.lower() if character.isalnum())


def _horizontal_position_uncertainty_m(covariance: object) -> float:
    values = tuple(float(value) for value in covariance)
    if len(values) != 36:
        return 0.0
    variance_x = max(0.0, values[0])
    variance_y = max(0.0, values[7])
    return math.sqrt(variance_x + variance_y)


def _viewpoint_dict(viewpoint: Viewpoint) -> Dict[str, float]:
    return {
        "x": viewpoint.x,
        "y": viewpoint.y,
        "z": viewpoint.z,
        "yaw": viewpoint.yaw,
        "pitch": viewpoint.pitch,
    }


def main(args=None) -> None:
    rclpy.init(args=args)
    node = GsiSearchNode()
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
