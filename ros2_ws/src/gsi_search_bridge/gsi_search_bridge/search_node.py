"""Event-driven ROS 2 node running GSI active search over Gazebo sensors."""

from __future__ import annotations

import json
import math
import struct
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Tuple

import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import BatteryState, CameraInfo, Image, Imu, PointCloud2
from std_msgs.msg import String
from vision_msgs.msg import Detection3DArray

from .building_route_planner import (
    BuildingObstacle,
    load_building_obstacles,
    plan_building_avoiding_route,
    point_has_building_clearance,
)
from .pointcloud_projection import PointCloudGroundProjector
from .scenario_context import load_search_scenario_context

from modules.search_intelligence import (
    AdaptiveBeliefLookaheadPolicy,
    AdaptiveActiveSearchPolicy,
    ActiveSearchPolicy,
    BayesianBeliefUpdater,
    BinarySensorModel,
    CandidateViewpointGenerator,
    CoveragePolicy,
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
        self._navigation_goal: Optional[Viewpoint] = None
        self._remaining_navigation_goals: Tuple[Viewpoint, ...] = ()
        self._building_obstacles: Tuple[BuildingObstacle, ...] = ()
        self._planning_footprint_radius_m = float(
            self._parameter("sensor_footprint_radius_m")
        )
        self._command_start_time_s = 0.0
        self._command_start_distance_m = 0.0
        self._command_start_battery: Optional[float] = None
        self._observation_checkpoint_time_s = 0.0
        self._observation_checkpoint_distance_m = 0.0
        self._observation_checkpoint_battery: Optional[float] = None
        self._settled_since_s: Optional[float] = None
        self._last_goal_publish_s = float("-inf")
        self._last_progress_log_s = float("-inf")
        self._transit_detection_recorded = False
        self._transit_suspect_snapshot_count = 0
        self._transit_suspect_inspection_counts: Dict[str, int] = {}
        self._verification_snapshot_count = 0
        self._outcome_published = False
        self._safety_abort_reason: Optional[str] = None
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
        self.create_subscription(
            String,
            self._parameter("safety_status_topic"),
            self._on_safety_status,
            10,
        )
        self.create_timer(0.1, self._tick)
        self.get_logger().info("GSI search node waiting for odometry and sensors")

    def _declare_parameters(self) -> None:
        defaults = {
            "target_query": "yellow-van",
            "search_policy": "active",
            "area_id": "gazebo-search-area",
            "area_min_x_m": -50.0,
            "area_min_y_m": -40.0,
            "area_max_x_m": 50.0,
            "area_max_y_m": 40.0,
            "grid_resolution_m": 10.0,
            "flight_altitude_m": 20.0,
            "sensor_footprint_radius_m": 15.0,
            "camera_horizontal_fov_rad": 1.0471975512,
            "camera_image_width_px": 160,
            "camera_image_height_px": 120,
            "planning_footprint_scale": 0.95,
            "max_viewpoints": 30,
            "search_time_budget_s": 180.0,
            "min_confirmations": 2,
            "max_localization_error_m": 5.0,
            "verification_followup_limit": 0,
            "verification_max_horizontal_offset_m": 0.0,
            "semantic_map_path": "",
            "search_prior_path": "",
            "sensor_detection_probability": 0.85,
            "sensor_false_positive_probability": 0.01,
            "active_detection_weight": 1.0,
            "active_information_gain_weight": 1.0,
            "active_novelty_weight": 0.25,
            "active_travel_weight": 0.1,
            "active_revisit_weight": 0.2,
            "active_risk_weight": 0.25,
            "active_adaptive_weights_enabled": True,
            "active_belief_lookahead_enabled": False,
            "active_lookahead_discount_factor": 0.7,
            "active_lookahead_candidate_limit": 16,
            "active_candidate_exploitation_fraction": 0.3,
            "active_candidate_exploration_fraction": 0.4,
            "active_candidate_semantic_fraction": 0.3,
            "active_candidate_frontier_fraction": 0.5,
            "active_planning_speed_mps": 1.0,
            "active_completion_time_reserve_s": 5.0,
            "coverage_pass_spacing_m": 20.0,
            "coverage_observation_spacing_m": 0.0,
            "coverage_camera_pitch_rad": -1.5707963267948966,
            "coverage_start_from_nearest_endpoint": True,
            "coverage_recovery_enabled": False,
            "coverage_recovery_min_quality": 0.5,
            "coverage_recovery_offset_m": 0.0,
            "active_distance_scale_mode": "fixed",
            "active_distance_scale_m": 100.0,
            "position_tolerance_m": 0.75,
            "velocity_tolerance_mps": 0.25,
            "settle_time_s": 1.0,
            "goal_republish_interval_s": 0.5,
            "progress_log_interval_s": 5.0,
            "sensor_timeout_s": 0.5,
            "maximum_sensor_skew_s": 0.25,
            "minimum_observation_quality": 0.0,
            "minimum_negative_observation_quality": 0.5,
            "minimum_projected_ground_points": 10,
            "building_route_planning_enabled": False,
            "building_horizontal_clearance_m": 3.0,
            "building_vertical_clearance_m": 2.0,
            "building_corner_offset_m": 0.1,
            "require_detections": True,
            "require_point_cloud": False,
            "record_first_positive_detection_in_transit": False,
            "transit_detection_max_sensor_skew_s": 1.0,
            "transit_detection_min_observation_quality": 0.5,
            "record_negative_observations_in_transit": False,
            "transit_negative_min_interval_s": 10.0,
            "transit_negative_min_distance_m": 20.0,
            "transit_negative_min_new_cells": 3,
            "transit_suspect_min_belief": 0.05,
            "transit_suspect_inspection_limit": 0,
            "transit_suspect_min_angle_rad": 1.5707963267948966,
            "transit_suspect_snapshot_dir": "",
            "transit_suspect_snapshot_limit": 20,
            "verification_snapshot_dir": "",
            "verification_snapshot_limit": 20,
            "replan_min_interval_s": 20.0,
            "replan_belief_total_variation_threshold": 0.1,
            "replan_kl_divergence_threshold_nats": 0.05,
            "replan_expected_reward_change_threshold": 0.2,
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
            "map_origin_x_m": 0.0,
            "map_origin_y_m": 0.0,
            "map_origin_z_m": 0.0,
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
            "safety_status_topic": "/gsi/uav/safety_status",
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
        if self._safety_abort_reason is not None and not self._session.completed:
            self._session.abort(self._safety_abort_reason)
            self._commanded_viewpoint = None
            self._navigation_goal = None
            self._remaining_navigation_goals = ()
            self._write_trace("safety_abort", {
                "reason": self._safety_abort_reason,
                "viewpoint": _viewpoint_dict(self._actual_viewpoint()),
            })
        if self._session.completed:
            self._publish_outcome()
            return
        if self._time_budget_exhausted():
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
        if self._can_record_positive_detection_in_transit():
            self._transit_detection_recorded = True
            self._record_sensor_observation(
                "positive_detection_in_transit",
                transit=True,
            )
            return
        if self._can_record_negative_detection_in_transit():
            if self._record_sensor_observation(
                "negative_detection_in_transit",
                transit=True,
            ):
                return
        assert self._navigation_goal is not None
        if not self._is_settled_at(self._navigation_goal):
            self._settled_since_s = None
            self._log_progress("traveling")
            return
        now = self._now_s()
        if self._settled_since_s is None:
            self._settled_since_s = now
            return
        if now - self._settled_since_s < float(self._parameter("settle_time_s")):
            self._log_progress("settling")
            return
        if self._navigation_goal.key != self._commanded_viewpoint.key:
            self._advance_navigation_route()
            return
        if not self._required_sensors_are_fresh():
            self._log_progress("waiting_for_sensors")
            return
        if self._predicted_observation_quality() < float(
            self._parameter("minimum_observation_quality")
        ):
            self._log_progress("waiting_for_observation_quality")
            return
        self._record_sensor_observation("settled_viewpoint")

    def _on_safety_status(self, message: String) -> None:
        reason = str(message.data).strip()
        if not reason or self._safety_abort_reason is not None:
            return
        self._safety_abort_reason = f"PX4 safety stop: {reason}"
        self.get_logger().error(self._safety_abort_reason)

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
            "time_budget_s": float(self._parameter("search_time_budget_s")),
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
        if bool(self._parameter("building_route_planning_enabled")):
            self._building_obstacles = load_building_obstacles(
                str(self._parameter("semantic_map_path"))
            )
        configured_radius = float(self._parameter("sensor_footprint_radius_m"))
        footprint_half_extents = None
        if configured_radius > 0:
            self._planning_footprint_radius_m = configured_radius
        else:
            footprint_half_extents = _nadir_footprint_half_extents(
                altitude_m=float(self._parameter("flight_altitude_m")),
                horizontal_fov_rad=float(
                    self._parameter("camera_horizontal_fov_rad")
                ),
                image_width_px=int(self._parameter("camera_image_width_px")),
                image_height_px=int(self._parameter("camera_image_height_px")),
                scale=float(self._parameter("planning_footprint_scale")),
            )
            self._planning_footprint_radius_m = min(footprint_half_extents)
        generator_kwargs = {
            "altitude_m": float(self._parameter("flight_altitude_m")),
        }
        if footprint_half_extents is None:
            generator_kwargs["footprint_radius_m"] = configured_radius
        else:
            generator_kwargs.update({
                "footprint_half_width_m": footprint_half_extents[0],
                "footprint_half_height_m": footprint_half_extents[1],
            })
        candidates = CandidateViewpointGenerator(**generator_kwargs).generate(grid)
        unfiltered_candidate_count = len(candidates)
        if self._building_obstacles:
            candidates = tuple(
                candidate for candidate in candidates
                if self._viewpoint_has_building_clearance(candidate.viewpoint)
            )
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
        policy_name = str(self._parameter("search_policy")).strip().lower()
        adaptive_weights_enabled = bool(
            self._parameter("active_adaptive_weights_enabled")
        )
        lookahead_enabled = bool(
            self._parameter("active_belief_lookahead_enabled")
        )
        if policy_name == "coverage":
            observation_spacing = float(
                self._parameter("coverage_observation_spacing_m")
            )
            policy = CoveragePolicy(
                pass_spacing_m=float(
                    self._parameter("coverage_pass_spacing_m")
                ),
                altitude_m=float(self._parameter("flight_altitude_m")),
                camera_pitch_rad=float(
                    self._parameter("coverage_camera_pitch_rad")
                ),
                observation_spacing_m=(
                    observation_spacing if observation_spacing > 0 else None
                ),
                start_from_nearest_endpoint=bool(
                    self._parameter("coverage_start_from_nearest_endpoint")
                ),
                route_start_hint=actual,
                search_grid=grid,
                recovery_enabled=bool(
                    self._parameter("coverage_recovery_enabled")
                ),
                recovery_min_quality=float(
                    self._parameter("coverage_recovery_min_quality")
                ),
                recovery_offset_m=(
                    float(self._parameter("coverage_recovery_offset_m"))
                    if float(self._parameter("coverage_recovery_offset_m")) > 0
                    else None
                ),
                viewpoint_filter=(
                    self._viewpoint_has_building_clearance
                    if self._building_obstacles else None
                ),
            )
            policy_type = CoveragePolicy
        elif policy_name in {"active", "adaptive_active", "lookahead_active"}:
            policy_type = ActiveSearchPolicy
            if policy_name == "lookahead_active" or (
                policy_name == "active"
                and adaptive_weights_enabled and lookahead_enabled
            ):
                policy_type = AdaptiveBeliefLookaheadPolicy
            elif policy_name == "adaptive_active" or (
                policy_name == "active" and adaptive_weights_enabled
            ):
                policy_type = AdaptiveActiveSearchPolicy
        else:
            raise ValueError(
                "search_policy must be coverage, active, adaptive_active, "
                "or lookahead_active"
            )
        policy_kwargs = {
            "candidates": candidates,
            "sensor_model": sensor_model,
            "detection_weight": float(self._parameter("active_detection_weight")),
            "information_gain_weight": float(
                self._parameter("active_information_gain_weight")
            ),
            "novelty_weight": float(self._parameter("active_novelty_weight")),
            "travel_weight": float(self._parameter("active_travel_weight")),
            "revisit_weight": float(self._parameter("active_revisit_weight")),
            "risk_weight": float(self._parameter("active_risk_weight")),
            "distance_scale_m": distance_scale_m,
            "planning_speed_mps": float(
                self._parameter("active_planning_speed_mps")
            ),
            "completion_time_reserve_s": float(
                self._parameter("active_completion_time_reserve_s")
            ),
            "verification_followup_limit": (
                verification_limit if verification_limit > 0 else None
            ),
            "verification_max_horizontal_offset_m": (
                float(self._parameter("verification_max_horizontal_offset_m"))
                if float(self._parameter("verification_max_horizontal_offset_m")) > 0
                else None
            ),
        }
        cell_by_id = {cell.cell_id: cell for cell in grid.searchable_cells}
        policy_kwargs.update({
            "visibility_probabilities": {
                candidate.candidate_id: _semantic_visibility_probability(
                    cell_by_id[candidate.anchor_cell_id].semantic_labels
                )
                for candidate in candidates
            },
            "candidate_risk_scores": {
                candidate.candidate_id: _semantic_risk_score(
                    cell_by_id[candidate.anchor_cell_id].semantic_labels
                )
                for candidate in candidates
            },
        })
        if policy_type is AdaptiveBeliefLookaheadPolicy:
            policy_kwargs.update({
                "discount_factor": float(
                    self._parameter("active_lookahead_discount_factor")
                ),
                "lookahead_candidate_limit": int(
                    self._parameter("active_lookahead_candidate_limit")
                ),
                "exploitation_fraction": float(
                    self._parameter("active_candidate_exploitation_fraction")
                ),
                "exploration_fraction": float(
                    self._parameter("active_candidate_exploration_fraction")
                ),
                "semantic_fraction": float(
                    self._parameter("active_candidate_semantic_fraction")
                ),
                "frontier_fraction_within_exploration": float(
                    self._parameter("active_candidate_frontier_fraction")
                ),
                "semantic_regions": {
                    candidate.candidate_id: cell_by_id[
                        candidate.anchor_cell_id
                    ].semantic_labels
                    for candidate in candidates
                },
                "transit_suspect_inspection_limit": int(
                    self._parameter("transit_suspect_inspection_limit")
                ),
                "transit_suspect_min_angle_rad": float(
                    self._parameter("transit_suspect_min_angle_rad")
                ),
            })
        if policy_type is not CoveragePolicy:
            policy = policy_type(**policy_kwargs)
        self._adapter = SearchObservationAdapter(
            grid,
            target_query=str(self._parameter("target_query")),
            fallback_footprint_radius_m=self._planning_footprint_radius_m,
            maximum_sensor_skew_s=float(self._parameter("maximum_sensor_skew_s")),
            minimum_negative_observation_quality=float(
                self._parameter("minimum_negative_observation_quality")
            ),
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
                    "revisit": float(self._parameter("active_revisit_weight")),
                    "risk": float(self._parameter("active_risk_weight")),
                },
                "active_distance_scale_mode": distance_scale_mode,
                "active_distance_scale_m": distance_scale_m,
                "active_adaptive_weights_enabled": adaptive_weights_enabled,
                "active_belief_lookahead_enabled": lookahead_enabled,
                "search_policy": policy_name,
                **scenario.policy_metadata,
            },
            search_grid=grid,
            belief_updater=BayesianBeliefUpdater(sensor_model),
        )
        self.get_logger().info(
            f"Initialized {policy_type.__name__} with "
            f"{len(candidates)} candidate viewpoints; "
            f"building_filtered={unfiltered_candidate_count - len(candidates)}; "
            f"building_obstacles={len(self._building_obstacles)}; "
            f"planning_footprint_radius_m={self._planning_footprint_radius_m:.2f}; "
            f"planning_footprint_half_extents_m={footprint_half_extents}; "
            f"semantics={scenario.policy_metadata['semantic_map_loaded']}, "
            f"prior={scenario.policy_metadata['prior_loaded']}"
        )

    def _start_viewpoint_action(self, viewpoint: Viewpoint) -> None:
        route = self._plan_navigation_route(self._actual_viewpoint(), viewpoint)
        if route is None:
            raise RuntimeError(
                "No collision-free building route to selected viewpoint "
                f"({viewpoint.x:.2f}, {viewpoint.y:.2f}, {viewpoint.z:.2f})"
            )
        self._commanded_viewpoint = viewpoint
        self._navigation_goal = route[0]
        self._remaining_navigation_goals = route[1:]
        self._command_start_time_s = self._now_s()
        self._command_start_distance_m = self._odometry_distance_m
        self._command_start_battery = self._battery_percentage
        self._observation_checkpoint_time_s = self._command_start_time_s
        self._observation_checkpoint_distance_m = self._command_start_distance_m
        self._observation_checkpoint_battery = self._command_start_battery
        self._settled_since_s = None
        self._publish_goal(self._navigation_goal)
        decision = self._session.pending_policy_metadata if self._session else {}
        self.get_logger().info(
            "Selected viewpoint "
            f"({viewpoint.x:.2f}, {viewpoint.y:.2f}, {viewpoint.z:.2f}); "
            f"verification_mode={bool(decision.get('verification_mode'))}; "
            f"verification_cell={decision.get('verification_cell_id')}; "
            f"transit_suspect_mode={bool(decision.get('transit_suspect_inspection_mode'))}; "
            f"transit_suspect_cell={decision.get('transit_suspect_cell_id')}; "
            f"route_segments={len(route)}"
        )
        self._write_trace("command", {
            "step_index": self._session.state.step_index if self._session else 0,
            "commanded_viewpoint": _viewpoint_dict(viewpoint),
            "navigation_route": [_viewpoint_dict(item) for item in route],
        })

    def _republish_goal_if_due(self) -> None:
        assert self._commanded_viewpoint is not None
        assert self._navigation_goal is not None
        if self._now_s() - self._last_goal_publish_s >= float(
            self._parameter("goal_republish_interval_s")
        ):
            self._publish_goal(self._navigation_goal)

    def _advance_navigation_route(self) -> None:
        assert self._navigation_goal is not None
        reached = self._navigation_goal
        if not self._remaining_navigation_goals:
            return
        self._navigation_goal = self._remaining_navigation_goals[0]
        self._remaining_navigation_goals = self._remaining_navigation_goals[1:]
        self._settled_since_s = None
        self._publish_goal(self._navigation_goal)
        self.get_logger().info(
            "Reached navigation waypoint "
            f"({reached.x:.2f}, {reached.y:.2f}, {reached.z:.2f}); "
            "continuing to "
            f"({self._navigation_goal.x:.2f}, {self._navigation_goal.y:.2f}, "
            f"{self._navigation_goal.z:.2f})"
        )
        self._write_trace("navigation_waypoint", {
            "reached": _viewpoint_dict(reached),
            "next_navigation_goal": _viewpoint_dict(self._navigation_goal),
            "final_viewpoint": _viewpoint_dict(self._commanded_viewpoint),
            "remaining_segments": 1 + len(self._remaining_navigation_goals),
        })

    def _plan_navigation_route(
        self,
        start: Viewpoint,
        goal: Viewpoint,
    ) -> Optional[Tuple[Viewpoint, ...]]:
        if not bool(self._parameter("building_route_planning_enabled")):
            return (goal,)
        points = plan_building_avoiding_route(
            (start.x, start.y, start.z),
            (goal.x, goal.y, goal.z),
            self._building_obstacles,
            horizontal_clearance_m=float(
                self._parameter("building_horizontal_clearance_m")
            ),
            vertical_clearance_m=float(
                self._parameter("building_vertical_clearance_m")
            ),
            corner_offset_m=float(self._parameter("building_corner_offset_m")),
            route_bounds=(
                float(self._parameter("area_min_x_m")),
                float(self._parameter("area_min_y_m")),
                float(self._parameter("area_max_x_m")),
                float(self._parameter("area_max_y_m")),
            ),
        )
        if points is None:
            return None
        route = []
        previous = start
        for index, point in enumerate(points):
            final = index == len(points) - 1
            yaw = goal.yaw if final else math.atan2(point[1] - previous.y, point[0] - previous.x)
            route.append(Viewpoint(point[0], point[1], point[2], yaw, goal.pitch))
            previous = route[-1]
        return tuple(route)

    def _viewpoint_has_building_clearance(self, viewpoint: Viewpoint) -> bool:
        return point_has_building_clearance(
            (viewpoint.x, viewpoint.y, viewpoint.z),
            self._building_obstacles,
            horizontal_clearance_m=float(
                self._parameter("building_horizontal_clearance_m")
            ),
            vertical_clearance_m=float(
                self._parameter("building_vertical_clearance_m")
            ),
        )

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

    def _record_sensor_observation(
        self,
        trigger: str,
        *,
        transit: bool = False,
    ) -> bool:
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
        minimum_projected_points = int(
            self._parameter("minimum_projected_ground_points")
        )
        visibility_probability = _projection_visibility_probability(
            len(visible_ground_points),
            minimum_projected_points,
        )
        rejection_reason = _negative_update_rejection_reason(
            rgb_available="rgb" in fresh,
            depth_available="depth" in fresh,
            point_cloud_available="point_cloud" in fresh,
            projected_ground_point_count=len(visible_ground_points),
            minimum_projected_ground_points=minimum_projected_points,
            observation_quality=self._predicted_observation_quality(),
            minimum_observation_quality=float(
                self._parameter("minimum_negative_observation_quality")
            ),
        )
        frame = SearchSensorFrame(
            timestamp_s=now,
            viewpoint=self._actual_viewpoint(),
            frame_id="map",
            detections=self._target_detections(),
            visible_ground_points_xy=visible_ground_points,
            observation_quality=self._observation_quality(),
            visibility_probability=visibility_probability,
            negative_update_strength=(0.0 if rejection_reason else 1.0),
            negative_update_rejection_reason=rejection_reason,
            travel_time_s=max(0.0, now - self._observation_checkpoint_time_s),
            travel_distance_m=max(
                0.0,
                self._odometry_distance_m
                - self._observation_checkpoint_distance_m,
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
                "observation_trigger": trigger,
                "rgb_available": "rgb" in fresh,
                "depth_available": "depth" in fresh,
                "point_cloud_available": "point_cloud" in fresh,
                "camera_info_available": "camera_info" in fresh,
                "detections_available": "detections" in fresh,
                "sensor_header_timestamps_s": {
                    name: item[1] for name, item in fresh.items()
                },
                "projected_ground_point_count": len(visible_ground_points),
                "projection_support_status": (
                    "sufficient"
                    if len(visible_ground_points) >= minimum_projected_points
                    else "insufficient"
                ),
                "geometric_occlusion_confirmed": False,
            },
        )
        observation = self._adapter.adapt(frame, self._commanded_viewpoint)
        replan_reason = None
        replan_diagnostics: Dict[str, object] = {}
        suspect_snapshot_path = None
        verification_snapshot_path = None
        verification_in_progress = bool(
            self._session.pending_policy_metadata.get("verification_mode")
        )
        suspect_inspection_in_progress = bool(
            self._session.pending_policy_metadata.get(
                "transit_suspect_inspection_mode"
            )
        )
        if transit:
            prior_belief = dict(self._session.state.belief)
            prior_utility = _selected_utility(
                self._session.pending_policy_metadata
            )
            new_cells = _new_visible_cell_count(
                self._session.state.observed_cell_quality,
                observation.visible_cell_ids,
                observation.observation_quality,
            )
            if (
                trigger == "negative_detection_in_transit"
                and new_cells
                < int(self._parameter("transit_negative_min_new_cells"))
            ):
                return False
            state = self._session.record_transit_observation(
                observation,
                replan=False,
            )
            belief_update = self._session.belief_updates[-1]
            next_utility = _best_policy_utility(self._session.policy, state)
            reward_change = _relative_reward_change(prior_utility, next_utility)
            last_replan_time = state.policy_metadata.get(
                "last_replan_timestamp_s"
            )
            time_since_replan = (
                max(0.0, now - float(last_replan_time))
                if last_replan_time is not None
                else max(0.0, now - self._command_start_time_s)
            )
            trajectory_valid = _policy_viewpoint_is_viable(
                self._session.policy,
                state,
                self._commanded_viewpoint,
            )
            belief_total_variation = _belief_total_variation(
                prior_belief,
                state.belief,
            )
            replan_reason = _replan_reason(
                positive_detection=trigger == "positive_detection_in_transit",
                belief_total_variation=belief_total_variation,
                kl_divergence_nats=belief_update.kl_divergence_nats,
                trajectory_valid=trajectory_valid,
                expected_reward_change=reward_change,
                time_since_replan_s=time_since_replan,
                minimum_interval_s=float(
                    self._parameter("replan_min_interval_s")
                ),
                belief_total_variation_threshold=float(self._parameter(
                    "replan_belief_total_variation_threshold"
                )),
                kl_divergence_threshold_nats=float(self._parameter(
                    "replan_kl_divergence_threshold_nats"
                )),
                expected_reward_change_threshold=float(self._parameter(
                    "replan_expected_reward_change_threshold"
                )),
            )
            replan_diagnostics = {
                "belief_total_variation": belief_total_variation,
                "kl_divergence_nats": belief_update.kl_divergence_nats,
                "trajectory_valid": trajectory_valid,
                "prior_expected_reward": prior_utility,
                "next_expected_reward": next_utility,
                "expected_reward_change": reward_change,
                "time_since_replan_s": time_since_replan,
            }
            protected_action = _transit_replan_protected(
                trigger=trigger,
                verification_in_progress=verification_in_progress,
                suspect_inspection_in_progress=suspect_inspection_in_progress,
            )
            suspect = (
                None
                if protected_action
                else self._transit_occlusion_suspect(observation, state)
            )
            if protected_action:
                replan_diagnostics["replan_suppressed_for_protected_action"] = (
                    "verification"
                    if verification_in_progress
                    else "transit_suspect_inspection"
                )
                replan_reason = None
            elif suspect is not None:
                suspect_cell_id, suspect_belief = suspect
                suspect_snapshot_path = self._save_transit_suspect_snapshot(
                    now,
                    suspect_cell_id,
                    suspect_belief,
                )
                replan_reason = "transit_occlusion_suspect"
                replan_diagnostics.update({
                    "suspect_cell_id": suspect_cell_id,
                    "suspect_belief": suspect_belief,
                    "completed_suspect_inspections": (
                        self._transit_suspect_inspection_counts.get(
                            suspect_cell_id,
                            0,
                        )
                    ),
                    "suspect_timestamp_s": now,
                    "suspect_viewpoint_xy": (
                        observation.viewpoint.x,
                        observation.viewpoint.y,
                    ),
                    "suspect_snapshot_path": suspect_snapshot_path,
                    "negative_update_rejection_reason": (
                        observation.negative_update_rejection_reason
                    ),
                })
            if replan_reason is not None:
                self._session.request_replan(
                    replan_reason,
                    timestamp_s=now,
                    diagnostics=replan_diagnostics,
                )
                state = self._session.state
        else:
            new_cells = None
            decision = self._session.pending_policy_metadata
            inspected_suspect_cell = (
                str(decision.get("transit_suspect_cell_id"))
                if bool(decision.get("transit_suspect_inspection_mode"))
                and decision.get("transit_suspect_cell_id") is not None
                else None
            )
            if bool(decision.get("verification_mode")):
                verification_snapshot_path = self._save_verification_snapshot(
                    now,
                    observation,
                    decision,
                )
            state = self._session.record_observation(observation)
            if inspected_suspect_cell is not None:
                completed = (
                    self._transit_suspect_inspection_counts.get(
                        inspected_suspect_cell,
                        0,
                    )
                    + 1
                )
                self._transit_suspect_inspection_counts[
                    inspected_suspect_cell
                ] = completed
                self.get_logger().info(
                    "Completed transit suspect inspection "
                    f"cell={inspected_suspect_cell}; count={completed}/"
                    f"{int(self._parameter('transit_suspect_inspection_limit'))}"
                )
        self._write_trace("observation", {
            "step_index": state.step_index,
            "observation": observation.to_dict(),
            "belief": dict(state.belief),
            "policy_decision": (
                self._session.policy_decisions[-1]
                if self._session.policy_decisions else None
            ),
            "transit_replan": transit and replan_reason is not None,
            "replan_reason": replan_reason,
            "replan_diagnostics": replan_diagnostics,
            "new_visible_cell_count": new_cells,
            "transit_suspect_snapshot_path": suspect_snapshot_path,
            "verification_snapshot_path": verification_snapshot_path,
        })
        self.get_logger().info(
            f"Recorded {'transit evidence' if transit else 'viewpoint'} "
            f"{state.step_index} via {trigger}; "
            f"detections={len(observation.detections)}, "
            f"visible_cells={len(observation.visible_cell_ids)}, "
            f"ground_points={len(visible_ground_points)}, "
            f"quality={observation.observation_quality:.3f}, "
            f"visibility={observation.visibility_probability:.3f}, "
            f"negative_strength={observation.negative_update_strength:.3f}, "
            f"negative_rejection={observation.negative_update_rejection_reason}"
        )
        self._observation_checkpoint_time_s = now
        self._observation_checkpoint_distance_m = self._odometry_distance_m
        self._observation_checkpoint_battery = self._battery_percentage
        if not transit or replan_reason is not None or self._session.completed:
            self._commanded_viewpoint = None
            self._navigation_goal = None
            self._remaining_navigation_goals = ()
            self._settled_since_s = None
        if self._session.completed:
            self._publish_outcome()
        return True

    def _transit_occlusion_suspect(
        self,
        observation: object,
        state: object,
    ) -> Optional[Tuple[str, float]]:
        if (
            observation.negative_update_rejection_reason != "geometric_occlusion"
            or self._grid is None
            or not state.belief
        ):
            return None
        footprint_ids = {
            cell.cell_id
            for cell in self._grid.cells_within_radius(
                observation.viewpoint.x,
                observation.viewpoint.y,
                self._planning_footprint_radius_m,
            )
        }
        inspection_limit = int(
            self._parameter("transit_suspect_inspection_limit")
        )
        local_cells = tuple(
            cell_id
            for cell_id in footprint_ids
            if cell_id in state.belief
            and _transit_suspect_cell_is_available(
                cell_id,
                self._transit_suspect_inspection_counts,
                inspection_limit,
            )
        )
        if not local_cells:
            return None
        suspect_cell_id = min(
            local_cells,
            key=lambda cell_id: (-float(state.belief[cell_id]), cell_id),
        )
        suspect_belief = float(state.belief[suspect_cell_id])
        if suspect_belief < float(self._parameter("transit_suspect_min_belief")):
            return None
        return suspect_cell_id, suspect_belief

    def _save_transit_suspect_snapshot(
        self,
        timestamp_s: float,
        cell_id: str,
        belief: float,
    ) -> Optional[str]:
        directory = str(self._parameter("transit_suspect_snapshot_dir")).strip()
        limit = int(self._parameter("transit_suspect_snapshot_limit"))
        rgb_item = self._latest.get("rgb")
        if (
            not directory
            or rgb_item is None
            or self._transit_suspect_snapshot_count >= limit
        ):
            return None
        try:
            ppm = _rgb_image_to_ppm(rgb_item[0])
            output = Path(directory)
            output.mkdir(parents=True, exist_ok=True)
            self._transit_suspect_snapshot_count += 1
            stem = (
                f"suspect_{self._transit_suspect_snapshot_count:03d}_"
                f"{int(timestamp_s * 1000):013d}"
            )
            image_path = output / f"{stem}.ppm"
            metadata_path = output / f"{stem}.json"
            image_path.write_bytes(ppm)
            metadata_path.write_text(json.dumps({
                "timestamp_s": timestamp_s,
                "suspect_cell_id": cell_id,
                "suspect_belief": belief,
                "viewpoint": _viewpoint_dict(self._actual_viewpoint()),
                "image": str(image_path),
            }, indent=2, sort_keys=True), encoding="utf-8")
            return str(image_path)
        except (OSError, TypeError, ValueError) as error:
            self.get_logger().warning(f"Suspect snapshot skipped: {error}")
            return None

    def _save_verification_snapshot(
        self,
        timestamp_s: float,
        observation: object,
        decision: Mapping[str, object],
    ) -> Optional[str]:
        directory = str(self._parameter("verification_snapshot_dir")).strip()
        limit = int(self._parameter("verification_snapshot_limit"))
        rgb_item = self._latest.get("rgb")
        if (
            not directory
            or rgb_item is None
            or self._verification_snapshot_count >= limit
        ):
            return None
        try:
            ppm = _rgb_image_to_ppm(rgb_item[0])
            output = Path(directory)
            output.mkdir(parents=True, exist_ok=True)
            self._verification_snapshot_count += 1
            stem = (
                f"verification_{self._verification_snapshot_count:03d}_"
                f"{int(timestamp_s * 1000):013d}"
            )
            image_path = output / f"{stem}.ppm"
            metadata_path = output / f"{stem}.json"
            image_path.write_bytes(ppm)
            metadata_path.write_text(json.dumps({
                "timestamp_s": timestamp_s,
                "verification_cell_id": decision.get("verification_cell_id"),
                "viewpoint": _viewpoint_dict(observation.viewpoint),
                "detections": [asdict(item) for item in observation.detections],
                "visibility_probability": observation.visibility_probability,
                "negative_update_rejection_reason": (
                    observation.negative_update_rejection_reason
                ),
                "image": str(image_path),
            }, indent=2, sort_keys=True), encoding="utf-8")
            self.get_logger().info(
                "Saved double-check snapshot "
                f"{image_path}; detections={len(observation.detections)}"
            )
            return str(image_path)
        except (OSError, TypeError, ValueError) as error:
            self.get_logger().warning(f"Double-check snapshot skipped: {error}")
            return None

    def _can_record_positive_detection_in_transit(self) -> bool:
        if (
            self._transit_detection_recorded
            or not bool(self._parameter("record_first_positive_detection_in_transit"))
            or self._session is None
            or not self._required_sensors_are_fresh()
        ):
            return False
        maximum_skew = float(
            self._parameter("transit_detection_max_sensor_skew_s")
        )
        minimum_quality = max(
            float(self._parameter("minimum_observation_quality")),
            float(self._parameter("transit_detection_min_observation_quality")),
        )
        sensor_gate_passed = (
            self._dynamic_sensor_receipt_skew_s() <= maximum_skew
            and self._predicted_observation_quality()
            >= minimum_quality
        )
        if not sensor_gate_passed:
            return False
        criteria = self._session.state.task.success_criteria
        return _can_use_transit_detection_evidence(
            self._session.state.observations,
            self._target_detections(),
            minimum_confidence=criteria.min_confidence,
            maximum_localization_error_m=criteria.max_localization_error_m,
        )

    def _can_record_negative_detection_in_transit(self) -> bool:
        if (
            not bool(self._parameter("record_negative_observations_in_transit"))
            or self._session is None
            or self._commanded_viewpoint is None
            or not self._required_sensors_are_fresh()
            or self._is_settled_at(self._commanded_viewpoint)
        ):
            return False
        actual = self._actual_viewpoint()
        if actual.z < (
            float(self._parameter("flight_altitude_m"))
            - float(self._parameter("position_tolerance_m"))
        ):
            return False
        if (
            self._now_s() - self._observation_checkpoint_time_s
            < float(self._parameter("transit_negative_min_interval_s"))
            or self._odometry_distance_m - self._observation_checkpoint_distance_m
            < float(self._parameter("transit_negative_min_distance_m"))
            or self._predicted_observation_quality()
            < float(self._parameter("minimum_observation_quality"))
        ):
            return False
        criteria = self._session.state.task.success_criteria
        if _has_acceptable_detection(
            self._target_detections(),
            minimum_confidence=criteria.min_confidence,
            maximum_localization_error_m=criteria.max_localization_error_m,
        ):
            return False
        # A prior transit hit must not permanently disable later path sampling:
        # it may be the only hit and therefore still require an independent
        # confirmation.  Only a detection in the current frame suppresses the
        # negative update.
        return True

    def _dynamic_sensor_receipt_skew_s(self) -> float:
        names = ["odom", "rgb", "depth"]
        if bool(self._parameter("require_detections")):
            names.append("detections")
        if bool(self._parameter("require_point_cloud")):
            names.append("point_cloud")
        receipt_times = [self._latest[name][2] for name in names]
        return max(receipt_times) - min(receipt_times)

    def _time_budget_exhausted(self) -> bool:
        assert self._session is not None
        elapsed_time_s = self._session.state.elapsed_time_s
        if self._commanded_viewpoint is not None:
            elapsed_time_s += max(
                0.0,
                self._now_s() - self._observation_checkpoint_time_s,
            )
        return self._session.expire_time_budget(elapsed_time_s)

    def _predicted_observation_quality(self) -> float:
        receipt_times = [
            self._latest[name][2]
            for name in ("odom", "rgb", "depth")
        ]
        return _quality_after_sensor_skew(
            self._observation_quality(),
            max(receipt_times) - min(receipt_times),
            float(self._parameter("maximum_sensor_skew_s")),
        )

    def _required_sensors_are_fresh(self) -> bool:
        now = self._now_s()
        timeout = float(self._parameter("sensor_timeout_s"))
        required = ["odom", "rgb", "depth"]
        if bool(self._parameter("require_detections")):
            required.append("detections")
        if bool(self._parameter("require_point_cloud")):
            required.append("point_cloud")
        dynamic_inputs_are_fresh = all(
            name in self._latest and now - self._latest[name][2] <= timeout
            for name in required
        )
        # Camera calibration is static and may be bridged only once at startup.
        return dynamic_inputs_are_fresh and "camera_info" in self._latest

    def _visible_ground_points(
        self,
        fresh: Dict[str, Tuple[object, float, float]],
    ) -> Tuple[Tuple[float, float], ...]:
        item = fresh.get("point_cloud")
        if item is None or self._odom is None:
            return ()
        try:
            points = self._pointcloud_projector.project(
                item[0],
                self._odom.pose.pose,
            )
            origin_x = float(self._parameter("map_origin_x_m"))
            origin_y = float(self._parameter("map_origin_y_m"))
            return tuple((point[0] + origin_x, point[1] + origin_y) for point in points)
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
        position_error, speed = self._motion_errors(viewpoint)
        return (
            position_error <= float(self._parameter("position_tolerance_m"))
            and speed <= float(self._parameter("velocity_tolerance_mps"))
        )

    def _motion_errors(self, viewpoint: Viewpoint) -> Tuple[float, float]:
        actual = self._actual_viewpoint()
        position_error = math.sqrt(
            (actual.x - viewpoint.x) ** 2
            + (actual.y - viewpoint.y) ** 2
            + (actual.z - viewpoint.z) ** 2
        )
        velocity = self._odom.twist.twist.linear
        speed = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
        return position_error, speed

    def _log_progress(self, gate: str) -> None:
        now = self._now_s()
        interval = float(self._parameter("progress_log_interval_s"))
        if now >= self._last_progress_log_s and now - self._last_progress_log_s < interval:
            return
        self._last_progress_log_s = now
        assert self._commanded_viewpoint is not None
        assert self._navigation_goal is not None
        actual = self._actual_viewpoint()
        position_error, speed = self._motion_errors(self._navigation_goal)
        sensor_ages = {
            name: round(max(0.0, now - item[2]), 3)
            for name, item in self._latest.items()
            if name in {"odom", "rgb", "camera_info", "depth", "point_cloud", "detections"}
        }
        self.get_logger().info(
            f"Search gate={gate}; actual=({actual.x:.2f}, {actual.y:.2f}, {actual.z:.2f}); "
            f"target=({self._navigation_goal.x:.2f}, "
            f"{self._navigation_goal.y:.2f}, {self._navigation_goal.z:.2f}); "
            f"position_error_m={position_error:.2f}; speed_mps={speed:.2f}; "
            f"sensor_age_s={sensor_ages}"
        )

    def _actual_viewpoint(self) -> Viewpoint:
        pose = self._odom.pose.pose
        return Viewpoint(
            float(pose.position.x) + float(self._parameter("map_origin_x_m")),
            float(pose.position.y) + float(self._parameter("map_origin_y_m")),
            float(pose.position.z) + float(self._parameter("map_origin_z_m")),
            _yaw_from_quaternion(pose.orientation),
            pitch=0.0,
        )

    def _energy_used(self) -> float:
        if (
            self._observation_checkpoint_battery is None
            or self._battery_percentage is None
        ):
            return 0.0
        fraction = max(
            0.0,
            self._observation_checkpoint_battery - self._battery_percentage,
        )
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


def _rgb_image_to_ppm(message: object) -> bytes:
    """Encode one ROS RGB/BGR image as a dependency-free PPM snapshot."""
    width = int(message.width)
    height = int(message.height)
    step = int(message.step)
    encoding = str(message.encoding).lower()
    channels = 4 if encoding in {"rgba8", "bgra8"} else 3
    if encoding not in {"rgb8", "bgr8", "rgba8", "bgra8"}:
        raise ValueError(f"unsupported RGB snapshot encoding: {message.encoding}")
    if width <= 0 or height <= 0 or step < width * channels:
        raise ValueError("invalid RGB snapshot dimensions")
    raw = bytes(message.data)
    if len(raw) < step * height:
        raise ValueError("RGB snapshot data is truncated")
    rgb = bytearray(width * height * 3)
    output_index = 0
    for row in range(height):
        row_start = row * step
        for column in range(width):
            pixel = row_start + column * channels
            if encoding.startswith("rgb"):
                red, green, blue = raw[pixel:pixel + 3]
            else:
                blue, green, red = raw[pixel:pixel + 3]
            rgb[output_index:output_index + 3] = bytes((red, green, blue))
            output_index += 3
    return f"P6\n{width} {height}\n255\n".encode("ascii") + bytes(rgb)


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


def _quality_after_sensor_skew(
    base_quality: float,
    sensor_skew_s: float,
    maximum_sensor_skew_s: float,
) -> float:
    if maximum_sensor_skew_s <= 0:
        raise ValueError("maximum_sensor_skew_s must be positive")
    skew_ratio = min(1.0, max(0.0, sensor_skew_s) / maximum_sensor_skew_s)
    return max(0.0, min(1.0, base_quality)) * (1.0 - skew_ratio)


def _projection_visibility_probability(
    projected_ground_point_count: int,
    minimum_projected_ground_points: int,
) -> float:
    if minimum_projected_ground_points <= 0:
        raise ValueError("minimum_projected_ground_points must be positive")
    return min(
        1.0,
        max(0, projected_ground_point_count) / minimum_projected_ground_points,
    )


def _nadir_footprint_half_extents(
    *,
    altitude_m: float,
    horizontal_fov_rad: float,
    image_width_px: int,
    image_height_px: int,
    scale: float = 1.0,
) -> Tuple[float, float]:
    """Compute a conservative flat-ground footprint from pinhole intrinsics."""
    if altitude_m <= 0:
        raise ValueError("altitude_m must be positive")
    if not 0 < horizontal_fov_rad < math.pi:
        raise ValueError("horizontal_fov_rad must be within (0, pi)")
    if image_width_px <= 0 or image_height_px <= 0:
        raise ValueError("camera image dimensions must be positive")
    if not 0 < scale <= 1:
        raise ValueError("planning footprint scale must be within (0, 1]")
    half_width = altitude_m * math.tan(horizontal_fov_rad / 2.0)
    half_height = half_width * image_height_px / image_width_px
    return half_width * scale, half_height * scale


def _negative_update_rejection_reason(
    *,
    rgb_available: bool,
    depth_available: bool,
    point_cloud_available: bool,
    projected_ground_point_count: int,
    minimum_projected_ground_points: int,
    observation_quality: float,
    minimum_observation_quality: float,
) -> Optional[str]:
    if not rgb_available or not depth_available:
        return "insufficient_rgbd_observation"
    if not point_cloud_available or projected_ground_point_count <= 0:
        return "no_valid_point_projection"
    if projected_ground_point_count < minimum_projected_ground_points:
        return "insufficient_ground_projection"
    if observation_quality < minimum_observation_quality:
        return "insufficient_observation_quality"
    return None


def _belief_total_variation(
    prior: Mapping[str, float],
    posterior: Mapping[str, float],
) -> float:
    return 0.5 * sum(
        abs(float(posterior.get(key, 0.0)) - float(prior.get(key, 0.0)))
        for key in set(prior) | set(posterior)
    )


def _semantic_visibility_probability(labels: Iterable[str]) -> float:
    normalized = " ".join(_normalize_label(label) for label in labels)
    if any(token in normalized for token in ("woodland", "forest", "vegetation")):
        return 0.35
    if any(token in normalized for token in ("passage", "alley", "corridor")):
        return 0.5
    if any(token in normalized for token in ("buildingfrontage", "streetedge")):
        return 0.65
    if any(token in normalized for token in ("road", "parking", "plaza", "open")):
        return 0.9
    return 0.75


def _semantic_risk_score(labels: Iterable[str]) -> float:
    normalized = " ".join(_normalize_label(label) for label in labels)
    if any(token in normalized for token in ("woodland", "forest", "vegetation")):
        return 0.7
    if any(token in normalized for token in ("passage", "alley", "corridor")):
        return 0.5
    if any(token in normalized for token in ("building", "frontage")):
        return 0.35
    return 0.1


def _selected_utility(metadata: Mapping[str, object]) -> Optional[float]:
    score = metadata.get("selected_viewpoint_score")
    if not isinstance(score, Mapping):
        return None
    value = score.get("utility")
    if value is None:
        return None
    utility = float(value)
    return utility if math.isfinite(utility) else None


def _best_policy_utility(policy: object, state: object) -> Optional[float]:
    viewpoint = policy.select_next(state)
    if viewpoint is None:
        return None
    return _selected_utility(policy.decision_metadata(state, viewpoint))


def _relative_reward_change(
    previous: Optional[float],
    current: Optional[float],
) -> float:
    if previous is None or current is None:
        return 0.0
    return abs(current - previous) / max(abs(previous), 1e-9)


def _policy_viewpoint_is_viable(
    policy: object,
    state: object,
    viewpoint: Viewpoint,
) -> bool:
    predicate = getattr(policy, "is_viewpoint_viable", None)
    if predicate is not None:
        return bool(predicate(state, viewpoint))
    return any(item.key == viewpoint.key for item in policy.plan(state))


def _replan_reason(
    *,
    positive_detection: bool,
    belief_total_variation: float,
    kl_divergence_nats: float,
    trajectory_valid: bool,
    expected_reward_change: float,
    time_since_replan_s: float,
    minimum_interval_s: float,
    belief_total_variation_threshold: float,
    kl_divergence_threshold_nats: float,
    expected_reward_change_threshold: float,
) -> Optional[str]:
    if positive_detection:
        return "positive_detection"
    if not trajectory_valid:
        return "trajectory_invalid"
    if time_since_replan_s < minimum_interval_s:
        return None
    if belief_total_variation >= belief_total_variation_threshold:
        return "belief_distribution_changed"
    if kl_divergence_nats >= kl_divergence_threshold_nats:
        return "kl_divergence_exceeded"
    if expected_reward_change >= expected_reward_change_threshold:
        return "expected_reward_changed"
    return None


def _transit_replan_protected(
    *,
    trigger: str,
    verification_in_progress: bool,
    suspect_inspection_in_progress: bool,
) -> bool:
    """Keep a committed inspection from being reset by more blocked frames.

    Positive evidence may interrupt an occlusion inspection so the policy can
    immediately schedule exact verification. Verification itself remains
    protected because it is already the response to positive evidence.
    """
    if verification_in_progress:
        return True
    return (
        suspect_inspection_in_progress
        and trigger != "positive_detection_in_transit"
    )


def _transit_suspect_cell_is_available(
    cell_id: str,
    inspection_counts: Mapping[str, int],
    inspection_limit: int,
) -> bool:
    """Bound repeated occlusion recovery across separate replan events."""
    return (
        inspection_limit > 0
        and int(inspection_counts.get(cell_id, 0)) < inspection_limit
    )


def _can_use_transit_detection_evidence(
    prior_observations: Iterable[object],
    current_detections: Iterable[TargetDetection],
    *,
    minimum_confidence: float,
    maximum_localization_error_m: Optional[float],
) -> bool:
    """Accept the first reliable positive on any flight leg.

    Once positive evidence exists, the next confirmation remains gated on a
    settled viewpoint so adjacent frames cannot satisfy both confirmations.
    """

    prior_positive = any(
        _has_acceptable_detection(
            observation.detections,
            minimum_confidence=minimum_confidence,
            maximum_localization_error_m=maximum_localization_error_m,
        )
        for observation in prior_observations
    )
    return not prior_positive and _has_acceptable_detection(
        current_detections,
        minimum_confidence=minimum_confidence,
        maximum_localization_error_m=maximum_localization_error_m,
    )


def _has_acceptable_detection(
    detections: Iterable[TargetDetection],
    *,
    minimum_confidence: float,
    maximum_localization_error_m: Optional[float],
) -> bool:
    for detection in detections:
        if detection.confidence < minimum_confidence:
            continue
        if maximum_localization_error_m is None:
            return True
        error = detection.attributes.get("localization_error_m")
        if error is not None and float(error) <= maximum_localization_error_m:
            return True
    return False


def _new_visible_cell_count(
    observed_cell_quality: Mapping[str, float],
    visible_cell_ids: Iterable[str],
    observation_quality: float,
) -> int:
    return sum(
        observation_quality > observed_cell_quality.get(cell_id, 0.0)
        for cell_id in set(visible_cell_ids)
    )


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
