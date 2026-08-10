"""MAVROS Offboard adapter from GSI map-frame goals to PX4 setpoints."""

from __future__ import annotations

import copy
import math
from typing import Optional

import rclpy
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from std_msgs.msg import String


class MavrosOffboardController(Node):
    """Stream ENU pose setpoints and manage the PX4 Offboard handshake."""

    def __init__(self) -> None:
        super().__init__("gsi_mavros_offboard_controller")
        defaults = {
            "state_topic": "/mavros/state",
            "odom_topic": "/mavros/local_position/odom",
            "goal_pose_topic": "/gsi/uav/goal_pose",
            "setpoint_topic": "/mavros/setpoint_position/local",
            "arming_service": "/mavros/cmd/arming",
            "set_mode_service": "/mavros/set_mode",
            "map_frame": "map",
            "map_origin_x_m": 0.0,
            "map_origin_y_m": 0.0,
            "map_origin_z_m": 0.0,
            "setpoint_rate_hz": 20.0,
            "prestream_setpoint_count": 40,
            "request_interval_s": 2.0,
            "auto_offboard": True,
            "auto_arm": True,
            "staged_takeoff": True,
            "takeoff_altitude_tolerance_m": 0.3,
            "horizontal_setpoint_speed_mps": 0.0,
            "horizontal_setpoint_max_lead_m": 0.0,
            "initial_pose_expected_x_m": 0.0,
            "initial_pose_expected_y_m": 0.0,
            "initial_pose_expected_z_m": 0.25,
            "initial_pose_horizontal_tolerance_m": 2.0,
            "initial_pose_vertical_tolerance_m": 1.0,
            "initial_pose_max_speed_mps": 1.0,
            "initial_pose_required_samples": 100,
            "safety_area_min_x_m": -1.0e9,
            "safety_area_min_y_m": -1.0e9,
            "safety_area_max_x_m": 1.0e9,
            "safety_area_max_y_m": 1.0e9,
            "safety_area_margin_m": 5.0,
            "safety_status_topic": "/gsi/uav/safety_status",
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

        rate_hz = float(self._parameter("setpoint_rate_hz"))
        if rate_hz < 2.0:
            raise ValueError("setpoint_rate_hz must be at least 2 Hz for Offboard")
        self._setpoint_period_s = 1.0 / rate_hz

        mavros_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self._state = State()
        self._odom: Optional[Odometry] = None
        self._setpoint: Optional[PoseStamped] = None
        self._pending_goal: Optional[PoseStamped] = None
        self._initial_pose: Optional[PoseStamped] = None
        self._prestream_count = 0
        self._last_request_s = float("-inf")
        self._takeoff_complete = not bool(self._parameter("staged_takeoff"))
        self._last_reported_state = None
        self._last_goal_signature = None
        self._last_progress_report_s = float("-inf")
        self._initial_pose_valid_samples = 0
        self._offboard_achieved = False
        self._safety_latched = False

        self._publisher = self.create_publisher(
            PoseStamped,
            self._parameter("setpoint_topic"),
            mavros_qos,
        )
        self.create_subscription(
            State,
            self._parameter("state_topic"),
            self._on_state,
            mavros_qos,
        )
        self.create_subscription(
            Odometry,
            self._parameter("odom_topic"),
            self._on_odom,
            mavros_qos,
        )
        self.create_subscription(
            PoseStamped,
            self._parameter("goal_pose_topic"),
            self._on_goal,
            10,
        )
        self._arming_client = self.create_client(
            CommandBool,
            self._parameter("arming_service"),
        )
        self._mode_client = self.create_client(
            SetMode,
            self._parameter("set_mode_service"),
        )
        self._safety_publisher = self.create_publisher(
            String,
            self._parameter("safety_status_topic"),
            10,
        )
        self.create_timer(1.0 / rate_hz, self._publish_setpoint)
        self.create_timer(0.2, self._manage_offboard)
        self.get_logger().info(
            "Waiting for MAVROS connection and local odometry (map frame is ENU)"
        )

    def _on_state(self, message: State) -> None:
        self._state = message
        if message.armed and message.mode == "OFFBOARD":
            self._offboard_achieved = True
        elif (
            self._offboard_achieved
            and message.armed
            and message.mode != "OFFBOARD"
        ):
            self._latch_safety_stop(
                f"OFFBOARD mode lost to {message.mode or '<unset>'}"
            )
        state_key = (message.connected, message.armed, message.mode)
        if state_key != self._last_reported_state:
            self.get_logger().info(
                f"MAVROS connected={message.connected}, armed={message.armed}, "
                f"mode={message.mode or '<unset>'}"
            )
            self._last_reported_state = state_key

    def _on_odom(self, message: Odometry) -> None:
        self._odom = message
        self._report_progress()
        initial_odometry_is_plausible = self._initial_odometry_is_plausible(message)
        if (
            self._initial_pose is not None
            and not self._state.armed
            and not initial_odometry_is_plausible
        ):
            self._reset_initial_pose_after_unstable_odometry()
        if self._initial_pose is not None:
            violation = self._flight_safety_violation(message)
            if violation is not None:
                self._latch_safety_stop(violation)
        if self._safety_latched:
            return
        if self._initial_pose is not None:
            self._update_takeoff_state()
            return

        if not initial_odometry_is_plausible:
            self._initial_pose_valid_samples = 0
            return
        self._initial_pose_valid_samples += 1
        if self._initial_pose_valid_samples < int(
            self._parameter("initial_pose_required_samples")
        ):
            return

        initial = PoseStamped()
        initial.header.frame_id = str(self._parameter("map_frame"))
        initial.pose = copy.deepcopy(message.pose.pose)
        if _quaternion_norm(initial.pose.orientation) < 1e-6:
            initial.pose.orientation.w = 1.0
        self._initial_pose = initial
        self._setpoint = copy.deepcopy(initial)
        self._apply_pending_goal()
        self.get_logger().info(
            "Received stable initial ENU pose: "
            f"x={initial.pose.position.x:.2f}, "
            f"y={initial.pose.position.y:.2f}, "
            f"z={initial.pose.position.z:.2f}"
        )

    def _initial_odometry_is_plausible(self, message: Odometry) -> bool:
        return _initial_odometry_is_plausible(
            message,
            expected_position=(
                float(self._parameter("initial_pose_expected_x_m")),
                float(self._parameter("initial_pose_expected_y_m")),
                float(self._parameter("initial_pose_expected_z_m")),
            ),
            horizontal_tolerance_m=float(
                self._parameter("initial_pose_horizontal_tolerance_m")
            ),
            vertical_tolerance_m=float(
                self._parameter("initial_pose_vertical_tolerance_m")
            ),
            maximum_speed_mps=float(
                self._parameter("initial_pose_max_speed_mps")
            ),
        )

    def _reset_initial_pose_after_unstable_odometry(self) -> None:
        self.get_logger().warning(
            "Initial odometry became unstable before arming; withdrawing "
            "setpoints and restarting the stability check"
        )
        self._initial_pose = None
        self._setpoint = None
        self._initial_pose_valid_samples = 0
        self._prestream_count = 0
        self._takeoff_complete = not bool(self._parameter("staged_takeoff"))

    def _on_goal(self, message: PoseStamped) -> None:
        if self._safety_latched:
            self.get_logger().error("Rejected GSI goal after safety stop")
            return
        expected_frame = str(self._parameter("map_frame"))
        if message.header.frame_id and message.header.frame_id != expected_frame:
            self.get_logger().error(
                f"Rejected goal in frame '{message.header.frame_id}'; "
                f"expected '{expected_frame}'"
            )
            return
        if not _pose_is_finite(message):
            self.get_logger().error("Rejected non-finite GSI goal")
            return

        goal = copy.deepcopy(message)
        goal.header.frame_id = expected_frame
        if _quaternion_norm(goal.pose.orientation) < 1e-6:
            goal.pose.orientation.w = 1.0
        local_goal = copy.deepcopy(goal)
        local_goal.pose.position.x -= float(self._parameter("map_origin_x_m"))
        local_goal.pose.position.y -= float(self._parameter("map_origin_y_m"))
        local_goal.pose.position.z -= float(self._parameter("map_origin_z_m"))
        signature = _goal_signature(goal)
        goal_changed = signature != self._last_goal_signature
        self._pending_goal = local_goal
        if goal_changed:
            self._apply_pending_goal()
            self.get_logger().info(
                "Accepted GSI ENU goal: "
                f"x={goal.pose.position.x:.2f}, "
                f"y={goal.pose.position.y:.2f}, "
                f"z={goal.pose.position.z:.2f}"
            )
            self._last_goal_signature = signature

    def _apply_pending_goal(self) -> None:
        if self._pending_goal is None or self._initial_pose is None:
            return
        if self._takeoff_complete:
            if float(self._parameter("horizontal_setpoint_speed_mps")) <= 0.0:
                self._setpoint = copy.deepcopy(self._pending_goal)
            elif self._odom is not None:
                # Start each new ramp from the vehicle, not from a setpoint that
                # may still be far ahead on the previous search leg.
                self._setpoint.pose = copy.deepcopy(self._odom.pose.pose)
                self._setpoint.pose.orientation = copy.deepcopy(
                    self._pending_goal.pose.orientation
                )
            return

        takeoff = copy.deepcopy(self._initial_pose)
        takeoff.pose.position.z = max(
            takeoff.pose.position.z,
            self._pending_goal.pose.position.z,
        )
        self._setpoint = takeoff

    def _update_takeoff_state(self) -> None:
        if (
            self._takeoff_complete
            or self._pending_goal is None
            or self._odom is None
        ):
            return
        target_z = max(
            self._initial_pose.pose.position.z,
            self._pending_goal.pose.position.z,
        )
        altitude_error = abs(self._odom.pose.pose.position.z - target_z)
        if altitude_error <= float(
            self._parameter("takeoff_altitude_tolerance_m")
        ):
            self._takeoff_complete = True
            if float(self._parameter("horizontal_setpoint_speed_mps")) <= 0.0:
                self._setpoint = copy.deepcopy(self._pending_goal)
            self.get_logger().info(
                "Staged takeoff complete; releasing horizontal search goals"
            )

    def _publish_setpoint(self) -> None:
        if not self._state.connected or self._setpoint is None:
            return
        if self._takeoff_complete and self._pending_goal is not None:
            speed = float(self._parameter("horizontal_setpoint_speed_mps"))
            if speed > 0.0:
                _advance_horizontal_setpoint(
                    self._setpoint,
                    self._pending_goal,
                    speed * self._setpoint_period_s,
                )
                maximum_lead = float(
                    self._parameter("horizontal_setpoint_max_lead_m")
                )
                if maximum_lead > 0.0 and self._odom is not None:
                    _limit_horizontal_setpoint_lead(
                        self._setpoint,
                        self._pending_goal,
                        self._odom.pose.pose,
                        maximum_lead,
                    )
        self._setpoint.header.stamp = self.get_clock().now().to_msg()
        self._publisher.publish(self._setpoint)
        self._prestream_count += 1

    def _manage_offboard(self) -> None:
        if self._safety_latched:
            return
        if not self._state.connected or self._odom is None:
            return
        if self._prestream_count < int(self._parameter("prestream_setpoint_count")):
            return
        now_s = self.get_clock().now().nanoseconds / 1e9
        if now_s - self._last_request_s < float(
            self._parameter("request_interval_s")
        ):
            return

        if bool(self._parameter("auto_offboard")) and self._state.mode != "OFFBOARD":
            if not self._mode_client.service_is_ready():
                self.get_logger().warning("Waiting for /mavros/set_mode")
                self._last_request_s = now_s
                return
            request = SetMode.Request()
            request.custom_mode = "OFFBOARD"
            self._mode_client.call_async(request).add_done_callback(
                self._on_mode_response
            )
            self._last_request_s = now_s
            return

        if bool(self._parameter("auto_arm")) and not self._state.armed:
            if not self._arming_client.service_is_ready():
                self.get_logger().warning("Waiting for /mavros/cmd/arming")
                self._last_request_s = now_s
                return
            request = CommandBool.Request()
            request.value = True
            self._arming_client.call_async(request).add_done_callback(
                self._on_arm_response
            )
            self._last_request_s = now_s

    def _flight_safety_violation(self, message: Odometry) -> Optional[str]:
        position = message.pose.pose.position
        map_x = float(position.x) + float(self._parameter("map_origin_x_m"))
        map_y = float(position.y) + float(self._parameter("map_origin_y_m"))
        margin = float(self._parameter("safety_area_margin_m"))
        if not (
            float(self._parameter("safety_area_min_x_m")) - margin
            <= map_x
            <= float(self._parameter("safety_area_max_x_m")) + margin
            and float(self._parameter("safety_area_min_y_m")) - margin
            <= map_y
            <= float(self._parameter("safety_area_max_y_m")) + margin
        ):
            return f"map position ({map_x:.2f}, {map_y:.2f}) left safety bounds"
        return None

    def _latch_safety_stop(self, reason: str) -> None:
        if self._safety_latched:
            return
        self._safety_latched = True
        self._pending_goal = None
        if self._odom is not None:
            hold = PoseStamped()
            hold.header.frame_id = str(self._parameter("map_frame"))
            hold.pose = copy.deepcopy(self._odom.pose.pose)
            if _quaternion_norm(hold.pose.orientation) < 1e-6:
                hold.pose.orientation.w = 1.0
            self._setpoint = hold
        self.get_logger().error(
            f"SAFETY STOP: {reason}; automatic OFFBOARD/arming recovery disabled"
        )
        status = String()
        status.data = reason
        self._safety_publisher.publish(status)

    def _on_mode_response(self, future) -> None:
        try:
            response = future.result()
            if response is None or not response.mode_sent:
                self.get_logger().warning("PX4 rejected OFFBOARD mode request")
        except Exception as error:  # pragma: no cover - ROS future boundary
            self.get_logger().error(f"OFFBOARD request failed: {error}")

    def _on_arm_response(self, future) -> None:
        try:
            response = future.result()
            if response is None or not response.success:
                self.get_logger().warning("PX4 rejected arming request")
        except Exception as error:  # pragma: no cover - ROS future boundary
            self.get_logger().error(f"Arming request failed: {error}")

    def _parameter(self, name: str):
        return self.get_parameter(name).value

    def _report_progress(self) -> None:
        if self._pending_goal is None or self._setpoint is None or self._odom is None:
            return
        now_s = self.get_clock().now().nanoseconds / 1e9
        if now_s >= self._last_progress_report_s and now_s - self._last_progress_report_s < 5.0:
            return
        self._last_progress_report_s = now_s
        position = self._odom.pose.pose.position
        velocity = self._odom.twist.twist.linear
        speed = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)
        setpoint = self._setpoint.pose.position
        goal = self._pending_goal.pose.position
        self.get_logger().info(
            f"Offboard progress: local=({position.x:.2f}, {position.y:.2f}, {position.z:.2f}); "
            f"setpoint=({setpoint.x:.2f}, {setpoint.y:.2f}, {setpoint.z:.2f}); "
            f"goal=({goal.x:.2f}, {goal.y:.2f}, {goal.z:.2f}); "
            f"speed_mps={speed:.2f}; takeoff_complete={self._takeoff_complete}"
        )


def _pose_is_finite(message: PoseStamped) -> bool:
    values = (
        message.pose.position.x,
        message.pose.position.y,
        message.pose.position.z,
        message.pose.orientation.x,
        message.pose.orientation.y,
        message.pose.orientation.z,
        message.pose.orientation.w,
    )
    return all(math.isfinite(float(value)) for value in values)


def _initial_odometry_is_plausible(
    message: Odometry,
    *,
    expected_position: tuple[float, float, float],
    horizontal_tolerance_m: float,
    vertical_tolerance_m: float,
    maximum_speed_mps: float,
) -> bool:
    """Reject stale estimator state before it can become a takeoff setpoint."""
    if min(horizontal_tolerance_m, vertical_tolerance_m, maximum_speed_mps) < 0:
        raise ValueError("initial odometry tolerances must not be negative")
    position = message.pose.pose.position
    values = (position.x, position.y, position.z, *expected_position)
    if not all(math.isfinite(float(value)) for value in values):
        return False
    horizontal_error = math.hypot(
        float(position.x) - expected_position[0],
        float(position.y) - expected_position[1],
    )
    vertical_error = abs(float(position.z) - expected_position[2])
    return (
        horizontal_error <= horizontal_tolerance_m
        and vertical_error <= vertical_tolerance_m
        and _odometry_speed_mps(message) <= maximum_speed_mps
    )


def _odometry_speed_mps(message: Odometry) -> float:
    velocity = message.twist.twist.linear
    return math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)


def _quaternion_norm(quaternion: object) -> float:
    return math.sqrt(
        float(quaternion.x) ** 2
        + float(quaternion.y) ** 2
        + float(quaternion.z) ** 2
        + float(quaternion.w) ** 2
    )


def _goal_signature(message: PoseStamped):
    pose = message.pose
    return (
        float(pose.position.x),
        float(pose.position.y),
        float(pose.position.z),
        float(pose.orientation.x),
        float(pose.orientation.y),
        float(pose.orientation.z),
        float(pose.orientation.w),
    )


def _advance_horizontal_setpoint(
    current: PoseStamped,
    target: PoseStamped,
    maximum_step_m: float,
) -> None:
    """Move a streamed XY setpoint toward a distant goal without a position jump."""

    dx = float(target.pose.position.x - current.pose.position.x)
    dy = float(target.pose.position.y - current.pose.position.y)
    distance = math.hypot(dx, dy)
    if distance <= maximum_step_m or distance < 1e-9:
        current.pose.position.x = target.pose.position.x
        current.pose.position.y = target.pose.position.y
    else:
        scale = maximum_step_m / distance
        current.pose.position.x += dx * scale
        current.pose.position.y += dy * scale
    current.pose.position.z = target.pose.position.z
    current.pose.orientation = copy.deepcopy(target.pose.orientation)


def _limit_horizontal_setpoint_lead(
    current: PoseStamped,
    target: PoseStamped,
    actual_pose: object,
    maximum_lead_m: float,
) -> None:
    """Keep the virtual XY target close enough for PX4 to brake cleanly."""

    if maximum_lead_m <= 0.0:
        raise ValueError("maximum_lead_m must be positive")
    actual_x = float(actual_pose.position.x)
    actual_y = float(actual_pose.position.y)
    target_dx = float(target.pose.position.x) - actual_x
    target_dy = float(target.pose.position.y) - actual_y
    target_distance = math.hypot(target_dx, target_dy)
    if target_distance <= maximum_lead_m:
        current.pose.position.x = target.pose.position.x
        current.pose.position.y = target.pose.position.y
        return
    current_lead = math.hypot(
        float(current.pose.position.x) - actual_x,
        float(current.pose.position.y) - actual_y,
    )
    if current_lead <= maximum_lead_m:
        return
    scale = maximum_lead_m / target_distance
    current.pose.position.x = actual_x + target_dx * scale
    current.pose.position.y = actual_y + target_dy * scale


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MavrosOffboardController()
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
