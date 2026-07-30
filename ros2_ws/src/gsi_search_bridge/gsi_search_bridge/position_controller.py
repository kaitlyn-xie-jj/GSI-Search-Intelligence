"""Generic proportional 3D pose-to-velocity controller for an initial UAV model."""

from __future__ import annotations

import math
from typing import Optional

import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data


class PositionController(Node):
    """Translate map-frame pose goals into body-frame velocity commands."""

    def __init__(self) -> None:
        super().__init__("gsi_position_controller")
        defaults = {
            "odom_topic": "/uav/odom",
            "goal_pose_topic": "/gsi/uav/goal_pose",
            "command_velocity_topic": "/uav/cmd_vel",
            "position_gain": 0.8,
            "yaw_gain": 1.0,
            "maximum_xy_speed_mps": 5.0,
            "maximum_z_speed_mps": 2.0,
            "maximum_yaw_rate_rad_s": 1.0,
            "position_tolerance_m": 0.3,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        self._odom: Optional[Odometry] = None
        self._goal: Optional[PoseStamped] = None
        self._publisher = self.create_publisher(
            Twist,
            self._parameter("command_velocity_topic"),
            10,
        )
        self.create_subscription(
            Odometry,
            self._parameter("odom_topic"),
            self._on_odom,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            PoseStamped,
            self._parameter("goal_pose_topic"),
            self._on_goal,
            10,
        )
        self.create_timer(0.05, self._tick)

    def _on_odom(self, message: Odometry) -> None:
        self._odom = message

    def _on_goal(self, message: PoseStamped) -> None:
        self._goal = message

    def _tick(self) -> None:
        command = Twist()
        if self._odom is None or self._goal is None:
            self._publisher.publish(command)
            return
        current = self._odom.pose.pose
        goal = self._goal.pose
        dx = float(goal.position.x - current.position.x)
        dy = float(goal.position.y - current.position.y)
        dz = float(goal.position.z - current.position.z)
        distance = math.sqrt(dx * dx + dy * dy + dz * dz)
        if distance <= float(self._parameter("position_tolerance_m")):
            self._publisher.publish(command)
            return

        yaw = _yaw_from_quaternion(current.orientation)
        goal_yaw = _yaw_from_quaternion(goal.orientation)
        body_x = math.cos(yaw) * dx + math.sin(yaw) * dy
        body_y = -math.sin(yaw) * dx + math.cos(yaw) * dy
        gain = float(self._parameter("position_gain"))
        maximum_xy = float(self._parameter("maximum_xy_speed_mps"))
        command.linear.x = gain * body_x
        command.linear.y = gain * body_y
        horizontal_speed = math.hypot(command.linear.x, command.linear.y)
        if horizontal_speed > maximum_xy:
            scale = maximum_xy / horizontal_speed
            command.linear.x *= scale
            command.linear.y *= scale
        command.linear.z = _clamp(
            gain * dz,
            float(self._parameter("maximum_z_speed_mps")),
        )
        command.angular.z = _clamp(
            float(self._parameter("yaw_gain"))
            * _wrapped_angle_difference(goal_yaw, yaw),
            float(self._parameter("maximum_yaw_rate_rad_s")),
        )
        self._publisher.publish(command)

    def _parameter(self, name: str):
        return self.get_parameter(name).value


def _yaw_from_quaternion(quaternion: object) -> float:
    return math.atan2(
        2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y),
        1.0 - 2.0 * (quaternion.y ** 2 + quaternion.z ** 2),
    )


def _wrapped_angle_difference(first: float, second: float) -> float:
    return math.atan2(math.sin(first - second), math.cos(first - second))


def _clamp(value: float, maximum_magnitude: float) -> float:
    return max(-maximum_magnitude, min(maximum_magnitude, value))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PositionController()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
