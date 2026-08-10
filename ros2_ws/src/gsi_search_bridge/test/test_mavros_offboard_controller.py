import math
import unittest

from geometry_msgs.msg import Pose, PoseStamped
from nav_msgs.msg import Odometry

from gsi_search_bridge.mavros_offboard_controller import (
    _advance_horizontal_setpoint,
    _initial_odometry_is_plausible,
    _limit_horizontal_setpoint_lead,
)


class HorizontalSetpointRampTests(unittest.TestCase):
    def test_rejects_stale_initial_estimator_altitude(self):
        odometry = Odometry()
        odometry.pose.pose.position.z = 81.38

        self.assertFalse(_initial_odometry_is_plausible(
            odometry,
            expected_position=(0.0, 0.0, 0.25),
            horizontal_tolerance_m=2.0,
            vertical_tolerance_m=1.0,
            maximum_speed_mps=1.0,
        ))

    def test_accepts_stationary_odometry_at_gazebo_spawn(self):
        odometry = Odometry()
        odometry.pose.pose.position.x = 0.1
        odometry.pose.pose.position.y = -0.1
        odometry.pose.pose.position.z = 0.3

        self.assertTrue(_initial_odometry_is_plausible(
            odometry,
            expected_position=(0.0, 0.0, 0.25),
            horizontal_tolerance_m=2.0,
            vertical_tolerance_m=1.0,
            maximum_speed_mps=1.0,
        ))

    def test_limits_diagonal_step_without_changing_target_altitude(self):
        current = PoseStamped()
        target = PoseStamped()
        target.pose.position.x = 3.0
        target.pose.position.y = 4.0
        target.pose.position.z = 12.0
        target.pose.orientation.w = 1.0

        _advance_horizontal_setpoint(current, target, maximum_step_m=1.0)

        self.assertAlmostEqual(current.pose.position.x, 0.6)
        self.assertAlmostEqual(current.pose.position.y, 0.8)
        self.assertAlmostEqual(current.pose.position.z, 12.0)
        self.assertAlmostEqual(
            math.hypot(current.pose.position.x, current.pose.position.y),
            1.0,
        )

    def test_snaps_to_goal_inside_one_step(self):
        current = PoseStamped()
        target = PoseStamped()
        target.pose.position.x = 0.2
        target.pose.position.y = -0.1

        _advance_horizontal_setpoint(current, target, maximum_step_m=1.0)

        self.assertEqual(current.pose.position.x, target.pose.position.x)
        self.assertEqual(current.pose.position.y, target.pose.position.y)

    def test_limits_setpoint_distance_from_actual_vehicle(self):
        current = PoseStamped()
        current.pose.position.x = 5.0
        target = PoseStamped()
        target.pose.position.x = 10.0
        actual = Pose()

        _limit_horizontal_setpoint_lead(
            current,
            target,
            actual,
            maximum_lead_m=1.0,
        )

        self.assertAlmostEqual(current.pose.position.x, 1.0)
        self.assertAlmostEqual(current.pose.position.y, 0.0)

    def test_lead_limiter_snaps_to_goal_near_actual_vehicle(self):
        current = PoseStamped()
        current.pose.position.x = 4.0
        target = PoseStamped()
        target.pose.position.x = 0.5
        target.pose.position.y = -0.25
        actual = Pose()

        _limit_horizontal_setpoint_lead(
            current,
            target,
            actual,
            maximum_lead_m=1.0,
        )

        self.assertEqual(current.pose.position.x, target.pose.position.x)
        self.assertEqual(current.pose.position.y, target.pose.position.y)

    def test_lead_limiter_rejects_nonpositive_limit(self):
        with self.assertRaisesRegex(ValueError, "maximum_lead_m"):
            _limit_horizontal_setpoint_lead(
                PoseStamped(),
                PoseStamped(),
                Pose(),
                maximum_lead_m=0.0,
            )


if __name__ == "__main__":
    unittest.main()
