import math
import unittest

from geometry_msgs.msg import PoseStamped

from gsi_search_bridge.mavros_offboard_controller import _advance_horizontal_setpoint


class HorizontalSetpointRampTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
