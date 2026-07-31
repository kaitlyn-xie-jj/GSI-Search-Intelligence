import math
import unittest

from gsi_search_bridge.stability import (
    classify_log_errors,
    evaluate_samples,
    memory_slope_mib_per_minute,
    parse_memory_mib,
)


class StabilityTests(unittest.TestCase):
    def test_parse_memory_mib(self):
        self.assertEqual(parse_memory_mib("512MiB / 16GiB"), 512.0)
        self.assertEqual(parse_memory_mib("1.5GiB / 16GiB"), 1536.0)

    def test_linear_memory_slope(self):
        samples = [
            {"elapsed_s": 0, "container_memory_mib": 1000},
            {"elapsed_s": 60, "container_memory_mib": 1010},
            {"elapsed_s": 120, "container_memory_mib": 1020},
        ]
        self.assertAlmostEqual(memory_slope_mib_per_minute(samples), 10.0)
        self.assertTrue(math.isnan(memory_slope_mib_per_minute(samples[:1])))

    def test_healthy_run_passes(self):
        samples = [
            {
                "elapsed_s": elapsed,
                "container_memory_mib": 1200 + index * 20,
                "container_cpu_percent": 80,
                "container_running": True,
                "oom_killed": False,
                "gazebo_processes": 1,
                "px4_processes": 1,
                "required_topics_present": True,
                "required_topics_active": index == 0,
                "mavros_connected": True,
            }
            for index, elapsed in enumerate((0, 600, 1200, 1500))
        ]
        result = evaluate_samples(
            samples,
            duration_s=1500,
            interval_s=10,
            max_memory_mib=8192,
            max_growth_mib=2560,
            max_slope_mib_per_minute=100,
            require_mavros=True,
            require_flight=False,
            min_observations=2,
            critical_error_count=0,
            max_critical_errors=0,
        )
        self.assertEqual(result["verdict"], "PASS")

    def test_resource_growth_and_sensor_loss_fail(self):
        samples = [
            {
                "elapsed_s": 0,
                "container_memory_mib": 1000,
                "container_running": True,
                "oom_killed": False,
                "gazebo_processes": 1,
                "px4_processes": 1,
                "required_topics_present": True,
                "required_topics_active": False,
            },
            {
                "elapsed_s": 1500,
                "container_memory_mib": 9000,
                "container_running": True,
                "oom_killed": False,
                "gazebo_processes": 1,
                "px4_processes": 1,
                "required_topics_present": False,
                "required_topics_active": False,
            },
        ]
        result = evaluate_samples(
            samples,
            duration_s=1500,
            interval_s=10,
            max_memory_mib=8192,
            max_growth_mib=2560,
            max_slope_mib_per_minute=100,
            require_mavros=False,
            require_flight=False,
            min_observations=2,
            critical_error_count=2,
            max_critical_errors=0,
        )
        self.assertEqual(result["verdict"], "FAIL")
        self.assertGreaterEqual(len(result["failures"]), 4)

    def test_flight_gate_requires_bounded_offboard_observations(self):
        samples = [
            {
                "elapsed_s": 0,
                "container_memory_mib": 1200,
                "container_running": True,
                "oom_killed": False,
                "gazebo_processes": 1,
                "px4_processes": 1,
                "required_topics_present": True,
                "required_topics_active": True,
                "mavros_connected": True,
                "mavros_armed": True,
                "mavros_mode": "OFFBOARD",
                "mavros_x_m": -15.0,
                "mavros_y_m": -15.0,
                "mavros_z_m": 12.0,
                "within_flight_bounds": True,
                "observation_count": 2,
            },
            {
                "elapsed_s": 1500,
                "container_memory_mib": 1300,
                "container_running": True,
                "oom_killed": False,
                "gazebo_processes": 1,
                "px4_processes": 1,
                "required_topics_present": True,
                "required_topics_active": False,
                "mavros_connected": True,
                "mavros_armed": True,
                "mavros_mode": "OFFBOARD",
                "mavros_x_m": -25.0,
                "mavros_y_m": -15.0,
                "mavros_z_m": 12.0,
                "within_flight_bounds": True,
                "observation_count": 3,
            },
        ]
        result = evaluate_samples(
            samples,
            duration_s=1500,
            interval_s=10,
            max_memory_mib=8192,
            max_growth_mib=2560,
            max_slope_mib_per_minute=100,
            require_mavros=True,
            require_flight=True,
            min_observations=2,
            critical_error_count=0,
            max_critical_errors=0,
        )
        self.assertEqual(result["verdict"], "PASS")

        samples[-1]["within_flight_bounds"] = False
        result = evaluate_samples(
            samples,
            duration_s=1500,
            interval_s=10,
            max_memory_mib=8192,
            max_growth_mib=2560,
            max_slope_mib_per_minute=100,
            require_mavros=True,
            require_flight=True,
            min_observations=2,
            critical_error_count=0,
            max_critical_errors=0,
        )
        self.assertEqual(result["verdict"], "FAIL")
        self.assertIn(
            "UAV left the configured SearchWorld flight bounds",
            result["failures"],
        )

    def test_log_classification_keeps_timesync_as_warning(self):
        result = classify_log_errors(
            "TM: Time jump detected. Resetting time synchroniser.\n"
            "IMU timestamp invalid\n"
        )
        self.assertEqual(result["counts"]["imu_timestamp"], 1)
        self.assertEqual(result["warning_counts"]["mavros_timesync_reset"], 1)
        self.assertEqual(result["total"], 1)


if __name__ == "__main__":
    unittest.main()
