import unittest

from modules.search_intelligence import (
    SearchGrid,
    SearchObservationAdapter,
    SearchSensorFrame,
    SearchTask,
    TargetDetection,
    Viewpoint,
)


class SearchSensorIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.task = SearchTask.from_skill_params({
            "task_id": "sensor-adapter",
            "area_token": "gazebo-area",
            "area": {
                "kind": "rectangle",
                "coords": [[0, 0], [40, 0], [40, 20], [0, 20]],
            },
            "target_token": "yellow-van",
        })
        self.grid = SearchGrid.from_task(self.task, resolution_m=20.0)
        self.commanded = Viewpoint(10.0, 10.0, 20.0, 0.0)

    def test_adapter_keeps_actual_pose_and_commanded_action_key(self):
        actual = Viewpoint(10.2, 9.9, 20.1, 0.05)
        frame = SearchSensorFrame(
            timestamp_s=4.0,
            viewpoint=actual,
            visible_ground_points_xy=((10.0, 10.0),),
        )

        observation = SearchObservationAdapter(self.grid).adapt(
            frame,
            self.commanded,
        )

        self.assertEqual(observation.viewpoint, actual)
        self.assertEqual(observation.action_viewpoint_key, self.commanded.key)
        self.assertEqual(observation.visible_cell_ids, ("gazebo-area:r0:c0",))
        self.assertGreater(observation.sensor_metadata["position_error_m"], 0.0)

    def test_adapter_filters_non_target_detections(self):
        frame = SearchSensorFrame(
            timestamp_s=4.0,
            viewpoint=self.commanded,
            detections=(
                TargetDetection("yellow van", 0.9),
                TargetDetection("person", 0.99),
            ),
        )

        observation = SearchObservationAdapter(
            self.grid,
            target_query="yellow-van",
            fallback_footprint_radius_m=1.0,
        ).adapt(frame, self.commanded)

        self.assertEqual(len(observation.detections), 1)
        self.assertEqual(observation.detections[0].label, "yellow van")

    def test_sensor_skew_reduces_observation_quality(self):
        frame = SearchSensorFrame(
            timestamp_s=4.0,
            viewpoint=self.commanded,
            observation_quality=0.8,
            sensor_timestamps_s={"rgb": 3.9, "depth": 4.0},
        )

        observation = SearchObservationAdapter(
            self.grid,
            fallback_footprint_radius_m=1.0,
            maximum_sensor_skew_s=0.2,
        ).adapt(frame, self.commanded)

        self.assertAlmostEqual(observation.observation_quality, 0.4)

    def test_fallback_footprint_is_used_without_depth_projection(self):
        frame = SearchSensorFrame(
            timestamp_s=4.0,
            viewpoint=self.commanded,
        )

        observation = SearchObservationAdapter(
            self.grid,
            fallback_footprint_radius_m=1.0,
        ).adapt(frame, self.commanded)

        self.assertEqual(observation.visible_cell_ids, ("gazebo-area:r0:c0",))
        self.assertEqual(observation.negative_update_strength, 0.0)
        self.assertEqual(
            observation.negative_update_rejection_reason,
            "no_valid_point_projection",
        )

    def test_valid_projection_preserves_confidence_gated_negative_strength(self):
        frame = SearchSensorFrame(
            timestamp_s=4.0,
            viewpoint=self.commanded,
            visible_ground_points_xy=((10.0, 10.0),),
            visibility_probability=0.7,
            negative_update_strength=0.8,
        )

        observation = SearchObservationAdapter(self.grid).adapt(
            frame,
            self.commanded,
        )

        self.assertAlmostEqual(observation.negative_update_strength, 0.56)
        self.assertIsNone(observation.negative_update_rejection_reason)

    def test_sensor_frame_rejects_invalid_quality(self):
        with self.assertRaises(ValueError):
            SearchSensorFrame(
                timestamp_s=1.0,
                viewpoint=self.commanded,
                observation_quality=1.1,
            )


if __name__ == "__main__":
    unittest.main()
