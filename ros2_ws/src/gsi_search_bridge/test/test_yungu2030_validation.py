import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROS2_WS = Path(__file__).resolve().parents[3]
TOOL_PATH = ROS2_WS / "tools" / "summarize_yungu2030_validation.py"
SPEC = importlib.util.spec_from_file_location("summarize_yungu2030_validation", TOOL_PATH)
validation = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = validation
SPEC.loader.exec_module(validation)


MANIFEST_PATH = ROS2_WS / "simulation" / "yungu2030_v1" / "validation_manifest.json"
SEARCH_PARAMS_PATH = ROS2_WS / "simulation" / "yungu2030_v1" / "yungu_search_params.yaml"
AIRFRAME_PATH = (
    ROS2_WS
    / "simulation"
    / "yungu2030_v1"
    / "airframes"
    / "4012_gz_x500_gsi_rgbd_nadir_longrange"
)
YUNGU_INSTALLER_PATH = ROS2_WS / "install_yungu2030_sitl.sh"
V1_1_INSTALLER_PATH = ROS2_WS / "install_search_world_v1_1.sh"
SEMANTIC_MAP_PATH = ROS2_WS.parent / "data" / "yungu2030_v1" / "semantic_map.json"


class ManifestAndStatisticsTests(unittest.TestCase):
    def test_visual_demo_uses_flat_road_and_overlapping_coverage(self):
        parameters = SEARCH_PARAMS_PATH.read_text(encoding="utf-8")

        self.assertIn("area_min_x_m: 90.0", parameters)
        self.assertIn("area_min_y_m: 57.0", parameters)
        self.assertIn("area_max_x_m: 140.0", parameters)
        self.assertIn("area_max_y_m: 72.0", parameters)
        self.assertIn("flight_altitude_m: 15.0", parameters)
        self.assertIn("grid_resolution_m: 7.0", parameters)
        self.assertIn("coverage_pass_spacing_m: 7.0", parameters)
        self.assertIn("coverage_observation_spacing_m: 7.0", parameters)
        self.assertIn("ground_plane_z_m: 0.28", parameters)
        self.assertIn("verification_max_horizontal_offset_m: 0.01", parameters)
        self.assertIn("operator_confirmation_enabled: true", parameters)
        self.assertIn(
            "operator_confirmation_topic: /gsi/operator_confirmation",
            parameters,
        )

    def test_yungu_uses_px4_trajectory_control_baseline(self):
        parameters = SEARCH_PARAMS_PATH.read_text(encoding="utf-8")
        airframe = AIRFRAME_PATH.read_text(encoding="utf-8")

        self.assertIn("horizontal_setpoint_speed_mps: 0.0", parameters)
        self.assertIn("horizontal_setpoint_max_lead_m: 0.0", parameters)
        self.assertIn("param set-default MPC_XY_VEL_MAX 1.5", airframe)
        self.assertIn("param set-default MPC_ACC_HOR 2.0", airframe)
        self.assertIn("param set-default MPC_ACC_HOR_MAX 2.0", airframe)
        self.assertIn("param set-default MPC_TILTMAX_AIR 20.0", airframe)

        controller_condition = (
            "SYS_AUTOSTART 4010 || param compare -s SYS_AUTOSTART 4011 || "
            "param compare -s SYS_AUTOSTART 4012"
        )
        self.assertIn(
            controller_condition,
            YUNGU_INSTALLER_PATH.read_text(encoding="utf-8"),
        )
        self.assertIn(
            controller_condition,
            V1_1_INSTALLER_PATH.read_text(encoding="utf-8"),
        )

    def test_frozen_manifest_positions_match_open_semantic_regions(self):
        result = validation.validate_manifest(MANIFEST_PATH, SEMANTIC_MAP_PATH)

        self.assertTrue(result["valid"])
        self.assertEqual(result["target_count"], 13)
        self.assertEqual(len(result["sha256"]), 64)

    def test_manifest_rejects_target_outside_declared_region(self):
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        manifest["positions"]["targets"][0]["x"] = 999.0
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(validation.ManifestValidationError, "outside"):
                validation.validate_manifest(path, SEMANTIC_MAP_PATH)

    def test_episode_expansion_freezes_expected_sample_sizes(self):
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

        self.assertEqual(len(validation.expand_episodes(manifest, "preflight")), 2)
        self.assertEqual(len(validation.expand_episodes(manifest, "stability")), 20)
        self.assertEqual(len(validation.expand_episodes(manifest, "positions")), 36)
        self.assertEqual(len(validation.expand_episodes(manifest, "all")), 56)

        timed_out = {
            "road_connector_north",
            "road_connector_south",
            "street_edge_mid",
        }
        filtered = [
            episode
            for episode in validation.expand_episodes(manifest, "positions")
            if episode["target_id"] in timed_out
        ]
        self.assertEqual(len(filtered), 9)

    def test_wilson_interval_matches_reference_values(self):
        interval = validation.wilson_interval(9, 10)

        self.assertAlmostEqual(interval["lower"], 0.5958499732)
        self.assertAlmostEqual(interval["upper"], 0.9821237869)
        self.assertIsNone(validation.wilson_interval(0, 0)["lower"])


def _classification_inputs():
    return {
        "outcome": {"status": "found", "found": True},
        "observation_qualities": [0.75, 0.7],
        "observation_triggers": [
            "positive_detection_in_transit",
            "settled_viewpoint",
        ],
        "localization_error_m": 3.0,
        "artifact_complete": True,
        "trace_error": None,
        "runner_status": 0,
        "search_exit_status": 0,
        "minimum_observations": 2,
        "minimum_quality": 0.5,
        "required_second_trigger": "settled_viewpoint",
        "maximum_localization_error_m": 8.0,
    }


class TrialClassificationTests(unittest.TestCase):
    def test_trial_manifest_falls_back_to_batch_local_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            batch = Path(directory)
            trial = batch / "episodes" / "trial-1"
            trial.mkdir(parents=True)
            manifest = batch / "validation_manifest.json"
            manifest.write_text("{}", encoding="utf-8")

            resolved = validation._resolve_trial_manifest(
                trial,
                "/mnt/c/moved/batch/validation_manifest.json",
            )

            self.assertEqual(resolved, manifest)

    def test_confirmation_observations_ignore_negative_transit_evidence(self):
        observations = [
            {
                "detections": [],
                "sensor_metadata": {
                    "observation_trigger": "negative_detection_in_transit",
                },
            },
            {
                "detections": [{"label": "yellow-van"}],
                "sensor_metadata": {
                    "observation_trigger": "positive_detection_in_transit",
                },
            },
            {
                "detections": [{"label": "yellow-van"}],
                "sensor_metadata": {"observation_trigger": "settled_viewpoint"},
            },
        ]

        confirmations = validation._confirmation_observations(observations)

        self.assertEqual(len(confirmations), 2)
        self.assertEqual(
            confirmations[1]["sensor_metadata"]["observation_trigger"],
            "settled_viewpoint",
        )

    def test_trial_classification_accepts_complete_contract(self):
        self.assertEqual(
            validation.classify_trial(**_classification_inputs()),
            (True, None),
        )

    def test_trial_classification_distinguishes_failure_modes(self):
        cases = [
            ({"search_exit_status": 124}, "search_timeout"),
            ({"outcome": {"status": "not_found", "found": False}}, "algorithm_failure"),
            ({"observation_triggers": ["settled_viewpoint"]}, "observation_independence_failure"),
            ({"observation_qualities": [0.75, 0.49]}, "observation_quality_failure"),
            ({"localization_error_m": 8.01}, "localization_failure"),
            ({"artifact_complete": False}, "infrastructure_artifact_failure"),
        ]
        for changes, category in cases:
            with self.subTest(category=category):
                inputs = _classification_inputs()
                inputs.update(changes)
                self.assertEqual(
                    validation.classify_trial(**inputs),
                    (False, category),
                )

    def test_batch_aggregation_keeps_position_target_coverage_separate(self):
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "episodes" / "stability").mkdir(parents=True)
            (root / "episodes" / "position").mkdir(parents=True)
            (root / "validation_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            (root / "batch_metadata.json").write_text(
                json.dumps({
                    "batch_mode": "preflight",
                    "expected_episode_count": 2,
                    "manifest_sha256": "test-sha",
                }),
                encoding="utf-8",
            )
            common = {
                "success": True,
                "artifact_complete": True,
                "failure_category": None,
                "elapsed_time_s": 10.0,
                "distance_travelled_m": 20.0,
                "localization_error_m": 1.0,
            }
            stability = {
                **common,
                "target_id": "stability_reference",
                "semantic_region": "SEM_PLAZA_NORTH",
                "cohort": "stability",
            }
            position = {
                **common,
                "target_id": "plaza_north_center",
                "semantic_region": "SEM_PLAZA_NORTH",
                "cohort": "positions",
            }
            (root / "episodes" / "stability" / "trial_summary.json").write_text(
                json.dumps(stability), encoding="utf-8"
            )
            (root / "episodes" / "position" / "trial_summary.json").write_text(
                json.dumps(position), encoding="utf-8"
            )

            summary = validation.aggregate_batch(root)

            self.assertEqual(summary["position_target_coverage"]["target_count"], 1)
            self.assertTrue((root / "trials.csv").is_file())
            self.assertTrue((root / "batch_summary.json").is_file())


if __name__ == "__main__":
    unittest.main()
