import copy
import json
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

from gsi_search_bridge.search_world_generator import (
    generate_artifacts,
    load_config,
    validate_config,
)


ROS2_WS = Path(__file__).resolve().parents[3]
CONFIG_PATH = ROS2_WS / "simulation" / "search_world_v1" / "search_world_v1.json"
V11_CONFIG_PATH = ROS2_WS / "simulation" / "search_world_v1_1" / "search_world_v1_1.json"
V11_MODEL_PATH = (
    ROS2_WS
    / "simulation"
    / "search_world_v1_1"
    / "models"
    / "x500_gsi_rgbd"
    / "model.sdf"
)
V2_ROOT = ROS2_WS / "simulation" / "search_world_v2"
V2_SCENARIOS = ("campus", "industrial", "suburban")
V2_MODEL_PATH = V2_ROOT / "models" / "x500_gsi_rgbd_nadir" / "model.sdf"


class SearchWorldGeneratorTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(CONFIG_PATH)

    def test_generation_is_deterministic(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            first_artifacts = generate_artifacts(self.config, first)
            second_artifacts = generate_artifacts(self.config, second)
            for key in first_artifacts:
                self.assertEqual(
                    first_artifacts[key].read_bytes(),
                    second_artifacts[key].read_bytes(),
                    key,
                )

    def test_sdf_contains_physical_complexity_and_target(self):
        with tempfile.TemporaryDirectory() as output:
            artifacts = generate_artifacts(self.config, output)
            root = ET.parse(artifacts["world"]).getroot()
            world = root.find("world")
            self.assertEqual(world.attrib["name"], "gsi_search_world_v1")
            self.assertEqual(
                world.find("spherical_coordinates/world_frame_orientation").text,
                "ENU",
            )
            model_names = {model.attrib["name"] for model in world.findall("model")}
            self.assertIn("yellow_search_van", model_names)
            self.assertIn("office-main", model_names)
            self.assertIn("restricted-fence-north", model_names)
            self.assertGreater(len(model_names), 40)

    def test_public_semantics_exclude_target_ground_truth(self):
        with tempfile.TemporaryDirectory() as output:
            artifacts = generate_artifacts(self.config, output)
            public = json.loads(artifacts["semantic_map"].read_text(encoding="utf-8"))
            private = json.loads(artifacts["ground_truth"].read_text(encoding="utf-8"))
            public_text = json.dumps(public)
            self.assertNotIn("yellow-search-van", public_text)
            self.assertNotIn("yellow_search_van", public_text)
            self.assertTrue(private["evaluator_only"])
            self.assertEqual(private["targets"][0]["entity_id"], "yellow-search-van")
            restricted = next(node for node in public["nodes"] if node["id"] == "restricted-northeast")
            self.assertEqual(restricted["properties"]["passability"], "restricted")

    def test_generated_runtime_configs_use_world_name(self):
        with tempfile.TemporaryDirectory() as output:
            artifacts = generate_artifacts(self.config, output)
            bridge = artifacts["gz_bridge"].read_text(encoding="utf-8")
            params = artifacts["search_params"].read_text(encoding="utf-8")
            profile = artifacts["visionflow_profile"].read_text(encoding="utf-8")
            self.assertIn("/world/gsi_search_world_v1/", bridge)
            self.assertIn("area_id: gsi_search_world_v1", params)
            self.assertIn("area_min_x_m: -40.0", params)
            self.assertIn("flight_altitude_m: 12.0", params)
            self.assertIn("ground_plane_z_m: 0.0", params)
            self.assertIn("semantic_map_path:", params)
            self.assertIn('--id "GSI SearchWorld V1"', profile)
            self.assertIn(
                '--target "gz_q940_ti_gripper4_gsi_search_world_v1"',
                profile,
            )

    def test_invalid_or_out_of_range_parameters_are_rejected(self):
        too_small = copy.deepcopy(self.config)
        too_small["world"]["size_m"]["x"] = 20
        with self.assertRaises(ValueError):
            validate_config(too_small)

        bad_slot = copy.deepcopy(self.config)
        bad_slot["target"]["slot_index"] = 99
        with tempfile.TemporaryDirectory() as output, self.assertRaises(ValueError):
            generate_artifacts(bad_slot, output)


class SearchWorldV11GeneratorTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(V11_CONFIG_PATH)

    def test_x500_rgbd_runtime_contract(self):
        with tempfile.TemporaryDirectory() as output:
            artifacts = generate_artifacts(self.config, output)
            bridge = artifacts["gz_bridge"].read_text(encoding="utf-8")
            params = artifacts["search_params"].read_text(encoding="utf-8")
            profile = artifacts["visionflow_profile"].read_text(encoding="utf-8")
            self.assertIn("gz_topic_name: /gsi/rgbd/image", bridge)
            self.assertIn("gz_topic_name: /gsi/rgbd/depth_image", bridge)
            self.assertIn("gz_topic_name: /gsi/rgbd/points", bridge)
            self.assertIn("ros_topic_name: /oakd1/rgb/image", bridge)
            self.assertIn("camera_translation_x_m: 0.12", params)
            self.assertIn("camera_pitch_rad: 0.785398", params)
            self.assertIn("pointcloud_frame_id: gsi_rgbd_link", params)
            self.assertIn("simulation/search_world_v1_1/generated", params)
            self.assertIn('--id "GSI SearchWorld V1.1"', profile)
            self.assertIn(
                '--target "gz_x500_gsi_rgbd_gsi_search_world_v1_1"',
                profile,
            )

    def test_vehicle_is_x500_with_exactly_one_rgbd_sensor(self):
        root = ET.parse(V11_MODEL_PATH).getroot()
        model = root.find("model")
        sensors = model.findall("link/sensor")
        self.assertEqual(model.attrib["name"], "x500_gsi_rgbd")
        self.assertEqual(len(sensors), 1)
        self.assertEqual(sensors[0].attrib["type"], "rgbd_camera")
        self.assertEqual(sensors[0].findtext("update_rate"), "10")
        self.assertEqual(sensors[0].findtext("camera/image/width"), "640")
        self.assertEqual(sensors[0].findtext("camera/image/height"), "480")
        model_text = V11_MODEL_PATH.read_text(encoding="utf-8").lower()
        for excluded in ("gripper", "manipulator", "oakd"):
            self.assertNotIn(excluded, model_text)

    def test_incomplete_sensor_contract_is_rejected(self):
        invalid = copy.deepcopy(self.config)
        del invalid["sensor"]["point_cloud_suffix"]
        with self.assertRaises(ValueError):
            validate_config(invalid)


class SearchWorldV2GeneratorTests(unittest.TestCase):
    def test_v2_vehicle_has_one_nadir_rgbd_sensor(self):
        root = ET.parse(V2_MODEL_PATH).getroot()
        model = root.find("model")
        sensors = model.findall("link/sensor")
        self.assertEqual(model.attrib["name"], "x500_gsi_rgbd_nadir")
        self.assertEqual(len(sensors), 1)
        self.assertEqual(sensors[0].attrib["type"], "rgbd_camera")
        self.assertEqual(model.findtext("link/pose").split()[4], "1.570796")

    def test_all_archetypes_generate_distinct_complete_contracts(self):
        world_hashes = set()
        target_regions = set()
        for scenario in V2_SCENARIOS:
            config = load_config(V2_ROOT / scenario / "scenario.json")
            with tempfile.TemporaryDirectory() as output:
                artifacts = generate_artifacts(config, output)
                root = ET.parse(artifacts["world"]).getroot()
                world = root.find("world")
                manifest = json.loads(artifacts["manifest"].read_text(encoding="utf-8"))
                public = json.loads(artifacts["semantic_map"].read_text(encoding="utf-8"))
                prior = json.loads(artifacts["search_prior"].read_text(encoding="utf-8"))
                truth = json.loads(artifacts["ground_truth"].read_text(encoding="utf-8"))
                params = artifacts["search_params"].read_text(encoding="utf-8")
                profile = artifacts["visionflow_profile"].read_text(encoding="utf-8")

                self.assertEqual(world.attrib["name"], f"gsi_search_world_v2_{scenario}")
                self.assertEqual(manifest["complexity"]["layout_archetype"], scenario)
                self.assertGreaterEqual(manifest["complexity"]["semantic_region_count"], 7)
                self.assertGreater(manifest["complexity"]["utility_pole_count"], 0)
                self.assertGreater(len(world.findall("model")), 50)
                self.assertNotIn("yellow_search_van", json.dumps(public))
                self.assertEqual(prior["projection_mode"], "label_mass")
                self.assertTrue(truth["evaluator_only"])
                self.assertIn("camera_pitch_rad: 1.570796", params)
                self.assertIn("gz_x500_gsi_rgbd_nadir_", profile)
                target_regions.add(truth["targets"][0]["semantic_region_id"])
                world_hashes.add(manifest["artifacts"]["world"]["sha256"])

        self.assertEqual(len(world_hashes), len(V2_SCENARIOS))
        self.assertEqual(len(target_regions), len(V2_SCENARIOS))

    def test_v2_generation_is_deterministic(self):
        config = load_config(V2_ROOT / "industrial" / "scenario.json")
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            first_artifacts = generate_artifacts(config, first)
            second_artifacts = generate_artifacts(config, second)
            for key in first_artifacts:
                self.assertEqual(first_artifacts[key].read_bytes(), second_artifacts[key].read_bytes())

    def test_v2_parameters_and_archetype_are_validated(self):
        config = load_config(V2_ROOT / "campus" / "scenario.json")
        invalid_scene = copy.deepcopy(config)
        invalid_scene["scene"]["archetype"] = "fantasy"
        with self.assertRaises(ValueError):
            validate_config(invalid_scene)

        too_small = copy.deepcopy(config)
        too_small["world"]["size_m"] = {"x": 80.0, "y": 60.0}
        with self.assertRaises(ValueError):
            validate_config(too_small)

        too_many_buildings = copy.deepcopy(config)
        too_many_buildings["complexity"]["building_count"] = 99
        with tempfile.TemporaryDirectory() as output, self.assertRaises(ValueError):
            generate_artifacts(too_many_buildings, output)


if __name__ == "__main__":
    unittest.main()
