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


if __name__ == "__main__":
    unittest.main()
