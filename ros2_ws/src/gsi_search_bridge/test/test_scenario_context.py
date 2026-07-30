import json
from pathlib import Path
import tempfile
import unittest

from gsi_search_bridge.scenario_context import load_search_scenario_context
from modules.search_intelligence import SearchGrid, SearchTask


class SearchScenarioContextTests(unittest.TestCase):
    def setUp(self):
        self.task = SearchTask.from_skill_params({
            "task_id": "gazebo-active-search",
            "area_token": "context-test",
            "area": {
                "kind": "rectangle",
                "coords": [[0, 0], [20, 0], [20, 10], [0, 10]],
            },
            "target_token": "yellow-van",
        })
        self.grid = SearchGrid.from_task(self.task, resolution_m=10.0)

    def test_semantics_and_prior_create_nonuniform_belief(self):
        semantic = {
            "nodes": [
                {
                    "id": "parking",
                    "properties": {
                        "category": "area",
                        "type": "parking",
                        "label": "parking",
                    },
                    "shape": {
                        "type": "rectangle",
                        "min_corner": [0, 0],
                        "max_corner": [10, 10],
                    },
                },
                {
                    "id": "park",
                    "properties": {
                        "category": "area",
                        "type": "park",
                        "label": "park",
                    },
                    "shape": {
                        "type": "rectangle",
                        "min_corner": [10, 0],
                        "max_corner": [20, 10],
                    },
                },
            ]
        }
        prior = {
            "semantic_weights": {"parking": 1.0, "park": 0.1},
            "confidence": 1.0,
        }
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            semantic_path = directory / "semantic.json"
            prior_path = directory / "prior.json"
            semantic_path.write_text(json.dumps(semantic), encoding="utf-8")
            prior_path.write_text(json.dumps(prior), encoding="utf-8")
            context = load_search_scenario_context(
                self.task,
                self.grid,
                semantic_map_path=str(semantic_path),
                search_prior_path=str(prior_path),
            )

        self.assertTrue(context.policy_metadata["semantic_map_loaded"])
        self.assertTrue(context.policy_metadata["prior_loaded"])
        self.assertGreater(
            context.initial_belief["context-test:r0:c0"],
            context.initial_belief["context-test:r0:c1"],
        )

    def test_empty_paths_preserve_uniform_baseline(self):
        context = load_search_scenario_context(self.task, self.grid)
        self.assertEqual(context.initial_belief, self.grid.uniform_belief())
        self.assertFalse(context.policy_metadata["semantic_map_loaded"])

    def test_ground_truth_shape_is_not_accepted_as_semantic_map(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ground_truth.json"
            path.write_text(json.dumps({"targets": []}), encoding="utf-8")
            with self.assertRaises(TypeError):
                load_search_scenario_context(
                    self.task,
                    self.grid,
                    semantic_map_path=str(path),
                )


if __name__ == "__main__":
    unittest.main()
