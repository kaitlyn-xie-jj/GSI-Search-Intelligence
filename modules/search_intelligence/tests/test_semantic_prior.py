import json
import unittest

from modules.search_intelligence import (
    SearchGrid,
    SearchPrior,
    SearchPriorRequest,
    SearchTask,
    SemanticGridBuilder,
)


class SemanticGridBuilderTests(unittest.TestCase):
    def setUp(self):
        task = SearchTask.from_skill_params({
            "task_id": "semantic-grid",
            "area_token": "Outdoor-A",
            "area": {
                "kind": "rectangle",
                "coords": [[0, 0], [60, 0], [60, 40], [0, 40]],
            },
            "target_token": "yellow-van",
        })
        self.grid = SearchGrid.from_task(task, resolution_m=20.0)

    def test_scene_environment_nodes_are_projected_to_cell_labels(self):
        nodes = [
            {
                "id": "parking-1",
                "properties": {
                    "category": "building",
                    "type": "parking",
                    "label": "Parking-1",
                },
                "shape": {
                    "type": "rectangle",
                    "min_corner": [0, 0],
                    "max_corner": [40, 40],
                },
            },
            {
                "id": "road-1",
                "properties": {
                    "category": "trans_facility",
                    "type": "street_segment",
                    "label": "Street Segment-1",
                },
                "shape": {
                    "type": "linestring",
                    "points": [[0, 10], [60, 10]],
                },
            },
        ]

        annotated = SemanticGridBuilder(line_buffer_m=5.0).annotate(self.grid, nodes)

        left = annotated.cell(0, 0)
        right = annotated.cell(1, 2)
        self.assertIn("parking", left.semantic_labels)
        self.assertIn("parking_1", left.semantic_labels)
        self.assertIn("street_segment", left.semantic_labels)
        self.assertNotIn("parking", right.semantic_labels)
        json.dumps(annotated.to_dict())

    def test_prop_ground_truth_is_never_projected_into_semantic_grid(self):
        nodes = [{
            "id": "vehicle-7",
            "properties": {
                "category": "prop",
                "type": "vehicle",
                "label": "Yellow Van Ground Truth",
            },
            "shape": {"type": "point", "center": [10, 10]},
        }]

        annotated = SemanticGridBuilder().annotate(self.grid, nodes)

        all_labels = {
            label for cell in annotated.cells for label in cell.semantic_labels
        }
        self.assertNotIn("vehicle", all_labels)
        self.assertNotIn("yellow_van_ground_truth", all_labels)

    def test_explicitly_restricted_environment_feature_blocks_cells(self):
        nodes = [{
            "id": "restricted-1",
            "properties": {
                "category": "area",
                "type": "restricted_zone",
                "label": "Restricted Zone-1",
                "passability": "restricted",
            },
            "shape": {
                "type": "rectangle",
                "min_corner": [0, 0],
                "max_corner": [20, 20],
            },
        }]

        annotated = SemanticGridBuilder().annotate(self.grid, nodes)

        self.assertFalse(annotated.cell(0, 0).searchable)
        self.assertEqual(
            annotated.cell(0, 0).metadata["blocked_by_feature_ids"],
            ("restricted-1",),
        )
        self.assertTrue(annotated.cell(0, 1).searchable)

    def test_scene_graph_public_interface_is_supported(self):
        class SceneGraphStub:
            def get_all_nodes(self):
                return [{
                    "id": "garden-1",
                    "properties": {
                        "category": "area",
                        "type": "garden",
                        "label": "Garden-1",
                    },
                    "shape": {
                        "type": "polygon",
                        "vertices": [[0, 0], [20, 0], [20, 20], [0, 20]],
                    },
                }]

        annotated = SemanticGridBuilder().from_scene_graph(
            self.grid,
            SceneGraphStub(),
        )

        self.assertIn("garden", annotated.cell(0, 0).semantic_labels)


class SearchPriorTests(unittest.TestCase):
    def setUp(self):
        task = SearchTask.from_skill_params({
            "task_id": "prior-task",
            "area_token": "Outdoor-A",
            "area": {
                "kind": "rectangle",
                "coords": [[0, 0], [40, 0], [40, 20], [0, 20]],
            },
            "target_token": "yellow-van",
        })
        grid = SearchGrid.from_task(task, resolution_m=20.0)
        nodes = [
            {
                "id": "parking",
                "properties": {
                    "category": "building",
                    "type": "parking",
                    "label": "Parking-A",
                },
                "shape": {
                    "type": "rectangle",
                    "min_corner": [0, 0],
                    "max_corner": [20, 20],
                },
            },
            {
                "id": "garden",
                "properties": {
                    "category": "area",
                    "type": "garden",
                    "label": "Garden-A",
                },
                "shape": {
                    "type": "rectangle",
                    "min_corner": [20, 0],
                    "max_corner": [40, 20],
                },
            },
        ]
        self.grid = SemanticGridBuilder().annotate(grid, nodes)

    def test_llm_semantic_weights_become_normalized_cell_prior(self):
        prior = SearchPrior.from_llm_output("prior-task", {
            "semantic_region_weights": {"Parking": 0.8, "Garden": 0.2},
            "confidence": 1.0,
        })

        projection = prior.project(self.grid)

        self.assertAlmostEqual(projection.belief["Outdoor-A:r0:c0"], 0.8)
        self.assertAlmostEqual(projection.belief["Outdoor-A:r0:c1"], 0.2)
        self.assertEqual(projection.matched_labels, ("garden", "parking"))
        self.assertEqual(projection.unmatched_labels, ())
        self.assertAlmostEqual(sum(projection.belief.values()), 1.0)
        json.dumps(projection.to_dict())

    def test_label_mass_projection_does_not_reward_larger_regions(self):
        task = SearchTask.from_skill_params({
            "task_id": "area-normalized-prior",
            "area_token": "Outdoor-B",
            "area": {
                "kind": "rectangle",
                "coords": [[0, 0], [60, 0], [60, 20], [0, 20]],
            },
            "target_token": "yellow-van",
        })
        grid = SearchGrid.from_task(task, resolution_m=20.0)
        annotated = SemanticGridBuilder().annotate(grid, [
            {
                "id": "large-parking",
                "properties": {"category": "area", "type": "parking"},
                "shape": {
                    "type": "rectangle",
                    "min_corner": [0, 0],
                    "max_corner": [40, 20],
                },
            },
            {
                "id": "small-entrance",
                "properties": {"category": "area", "type": "building_entrance"},
                "shape": {
                    "type": "rectangle",
                    "min_corner": [40, 0],
                    "max_corner": [60, 20],
                },
            },
        ])
        projection = SearchPrior(
            task_id=task.task_id,
            semantic_weights={"parking": 1.0, "building_entrance": 1.0},
            confidence=1.0,
            projection_mode="label_mass",
        ).project(annotated)
        cells = {cell.cell_id: cell for cell in annotated.cells}

        parking_mass = sum(
            probability
            for cell_id, probability in projection.belief.items()
            if "parking" in cells[cell_id].semantic_labels
        )
        entrance_mass = sum(
            probability
            for cell_id, probability in projection.belief.items()
            if "building_entrance" in cells[cell_id].semantic_labels
        )
        self.assertAlmostEqual(parking_mass, 0.5)
        self.assertAlmostEqual(entrance_mass, 0.5)
        self.assertEqual(projection.projection_mode, "label_mass")

    def test_low_confidence_prior_is_mixed_with_uniform_uncertainty(self):
        prior = SearchPrior(
            task_id="prior-task",
            semantic_weights={"parking": 1.0, "garden": 0.0},
            confidence=0.5,
        )

        projection = prior.project(self.grid)

        self.assertAlmostEqual(projection.belief["Outdoor-A:r0:c0"], 0.75)
        self.assertAlmostEqual(projection.belief["Outdoor-A:r0:c1"], 0.25)

    def test_unmatched_labels_are_reported_without_breaking_normalization(self):
        prior = SearchPrior(
            task_id="prior-task",
            semantic_weights={"airport": 1.0},
        )

        projection = prior.project(self.grid)

        self.assertEqual(projection.unmatched_labels, ("airport",))
        self.assertAlmostEqual(projection.belief["Outdoor-A:r0:c0"], 0.5)
        self.assertAlmostEqual(projection.belief["Outdoor-A:r0:c1"], 0.5)

    def test_negative_weights_are_rejected(self):
        with self.assertRaises(ValueError):
            SearchPrior(
                task_id="prior-task",
                semantic_weights={"parking": -1.0},
            )

    def test_prior_request_exposes_target_and_available_scene_semantics(self):
        task = SearchTask.from_skill_params({
            "task_id": "prior-task",
            "area_token": "Outdoor-A",
            "area": {
                "kind": "rectangle",
                "coords": [[0, 0], [40, 0], [40, 20], [0, 20]],
            },
            "target_token": "yellow-van",
            "target": {
                "class": "vehicle",
                "type": "van",
                "features": {"color": "yellow"},
            },
            "object_id": "ground-truth-must-not-appear",
        })

        request = SearchPriorRequest.from_task_and_grid(task, self.grid)

        self.assertEqual(request.target_query, "yellow-van")
        self.assertEqual(request.target_attributes, {"color": "yellow"})
        self.assertEqual(request.semantic_inventory["parking"], 1)
        self.assertNotIn("ground-truth-must-not-appear", json.dumps(request.to_dict()))
        schema = SearchPrior.llm_output_schema()
        self.assertIn("semantic_weights", schema["properties"])

    def test_excluded_semantic_region_keeps_zero_probability(self):
        prior = SearchPrior(
            task_id="prior-task",
            semantic_weights={"parking": 1.0, "garden": 1.0},
            confidence=0.2,
            excluded_labels=("garden",),
        )

        projection = prior.project(self.grid)

        self.assertEqual(projection.belief["Outdoor-A:r0:c1"], 0.0)
        self.assertEqual(projection.belief["Outdoor-A:r0:c0"], 1.0)


if __name__ == "__main__":
    unittest.main()
