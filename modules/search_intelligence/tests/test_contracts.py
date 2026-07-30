import json
import unittest

from modules.search_intelligence.contracts import (
    SearchObservation,
    SearchOutcome,
    SearchOutcomeStatus,
    SearchState,
    SearchTask,
    TargetDetection,
    Viewpoint,
)


class SearchContractTests(unittest.TestCase):
    def setUp(self):
        self.params = {
            "task_id": "T1",
            "area": {
                "kind": "polygon",
                "coords": [[0, 0], [100, 0], [100, 50], [0, 50]],
            },
            "area_token": "Parking-A",
            "target": {
                "class": "vehicle",
                "type": "van",
                "features": {"color": "yellow"},
            },
            "target_token": "yellow-van",
            "object_id": "ground-truth-object",
            "target_ids": ["ground-truth-object"],
            "conf_ge": 0.8,
            "persist_ge_s": 1.5,
            "time_budget_s": 30.0,
            "max_viewpoints": 2,
            "context_priors": "parking-space",
        }
        self.task = SearchTask.from_skill_params(
            self.params,
            instruction="Find the yellow van in Parking-A",
        )

    def test_skill_params_create_public_search_task_without_ground_truth(self):
        self.assertEqual(self.task.task_id, "T1")
        self.assertEqual(self.task.search_area.area_id, "Parking-A")
        self.assertEqual(self.task.target.category, "vehicle")
        self.assertEqual(self.task.target.subtype, "van")
        self.assertEqual(self.task.target.attributes["color"], "yellow")
        self.assertEqual(self.task.context_priors, ("parking-space",))
        serialized = self.task.to_dict()
        self.assertNotIn("object_id", json.dumps(serialized))
        self.assertNotIn("target_ids", json.dumps(serialized))
        json.dumps(serialized)

    def test_observation_validates_detection_confidence(self):
        with self.assertRaises(ValueError):
            TargetDetection(label="van", confidence=1.1)

    def test_state_advance_accumulates_resources_and_coverage(self):
        state = SearchState.initial(self.task, {"cell-a": 2.0, "cell-b": 1.0})
        observation = SearchObservation(
            viewpoint=Viewpoint(10.0, 20.0, 30.0, 1.57, -0.5),
            timestamp_s=1.0,
            visible_cell_ids=("cell-a", "cell-c"),
            observation_quality=0.7,
            travel_time_s=5.0,
            travel_distance_m=12.0,
            energy_used=2.5,
        )

        updated = state.advance(observation, belief={"cell-a": 0.2, "cell-b": 0.8})

        self.assertEqual(updated.step_index, 1)
        self.assertEqual(updated.current_viewpoint, observation.viewpoint)
        self.assertAlmostEqual(updated.elapsed_time_s, 5.0)
        self.assertAlmostEqual(updated.distance_travelled_m, 12.0)
        self.assertAlmostEqual(updated.energy_used, 2.5)
        self.assertEqual(updated.observed_cell_quality["cell-c"], 0.7)
        self.assertIsNone(updated.exhausted_budget)
        json.dumps(updated.to_dict())

    def test_outcome_adapts_to_platform_result(self):
        state = SearchState.initial(self.task, {"cell-a": 1.0})
        detection = TargetDetection(
            label="yellow van",
            confidence=0.91,
            estimated_position=(12.0, 8.0, 0.0),
            entity_id="observed-vehicle-7",
        )
        outcome = SearchOutcome.from_state(
            state,
            status=SearchOutcomeStatus.FOUND,
            reason="success criteria met",
            detections=(detection,),
        )

        platform_result = outcome.to_platform_result()
        self.assertTrue(platform_result["success"])
        self.assertEqual(platform_result["targets_found"], ["observed-vehicle-7"])
        self.assertEqual(platform_result["outcome"], "found")
        self.assertEqual(platform_result["search_metrics"]["belief_cell_count"], 1)
        self.assertEqual(platform_result["search_metrics"]["coverage_fraction"], 0.0)
        json.dumps(outcome.to_dict())

    def test_outcome_reports_grid_coverage_fraction(self):
        state = SearchState.initial(
            self.task,
            {"cell-a": 0.5, "cell-b": 0.5},
        ).advance(SearchObservation(
            viewpoint=Viewpoint(0.0, 0.0, 20.0, 0.0),
            timestamp_s=1.0,
            visible_cell_ids=("cell-a",),
        ))

        outcome = SearchOutcome.from_state(
            state,
            status=SearchOutcomeStatus.NOT_FOUND,
            reason="route exhausted",
        )

        self.assertEqual(outcome.metrics["observed_cell_count"], 1)
        self.assertEqual(outcome.metrics["belief_cell_count"], 2)
        self.assertEqual(outcome.metrics["coverage_fraction"], 0.5)

    def test_viewpoint_budget_is_reported(self):
        state = SearchState.initial(self.task, {"cell-a": 1.0})
        for step in range(2):
            state = state.advance(
                SearchObservation(
                    viewpoint=Viewpoint(float(step), 0.0, 20.0, 0.0),
                    timestamp_s=float(step),
                )
            )
        self.assertEqual(state.exhausted_budget, "viewpoints")


if __name__ == "__main__":
    unittest.main()
