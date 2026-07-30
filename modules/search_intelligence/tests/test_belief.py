import json
import unittest

from modules.search_intelligence import (
    BayesianBeliefUpdater,
    BeliefMap,
    BinarySensorModel,
    CoveragePolicy,
    SearchGrid,
    SearchObservation,
    SearchSession,
    SearchTask,
    TargetDetection,
    Viewpoint,
)


class BayesianBeliefTests(unittest.TestCase):
    def setUp(self):
        self.task = SearchTask.from_skill_params({
            "task_id": "belief-task",
            "area_token": "Area-A",
            "area": {
                "kind": "rectangle",
                "coords": [[0, 0], [40, 0], [40, 20], [0, 20]],
            },
            "target_token": "yellow-van",
            "conf_ge": 0.5,
        })
        self.grid = SearchGrid.from_task(self.task, resolution_m=20.0)
        self.cell_a = "Area-A:r0:c0"
        self.cell_b = "Area-A:r0:c1"
        self.prior = BeliefMap.for_grid(self.grid)
        self.updater = BayesianBeliefUpdater(BinarySensorModel(
            detection_probability=0.8,
            false_positive_probability=0.1,
        ))

    def _observation(self, **overrides):
        params = {
            "viewpoint": Viewpoint(10.0, 10.0, 30.0, 0.0),
            "timestamp_s": 1.0,
            "visible_cell_ids": (self.cell_a,),
        }
        params.update(overrides)
        return SearchObservation(**params)

    def test_grid_aligned_belief_is_uniform_and_serializable(self):
        self.assertEqual(self.prior.probabilities[self.cell_a], 0.5)
        self.assertEqual(self.prior.probabilities[self.cell_b], 0.5)
        self.assertAlmostEqual(self.prior.entropy_nats, 0.6931471805599453)
        self.assertEqual(self.prior.effective_cell_count, 2.0)
        json.dumps(self.prior.to_dict())

    def test_negative_observation_reduces_probability_in_visible_cell(self):
        update = self.updater.update(
            self.prior,
            self._observation(),
            self.grid,
        )

        self.assertEqual(update.evidence_type, "negative")
        self.assertAlmostEqual(update.posterior.probabilities[self.cell_a], 2.0 / 11.0)
        self.assertAlmostEqual(update.posterior.probabilities[self.cell_b], 9.0 / 11.0)
        self.assertGreater(update.kl_divergence_nats, 0.0)
        self.assertAlmostEqual(sum(update.posterior.probabilities.values()), 1.0)

    def test_localized_positive_detection_concentrates_target_probability(self):
        detection = TargetDetection(
            label="yellow van",
            confidence=1.0,
            estimated_position=(10.0, 10.0, 0.0),
        )

        update = self.updater.update(
            self.prior,
            self._observation(detections=(detection,)),
            self.grid,
        )

        self.assertEqual(update.evidence_type, "positive_localized")
        self.assertEqual(update.evidence_cell_ids, (self.cell_a,))
        self.assertAlmostEqual(update.posterior.probabilities[self.cell_a], 8.0 / 9.0)
        self.assertAlmostEqual(update.posterior.probabilities[self.cell_b], 1.0 / 9.0)

    def test_unlocalized_detection_supports_all_visible_cells(self):
        detection = TargetDetection(label="yellow van", confidence=1.0)

        update = self.updater.update(
            self.prior,
            self._observation(
                detections=(detection,),
                visible_cell_ids=(self.cell_a, self.cell_b),
            ),
            self.grid,
        )

        self.assertEqual(update.evidence_type, "positive_unlocalized")
        self.assertEqual(update.evidence_cell_ids, (self.cell_a, self.cell_b))
        self.assertEqual(update.posterior.probabilities, self.prior.probabilities)

    def test_zero_quality_negative_observation_is_uninformative(self):
        update = self.updater.update(
            self.prior,
            self._observation(observation_quality=0.0),
            self.grid,
        )

        self.assertAlmostEqual(update.posterior.probabilities[self.cell_a], 0.5)
        self.assertAlmostEqual(update.posterior.probabilities[self.cell_b], 0.5)
        self.assertAlmostEqual(update.entropy_reduction_nats, 0.0)

    def test_detection_below_task_threshold_is_negative_evidence(self):
        detection = TargetDetection(label="yellow van", confidence=0.4)

        update = self.updater.update(
            self.prior,
            self._observation(detections=(detection,)),
            self.grid,
            min_detection_confidence=0.5,
        )

        self.assertEqual(update.evidence_type, "negative")
        self.assertLess(update.posterior.probabilities[self.cell_a], 0.5)

    def test_excess_localization_error_is_not_positive_evidence(self):
        detection = TargetDetection(
            label="yellow van",
            confidence=1.0,
            estimated_position=(10.0, 10.0, 0.0),
            attributes={"localization_error_m": 20.0},
        )

        update = self.updater.update(
            self.prior,
            self._observation(detections=(detection,)),
            self.grid,
            max_localization_error_m=10.0,
        )

        self.assertEqual(update.evidence_type, "negative")
        self.assertLess(update.posterior.probabilities[self.cell_a], 0.5)

    def test_belief_rejects_cells_outside_searchable_grid(self):
        with self.assertRaises(ValueError):
            BeliefMap.for_grid(self.grid, {"unknown-cell": 1.0})

    def test_sensor_model_rejects_non_discriminative_parameters(self):
        with self.assertRaises(ValueError):
            BinarySensorModel(
                detection_probability=0.1,
                false_positive_probability=0.1,
            )


class BayesianSearchSessionTests(unittest.TestCase):
    def test_session_updates_belief_and_records_diagnostics(self):
        task = SearchTask.from_skill_params({
            "task_id": "belief-session",
            "area_token": "Area-A",
            "area": {
                "kind": "rectangle",
                "coords": [[0, 0], [40, 0], [40, 20], [0, 20]],
            },
            "target_token": "yellow-van",
        })
        grid = SearchGrid.from_task(task, resolution_m=20.0)
        session = SearchSession(
            task,
            CoveragePolicy(pass_spacing_m=20.0, altitude_m=30.0),
            search_grid=grid,
            belief_updater=BayesianBeliefUpdater(BinarySensorModel(
                detection_probability=0.8,
                false_positive_probability=0.1,
            )),
        )
        viewpoint = session.next_viewpoint()

        state = session.record_observation(SearchObservation(
            viewpoint=viewpoint,
            timestamp_s=1.0,
            visible_cell_ids=("Area-A:r0:c0",),
        ))

        self.assertAlmostEqual(state.belief["Area-A:r0:c0"], 2.0 / 11.0)
        self.assertEqual(state.policy_metadata["belief_update_count"], 1)
        self.assertEqual(state.policy_metadata["last_evidence_type"], "negative")
        self.assertEqual(len(session.belief_updates), 1)
        self.assertGreater(
            state.policy_metadata["cumulative_kl_divergence_nats"],
            0.0,
        )

    def test_grid_and_updater_must_be_configured_together(self):
        task = SearchTask.from_skill_params({
            "task_id": "invalid-session",
            "area_token": "Area-A",
            "area": {
                "kind": "rectangle",
                "coords": [[0, 0], [20, 0], [20, 20], [0, 20]],
            },
            "target_token": "target",
        })

        with self.assertRaises(ValueError):
            SearchSession(
                task,
                CoveragePolicy(),
                belief_updater=BayesianBeliefUpdater(),
            )

    def test_terminal_outcome_contains_belief_metrics(self):
        task = SearchTask.from_skill_params({
            "task_id": "belief-outcome",
            "area_token": "Area-A",
            "area": {
                "kind": "rectangle",
                "coords": [[0, 0], [40, 0], [40, 20], [0, 20]],
            },
            "target_token": "target",
            "max_viewpoints": 1,
        })
        grid = SearchGrid.from_task(task, resolution_m=20.0)
        session = SearchSession(
            task,
            CoveragePolicy(pass_spacing_m=20.0),
            search_grid=grid,
            belief_updater=BayesianBeliefUpdater(BinarySensorModel(
                detection_probability=0.8,
                false_positive_probability=0.1,
            )),
        )
        viewpoint = session.next_viewpoint()

        session.record_observation(SearchObservation(
            viewpoint=viewpoint,
            timestamp_s=1.0,
            visible_cell_ids=("Area-A:r0:c0",),
        ))

        platform_result = session.outcome.to_platform_result()
        metrics = platform_result["search_metrics"]
        self.assertEqual(metrics["belief_update_count"], 1)
        self.assertEqual(metrics["last_evidence_type"], "negative")
        self.assertLess(
            metrics["belief_entropy_nats"],
            metrics["initial_belief_entropy_nats"],
        )
        self.assertEqual(platform_result["final_belief"], session.state.belief)


if __name__ == "__main__":
    unittest.main()
