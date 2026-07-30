import unittest

from modules.search_intelligence import (
    ActiveSearchPolicy,
    BayesianBeliefUpdater,
    BinarySensorModel,
    GreedyPriorPolicy,
    RandomPolicy,
    SearchGrid,
    SearchObservation,
    SearchSession,
    SearchState,
    SearchTask,
    Viewpoint,
    ViewpointCandidate,
)


class CandidatePolicyTests(unittest.TestCase):
    def setUp(self):
        self.task = SearchTask.from_skill_params({
            "task_id": "active-policy",
            "area_token": "Area-A",
            "area": {
                "kind": "rectangle",
                "coords": [[0, 0], [60, 0], [60, 20], [0, 20]],
            },
            "target_token": "yellow-van",
            "max_viewpoints": 3,
        })
        self.grid = SearchGrid.from_task(self.task, resolution_m=20.0)
        self.cell_ids = tuple(cell.cell_id for cell in self.grid.searchable_cells)
        self.candidates = tuple(
            ViewpointCandidate(
                candidate_id=f"candidate-{index}",
                viewpoint=Viewpoint(cell.center[0], cell.center[1], 30.0, 0.0),
                anchor_cell_id=cell.cell_id,
                visible_cell_ids=(cell.cell_id,),
            )
            for index, cell in enumerate(self.grid.searchable_cells)
        )

    def _state(self, belief=None, current=None):
        return SearchState.initial(
            self.task,
            belief or dict(zip(self.cell_ids, (0.6, 0.3, 0.1))),
            current_viewpoint=current,
        )

    def test_active_policy_prefers_high_detection_probability(self):
        policy = ActiveSearchPolicy(
            self.candidates,
            sensor_model=BinarySensorModel(0.9, 0.1),
            detection_weight=1.0,
            information_gain_weight=0.0,
            novelty_weight=0.0,
            travel_weight=0.0,
        )

        scores = policy.score_candidates(self._state())

        self.assertEqual(scores[0].candidate_id, "candidate-0")
        self.assertAlmostEqual(scores[0].belief_mass_visible, 0.6)
        self.assertAlmostEqual(scores[0].detection_probability, 0.58)
        self.assertEqual(policy.select_next(self._state()), self.candidates[0].viewpoint)

    def test_information_gain_prefers_viewpoint_that_splits_belief(self):
        state = self._state({
            self.cell_ids[0]: 0.5,
            self.cell_ids[1]: 0.5,
            self.cell_ids[2]: 0.0,
        })
        split_candidate = self.candidates[0]
        all_candidate = ViewpointCandidate(
            candidate_id="candidate-all",
            viewpoint=Viewpoint(70.0, 10.0, 30.0, 0.0),
            anchor_cell_id=self.cell_ids[1],
            visible_cell_ids=self.cell_ids,
        )
        policy = ActiveSearchPolicy(
            (split_candidate, all_candidate),
            sensor_model=BinarySensorModel(0.9, 0.1),
            detection_weight=0.0,
            information_gain_weight=1.0,
            novelty_weight=0.0,
            travel_weight=0.0,
        )

        scores = policy.score_candidates(state)

        self.assertEqual(scores[0].candidate_id, split_candidate.candidate_id)
        self.assertGreater(scores[0].information_gain_nats, 0.0)
        self.assertAlmostEqual(scores[1].information_gain_nats, 0.0)

    def test_zero_quality_sensor_has_no_detection_information(self):
        policy = ActiveSearchPolicy(
            self.candidates,
            sensor_model=BinarySensorModel(0.9, 0.1),
            observation_quality=0.0,
            detection_weight=1.0,
            information_gain_weight=1.0,
            novelty_weight=0.0,
            travel_weight=0.0,
        )

        scores = policy.score_candidates(self._state())

        self.assertTrue(all(
            abs(score.detection_probability - 0.1) < 1e-12
            for score in scores
        ))
        self.assertTrue(all(
            abs(score.information_gain_nats) < 1e-12
            for score in scores
        ))

    def test_travel_cost_breaks_equal_information_tie(self):
        near = ViewpointCandidate(
            "near",
            Viewpoint(10.0, 10.0, 30.0, 0.0),
            self.cell_ids[0],
            (self.cell_ids[0],),
        )
        far = ViewpointCandidate(
            "far",
            Viewpoint(110.0, 10.0, 30.0, 0.0),
            self.cell_ids[0],
            (self.cell_ids[0],),
        )
        state = self._state(current=Viewpoint(0.0, 10.0, 30.0, 0.0))
        policy = ActiveSearchPolicy(
            (far, near),
            detection_weight=0.0,
            information_gain_weight=0.0,
            novelty_weight=0.0,
            travel_weight=1.0,
            distance_scale_m=100.0,
        )

        self.assertEqual(policy.select_next(state), near.viewpoint)

    def test_greedy_prior_ignores_changed_posterior(self):
        fixed_prior = dict(zip(self.cell_ids, (0.7, 0.2, 0.1)))
        changed_state = self._state(dict(zip(self.cell_ids, (0.05, 0.15, 0.8))))
        policy = GreedyPriorPolicy(self.candidates, fixed_prior)

        self.assertEqual(policy.select_next(changed_state), self.candidates[0].viewpoint)

    def test_random_policy_is_reproducible_and_skips_visited(self):
        policy = RandomPolicy(self.candidates, seed=17)
        first_plan = policy.plan(self._state())
        second_plan = policy.plan(self._state())
        observed_state = self._state().advance(SearchObservation(
            viewpoint=first_plan[0],
            timestamp_s=1.0,
        ))

        self.assertEqual(first_plan, second_plan)
        self.assertNotIn(first_plan[0], policy.plan(observed_state))

    def test_baseline_policy_decision_has_common_trace_metadata(self):
        policy = GreedyPriorPolicy(
            self.candidates,
            dict(zip(self.cell_ids, (0.6, 0.3, 0.1))),
        )
        session = SearchSession(
            self.task,
            policy,
            initial_belief=dict(zip(self.cell_ids, (0.6, 0.3, 0.1))),
        )

        viewpoint = session.next_viewpoint()
        decision = session.policy_decisions[0]

        self.assertEqual(decision["policy_name"], "GreedyPriorPolicy")
        self.assertEqual(decision["step_index"], 0)
        self.assertEqual(decision["selected_viewpoint_key"], viewpoint.key)

    def test_negative_observation_changes_active_policy_next_viewpoint(self):
        sensor = BinarySensorModel(0.9, 0.01)
        policy = ActiveSearchPolicy(
            self.candidates,
            sensor_model=sensor,
            detection_weight=1.0,
            information_gain_weight=0.0,
            novelty_weight=0.0,
            travel_weight=0.0,
        )
        session = SearchSession(
            self.task,
            policy,
            initial_belief=dict(zip(self.cell_ids, (0.6, 0.3, 0.1))),
            search_grid=self.grid,
            belief_updater=BayesianBeliefUpdater(sensor),
        )

        first = session.next_viewpoint()
        session.record_observation(SearchObservation(
            viewpoint=first,
            timestamp_s=1.0,
            visible_cell_ids=(self.cell_ids[0],),
        ))
        second = session.next_viewpoint()

        self.assertEqual(first, self.candidates[0].viewpoint)
        self.assertEqual(second, self.candidates[1].viewpoint)
        self.assertEqual(
            session.state.policy_metadata["selected_candidate_id"],
            "candidate-0",
        )
        self.assertEqual(len(session.policy_decisions), 2)


if __name__ == "__main__":
    unittest.main()
