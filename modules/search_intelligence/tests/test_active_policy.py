import unittest
from dataclasses import replace

from modules.search_intelligence import (
    AdaptiveActiveSearchPolicy,
    AdaptiveBeliefLookaheadPolicy,
    ActiveSearchPolicy,
    BeliefLookaheadPolicy,
    BayesianBeliefUpdater,
    BinarySensorModel,
    GreedyPriorPolicy,
    OriginalActiveSearchPolicy,
    RandomPolicy,
    SearchGrid,
    SearchObservation,
    SearchSession,
    SearchState,
    SearchTask,
    TargetDetection,
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

    def test_found_probability_separates_target_visibility_and_sensor_terms(self):
        policy = ActiveSearchPolicy(
            self.candidates,
            sensor_model=BinarySensorModel(0.9, 0.1),
            visibility_probabilities={"candidate-0": 0.5},
            detection_weight=1.0,
            information_gain_weight=0.0,
            novelty_weight=0.0,
            travel_weight=0.0,
        )

        score = next(
            item for item in policy.score_candidates(self._state())
            if item.candidate_id == "candidate-0"
        )

        self.assertEqual(score.target_probability, 0.6)
        self.assertEqual(score.visibility_probability, 0.5)
        self.assertEqual(score.sensor_detection_probability, 0.9)
        self.assertAlmostEqual(score.found_probability, 0.27)
        self.assertAlmostEqual(score.utility, 0.27)

    def test_frozen_original_active_uses_positive_observation_utility(self):
        policy = OriginalActiveSearchPolicy(
            self.candidates,
            sensor_model=BinarySensorModel(0.9, 0.1),
            detection_weight=1.0,
            information_gain_weight=0.0,
            novelty_weight=0.0,
            travel_weight=0.0,
        )

        score = policy.score_candidates(self._state())[0]

        self.assertAlmostEqual(score.detection_probability, 0.58)
        self.assertAlmostEqual(score.detection_contribution, 0.58)
        self.assertAlmostEqual(score.utility, 0.58)

    def test_reward_explanation_includes_all_weighted_contributions(self):
        state = replace(
            self._state(),
            observed_cell_quality={self.cell_ids[0]: 1.0},
        )
        policy = ActiveSearchPolicy(
            (self.candidates[0],),
            revisit_weight=0.2,
            risk_weight=0.5,
            candidate_risk_scores={"candidate-0": 0.8},
        )

        score = policy.score_candidates(state)[0]

        self.assertEqual(score.revisit_score, 1.0)
        self.assertEqual(score.risk_score, 0.8)
        self.assertAlmostEqual(score.revisit_cost_contribution, -0.2)
        self.assertAlmostEqual(score.risk_cost_contribution, -0.4)
        self.assertAlmostEqual(score.utility, sum((
            score.detection_contribution,
            score.information_gain_contribution,
            score.exploration_contribution,
            score.flight_cost_contribution,
            score.revisit_cost_contribution,
            score.risk_cost_contribution,
        )))
        explanation = policy.decision_metadata(
            state,
            self.candidates[0].viewpoint,
        )["selected_viewpoint_score"]
        self.assertIn("detection_contribution", explanation)
        self.assertIn("exploration_contribution", explanation)
        self.assertIn("flight_cost_contribution", explanation)

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

    def test_adaptive_policy_boosts_exploration_when_uncertain_and_early(self):
        state = SearchState.initial(
            self.task,
            dict.fromkeys(self.cell_ids, 1.0 / len(self.cell_ids)),
            policy_metadata={
                "initial_belief_entropy_nats": 1.0986122886681098,
                "prior_confidence": 0.1,
            },
        )
        adaptive = AdaptiveActiveSearchPolicy(self.candidates).adaptive_weight_state(state)

        self.assertGreater(
            adaptive.multipliers["information_gain"],
            adaptive.multipliers["detection"],
        )
        self.assertGreater(
            adaptive.multipliers["novelty"],
            adaptive.multipliers["detection"],
        )
        self.assertEqual(adaptive.budget_progress, 0.0)

    def test_adaptive_policy_boosts_detection_and_travel_late(self):
        early = SearchState.initial(
            self.task,
            dict(zip(self.cell_ids, (0.8, 0.15, 0.05))),
            policy_metadata={
                "initial_belief_entropy_nats": 1.0986122886681098,
                "prior_confidence": 0.9,
            },
        )
        late = SearchState(
            task=self.task,
            belief=dict(zip(self.cell_ids, (0.97, 0.02, 0.01))),
            observed_cell_quality=dict.fromkeys(self.cell_ids, 1.0),
            step_index=2,
            policy_metadata=early.policy_metadata,
        )
        policy = AdaptiveActiveSearchPolicy(self.candidates)
        early_weights = policy.adaptive_weight_state(early)
        late_weights = policy.adaptive_weight_state(late)

        self.assertGreater(
            late_weights.multipliers["detection"],
            early_weights.multipliers["detection"],
        )
        self.assertGreater(
            late_weights.multipliers["travel"],
            early_weights.multipliers["travel"],
        )
        self.assertGreater(late_weights.budget_progress, 0.5)

    def test_adaptive_weights_are_normalized_and_auditable(self):
        state = self._state()
        policy = AdaptiveActiveSearchPolicy(self.candidates)
        weights = policy.adaptive_weight_state(state).adaptive_weights
        selected = policy.select_next(state)
        metadata = policy.decision_metadata(state, selected)

        self.assertAlmostEqual(sum(weights.values()), 1.0)
        self.assertTrue(all(value >= 0.0 for value in weights.values()))
        self.assertIn("adaptive_weight_state", metadata)
        self.assertEqual(
            metadata["adaptive_weight_state"]["adaptive_weights"],
            weights,
        )

    def test_elapsed_time_drives_adaptive_budget_progress(self):
        task = SearchTask.from_skill_params({
            "task_id": "time-budget",
            "area_token": "Area-A",
            "area": {
                "kind": "rectangle",
                "coords": [[0, 0], [60, 0], [60, 20], [0, 20]],
            },
            "target_token": "yellow-van",
            "time_budget_s": 100.0,
            "max_viewpoints": 20,
        })
        state = SearchState(
            task=task,
            belief=dict(zip(self.cell_ids, (0.6, 0.3, 0.1))),
            elapsed_time_s=75.0,
        )

        weights = AdaptiveActiveSearchPolicy(
            self.candidates,
        ).adaptive_weight_state(state)

        self.assertEqual(weights.budget_progress, 0.75)

    def test_time_budget_filters_unreachable_candidates(self):
        task = SearchTask.from_skill_params({
            "task_id": "time-feasibility",
            "area_token": "Area-A",
            "area": {
                "kind": "rectangle",
                "coords": [[0, 0], [60, 0], [60, 20], [0, 20]],
            },
            "target_token": "yellow-van",
            "time_budget_s": 100.0,
        })
        near = ViewpointCandidate(
            "near",
            Viewpoint(10.0, 0.0, 30.0, 0.0),
            self.cell_ids[0],
            (self.cell_ids[0],),
        )
        far = ViewpointCandidate(
            "far",
            Viewpoint(40.0, 0.0, 30.0, 0.0),
            self.cell_ids[1],
            (self.cell_ids[1],),
        )
        state = SearchState(
            task=task,
            belief=dict(zip(self.cell_ids, (0.1, 0.8, 0.1))),
            current_viewpoint=Viewpoint(0.0, 0.0, 30.0, 0.0),
            elapsed_time_s=80.0,
        )
        policy = ActiveSearchPolicy(
            (far, near),
            planning_speed_mps=1.0,
            completion_time_reserve_s=5.0,
        )

        self.assertEqual(
            [score.candidate_id for score in policy.score_candidates(state)],
            ["near"],
        )

    def test_adaptive_lookahead_uses_bounded_diverse_candidate_pool(self):
        state = self._state()
        immediate = AdaptiveActiveSearchPolicy(
            self.candidates,
        ).score_candidates(state)
        lookahead = AdaptiveBeliefLookaheadPolicy(
            self.candidates,
            discount_factor=0.0,
            lookahead_candidate_limit=2,
        ).score_candidates(state)

        self.assertEqual(len(lookahead), 2)
        selected = {score.candidate_id for score in lookahead}
        self.assertIn(immediate[0].candidate_id, selected)
        self.assertIn("candidate-2", selected)
        self.assertTrue(all(len(score.branches) == 2 for score in lookahead))
        self.assertTrue(all(score.candidate_pool_sources for score in lookahead))

    def test_candidate_pool_includes_semantic_representative(self):
        state = self._state()
        policy = AdaptiveBeliefLookaheadPolicy(
            self.candidates,
            discount_factor=0.0,
            lookahead_candidate_limit=3,
            exploitation_fraction=1 / 3,
            exploration_fraction=1 / 3,
            semantic_fraction=1 / 3,
            semantic_regions={"candidate-1": ("street",)},
        )

        scores = policy.score_candidates(state)

        semantic = next(score for score in scores if score.candidate_id == "candidate-1")
        self.assertIn("semantic", semantic.candidate_pool_sources)

    def test_fixed_active_policy_metadata_remains_non_adaptive(self):
        state = self._state()
        policy = ActiveSearchPolicy(self.candidates)
        selected = policy.select_next(state)

        self.assertNotIn(
            "adaptive_weight_state",
            policy.decision_metadata(state, selected),
        )

    def test_lookahead_score_exposes_normalized_binary_branches(self):
        policy = BeliefLookaheadPolicy(self.candidates, discount_factor=0.8)

        score = policy.score_candidates(self._state())[0]

        self.assertEqual({branch.observation for branch in score.branches}, {
            "positive",
            "negative",
        })
        self.assertAlmostEqual(
            sum(branch.probability for branch in score.branches),
            1.0,
        )
        self.assertAlmostEqual(
            score.utility,
            score.immediate_score.utility
            + 0.8 * score.expected_continuation_utility,
        )
        self.assertTrue(all(
            branch.posterior_entropy_nats >= 0.0
            for branch in score.branches
        ))

    def test_zero_discount_reduces_lookahead_to_greedy_utility(self):
        state = self._state()
        greedy = ActiveSearchPolicy(self.candidates).score_candidates(state)
        lookahead = BeliefLookaheadPolicy(
            self.candidates,
            discount_factor=0.0,
        ).score_candidates(state)

        self.assertEqual(
            [score.candidate_id for score in lookahead],
            [score.candidate_id for score in greedy],
        )
        self.assertEqual(
            [score.utility for score in lookahead],
            [score.utility for score in greedy],
        )

    def _verification_case(
        self,
        *,
        min_confirmations=2,
        localized=True,
        verification_followup_limit=None,
    ):
        task = SearchTask.from_skill_params({
            "task_id": "active-policy-verification",
            "area_token": "Area-A",
            "area": {
                "kind": "rectangle",
                "coords": [[0, 0], [60, 0], [60, 20], [0, 20]],
            },
            "target_token": "yellow-van",
            "max_viewpoints": 3,
            "min_confirmations": min_confirmations,
        })
        detected_cell = self.cell_ids[0]
        exploratory = ViewpointCandidate(
            "exploratory",
            Viewpoint(20.0, 10.0, 30.0, 0.0),
            self.cell_ids[1],
            (self.cell_ids[1],),
        )
        near_verification = ViewpointCandidate(
            "near-verification",
            Viewpoint(10.0, 10.0, 30.0, 0.0),
            detected_cell,
            (detected_cell,),
        )
        far_verification = ViewpointCandidate(
            "far-verification",
            Viewpoint(50.0, 10.0, 30.0, 0.0),
            detected_cell,
            (detected_cell,),
        )
        policy = ActiveSearchPolicy(
            (exploratory, far_verification, near_verification),
            detection_weight=1.0,
            information_gain_weight=0.0,
            novelty_weight=0.0,
            travel_weight=0.0,
            verification_followup_limit=verification_followup_limit,
        )
        attributes = {}
        if localized:
            attributes["localized_cell_id"] = detected_cell
        state = SearchState.initial(
            task,
            {
                detected_cell: 0.01,
                self.cell_ids[1]: 0.98,
                self.cell_ids[2]: 0.01,
            },
            current_viewpoint=Viewpoint(0.0, 10.0, 30.0, 0.0),
        ).advance(SearchObservation(
            viewpoint=Viewpoint(0.0, 10.0, 30.0, 0.0),
            timestamp_s=1.0,
            detections=(TargetDetection(
                label="yellow-van",
                confidence=0.9,
                entity_id="candidate-target",
                attributes=attributes,
            ),),
            visible_cell_ids=(detected_cell,),
        ))
        return policy, state, exploratory, near_verification

    def test_pending_detection_prefers_nearest_verification_viewpoint(self):
        policy, state, exploratory, near_verification = self._verification_case()

        self.assertEqual(policy.score_candidates(state)[0].viewpoint, exploratory.viewpoint)
        self.assertEqual(policy.select_next(state), near_verification.viewpoint)
        metadata = policy.decision_metadata(state, near_verification.viewpoint)
        self.assertTrue(metadata["verification_mode"])
        self.assertEqual(metadata["verification_cell_id"], self.cell_ids[0])

    def test_verification_can_hover_directly_above_localized_detection(self):
        policy, state, _, _ = self._verification_case(
            verification_followup_limit=1,
        )
        detection = state.observations[-1].detections[0]
        localized = replace(
            detection,
            estimated_position=(12.25, 8.75, 0.5),
        )
        state = replace(
            state,
            observations=(replace(
                state.observations[-1],
                detections=(localized,),
            ),),
        )
        policy = replace(
            policy,
            verification_max_horizontal_offset_m=1.0,
        )

        selected = policy.select_next(state)

        self.assertEqual((selected.x, selected.y), (12.25, 8.75))
        self.assertEqual(selected.z, 30.0)
        self.assertTrue(policy.is_viewpoint_viable(state, selected))
        self.assertTrue(policy.decision_metadata(state, selected)["verification_mode"])

    def test_single_confirmation_task_keeps_normal_active_ranking(self):
        policy, state, exploratory, _ = self._verification_case(
            min_confirmations=1,
        )

        self.assertEqual(policy.select_next(state), exploratory.viewpoint)
        self.assertFalse(
            policy.decision_metadata(state, exploratory.viewpoint)["verification_mode"]
        )

    def test_missing_detection_localization_keeps_normal_active_ranking(self):
        policy, state, exploratory, _ = self._verification_case(localized=False)

        self.assertEqual(policy.select_next(state), exploratory.viewpoint)
        self.assertIsNone(
            policy.decision_metadata(state, exploratory.viewpoint)[
                "verification_cell_id"
            ]
        )

    def test_negative_verification_followup_resumes_active_ranking(self):
        policy, state, exploratory, near_verification = self._verification_case(
            verification_followup_limit=1,
        )
        state = state.advance(SearchObservation(
            viewpoint=near_verification.viewpoint,
            timestamp_s=2.0,
            visible_cell_ids=(self.cell_ids[0],),
        ))

        self.assertEqual(policy.select_next(state), exploratory.viewpoint)
        metadata = policy.decision_metadata(state, exploratory.viewpoint)
        self.assertFalse(metadata["verification_mode"])
        self.assertIsNone(metadata["verification_cell_id"])

    def test_unlimited_verification_keeps_pending_detection_active(self):
        policy, state, _, near_verification = self._verification_case()
        state = state.advance(SearchObservation(
            viewpoint=near_verification.viewpoint,
            timestamp_s=2.0,
            visible_cell_ids=(self.cell_ids[0],),
        ))

        self.assertEqual(
            policy.decision_metadata(state, policy.select_next(state))[
                "verification_cell_id"
            ],
            self.cell_ids[0],
        )

    def test_transit_suspect_selects_a_cross_angle_inspection_viewpoint(self):
        target_cell = self.cell_ids[1]
        anchor = ViewpointCandidate(
            "anchor",
            Viewpoint(30.0, 10.0, 30.0, 0.0),
            target_cell,
            (target_cell,),
        )
        west = ViewpointCandidate(
            "west",
            Viewpoint(10.0, 10.0, 30.0, 0.0),
            self.cell_ids[0],
            (target_cell,),
        )
        east = ViewpointCandidate(
            "east",
            Viewpoint(50.0, 10.0, 30.0, 0.0),
            self.cell_ids[2],
            (target_cell,),
        )
        policy = AdaptiveBeliefLookaheadPolicy(
            (anchor, west, east),
            transit_suspect_inspection_limit=2,
            transit_suspect_min_angle_rad=1.570796,
        )
        state = SearchState.initial(
            self.task,
            dict(zip(self.cell_ids, (0.05, 0.9, 0.05))),
            current_viewpoint=west.viewpoint,
            policy_metadata={
                "last_replan_reason": "transit_occlusion_suspect",
                "last_replan_diagnostics": {
                    "suspect_cell_id": target_cell,
                    "suspect_timestamp_s": 1.0,
                    "suspect_viewpoint_xy": (10.0, 10.0),
                },
            },
        )

        selected = policy.select_next(state)

        self.assertEqual(selected, east.viewpoint)
        metadata = policy.decision_metadata(state, selected)
        self.assertTrue(metadata["transit_suspect_inspection_mode"])
        self.assertEqual(metadata["transit_suspect_cell_id"], target_cell)


if __name__ == "__main__":
    unittest.main()
