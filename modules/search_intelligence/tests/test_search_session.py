import unittest

from modules.search_intelligence import (
    CoveragePolicy,
    SearchObservation,
    SearchOutcomeStatus,
    SearchSession,
    SearchTask,
    TargetDetection,
    Viewpoint,
)


class SearchSessionTests(unittest.TestCase):
    def _task(self, **overrides):
        params = {
            "task_id": "session-1",
            "area_token": "Parking-A",
            "area": {
                "kind": "rectangle",
                "coords": [[0, 0], [80, 0], [80, 60], [0, 60]],
            },
            "target_token": "yellow-van",
            "conf_ge": 0.8,
            "min_confirmations": 2,
            "persist_ge_s": 1.0,
            "max_viewpoints": 3,
        }
        params.update(overrides)
        return SearchTask.from_skill_params(params)

    def test_task_observation_state_and_outcome_form_closed_loop(self):
        session = SearchSession(
            self._task(),
            CoveragePolicy(pass_spacing_m=20.0, altitude_m=25.0),
        )

        first = session.next_viewpoint()
        session.record_observation(SearchObservation(
            viewpoint=first,
            timestamp_s=1.0,
            detections=(TargetDetection(
                label="yellow van", confidence=0.91, entity_id="vehicle-7"
            ),),
            travel_time_s=2.0,
        ))
        self.assertFalse(session.completed)

        second = session.next_viewpoint()
        session.record_observation(SearchObservation(
            viewpoint=second,
            timestamp_s=2.5,
            detections=(TargetDetection(
                label="yellow van",
                confidence=0.93,
                entity_id="vehicle-7",
                estimated_position=(12.0, 8.0, 0.0),
            ),),
            travel_time_s=2.0,
        ))

        self.assertTrue(session.completed)
        self.assertEqual(session.outcome.status, SearchOutcomeStatus.FOUND)
        self.assertEqual(session.outcome.steps, 2)
        self.assertEqual(
            session.outcome.to_platform_result()["targets_found"], ["vehicle-7"]
        )

    def test_budget_exhaustion_produces_terminal_outcome(self):
        session = SearchSession(
            self._task(max_viewpoints=1, min_confirmations=1),
            CoveragePolicy(pass_spacing_m=20.0),
        )
        viewpoint = session.next_viewpoint()

        session.record_observation(SearchObservation(viewpoint=viewpoint, timestamp_s=1.0))

        self.assertTrue(session.completed)
        self.assertEqual(session.outcome.status, SearchOutcomeStatus.BUDGET_EXHAUSTED)
        self.assertEqual(session.outcome.reason, "viewpoints budget exhausted")

    def test_wall_clock_time_budget_can_expire_during_an_action(self):
        session = SearchSession(
            self._task(
                time_budget_s=10.0,
                max_viewpoints=3,
                min_confirmations=1,
            ),
            CoveragePolicy(pass_spacing_m=20.0),
        )
        session.next_viewpoint()

        self.assertFalse(session.expire_time_budget(9.9))
        self.assertTrue(session.expire_time_budget(10.0))
        self.assertEqual(session.outcome.status, SearchOutcomeStatus.BUDGET_EXHAUSTED)
        self.assertEqual(session.outcome.reason, "time budget exhausted")
        self.assertEqual(session.state.elapsed_time_s, 10.0)

    def test_actual_pose_can_differ_from_commanded_viewpoint(self):
        session = SearchSession(
            self._task(max_viewpoints=2, min_confirmations=1),
            CoveragePolicy(pass_spacing_m=20.0, altitude_m=25.0),
        )
        commanded = session.next_viewpoint()
        actual = Viewpoint(
            commanded.x + 0.2,
            commanded.y - 0.1,
            commanded.z + 0.05,
            commanded.yaw,
            commanded.pitch,
        )

        state = session.record_observation(SearchObservation(
            viewpoint=actual,
            action_viewpoint_key=commanded.key,
            timestamp_s=1.0,
        ))

        self.assertEqual(state.current_viewpoint, actual)
        self.assertIn(commanded.key, state.visited_viewpoint_keys)
        self.assertNotIn(actual.key, state.visited_viewpoint_keys)

    def test_transit_observation_updates_pose_without_consuming_viewpoint(self):
        session = SearchSession(
            self._task(max_viewpoints=1, min_confirmations=1),
            CoveragePolicy(pass_spacing_m=20.0, altitude_m=25.0),
        )
        commanded = session.next_viewpoint()
        actual = Viewpoint(
            commanded.x + 5.0,
            commanded.y,
            commanded.z,
            commanded.yaw,
            commanded.pitch,
        )

        state = session.record_transit_observation(SearchObservation(
            viewpoint=actual,
            action_viewpoint_key=commanded.key,
            timestamp_s=1.0,
            visible_cell_ids=("cell-a",),
            travel_time_s=4.0,
            travel_distance_m=5.0,
        ))

        self.assertEqual(state.current_viewpoint, actual)
        self.assertEqual(state.step_index, 0)
        self.assertEqual(state.visited_viewpoint_keys, ())
        self.assertFalse(session.completed)
        self.assertEqual(session.next_viewpoint(), commanded)

    def test_hysteresis_can_keep_pending_trajectory_then_request_replan(self):
        session = SearchSession(
            self._task(max_viewpoints=2, min_confirmations=1),
            CoveragePolicy(pass_spacing_m=20.0, altitude_m=25.0),
        )
        commanded = session.next_viewpoint()
        actual = Viewpoint(
            commanded.x + 5.0,
            commanded.y,
            commanded.z,
            commanded.yaw,
            commanded.pitch,
        )

        session.record_transit_observation(
            SearchObservation(
                viewpoint=actual,
                action_viewpoint_key=commanded.key,
                timestamp_s=5.0,
                travel_time_s=5.0,
                travel_distance_m=5.0,
            ),
            replan=False,
        )

        self.assertEqual(session.pending_viewpoint, commanded)
        self.assertTrue(session.request_replan(
            "belief_distribution_changed",
            timestamp_s=5.0,
            diagnostics={"belief_total_variation": 0.2},
        ))
        self.assertIsNone(session.pending_viewpoint)
        self.assertEqual(session.state.policy_metadata["replan_count"], 1)
        self.assertEqual(
            session.state.policy_metadata["last_replan_reason"],
            "belief_distribution_changed",
        )

    def test_empty_route_produces_not_found_outcome(self):
        session = SearchSession(
            self._task(area={"kind": "unsupported"}),
            CoveragePolicy(),
        )

        self.assertIsNone(session.next_viewpoint())
        self.assertEqual(session.outcome.status, SearchOutcomeStatus.NOT_FOUND)

    def test_initial_prior_metadata_is_kept_in_policy_state(self):
        session = SearchSession(
            self._task(),
            CoveragePolicy(),
            initial_belief={"cell-a": 1.0},
            initial_policy_metadata={
                "prior_source": "llm_semantic",
                "matched_prior_labels": ("parking",),
            },
        )

        self.assertEqual(session.state.policy_metadata["prior_source"], "llm_semantic")
        self.assertEqual(
            session.state.policy_metadata["matched_prior_labels"],
            ("parking",),
        )


if __name__ == "__main__":
    unittest.main()
