import math
import unittest

from modules.search_intelligence import (
    HybridSearchSupervisorPolicy,
    SearchObservation,
    SearchPolicy,
    SearchState,
    SearchTask,
    Viewpoint,
)


class _FixedPolicy(SearchPolicy):
    def __init__(self, viewpoint):
        self.viewpoint = viewpoint

    def plan(self, state):
        if self.viewpoint.key in state.visited_viewpoint_keys:
            return ()
        return (self.viewpoint,)


class HybridSearchSupervisorTests(unittest.TestCase):
    def setUp(self):
        self.task = SearchTask.from_skill_params({
            "task_id": "hybrid-supervisor-test",
            "area_token": "test-area",
            "area": {
                "kind": "rectangle",
                "coords": [[0, 0], [40, 0], [40, 40], [0, 40]],
            },
            "target_token": "yellow-van",
            "max_viewpoints": 10,
            "time_budget_s": 100.0,
        })
        viewpoints = tuple(
            Viewpoint(float(index * 10), 10.0, 30.0, 0.0)
            for index in range(1, 5)
        )
        self.policy = HybridSearchSupervisorPolicy(
            improved_policy=_FixedPolicy(viewpoints[0]),
            coverage_policy=_FixedPolicy(viewpoints[1]),
            random_policy=_FixedPolicy(viewpoints[2]),
            visibility_fallback_policy=_FixedPolicy(viewpoints[3]),
        )
        self.initial = SearchState.initial(
            self.task,
            {"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25},
            current_viewpoint=Viewpoint(0.0, 0.0, 30.0, 0.0),
            policy_metadata={
                "initial_belief_entropy_nats": math.log(4),
                "belief_entropy_nats": math.log(4),
            },
        )

    def test_improved_active_is_the_default_mode(self):
        decision = self.policy.mode_decision(self.initial)

        self.assertEqual(decision.mode, "improved_active")
        self.assertEqual(decision.reason, "default_improved_active")

    def test_repeated_blocked_views_trigger_visibility_fallback(self):
        state = self.initial
        for index in range(2):
            state = state.advance(SearchObservation(
                viewpoint=Viewpoint(float(index), 0.0, 30.0, 0.0),
                timestamp_s=float(index + 1),
                visible_cell_ids=("a",),
                observation_quality=0.8,
                visibility_probability=0.0,
                negative_update_strength=0.0,
                negative_update_rejection_reason="blocked_view",
            ))

        decision = self.policy.mode_decision(state)

        self.assertEqual(decision.mode, "visibility_fallback")
        self.assertEqual(decision.reason, "visibility_model_unreliable")

    def test_minimum_residence_prevents_immediate_mode_switch(self):
        state = SearchState(
            task=self.task,
            belief=self.initial.belief,
            current_viewpoint=self.initial.current_viewpoint,
            step_index=3,
            policy_metadata={
                **self.initial.policy_metadata,
                "hybrid_mode": "coverage_fallback",
                "hybrid_mode_entered_step": 2,
            },
        )

        decision = self.policy.mode_decision(state)

        self.assertEqual(decision.mode, "coverage_fallback")
        self.assertEqual(decision.reason, "minimum_mode_residence")
        self.assertFalse(decision.switched)

    def test_decision_metadata_exposes_mode_and_delegated_policy(self):
        viewpoint = self.policy.select_next(self.initial)
        metadata = self.policy.decision_metadata(self.initial, viewpoint)

        self.assertEqual(metadata["policy_name"], "HybridSearchSupervisorPolicy")
        self.assertEqual(metadata["hybrid_mode"], "improved_active")
        self.assertIn("delegated_policy_metadata", metadata)


if __name__ == "__main__":
    unittest.main()
