import unittest

from modules.search_intelligence import (
    HIGH_RES_CAMERA_PROFILE,
    SearchBenchmarkConfig,
    SearchEpisodeRunner,
    SearchObservation,
    SearchPolicy,
    SearchState,
    SearchTask,
    SuccessConstrainedSupervisorPolicy,
    TargetDetection,
    Viewpoint,
    default_benchmark_scenarios,
)


class _FixedPolicy(SearchPolicy):
    def __init__(self, viewpoint):
        self.viewpoint = viewpoint

    def plan(self, state):
        return (self.viewpoint,)


class SuccessConstrainedSupervisorTests(unittest.TestCase):
    def setUp(self):
        self.task = SearchTask.from_skill_params({
            "task_id": "success-supervisor-test",
            "area_token": "test-area",
            "area": {
                "kind": "rectangle",
                "coords": [[0, 0], [40, 0], [40, 40], [0, 40]],
            },
            "target_token": "yellow-van",
            "max_viewpoints": 10,
            "time_budget_s": 100.0,
        })
        self.active = Viewpoint(10.0, 10.0, 30.0, 0.0)
        self.recovery = Viewpoint(30.0, 30.0, 30.0, 0.0)
        self.policy = SuccessConstrainedSupervisorPolicy(
            default_policy=_FixedPolicy(self.active),
            recovery_policy=_FixedPolicy(self.recovery),
            recovery_reserve_actions=2,
            required_quality_coverage=0.6,
            estimated_action_time_s=10.0,
        )

    def test_active_search_is_default_while_budget_is_available(self):
        state = self._state(step_index=2, elapsed_time_s=20.0)

        self.assertEqual(self.policy.select_next(state), self.active)

    def test_low_coverage_uses_recovery_at_action_reserve(self):
        state = self._state(step_index=8, elapsed_time_s=80.0)

        selected = self.policy.select_next(state)
        metadata = self.policy.decision_metadata(state, selected)

        self.assertEqual(selected, self.recovery)
        self.assertEqual(metadata["success_supervisor_mode"], "global_recovery")
        self.assertEqual(metadata["success_supervisor_reason"], "coverage_reserve_reached")

    def test_positive_detection_returns_to_active_confirmation(self):
        state = self._state(step_index=8, elapsed_time_s=80.0)
        positive = SearchObservation(
            viewpoint=state.current_viewpoint,
            timestamp_s=81.0,
            detections=(TargetDetection(
                label="yellow-van",
                confidence=0.9,
            ),),
        )
        state = SearchState(
            task=state.task,
            belief=state.belief,
            current_viewpoint=state.current_viewpoint,
            observations=(positive,),
            elapsed_time_s=state.elapsed_time_s,
            step_index=state.step_index,
        )

        self.assertEqual(self.policy.select_next(state), self.active)

    def test_benchmark_runner_supports_success_constrained_policy(self):
        runner = SearchEpisodeRunner(SearchBenchmarkConfig(
            policy_names=("success_constrained",),
            repetitions=1,
        ))

        result = runner.run(
            default_benchmark_scenarios()[0],
            "success_constrained",
        )

        self.assertTrue(result.policy_trace)
        self.assertTrue(all(
            decision["policy_name"] == "SuccessConstrainedSupervisorPolicy"
            for decision in result.policy_trace
        ))

    def test_high_resolution_profile_declares_power_model_limit(self):
        self.assertEqual(
            HIGH_RES_CAMERA_PROFILE["assumed_resolution_px"],
            [3840, 2160],
        )
        self.assertFalse(
            HIGH_RES_CAMERA_PROFILE["camera_and_compute_power_model_included"]
        )

    def _state(self, *, step_index, elapsed_time_s):
        return SearchState(
            task=self.task,
            belief={"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25},
            current_viewpoint=Viewpoint(0.0, 0.0, 30.0, 0.0),
            elapsed_time_s=elapsed_time_s,
            step_index=step_index,
        )


if __name__ == "__main__":
    unittest.main()
