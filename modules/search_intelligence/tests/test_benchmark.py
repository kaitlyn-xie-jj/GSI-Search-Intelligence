import json
import tempfile
import unittest
from pathlib import Path

from modules.search_intelligence import (
    BinarySensorModel,
    AdaptiveBeliefLookaheadPolicy,
    OriginalActiveSearchPolicy,
    SearchBenchmarkConfig,
    SearchBenchmarkRunner,
    SearchEpisodeRunner,
    Viewpoint,
    default_benchmark_scenarios,
    stress_benchmark_scenarios,
    write_benchmark_report,
    run_unified_benchmark,
    search_skill_scenario_matrix,
)
from modules.search_intelligence.evaluation import (
    SearchBenchmarkScenario,
    estimate,
    focused_grid_belief,
)


class SearchBenchmarkContractTests(unittest.TestCase):
    def setUp(self):
        self.scenario = default_benchmark_scenarios()[0]

    def test_default_scenarios_cover_prior_quality_conditions(self):
        scenarios = default_benchmark_scenarios()

        self.assertEqual(len(scenarios), 4)
        self.assertEqual(
            {item.prior_condition for item in scenarios},
            {"correct", "uniform", "noisy", "misleading"},
        )
        self.assertEqual(len({item.scenario_id for item in scenarios}), 4)

    def test_scenario_rejects_target_outside_searchable_grid(self):
        with self.assertRaises(ValueError):
            SearchBenchmarkScenario(
                scenario_id="invalid",
                task=self.scenario.task,
                grid=self.scenario.grid,
                target_cell_id="missing-cell",
                initial_belief=self.scenario.initial_belief,
                start_xy=(0.0, 0.0),
            )

    def test_focused_belief_is_normalized(self):
        target_id = self.scenario.target_cell_id

        belief = focused_grid_belief(
            self.scenario.grid,
            (target_id,),
            0.7,
        )

        self.assertAlmostEqual(sum(belief.values()), 1.0)
        self.assertAlmostEqual(belief[target_id], 0.7)

    def test_metric_estimate_contains_95_percent_interval(self):
        metric = estimate((0.0, 1.0, 1.0, 0.0), bounded=True)

        self.assertEqual(metric.mean, 0.5)
        self.assertEqual(metric.sample_count, 4)
        self.assertGreaterEqual(metric.ci95_low, 0.0)
        self.assertLessEqual(metric.ci95_high, 1.0)

    def test_map_diagonal_distance_scale_is_layout_relative(self):
        config = SearchBenchmarkConfig(distance_scale_mode="map_diagonal")
        scenario = default_benchmark_scenarios()[0]

        self.assertAlmostEqual(config.distance_scale_for(scenario), 128.0624847)

    def test_unknown_distance_scale_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            SearchBenchmarkConfig(distance_scale_mode="unknown")

    def test_default_policy_set_is_unchanged_by_hybrid_experiment(self):
        self.assertEqual(
            SearchBenchmarkConfig().policy_names,
            (
                "coverage",
                "random",
                "greedy_prior",
                "active",
                "adaptive_active",
                "lookahead_active",
            ),
        )

    def test_search_skill_matrix_covers_all_required_conditions(self):
        scenarios = search_skill_scenario_matrix()

        self.assertEqual(len(scenarios), 24)
        self.assertEqual(
            {item.metadata["environment"] for item in scenarios},
            {"open_area", "street_edge", "woodland", "building_passage"},
        )
        self.assertEqual(
            {item.prior_condition for item in scenarios},
            {"correct", "wrong", "uniform"},
        )
        self.assertEqual(
            {item.metadata["sensor_condition"] for item in scenarios},
            {"normal", "reduced_quality"},
        )

    def test_unified_report_contains_required_acceptance_metrics(self):
        payload = run_unified_benchmark(repetitions=1)

        self.assertEqual(payload["configuration"]["episode_count"], 96)
        self.assertEqual(
            set(payload["policy_results"]),
            {"coverage", "random", "active", "improved_active"},
        )
        improved = payload["policy_results"]["improved_active"]
        self.assertIn("success_rate", improved)
        self.assertIn("mean_detection_distance_m", improved)
        self.assertIn("mean_detection_time_s", improved)
        self.assertIn("success_per_km", improved)
        self.assertIn("replans", improved)
        self.assertIn("belief_calibration_brier", improved)
        self.assertIn("worst_case_scenario_success_rate", improved)
        self.assertEqual(
            payload["method_isolation"]["active"],
            {
                "implementation": "OriginalActiveSearchPolicy",
                "confidence_gating_enabled": False,
            },
        )
        self.assertTrue(
            payload["method_isolation"]["improved_active"][
                "confidence_gating_enabled"
            ]
        )
        self.assertIn("comparison_against_original_active", payload)


class SearchBenchmarkRunnerTests(unittest.TestCase):
    def setUp(self):
        self.scenario = default_benchmark_scenarios()[0]
        self.config = SearchBenchmarkConfig(
            repetitions=2,
            base_seed=13,
            sensor_model=BinarySensorModel(1.0, 0.0),
        )

    def test_episode_is_reproducible(self):
        runner = SearchEpisodeRunner(self.config)

        first = runner.run(self.scenario, "active", repetition=1)
        second = runner.run(self.scenario, "active", repetition=1)

        self.assertEqual(first, second)
        self.assertTrue(first.target_found)
        self.assertFalse(first.false_positive)
        self.assertEqual(len(first.belief_entropy_trace), first.steps + 1)

    def test_unified_baseline_c_and_improved_d_use_isolated_implementations(self):
        runner = SearchEpisodeRunner(self.config)
        candidates = runner._candidates(self.scenario)

        baseline = runner._policy("active", self.scenario, candidates, seed=1)
        improved = runner._policy(
            "improved_active",
            self.scenario,
            candidates,
            seed=1,
        )

        self.assertIsInstance(baseline, OriginalActiveSearchPolicy)
        self.assertIsInstance(improved, AdaptiveBeliefLookaheadPolicy)

    def test_adaptive_episode_is_reproducible_and_logs_weights(self):
        runner = SearchEpisodeRunner(self.config)

        first = runner.run(self.scenario, "adaptive_active", repetition=1)
        second = runner.run(self.scenario, "adaptive_active", repetition=1)

        self.assertEqual(first, second)
        self.assertTrue(first.policy_trace)
        self.assertTrue(all(
            "adaptive_weight_state" in decision
            for decision in first.policy_trace
        ))

    def test_lookahead_episode_is_reproducible_and_logs_value_decomposition(self):
        runner = SearchEpisodeRunner(self.config)

        first = runner.run(self.scenario, "lookahead_active", repetition=1)
        second = runner.run(self.scenario, "lookahead_active", repetition=1)

        self.assertEqual(first, second)
        self.assertTrue(first.policy_trace)
        selected = first.policy_trace[0]["selected_viewpoint_score"]
        self.assertIn("branches", selected)
        self.assertIn("expected_continuation_utility", selected)

    def test_suite_runs_every_policy_on_every_repetition(self):
        report = SearchBenchmarkRunner(self.config).run((self.scenario,))

        self.assertEqual(len(report.episodes), 12)
        self.assertEqual(len(report.aggregates), 6)
        self.assertEqual(len(report.condition_aggregates), 6)
        self.assertEqual(
            {item.policy_name for item in report.aggregates},
            {
                "coverage",
                "random",
                "greedy_prior",
                "active",
                "adaptive_active",
                "lookahead_active",
            },
        )
        self.assertTrue(all(item.episode_count == 2 for item in report.aggregates))

    def test_false_positive_is_not_counted_as_ground_truth_success(self):
        config = SearchBenchmarkConfig(
            policy_names=("coverage",),
            repetitions=1,
            base_seed=0,
            footprint_radius_m=0.0,
            sensor_model=BinarySensorModel(1.0, 0.999999),
        )

        result = SearchEpisodeRunner(config).run(
            self.scenario,
            "coverage",
        )

        self.assertTrue(result.declared_found)
        self.assertTrue(result.false_positive)
        self.assertFalse(result.target_found)
        self.assertEqual(result.spl, 0.0)

    def test_zero_observation_quality_uses_effective_sensor_probability(self):
        config = SearchBenchmarkConfig(
            policy_names=("coverage",),
            repetitions=1,
            base_seed=0,
            footprint_radius_m=1000.0,
            observation_quality=0.0,
            sensor_model=BinarySensorModel(1.0, 0.0),
        )

        result = SearchEpisodeRunner(config).run(
            self.scenario,
            "coverage",
        )

        self.assertFalse(result.declared_found)
        self.assertFalse(result.target_found)

    def test_independent_false_alarms_have_distinct_entity_ids(self):
        config = SearchBenchmarkConfig(
            policy_names=("active",),
            repetitions=1,
            sensor_model=BinarySensorModel(1.0, 0.999999999999),
        )
        runner = SearchEpisodeRunner(config)

        first = runner._detections(
            self.scenario,
            Viewpoint(10.0, 10.0, 30.0, 0.0),
            (),
            seed=1,
        )
        second = runner._detections(
            self.scenario,
            Viewpoint(30.0, 10.0, 30.0, 0.0),
            (),
            seed=1,
        )

        self.assertEqual(len(first), 1)
        self.assertEqual(len(second), 1)
        self.assertNotEqual(first[0].entity_id, second[0].entity_id)

    def test_persistent_distractor_keeps_identity_across_viewpoints(self):
        scenario = stress_benchmark_scenarios(min_confirmations=2)[0]
        distractor_cell_id = scenario.metadata["distractor_cell_id"]
        runner = SearchEpisodeRunner(SearchBenchmarkConfig(
            policy_names=("active",),
            persistent_distractor_probability=1.0,
        ))

        first = runner._detections(
            scenario,
            Viewpoint(10.0, 10.0, 30.0, 0.0),
            (distractor_cell_id,),
            seed=3,
        )
        second = runner._detections(
            scenario,
            Viewpoint(30.0, 10.0, 30.0, 0.0),
            (distractor_cell_id,),
            seed=3,
        )

        self.assertEqual(first[0].entity_id, second[0].entity_id)
        self.assertEqual(
            first[0].attributes["source_kind"],
            "persistent_distractor",
        )

    def test_correlated_false_alarms_can_share_identity(self):
        runner = SearchEpisodeRunner(SearchBenchmarkConfig(
            policy_names=("active",),
            sensor_model=BinarySensorModel(1.0, 0.999999999999),
            false_alarm_correlation=1.0,
            correlated_false_alarm_shared_identity=True,
        ))

        first = runner._detections(
            self.scenario,
            Viewpoint(10.0, 10.0, 30.0, 0.0),
            (),
            seed=7,
        )
        second = runner._detections(
            self.scenario,
            Viewpoint(30.0, 10.0, 30.0, 0.0),
            (),
            seed=7,
        )

        self.assertEqual(first[0].entity_id, second[0].entity_id)
        self.assertEqual(
            first[0].attributes["source_kind"],
            "correlated_false_alarm",
        )

    def test_localization_noise_is_deterministic_and_auditable(self):
        runner = SearchEpisodeRunner(SearchBenchmarkConfig(
            policy_names=("active",),
            sensor_model=BinarySensorModel(1.0, 0.0),
            localization_error_std_m=10.0,
        ))
        viewpoint = Viewpoint(10.0, 10.0, 30.0, 0.0)

        first = runner._detections(
            self.scenario,
            viewpoint,
            (self.scenario.target_cell_id,),
            seed=11,
        )
        second = runner._detections(
            self.scenario,
            viewpoint,
            (self.scenario.target_cell_id,),
            seed=11,
        )

        self.assertEqual(first, second)
        self.assertGreater(first[0].attributes["localization_error_m"], 0.0)
        self.assertEqual(first[0].attributes["source_kind"], "target")

    def test_episode_records_sensor_trace_and_detection_counts(self):
        result = SearchEpisodeRunner(self.config).run(self.scenario, "active")

        self.assertEqual(len(result.sensor_trace), result.steps)
        self.assertEqual(
            result.detection_count,
            sum(len(item["detections"]) for item in result.sensor_trace),
        )

    def test_report_writer_creates_json_and_csv_artifacts(self):
        report = SearchBenchmarkRunner(self.config).run((self.scenario,))

        with tempfile.TemporaryDirectory() as directory:
            paths = write_benchmark_report(report, directory)

            self.assertEqual(set(paths), {
                "report_json",
                "episodes_csv",
                "aggregates_csv",
            })
            self.assertTrue(all(Path(path).is_file() for path in paths.values()))
            payload = json.loads(Path(paths["report_json"]).read_text("utf-8"))
            self.assertEqual(len(payload["episodes"]), 12)
            self.assertEqual(len(payload["aggregates"]), 6)
            self.assertEqual(len(payload["condition_aggregates"]), 6)


if __name__ == "__main__":
    unittest.main()
