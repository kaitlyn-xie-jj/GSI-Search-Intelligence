import json
import tempfile
import unittest
from pathlib import Path

from modules.search_intelligence import (
    BinarySensorModel,
    SearchBenchmarkConfig,
    SearchBenchmarkRunner,
    SearchEpisodeRunner,
    Viewpoint,
    default_benchmark_scenarios,
    write_benchmark_report,
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

    def test_suite_runs_every_policy_on_every_repetition(self):
        report = SearchBenchmarkRunner(self.config).run((self.scenario,))

        self.assertEqual(len(report.episodes), 8)
        self.assertEqual(len(report.aggregates), 4)
        self.assertEqual(len(report.condition_aggregates), 4)
        self.assertEqual(
            {item.policy_name for item in report.aggregates},
            {"coverage", "random", "greedy_prior", "active"},
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
            self.assertEqual(len(payload["episodes"]), 8)
            self.assertEqual(len(payload["aggregates"]), 4)
            self.assertEqual(len(payload["condition_aggregates"]), 4)


if __name__ == "__main__":
    unittest.main()
