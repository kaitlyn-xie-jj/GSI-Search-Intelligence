import csv
import json
import tempfile
import unittest
from pathlib import Path

from modules.search_intelligence import (
    default_stress_profiles,
    run_stress_benchmark,
    stress_benchmark_scenarios,
    write_stress_benchmark_results,
)


class SearchStressScenarioTests(unittest.TestCase):
    def test_matrix_spans_layout_target_and_prior_axes(self):
        scenarios = stress_benchmark_scenarios()

        self.assertEqual(len(scenarios), 24)
        self.assertEqual(len({item.scenario_id for item in scenarios}), 24)
        self.assertEqual(
            {item.metadata["layout"] for item in scenarios},
            {"compact_rectangle", "large_rectangle", "l_shape"},
        )
        self.assertEqual(
            {item.metadata["target_position"] for item in scenarios},
            {"near", "far"},
        )
        self.assertEqual(
            {item.prior_condition for item in scenarios},
            {"correct", "diffuse", "uniform", "misleading"},
        )

    def test_budget_scale_changes_only_the_viewpoint_budget(self):
        nominal = stress_benchmark_scenarios(budget_scale=1.0)
        tight = stress_benchmark_scenarios(budget_scale=0.45)

        self.assertEqual(
            [item.scenario_id for item in nominal],
            [item.scenario_id for item in tight],
        )
        self.assertTrue(all(
            tight_item.task.budget.max_viewpoints
            < nominal_item.task.budget.max_viewpoints
            for nominal_item, tight_item in zip(nominal, tight)
        ))

    def test_default_profiles_cover_sensor_quality_and_budget_stress(self):
        profiles = default_stress_profiles()

        self.assertEqual(len(profiles), 5)
        self.assertEqual(
            {item.profile_id for item in profiles},
            {
                "nominal",
                "degraded_sensor",
                "low_observation_quality",
                "high_false_alarm",
                "tight_budget",
            },
        )


class SearchStressReportingTests(unittest.TestCase):
    def test_writer_creates_long_form_and_profile_artifacts(self):
        runs = run_stress_benchmark(
            default_stress_profiles()[:1],
            repetitions=1,
            base_seed=7,
            policy_names=("active",),
        )

        with tempfile.TemporaryDirectory() as directory:
            paths = write_stress_benchmark_results(runs, directory)

            self.assertTrue(all(Path(path).is_file() for path in paths.values()))
            manifest = json.loads(
                Path(paths["manifest_json"]).read_text("utf-8")
            )
            self.assertEqual(manifest["episode_count"], 24)
            with Path(paths["episodes_csv"]).open(
                newline="", encoding="utf-8"
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 24)
            self.assertIn("layout", rows[0])
            self.assertIn("observation_quality", rows[0])


if __name__ == "__main__":
    unittest.main()
