import json
import tempfile
import unittest
from pathlib import Path

from modules.search_intelligence import (
    DEFAULT_UTILITY_WEIGHTS,
    OfflineOptimizationConfig,
    OfflineUtilityOptimizer,
    UtilityWeights,
    default_offline_splits,
    generate_weight_candidates,
    write_offline_optimization_result,
)


class UtilityWeightTests(unittest.TestCase):
    def test_weights_normalize_to_simplex(self):
        weights = UtilityWeights(1.0, 1.0, 0.25, 0.1).normalized()

        self.assertAlmostEqual(sum(weights.as_tuple()), 1.0)
        self.assertEqual(weights, DEFAULT_UTILITY_WEIGHTS)

    def test_candidate_generation_is_deterministic(self):
        first = generate_weight_candidates(10, seed=17)
        second = generate_weight_candidates(10, seed=17)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 10)
        self.assertEqual(first[0], DEFAULT_UTILITY_WEIGHTS)
        self.assertTrue(all(abs(item.total - 1.0) < 1e-9 for item in first))

    def test_negative_practical_improvement_is_rejected(self):
        with self.assertRaises(ValueError):
            OfflineOptimizationConfig(minimum_validation_improvement=-0.1)


class OfflineOptimizationTests(unittest.TestCase):
    def setUp(self):
        self.config = OfflineOptimizationConfig(
            candidate_count=6,
            validation_candidate_count=2,
            repetitions=1,
            base_seed=23,
        )

    def test_default_splits_hold_out_layouts(self):
        splits = default_offline_splits(self.config)

        self.assertEqual({key: len(value) for key, value in splits.items()}, {
            "train": 8,
            "validation": 8,
            "test": 8,
        })
        ids = [item.scenario_id for values in splits.values() for item in values]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(
            {item.metadata["layout"] for item in splits["test"]},
            {"l_shape"},
        )

    def test_optimizer_is_reproducible_and_validates_only_top_candidates(self):
        first = OfflineUtilityOptimizer(self.config).run()
        second = OfflineUtilityOptimizer(self.config).run()

        self.assertEqual(first, second)
        self.assertEqual(len(first.candidates), 6)
        self.assertEqual(
            sum(item.validation is not None for item in first.candidates),
            2,
        )
        self.assertIsNotNone(first.candidates[0].validation)
        self.assertEqual(
            first.candidates[0].validation_vs_default.mean_difference,
            0.0,
        )
        self.assertFalse(
            first.candidates[0].validation_vs_default.confidently_better
        )
        self.assertIn(first.selected_candidate_id, {
            item.candidate_id
            for item in first.candidates
            if item.validation is not None
        })
        self.assertEqual(set(first.selected_scores), {"train", "validation", "test"})
        self.assertEqual(set(first.default_scores), {"train", "validation", "test"})

    def test_writer_creates_auditable_and_deployable_artifacts(self):
        result = OfflineUtilityOptimizer(self.config).run()

        with tempfile.TemporaryDirectory() as directory:
            paths = write_offline_optimization_result(result, directory)

            self.assertEqual(set(paths), {
                "report_json",
                "candidates_csv",
                "split_summary_csv",
                "selected_episodes_csv",
                "selected_policy_json",
            })
            self.assertTrue(all(Path(path).is_file() for path in paths.values()))
            policy = json.loads(
                Path(paths["selected_policy_json"]).read_text("utf-8")
            )
            self.assertEqual(
                policy["schema_version"],
                "gsi-active-search-policy-weights-v1",
            )
            self.assertAlmostEqual(sum(policy["weights"].values()), 1.0)
            self.assertEqual(
                policy["ros_parameters"]["active_distance_scale_mode"],
                "map_diagonal",
            )


if __name__ == "__main__":
    unittest.main()
