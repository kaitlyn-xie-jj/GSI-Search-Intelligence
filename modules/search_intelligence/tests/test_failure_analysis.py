import unittest

from modules.search_intelligence import analyze_detection_failures


class DetectionFailureAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = analyze_detection_failures(repetitions=1, base_seed=10)

    def test_taxonomy_is_exhaustive_and_mutually_exclusive(self):
        category_total = sum(
            item["count"]
            for item in self.payload["failure_summary"].values()
        )

        self.assertEqual(category_total, self.payload["failure_count"])
        self.assertEqual(
            self.payload["success_count"] + category_total,
            self.payload["episode_count"],
        )

    def test_probability_product_reconstructs_success_rate(self):
        probabilities = self.payload[
            "probability_decomposition"
        ]["probabilities"]

        self.assertAlmostEqual(
            probabilities["product_to_success"],
            self.payload["success_rate"],
        )

    def test_analysis_does_not_change_the_model(self):
        self.assertFalse(self.payload["model_changed"])


if __name__ == "__main__":
    unittest.main()
