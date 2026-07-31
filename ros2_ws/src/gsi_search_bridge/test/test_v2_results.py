import json
from pathlib import Path
import tempfile
import unittest

from gsi_search_bridge.v2_results import aggregate_summaries, summarize_trial


class SearchWorldV2ResultsTests(unittest.TestCase):
    def test_trial_summary_extracts_policy_and_accuracy_metrics(self):
        events = [
            {"event": "command"},
            {"event": "observation"},
            {
                "event": "outcome",
                "outcome": {
                    "status": "found",
                    "found": True,
                    "steps": 1,
                    "elapsed_time_s": 12.0,
                    "distance_travelled_m": 18.0,
                    "confidence": 0.9,
                    "estimated_target_position": [4.0, 6.0, 0.5],
                    "metrics": {"coverage_fraction": 0.25, "belief_entropy_nats": 1.2},
                },
            },
        ]
        truth = {
            "seed": 9,
            "targets": [{
                "slot_id": "slot-1",
                "semantic_region_id": "parking",
                "pose_enu_m": {"x": 1.0, "y": 2.0, "z": 0.5},
            }],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace = root / "trace.jsonl"
            trace.write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")
            ground_truth = root / "ground_truth.json"
            ground_truth.write_text(json.dumps(truth), encoding="utf-8")
            summary = summarize_trial("test-map", trace, ground_truth)

        self.assertTrue(summary["found"])
        self.assertEqual(summary["observation_count"], 1)
        self.assertEqual(summary["command_count"], 1)
        self.assertEqual(summary["localization_error_m"], 5.0)

    def test_batch_summary_includes_failures_in_success_rate(self):
        rows = [
            {"scenario": "a", "found": True, "elapsed_time_s": 10, "distance_travelled_m": 20, "localization_error_m": 1},
            {"scenario": "b", "found": False, "elapsed_time_s": None, "distance_travelled_m": None, "localization_error_m": None},
        ]
        with tempfile.TemporaryDirectory() as directory:
            aggregate = aggregate_summaries(rows, directory)
            self.assertTrue((Path(directory) / "trials.csv").is_file())
            self.assertTrue((Path(directory) / "batch_summary.json").is_file())
        self.assertEqual(aggregate["success_rate"], 0.5)
        self.assertEqual(aggregate["mean_distance_m_success"], 20.0)


if __name__ == "__main__":
    unittest.main()
