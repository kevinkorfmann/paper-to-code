"""Check the claims and exact output, including CLI behavior."""
import subprocess
import sys
import unittest

from examples.causal_index_routing import demo


class DemoTests(unittest.TestCase):
    def test_constructed_routing_and_recall(self):
        report = demo.build_report()["cluster_cancellation"]
        self.assertEqual(report["native_scores"], (9, 0, 4, 4))
        self.assertEqual(report["contiguous_selection"]["selected_ids"], (2,))
        self.assertEqual(report["similarity_regrouping"]["selected_ids"], (0,))
        self.assertEqual(report["full_budget_selection"]["selected_ids"], (0,))
        self.assertEqual(report["exhaustive_top_1"], (0,))

    def test_causal_leakage_is_not_silently_allowed(self):
        report = demo.build_report()["causality"]
        self.assertEqual(report["correct_before"], (0,))
        self.assertEqual(report["correct_after_future_append"], (0,))
        self.assertEqual(report["future_derived_means"], ((-49,), (1,)))
        self.assertEqual(report["wrong_global_route_then_mask_candidates"], [1])
        self.assertTrue(report["future_snapshot_rejected"])

    def test_reserved_does_not_mean_selected_and_truncation_can_miss(self):
        report = demo.build_report()
        self.assertEqual(report["reserved_candidate_slots"]["reserved_ids"], (0, 3))
        self.assertEqual(report["reserved_candidate_slots"]["candidate_ids"], (0, 3, 1))
        self.assertEqual(report["reserved_candidate_slots"]["selected_ids"], (3,))
        self.assertEqual(report["truncation_counterexample"]["selection"]["selected_ids"], (0,))
        self.assertEqual(report["truncation_counterexample"]["exhaustive_top_1"], (1,))

    def test_checked_sample_and_cli(self):
        expected = demo.SAMPLE.read_text(encoding="utf-8")
        self.assertEqual(demo.render(demo.build_report()), expected)
        result = subprocess.run([sys.executable, "-m", "examples.causal_index_routing.demo"],
                                check=True, capture_output=True, text=True)
        self.assertEqual(result.stdout, expected)
        result = subprocess.run([sys.executable, "-m", "examples.causal_index_routing.demo",
                                 "--check-sample"], check=True, capture_output=True, text=True)
        self.assertEqual(result.stdout, "sample output matches\n")

    def test_unknown_cli_flag_fails(self):
        result = subprocess.run([sys.executable, "-m", "examples.causal_index_routing.demo",
                                 "--rewrite-sample"], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
