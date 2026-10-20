import json
from pathlib import Path
import subprocess
import sys
import unittest

from examples.differential_attention.demo import build_report

ROOT = Path(__file__).resolve().parents[1]


class DemoTests(unittest.TestCase):
    def test_checked_sample_output(self):
        path = ROOT / "examples" / "differential_attention" / "sample_output.json"
        self.assertEqual(build_report(), json.loads(path.read_text(encoding="utf-8")))

    def test_constructed_cancellation_and_both_counterexamples(self):
        report = build_report()
        row = report["last_query"]
        self.assertEqual(row["softmax_map_1"], [0.4, 0.2, 0.2, 0.2])
        self.assertEqual(row["softmax_map_2"], [0.1, 0.3, 0.3, 0.3])
        self.assertEqual(row["signed_difference_weights"], [0.3, -0.1, -0.1, -0.1])
        self.assertEqual(row["standard_attention_output"], [10, 0.4])
        self.assertEqual(row["differential_attention_output"], [0, 0.3])
        cases = report["counterexamples_last_query"]
        self.assertEqual(cases["identical_maps_output"], [0, 0])
        self.assertEqual(cases["nonconstant_background_output"], [-0.2, 0.3])

    def test_lambda_sweep_has_hand_calculated_outputs(self):
        for coefficient, target in ((0, [10, 0.4]), (0.5, [5, 0.35]), (1, [0, 0.3])):
            self.assertEqual(
                build_report(coefficient)["last_query"]["differential_attention_output"], target
            )

    def test_cli_honors_parameters(self):
        result = subprocess.run(
            [sys.executable, "-m", "examples.differential_attention.demo",
             "--lambda-value", "0.5", "--background", "20"],
            cwd=ROOT, text=True, capture_output=True, check=True,
        )
        report = json.loads(result.stdout)
        self.assertEqual(report["last_query"]["differential_attention_output"], [10, 0.35])

    def test_cli_rejects_invalid_values_without_traceback(self):
        for options in (
            ["--lambda-value", "NaN"], ["--lambda-value", "inf"],
            ["--background", "NaN"], ["--background", "inf"],
            ["--lambda-value", "wrong"],
        ):
            result = subprocess.run(
                [sys.executable, "-m", "examples.differential_attention.demo", *options],
                cwd=ROOT, text=True, capture_output=True,
            )
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("error:", result.stderr)
            self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
