import json
import math
from pathlib import Path
import subprocess
import sys
import unittest

from examples.slimulations.migration.demo import build_report, expected_difference, run_counts


class MigrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = build_report()

    def test_algebra_against_iterated_two_deme_recursion(self):
        for m in (0, .05, .5, 1):
            p, q = 1., 0.
            for t in range(21):
                self.assertAlmostEqual(p-q, expected_difference(m, t), places=13)
                self.assertAlmostEqual((p+q)/2, .5, places=13)
                p, q = (1-m)*p+m*q, (1-m)*q+m*p

    def test_initialization_has_no_hidden_generation(self):
        counts = run_counts(generations=0, n=8, replicates=4)
        self.assertEqual(counts, {0: [(16, 0)]*4})

    def test_no_migration_preserves_opposite_fixations(self):
        counts = run_counts(0, n=8, replicates=4, generations=7)
        for rows in counts.values():
            self.assertEqual(rows, [(16, 0)]*4)

    def test_complete_migration_swaps_fixed_demes(self):
        counts = run_counts(1, n=8, replicates=4, generations=7)
        for t, rows in counts.items():
            self.assertEqual(rows, ([(16, 0)] if t % 2 == 0 else [(0, 16)])*4)

    def test_ensemble_expectations_with_simultaneous_bound(self):
        tolerance = self.report["simultaneous_difference_tolerance_alpha_1e_minus_6"]
        for group in self.report["treatments"]:
            for row in group["checkpoints"]:
                self.assertLessEqual(abs(row["mean_p1_minus_p2"]-row["expected_p1_minus_p2"]), tolerance)
                self.assertLessEqual(abs(row["mean_pooled_frequency"]-.5), tolerance/2)

    def test_default_contrast_and_finite_scatter(self):
        control, migration = [group["checkpoints"][-1] for group in self.report["treatments"]]
        self.assertEqual(control["mean_p1_minus_p2"], 1)
        self.assertLess(migration["mean_p1_minus_p2"], .3)
        self.assertGreater(migration["difference_standard_error"], 0)
        self.assertLess(migration["min_realized_difference"], 0)

    def test_seed_reproducibility(self):
        kwargs = dict(n=8, replicates=8, generations=4, seed=401)
        self.assertEqual(run_counts(**kwargs), run_counts(**kwargs))

    def test_seed_changes_realizations(self):
        kwargs = dict(n=8, replicates=8, generations=4)
        self.assertNotEqual(run_counts(**kwargs, seed=401), run_counts(**kwargs, seed=402))

    def test_invalid_inputs_fail_before_execution(self):
        for kwargs in [dict(migration=math.nan), dict(migration=-.1), dict(migration=1.1),
                       dict(replicates=0), dict(n=2.2), dict(generations=-1), dict(seed=0)]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                run_counts(**kwargs, executable="nonexistent-executable")

    def test_checked_sample(self):
        self.assertEqual(self.report, json.loads(Path(__file__).with_name("sample_summary.json").read_text()))

    def test_cli_invalid(self):
        result = subprocess.run([sys.executable, "-m", "examples.slimulations.migration.demo", "--replicates", "0"],
                                capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("replicates must", result.stderr)


if __name__ == "__main__":
    unittest.main()
