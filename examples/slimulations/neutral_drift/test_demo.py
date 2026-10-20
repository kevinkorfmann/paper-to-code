import json
from pathlib import Path
import shutil
import subprocess
import sys
import unittest

from examples.slimulations.neutral_drift.demo import (
    SCRIPT, build_report, check_slim_version, expected_diversity, run_counts, summarize,
)


class FormulaTests(unittest.TestCase):
    def test_initial_and_absorbing_states(self):
        for n in (2, 20, 100):
            self.assertEqual(expected_diversity(n, n, 0), 0.5)
            for t in (0, 1, 80):
                self.assertEqual(expected_diversity(n, 0, t), 0)
                self.assertEqual(expected_diversity(n, 2 * n, t), 0)

    def test_one_step_and_population_size_effect(self):
        self.assertAlmostEqual(expected_diversity(20, 20, 1), 0.4875)
        self.assertLess(expected_diversity(20, 20, 80), expected_diversity(100, 100, 80))

    def test_summary_counts_and_standard_error(self):
        row = summarize({1: [0, 2, 4]}, 2, 2)[0]
        self.assertEqual(row["mean_allele_frequency"], 0.5)
        self.assertAlmostEqual(row["mean_gene_diversity"], 1 / 6, places=8)
        self.assertEqual([row[k] for k in (
            "lost_replicates", "segregating_replicates", "fixed_replicates"
        )], [1, 1, 1])
        self.assertAlmostEqual(row["gene_diversity_standard_error"], 1 / 6, places=8)
        self.assertIsNone(summarize({0: [2]}, 2, 2)[0]["gene_diversity_standard_error"])

    def test_invalid_parameters_fail_before_executable_runs(self):
        for kwargs in ({"n": 0}, {"n": True}, {"replicates": 0}, {"generations": -1},
                       {"generations": 501}, {"seed": 0}, {"initial_copies": 41}):
            with self.assertRaises(ValueError):
                run_counts(**kwargs, executable="not-an-executable")


class SlimIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if shutil.which("slim") is None:
            raise RuntimeError("Install the pinned SLiM 4.0.1; these integration tests must not be skipped")
        cls.report = build_report()

    def test_real_engine_version(self):
        self.assertEqual(check_slim_version("slim"), "4.0.1")

    def test_initialization_is_before_the_first_counted_generation(self):
        counts = run_counts(n=4, replicates=16, generations=1)
        self.assertEqual(counts[0], [4] * 16)
        self.assertEqual(set(counts), {0, 1})
        self.assertNotEqual(counts[1], counts[0])

    def test_zero_horizon_and_absorbing_alleles(self):
        self.assertEqual(run_counts(4, 16, 0), {0: [4] * 16})
        for initial in (0, 8):
            for row in run_counts(4, 16, 10, initial_copies=initial).values():
                self.assertEqual(row, [initial] * 16)

    def test_fixed_seed_repeats(self):
        self.assertEqual(run_counts(4, 16, 10, 123), run_counts(4, 16, 10, 123))
        self.assertNotEqual(run_counts(4, 16, 10, 123), run_counts(4, 16, 10, 124))

    def test_replicate_means_against_distribution_free_tolerance(self):
        tolerance = self.report["simultaneous_hoeffding_tolerance_alpha_1e_minus_6"]
        for group in self.report["populations"]:
            for row in group["checkpoints"]:
                self.assertLessEqual(abs(row["mean_gene_diversity"] - row["expected_gene_diversity"]), tolerance)
                # p lies in [0,1], twice the range of H, so its bound is twice as wide.
                self.assertLessEqual(abs(row["mean_allele_frequency"] - 0.5), 2 * tolerance)
                self.assertEqual(row["fixed_replicates"] + row["lost_replicates"] + row["segregating_replicates"], 256)
        small, large = self.report["populations"]
        self.assertLess(small["checkpoints"][-1]["mean_gene_diversity"], large["checkpoints"][-1]["mean_gene_diversity"])

    def test_checked_output_and_cli(self):
        reference = json.loads(Path(__file__).with_name("sample_summary.json").read_text())
        self.assertEqual(self.report, reference)
        result = subprocess.run(
            [sys.executable, "-m", "examples.slimulations.neutral_drift.demo",
             "--replicates", "8", "--generations", "0"],
            capture_output=True, text=True, check=True,
        )
        self.assertEqual(json.loads(result.stdout)["replicates_per_population_size"], 8)

    def test_eidos_validates_direct_cli_inputs(self):
        result = subprocess.run(["slim", "-d", "N=0", str(SCRIPT)], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("N must be", result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
