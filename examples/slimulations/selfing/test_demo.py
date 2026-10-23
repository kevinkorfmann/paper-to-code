import json
from math import comb, nan
from pathlib import Path
import subprocess
import sys
import unittest

from examples.slimulations.selfing.demo import SCRIPT, build_report, expected_heterozygosity, run_counts


class SelfingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = build_report()

    def test_selfing_formula_from_mendelian_genotype_transition(self):
        aa, hetero, other = 0., 1., 0.
        for t in range(9):
            self.assertEqual(hetero, expected_heterozygosity(1, 100, t))
            self.assertEqual(aa+hetero/2, .5)
            aa, hetero, other = aa+hetero/4, hetero/2, other+hetero/4

    def test_random_mating_second_generation_from_exact_binomial(self):
        n = 4
        exact = sum(comb(2*n, k)/2**(2*n) * 2*(k/(2*n))*(1-k/(2*n)) for k in range(2*n+1))
        self.assertAlmostEqual(exact, expected_heterozygosity(0, n, 2), places=14)
        self.assertEqual(expected_heterozygosity(0, n, 0), 1)
        self.assertEqual(expected_heterozygosity(0, n, 1), .5)

    def test_exact_initial_genotypes_and_zero_duration(self):
        for mode in (0, 1):
            self.assertEqual(run_counts(mode, n=8, replicates=4, generations=0), {0: [(8, 8)]*4})

    def test_fixation_and_absence_are_absorbing(self):
        for mode in (0, 1):
            for state in (0, 2):
                counts = run_counts(mode, n=8, replicates=4, generations=5, initial_state=state)
                for rows in counts.values():
                    self.assertEqual(rows, [(8*state, 0)]*4)

    def test_first_generation_mendelian_proportions(self):
        rows = run_counts(1, generations=1)[1]
        mean_aa = sum((copies-hetero)/2 for copies, hetero in rows)/(256*100)
        mean_hetero = sum(hetero for _, hetero in rows)/(256*100)
        mean_other = 1-mean_aa-mean_hetero
        for observed, expected in zip((mean_aa, mean_hetero, mean_other), (.25, .5, .25)):
            self.assertLess(abs(observed-expected), .03)

    def test_ensemble_means_with_simultaneous_bound(self):
        tolerance = self.report["simultaneous_mean_tolerance_alpha_1e_minus_6"]
        for treatment in self.report["treatments"]:
            for row in treatment["checkpoints"]:
                self.assertLessEqual(abs(row["mean_observed_heterozygosity"]-row["expected_observed_heterozygosity"]), tolerance)
                self.assertLessEqual(abs(row["mean_allele_frequency"]-.5), tolerance)

    def test_heterozygosity_is_not_gene_diversity(self):
        random, selfing = [t["checkpoints"][-1] for t in self.report["treatments"]]
        self.assertGreater(random["mean_observed_heterozygosity"], .4)
        self.assertLess(selfing["mean_observed_heterozygosity"], .02)
        self.assertGreater(selfing["mean_gene_diversity_2p1_minus_p"], .4)
        self.assertGreater(selfing["heterozygosity_standard_error"], 0)

    def test_seed_reproducibility_and_variability(self):
        kwargs = dict(n=8, replicates=8, generations=4)
        first = run_counts(seed=401, **kwargs)
        self.assertEqual(first, run_counts(seed=401, **kwargs))
        self.assertNotEqual(first, run_counts(seed=402, **kwargs))

    def test_invalid_python_inputs_before_engine_execution(self):
        for patch in [dict(selfing=.5), dict(selfing=nan), dict(selfing=True), dict(n=1),
                      dict(initial_state=3), dict(generations=-1), dict(replicates=0), dict(seed=0)]:
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                run_counts(**patch, executable="nonexistent-executable")

    def test_direct_eidos_validation(self):
        result = subprocess.run(["slim", "-d", "SELFING=0.5", str(SCRIPT)], capture_output=True, text=True, timeout=30)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("SELFING must be", result.stderr)

    def test_checked_sample(self):
        self.assertEqual(self.report, json.loads(Path(__file__).with_name("sample_summary.json").read_text()))

    def test_cli_and_invalid_duration(self):
        result = subprocess.run([sys.executable, "-m", "examples.slimulations.selfing.demo", "--generations", "0", "--replicates", "2"],
                                capture_output=True, text=True, check=True, timeout=30)
        report = json.loads(result.stdout)
        self.assertTrue(all(t["checkpoints"][0]["mean_observed_heterozygosity"] == 1 for t in report["treatments"]))
        bad = subprocess.run([sys.executable, "-m", "examples.slimulations.selfing.demo", "--generations", "-1"],
                             capture_output=True, text=True, timeout=30)
        self.assertNotEqual(bad.returncode, 0)


if __name__ == "__main__":
    unittest.main()
