"""Independent mathematics, genuine engine calls, controls, and exact output."""

import copy
from fractions import Fraction
import json
import math
from pathlib import Path
import subprocess
import sys
import unittest
from unittest import mock

import allel
import numpy as np

from examples.inference_showcase.scikit_allel_fst import demo


def rational_components(ac_a, ac_b):
    """Separate scalar implementation of the finite-sample biallelic formula."""
    nums, dens = [], []
    for row_a, row_b in zip(ac_a, ac_b):
        n_a, n_b = sum(row_a), sum(row_b)
        p_a, p_b = Fraction(row_a[1], n_a), Fraction(row_b[1], n_b)
        nums.append((p_a-p_b)**2-p_a*(1-p_a)/(n_a-1)-p_b*(1-p_b)/(n_b-1))
        dens.append(p_a*(1-p_b)+p_b*(1-p_a))
    return nums, dens


class HudsonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = json.loads(demo.FIXTURE.read_text())
        scenario = cls.data["scenarios"]["differentiated"]
        cls.a = np.array(scenario["population_a"])
        cls.b = np.array(scenario["population_b"])

    def test_01_fixture_regenerates_exactly(self):
        self.assertEqual(demo.generate_input(), self.data)

    def test_02_seed_controls_sample(self):
        other = demo.generate_input(demo.SEED+1)
        self.assertNotEqual(other["scenarios"]["null"]["population_a"],
                            self.data["scenarios"]["null"]["population_a"])
        for seed in (-1, True, 1.5):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                demo.generate_input(seed)

    def test_03_engine_components_match_exact_rational_formula(self):
        a = [[8, 2], [2, 7], [0, 2], [9, 0], [99, 1]]
        b = [[3, 5], [8, 4], [2, 0], [6, 0], [100, 0]]
        expected = rational_components(a, b)
        actual = demo.components(a, b)
        for got, want in zip(actual, expected):
            np.testing.assert_allclose(got, list(map(float, want)), rtol=0, atol=2e-16)

    def test_04_real_engine_is_called(self):
        with mock.patch.object(allel, "average_hudson_fst", wraps=allel.average_hudson_fst) as engine:
            demo.estimate(self.a, self.b)
            engine.assert_called_once()

    def test_05_null_and_differentiated_population_targets(self):
        report = demo.build_report(self.data)["scenarios"]
        self.assertAlmostEqual(demo.population_target(demo.BASE_FREQUENCIES, .15), 54/289)
        self.assertLess(abs(report["null"]["fst_ratio_of_sums"]), .003)
        self.assertLess(abs(report["differentiated"]["fst_ratio_of_sums"]-54/289), .01)
        self.assertGreater(report["differentiated"]["fst_ratio_of_sums"], .1)

    def test_06_independent_delete_block_jackknife(self):
        result = demo.estimate(self.a, self.b)
        nums, dens = rational_components(self.a.tolist(), self.b.tolist())
        expected = []
        for start in range(0, len(nums), demo.BLOCK_SIZE):
            keep_n = nums[:start]+nums[start+demo.BLOCK_SIZE:]
            keep_d = dens[:start]+dens[start+demo.BLOCK_SIZE:]
            expected.append(float(sum(keep_n)/sum(keep_d)))
        mean = sum(expected)/len(expected)
        se = math.sqrt((len(expected)-1)/len(expected)*sum((x-mean)**2 for x in expected))
        np.testing.assert_allclose(result["leave_one_block_out"], expected, rtol=0, atol=2e-16)
        self.assertAlmostEqual(result["block_jackknife_se"], se, places=15)
        self.assertAlmostEqual(result["fst_ratio_of_sums"], float(sum(nums)/sum(dens)), places=15)

    def test_07_population_swap_invariance(self):
        first, second = demo.estimate(self.a, self.b), demo.estimate(self.b, self.a)
        for key in ("fst_ratio_of_sums", "block_jackknife_se"):
            self.assertAlmostEqual(first[key], second[key], places=15)

    def test_08_allele_relabel_invariance(self):
        first = demo.estimate(self.a, self.b)
        second = demo.estimate(self.a[:, ::-1], self.b[:, ::-1])
        for key in ("fst_ratio_of_sums", "block_jackknife_se"):
            self.assertAlmostEqual(first[key], second[key], places=15)

    def test_09_equal_observed_frequencies_can_be_negative(self):
        result = demo.estimate([[40, 40]]*4, [[40, 40]]*4, 2)
        self.assertAlmostEqual(result["fst_ratio_of_sums"], -1/79)
        self.assertEqual(result["negative_per_locus_estimates"], 4)

    def test_10_fixed_differences_are_one(self):
        result = demo.estimate([[80, 0]]*4, [[0, 80]]*4, 2)
        self.assertEqual(result["fst_ratio_of_sums"], 1)
        self.assertEqual(result["block_jackknife_se"], 0)

    def test_11_shared_monomorphic_rows_contribute_zero(self):
        num, den = demo.components([[80, 0], [0, 80]], [[80, 0], [80, 0]])
        np.testing.assert_array_equal(num, [0, 1])
        np.testing.assert_array_equal(den, [0, 1])
        result = demo.estimate([[80, 0], [0, 80]]*2, [[80, 0], [80, 0]]*2, 2)
        self.assertEqual(result["fst_ratio_of_sums"], 1)
        self.assertEqual(result["undefined_per_locus_ratios"], 2)

    def test_12_all_shared_monomorphic_panel_is_undefined(self):
        with self.assertRaisesRegex(ValueError, "undefined"):
            demo.estimate([[80, 0]]*4, [[80, 0]]*4, 2)

    def test_13_invalid_shapes(self):
        for bad in ([], [1, 2], [[1, 2, 3]], [[1, 2]]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                demo.validate_counts(bad, [[2, 2], [2, 2]])

    def test_14_invalid_counts(self):
        for bad in ([[1., 2.]], [[-1, 3]], [[True, True]], [["1", "2"]],
                    [[2**63, 0]], [[float("nan"), 2]]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                demo.validate_counts(bad, [[2, 2]])

    def test_15_missing_or_single_call_rows_rejected(self):
        for bad in ([[0, 0]], [[1, 0]]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                demo.validate_counts(bad, [[2, 2]])

    def test_16_incomplete_or_insufficient_blocks_rejected(self):
        for size in (0, -1, 1.5, True, 59, 1200, 2400):
            with self.subTest(size=size), self.assertRaises(ValueError):
                demo.estimate(self.a, self.b, size)

    def test_17_uninformative_block_rejected(self):
        with self.assertRaisesRegex(ValueError, "every jackknife block"):
            demo.estimate([[80, 0]]*4, [[80, 0]]*2+[[0, 80]]*2, 2)

    def test_18_block_choice_changes_se_not_point_estimate(self):
        first = demo.estimate(self.a, self.b, 60)
        second = demo.estimate(self.a, self.b, 120)
        self.assertEqual(first["fst_ratio_of_sums"], second["fst_ratio_of_sums"])
        self.assertNotEqual(first["block_jackknife_se"], second["block_jackknife_se"])

    def test_19_evaluation_truth_does_not_enter_estimator(self):
        changed = copy.deepcopy(self.data)
        changed["scenarios"]["differentiated"]["frequency_offset_for_evaluation_only"] = 0
        before, after = demo.build_report(self.data), demo.build_report(changed)
        for name in before["scenarios"]:
            for key in ("fst_ratio_of_sums", "block_jackknife_se"):
                self.assertEqual(before["scenarios"][name][key], after["scenarios"][name][key])

    def test_20_aggregation_counterexample_matches_rational_result(self):
        report = demo.aggregation_counterexample()
        nums, dens = rational_components([[80, 20], [99, 1]], [[20, 80], [100, 0]])
        expected = float(20*nums[0]/(20*dens[0]+180*dens[1]))
        result = report["plus_180_rare_loci"]
        self.assertAlmostEqual(result["ratio_of_sums"], expected, places=8)
        self.assertGreater(result["ratio_of_sums"], result["mean_of_per_locus_ratios"]*8)

    def test_21_sample_and_cli_are_exact(self):
        process = subprocess.run([sys.executable, "-m", demo.__name__, "--check-sample"],
                                 text=True, capture_output=True, check=True)
        self.assertEqual(process.stdout, demo.SAMPLE.read_text())
        self.assertEqual(json.loads(process.stdout), demo.build_report(self.data))

    def test_22_cli_invalid_block_size(self):
        process = subprocess.run([sys.executable, "-m", demo.__name__, "--block-size", "59"],
                                 text=True, capture_output=True)
        self.assertEqual(process.returncode, 2)
        self.assertIn("complete, equally sized blocks", process.stderr)


if __name__ == "__main__":
    unittest.main()
