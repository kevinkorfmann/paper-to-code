"""Independent allele-tuple enumeration and exact jackknife controls."""

from fractions import Fraction as F
from itertools import product
import json
import math
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np

from examples.inference_showcase.patterson_blocks import demo


def enumerated_components(site):
    """Enumerate independent allele draws; do not reuse frequency formulas."""
    positive = negative = F(0)
    for alleles in product((0, 1), repeat=4):
        probability = F(1)
        for p, allele in enumerate(alleles):
            probability *= F(int(site[p, allele]), sum(int(c) for c in site[p]))
        a, b, c, d = alleles
        if a == c and b == d and a != b:
            positive += probability
        if a == d and b == c and a != b:
            negative += probability
    return positive - negative, positive + negative


def exact_jackknife(counts, block_size):
    pairs = [enumerated_components(site) for site in counts]
    blocks = [(sum(n for n, _ in pairs[i:i+block_size]),
               sum(d for _, d in pairs[i:i+block_size]))
              for i in range(0, len(pairs), block_size)]
    total_n = sum(n for n, _ in blocks)
    total_d = sum(d for _, d in blocks)
    estimates = [(total_n - n) / (total_d - d) for n, d in blocks]
    b = len(blocks)
    mean = sum(estimates) / b
    variance = F(b - 1, b) * sum((v - mean) ** 2 for v in estimates)
    return total_n / total_d, variance, estimates


class PattersonBlocksTests(unittest.TestCase):
    def setUp(self):
        self.data = demo.fixtures()

    def test_original_exact_output(self):
        actual = demo.estimate(self.data["original"], 1)
        self.assertEqual(actual["d"], 0.5)
        self.assertEqual(actual["numerator_sum"], 4)
        self.assertEqual(actual["denominator_sum"], 8)
        self.assertAlmostEqual(actual["jackknife_se"], math.sqrt(3 / 28), places=14)
        self.assertAlmostEqual(actual["z"], math.sqrt(7 / 3), places=14)

    def test_copy_group_jackknife_preserves_information(self):
        original = demo.estimate(self.data["original"], 1)
        copied = demo.estimate(self.data["copies"], 10)
        for key in ("d", "jackknife_se", "z", "per_block_d", "leave_one_block_out_d"):
            self.assertEqual(original[key], copied[key])

    def test_pseudo_independent_copies_inflate_z(self):
        actual = demo.estimate(self.data["copies"], 1)
        self.assertEqual(actual["d"], 0.5)
        self.assertAlmostEqual(actual["jackknife_se"], math.sqrt(3 / 316), places=14)
        self.assertAlmostEqual(actual["z"], math.sqrt(79 / 3), places=13)

    def test_all_353_small_population_count_combinations(self):
        cases = 0
        for n in (1, 2, 3):
            x = demo.from_alt_counts(list(product(range(n + 1), repeat=4)), chromosomes=n)
            num, den = demo.site_components(x)
            for site, actual_n, actual_d in zip(x, num, den):
                expected_n, expected_d = enumerated_components(site)
                self.assertAlmostEqual(actual_n, float(expected_n), places=14)
                self.assertAlmostEqual(actual_d, float(expected_d), places=14)
                self.assertLessEqual(abs(actual_n), actual_d + 1e-15)
                cases += 1
        self.assertEqual(cases, 353)

    def test_polymorphic_jackknife_independent_exact_oracle(self):
        x = self.data["polymorphic"]
        expected, variance, leave_out = exact_jackknife(x, 2)
        actual = demo.estimate(x, 2)
        self.assertAlmostEqual(actual["d"], float(expected), places=14)
        self.assertAlmostEqual(actual["jackknife_se"], math.sqrt(variance), places=14)
        np.testing.assert_allclose(actual["leave_one_block_out_d"], [float(v) for v in leave_out], atol=1e-14)

    def test_nonzero_outgroup_fractional_case(self):
        x = demo.from_alt_counts([[15, 5, 15, 5]] * 2)
        n, d = demo.site_components(x)
        np.testing.assert_allclose(n, [1/4, 1/4])
        np.testing.assert_allclose(d, [25/64, 25/64])
        self.assertAlmostEqual(demo.estimate(x, 1)["d"], 16/25)

    def test_aggregate_ratio_not_mean_of_ratios(self):
        x = demo.from_alt_counts([[20, 0, 20, 0], [0, 5, 20, 0]])
        n, d = demo.site_components(x)
        self.assertEqual(float(np.mean(n/d)), 0)
        self.assertAlmostEqual(demo.estimate(x, 1)["d"], 3/5)

    def test_d_is_not_constructed_mixture_fraction(self):
        # A is alpha*C + (1-alpha)*B by construction, not a demographic model.
        for alt in (5, 15):
            x = demo.from_alt_counts([[alt, 0, 20, 0]] * 2)
            self.assertEqual(demo.estimate(x, 1)["d"], 1)

    def test_allele_reversal_invariant(self):
        x = self.data["polymorphic"]
        a, b = demo.estimate(x, 2), demo.estimate(x[:, :, ::-1], 2)
        for key in ("d", "jackknife_se", "z"):
            self.assertAlmostEqual(a[key], b[key], places=14)

    def test_A_B_swap_negates_d_and_z_preserves_se(self):
        x = self.data["polymorphic"]
        a, b = demo.estimate(x, 2), demo.estimate(x[:, [1, 0, 2, 3], :], 2)
        self.assertAlmostEqual(a["d"], -b["d"], places=14)
        self.assertAlmostEqual(a["z"], -b["z"], places=14)
        self.assertAlmostEqual(a["jackknife_se"], b["jackknife_se"], places=14)

    def test_both_pair_swaps_preserve_d(self):
        x = self.data["polymorphic"]
        self.assertAlmostEqual(demo.estimate(x, 2)["d"], demo.estimate(x[:, [1, 0, 3, 2], :], 2)["d"])

    def test_complete_block_reordering_preserves_summary(self):
        x = self.data["polymorphic"]
        reordered = x.reshape(4, 2, 4, 2)[[2, 0, 3, 1]].reshape(8, 4, 2)
        a, b = demo.estimate(x, 2), demo.estimate(reordered, 2)
        for key in ("d", "jackknife_se", "z"):
            self.assertAlmostEqual(a[key], b[key], places=14)

    def test_proportional_sample_counts_preserve_estimate(self):
        x = self.data["polymorphic"]
        scale = np.array([1, 2, 3, 4]).reshape(1, 4, 1)
        a, b = demo.estimate(x, 2), demo.estimate(x * scale, 2)
        for key in ("d", "jackknife_se", "z"):
            self.assertEqual(a[key], b[key])

    def test_monomorphic_rows_do_not_supply_information(self):
        x = demo.from_alt_counts([[0, 0, 0, 0]] * 8)
        actual = demo.estimate(x, 1)
        self.assertEqual(actual["denominator_sum"], 0)
        self.assertIsNone(actual["d"])
        self.assertIsNone(actual["jackknife_se"])
        self.assertIsNone(actual["z"])

    def test_degenerate_zero_se_has_null_z(self):
        x = demo.from_alt_counts([[20, 0, 20, 0]] * 8)
        actual = demo.estimate(x, 1)
        self.assertEqual(actual["d"], 1)
        self.assertEqual(actual["jackknife_se"], 0)
        self.assertIsNone(actual["z"])

    def test_missing_population_rejected(self):
        x = self.data["original"].copy()
        x[0, 2] = 0
        with self.assertRaises(ValueError):
            demo.estimate(x, 1)

    def test_negative_counts_rejected(self):
        x = self.data["original"].copy()
        x[0, 0, 0] = -1
        with self.assertRaises(ValueError):
            demo.estimate(x, 1)

    def test_noninteger_counts_rejected(self):
        for x in (self.data["original"].astype(float), np.ones((8, 4, 2), dtype=bool), np.full((8, 4, 2), np.nan)):
            with self.subTest(dtype=x.dtype), self.assertRaises(ValueError):
                demo.estimate(x, 1)

    def test_wrong_shape_rejected(self):
        for shape in ((0, 4, 2), (8, 3, 2), (8, 4, 3), (8, 8)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                demo.estimate(np.ones(shape, dtype=int), 1)

    def test_large_counts_rejected_without_overflow(self):
        with self.assertRaises(ValueError):
            demo.estimate(np.full((8, 4, 2), 2**63, dtype=np.uint64), 1)

    def test_incomplete_or_insufficient_blocks_rejected(self):
        for size in (0, -1, 3, 8, 9, True, 1.0, "2"):
            with self.subTest(size=size), self.assertRaises(ValueError):
                demo.estimate(self.data["original"], size)

    def test_input_not_mutated(self):
        x = self.data["polymorphic"]
        before = x.copy()
        demo.estimate(x, 2)
        np.testing.assert_array_equal(x, before)

    def test_fixture_is_exact_tenfold_contiguous_duplication(self):
        np.testing.assert_array_equal(self.data["copies"], np.repeat(self.data["original"], 10, axis=0))

    def test_sample_matches_and_json_has_no_nan(self):
        text = demo.render()
        self.assertEqual(text, (demo.HERE / "sample_output.json").read_text())
        json.loads(text, parse_constant=lambda value: self.fail(value))

    def test_cli_check_sample(self):
        result = subprocess.run([sys.executable, "-m", "examples.inference_showcase.patterson_blocks.demo", "--check-sample"], check=True, capture_output=True, text=True)
        self.assertEqual(result.stdout, "sample output matches\n")

    def test_cli_output(self):
        result = subprocess.run([sys.executable, "-m", "examples.inference_showcase.patterson_blocks.demo"], check=True, capture_output=True, text=True)
        self.assertEqual(result.stdout, demo.render())


if __name__ == "__main__":
    unittest.main()
