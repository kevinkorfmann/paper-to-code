"""Independent pair-count oracle and native scikit-allel checks."""
from fractions import Fraction
from itertools import combinations
import math
import subprocess
import sys
import unittest

import allel
import numpy as np

from examples.inference_showcase.pbs_ranking import demo


def oracle_fst(rows, i, j, n=20):
    # Count differing ordered draws directly, using replacement between
    # populations, no replacement within either population.
    between, within = Fraction(0), Fraction(0)
    for row in rows:
        x, y = row[i], row[j]
        between += Fraction(x * (n-y) + (n-x) * y, n*n)
        within += Fraction(x * (n-x) + y * (n-y), n*(n-1))
    return (between - within) / between


def oracle_pbs(fst):
    cap = Fraction(99999, 100000)
    ab, ac, bc = (-math.log1p(-float(min(cap, max(Fraction(0), f)))) for f in fst)
    raw = (ab + ac - bc) / 2
    return raw, raw / (1 + (ab + ac + bc) / 2)


class Tests(unittest.TestCase):
    def setUp(self):
        self.ac = demo.allele_counts()
        self.raw = allel.pbs(*self.ac, window_size=3, normed=False)
        self.default = allel.pbs(*self.ac, window_size=3)

    def test_input_shape_and_call_totals(self):
        for ac in self.ac:
            self.assertEqual(ac.shape, (12, 2))
            self.assertTrue(np.all(ac.sum(axis=1) == 20))
            self.assertTrue(np.all(ac > 0))

    def test_pairwise_hudson_oracle(self):
        for i, j in combinations(range(3), 2):
            native = allel.moving_hudson_fst(self.ac[i], self.ac[j], size=3)
            expected = [float(oracle_fst(demo.ALT[k:k+3], i, j)) for k in range(0, 12, 3)]
            np.testing.assert_allclose(native, expected, atol=1e-15, rtol=0)

    def test_independent_pbs_oracle(self):
        for k in range(4):
            rows = demo.ALT[3*k:3*k + 3]
            fst = [oracle_fst(rows, i, j) for i, j in combinations(range(3), 2)]
            raw, normalized = oracle_pbs(fst)
            self.assertAlmostEqual(self.raw[k], raw, places=14)
            self.assertAlmostEqual(self.default[k], normalized, places=14)

    def test_default_matches_explicit_normalization(self):
        np.testing.assert_array_equal(self.default, allel.pbs(*self.ac, window_size=3, normed=True))

    def test_rank_reversal(self):
        self.assertEqual(np.argmax(self.raw), 0)
        self.assertEqual(np.argmax(self.default), 1)
        self.assertGreater(self.raw[0] - self.raw[1], .07)
        self.assertGreater(self.default[1] - self.default[0], .09)

    def test_negative_pbs_with_positive_fst(self):
        rows = demo.ALT[6:9]
        self.assertTrue(all(oracle_fst(rows, i, j) > 0 for i, j in combinations(range(3), 2)))
        self.assertLess(self.raw[2], 0)
        self.assertLess(self.default[2], 0)

    def test_equal_counts_negative_fst_clips_to_zero(self):
        self.assertEqual(oracle_fst(demo.ALT[9:12], 0, 1), -Fraction(1, 19))
        self.assertEqual(self.raw[3], 0)
        self.assertEqual(self.default[3], 0)

    def test_fixed_difference_upper_clip(self):
        ac = demo.allele_counts(((20, 0, 10),))
        raw = allel.pbs(*ac, window_size=1, normed=False)[0]
        self.assertAlmostEqual(raw, -.5 * math.log1p(-.99999), places=13)

    def test_joint_ref_alt_swap_invariant(self):
        np.testing.assert_allclose(self.raw, allel.pbs(*(a[:, ::-1] for a in self.ac), window_size=3,
                                                           normed=False), atol=1e-14, rtol=0)

    def test_swap_reference_groups_invariant(self):
        a, b, c = self.ac
        np.testing.assert_allclose(self.default, allel.pbs(a, c, b, window_size=3), atol=1e-14, rtol=0)

    def test_cli_replay(self):
        output = subprocess.check_output([sys.executable, "-m", demo.__spec__.name])
        self.assertEqual(output, demo.SAMPLE.read_bytes())
        result = subprocess.run([sys.executable, "-m", demo.__spec__.name, "--check-sample"],
                                check=True, capture_output=True, text=True)
        self.assertIn("matches", result.stdout)


if __name__ == "__main__":
    unittest.main()
