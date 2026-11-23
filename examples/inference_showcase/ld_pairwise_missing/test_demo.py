from fractions import Fraction
from itertools import combinations
from math import sqrt
from pathlib import Path
from subprocess import check_output
import sys
import unittest

import numpy as np

from . import demo


def independently_measured_r(a, b):
    pairs = [(int(x),int(y)) for x,y in zip(a,b) if min(x,y) >= 0]
    n = len(pairs)
    x, y = zip(*pairs)
    cov = Fraction(sum(a*b for a,b in pairs), n) - Fraction(sum(x)*sum(y), n*n)
    vx = Fraction(sum(a*a for a in x), n) - Fraction(sum(x)**2, n*n)
    vy = Fraction(sum(b*b for b in y), n) - Fraction(sum(y)**2, n*n)
    return float(cov) / sqrt(vx*vy), n


class Tests(unittest.TestCase):
    def test_fixture(self):
        full, masked = demo.fixture()
        self.assertEqual(full.shape, (3,28))
        self.assertEqual((masked[masked < 0] == -1).sum(), 24)
        self.assertTrue((full[masked < 0] == 1).all())
        self.assertTrue((full[masked >= 0] == masked[masked >= 0]).all())

    def test_masked_and_complete_pair_results_against_rational_oracle(self):
        full, masked = demo.fixture()
        for data in (full, masked, masked[:,(masked >= 0).all(axis=0)]):
            results = demo.statistic(data)
            for i,j in combinations(range(3), 2):
                r,n = independently_measured_r(data[i], data[j])
                self.assertAlmostEqual(r, results["signed_ld_matrix"][i][j], places=6)
                self.assertEqual(n, results["pair_counts"][demo.NAMES[i]+demo.NAMES[j]])

    def test_negative_quadratic_form(self):
        matrix = np.array(demo.report()["masked_pairwise"]["signed_ld_matrix"])
        vector = np.array([1,-1,1])
        self.assertAlmostEqual(vector @ matrix @ vector, -1, places=5)

    def test_eigenvalues(self):
        r = demo.report()
        self.assertEqual(r["common_complete_people"], 4)
        self.assertEqual(r["masked_pairwise"]["eigenvalues"], [-0.333333,1.666667,1.666667])
        self.assertEqual(r["same_complete_people_for_all_pairs"]["eigenvalues"], [1,1,1])
        self.assertEqual(r["synthetic_full_truth_restored"]["eigenvalues"], [0.2,1.4,1.4])

    def test_invalid_calls(self):
        with self.assertRaisesRegex(ValueError, "three rows"):
            demo.statistic([[0,1],[1,2],[0,3]])
        with self.assertRaisesRegex(ValueError, "undefined"):
            demo.statistic([[0,0],[0,1],[1,2]])

    def test_checked_output(self):
        actual = check_output([sys.executable, "-m", "examples.inference_showcase.ld_pairwise_missing.demo"])
        self.assertEqual(actual, Path(__file__).with_name("sample_output.json").read_bytes())
