from fractions import Fraction
from itertools import combinations, product
from math import sqrt
import os
from unittest.mock import patch
import unittest

from . import demo


class Tests(unittest.TestCase):
    def test_equal_pair_distinct_triple_targets(self):
        w0, w1 = demo.WEIGHTS
        self.assertEqual(demo.expectation(w0, 2), Fraction(25, 64))
        self.assertEqual(demo.expectation(w0, 2), demo.expectation(w1, 2))
        self.assertEqual(demo.expectation(w0, 3), Fraction(43, 256))
        self.assertEqual(demo.expectation(w1, 3), Fraction(49, 256))

    def test_distinct_sampling_oracle(self):
        for labels in product(range(4), repeat=5):
            counts = [labels.count(i) for i in range(4)]
            for k in (2, 3):
                groups = list(combinations(labels, k))
                expected = Fraction(sum(len(set(g)) == 1 for g in groups), len(groups))
                self.assertEqual(demo.sharing(counts, k), expected)
        self.assertEqual(demo.sharing((1,1,1,1), 3), 0)
        self.assertEqual(demo.sharing((0,0,0,4), 3), 1)

    def test_parser_rejects_bad_or_partial_censuses(self):
        self.assertEqual(demo.parse_counts("DATA,0,1,2,0,0\nDATA,1,0,3,0,0\n", 2, 3),
                         [(1,2,0,0),(0,3,0,0)])
        for data in ("DATA,0,1,2,0,0\n", "DATA,0,5,0,0,0\n",
                     "DATA,0,1,2,0,0\nDATA,0,1,2,0,0\n"):
            with self.assertRaises(ValueError):
                demo.parse_counts(data, 2, 3)

    def test_simulator_and_observed_contrast(self):
        x, y = demo.report(slim=os.environ.get("SLIM_FOR_TEST", "slim"))["results"]
        for result in (x,y):
            for key in ("pair", "triple"):
                row = result[key]
                self.assertLess(abs(row["mean"]-row["exact_expectation"]),
                                5*row["monte_carlo_se"])
        for name, expected in (("pair", 0), ("triple", 6/256)):
            a,b = x[name],y[name]
            se = sqrt(a["monte_carlo_se"]**2+b["monte_carlo_se"]**2)
            self.assertLess(abs((b["mean"]-a["mean"])-expected), 5*se)
        self.assertGreater(y["triple"]["mean"]-x["triple"]["mean"], 4*se)

    def test_cli_verified_sample_and_mismatch(self):
        binary = os.environ.get("SLIM_FOR_TEST", "slim")
        demo.main(["--slim", binary, "--check-sample"])
        with patch.object(demo.Path, "read_bytes", return_value=b"{}"), \
                self.assertRaisesRegex(SystemExit, "sample output mismatch"):
            demo.main(["--slim", binary, "--check-sample"])
