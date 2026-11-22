from collections import defaultdict
from fractions import Fraction
from itertools import product
from math import comb
import shutil
import unittest

from . import demo


def exact_products():
    """Independent exact Wright–Fisher transition / trajectory oracle (2N=6)."""
    size, start = 6, 2
    frequencies = [Fraction(i, size) for i in range(size + 1)]
    paths = {(start,): Fraction(1)}
    for _ in range(3):
        out = defaultdict(Fraction)
        for history, weight in paths.items():
            p = frequencies[history[-1]]
            for k in range(size + 1):
                transition = comb(size, k) * p**k * (1-p)**(size-k)
                if transition:
                    out[(*history, k)] += weight * transition
        paths = out
    return paths, frequencies


class Tests(unittest.TestCase):
    def test_exact_markov_oracle(self):
        paths, frequencies = exact_products()
        self.assertEqual(sum(paths.values()), 1)
        intervals = [(a, b) for a in range(4) for b in range(a, 4)]
        for (a, b), (c, d) in product(intervals, repeat=2):
            expected = sum(
                weight * (frequencies[path[b]] - frequencies[path[a]])
                       * (frequencies[path[d]] - frequencies[path[c]])
                for path, weight in paths.items()
            )
            self.assertAlmostEqual(demo.model_covariance(3, 1/3, (a,b), (c,d)),
                                   float(expected), places=14)

    def test_levels_and_shared_increment(self):
        self.assertAlmostEqual(
            demo.model_covariance(20, 0.25, (0, 6), (0, 12)),
            0.25*0.75*(1-0.975**6),
        )
        self.assertEqual(demo.model_covariance(20, 0.25, (0, 6), (6, 12)), 0)
        self.assertAlmostEqual(
            demo.model_covariance(20, 0.25, (0, 12), (6, 18)),
            0.25*0.75*(0.975**6-0.975**12),
        )

    def test_parse_keeps_lost_and_fixed(self):
        complete = "DATA,1,0,2\nDATA,1,6,2\nDATA,1,12,0\nDATA,1,18,0\n"
        complete += "DATA,2,0,2\nDATA,2,6,4\nDATA,2,12,4\nDATA,2,18,4\n"
        rows = demo.parse_counts(complete, 2, 2, 2)
        self.assertEqual(rows, [[2,2,0,0], [2,4,4,4]])
        result = demo.summarize(rows, 2, 2, 42)
        self.assertEqual(result["lost_at_final_census"], 1)
        self.assertEqual(result["fixed_at_final_census"], 1)

    def test_parse_invalid(self):
        with self.assertRaisesRegex(ValueError, "missing"):
            demo.parse_counts("DATA,1,0,2\n", 2, 2, 2)
        with self.assertRaisesRegex(ValueError, "unexpected"):
            demo.parse_counts("DATA,1,0,5\n", 2, 2, 2)
        with self.assertRaisesRegex(ValueError, "unexpected"):
            demo.parse_counts("DATA,1,0,2\nDATA,1,0,2\n", 2, 2, 2)

    def test_boundaries_have_zero_covariance(self):
        for p in (0,1):
            self.assertEqual(demo.model_covariance(2, p, (0,12), (0,12)), 0)
        with self.assertRaises(ValueError):
            demo.model_covariance(2, 0.5, (2,1), (0,12))

    @unittest.skipUnless(shutil.which("slim"), "SLiM 4.0.1 not on PATH")
    def test_slim_matches_neutral_null_and_retains_boundaries(self):
        rows = demo.simulate("slim", 20, 2048, 10, demo.REFERENCE_SEED)
        self.assertEqual(len(rows), 2048)
        result = demo.summarize(rows, 20, 10, demo.REFERENCE_SEED)
        self.assertGreater(result["lost_at_final_census"], 0)
        self.assertGreater(result["fixed_at_final_census"], 0)
        for row in result["covariances"].values():
            error = abs(row["estimate"] - row["wf_expectation"])
            self.assertLess(error, 5 * row["monte_carlo_se"])
        self.assertGreater(
            result["covariances"]["levels_generations_6_and_12"]["estimate"], 0.02
        )
        self.assertLess(
            abs(result["covariances"]["disjoint_increments_0_6_and_6_12"]["estimate"]),
            0.004,
        )

    @unittest.skipUnless(shutil.which("slim"), "SLiM 4.0.1 not on PATH")
    def test_slim_initially_fixed_controls(self):
        for initial in (0, 12):
            rows = demo.simulate("slim", 6, 5, initial, demo.REFERENCE_SEED)
            self.assertEqual(rows, [[initial]*4]*5)
