from fractions import Fraction
from itertools import product
from math import factorial
import shutil
import unittest

from . import demo


class Tests(unittest.TestCase):
    def test_ld(self):
        self.assertEqual(demo.ld((5,5,5,5)), 0)
        self.assertEqual(demo.ld((10,0,0,10)), 0.25)
        self.assertEqual(demo.ld((0,10,10,0)), -0.25)
        self.assertEqual(demo.ld((0,0,0,20)), 0)
        with self.assertRaises(ValueError):
            demo.ld((0,0,0,0))

    def test_exact_multinomial(self):
        # Oracle sums exact multinomial probabilities over all count states.
        # Compute D directly from marginals rather than via the determinant.
        for n in range(1, 13):
            mass, first, second = Fraction(0), Fraction(0), Fraction(0)
            for a, b, c in product(range(n+1), repeat=3):
                d = n-a-b-c
                if d < 0:
                    continue
                count = factorial(n) // (factorial(a)*factorial(b)*factorial(c)*factorial(d))
                weight = Fraction(count, 4**n)
                ld = Fraction(a,n) - Fraction(a+b,n)*Fraction(a+c,n)
                mass += weight
                first += weight*ld
                second += weight*ld**2
            self.assertEqual(mass, 1)
            self.assertEqual(first, 0)
            self.assertAlmostEqual(float(second), demo.second_moment(n), places=15)

    def test_founder_and_all_offspring_counts(self):
        good = ("DATA,1,0,1,1,1,1\nDATA,1,1,4,0,0,0\n"
                "DATA,2,0,1,1,1,1\nDATA,2,1,0,0,0,4\n")
        self.assertEqual(demo.parse_counts(good, 2, 2), [(4,0,0,0), (0,0,0,4)])
        with self.assertRaisesRegex(ValueError, "balanced"):
            demo.parse_counts(good.replace("1,0,1,1,1,1", "1,0,4,0,0,0"), 2, 2)
        with self.assertRaisesRegex(ValueError, "missing"):
            demo.parse_counts(good.splitlines()[0], 2, 2)
        with self.assertRaisesRegex(ValueError, "invalid"):
            demo.parse_counts(good + "DATA,1,1,4,0,0,0\n", 2, 2)

    def test_reject_impossible_design(self):
        with self.assertRaises(ValueError):
            demo.simulate("slim", 3, 20, 1)
        with self.assertRaises(ValueError):
            demo.second_moment(0)

    @unittest.skipUnless(shutil.which("slim"), "SLiM 4.0.1 not on PATH")
    def test_simulated_moments(self):
        a, b = [demo.simulate("slim", n, 2048, demo.SEED+i)
                for i, n in enumerate((10,40))]
        for row in (a,b):
            self.assertEqual(sum(row[k] for k in ("positive_D","negative_D","zero_D")), 2048)
            self.assertGreater(row["positive_D"], 750)
            self.assertGreater(row["negative_D"], 750)
            self.assertLess(abs(row["mean_D"]), 5*row["mean_D_mcse"])
            self.assertLess(abs(row["mean_D_squared"]-row["mean_D_squared_expected"]),
                            5*row["mean_D_squared_mcse"])
        self.assertGreater(a["mean_D_squared"], 3*b["mean_D_squared"])
