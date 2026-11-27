import itertools
import os
import statistics
import unittest

from demo import REPS, parse, run, summary


class SiblingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = run(os.getenv("SLIM", "slim"))
        cls.groups = summary(cls.rows)["groups"]

    def test_rows(self):
        self.assertEqual(len(self.rows), REPS)

    def test_parser_rejects_partial_output(self):
        with self.assertRaises(ValueError):
            parse("DATA,0,10,10")

    def test_unrelated_variance(self):
        self.assertAlmostEqual(self.groups["unrelated"]["variance"], 0.0125,
                               delta=0.002)

    def test_sibling_variance(self):
        self.assertAlmostEqual(self.groups["siblings"]["variance"], 0.06875,
                               delta=0.008)

    def test_both_means(self):
        for group in self.groups.values():
            self.assertAlmostEqual(group["mean_frequency"], 0.5, delta=0.02)

    def test_enumerated_parent_variance(self):
        parents = list(itertools.product((0, 1), repeat=4))
        between = statistics.pvariance(sum(a) / 4 for a in parents)
        within = statistics.mean(sum(p * (1 - p) for p in
                                      (sum(a[:2]) / 2, sum(a[2:]) / 2)) / 40
                                 for a in parents)
        self.assertEqual(between + within, 11 / 160)
