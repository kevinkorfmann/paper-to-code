from math import sqrt
import shutil
import unittest

from demo import MODELS, exact, report, transition


class TheoryTests(unittest.TestCase):
    def test_one_generation(self):
        self.assertEqual(transition([0, 1, 0], 4),
                         [1/16, 4/16, 6/16, 4/16, 1/16])

    def test_absorption(self):
        self.assertEqual(transition([0.3, 0, 0.7], 3), [0.3, 0, 0, 0.7])

    def test_symmetric_normalized_distribution(self):
        for path, _ in MODELS.values():
            d = exact(path)
            self.assertAlmostEqual(sum(d), 1)
            for a, b in zip(d, reversed(d)):
                self.assertAlmostEqual(a, b)

    def test_reversing_order_changes_absorption_not_diversity(self):
        a = exact((4, 12, 40))
        b = exact((12, 4, 40))
        self.assertGreater(b[0], a[0] + 0.02)
        for d in (a, b):
            diversity = sum(w * 2 * k/40 * (1-k/40) for k, w in enumerate(d))
            self.assertAlmostEqual(diversity, 0.33515625)


@unittest.skipUnless(shutil.which("slim"), "Requires SLiM executable")
class SimulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = report()

    def test_absorption_within_five_monte_carlo_se(self):
        for r in self.results.values():
            p = r["exact_monomorphic"]
            self.assertLess(abs(r["observed_monomorphic"] - p),
                            5 * sqrt(p * (1-p) / r["replicates"]))

    def test_diversity_and_neutral_mean(self):
        for r in self.results.values():
            self.assertAlmostEqual(r["observed_gene_diversity"],
                                   r["exact_gene_diversity"], delta=0.015)
            self.assertAlmostEqual(r["mean_frequency"], 0.5, delta=0.025)
