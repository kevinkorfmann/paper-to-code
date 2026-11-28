from itertools import combinations
import unittest

from .demo import evaluate, route, run, softmax, teacher_consensus


class RoutingTests(unittest.TestCase):
    def test_stable_softmax(self):
        for base, shifted in zip(softmax([0, 1, 2]), softmax([1000, 1001, 1002])):
            self.assertAlmostEqual(base, shifted)

    def test_k_and_ties(self):
        self.assertEqual(route([4, 4, 8], 2), (2, 0))
        for k in (0, 4):
            with self.assertRaises(ValueError):
                route([4, 4, 8], k)

    def test_consensus_maximizes_average_retained_mass(self):
        rows = [(0.65, 0.25, 0.1), (0.1, 0.3, 0.6)]
        for k in (1, 2, 3):
            ids = route(teacher_consensus(rows), k)
            mass = lambda selected: sum(w[i] for w in rows for i in selected)
            self.assertAlmostEqual(mass(ids), max(map(mass, combinations(range(3), k))))

    def test_consensus_does_not_optimize_worst_layer(self):
        r = run(1)
        self.assertEqual(r["consensus"]["retained_mass"], [0.65, 0.10])
        self.assertEqual(r["maximin_diagnostic"]["retained_mass"], [0.25, 0.30])
        self.assertEqual(r["per_layer_oracle"]["retained_mass"], [0.65, 0.60])

    def test_restricted_attention_renormalizes(self):
        mass, output = evaluate((0.6, 0.3, 0.1), (0, 1, 3), (1, 2))
        self.assertAlmostEqual(mass, 0.4)
        self.assertAlmostEqual(output, 1.5)

    def test_full_budget_recovers_dense_output(self):
        r = run(3)
        for name in ("consensus", "maximin_diagnostic", "per_layer_oracle"):
            self.assertEqual(r[name]["sparse_output"], r["dense_output"])
            self.assertEqual(r[name]["retained_mass"], [1, 1])
