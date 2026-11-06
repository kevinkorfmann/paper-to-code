"""Independent high-precision oracle, identities, controls, and CLI checks."""

from decimal import Decimal, localcontext
import json
import math
import random
import subprocess
import sys
import unittest

from examples.hoba_triplet_attention import demo
from examples.hoba_triplet_attention.model import block_attention, triplet_read


def oracle(q, a, b, u, v):
    """Direct 80-digit exponentiation, independently flattened pair indexing."""
    with localcontext() as context:
        context.prec = 80
        D = lambda x: Decimal(str(x))
        pairs = [(j, k) for j in range(len(a)) for k in range(len(b))]
        logits = [sum(D(q[r]) * D(a[j][r]) * D(b[k][r])
                      for r in range(len(q))) / D(len(q)).sqrt() for j, k in pairs]
        masses = [x.exp() for x in logits]
        probabilities = [x / sum(masses) for x in masses]
        output = [sum(p * D(u[j][r]) * D(v[k][r])
                      for p, (j, k) in zip(probabilities, pairs))
                  for r in range(len(u[0]))]
        return probabilities, output


class TripletTests(unittest.TestCase):
    def setUp(self):
        self.q = [0.5, -1.25]
        self.a = [[1, 2], [-2, 0.5], [0.25, -1]]
        self.b = [[-0.5, 1], [1.5, -2], [0.5, 0.5]]
        self.u = [[1, 2], [-1, 3], [2, -2]]
        self.v = [[-2, 1], [1, 0], [0.5, 2]]

    def read(self, **changes):
        args = {"query": self.q, "keys1": self.a, "keys2": self.b,
                "values1": self.u, "values2": self.v}
        args.update(changes)
        return triplet_read(**args)

    def close(self, a, b, tolerance=2e-12):
        self.assertEqual(len(a), len(b))
        for x, y in zip(a, b):
            self.assertAlmostEqual(float(x), float(y), delta=tolerance)

    def test_decimal_oracle_and_dimension_scale(self):
        weights, output = oracle(self.q, self.a, self.b, self.u, self.v)
        result = self.read()
        self.close([x for row in result.weights for x in row], weights)
        self.close(result.output, output)

    def test_seeded_decimal_oracle(self):
        rng = random.Random(20261106)
        for n, d, dv in [(1, 1, 1), (2, 3, 2), (3, 2, 3), (4, 4, 1)]:
            for _ in range(5):
                make = lambda rows, cols: [[rng.randint(-6, 6) / 4 for _ in range(cols)]
                                           for _ in range(rows)]
                q, a, b, u, v = make(1, d)[0], make(n, d), make(n, d), make(n, dv), make(n, dv)
                expected_weights, expected_output = oracle(q, a, b, u, v)
                got = triplet_read(q, a, b, u, v)
                self.close([x for row in got.weights for x in row], expected_weights)
                self.close(got.output, expected_output)

    def test_joint_normalization_not_each_row(self):
        result = self.read()
        self.assertAlmostEqual(math.fsum(x for row in result.weights for x in row), 1)
        self.assertTrue(all(0 <= x <= 1 for row in result.weights for x in row))
        self.assertTrue(all(math.fsum(row) < 1 for row in result.weights))

    def test_zero_query_product_of_means(self):
        result = self.read(query=[0, 0])
        self.close([x for row in result.weights for x in row], [1 / 9] * 9)
        self.close(result.output, [math.fsum(row[r] for row in self.u) / 3
                                   * math.fsum(row[r] for row in self.v) / 3
                                   for r in range(2)])

    def test_sign_pair_exact_tanh_and_identical_marginals(self):
        for query in [-2, -1, 0, 1, 2]:
            report = demo.describe(query)
            self.close(report["first_marginal"], [0.5, 0.5])
            self.close(report["second_marginal"], [0.5, 0.5])
            self.assertAlmostEqual(report["joint_output"], math.tanh(query))
            self.assertEqual(report["product_of_marginals_output"], 0)

    def test_diagonal_pairs_are_included(self):
        result = triplet_read([0], [[1], [1]], [[1], [1]], [[1], [0]], [[1], [0]])
        self.assertEqual(result.output, (0.25,))
        self.assertEqual(result.weights, ((0.25, 0.25), (0.25, 0.25)))

    def test_branch_swap_transposes_joint(self):
        original = self.read()
        swapped = self.read(keys1=self.b, keys2=self.a, values1=self.v, values2=self.u)
        self.close(swapped.output, original.output)
        for j in range(3):
            self.close(swapped.weights[j], [original.weights[k][j] for k in range(3)])

    def test_independent_token_permutations(self):
        p, t = [2, 0, 1], [1, 2, 0]
        original = self.read()
        result = self.read(keys1=[self.a[i] for i in p], values1=[self.u[i] for i in p],
                           keys2=[self.b[i] for i in t], values2=[self.v[i] for i in t])
        self.close(result.output, original.output)
        for j in range(3):
            self.close(result.weights[j], [original.weights[p[j]][t[k]] for k in range(3)])

    def test_reciprocal_key_rescaling(self):
        result = self.read(keys1=[[2 * x for x in row] for row in self.a],
                           keys2=[[x / 2 for x in row] for row in self.b])
        self.assertEqual(result, self.read())

    def test_value_bilinearity(self):
        result = self.read(values1=[[2 * x for x in row] for row in self.u],
                           values2=[[-3 * x for x in row] for row in self.v])
        self.close(result.output, [-6 * x for x in self.read().output])
        self.assertEqual(result.weights, self.read().weights)

    def test_zero_values(self):
        self.assertEqual(self.read(values1=[[0, 0]] * 3).output, (0, 0))

    def test_constant_values(self):
        self.close(self.read(values1=[[2, -3]] * 3, values2=[[4, 5]] * 3).output, [8, -15])

    def test_single_pair(self):
        result = triplet_read([2], [[3]], [[4]], [[-2, 3]], [[4, 5]])
        self.assertEqual(result.weights, ((1.0,),))
        self.assertEqual(result.output, (-8, 15))

    def test_duplicate_tokens_preserve_output(self):
        duplicate = lambda matrix: [row for row in matrix for _ in range(2)]
        result = triplet_read(self.q, duplicate(self.a), duplicate(self.b),
                              duplicate(self.u), duplicate(self.v))
        self.close(result.output, self.read().output)

    def test_large_finite_scores_are_stable(self):
        result = triplet_read([1000], [[-1], [1]], [[-1], [1]], [[-1], [1]], [[-1], [1]])
        self.assertEqual(result.weights, ((0.5, 0), (0, 0.5)))
        self.assertEqual(result.output, (1,))

    def test_opposite_extreme_score_subtraction(self):
        result = triplet_read([1e308], [[-1], [1]], [[-1], [1]], [[-1], [1]], [[-1], [1]])
        self.assertEqual(result.output, (1,))

    def test_reject_invalid_shapes(self):
        cases = [{"query": []}, {"keys1": []}, {"keys1": [[1], [1, 2]]},
                 {"query": [1]}, {"values1": [[1, 2]]}, {"values2": [[1]] * 3},
                 {"keys2": None}, {"values1": [[], [], []]}]
        for change in cases:
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.read(**change)

    def test_reject_nonfinite_and_nonreal_values(self):
        for value in [float("nan"), float("inf"), -float("inf"), 1j, None, True]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.read(query=[value, 1])

    def test_reject_intermediate_overflow(self):
        for args in [([1e308], [[1e308]], [[1]], [[1]], [[1]]),
                     ([1], [[1]], [[1]], [[1e308]], [[1e308]])]:
            with self.assertRaisesRegex(ValueError, "overflow"):
                triplet_read(*args)

    def test_inputs_are_not_mutated_and_iterators_work(self):
        before = json.dumps([self.q, self.a, self.b, self.u, self.v])
        result = self.read(query=iter(self.q))
        self.assertEqual(result, self.read())
        self.assertEqual(before, json.dumps([self.q, self.a, self.b, self.u, self.v]))


class BlockTests(unittest.TestCase):
    def setUp(self):
        self.q = [[1], [-1], [0]]
        self.k = [[-1], [1], [2]]
        self.v = [[1], [3], [10]]

    def run_block(self, size, values=None):
        return block_attention(self.q, self.k, self.k, values or self.v, self.v, size)

    def test_full_block_equals_direct(self):
        expected = tuple(triplet_read(q, self.k, self.k, self.v, self.v) for q in self.q)
        self.assertEqual(self.run_block(3), expected)
        self.assertEqual(self.run_block(10), expected)

    def test_singleton_blocks_include_diagonal(self):
        self.assertEqual([x.output[0] for x in self.run_block(1)], [1, 9, 100])

    def test_short_final_block(self):
        result = self.run_block(2)
        self.assertEqual([len(x.weights) for x in result], [2, 2, 1])
        self.assertEqual(result[-1].output, (100,))

    def test_other_block_does_not_change_first_block(self):
        self.assertEqual(self.run_block(2)[:2], self.run_block(2, [[1], [3], [-999]])[:2])

    def test_local_kernel_is_not_causal(self):
        before = self.run_block(2)[0].output
        after = self.run_block(2, [[1], [30], [10]])[0].output
        self.assertNotEqual(before, after)

    def test_boundary_changes_answer(self):
        report = demo.build_report()["block_boundary_control"]
        self.assertEqual(report["block_size_two_outputs"], [1, 1, 100])
        for value in report["one_full_block_outputs"]:
            self.assertAlmostEqual(value, 16)

    def test_reject_invalid_blocks(self):
        for size in [0, -1, 1.5, True, "2"]:
            with self.subTest(size=size), self.assertRaises(ValueError):
                self.run_block(size)
        with self.assertRaises(ValueError):
            block_attention(self.q, self.k[:1], self.k, self.v, self.v, 2)


class DemoTests(unittest.TestCase):
    def test_checked_sample_and_cli(self):
        expected = demo.SAMPLE.read_text(encoding="utf-8")
        self.assertEqual(demo.render(demo.build_report()), expected)
        result = subprocess.run([sys.executable, "-m", "examples.hoba_triplet_attention.demo"],
                                check=True, capture_output=True, text=True)
        self.assertEqual(result.stdout, expected)
        result = subprocess.run([sys.executable, "-m", "examples.hoba_triplet_attention.demo",
                                 "--check-sample"], check=True, capture_output=True, text=True)
        self.assertEqual(result.stdout, "sample output matches\n")

    def test_unknown_flag_fails(self):
        result = subprocess.run([sys.executable, "-m", "examples.hoba_triplet_attention.demo",
                                 "--rewrite-sample"], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
