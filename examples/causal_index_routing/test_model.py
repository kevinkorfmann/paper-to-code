"""Independent exact-arithmetic oracles and causal selector invariants."""
import copy
from dataclasses import FrozenInstanceError
from fractions import Fraction as F
import itertools
import random
import unittest

from examples.causal_index_routing import model


def rational_score(heads, weights, key):
    """Eq. 1 evaluated without production helpers or floating-point arithmetic."""
    total = F(0)
    for h in range(len(heads)):
        dot = sum((F(heads[h][d]) * F(key[d])
                   for d in range(len(key))), F(0))
        if dot > 0:
            total += F(weights[h]) * dot
    return total


def rational_rank(heads, weights, keys, ids):
    scores = {i: rational_score(heads, weights, keys[i]) for i in ids}
    return tuple(sorted(scores, key=lambda i: (-scores[i], i)))


def rational_selection(keys, groups, sinks, visible, heads, weights, budget, k):
    """Independent specification including exact rational representative means."""
    means = [[sum((F(keys[i][d]) for i in group), F(0)) / len(group)
              for d in range(len(keys[0]))] for group in groups]
    scores = tuple(rational_score(heads, weights, mean) for mean in means)
    order = tuple(sorted(range(len(groups)), key=lambda i: (-scores[i], i)))
    reserved = tuple(sorted(sinks)) + tuple(range(len(keys), len(visible)))
    stream = list(reserved)
    for group_id in order:
        stream += sorted(groups[group_id])
    candidates = tuple(stream[:budget])
    winners = rational_rank(heads, weights, visible, candidates)[:k]
    return scores, order, reserved, candidates, winners


class NativeScoreTests(unittest.TestCase):
    def test_exhaustive_small_integer_multihead_oracle(self):
        keys = list(itertools.product((-2, 0, 3), repeat=2))
        for first in itertools.product((-1, 0, 2), repeat=2):
            for weights in itertools.product((-2, 0, 1), repeat=2):
                heads = [first, (2, -1)]
                with self.subTest(first=first, weights=weights):
                    expected = tuple(float(rational_score(heads, weights, key))
                                     for key in keys)
                    self.assertEqual(model.native_scores(heads, weights, keys), expected)
                    for k in (1, 4, len(keys)):
                        self.assertEqual(model.exhaustive_top_k(heads, weights, keys, k),
                                         rational_rank(heads, weights, keys, range(len(keys)))[:k])

    def test_signed_weights_and_relu_before_head_sum(self):
        self.assertEqual(model.native_scores([[1], [-1]], [1, 1], [[2], [-2]]), (2, 2))
        self.assertEqual(model.native_scores([[1], [-1]], [-3, 2], [[2], [-2]]), (-6, 4))
        self.assertEqual(model.native_scores([[1, 1]], [1], [[2, 2]]), (4,))

    def test_fractional_inputs(self):
        heads = [[0.5, -0.25], [-2, 0.125]]
        keys = [[0.25, 2], [-4, 0.5], [0, 0]]
        weights = [-0.5, 1.25]
        self.assertEqual(model.native_scores(heads, weights, keys),
                         tuple(float(rational_score(heads, weights, key)) for key in keys))

    def test_invalid_shapes_and_numeric_values(self):
        fixtures = [([], [1], [[1]]), ([[1]], [], [[1]]), ([[1]], [1], []),
                    ([[1], [1, 2]], [1, 1], [[1]]), ([[1, 2]], [1], [[1]]),
                    ([[1]], [1, 2], [[1]]), ([[1]], [1], [[1], [1, 2]]),
                    ([[True]], [1], [[1]]), ([[1]], [False], [[1]]),
                    ([[1]], [1], [[True]]), ([[1]], [1], [[1j]]),
                    ([[1]], [1], [["1"]]), ([[1]], [1], [[10**1000]]),
                    ([[1]], [1], [[float('nan')]]), ([[float('inf')]], [1], [[1]]),
                    ([[1]], [float('-inf')], [[1]])]
        for args in fixtures:
            with self.subTest(args=repr(args)), self.assertRaises(ValueError):
                model.native_scores(*args)

    def test_overflow_is_explicit(self):
        for args in [([[1e308]], [1], [[2]]),
                     ([[1]], [1e308], [[2]]),
                     ([[1, 1]], [1], [[1e308, 1e308]]),
                     ([[1], [1]], [1, 1], [[1e308]])]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                model.native_scores(*args)

    def test_invalid_exhaustive_k(self):
        for k in (True, False, 0, -1, 2, 1.0, None):
            with self.subTest(k=k), self.assertRaises(ValueError):
                model.exhaustive_top_k([[1]], [1], [[1]], k)


class PrefixIndexTests(unittest.TestCase):
    def test_exact_means_and_sorted_members(self):
        index = model.PrefixIndex([[1, 2], [4, 5], [-2, -1], [8, 0]], [[2, 0, 1]], [3])
        self.assertEqual(index.groups, ((0, 1, 2),))
        self.assertEqual(index.representatives, ((1, 2),))

    def test_large_finite_mean_does_not_overflow(self):
        index = model.PrefixIndex([[1e308], [1e308]], [[0, 1]])
        self.assertEqual(index.representatives, ((1e308,),))

    def test_invalid_partition_and_ids(self):
        for groups, sinks in [([[0]], []), ([[0, 1], [1]], []), ([[0, 1]], [0]),
                              ([[0, 0, 1]], []), ([[]], [0, 1]), ([[0, 2]], []),
                              ([[-1, 0]], []), ([[False, 1]], []), ([[0.0, 1]], []),
                              ([[0]], [1, 1]), ([], [0]), ({0: [0, 1]}, []),
                              ([[0, 1]], {0}), ([{0, 1}], [])]:
            with self.subTest(groups=groups, sinks=sinks), self.assertRaises(ValueError):
                model.PrefixIndex([[1], [2]], groups, sinks)

    def test_copied_and_immutable(self):
        keys, groups, sinks = [[1], [3], [5]], [[1, 0]], [2]
        original = copy.deepcopy((keys, groups, sinks))
        index = model.PrefixIndex(keys, groups, sinks)
        self.assertEqual((keys, groups, sinks), original)
        keys[0][0] = 999
        groups[0].clear()
        sinks.clear()
        self.assertEqual(index.keys, ((1,), (3,), (5,)))
        self.assertEqual(index.groups, ((0, 1),))
        self.assertEqual(index.sinks, (2,))
        self.assertEqual(index.representatives, ((2,),))
        with self.assertRaises(FrozenInstanceError):
            index.keys = ()
        with self.assertRaises(TypeError):
            index.keys[0][0] = 2


class SelectionTests(unittest.TestCase):
    def test_independent_fraction_oracle_across_budgets(self):
        rng = random.Random(261102508)
        for trial in range(30):
            keys = [[rng.randrange(-5, 6) for _ in range(3)] for _ in range(7)]
            visible = keys + [[rng.randrange(-5, 6) for _ in range(3)]]
            groups, sinks = [[4, 0, 2], [5, 1, 3]], [6]
            heads = [[rng.randrange(-3, 4) for _ in range(3)] for _ in range(3)]
            weights = [rng.choice((-2, -0.5, 0, 1, 3)) for _ in heads]
            index = model.PrefixIndex(keys, groups, sinks)
            for budget in range(2, 9):
                for k in range(1, budget + 1):
                    with self.subTest(trial=trial, budget=budget, k=k):
                        expected = rational_selection(keys, groups, sinks, visible,
                                                      heads, weights, budget, k)
                        got = model.select(index, visible, heads, weights, budget=budget, top_k=k)
                        for actual, exact in zip(got.cluster_scores, expected[0]):
                            self.assertAlmostEqual(actual, float(exact), places=12)
                        self.assertEqual((got.cluster_order, got.reserved_ids,
                                          got.candidate_ids, got.selected_ids), expected[1:])
                        self.assertEqual(got.selected_scores, tuple(float(rational_score(heads, weights, visible[i]))
                                                                   for i in expected[-1]))

    def test_full_budget_matches_independent_exhaustive_for_every_k(self):
        keys = [[3, -2], [-4, 1], [0, 3], [2, 2], [-1, -1]]
        heads, weights = [[1, 2], [-2, 3]], [-1, 2]
        index = model.PrefixIndex(keys[:4], [[2, 0], [1]], [3])
        expected = rational_rank(heads, weights, keys, range(5))
        for k in range(1, 6):
            got = model.select(index, keys, heads, weights, budget=5, top_k=k)
            self.assertEqual(got.selected_ids, expected[:k])
            self.assertEqual(set(got.candidate_ids), set(range(5)))

    def test_cancellation_and_regrouping(self):
        keys = [[9], [-9], [4], [4]]
        bad = model.PrefixIndex(keys, [[0, 1], [2, 3]])
        good = model.PrefixIndex(keys, [[0, 2], [1, 3]])
        self.assertEqual(model.select(bad, keys, [[1]], [1], budget=2, top_k=1).selected_ids, (2,))
        self.assertEqual(model.select(good, keys, [[1]], [1], budget=2, top_k=1).selected_ids, (0,))

    def test_old_snapshot_and_open_window(self):
        prefix = [[2], [1]]
        index = model.PrefixIndex(prefix, [[0], [1]])
        before = model.select(index, prefix, [[1]], [1], budget=1, top_k=1)
        extended = prefix + [[-100]]
        later = model.select(index, extended, [[1]], [1], budget=2, top_k=1)
        self.assertEqual(later.reserved_ids, (2,))
        self.assertEqual(later.selected_ids, (0,))
        self.assertEqual(model.select(index, prefix, [[1]], [1], budget=1, top_k=1), before)
        newer = model.PrefixIndex(extended, [[0, 2], [1]])
        with self.assertRaises(ValueError):
            model.select(newer, prefix, [[1]], [1], budget=1, top_k=1)
        with self.assertRaises(ValueError):
            model.select(index, [[2], [7]], [[1]], [1], budget=1, top_k=1)

    def test_reserved_candidate_not_guaranteed_final_membership(self):
        keys = [[-3], [8], [5], [-2]]
        index = model.PrefixIndex(keys[:3], [[1, 2]], [0])
        got = model.select(index, keys, [[1]], [1], budget=3, top_k=1)
        self.assertEqual(got.reserved_ids, (0, 3))
        self.assertEqual(got.candidate_ids, (0, 3, 1))
        self.assertEqual(got.selected_ids, (1,))

    def test_sorted_member_truncation_not_best_member(self):
        keys = [[1], [100], [2]]
        index = model.PrefixIndex(keys, [[1, 0], [2]])
        got = model.select(index, keys, [[1]], [1], budget=1, top_k=1)
        self.assertEqual(got.candidate_ids, (0,))
        self.assertEqual(got.selected_ids, (0,))
        self.assertEqual(model.exhaustive_top_k([[1]], [1], keys, 1), (1,))

    def test_ties_cluster_order_then_token_id(self):
        keys = [[2], [2], [2], [2]]
        index = model.PrefixIndex(keys, [[3, 1], [2, 0]])
        got = model.select(index, keys, [[1]], [0], budget=3, top_k=3)
        self.assertEqual(got.cluster_order, (0, 1))
        self.assertEqual(got.candidate_ids, (1, 3, 0))
        self.assertEqual(got.selected_ids, (0, 1, 3))

    def test_all_sink_index(self):
        keys = [[1], [2]]
        index = model.PrefixIndex(keys, [], [1, 0])
        got = model.select(index, keys, [[1]], [1], budget=2, top_k=1)
        self.assertEqual(got.cluster_scores, ())
        self.assertEqual(got.reserved_ids, (0, 1))
        self.assertEqual(got.selected_ids, (1,))

    def test_nonmutation_and_frozen_selection(self):
        keys, groups, heads, weights = [[1], [2]], [[1, 0]], [[1]], [1]
        before = copy.deepcopy((keys, groups, heads, weights))
        index = model.PrefixIndex(keys, groups)
        got = model.select(index, keys, heads, weights, budget=2, top_k=1)
        self.assertEqual((keys, groups, heads, weights), before)
        with self.assertRaises(FrozenInstanceError):
            got.selected_ids = ()

    def test_invalid_budget_and_k(self):
        keys = [[1], [2]]
        index = model.PrefixIndex(keys, [[0, 1]])
        for budget, k in [(0, 1), (3, 1), (1, 0), (1, 2), (True, 1),
                          (2, True), (2.0, 1), (2, 1.0), (-1, 1)]:
            with self.subTest(budget=budget, k=k), self.assertRaises(ValueError):
                model.select(index, keys, [[1]], [1], budget=budget, top_k=k)
        with self.assertRaises(ValueError):
            model.select(None, keys, [[1]], [1], budget=1, top_k=1)

    def test_reserved_budget_overflow(self):
        index = model.PrefixIndex([[1], [2]], [[1]], [0])
        with self.assertRaises(ValueError):
            model.select(index, [[1], [2], [3]], [[1]], [1], budget=1, top_k=1)

    def test_selection_validates_queries_and_overflow(self):
        index = model.PrefixIndex([[2]], [[0]])
        for heads, weights in [([[1, 2]], [1]), ([[1]], [1, 2]),
                               ([[float('nan')]], [1]), ([[1e308]], [1])]:
            with self.subTest(heads=heads, weights=weights), self.assertRaises(ValueError):
                model.select(index, [[2]], heads, weights, budget=1, top_k=1)
        # Representative cancellation is harmless, but candidate refinement overflows.
        index = model.PrefixIndex([[1e308], [-1e308]], [[0, 1]])
        with self.assertRaises(ValueError):
            model.select(index, [[1e308], [-1e308]], [[2]], [1], budget=2, top_k=1)


if __name__ == '__main__':
    unittest.main()
