"""Independent high-precision and algebraic checks; no external dependencies."""

from decimal import Decimal, localcontext
import json
import math
from pathlib import Path
import subprocess
import sys
import unittest

from examples.tail_proxy_attention.model import dense_attention, tail_proxy_attention
from examples.tail_proxy_attention.demo import build_report


def decimal_oracle(logits, values, groups=None, exact=(), mask=None):
    """70-digit per-original-token oracle; does not construct virtual tokens."""
    with localcontext() as context:
        context.prec = 70
        z = [Decimal(str(x)) for x in logits]
        v = [[Decimal(str(x)) for x in row] for row in values]
        visible = [True]*len(z) if mask is None else mask
        weights = [Decimal(0)]*len(z)
        if groups is None:
            for i, ok in enumerate(visible):
                if ok:
                    weights[i] = z[i].exp()
        else:
            chosen = set(exact)
            for i in exact:
                if visible[i]:
                    weights[i] = z[i].exp()
            for group in groups:
                members = [i for i in group if visible[i] and i not in chosen]
                if members:
                    mean = sum((z[i] for i in members), Decimal(0))/len(members)
                    for i in members:
                        weights[i] = mean.exp()
        total = sum(weights)
        probabilities = [w/total for w in weights]
        output = [sum((p*row[j] for p, row in zip(probabilities, v, strict=True)), Decimal(0))
                  for j in range(len(v[0]))]
        return tuple(float(x) for x in output), tuple(float(p) for p in probabilities)


class TailProxyTests(unittest.TestCase):
    def assertVectorClose(self, first, second, delta=2e-12):
        self.assertEqual(len(first), len(second))
        for a, b in zip(first, second, strict=True):
            self.assertAlmostEqual(a, b, delta=delta)

    def test_decimal_proxy_oracle_multiple_groups_and_masks(self):
        z = [2, -1, 3, 0.5, -4, 2.5]
        v = [[1, -2], [2, 4], [7, 3], [-1, 8], [5, -3], [9, 6]]
        groups = [[0, 1, 2], [3, 4, 5]]
        for exact in [[], [0], [0, 3], [1, 2, 5]]:
            for mask in [[True]*6, [True, True, False, True, False, True]]:
                row = tail_proxy_attention(z, v, groups, exact=exact, mask=mask)
                output, weights = decimal_oracle(z, v, groups, exact, mask)
                self.assertVectorClose(row.output, output)
                self.assertVectorClose(row.token_weights, weights)

    def test_decimal_dense_oracle(self):
        z, v = [1, -3, 2], [[3, 4], [7, -1], [-2, 8]]
        output, weights = decimal_oracle(z, v)
        row = dense_attention(z, v)
        self.assertVectorClose(row.output, output)
        self.assertVectorClose(row.token_weights, weights)

    def test_equal_logit_tail_is_exact_with_unequal_values(self):
        z, v = [math.log(4), 0, 0, 0], [[1], [2], [4], [6]]
        row = tail_proxy_attention(z, v, [[0, 1, 2, 3]], exact=[0])
        self.assertVectorClose(row.output, [16/7])
        self.assertVectorClose(row.token_weights, [4/7, 1/7, 1/7, 1/7])
        self.assertVectorClose(row.output, dense_attention(z, v).output)

    def test_omitting_count_correction_changes_result(self):
        row = tail_proxy_attention([math.log(4), 0, 0, 0], [[1], [2], [4], [6]],
                                   [[0, 1, 2, 3]], exact=[0], count_weighting=False)
        self.assertVectorClose(row.output, [8/5])
        self.assertGreater(abs(row.output[0]-16/7), 0.6)

    def test_variable_logits_counterexample_discard_is_closer(self):
        z, v = [0, -2, 2], [[0], [1], [0]]
        exact = math.exp(-2)/(1+math.exp(-2)+math.exp(2))
        dense = dense_attention(z, v)
        proxy = tail_proxy_attention(z, v, [[0, 1, 2]], exact=[0])
        discard = dense_attention(z, v, mask=[True, False, False])
        self.assertVectorClose(dense.output, [exact])
        self.assertVectorClose(proxy.output, [1/3])
        self.assertLess(abs(discard.output[0]-exact), abs(proxy.output[0]-exact))

    def test_singleton_tails_restore_dense(self):
        z, v = [0, -2, 2, 1], [[0, 1], [1, 0], [0, 5], [-2, 9]]
        row = tail_proxy_attention(z, v, [[0], [1], [2], [3]], exact=[0])
        self.assertVectorClose(row.output, dense_attention(z, v).output)
        self.assertVectorClose(row.token_weights, dense_attention(z, v).token_weights)

    def test_all_exact_restores_dense_and_has_no_proxies(self):
        z, v = [0, -2, 2], [[0], [1], [3]]
        row = tail_proxy_attention(z, v, [[0, 1, 2]], exact=[2, 0, 1])
        self.assertEqual(row.proxies, ())
        self.assertVectorClose(row.output, dense_attention(z, v).output)

    def test_no_exact_one_proxy_returns_arithmetic_value_mean(self):
        row = tail_proxy_attention([1, -2, 4], [[2, 7], [4, -3], [9, 2]], [[0, 1, 2]])
        self.assertVectorClose(row.output, [5, 2])
        self.assertVectorClose(row.token_weights, [1/3]*3)

    def test_exact_removal_recomputes_all_residual_statistics(self):
        row = tail_proxy_attention([9, 0, 2], [[100], [2], [6]], [[0, 1, 2]], exact=[0])
        proxy = row.proxies[0]
        self.assertEqual(proxy.members, (1, 2))
        self.assertEqual(proxy.count, 2)
        self.assertEqual(proxy.mean_logit, 1)
        self.assertEqual(proxy.mean_value, (4,))
        self.assertAlmostEqual(proxy.centered_logit, 1-9+math.log(2))

    def test_empty_residual_group_is_omitted(self):
        row = tail_proxy_attention([0, 1, 2], [[1], [2], [3]], [[0], [1, 2]], exact=[0])
        self.assertEqual(len(row.proxies), 1)
        self.assertEqual(row.proxies[0].members, (1, 2))

    def test_masked_exact_and_masked_groups_contribute_nothing(self):
        row = tail_proxy_attention([0, 10000, 1], [[1], [1e100], [3]], [[0, 2], [1]],
                                   exact=[1], mask=[True, False, True])
        self.assertEqual(row.exact_indices, ())
        self.assertEqual(row.token_weights[1], 0)
        self.assertEqual(row.proxies[0].count, 2)
        self.assertVectorClose(row.output, [2])

    def test_causal_prefix_invariant_to_extreme_future_token(self):
        z, v = [1, 0, -1], [[2], [4], [8]]
        before = tail_proxy_attention(z, v, [[0, 1, 2]], exact=[0])
        after = tail_proxy_attention(z+[1e6], v+[[1e100]], [[0, 1, 2, 3]],
                                     exact=[0, 3], mask=[True, True, True, False])
        self.assertEqual(before.output, after.output)
        self.assertEqual(before.proxies, after.proxies)
        self.assertEqual(before.token_weights, after.token_weights[:-1])

    def test_mean_scaled_logit_matches_query_dot_mean_key(self):
        q, keys = [2, -1], [[3, 4], [-1, 2], [5, 0]]
        z = [sum(a*b for a, b in zip(q, k, strict=True))/math.sqrt(2) for k in keys]
        row = tail_proxy_attention(z, [[1]]*3, [[0, 1, 2]])
        mean_key = [sum(k[j] for k in keys)/3 for j in range(2)]
        expected = sum(a*b for a, b in zip(q, mean_key, strict=True))/math.sqrt(2)
        self.assertAlmostEqual(row.proxies[0].mean_logit, expected)

    def test_constant_values_are_preserved(self):
        row = tail_proxy_attention([3, -1, 4, 0], [[7, -5]]*4, [[0, 1], [2, 3]], exact=[0])
        self.assertVectorClose(row.output, [7, -5])

    def test_common_logit_shift_invariance(self):
        z, v = [1, -3, 2, 0], [[1], [3], [7], [-2]]
        a = tail_proxy_attention(z, v, [[0, 1], [2, 3]], exact=[0])
        b = tail_proxy_attention([x+10000 for x in z], v, [[0, 1], [2, 3]], exact=[0])
        self.assertVectorClose(a.output, b.output, delta=1e-11)
        self.assertVectorClose(a.token_weights, b.token_weights, delta=1e-11)

    def test_permutation_equivariance(self):
        z, v = [1, -3, 2, 0], [[1], [3], [7], [-2]]
        groups, exact = [[0, 1], [2, 3]], [0]
        a = tail_proxy_attention(z, v, groups, exact=exact)
        permutation = [2, 0, 3, 1]
        inverse = {old: new for new, old in enumerate(permutation)}
        b = tail_proxy_attention([z[i] for i in permutation], [v[i] for i in permutation],
                                 [[inverse[i] for i in g] for g in groups], exact=[inverse[i] for i in exact])
        self.assertVectorClose(a.output, b.output)
        self.assertVectorClose([a.token_weights[i] for i in permutation], b.token_weights)

    def test_group_and_member_order_do_not_change_output(self):
        z, v = [1, -3, 2, 0], [[1], [3], [7], [-2]]
        a = tail_proxy_attention(z, v, [[0, 1], [2, 3]], exact=[0, 2])
        b = tail_proxy_attention(z, v, [[3, 2], [1, 0]], exact=[2, 0])
        self.assertVectorClose(a.output, b.output)
        self.assertVectorClose(a.token_weights, b.token_weights)

    def test_linearity_in_values_for_fixed_partition(self):
        z = [1, -3, 2, 0]
        v, u = [[1], [3], [7], [-2]], [[4], [2], [-3], [6]]
        def run(values):
            return tail_proxy_attention(z, values, [[0, 1], [2, 3]], exact=[0]).output[0]
        mixed = [[2*a[0]-3*b[0]] for a, b in zip(v, u, strict=True)]
        self.assertAlmostEqual(run(mixed), 2*run(v)-3*run(u), places=12)

    def test_weights_are_nonnegative_normalized_and_output_is_convex(self):
        v = [[1, -5], [3, 1], [7, 3], [-2, 8]]
        row = tail_proxy_attention([1, -3, 2, 0], v, [[0, 1], [2, 3]], exact=[0])
        for weights in [row.token_weights, row.component_weights]:
            self.assertTrue(all(w >= 0 for w in weights))
            self.assertAlmostEqual(math.fsum(weights), 1)
        for j, value in enumerate(row.output):
            self.assertGreaterEqual(value, min(r[j] for r in v))
            self.assertLessEqual(value, max(r[j] for r in v))

    def test_proxy_mass_is_shared_uniformly_between_members(self):
        row = tail_proxy_attention([1, -3, 2, 0], [[1], [3], [7], [-2]], [[0, 1, 2, 3]], exact=[0])
        proxy_weight = row.component_weights[1]
        self.assertVectorClose(row.token_weights[1:], [proxy_weight/3]*3)
        expanded_output = sum(w*v for w, v in zip(row.token_weights, [1, 3, 7, -2], strict=True))
        self.assertAlmostEqual(expanded_output, row.output[0])

    def test_jensen_mass_gap_and_equal_logit_equality(self):
        for z in [[-2, 2], [1, 2, 3], [4, 4, 4]]:
            row = tail_proxy_attention(z, [[1]]*len(z), [list(range(len(z)))])
            proxy = row.proxies[0]
            approximate = proxy.count*math.exp(proxy.mean_logit)
            dense = sum(math.exp(x) for x in z)
            self.assertLessEqual(approximate, dense+1e-12)
            if len(set(z)) == 1:
                self.assertAlmostEqual(approximate, dense)
            else:
                self.assertLess(approximate, dense)

    def test_splitting_equal_logit_tail_preserves_output(self):
        z, v = [2, 0, 0, 0, 0], [[2], [1], [3], [5], [9]]
        a = tail_proxy_attention(z, v, [[0, 1, 2, 3, 4]], exact=[0])
        b = tail_proxy_attention(z, v, [[0], [1, 2], [3, 4]], exact=[0])
        self.assertVectorClose(a.output, b.output)

    def test_extreme_logits_match_decimal_without_exp_overflow(self):
        z, v = [10000, 9999, -10000, 9998], [[1], [2], [3], [4]]
        groups, exact = [[0, 1], [2, 3]], [0]
        row = tail_proxy_attention(z, v, groups, exact=exact)
        output, weights = decimal_oracle(z, v, groups, exact)
        self.assertVectorClose(row.output, output)
        self.assertVectorClose(row.token_weights, weights)

    def test_huge_common_offset_does_not_swallow_log_count(self):
        row = tail_proxy_attention([1e308]*4, [[1], [2], [4], [6]], [[0, 1, 2, 3]], exact=[0])
        self.assertVectorClose(row.token_weights, [0.25]*4)
        self.assertVectorClose(row.output, [13/4])

    def test_opposite_extreme_logits_and_values_remain_finite(self):
        row = tail_proxy_attention([1e308, -1e308, -1e308], [[1e308], [-1e308], [0]], [[0, 1, 2]])
        self.assertVectorClose(row.output, [0])
        self.assertVectorClose(row.token_weights, [1/3]*3)
        self.assertTrue(math.isfinite(row.proxies[0].mean_logit))
        dense = dense_attention([1e308, -1e308], [[1], [2]])
        self.assertEqual(dense.output, (1,))

    def test_invalid_numeric_inputs_and_shapes(self):
        for z, v in [([], []), ([1], []), ([1], [[]]), ([1, 2], [[1], [1, 2]]),
                     ([math.nan], [[1]]), ([math.inf], [[1]]), ([True], [[1]]),
                     ([1], [[math.inf]]), ([1], [[True]]), (None, [[1]]), ([1], [1])]:
            with self.subTest(z=z, v=v), self.assertRaises(ValueError):
                tail_proxy_attention(z, v, [[0]])

    def test_invalid_partitions_rejected(self):
        for groups in [[], [[]], [[0, 1]], [[0, 1], [1, 2]], [[0, 0, 1, 2]],
                       [[0, 1, 3]], [[0, 1, -1]], [[0, 1, True]], [[0, 1, 2.0]], None, [3]]:
            with self.subTest(groups=groups), self.assertRaises(ValueError):
                tail_proxy_attention([1, 2, 3], [[1]]*3, groups)

    def test_invalid_exact_mask_and_ablation_flags(self):
        for kwargs in [{"exact": [0, 0]}, {"exact": [3]}, {"exact": [True]}, {"exact": None},
                       {"mask": [False]*3}, {"mask": [True]}, {"mask": [1, 1, 1]},
                       {"mask": 1}, {"count_weighting": 1}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                tail_proxy_attention([1, 2, 3], [[1]]*3, [[0, 1, 2]], **kwargs)

    def test_sample_output_matches_executed_cli_bytes(self):
        proc = subprocess.run([sys.executable, '-m', 'examples.tail_proxy_attention.demo'],
                              check=True, capture_output=True, text=True)
        sample = Path(__file__).with_name('sample_output.json').read_text()
        self.assertEqual(proc.stdout, sample)
        self.assertEqual(json.loads(sample), build_report())

    def test_cli_spread_and_invalid_arguments(self):
        cmd = [sys.executable, '-m', 'examples.tail_proxy_attention.demo']
        proc = subprocess.run(cmd+['--spread', '0'], check=True, capture_output=True, text=True)
        case = json.loads(proc.stdout)['variable_logit_counterexample']
        self.assertEqual(case['dense_output'], case['proxy']['output'])
        for value in ['-1', '101', 'nan', 'inf']:
            proc = subprocess.run(cmd+['--spread', value], capture_output=True, text=True)
            self.assertEqual(proc.returncode, 2)
            self.assertIn('error:', proc.stderr)


if __name__ == '__main__':
    unittest.main()
