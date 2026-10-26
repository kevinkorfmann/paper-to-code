import copy
from decimal import Decimal, localcontext
import json
from math import exp, fsum, log
from pathlib import Path
import random
import subprocess
import sys
import unittest

from examples.first_passage_attention.demo import build_report
from examples.first_passage_attention.model import (
    IncrementalAttention, causal_attention, first_passage_bias, walk_from_increments,
)


def decimal_oracle(queries, keys, values, field, levels, beta):
    """Direct slice minima and unshifted exponentials at 60-digit precision.

    Independent of production scans, caches, dot products and softmax.
    """
    with localcontext() as context:
        context.prec = 60
        d = lambda x: Decimal(str(x))
        scale, strength = d(len(queries[0])).sqrt(), d(beta)
        result = []
        for t, query in enumerate(queries):
            biases = [strength * min(Decimal(0), min(map(d, field[s+1:t+1])) - d(levels[s]))
                      for s in range(t)] + [Decimal(0)]
            scores = [sum(d(a)*d(b) for a, b in zip(query, keys[s]))/scale + biases[s]
                      for s in range(t+1)]
            masses = [score.exp() for score in scores]
            weights = [mass/sum(masses) for mass in masses]
            output = [sum(weights[s]*d(values[s][j]) for s in range(t+1))
                      for j in range(len(values[0]))]
            result.append(tuple(tuple(map(float, vector)) for vector in (biases, weights, output)))
        return result


class FirstPassageTests(unittest.TestCase):
    def assertVectorClose(self, actual, expected, places=12):
        self.assertEqual(len(actual), len(expected))
        for a, b in zip(actual, expected, strict=True):
            self.assertAlmostEqual(a, b, places=places)

    def test_independent_decimal_oracle_with_separate_fields_and_levels(self):
        rng = random.Random(261022714)
        for n, width in ((1, 1), (4, 2), (7, 3)):
            q = [[rng.uniform(-2, 2) for _ in range(width)] for _ in range(n)]
            k = [[rng.uniform(-2, 2) for _ in range(width)] for _ in range(n)]
            v = [[rng.uniform(-3, 3) for _ in range(2)] for _ in range(n)]
            phi = [rng.uniform(-2, 2) for _ in range(n)]
            psi = [rng.uniform(-2, 2) for _ in range(n)]
            for beta in (0.1, 1.0, 4.0):
                actual = causal_attention(q, k, v, phi, psi, beta)
                reference = decimal_oracle(q, k, v, phi, psi, beta)
                for row, (bias, weights, output) in zip(actual, reference, strict=True):
                    self.assertVectorClose(row.biases, bias)
                    self.assertVectorClose(row.weights, weights)
                    self.assertVectorClose(row.output, output)

    def test_full_and_incremental_agree_on_seeded_inputs(self):
        rng = random.Random(42)
        for n in (1, 2, 9):
            q = [[rng.uniform(-1, 1) for _ in range(3)] for _ in range(n)]
            k = [[rng.uniform(-1, 1) for _ in range(3)] for _ in range(n)]
            v = [[rng.uniform(-2, 2) for _ in range(2)] for _ in range(n)]
            phi, psi = ([rng.uniform(-2, 2) for _ in range(n)] for _ in range(2))
            decoder = IncrementalAttention(1.7)
            dense = causal_attention(q, k, v, phi, psi, 1.7)
            for t in range(n):
                row = decoder.step(q[t], k[t], v[t], phi[t], psi[t])
                self.assertEqual(row, dense[t])
                self.assertEqual(len(decoder), t+1)

    def test_range_excludes_key_field_and_includes_query_field(self):
        rows = first_passage_bias([-100, 2, 1], [3, 2, 999], 2)
        self.assertEqual(rows, ((0,), (-2, 0), (-4, -2, 0)))

    def test_original_rollback_bias_and_payload_mass(self):
        report = build_report()
        row = report["first_passage_attention"]
        self.assertEqual(row["biases"], [0, -4, -8, 0, 0, 0])
        self.assertGreater(row["weights"][0], 0.48)
        self.assertEqual(row["weights"][0], row["weights"][4])
        self.assertLess(row["draft_and_detail_mass"], 0.01)
        self.assertGreater(row["draft_and_detail_mass"], 0)

    def test_future_tokens_cannot_change_prefix_outputs(self):
        q, k, v = [[1], [2], [3]], [[3], [2], [1]], [[1], [-1], [4]]
        phi, psi = [2, 1, 0], [3, 2, 1]
        all_rows = causal_attention(q, k, v, phi, psi)
        prefix = causal_attention(q[:2], k[:2], v[:2], phi[:2], psi[:2])
        self.assertEqual(all_rows[:2], prefix)
        changed = causal_attention(q[:2]+[[-999]], k[:2]+[[999]], v[:2]+[[999]],
                                   phi[:2]+[-999], psi[:2]+[999])
        self.assertEqual(changed[:2], prefix)

    def test_constant_walk_reduces_to_ordinary_causal_attention(self):
        rows = causal_attention([[1]]*3, [[-1], [0], [1]], [[2], [3], [5]], [7]*3, [7]*3)
        for t, row in enumerate(rows):
            expected = [exp(x) for x in [-1, 0, 1][:t+1]]
            expected = [x/fsum(expected) for x in expected]
            self.assertEqual(row.biases, (0,)*(t+1))
            self.assertVectorClose(row.weights, expected)
            self.assertVectorClose(row.output, [fsum(a*b for a, b in zip(expected, [2, 3, 5]))])

    def test_diagonal_is_zero_even_when_current_field_is_below_own_level(self):
        rows = causal_attention([[1]], [[2]], [[8, -4]], [-100], [100])
        self.assertEqual(rows[0].biases, (0,))
        self.assertEqual(rows[0].weights, (1,))
        self.assertEqual(rows[0].output, (8, -4))

    def test_repeated_dip_preserves_bias_not_normalized_weight(self):
        short = causal_attention([[0]]*3, [[0]]*3, [[1]]*3, [2, 0, 2], [2, 0, 2])[-1]
        longer = causal_attention([[0]]*4, [[0]]*4, [[1]]*4, [2, 0, 0, 2], [2, 0, 0, 2])[-1]
        self.assertEqual(short.biases[0], longer.biases[0])
        self.assertLess(longer.weights[0], short.weights[0])

    def test_negative_bias_cannot_recover_after_field_rises(self):
        rows = first_passage_bias([2, 0, 3, -1, 9], [2, 0, 3, -1, 9])
        self.assertEqual([row[0] for row in rows], [0, -8, -8, -12, -12])
        for s in range(len(rows)):
            history = [row[s] for row in rows[s:]]
            self.assertTrue(all(a >= b for a, b in zip(history, history[1:])))

    def test_common_field_and_level_shift_leaves_bias_unchanged(self):
        phi, psi = [1, -2, 3, 0], [2, 1, 4, -1]
        self.assertEqual(first_passage_bias(phi, psi),
                         first_passage_bias([x+123 for x in phi], [x+123 for x in psi]))

    def test_nonincreasing_walk_matches_summed_log_forget_gates(self):
        increments, beta = [-0.2, -0.5, 0.0, -0.25], 1.3
        walk = walk_from_increments(increments)
        rows = first_passage_bias(walk, walk, beta)
        for t, row in enumerate(rows):
            for s, bias in enumerate(row):
                expected = fsum(log(exp(beta*delta)) for delta in increments[s+1:t+1])
                self.assertAlmostEqual(bias, expected, places=14)

    def test_expiry_specialization(self):
        walk, expiry, beta = [-1, -2, -3, -4], [2, 0, 1, 3], 2.0
        levels = [w-e for w, e in zip(walk, expiry, strict=True)]
        for t, row in enumerate(first_passage_bias(walk, levels, beta)):
            for s, bias in enumerate(row):
                self.assertEqual(bias, beta*min(0, expiry[s]-(t-s)))

    def test_sqrt_head_width_scale(self):
        rows = causal_attention([[2, 0]]*2, [[0, 0], [1, 0]], [[1], [0]], [0, 0], [0, 0])
        expected = 1/(1+exp(2/(2**0.5)))
        self.assertAlmostEqual(rows[-1].weights[0], expected, places=14)

    def test_content_logit_shift_preserves_weights(self):
        args = dict(queries=[[1]]*3, values=[[1], [2], [3]], field=[2, 0, 3], levels=[2, 0, 3])
        a = causal_attention(keys=[[-1], [0], [2]], **args)
        b = causal_attention(keys=[[999], [1000], [1002]], **args)
        for first, second in zip(a, b, strict=True):
            self.assertVectorClose(first.weights, second.weights)

    def test_linearity_in_values_with_fixed_fields(self):
        args = dict(queries=[[1]]*3, keys=[[0], [1], [2]], field=[2, 0, 3], levels=[2, 0, 3])
        a, b = [[1, 2], [3, 4], [5, 6]], [[-1, 3], [0, 2], [2, 1]]
        added = [[x+y for x, y in zip(r, s)] for r, s in zip(a, b)]
        for x, y, z in zip(causal_attention(values=a, **args), causal_attention(values=b, **args),
                           causal_attention(values=added, **args), strict=True):
            self.assertVectorClose(z.output, [v+w for v, w in zip(x.output, y.output)])

    def test_constant_values_and_valid_prefix_distributions(self):
        rows = causal_attention([[1]]*4, [[-3], [0], [1], [4]], [[3, -2]]*4,
                                [1, 2, 0, 3], [1, 2, 0, 3])
        for t, row in enumerate(rows):
            self.assertEqual(len(row.weights), t+1)
            self.assertTrue(all(weight >= 0 for weight in row.weights))
            self.assertAlmostEqual(fsum(row.weights), 1, places=14)
            self.assertVectorClose(row.output, [3, -2])

    def test_finite_bias_is_attenuation_with_expected_odds(self):
        for beta in (0.1, 1, 4, 20):
            row = causal_attention([[0]]*2, [[0]]*2, [[1], [0]], [1, 0], [1, 0], beta)[-1]
            self.assertGreater(row.weights[0], 0)
            self.assertAlmostEqual(row.weights[0]/row.weights[1], exp(-beta), places=14)

    def test_stable_softmax_and_documented_binary64_underflow(self):
        equal = causal_attention([[1]]*2, [[1000], [1000]], [[2], [4]], [0]*2, [0]*2)
        self.assertEqual(equal[-1].weights, (0.5, 0.5))
        tiny = causal_attention([[0]]*2, [[0]]*2, [[2], [4]], [1, 0], [1, 0], 1000)[-1]
        self.assertEqual(tiny.biases, (-1000, 0))
        self.assertEqual(tiny.weights, (0, 1))  # Numerical underflow, not a hard-mask rule.

    def test_invalid_strength(self):
        for beta in (0, -1, float("inf"), float("nan"), True, "4"):
            with self.subTest(beta=beta), self.assertRaises((TypeError, ValueError)):
                first_passage_bias([0], [0], beta)
            with self.subTest(cache_beta=beta), self.assertRaises((TypeError, ValueError)):
                IncrementalAttention(beta)

    def test_invalid_shapes(self):
        base = dict(queries=[[1], [1]], keys=[[1], [1]], values=[[1], [1]],
                    field=[0, 0], levels=[0, 0])
        for key, bad in (("queries", []), ("queries", [[1], [1, 2]]), ("keys", [[1]]),
                         ("keys", [[1, 2], [1, 2]]), ("values", [[1]]),
                         ("values", [[], []]), ("field", [0]), ("levels", [])):
            args = {**base, key: bad}
            with self.subTest(key=key, bad=bad), self.assertRaises(ValueError):
                causal_attention(**args)

    def test_reject_nonfinite_or_nonreal_inputs(self):
        base = dict(queries=[[1]], keys=[[1]], values=[[1]], field=[0], levels=[0])
        for name in base:
            for bad in (float("inf"), float("-inf"), float("nan"), True, "x"):
                shape = [[bad]] if name in ("queries", "keys", "values") else [bad]
                with self.subTest(name=name, bad=bad), self.assertRaises((TypeError, ValueError)):
                    causal_attention(**{**base, name: shape})

    def test_reject_unrepresentable_arithmetic(self):
        with self.assertRaises(ValueError):
            first_passage_bias([0, -1e308], [1e308, 0], 4)
        with self.assertRaises(ValueError):
            causal_attention([[1e308]], [[1e308]], [[1]], [0], [0])
        with self.assertRaises(ValueError):
            walk_from_increments([1e308, 1e308])
        # A positive overflowing subtraction would be clipped to zero; no need to form it.
        self.assertEqual(first_passage_bias([0, 1e308], [-1e308, 0]), ((0,), (0, 0)))

    def test_inputs_are_not_mutated(self):
        args = dict(queries=[[1], [2]], keys=[[2], [1]], values=[[2], [4]], field=[1, 0], levels=[1, 0])
        before = copy.deepcopy(args)
        causal_attention(**args)
        self.assertEqual(args, before)

    def test_rejected_incremental_steps_do_not_mutate_cache(self):
        actual, expected = IncrementalAttention(), IncrementalAttention()
        for decoder in (actual, expected):
            decoder.step([1], [2], [3], 1, 1)
        invalid = (([1, 2], [1, 2], [3], 0, 0), ([1], [1], [2, 3], 0, 0),
                   ([1], [1], [3], float("nan"), 0), ([1e308], [1e308], [3], 0, 0),
                   ([], [], [3], 0, 0))
        for arguments in invalid:
            with self.assertRaises((TypeError, ValueError)):
                actual.step(*arguments)
            self.assertEqual(len(actual), 1)
        self.assertEqual(actual.step([1], [2], [4], 2, 2), expected.step([1], [2], [4], 2, 2))

    def test_cache_copies_caller_owned_vectors(self):
        key, value = [2], [3]
        decoder = IncrementalAttention()
        decoder.step([1], key, value, 0, 0)
        key[0], value[0] = -999, -999
        row = decoder.step([1], [2], [5], 0, 0)
        self.assertEqual(row.weights, (0.5, 0.5))
        self.assertEqual(row.output, (4,))

    def test_walk_prefix_sums_and_validation(self):
        self.assertEqual(walk_from_increments([1, 1, 1, -2, 1, 0]), (1, 2, 3, 1, 2, 2))
        for bad in ([], [True], [float("nan")]):
            with self.assertRaises((TypeError, ValueError)):
                walk_from_increments(bad)

    def test_checked_sample_output(self):
        expected = json.loads(Path(__file__).with_name("sample_output.json").read_text())
        self.assertEqual(build_report(), expected)

    def test_cli_sample_changed_strength_and_rejected_arguments(self):
        command = [sys.executable, "-m", "examples.first_passage_attention.demo"]
        result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=10)
        self.assertEqual(result.stdout, Path(__file__).with_name("sample_output.json").read_text())
        stronger = subprocess.run(command+["--beta", "8"], capture_output=True, text=True, check=True, timeout=10)
        self.assertLess(json.loads(stronger.stdout)["first_passage_attention"]["draft_and_detail_mass"],
                        build_report()["first_passage_attention"]["draft_and_detail_mass"])
        for invalid in ("0", "-1", "nan", "inf"):
            bad = subprocess.run(command+["--beta", invalid], capture_output=True, text=True, timeout=10)
            self.assertNotEqual(bad.returncode, 0)
            self.assertIn("error:", bad.stderr)


if __name__ == "__main__":
    unittest.main()
