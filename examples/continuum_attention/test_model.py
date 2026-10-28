"""Analytical, independent quadrature, Decimal and causal-prefix verification."""

from decimal import Decimal, localcontext
import json
import math
from pathlib import Path
import subprocess
import sys
import unittest

from examples.continuum_attention.model import attend, segment_integrals
from examples.continuum_attention.demo import build_report


def decimal_reference(logits, values, caps=True):
    """70-digit direct positive/negative exponential moments, without reversal."""
    with localcontext() as context:
        context.prec = 70
        z = [Decimal(str(s)) for s in logits]
        shift = max(z)
        weights = [Decimal(0)]*len(z)
        if caps:
            weights[0] += (z[0]-shift).exp()/2
            weights[-1] += (z[-1]-shift).exp()/2
        for j in range(1, len(z)):
            delta = z[j]-z[j-1]
            if delta == 0:
                j0, j1 = Decimal(1), Decimal('0.5')
            else:
                j0 = (delta.exp()-1)/delta
                j1 = ((delta-1)*delta.exp()+1)/(delta*delta)
            scale = (z[j-1]-shift).exp()
            weights[j-1] += scale*(j0-j1)
            weights[j] += scale*j1
        total = sum(weights)
        p = [w/total for w in weights]
        out = [sum((w*Decimal(str(row[d])) for w, row in zip(p, values, strict=True)), Decimal(0))
               for d in range(len(values[0]))]
        return tuple(float(x) for x in out), tuple(float(x) for x in p)


def quadrature_reference(logits, values, panels=2048):
    """Composite Simpson integration of the interpolated functions themselves."""
    peak = max(logits)
    denominator = [0.5*math.exp(logits[0]-peak), 0.5*math.exp(logits[-1]-peak)]
    numerator = [[denominator[0]*values[0][d], denominator[1]*values[-1][d]]
                 for d in range(len(values[0]))]
    for j in range(1, len(logits)):
        for k in range(panels+1):
            u = k/panels
            simpson = (1 if k in (0, panels) else 4 if k % 2 else 2)/(3*panels)
            density = math.exp((1-u)*logits[j-1]+u*logits[j]-peak)
            weight = simpson*density
            denominator.append(weight)
            for d in range(len(values[0])):
                value = (1-u)*values[j-1][d]+u*values[j][d]
                numerator[d].append(weight*value)
    return tuple(math.fsum(terms)/math.fsum(denominator) for terms in numerator)


class ContinuumTests(unittest.TestCase):
    def assertVectorClose(self, a, b, delta=2e-12):
        self.assertEqual(len(a), len(b))
        for x, y in zip(a, b, strict=True):
            self.assertAlmostEqual(x, y, delta=delta)

    def test_near_zero_segments_match_decimal(self):
        for delta in [0, 1e-12, -1e-12, 1e-6, -1e-6, 0.499999, 0.5, 0.500001]:
            with self.subTest(delta=delta):
                s = segment_integrals(0, delta)
                expected, _ = decimal_reference([0, delta], [[0], [1]], caps=False)
                self.assertAlmostEqual(s.mean_coordinate, expected[0], delta=2e-13)
                self.assertAlmostEqual(s.left_coefficient+s.right_coefficient, s.mass, delta=1e-15)

    def test_general_segments_match_decimal(self):
        for a, b in [(2, -3), (-4, 1), (0, 50), (50, 0), (1000, -1000)]:
            s = segment_integrals(a, b)
            expected, _ = decimal_reference([a, b], [[0], [1]], caps=False)
            self.assertAlmostEqual(s.mean_coordinate, expected[0], delta=2e-13)

    def test_segment_orientation_mass_and_shift(self):
        forward, backward = segment_integrals(0, 2), segment_integrals(2, 0)
        self.assertEqual(forward.mass, backward.mass)
        self.assertEqual(forward.left_coefficient, backward.right_coefficient)
        self.assertEqual(forward.right_coefficient, backward.left_coefficient)
        self.assertAlmostEqual(forward.mass, (1-math.exp(-2))/2)
        shifted = segment_integrals(0, 2, shift=3)
        self.assertAlmostEqual(shifted.mass, forward.mass/math.e)
        self.assertEqual(shifted.mean_coordinate, forward.mean_coordinate)

    def test_retrieval_coordinate_monotonic_and_symmetric(self):
        means = [segment_integrals(0, d).mean_coordinate for d in [-10, -4, -1, 0, 1, 4, 10]]
        self.assertEqual(means, sorted(means))
        self.assertEqual(means[3], 0.5)
        self.assertTrue(all(0 < u < 1 for u in means))
        for d in [1e-8, 0.5, 4, 100]:
            self.assertAlmostEqual(segment_integrals(0, d).mean_coordinate+
                                   segment_integrals(0, -d).mean_coordinate, 1)

    def test_complete_operator_matches_decimal(self):
        v = [[1, -3], [5, 2], [-2, 7], [9, 1]]
        for z in [[0, 0, 0, 0], [1, -2, 3, 0], [0, 1e-12, -1e-12, 0.5], [1000, 9999, -1000, 2]]:
            for caps in [True, False]:
                with self.subTest(z=z, caps=caps):
                    output, weights = decimal_reference(z, v, caps)
                    row = attend(z, v, boundary_caps=caps)
                    self.assertVectorClose(row.output, output)
                    self.assertVectorClose(row.token_weights, weights)

    def test_independent_numerical_integration(self):
        v = [[1, -3], [5, 2], [-2, 7], [9, 1]]
        for z in [[0, 0, 0, 0], [1, -2, 2, 0], [0, 1e-8, -1e-8, 0.5]]:
            self.assertVectorClose(attend(z, v).output, quadrature_reference(z, v), delta=2e-11)

    def test_equal_logits_give_uniform_weights_for_every_prefix_size(self):
        for n in range(1, 10):
            row = attend([3]*n, [[i] for i in range(n)])
            self.assertVectorClose(row.token_weights, [1/n]*n)
            self.assertVectorClose(row.output, [(n-1)/2])

    def test_endpoint_caps_restore_uniform_mass(self):
        z, v = [0, 0, 0], [[0], [1], [0]]
        correct, ablation = attend(z, v), attend(z, v, boundary_caps=False)
        self.assertVectorClose(correct.output, [1/3])
        self.assertVectorClose(ablation.token_weights, [1/4, 1/2, 1/4])
        self.assertVectorClose(ablation.output, [1/2])

    def test_one_visible_token_has_two_half_caps_and_no_segments(self):
        row = attend([999], [[3, -2]])
        self.assertEqual(row.output, (3, -2))
        self.assertEqual(row.token_weights, (1,))
        self.assertEqual(row.boundary_probability_mass, 1)
        self.assertEqual(row.segment_probability_masses, ())

    def test_constant_values_are_preserved(self):
        self.assertVectorClose(attend([4, -2, 1, 3], [[7, -5]]*4).output, [7, -5])

    def test_linearity_in_values(self):
        z = [1, -3, 2]
        v, u = [[1], [4], [-2]], [[-3], [8], [2]]
        mixed = [[2*a[0]-3*b[0]] for a, b in zip(v, u, strict=True)]
        self.assertVectorClose(attend(z, mixed).output,
                               [2*attend(z, v).output[0]-3*attend(z, u).output[0]])

    def test_common_logit_shift_invariance(self):
        z, v = [1, -3, 2], [[1], [4], [-2]]
        a, b = attend(z, v), attend([x+10000 for x in z], v)
        self.assertVectorClose(a.output, b.output)
        self.assertVectorClose(a.token_weights, b.token_weights)
        self.assertEqual(a.segment_mean_coordinates, b.segment_mean_coordinates)

    def test_reversing_entire_memory_curve_preserves_output(self):
        z, v = [1, -3, 2, 0], [[1], [4], [-2], [8]]
        a, b = attend(z, v), attend(z[::-1], v[::-1])
        self.assertVectorClose(a.output, b.output)
        self.assertVectorClose(a.token_weights, b.token_weights[::-1])

    def test_arbitrary_permutation_changes_adjacency_and_output(self):
        a = attend([0, 2, -1], [[0], [1], [0]])
        b = attend([2, 0, -1], [[1], [0], [0]])
        self.assertGreater(abs(a.output[0]-b.output[0]), 0.1)

    def test_weights_and_segment_masses_normalize_and_output_is_convex(self):
        v = [[1, -3], [5, 2], [-2, 7], [9, 1]]
        row = attend([1, -2, 3, 0], v)
        self.assertTrue(all(w >= 0 for w in row.token_weights))
        self.assertAlmostEqual(math.fsum(row.token_weights), 1)
        self.assertAlmostEqual(math.fsum(row.segment_probability_masses)+row.boundary_probability_mass, 1)
        for d, value in enumerate(row.output):
            self.assertGreaterEqual(value, min(r[d] for r in v))
            self.assertLessEqual(value, max(r[d] for r in v))

    def test_prefix_call_matches_explicit_truncation(self):
        z, v = [1, -2, 3, 0], [[1], [5], [-2], [9]]
        for n in range(1, 5):
            row = attend(z, v, prefix_length=n)
            truncated = attend(z[:n], v[:n])
            self.assertEqual(row.output, truncated.output)
            self.assertEqual(row.token_weights[:n], truncated.token_weights)
            self.assertEqual(row.token_weights[n:], (0.0,)*(4-n))

    def test_future_cannot_change_visible_segments_or_last_cap(self):
        z, v = [0, 1], [[0], [1]]
        before = attend(z, v)
        after = attend(z+[1e6], v+[[1e100]], prefix_length=2)
        self.assertEqual(before.output, after.output)
        self.assertEqual(before.segment_probability_masses, after.segment_probability_masses)
        self.assertEqual(before.boundary_probability_mass, after.boundary_probability_mass)
        self.assertEqual(after.token_weights[-1], 0)

    def test_interpolated_key_score_is_affine(self):
        q, a, b = [2, -1], [3, 4], [-2, 5]
        def score(k):
            return sum(x*y for x, y in zip(q, k, strict=True))/math.sqrt(2)
        for u in [0, 0.25, 0.5, 1]:
            k = [(1-u)*x+u*y for x, y in zip(a, b, strict=True)]
            self.assertAlmostEqual(score(k), (1-u)*score(a)+u*score(b))

    def test_extreme_scores_stay_finite_and_match_decimal(self):
        z, v = [-10000, 10000, -9999], [[0], [1], [3]]
        row = attend(z, v)
        expected, weights = decimal_reference(z, v)
        self.assertTrue(math.isfinite(row.output[0]))
        self.assertVectorClose(row.output, expected)
        self.assertVectorClose(row.token_weights, weights)

    def test_huge_common_offset_keeps_uniform_weights(self):
        row = attend([1e308]*3, [[1], [2], [6]])
        self.assertVectorClose(row.token_weights, [1/3]*3)
        self.assertVectorClose(row.output, [3])

    def test_large_opposite_values_avoid_intermediate_overflow(self):
        self.assertEqual(attend([0, 0], [[1e308], [-1e308]]).output, (0,))

    def test_invalid_data_are_rejected(self):
        for z, v in [([], []), ([1], []), ([1], [[]]), ([1, 2], [[1], [1, 2]]),
                     ([math.nan], [[1]]), ([math.inf], [[1]]), ([True], [[1]]),
                     ([1], [[math.inf]]), ([1], [[True]]), (None, [[1]]), ([1], [1])]:
            with self.subTest(z=z), self.assertRaises(ValueError):
                attend(z, v)
        with self.assertRaises(ValueError):
            attend([0, math.nan], [[1], [2]], prefix_length=1)

    def test_invalid_prefix_and_boundary_options(self):
        for kwargs in [{"prefix_length": 0}, {"prefix_length": 3}, {"prefix_length": True},
                       {"prefix_length": 1.0}, {"boundary_caps": 1},
                       {"prefix_length": 1, "boundary_caps": False}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                attend([0, 1], [[1], [2]], **kwargs)

    def test_invalid_segment_shift_and_overflowing_difference(self):
        for a, b, kwargs in [(0, 1, {"shift": 0}), (0, 1, {"shift": math.nan}),
                             (math.inf, 1, {}), (1e308, -1e308, {})]:
            with self.subTest(a=a, b=b), self.assertRaises(ValueError):
                segment_integrals(a, b, **kwargs)

    def test_checked_output_matches_cli_bytes(self):
        proc = subprocess.run([sys.executable, '-m', 'examples.continuum_attention.demo'],
                              check=True, capture_output=True, text=True)
        sample = Path(__file__).with_name('sample_output.json').read_text()
        self.assertEqual(proc.stdout, sample)
        self.assertEqual(json.loads(sample), build_report())

    def test_cli_gap_control_and_invalid_values(self):
        cmd = [sys.executable, '-m', 'examples.continuum_attention.demo']
        proc = subprocess.run(cmd+['--gap', '0'], check=True, capture_output=True, text=True)
        row = json.loads(proc.stdout)['two_token_read']
        self.assertEqual(row['continuum']['output'], row['discrete_softmax_output'])
        for value in ['-101', '101', 'nan', 'inf']:
            proc = subprocess.run(cmd+['--gap', value], capture_output=True, text=True)
            self.assertEqual(proc.returncode, 2)
            self.assertIn('error:', proc.stderr)


if __name__ == '__main__':
    unittest.main()
