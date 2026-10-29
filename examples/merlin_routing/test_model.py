"""Independent scalar-score Decimal oracle and structural regression tests."""

from decimal import Decimal, localcontext
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import unittest

from examples.merlin_routing.demo import build_report, dense_segment_mass
from examples.merlin_routing.model import SegmentMoments, route


def decimal_oracle(segments, query, readouts, gamma=0.5, tm=1.0, tv=1.0):
    """80-digit oracle directly from token scores: no covariance matrices."""
    D = lambda x: Decimal(str(x))
    with localcontext() as context:
        context.prec = 80
        scores = [[sum((D(q)*D(k) for q, k in zip(query, key, strict=True)), Decimal(0))
                   for key in segment] for segment in segments]
        means = [sum(row)/len(row) for row in scores]
        variances = [sum((x-m)**2 for x in row)/len(row)
                     for row, m in zip(scores, means, strict=True)]
        def normalized(row, temperature):
            exps = [(D(temperature)*(x-max(row))).exp() for x in row]
            return [x/sum(exps) for x in exps]
        pm, pv = normalized(means, tm), normalized(variances, tv)
        weights = [D(gamma)*a+(1-D(gamma))*b for a, b in zip(pm, pv, strict=True)]
        output = [sum(w*D(row[d]) for w, row in zip(weights, readouts, strict=True))
                  for d in range(len(readouts[0]))]
        return tuple(tuple(float(x) for x in row) for row in (means, variances, pm, pv, weights, output))


def run(segments, query=(1,), readouts=((1, 0), (0, 1)), **kwargs):
    return route([SegmentMoments(s) for s in segments], query, readouts, **kwargs)


class MerlinTests(unittest.TestCase):
    def assertVector(self, actual, expected, tolerance=2e-13):
        self.assertEqual(len(actual), len(expected))
        for a, b in zip(actual, expected, strict=True):
            self.assertTrue(math.isclose(a, b, abs_tol=tolerance, rel_tol=tolerance), (a, b))

    def test_decimal_oracle_multidimensional(self):
        segments = [[[1, 2], [-3, 4], [0, -2]], [[2, -1], [3, 0]], [[-1, -1]]]
        query, readouts = [0.5, -1.25], [[2, -3], [0, 7], [5, 1]]
        result = run(segments, query, readouts, gamma=0.37, tau_mean=0.8, tau_variance=0.2)
        oracle = decimal_oracle(segments, query, readouts, 0.37, 0.8, 0.2)
        for actual, expected in zip(vars(result).values(), oracle, strict=True):
            self.assertVector(actual, expected)

    def test_seeded_decimal_oracle(self):
        rng = random.Random(20261029)
        for _ in range(40):
            segments = [[[rng.randint(-12, 12)/4 for _ in range(3)]
                         for _ in range(rng.randint(1, 6))] for _ in range(4)]
            query = [rng.randint(-4, 4)/4 for _ in range(3)]
            readouts = [[rng.randint(-8, 8)/4 for _ in range(2)] for _ in segments]
            result = run(segments, query, readouts, gamma=0.3, tau_mean=1.5, tau_variance=0.2)
            for actual, expected in zip(vars(result).values(), decimal_oracle(segments, query, readouts, 0.3, 1.5, 0.2), strict=True):
                self.assertVector(actual, expected)

    def test_equal_mean_variance_selects_first(self):
        result = run([[[-2], [2]], [[0], [0]]])
        self.assertVector(result.mean_scores, [0, 0])
        self.assertVector(result.variance_scores, [4, 0])
        self.assertVector(result.weights, [0.25+0.5/(1+math.exp(-4)), 0.25+0.5/(1+math.exp(4))])

    def test_not_softmax_of_combined_scores(self):
        result = run([[[-2], [2]], [[0], [0]]])
        wrong = 1/(1+math.exp(-2))
        self.assertGreater(abs(result.weights[0]-wrong), 0.13)

    def test_population_not_sample_variance(self):
        self.assertEqual(SegmentMoments([[0], [2]]).scores([1]), (1, 1))

    def test_off_diagonal_covariance_cancels(self):
        moments = SegmentMoments([[1, 1], [-1, -1]])
        self.assertEqual(moments.covariance, ((1, 1), (1, 1)))
        self.assertEqual(moments.scores([1, -1]), (0, 0))
        self.assertEqual(moments.scores([1, 1]), (0, 4))

    def test_gate_endpoints(self):
        for gate, name in [(0, 'variance_weights'), (1, 'mean_weights')]:
            result = run([[[-2], [2]], [[1], [1]]], gamma=gate)
            self.assertEqual(result.weights, getattr(result, name))

    def test_zero_query(self):
        result = run([[[-2], [2]], [[1], [3]]], query=[0])
        self.assertEqual(result.weights, (0.5, 0.5))

    def test_zero_temperatures(self):
        result = run([[[-2], [2]], [[1], [3]]], tau_mean=0, tau_variance=0)
        self.assertEqual(result.weights, (0.5, 0.5))

    def test_singleton_segments(self):
        result = run([[[2]], [[-1]]])
        self.assertEqual(result.variance_scores, (0, 0))
        self.assertEqual(result.variance_weights, (0.5, 0.5))

    def test_one_segment_returns_readout(self):
        result = run([[[1], [-2]]], readouts=[[3, -4]])
        self.assertEqual(result.weights, (1,))
        self.assertEqual(result.output, (3, -4))

    def test_equal_segments_uniform(self):
        result = run([[[1], [-2]], [[1], [-2]]], gamma=0.9)
        self.assertEqual(result.weights, (0.5, 0.5))

    def test_probability_normalization(self):
        result = run([[[1, 2], [-2, 3]], [[4, -1]]], query=[-2, 3])
        for weights in [result.weights, result.mean_weights, result.variance_weights]:
            self.assertTrue(all(0 <= w <= 1 for w in weights))
            self.assertAlmostEqual(math.fsum(weights), 1)

    def test_within_segment_permutation(self):
        segments = [[[1, 2], [-2, 3], [3, 5]], [[4, -1], [0, 0]]]
        a = run(segments, [0.2, -0.3])
        b = run([list(reversed(s)) for s in segments], [0.2, -0.3])
        self.assertVector(a.weights, b.weights)

    def test_segment_permutation_equivariance(self):
        segments = [[[-2], [2]], [[0], [0]]]
        a = run(segments)
        b = run(segments[::-1], readouts=[[0, 1], [1, 0]])
        self.assertVector(a.weights, b.weights[::-1])
        self.assertVector(a.output, b.output)

    def test_replication_invariant_but_dense_mass_changes(self):
        segments = [[[-2], [2]], [[0], [0]]]
        copied = [segments[0]*3, segments[1]]
        self.assertVector(run(segments).weights, run(copied).weights)
        self.assertGreater(dense_segment_mass(copied, [1])[0], dense_segment_mass(segments, [1])[0])

    def test_prefix_excludes_future(self):
        a = SegmentMoments([[1, 2], [3, -1], [999, 999]], prefix_length=2)
        b = SegmentMoments([[1, 2], [3, -1]])
        self.assertEqual(a, b)
        self.assertEqual(a.count, 2)
        self.assertFalse(hasattr(a, 'keys'))

    def test_causal_routing_uses_visible_prefix_only(self):
        fixed = SegmentMoments([[-1], [1]])
        a = route([fixed, SegmentMoments([[2], [1000]], prefix_length=1)], [1], [[1], [0]])
        b = route([fixed, SegmentMoments([[2], [-1000]], prefix_length=1)], [1], [[1], [0]])
        self.assertEqual(a, b)

    def test_large_offset_small_variance(self):
        summary = SegmentMoments([[1e12-2], [1e12+2]])
        self.assertEqual(summary.scores([1]), (1e12, 4))
        raw_variance = ((1e12-2)**2+(1e12+2)**2)/2-1e24
        self.assertNotEqual(raw_variance, 4)

    def test_stable_large_logits_and_temperatures(self):
        result = run([[[1e150]], [[-1e150]]], tau_mean=1e200, tau_variance=1e200)
        self.assertEqual(result.mean_weights, (1, 0))
        self.assertEqual(result.variance_weights, (0.5, 0.5))

    def test_huge_opposite_logits_small_temperature(self):
        result = run([[[1e308]], [[-1e308]]], gamma=1, tau_mean=1e-308)
        self.assertVector(result.weights, [1/(1+math.exp(-2)), 1/(1+math.exp(2))])

    def test_max_float_constant_readout(self):
        maximum = sys.float_info.max
        result = run([[[-4]], [[-1]]], gamma=0.7, readouts=[[maximum], [maximum]])
        self.assertEqual(result.output, (maximum,))

    def test_large_finite_readouts(self):
        result = run([[[0]], [[0]]], readouts=[[1e308, -1e308], [1e308, 1e308]])
        self.assertEqual(result.output, (1e308, 0))

    def test_linear_in_readouts(self):
        segments = [[[-2], [2]], [[0], [0]]]
        a = run(segments, readouts=[[2], [-1]])
        b = run(segments, readouts=[[3], [7]])
        combined = run(segments, readouts=[[2*2-3], [2*(-1)-7]])
        self.assertVector(combined.output, [2*a.output[0]-b.output[0]])

    def test_query_scaling_mean_and_variance(self):
        moments = SegmentMoments([[1, 3], [-2, 4]])
        mean, variance = moments.scores([0.5, -1])
        scaled_mean, scaled_variance = moments.scores([-1, 2])
        self.assertAlmostEqual(scaled_mean, -2*mean)
        self.assertAlmostEqual(scaled_variance, 4*variance)

    def test_two_moments_do_not_recover_dense_attention(self):
        segments = [[[-5], [0], [0], [5]], [[-4], [-3], [3], [4]]]
        result = run(segments)
        self.assertEqual(result.mean_scores, (0, 0))
        self.assertEqual(result.variance_scores, (12.5, 12.5))
        self.assertEqual(result.weights, (0.5, 0.5))
        self.assertGreater(dense_segment_mass(segments, [1])[0], 0.66)

    def test_invalid_key_shapes(self):
        for keys in [[], [[]], [[1], [1, 2]], None, [1, 2]]:
            with self.subTest(keys=keys), self.assertRaises(ValueError):
                SegmentMoments(keys)

    def test_invalid_prefix(self):
        for prefix in [0, -1, 3, True, 1.0]:
            with self.subTest(prefix=prefix), self.assertRaises(ValueError):
                SegmentMoments([[0], [1]], prefix_length=prefix)

    def test_nonfinite_and_boolean_inputs(self):
        for value in [float('inf'), float('-inf'), float('nan'), True]:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    SegmentMoments([[value]])
                for kwargs in [dict(query=[value]), dict(readouts=[[value], [0]]), dict(gamma=value), dict(tau_mean=value), dict(tau_variance=value)]:
                    with self.assertRaises(ValueError):
                        run([[[0]], [[1]]], **kwargs)

    def test_invalid_query_and_readout_shapes(self):
        for kwargs in [dict(query=[]), dict(query=[1, 2]), dict(query=None), dict(readouts=[]), dict(readouts=[[1]]), dict(readouts=[[1], [2, 3]]), dict(readouts=[[], []]), dict(readouts=None)]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                run([[[0]], [[1]]], **kwargs)

    def test_invalid_gate_and_temperature_ranges(self):
        for kwargs in [dict(gamma=-0.01), dict(gamma=1.01), dict(tau_mean=-1), dict(tau_variance=-1)]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                run([[[0]], [[1]]], **kwargs)

    def test_reject_unrepresentable_intermediates(self):
        with self.assertRaises(ValueError):
            SegmentMoments([[-1e308], [1e308]])
        with self.assertRaises(ValueError):
            SegmentMoments([[-1e200], [1e200]])
        with self.assertRaises(ValueError):
            SegmentMoments([[1e308]]).scores([2])

    def test_invalid_segment_objects(self):
        for segments in [[], None, [[1]], [object()]]:
            with self.subTest(segments=segments), self.assertRaises(ValueError):
                route(segments, [1], [[1]])

    def test_supplied_readouts_not_token_values(self):
        a = run([[[-2], [2]], [[0], [0]]], readouts=[[8], [-3]])
        self.assertVector(a.output, [8*a.weights[0]-3*a.weights[1]])
        self.assertVector(a.weights, run([[[-2], [2]], [[0], [0]]]).weights)

    def test_checked_report(self):
        expected = json.loads(Path(__file__).with_name('sample_output.json').read_text())
        self.assertEqual(build_report(), expected)

    def test_cli_byte_reproduction(self):
        result = subprocess.run([sys.executable, '-m', 'examples.merlin_routing.demo'], check=True, capture_output=True, text=True)
        self.assertEqual(result.stdout, Path(__file__).with_name('sample_output.json').read_text())
        self.assertEqual(result.stderr, '')


if __name__ == '__main__':
    unittest.main()
