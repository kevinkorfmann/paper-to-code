from decimal import Decimal, localcontext
import json
from math import exp, fsum, inf, nan, sqrt
from pathlib import Path
import subprocess
import sys
import unittest

from examples.multi_head_attention_residuals.demo import SOURCES, build_report
from examples.multi_head_attention_residuals.model import rms_keys, route_depth


class DepthRoutingTests(unittest.TestCase):
    def test_independent_decimal_reference(self):
        sources = [[3, -1, 2, 0], [0, 4, -2, 1], [-2, 1, 3, -4]]
        query, gain = [.5, -.2, 1, -.3], [1.2, .7, 1.1, .8]
        with localcontext() as context:
            context.prec = 60
            D = lambda x: Decimal(str(x))
            keys = [[D(x)*D(g)/(sum(D(z)**2 for z in row)/4+D('1e-6')).sqrt()
                     for x, g in zip(row, gain)] for row in sources]
            for heads in (1, 2, 4):
                actual = route_depth(sources, query, heads=heads, gain=gain)
                expected = []
                for start in range(0, 4, 4//heads):
                    stop = start+4//heads
                    scores = [sum(D(query[c])*row[c] for c in range(start, stop)).exp() for row in keys]
                    for c in range(start, stop):
                        expected.append(sum(scores[i]*D(sources[i][c]) for i in range(3))/sum(scores))
                for a, b in zip(actual.output, expected):
                    self.assertAlmostEqual(a, float(b), places=13)

    def test_single_head_reduces_to_full_width_attention_residual(self):
        # Full-row RMS is equal for the two sources. No sqrt(D) logit scaling.
        read = route_depth([[3, 4], [4, 3]], [.5, -.25])
        delta = .75/sqrt(12.5+1e-6)
        second = 1/(1+exp(-delta))
        self.assertAlmostEqual(read.weights[0][1], second, places=14)
        self.assertAlmostEqual(read.output[0], 3+second, places=14)

    def test_zero_query_is_depth_mean(self):
        sources = [[3, 2, -4, 1], [1, 4, 2, 7], [-1, 0, 8, -2]]
        for heads in (1, 2, 4):
            read = route_depth(sources, [0]*4, heads=heads)
            for row in read.weights:
                self.assertEqual(row, (1/3,)*3)
            for actual, column in zip(read.output, zip(*sources)):
                self.assertAlmostEqual(actual, fsum(column)/3)

    def test_one_source_returns_raw_source(self):
        for heads in (1, 2, 4):
            self.assertEqual(route_depth([[30, -4, 8, 2]], [1, 2, -1, 3], heads=heads).output, (30, -4, 8, 2))

    def test_rms_statistic_spans_full_row(self):
        keys = rms_keys([[3, 4, 0, 0]])[0]
        expected = [3/sqrt(6.25+1e-6), 4/sqrt(6.25+1e-6), 0, 0]
        for a, b in zip(keys, expected):
            self.assertAlmostEqual(a, b, places=14)

    def test_gain_changes_keys_but_not_values(self):
        read = route_depth([[10, 0], [0, 20]], [3, 2], gain=[0, 0], heads=2)
        self.assertEqual(read.keys, ((0, 0), (0, 0)))
        self.assertEqual(read.output, (5, 10))

    def test_head_weights_are_probabilities(self):
        for row in route_depth(SOURCES, [1, -2, 3, -4], heads=2).weights:
            self.assertAlmostEqual(fsum(row), 1)
            self.assertTrue(all(0 <= x <= 1 for x in row))

    def test_depth_permutation_reorders_weights_only(self):
        sources = [[1, 3, 4, 2], [-1, 0, 2, 5], [0, -2, 1, 3]]
        order = [2, 0, 1]
        original = route_depth(sources, [1, -1, 2, -2], heads=2)
        permuted = route_depth([sources[i] for i in order], [1, -1, 2, -2], heads=2)
        for a, b in zip(original.output, permuted.output):
            self.assertAlmostEqual(a, b)
        for a, b in zip(original.weights, permuted.weights):
            self.assertEqual(tuple(a[i] for i in order), b)

    def test_head_permutation_equivariance(self):
        sources = [[1, 3, 4, 2], [-1, 0, 2, 5]]
        query = [1, -1, 2, -2]
        swap = lambda row: row[2:]+row[:2]
        original = route_depth(sources, query, heads=2)
        swapped = route_depth([swap(row) for row in sources], swap(query), heads=2)
        self.assertEqual(swapped.output, swap(original.output))
        self.assertEqual(swapped.weights, original.weights[::-1])

    def test_head_query_does_not_change_other_head(self):
        original = route_depth(SOURCES, [1, -1, 1, -1], heads=2)
        changed = route_depth(SOURCES, [-2, 3, 1, -1], heads=2)
        self.assertNotEqual(original.weights[0], changed.weights[0])
        self.assertEqual(original.weights[1], changed.weights[1])
        self.assertEqual(original.output[2:], changed.output[2:])

    def test_mixture_is_coordinate_bounded_but_not_full_vector_convex(self):
        report = build_report()
        output = report["two_heads"]["output"]
        for y, column in zip(output, zip(*SOURCES)):
            self.assertLessEqual(y, max(column))
            self.assertGreaterEqual(y, min(column))
        # Every full-vector convex mixture has coordinate 0 + coordinate 2 == 2.
        self.assertGreater(output[0]+output[2], 2)

    def test_query_derivative_against_softmax_identity(self):
        query = [0.3, -0.7, 0.8, -0.2]
        read = route_depth(SOURCES, query, heads=2)
        step = 1e-6
        plus, minus = query.copy(), query.copy()
        plus[0] += step
        minus[0] -= step
        numerical = (route_depth(SOURCES, plus, heads=2).output[0]-route_depth(SOURCES, minus, heads=2).output[0])/(2*step)
        mean_key = fsum(p*k[0] for p,k in zip(read.weights[0], read.keys))
        expected = fsum(p*s[0]*(k[0]-mean_key) for p,s,k in zip(read.weights[0], SOURCES, read.keys))
        self.assertAlmostEqual(numerical, expected, places=9)

    def test_zero_and_extreme_inputs_are_finite(self):
        self.assertEqual(rms_keys([[0, 0]]), ((0, 0),))
        key = rms_keys([[1e300, -1e300]])[0]
        self.assertEqual(key, (1, -1))
        read = route_depth([[1, -1], [-1, 1]], [1000, -1000], heads=2)
        self.assertEqual(read.output, (1, -1))

    def test_identical_sources_cannot_be_improved_by_routing(self):
        source = [2, 0, -3, 8]
        for heads in (1, 2, 4):
            self.assertEqual(route_depth([source]*3, [4, -2, 6, 1], heads=heads).output, tuple(source))

    def test_invalid_shapes_values_and_head_counts(self):
        bad = [dict(sources=[]), dict(sources=[[1, 2], [1]]), dict(query=[1]),
               dict(heads=0), dict(heads=3), dict(heads=True), dict(heads=1.0),
               dict(query=[nan, 0, 0, 0]), dict(epsilon=0), dict(epsilon=-1),
               dict(epsilon=inf), dict(gain=[1]), dict(gain=[1, 1, 1, inf])]
        for patch in bad:
            args = dict(sources=SOURCES, query=[1, -1, 1, -1], heads=2)
            args.update(patch)
            with self.subTest(patch=patch), self.assertRaises((ValueError, TypeError)):
                route_depth(**args)

    def test_input_not_mutated(self):
        sources = [list(row) for row in SOURCES]
        query = [1, -1, 1, -1]
        before = json.dumps([sources, query])
        route_depth(sources, query, heads=2)
        self.assertEqual(json.dumps([sources, query]), before)

    def test_checked_output(self):
        report = build_report()
        self.assertEqual(report, json.loads(Path(__file__).with_name("sample_output.json").read_text()))
        self.assertEqual(report["single_head"]["output"], [1, 1, 1, 1])
        self.assertEqual(report["two_heads"]["output"], [1.8, .2, 1.8, .2])

    def test_cli_counterexample_and_invalid_logit(self):
        result = subprocess.run([sys.executable, "-m", "examples.multi_head_attention_residuals.demo", "--strength", "0"],
                                capture_output=True, text=True, check=True, timeout=30)
        report = json.loads(result.stdout)
        self.assertEqual(report["single_head"]["output"], report["two_heads"]["output"])
        bad = subprocess.run([sys.executable, "-m", "examples.multi_head_attention_residuals.demo", "--strength", "nan"],
                             capture_output=True, text=True, timeout=30)
        self.assertNotEqual(bad.returncode, 0)


if __name__ == "__main__":
    unittest.main()
