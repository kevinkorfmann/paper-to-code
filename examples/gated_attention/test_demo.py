import decimal
import json
import math
from pathlib import Path
import subprocess
import sys
import unittest

from examples.gated_attention.demo import build_report
from examples.gated_attention.model import gated_attention, sigmoid


class GatedAttentionTests(unittest.TestCase):
    def compute(self, **kwargs):
        inputs = dict(queries=[[1, 2], [-1, 0.5]], keys=[[0.5, -1], [2, 1]],
                      values=[[3, -4], [7, 2]], hidden=[[2, 1], [-1, 3]],
                      gate_weights=[0.3, -0.2])
        inputs.update(kwargs)
        return gated_attention(**inputs)

    def test_independent_decimal_reference(self):
        with decimal.localcontext() as context:
            context.prec = 60
            D = decimal.Decimal
            q, k, v, x, w = ([[1, 2], [-1, D('0.5')]], [[D('0.5'), -1], [2, 1]],
                              [[3, -4], [7, 2]], [[2, 1], [-1, 3]], [D('.3'), D('-.2')])
            observed = self.compute()
            for i in range(2):
                weights = [(sum(D(a)*D(b) for a,b in zip(q[i], row))/D(2).sqrt()).exp() for row in k]
                total = sum(weights)
                g = 1/(1+(-sum(D(a)*b for a,b in zip(x[i], w))).exp())
                for c in range(2):
                    expected = g*sum(weights[j]*D(v[j][c])/total for j in range(2))
                    self.assertAlmostEqual(observed.gated_output[i][c], float(expected), places=13)

    def test_zero_projection_halves_output(self):
        result = self.compute(gate_weights=[0, 0])
        self.assertEqual(result.gates, (0.5, 0.5))
        self.assertEqual(result.gated_output, tuple(tuple(a/2 for a in row) for row in result.ordinary_output))

    def test_extreme_sigmoid_is_stable(self):
        self.assertEqual(sigmoid(-1000), 0)
        self.assertEqual(sigmoid(1000), 1)
        for z in [-4, -1, 0, 1, 4]:
            self.assertAlmostEqual(sigmoid(z)+sigmoid(-z), 1)

    def test_gate_does_not_renormalize_attention(self):
        a, b = self.compute(), self.compute(gate_weights=[-4, 5])
        self.assertEqual(a.attention, b.attention)
        self.assertEqual(a.ordinary_output, b.ordinary_output)
        self.assertNotEqual(a.gated_output, b.gated_output)
        for row, g in zip(a.attention, a.gates):
            self.assertAlmostEqual(sum(row), 1)
            self.assertAlmostEqual(sum(g*p for p in row), g)

    def test_causal_future_values_do_not_leak(self):
        a = self.compute(causal=True)
        b = self.compute(causal=True, values=[[3, -4], [999, 888]])
        self.assertEqual(a.attention[0], (1, 0))
        self.assertEqual(a.gated_output[0], b.gated_output[0])

    def test_query_gate_uses_query_hidden_state(self):
        a = self.compute()
        b = self.compute(hidden=[[2, 1], [99, 99]])
        self.assertEqual(a.gates[0], b.gates[0])
        self.assertNotEqual(a.gates[1], b.gates[1])

    def test_magnitude_and_sign(self):
        result = self.compute()
        for old, new in zip(result.ordinary_output, result.gated_output):
            for a, b in zip(old, new):
                self.assertLessEqual(abs(b), abs(a))
                self.assertGreaterEqual(a*b, 0)

    def test_gate_gradient(self):
        h = 1e-6
        for z in [-3, 0, 2]:
            numerical = (sigmoid(z+h)-sigmoid(z-h))/(2*h)
            self.assertAlmostEqual(numerical, sigmoid(z)*(1-sigmoid(z)), places=9)

    def test_post_gate_differs_from_key_value_gate(self):
        # Uniform A: query 0 output gate=.1 gives .1*(1+3)/2=.2.
        # Applying the two gates to values first gives (.1*1+.9*3)/2=1.4.
        result = gated_attention([[0], [0]], [[0], [0]], [[1], [3]],
                                 [[-math.log(9)], [math.log(9)]], [1])
        self.assertAlmostEqual(result.gated_output[0][0], 0.2)
        self.assertNotAlmostEqual(result.gated_output[0][0], 1.4)

    def test_invalid_inputs(self):
        for patch in [dict(keys=[[1]]), dict(hidden=[[1]]), dict(gate_weights=[]),
                      dict(values=[[math.nan], [0]]), dict(causal=1)]:
            with self.subTest(patch=patch), self.assertRaises((ValueError, TypeError)):
                self.compute(**patch)

    def test_checked_output_and_constructed_cases(self):
        report = build_report()
        self.assertEqual(report, json.loads(Path(__file__).with_name("sample_output.json").read_text()))
        self.assertEqual([row["gated_output"] for row in report["queries"]],
                         [[1, 0.08], [5, 0.4], [9, 0.72]])

    def test_cli(self):
        result = subprocess.run([sys.executable, "-m", "examples.gated_attention.demo", "--gate-logit", "0"],
                                capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(result.stdout)["queries"][0]["gate"], 0.5)
        bad = subprocess.run([sys.executable, "-m", "examples.gated_attention.demo", "--gate-logit", "nan"],
                             capture_output=True, text=True)
        self.assertNotEqual(bad.returncode, 0)


if __name__ == "__main__":
    unittest.main()
