import copy
from decimal import Decimal, localcontext
import json
from math import fsum, log
from pathlib import Path
import subprocess
import sys
import unittest

from examples.attention_abstention.demo import build_report
from examples.attention_abstention.model import attend, projection_gates


def decimal_oracle(scores, values, gates, sink, mask):
    """Independent direct exponentiation at 60 digits, without production softmax."""
    with localcontext() as context:
        context.prec = 60
        d = lambda x: Decimal(str(x))
        e = [d(x).exp() if allowed else Decimal(0) for x, allowed in zip(scores, mask)]
        phantom = Decimal(0) if sink is None else d(sink).exp()
        denominator = sum(e)+phantom
        p = [x/denominator for x in e]
        c = [x*d(g) for x, g in zip(p, gates)]
        mass = sum(c)
        output = [sum(c[i]*d(values[i][j]) for i in range(len(p))) for j in range(len(values[0]))]
        return tuple(map(float, p)), float(phantom/denominator), float(mass), tuple(map(float, output))


class AbstentionTests(unittest.TestCase):
    def assertVectorClose(self, actual, expected, places=12):
        self.assertEqual(len(actual), len(expected))
        for a, b in zip(actual, expected):
            self.assertAlmostEqual(a, b, places=places)

    def test_independent_decimal_reference(self):
        scores, values, gates = [-8, 0.7, 2.3], [[2, -1], [-3, 4], [0.5, 2]], [0.2, 0.7, 0.9]
        for sink in (None, -2.0, 0.0, 3.0):
            for mask in ([True, True, True], [True, False, True]):
                result = attend(scores, values, gates=gates, sink_logit=sink, mask=mask)
                p, phantom, mass, output = decimal_oracle(scores, values, gates, sink, mask)
                self.assertVectorClose(result.routing_weights, p)
                self.assertAlmostEqual(result.phantom_mass, phantom, places=14)
                self.assertAlmostEqual(result.retained_mass, mass, places=14)
                self.assertVectorClose(result.output, output)

    def test_phantom_equals_appended_zero_value(self):
        result = attend([0, 0], [[2, 0], [0, 2]], sink_logit=log(2))
        appended = attend([0, 0, log(2)], [[2, 0], [0, 2], [0, 0]])
        self.assertVectorClose(result.routing_weights, [0.25, 0.25])
        self.assertAlmostEqual(result.phantom_mass, 0.5)
        self.assertVectorClose(result.output, appended.output)
        self.assertVectorClose(result.output, [0.5, 0.5])

    def test_off_by_one_is_zero_phantom_logit(self):
        result = attend([0, 0], [[2, 0], [0, 2]], sink_logit=0)
        self.assertVectorClose(result.routing_weights, [1/3, 1/3])
        self.assertVectorClose(result.output, [2/3, 2/3])
        self.assertAlmostEqual(result.phantom_mass, 1/3)

    def test_sink_only_is_scalar_attenuation_of_ordinary_read(self):
        scores, values = [-1, 2, 0.3], [[1, 4], [2, 0], [-1, 3]]
        ordinary = attend(scores, values)
        result = attend(scores, values, sink_logit=1.1)
        self.assertVectorClose(result.output, [result.retained_mass*x for x in ordinary.output])
        self.assertVectorClose(result.renormalized_output, ordinary.output)
        near_absent = attend(scores, values, sink_logit=-1000)
        self.assertVectorClose(near_absent.output, ordinary.output)

    def test_retained_mass_decomposition_and_zero_case(self):
        for sink in (None, 0.7):
            for gates in ([0.2, 0.8], [1, 1], [0, 0]):
                result = attend([0.2, -1], [[3, -2], [1, 4]], sink_logit=sink, gates=gates)
                self.assertTrue(0 <= result.retained_mass <= 1)
                self.assertAlmostEqual(fsum(result.routing_weights)+result.phantom_mass, 1)
                if result.retained_mass:
                    self.assertAlmostEqual(fsum(result.renormalized_weights), 1)
                    self.assertVectorClose(result.output, [result.retained_mass*x for x in result.renormalized_output])
                else:
                    self.assertIsNone(result.renormalized_weights)
                    self.assertIsNone(result.renormalized_output)
                    self.assertEqual(result.output, (0, 0))

    def test_all_logit_shift_includes_phantom(self):
        args = dict(values=[[2, 1], [-3, 4]], gates=[0.7, 0.2])
        original = attend([-3, 2], sink_logit=0.4, **args)
        shifted = attend([1231, 1236], sink_logit=1234.4, **args)
        self.assertVectorClose(original.routing_weights, shifted.routing_weights)
        self.assertVectorClose(original.output, shifted.output)
        self.assertAlmostEqual(original.phantom_mass, shifted.phantom_mass, places=12)

    def test_shifting_only_real_logits_changes_mass_not_conditional_routing(self):
        low = attend([-8, -8], [[2, 0], [0, 2]], sink_logit=0)
        high = attend([8, 8], [[2, 0], [0, 2]], sink_logit=0)
        self.assertLess(low.retained_mass, 0.001)
        self.assertGreater(high.retained_mass, 0.999)
        self.assertVectorClose(low.renormalized_weights, high.renormalized_weights)

    def test_mask_excludes_future_key_before_normalizing(self):
        masked = attend([0, 1, 1000], [[2, 0], [0, 2], [1e300, -1e300]],
                        sink_logit=0.5, gates=[0.4, 0.8, 1], mask=[True, True, False])
        prefix = attend([0, 1], [[2, 0], [0, 2]], sink_logit=0.5, gates=[0.4, 0.8])
        self.assertEqual(masked.routing_weights[-1], 0)
        self.assertVectorClose(masked.output, prefix.output)
        self.assertAlmostEqual(masked.phantom_mass, prefix.phantom_mass)

    def test_all_masked_row_can_abstain_only_with_phantom(self):
        result = attend([1, 2], [[3], [4]], sink_logit=0, mask=[False, False])
        self.assertEqual(result.phantom_mass, 1)
        self.assertEqual(result.retained_mass, 0)
        self.assertEqual(result.output, (0,))
        self.assertIsNone(result.renormalized_output)
        with self.assertRaises(ValueError):
            attend([1, 2], [[3], [4]], mask=[False, False])

    def test_uniform_gates_lose_attenuation_when_renormalized(self):
        result = attend([0, 0], [[2, 0], [0, 2]], gates=[0.1, 0.1])
        self.assertVectorClose(result.output, [0.1, 0.1])
        self.assertVectorClose(result.renormalized_output, [1, 1])
        self.assertAlmostEqual(result.retained_mass, 0.1)

    def test_per_read_gate_differs_from_post_aggregation_scalar(self):
        report = build_report()
        self.assertVectorClose(report["value_gates_only"]["output"], [0.9, 0.1])
        self.assertVectorClose(report["post_aggregation_scalar_at_same_retained_mass"], [0.5, 0.5])
        self.assertVectorClose(report["phantom_plus_value_gates"]["output"], [0.45, 0.05])

    def test_projection_gate_uses_each_values_content(self):
        values, weight = [[2, 0], [0, 2]], [log(9)/2, -log(9)/2]
        self.assertVectorClose(projection_gates(values, weight), [0.9, 0.1])
        self.assertVectorClose(projection_gates(values[::-1], weight), [0.1, 0.9])
        self.assertVectorClose(projection_gates(values, [0, 0]), [0.5, 0.5])

    def test_sink_logit_derivative(self):
        step, beta = 1e-5, 0.7
        args = dict(logits=[0.3, -1.2], values=[[2, -1], [1, 4]], gates=[0.3, 0.9])
        central = attend(sink_logit=beta, **args)
        upper, lower = attend(sink_logit=beta+step, **args), attend(sink_logit=beta-step, **args)
        derivative = [(a-b)/(2*step) for a, b in zip(upper.output, lower.output)]
        self.assertVectorClose(derivative, [-central.phantom_mass*y for y in central.output], places=9)

    def test_per_read_gate_derivative(self):
        step = 1e-5
        args = dict(logits=[0.3, -1.2], values=[[2, -1], [1, 4]], sink_logit=0.7)
        middle = attend(gates=[0.3, 0.9], **args)
        upper, lower = attend(gates=[0.3+step, 0.9], **args), attend(gates=[0.3-step, 0.9], **args)
        derivative = [(a-b)/(2*step) for a, b in zip(upper.output, lower.output)]
        self.assertVectorClose(derivative, [middle.routing_weights[0]*v for v in [2, -1]], places=9)

    def test_real_key_permutation(self):
        first = attend([0, 1, 2], [[1, 3], [-2, 0], [4, 5]], gates=[0.8, 0.2, 0.6], sink_logit=0.4)
        second = attend([2, 0, 1], [[4, 5], [1, 3], [-2, 0]], gates=[0.6, 0.8, 0.2], sink_logit=0.4)
        self.assertVectorClose(first.output, second.output)
        self.assertAlmostEqual(first.retained_mass, second.retained_mass)

    def test_linearity_in_values_when_gates_are_held_fixed(self):
        args = dict(logits=[0, 1], gates=[0.2, 0.8], sink_logit=0.7)
        a, b = [[1, 2], [3, 4]], [[-2, 1], [1, 3]]
        summed = [[x+y for x, y in zip(r, s)] for r, s in zip(a, b)]
        result = attend(values=summed, **args)
        self.assertVectorClose(result.output, [x+y for x, y in zip(attend(values=a, **args).output,
                                                                 attend(values=b, **args).output)])

    def test_extreme_logits_and_representable_mass(self):
        equal = attend([1000, 1000], [[2, 0], [0, 2]], sink_logit=1000)
        self.assertVectorClose(equal.output, [2/3, 2/3])
        sink = attend([-1000, -1000], [[2, 0], [0, 2]], sink_logit=1000)
        self.assertEqual(sink.phantom_mass, 1)
        self.assertEqual(sink.retained_mass, 0)  # Underflow in binary64, not exact real arithmetic.
        self.assertIsNone(sink.renormalized_output)
        dominant = attend([1000, -1000], [[2, 1], [0, 3]], sink_logit=0)
        self.assertEqual(dominant.output, (2, 1))

    def test_zero_output_does_not_imply_abstention(self):
        result = attend([0, 0], [[1, 0], [-1, 0]])
        self.assertEqual(result.output, (0, 0))
        self.assertEqual(result.retained_mass, 1)
        self.assertEqual(result.phantom_mass, 0)

    def test_invalid_inputs(self):
        cases = [dict(logits=[], values=[]), dict(logits=[0, 1], values=[[1]]),
                 dict(logits=[float("inf")], values=[[1]]), dict(logits=[True], values=[[1]]),
                 dict(logits=[0], values=[[float("nan")]]),
                 dict(logits=[0], values=[[1]], sink_logit=float("nan")),
                 dict(logits=[0], values=[[1]], gates=[-1]), dict(logits=[0], values=[[1]], gates=[1.1]),
                 dict(logits=[0], values=[[1]], gates=[]), dict(logits=[0], values=[[1]], gates=[True]),
                 dict(logits=[0], values=[[1]], mask=[1]), dict(logits=[0], values=[[1]], mask=[])]
        for arguments in cases:
            with self.subTest(arguments=arguments), self.assertRaises((ValueError, TypeError)):
                attend(**arguments)
        with self.assertRaises(ValueError):
            projection_gates([[1, 2]], [1])

    def test_inputs_are_not_mutated(self):
        args = dict(logits=[0, 1], values=[[1, 2], [3, 4]], gates=[0.3, 0.7], mask=[True, False])
        original = copy.deepcopy(args)
        attend(**args, sink_logit=0)
        projection_gates(args["values"], [1, -1])
        self.assertEqual(args, original)

    def test_checked_output(self):
        expected = json.loads(Path(__file__).with_name("sample_output.json").read_text())
        self.assertEqual(build_report(), expected)

    def test_cli_and_invalid_argument(self):
        command = [sys.executable, "-m", "examples.attention_abstention.demo"]
        output = subprocess.run(command, capture_output=True, text=True, check=True, timeout=10)
        self.assertEqual(json.loads(output.stdout), build_report())
        changed = subprocess.run(command+["--real-shift", "-5"], capture_output=True, text=True, check=True, timeout=10)
        self.assertLess(json.loads(changed.stdout)["phantom_only"]["retained_mass"], 0.01)
        invalid = subprocess.run(command+["--sink-logit", "nan"], capture_output=True, text=True, timeout=10)
        self.assertNotEqual(invalid.returncode, 0)


if __name__ == "__main__":
    unittest.main()
