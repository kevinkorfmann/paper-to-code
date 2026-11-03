"""Independent numerical oracles, analytic identities, failure and CLI controls."""

from decimal import Decimal, localcontext
import json
import math
from pathlib import Path
import subprocess
import sys
import unittest

from examples.dami_calibration.demo import SAMPLE, build_report, render
from examples.dami_calibration.model import (
    Gate, calibrate, diagnose, geometric_timescales, retention_path, sigmoid, softplus,
)


SAMPLE_Z = [-2.0, -0.5, 0.5, 2.0]


def decimal_oracle(activations, tau, sensitivity):
    """80-digit Newton solve, not production bisection or production primitives."""
    with localcontext() as ctx:
        ctx.prec = 80
        z = [Decimal(str(x)) for x in activations]
        target = Decimal(str(sensitivity))
        bias = (target / (1 - target)).ln() - sum(z) / len(z)
        for _ in range(150):
            probabilities = [1 / (1 + (-x - bias).exp()) for x in z]
            residual = sum(probabilities) / len(z) - target
            if abs(residual) < Decimal('1e-65'):
                break
            slope = sum(p * (1 - p) for p in probabilities) / len(z)
            # Damping prevents Newton steps crossing far into saturated tails.
            step = min(Decimal(2), max(Decimal(-2), residual / slope))
            bias -= step
        else:
            raise AssertionError('Decimal Newton oracle did not converge')
        mu = sum((1 + (x + bias).exp()).ln() for x in z) / len(z)
        a_log = -Decimal(str(tau)).ln() - mu.ln()
        logs = [-a_log.exp() * (1 + (x + bias).exp()).ln() for x in z]
        return float(bias), float(a_log), [float(x) for x in logs]


class CalibrationTests(unittest.TestCase):
    def test_decimal_newton_oracle(self):
        for z in (SAMPLE_Z, [-5, -1, 0, 0.2, 3], [0.2, 1.3, 2.2]):
            for s in (0.1, 0.25, 0.5, 0.9):
                for tau in (1, 16, 1024):
                    with self.subTest(z=z, s=s, tau=tau):
                        expected_b, expected_a, expected_logs = decimal_oracle(z, tau, s)
                        actual = calibrate(z, tau, s)
                        self.assertAlmostEqual(actual.bias, expected_b, delta=2e-13)
                        self.assertAlmostEqual(actual.a_log, expected_a, delta=2e-13)
                        for value, expected in zip(z, expected_logs):
                            self.assertAlmostEqual(actual.log_retention(value), expected, delta=2e-13)

    def test_constant_input_closed_form(self):
        for c in (-100, -2, 0, 2, 100):
            for s in (1e-6, 0.1, 0.5, 0.9, 1 - 1e-6):
                gate = calibrate([c] * 7, 16, s)
                self.assertAlmostEqual(gate.bias, math.log(s / (1-s)) - c, delta=3e-14)
                self.assertAlmostEqual(gate.a_log, -math.log(16) - math.log(-math.log1p(-s)), delta=3e-14)
                self.assertAlmostEqual(gate.log_retention(c), -1/16, delta=3e-15)

    def test_target_identities_across_grid(self):
        for s in (0.05, 0.25, 0.5, 0.75, 0.95):
            for tau in (1e-6, 1, 16, 4096, 1e12):
                d = diagnose(calibrate(SAMPLE_Z, tau, s), SAMPLE_Z)
                self.assertAlmostEqual(d['mean_softplus_sensitivity'], s, delta=2e-14)
                self.assertTrue(math.isclose(d['mean_log_retention'], -1/tau, rel_tol=1e-13))

    def test_activation_translation(self):
        for shift in (-10, -1.5, 3, 50):
            a = calibrate(SAMPLE_Z, 16, 0.25)
            b = calibrate([z+shift for z in SAMPLE_Z], 16, 0.25)
            self.assertAlmostEqual(b.bias, a.bias-shift, places=13)
            self.assertAlmostEqual(b.a_log, a.a_log, places=13)
            for z in SAMPLE_Z:
                self.assertAlmostEqual(a.log_retention(z), b.log_retention(z+shift), places=13)

    def test_sample_permutation(self):
        self.assertEqual(calibrate(SAMPLE_Z, 16, .25), calibrate(SAMPLE_Z[::-1], 16, .25))

    def test_sample_replication(self):
        a, b = calibrate(SAMPLE_Z, 16, .25), calibrate(SAMPLE_Z * 5, 16, .25)
        self.assertAlmostEqual(a.bias, b.bias, places=14)
        self.assertAlmostEqual(a.a_log, b.a_log, places=14)

    def test_symmetric_samples(self):
        gate = calibrate([-5, -2, 0, 2, 5], 10, .5)
        self.assertEqual(gate.bias, 0)

    def test_timescale_rescaling(self):
        a, b = calibrate(SAMPLE_Z, 4, .25), calibrate(SAMPLE_Z, 64, .25)
        self.assertEqual(a.bias, b.bias)
        self.assertAlmostEqual(b.a_log-a.a_log, -math.log(16), places=14)
        for z in SAMPLE_Z:
            self.assertAlmostEqual(b.log_retention(z), a.log_retention(z)/16, places=14)

    def test_bias_monotonicity(self):
        biases = [calibrate(SAMPLE_Z, 16, s).bias for s in (.1,.25,.5,.75,.9)]
        self.assertTrue(all(x < y for x,y in zip(biases, biases[1:])))

    def test_bias_only_does_not_preserve_timescale(self):
        a, b = calibrate(SAMPLE_Z, 16, .25), calibrate(SAMPLE_Z, 16, .75)
        wrong = diagnose(Gate(b.bias, a.a_log), SAMPLE_Z)
        self.assertAlmostEqual(wrong['mean_softplus_sensitivity'], .75, places=14)
        self.assertLess(wrong['mean_log_retention'], -5/16)

    def test_exact_arithmetic_geometric_counterexample(self):
        z = [-math.log(3), math.log(3)]
        tau = 2/math.log(16/3)
        gate = calibrate(z, tau, .5)
        self.assertAlmostEqual(gate.bias, 0, places=14)
        self.assertAlmostEqual(gate.a_log, 0, places=14)
        self.assertAlmostEqual(gate.retention(z[0]), .75, places=14)
        self.assertAlmostEqual(gate.retention(z[1]), .25, places=14)
        d = diagnose(gate, z)
        self.assertAlmostEqual(d['arithmetic_mean_retention'], .5, places=14)
        self.assertAlmostEqual(d['geometric_mean_retention'], math.sqrt(3)/4, places=14)

    def test_jensen_strict_inequality(self):
        for s in (.1,.25,.75,.9):
            d = diagnose(calibrate(SAMPLE_Z, 4, s), SAMPLE_Z)
            self.assertGreater(d['arithmetic_mean_retention'], d['geometric_mean_retention'])

    def test_full_gate_sensitivity_can_reverse(self):
        low = diagnose(calibrate([0], 1, .1), [0])
        high = diagnose(calibrate([0], 1, .9), [0])
        self.assertLess(low['mean_softplus_sensitivity'], high['mean_softplus_sensitivity'])
        self.assertGreater(low['mean_log_decay_slope_magnitude'], high['mean_log_decay_slope_magnitude'])

    def test_softplus_derivative(self):
        for z in (-10, -2, 0, 2, 10):
            h = 1e-5
            numerical = (softplus(z+h)-softplus(z-h))/(2*h)
            self.assertAlmostEqual(numerical, sigmoid(z), delta=1e-10)

    def test_fixed_parameter_gate_derivative(self):
        gate = calibrate(SAMPLE_Z, 16, .25)
        h = 1e-5
        slopes = [-(gate.log_retention(z+h)-gate.log_retention(z-h))/(2*h) for z in SAMPLE_Z]
        self.assertAlmostEqual(sum(slopes)/len(slopes), diagnose(gate,SAMPLE_Z)['mean_log_decay_slope_magnitude'], delta=1e-11)

    def test_retention_derivative_includes_retention_factor(self):
        gate = calibrate(SAMPLE_Z, 16, .25)
        for z in SAMPLE_Z:
            h = 1e-5
            numerical = -(gate.retention(z+h)-gate.retention(z-h))/(2*h)
            expected = gate.retention(z)*math.exp(gate.a_log)*sigmoid(z+gate.bias)
            self.assertAlmostEqual(numerical, expected, delta=1e-11)

    def test_repeated_calibration_cycle(self):
        gate = calibrate(SAMPLE_Z, 16, .25)
        for repeats in (1,2,4,10):
            path = retention_path(gate, SAMPLE_Z*repeats)
            self.assertAlmostEqual(path['retention'][-1], math.exp(-len(SAMPLE_Z)*repeats/16), places=14)

    def test_prefix_causality(self):
        gate = calibrate(SAMPLE_Z, 16, .25)
        short = retention_path(gate, SAMPLE_Z)
        long = retention_path(gate, SAMPLE_Z + [-1000, 1000])
        for key in short:
            self.assertEqual(short[key], long[key][:len(short[key])])

    def test_path_matches_independent_products(self):
        gate = calibrate(SAMPLE_Z, 16, .25)
        p = retention_path(gate, SAMPLE_Z)
        for i in range(len(SAMPLE_Z)+1):
            direct = math.prod((1+math.exp(z+gate.bias))**(-math.exp(gate.a_log)) for z in SAMPLE_Z[:i])
            self.assertAlmostEqual(p['retention'][i], direct, places=14)

    def test_path_monotone_and_empty(self):
        gate = calibrate(SAMPLE_Z, 16, .25)
        self.assertEqual(retention_path(gate, []), {'log_retention': (0.0,), 'retention': (1.0,)})
        p = retention_path(gate, SAMPLE_Z)['retention']
        self.assertTrue(all(0 < b <= a <= 1 for a,b in zip(p,p[1:])))

    def test_distribution_shift_changes_envelope(self):
        report = build_report()['decay_envelope']
        self.assertAlmostEqual(report['calibration_sample_repeated'], math.exp(-1), places=14)
        self.assertLess(report['activations_shifted_by_two'], .04)

    def test_delta_write_erases_despite_high_retention(self):
        d = build_report()['write_interference_counterexample']
        self.assertGreater(d['explicit_retention'], .99)
        self.assertEqual(d['state_after_delta_write'], 0)

    def test_geometric_timescales(self):
        t = geometric_timescales(5, 1, 4096)
        self.assertEqual((t[0], t[-1]), (4096,1))
        for a,b in zip(t,t[1:]):
            self.assertAlmostEqual(a/b, 8, places=13)
        self.assertEqual(geometric_timescales(3,16,16), (16,16,16))

    def test_stable_primitive_extremes(self):
        self.assertEqual(softplus(1000), 1000)
        self.assertEqual(softplus(-1000), 0)
        self.assertEqual(sigmoid(1000), 1)
        self.assertEqual(sigmoid(-1000), 0)
        self.assertAlmostEqual(softplus(0), math.log(2), places=15)

    def test_log_path_survives_retention_underflow(self):
        p = retention_path(Gate(0, 0), [1000, 1000])
        self.assertEqual(p['retention'][-1], 0)
        self.assertEqual(p['log_retention'][-1], -2000)

    def test_extreme_supported_targets(self):
        for z in ([-100], [100], [-100,100]):
            for s in (1e-6, .25, .75, 1-1e-6):
                d = diagnose(calibrate(z, 1e12, s), z)
                self.assertAlmostEqual(d['mean_softplus_sensitivity'], s, delta=2e-13)
                self.assertTrue(math.isclose(d['mean_log_retention'], -1e-12, rel_tol=1e-13))

    def test_large_multiplier_tiny_softplus_composition(self):
        actual = Gate(0,700).log_retention(-750)
        self.assertNotEqual(actual, 0)
        self.assertTrue(math.isclose(actual, -math.exp(-50), rel_tol=1e-14))

    def test_accepted_near_flat_root_matches_behavior_not_bias(self):
        z, s = [-23,23], .5000000001
        gate = calibrate(z,16,s)
        d = diagnose(gate,z)
        self.assertAlmostEqual(d["mean_softplus_sensitivity"],s,delta=2e-13)
        self.assertAlmostEqual(d["mean_log_retention"],-1/16,delta=2e-14)

    def test_reject_flat_numerically_unidentified_root(self):
        with self.assertRaisesRegex(ValueError, 'too flat'):
            calibrate([-100,100], 16, .5)

    def test_reject_invalid_samples(self):
        for z in ([], None, [math.inf], [math.nan], [True], ['1'], [101], [10**500]):
            with self.subTest(z=z), self.assertRaises(ValueError):
                calibrate(z,16,.25)

    def test_reject_invalid_timescales(self):
        for tau in (0, -1, 1e-7, 1e13, math.inf, math.nan, True, '16'):
            with self.subTest(tau=tau), self.assertRaises(ValueError):
                calibrate(SAMPLE_Z,tau,.25)

    def test_reject_invalid_sensitivities(self):
        for s in (0,1,-1,1e-7,1-1e-7,math.inf,math.nan,True,'0.5'):
            with self.subTest(s=s), self.assertRaises(ValueError):
                calibrate(SAMPLE_Z,16,s)

    def test_reject_invalid_gate_and_primitives(self):
        for value in (math.inf, math.nan, True, '1'):
            for fn in (softplus, sigmoid, lambda x: Gate(x,0), lambda x: Gate(0,x)):
                with self.subTest(fn=fn, value=value), self.assertRaises(ValueError):
                    fn(value)
        for a in (-701,701):
            with self.assertRaises(ValueError):
                Gate(0,a)
        with self.assertRaisesRegex(ValueError,'overflow'):
            Gate(0,700).log_retention(1e300)
        with self.assertRaises(ValueError):
            diagnose(Gate(0,0),[])

    def test_reject_invalid_geometric_grid(self):
        for args in ((1,1,16),(True,1,16),(2.5,1,16),(3,0,16),(3,16,1),(3,1,math.inf)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                geometric_timescales(*args)

    def test_input_not_mutated_and_gate_immutable(self):
        z = SAMPLE_Z.copy()
        gate = calibrate(z,16,.25)
        self.assertEqual(z,SAMPLE_Z)
        with self.assertRaises(AttributeError):
            gate.bias = 0

    def test_sample_matches_report(self):
        self.assertEqual(render(build_report()), SAMPLE.read_text())
        self.assertEqual(json.loads(SAMPLE.read_text())['python_version'], '3.14.5')

    def test_cli_reproduces_sample(self):
        result = subprocess.run([sys.executable, '-m', 'examples.dami_calibration.demo', '--check-sample'], check=True, capture_output=True, text=True)
        self.assertEqual(result.stdout,SAMPLE.read_text())
        self.assertEqual(result.stderr,'')

    def test_cli_rejects_unknown_argument(self):
        result = subprocess.run([sys.executable, '-m', 'examples.dami_calibration.demo', '--unknown'], capture_output=True,text=True)
        self.assertNotEqual(result.returncode,0)
        self.assertIn('unrecognized arguments',result.stderr)


if __name__ == '__main__':
    unittest.main()
