import math
import unittest

from .demo import causal_bias, fixture_step, normalized_weights, step_log_survival


class VetoTests(unittest.TestCase):
    def test_single_zero_veto(self):
        self.assertEqual(step_log_survival((0.9, 0.0, 0.8)), 0)
        self.assertLess(step_log_survival((0.9, 0.0, 0.8), geometric=False), 0)

    def test_equal_gates(self):
        for p in (0.001, 0.1, 0.5, 0.99, 1 - 1e-12):
            self.assertAlmostEqual(step_log_survival((p,) * 3), math.log1p(-p))

    def test_boundaries(self):
        self.assertEqual(step_log_survival((1, 1, 1)), -math.inf)
        self.assertEqual(causal_bias(fixture_step, 1, 2), -math.inf)
        self.assertEqual(causal_bias(fixture_step, 1, 1), 0)

    def test_accumulates_after_key_only(self):
        self.assertAlmostEqual(math.exp(causal_bias(fixture_step, 3, 1)), 0.25)

    def test_readout(self):
        for g, old in ((True, 1), (False, (2 / 3) ** 3)):
            b = [causal_bias(fixture_step, 3, j, geometric=g) for j in range(4)]
            self.assertAlmostEqual(normalized_weights(b)[0], old / (old + 1.75))

    def test_rejects_invalid(self):
        for gates in ((), (float("nan"),), (-1,), (2,)):
            with self.assertRaises(ValueError):
                step_log_survival(gates)


if __name__ == "__main__":
    unittest.main()
