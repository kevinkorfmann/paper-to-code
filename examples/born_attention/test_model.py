"""Independent rational oracle and invariants for the two small kernels."""
import copy
from fractions import Fraction as F
import math
import subprocess
import sys
import unittest
from examples.born_attention import demo, model


def oracle(query, keys, values):
    # Real/imaginary Fraction arithmetic, without complex multiplication or
    # production helpers. Fixtures use integers exactly representable as floats.
    scores = []
    for key in keys:
        real = sum(F(q.real) * F(k.real) + F(q.imag) * F(k.imag)
                   for q, k in zip(query, key))
        imag = sum(F(q.real) * F(k.imag) - F(q.imag) * F(k.real)
                   for q, k in zip(query, key))
        scores.append((real * real + imag * imag) ** 2)
    weights = [x / sum(scores) for x in scores]
    mixed = [complex(float(sum(w * F(v[j].real) for w, v in zip(weights, values))),
                     float(sum(w * F(v[j].imag) for w, v in zip(weights, values))))
             for j in range(len(values[0]))]
    return [float(w) for w in weights], mixed


class BornAttentionTests(unittest.TestCase):
    def assertClose(self, actual, expected):
        self.assertEqual(len(actual), len(expected))
        for a, b in zip(actual, expected):
            self.assertLessEqual(abs(a - b), 2e-13 * max(1, abs(b)))

    def test_rational_oracle(self):
        keys = [[1+2j, 3-1j], [-2+1j, 1+4j], [3-3j, -2j]]
        values = [[1+2j, -1], [2, 3j], [-3j, 4-2j]]
        for query in ([2-1j, -1+3j], [1j, 2], [-3, 4j]):
            w, y = model.attention([query], keys, values)
            ref_w, ref_y = oracle(query, keys, values)
            self.assertClose(w[0], ref_w)
            self.assertClose(y[0], ref_y)
            self.assertAlmostEqual(sum(w[0]), 1)
            self.assertTrue(all(x >= 0 for x in w[0]))

    def test_conjugation_matters(self):
        w, _ = model.attention([[1, 1j]], [[1, 1j], [1, -1j]], [[1], [0]])
        self.assertEqual(w, [[1, 0]])

    def test_phase_invariance_with_fixed_values(self):
        q = [[1+1j, 2-1j]]
        k = [[2, 1j], [-1j, 3]]
        v = [[1], [2j]]
        w, y = model.attention(q, k, v)
        wp, yp = model.attention([[x * 1j for x in q[0]]],
                                 [[x * -1j for x in k[0]], [-x for x in k[1]]], v)
        self.assertClose(w[0], wp[0])
        self.assertClose(y[0], yp[0])

    def test_common_scaling_and_extreme_fourth_powers(self):
        base, _ = model.attention([[1]], [[1], [2]], [[1], [0]])
        for scale in (1e-100, 1e100):
            result, _ = model.attention([[1]], [[scale], [2*scale]], [[1], [0]])
            self.assertClose(result[0], base[0])
        scaled, _ = model.attention([[3j]], [[-2], [-4]], [[1], [0]])
        self.assertClose(scaled[0], base[0])

    def test_norm_sensitivity_counterexample(self):
        scaled, _ = model.attention([[1, 1j]], [[1, 0], [0, 2j]], [[1], [0]])
        unit, _ = model.attention([[1, 1j]], [[1, 0], [0, 1j]], [[1], [0]])
        self.assertClose(scaled[0], [1/17, 16/17])
        self.assertEqual(unit[0], [0.5, 0.5])

    def test_permutation_and_no_mutation(self):
        q = [[1, 2], [2, 1j]]
        k = [[1, 1j], [2, -1]]
        v = [[1, 3j], [-2, 1]]
        original = copy.deepcopy((q, k, v))
        w, y = model.attention(q, k, v)
        wp, yp = model.attention(q[::-1], k[::-1], v[::-1])
        for i in range(2):
            self.assertClose(w[i], wp[1-i][::-1])
            self.assertClose(y[i], yp[1-i])
        self.assertEqual((q, k, v), original)

    def test_readout_rational_oracle_and_baseline(self):
        p, b = model.born_readout([1+2j, 3j, 2-1j], [[0, 2], [1]])
        self.assertClose(p, [10/19, 9/19])
        self.assertClose(b, [2/3, 1/3])
        uniform, _ = model.born_readout([1, 1j, -1], [[0, 2], [1]])
        self.assertClose(uniform, b)
        for scale in (1e-200, 1e200, 2j):
            pp, _ = model.born_readout([scale*(1+2j), scale*3j, scale*(2-1j)], [[0, 2], [1]])
            self.assertClose(pp, p)

    def test_coherent_versus_incoherent_pooling(self):
        slots = [[1, 1], [-1, 1]]
        phi = [sum(slot[j] for slot in slots) for j in range(2)]
        coherent, _ = model.born_readout(phi, [[0], [1]])
        self.assertEqual(coherent, [0, 1])
        intensities = [sum(abs(slot[j])**2 for slot in slots) for j in range(2)]
        self.assertEqual([x / sum(intensities) for x in intensities], [0.5, 0.5])
        global_phase, _ = model.born_readout([1j*x for x in phi], [[0], [1]])
        self.assertEqual(global_phase, coherent)
        # Reverse only the second slot phase; the per-slot intensities stay fixed.
        relative_phi = [slots[0][j] - slots[1][j] for j in range(2)]
        relative_phase, _ = model.born_readout(relative_phi, [[0], [1]])
        self.assertEqual(relative_phase, [1, 0])

    def test_readout_phase_cancellation_before_detection(self):
        # Detection follows coherent slot summation, not per-slot intensities.
        with self.assertRaises(ValueError):
            model.born_readout([1 + (-1), 1j + (-1j)], [[0], [1]])

    def test_invalid_attention(self):
        invalid = [([], [[1]], [[1]]), ([[1]], [], [[1]]),
                   ([[1, 2]], [[1]], [[1]]), ([[1]], [[1], [2]], [[1]]),
                   ([[1], [1, 2]], [[1]], [[1]]), ([[0]], [[1]], [[1]]),
                   ([[1, 0]], [[0, 1]], [[1]]), ([[True]], [[1]], [[1]]),
                   ([["1"]], [[1]], [[1]]), ([[math.nan]], [[1]], [[1]]),
                   ([[1]], [[complex(0, math.inf)]], [[1]]),
                   ([[1]], [[1]], [[]]), ([[1e308]], [[1e308]], [[1]])]
        for args in invalid:
            with self.subTest(args=args), self.assertRaises(ValueError):
                model.attention(*args)

    def test_invalid_readout(self):
        cases = [([], [[0]]), ([0, 0], [[0], [1]]), ([1, 2], [[0], [0]]),
                 ([1, 2], [[0]]), ([1, 2], [[0], [2]]), ([1], [[]]),
                 ([1], [[True]]), ([1], [[0.0]]), ([math.inf], [[0]]), ([complex(1.7e308, 1.7e308)], [[0]])]
        for args in cases:
            with self.subTest(args=args), self.assertRaises(ValueError):
                model.born_readout(*args)

    def test_demo_and_sample(self):
        report = demo.build_report()
        self.assertClose(report["born_readout_probabilities"], [1/257, 256/257])
        self.assertEqual(demo.render(report), demo.SAMPLE.read_text())
        subprocess.run([sys.executable, "-m", "examples.born_attention.demo", "--check-sample"],
                       check=True, stdout=subprocess.PIPE, text=True)


if __name__ == "__main__":
    unittest.main()
