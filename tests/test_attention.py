from decimal import Decimal, localcontext
from math import isfinite, log, sqrt
from random import Random
import unittest

from examples.differential_attention.model import (
    differential_attention, lambda_from_vectors, lambda_initial, softmax,
)


def fixture():
    return (
        [[0.2, -0.5], [1.2, 0.1], [-0.3, 0.7]],
        [[-0.4, 0.1], [0.5, 0.6], [0.8, -0.2]],
        [[0.6, 0.3], [-0.2, 0.7], [0.5, -0.4]],
        [[0.2, -0.8], [0.4, 0.1], [-0.1, 0.9]],
        [[2, -1, 0.2, 0.7], [0.5, 3, -0.4, 0.1], [-1, 0, 1, 2]],
    )


def decimal_reference(q1, q2, k1, k2, values, coefficient, causal):
    """Independent 60-digit scalar-loop oracle; no production helpers."""
    with localcontext() as context:
        context.prec = 60
        d = Decimal(len(q1[0])).sqrt()
        c = Decimal.from_float(coefficient)
        result = []
        for i in range(len(q1)):
            probabilities = []
            for queries, keys in ((q1, k1), (q2, k2)):
                exps = []
                for j in range(len(keys)):
                    if causal and j > i:
                        exps.append(Decimal(0))
                    else:
                        score = sum(
                            Decimal.from_float(float(x)) * Decimal.from_float(float(y))
                            for x, y in zip(queries[i], keys[j])
                        ) / d
                        exps.append(score.exp())
                probabilities.append([x / sum(exps) for x in exps])
            output = []
            for channel in range(len(values[0])):
                output.append(float(sum(
                    (probabilities[0][j] - c * probabilities[1][j])
                    * Decimal.from_float(float(values[j][channel]))
                    for j in range(len(values))
                )))
            result.append(output)
        return result


class SoftmaxTests(unittest.TestCase):
    def test_known_probabilities(self):
        for actual, expected in zip(softmax([0, log(3)]), (0.25, 0.75)):
            self.assertAlmostEqual(actual, expected, places=14)

    def test_stability_shift_invariance_and_mask(self):
        baseline = softmax([0, -2, -4])
        shifted = softmax([1000, 998, 996])
        self.assertEqual(baseline, shifted)
        self.assertAlmostEqual(sum(shifted), 1, places=14)
        self.assertEqual(softmax([float("-inf"), 1000, -1000]), (0, 1, 0))
        self.assertTrue(all(isfinite(x) for x in softmax([-1e308, 1e308])))

    def test_invalid_rows(self):
        for values in ([], [float("-inf")], [float("nan")], [float("inf")]):
            with self.assertRaises(ValueError):
                softmax(values)
        for values in ([True], ["1"]):
            with self.assertRaises(TypeError):
                softmax(values)


class AttentionTests(unittest.TestCase):
    def assertMatrixAlmostEqual(self, actual, expected, places=13):
        self.assertEqual(len(actual), len(expected))
        for row, target in zip(actual, expected):
            self.assertEqual(len(row), len(target))
            for x, y in zip(row, target):
                self.assertAlmostEqual(x, y, places=places)

    def test_matches_independent_high_precision_oracle(self):
        for causal in (False, True):
            for coefficient in (-0.2, 0.0, 0.37, 1.0, 1.4):
                data = fixture()
                result = differential_attention(*data, coefficient, causal=causal)
                expected = decimal_reference(*data, coefficient, causal)
                self.assertMatrixAlmostEqual(result.output, expected)

    def test_scale_is_sqrt_d_not_sqrt_2d(self):
        q = [[sqrt(2), 0]] * 2
        k = [[0, 0], [log(3), 0]]
        values = [[1, 0, 0, 0], [0, 1, 0, 0]]
        result = differential_attention(q, q, k, k, values, 0.0)
        self.assertMatrixAlmostEqual(result.output, [[0.25, 0.75, 0, 0]] * 2)

    def test_lambda_zero_is_standard_attention(self):
        q = [[0]] * 3
        k = [[1], [2], [3]]
        values = [[1, 2], [2, 4], [6, 9]]
        result = differential_attention(q, q, k, k, values, 0.0)
        self.assertMatrixAlmostEqual(result.output, [[3, 5]] * 3)
        self.assertEqual(result.weights, result.map1)

    def test_each_map_normalizes_but_difference_sums_to_one_minus_lambda(self):
        for causal in (False, True):
            for coefficient in (-2, 0, 0.3, 1, 2):
                result = differential_attention(*fixture(), coefficient, causal=causal)
                for a, b, weights in zip(result.map1, result.map2, result.weights):
                    self.assertAlmostEqual(sum(a), 1, places=14)
                    self.assertAlmostEqual(sum(b), 1, places=14)
                    self.assertAlmostEqual(sum(weights), 1 - coefficient, places=14)

    def test_identical_maps_cancel_all_values_at_lambda_one(self):
        q1, _, k1, _, values = fixture()
        result = differential_attention(q1, q1, k1, k1, values, 1.0)
        self.assertEqual(result.output, ((0.0,) * 4,) * 3)

    def test_shared_value_shift_has_known_response(self):
        q1, q2, k1, k2, values = fixture()
        shift = [10, -3, 2, 7]
        shifted = [[x + y for x, y in zip(row, shift)] for row in values]
        for causal in (False, True):
            for coefficient in (0.0, 0.5, 1.0):
                before = differential_attention(q1, q2, k1, k2, values, coefficient, causal=causal)
                after = differential_attention(q1, q2, k1, k2, shifted, coefficient, causal=causal)
                expected = [[x + (1 - coefficient) * y for x, y in zip(row, shift)]
                            for row in before.output]
                self.assertMatrixAlmostEqual(after.output, expected)

    def test_causal_mask_has_no_future_leakage_in_either_branch(self):
        data = fixture()
        before = differential_attention(*data, 0.6, causal=True)
        changed = fixture()
        for matrix in changed:
            matrix[-1] = [100 + i for i in range(len(matrix[-1]))]
        after = differential_attention(*changed, 0.6, causal=True)
        self.assertEqual(before.output[:2], after.output[:2])
        self.assertNotEqual(before.output[-1], after.output[-1])
        for matrix in (before.map1, before.map2, before.weights):
            for i, row in enumerate(matrix):
                self.assertTrue(all(x == 0 for x in row[i + 1:]))
        self.assertAlmostEqual(before.map1[0][0], 1)
        self.assertAlmostEqual(before.map2[0][0], 1)

    def test_noncausal_token_permutation_equivariance(self):
        data = fixture()
        permutation = [2, 0, 1]
        result = differential_attention(*data, 0.6)
        permuted_data = [[matrix[i] for i in permutation] for matrix in data]
        permuted = differential_attention(*permuted_data, 0.6)
        self.assertMatrixAlmostEqual(permuted.output, [result.output[i] for i in permutation])

    def test_linear_in_values(self):
        q1, q2, k1, k2, values = fixture()
        rng = Random(2410)
        other = [[rng.uniform(-2, 2) for _ in row] for row in values]
        combined = [[2 * x - 3 * y for x, y in zip(row, second)]
                    for row, second in zip(values, other)]
        a = differential_attention(q1, q2, k1, k2, values, 0.6)
        b = differential_attention(q1, q2, k1, k2, other, 0.6)
        c = differential_attention(q1, q2, k1, k2, combined, 0.6)
        self.assertMatrixAlmostEqual(c.output, [[2 * x - 3 * y for x, y in zip(row, second)]
                                             for row, second in zip(a.output, b.output)])

    def test_lambda_derivative_matches_finite_difference(self):
        data = fixture()
        step = 1e-5
        plus = differential_attention(*data, 0.4 + step)
        minus = differential_attention(*data, 0.4 - step)
        center = differential_attention(*data, 0.4)
        expected = [
            [-sum(weight * row[channel] for weight, row in zip(weights, data[-1]))
             for channel in range(4)]
            for weights in center.map2
        ]
        numerical = [[(a - b) / (2 * step) for a, b in zip(row, second)]
                     for row, second in zip(plus.output, minus.output)]
        self.assertMatrixAlmostEqual(numerical, expected, places=9)

    def test_inputs_are_not_mutated(self):
        data = fixture()
        differential_attention(*data, 0.7, causal=True)
        self.assertEqual(data, fixture())


class LambdaTests(unittest.TestCase):
    def test_initial_schedule_and_zero_vectors(self):
        self.assertAlmostEqual(lambda_initial(1), 0.2)
        self.assertGreater(lambda_initial(2), lambda_initial(1))
        self.assertAlmostEqual(lambda_initial(100), 0.8)
        self.assertEqual(lambda_from_vectors([0, 0], [0, 0], [0, 0], [0, 0], 2), lambda_initial(2))

    def test_parameterization_is_not_clipped(self):
        self.assertAlmostEqual(lambda_from_vectors([log(2)], [1], [0], [1], 1), 1.2)
        self.assertAlmostEqual(lambda_from_vectors([0], [1], [log(2)], [1], 1), -0.8)

    def test_invalid_parameter_vectors_and_layer(self):
        for vectors in (([], [], [], []), ([1], [1, 2], [1], [1])):
            with self.assertRaises(ValueError):
                lambda_from_vectors(*vectors, 1)
        with self.assertRaises(ValueError):
            lambda_from_vectors([1000], [1], [0], [1], 1)
        with self.assertRaises(ValueError):
            lambda_initial(0)
        for layer in (True, 1.5):
            with self.assertRaises(TypeError):
                lambda_initial(layer)


class ValidationTests(unittest.TestCase):
    def test_invalid_shapes(self):
        for bad in ([], [[]], [[1], [1, 2]], [[1, 2]], [[1], [2], [3]]):
            data = list(fixture())
            data[0] = bad
            with self.assertRaises(ValueError):
                differential_attention(*data, 0.5)
        data = list(fixture())
        data[-1] = [[1, 2]] * 3
        with self.assertRaises(ValueError):
            differential_attention(*data, 0.5)

    def test_nonfinite_inputs_and_overflow(self):
        for bad in (float("nan"), float("inf"), float("-inf")):
            data = fixture()
            data[2][0][0] = bad
            with self.assertRaises(ValueError):
                differential_attention(*data, 0.5)
            with self.assertRaises(ValueError):
                differential_attention(*fixture(), bad)
        data = fixture()
        data[0][0][0] = 1e308
        data[2][0][0] = 1e308
        with self.assertRaises(ValueError):
            differential_attention(*data, 0.5)

    def test_rejects_booleans_and_invalid_mask_flag(self):
        with self.assertRaises(TypeError):
            differential_attention(*fixture(), True)
        with self.assertRaises(TypeError):
            differential_attention(*fixture(), 0.5, causal=1)
        data = fixture()
        data[0][0][0] = True
        with self.assertRaises(TypeError):
            differential_attention(*data, 0.5)


if __name__ == "__main__":
    unittest.main()
