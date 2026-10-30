"""Independent references and controls for the scalar association update."""

from decimal import Decimal, localcontext
import math
from pathlib import Path
import random
import subprocess
import sys
import unittest

from examples.kronecker_delta.demo import report
from examples.kronecker_delta.model import association, delta_update, kron_update, read


ROOT = Path(__file__).resolve().parents[2]


def decimal_reference(state, key, value, target, beta):
    """80-digit affine transition on COLUMN-major Kronecker features (Eq. 4).

    Deliberately forms the full transition matrix instead of using the
    implementation's matrix-vector read followed by an outer-product update.
    """
    with localcontext() as ctx:
        ctx.prec = 80
        d = lambda x: Decimal.from_float(float(x))
        rows, cols = len(value), len(key)
        a = [d(k)*d(v) for k in key for v in value]
        s = [d(state[i][j]) for j in range(cols) for i in range(rows)]
        n = len(s)
        updated = [sum(((Decimal(int(i == j))-d(beta)*a[i]*a[j])*s[j]
                        for j in range(n)), Decimal(0)) + d(beta)*d(target)*a[i]
                   for i in range(n)]
        return tuple(tuple(float(updated[j*rows+i]) for j in range(cols)) for i in range(rows))


class KroneckerTests(unittest.TestCase):
    def assertMatrixClose(self, actual, expected, tolerance=2e-12):
        self.assertEqual(len(actual), len(expected))
        for a, b in zip(actual, expected, strict=True):
            self.assertEqual(len(a), len(b))
            for x, y in zip(a, b, strict=True):
                self.assertTrue(math.isclose(x, y, rel_tol=tolerance, abs_tol=tolerance), (x, y))

    def test_seeded_decimal_transition_oracle(self):
        rng = random.Random(20261030)
        for rows, cols in [(1, 1), (2, 3), (3, 2), (4, 3)]:
            for _ in range(12):
                state = [[rng.uniform(-2, 2) for _ in range(cols)] for _ in range(rows)]
                key = [rng.uniform(-1, 1) for _ in range(cols)]
                value = [rng.uniform(-1, 1) for _ in range(rows)]
                target, beta = rng.uniform(-2, 2), rng.uniform(0, 1)
                self.assertMatrixClose(kron_update(state, key, value, target=target, beta=beta),
                                       decimal_reference(state, key, value, target, beta))

    def test_rectangular_read_and_association(self):
        state = ((1, 2, 3), (4, 5, 6))
        self.assertEqual(read(state, (2, -1, 1)), (3, 9))
        self.assertEqual(association(state, (2, -1, 1), (3, 2)), 27)

    def test_unit_pair_projection(self):
        state = kron_update(((0, 0), (0, 0)), (1, 0), (0, 1))
        self.assertEqual(state, ((0, 0), (1, 0)))

    def test_nonunit_projection_step(self):
        state, key, value = ((1, 2), (3, 4), (5, 6)), (2, -3), (1, 2, -2)
        beta = 1/(13*9)
        updated = kron_update(state, key, value, target=-2, beta=beta)
        self.assertAlmostEqual(association(updated, key, value), -2, places=12)

    def test_nonunit_naive_step_is_not_projection(self):
        updated = kron_update(((0,),), (2,), (3,))
        self.assertEqual(association(updated, (2,), (3,)), 36)

    def test_residual_recurrence(self):
        state, key, value, target, beta = ((0.2, -0.3), (0.4, 0.1)), (0.5, 1), (1, -0.5), -0.6, 0.3
        old_error = association(state, key, value)-target
        updated = kron_update(state, key, value, target=target, beta=beta)
        self.assertAlmostEqual(association(updated, key, value)-target, old_error*(1-beta*1.25**2))

    def test_repeated_pair_geometric_contraction(self):
        state = ((0,),)
        for step in range(1, 9):
            state = kron_update(state, (1,), (1,), beta=0.5)
            self.assertEqual(state[0][0]-1, -0.5**step)

    def test_reflection_boundary_does_not_contract(self):
        state = ((0,),)
        for expected in (2, 0, 2, 0):
            state = kron_update(state, (1,), (1,), beta=2)
            self.assertEqual(state, ((expected,),))

    def test_step_outside_interval_diverges(self):
        state = ((0,),)
        for step in range(1, 5):
            state = kron_update(state, (1,), (1,), beta=3)
            self.assertEqual(state[0][0]-1, -(-2)**step)

    def test_scalar_gradient_finite_difference(self):
        state = [[0.2, -0.4], [0.7, 0.1]]
        key, value, target = (0.3, -0.8), (0.6, 0.5), 0.9
        updated = kron_update(state, key, value, target=target, beta=0.2)
        def loss(s):
            return 0.5*(association(s, key, value)-target)**2
        for i in range(2):
            for j in range(2):
                plus, minus = [r[:] for r in state], [r[:] for r in state]
                plus[i][j] += 1e-6
                minus[i][j] -= 1e-6
                gradient = (loss(plus)-loss(minus))/2e-6
                self.assertAlmostEqual((state[i][j]-updated[i][j])/0.2, gradient, places=9)

    def test_projection_minimum_norm_control(self):
        # Every alternative below reaches the same scalar target by adding an
        # explicitly orthogonal perturbation. Pythagoras measures extra cost.
        state, key, value = ((0.2, -0.4), (0.7, 0.1)), (1, 0), (1, 0)
        projected = kron_update(state, key, value, target=2)
        base_cost = sum((projected[i][j]-state[i][j])**2 for i in range(2) for j in range(2))
        for z in (-2, -0.5, 0.5, 2):
            alternative = (projected[0], (projected[1][0]+z, projected[1][1]))
            self.assertEqual(association(alternative, key, value), 2)
            cost = sum((alternative[i][j]-state[i][j])**2 for i in range(2) for j in range(2))
            self.assertAlmostEqual(cost, base_cost+z*z)

    def test_both_overlaps_control_interference(self):
        state = ((0.2, 0.7), (-0.4, 0.1))
        key, value = (0.6, 0.8), (-0.8, 0.6)
        old_k, old_v, beta, target = (0.2, -0.3), (0.7, 0.4), 0.7, 1.2
        updated = kron_update(state, key, value, beta=beta, target=target)
        expected = beta*(target-association(state, key, value))*sum(a*b for a, b in zip(old_v, value))*sum(a*b for a, b in zip(key, old_k))
        actual = association(updated, old_k, old_v)-association(state, old_k, old_v)
        self.assertAlmostEqual(actual, expected)

    def test_orthogonal_key_protects_all_readout_coordinates(self):
        state = ((1, 2), (3, 4))
        updated = kron_update(state, (1, 0), (0.6, 0.8))
        self.assertEqual(read(updated, (0, 1)), read(state, (0, 1)))

    def test_orthogonal_value_protects_scalar_not_whole_readout(self):
        state = ((1, 0), (0, 0))
        updated = kron_update(state, (1, 0), (0, 1))
        self.assertEqual(association(updated, (1, 0), (1, 0)), 1)
        self.assertEqual(read(updated, (1, 0)), (1, 1))
        self.assertNotEqual(read(updated, (1, 0)), read(state, (1, 0)))

    def test_vector_delta_overwrites_unit_key(self):
        state = delta_update(((1, 0), (0, 0)), (1, 0), (0, 1))
        self.assertEqual(state, ((0, 0), (1, 0)))

    def test_vector_delta_independent_fraction_control(self):
        # S k=(3/2, 3), residual target-read=(1/2, -4).
        self.assertEqual(delta_update(((1, 1), (2, 2)), (1, 0.5), (2, -1), beta=0.5),
                         ((1.25, 1.125), (0, 1)))

    def test_zero_target_erases_only_requested_scalar(self):
        state = kron_update(((1, 0), (1, 0)), (1, 0), (1, 0), target=0)
        self.assertEqual(read(state, (1, 0)), (0, 1))

    def test_zero_feature_noop(self):
        state = ((1, 2), (3, 4))
        self.assertEqual(kron_update(state, (0, 0), (1, 1)), state)
        self.assertEqual(kron_update(state, (1, 1), (0, 0)), state)
        self.assertEqual(delta_update(state, (0, 0), (1, 1)), state)

    def test_zero_beta_noop(self):
        for update in (kron_update, delta_update):
            self.assertEqual(update(((1e308,),), (1e308,), (1e308,), beta=0), ((1e308,),))

    def test_satisfied_constraint_noop(self):
        state = ((1, 2), (3, 4))
        self.assertEqual(kron_update(state, (1, 0), (1, 0)), state)

    def test_negative_target_allowed(self):
        self.assertEqual(kron_update(((0,),), (1,), (1,), target=-2), ((-2,),))

    def test_inputs_not_mutated_and_output_immutable(self):
        state, key, value = [[1, 2], [3, 4]], [1, 0], [0, 1]
        for update in (kron_update, delta_update):
            result = update(state, key, value)
            self.assertIsInstance(result, tuple)
            self.assertIsInstance(result[0], tuple)
        self.assertEqual((state, key, value), ([[1, 2], [3, 4]], [1, 0], [0, 1]))

    def test_prefix_states_unchanged_by_later_writes(self):
        events = [((1, 0), (1, 0)), ((0, 1), (0, 1)), ((1, 1), (1, -1))]
        states = [((0, 0), (0, 0))]
        for k, v in events:
            states.append(kron_update(states[-1], k, v, beta=0.25))
        for end in range(1, 4):
            state = states[0]
            for k, v in events[:end]:
                state = kron_update(state, k, v, beta=0.25)
            self.assertEqual(state, states[end])

    def test_dimension_and_shape_validation(self):
        cases = [((), (1,), (1,)), (((),), (1,), (1,)), (((1,), (1, 2)), (1,), (1, 1)),
                 (((1,),), (), (1,)), (((1,),), (1,), ()), (((1,),), (1, 2), (1,)),
                 (((1,),), (1,), (1, 2)), (None, (1,), (1,))]
        for args in cases:
            for function in (kron_update, delta_update, association):
                with self.subTest(args=args, function=function.__name__), self.assertRaises(ValueError):
                    function(*args)
        with self.assertRaises(ValueError):
            read(((1,),), (1, 2))

    def test_nonfinite_and_non_numeric_validation(self):
        for bad in (math.nan, math.inf, -math.inf, True, "2", None, 1j):
            for function in (kron_update, delta_update, association):
                for args in ((((bad,),), (1,), (1,)), (((1,),), (bad,), (1,)), (((1,),), (1,), (bad,))):
                    with self.subTest(bad=bad, function=function.__name__), self.assertRaises(ValueError):
                        function(*args)
            with self.assertRaises(ValueError):
                kron_update(((0,),), (1,), (1,), target=bad)
            with self.assertRaises(ValueError):
                kron_update(((0,),), (1,), (1,), beta=bad)

    def test_negative_beta_rejected(self):
        for update in (kron_update, delta_update):
            with self.assertRaises(ValueError):
                update(((0,),), (1,), (1,), beta=-1)

    def test_overflow_is_rejected(self):
        for update in (kron_update, delta_update):
            with self.assertRaises(ValueError):
                update(((1e308,),), (2,), (1,))
            with self.assertRaises(ValueError):
                update(((0,),), (1e308,), (1e308,))
        with self.assertRaises(ValueError):
            read(((1e308,),), (2,))
        with self.assertRaises(ValueError):
            association(((1e308,),), (1,), (2,))

    def test_demo_controls(self):
        result = report()
        self.assertEqual(result["overwrite_counterexample"]["kronecker_squared_error"], 1)
        self.assertEqual(result["overwrite_counterexample"]["vector_delta_squared_error"], 0)
        self.assertAlmostEqual(result["overlapping_values_interfere"]["old_association_after"], 31/25)
        self.assertAlmostEqual(result["overlapping_values_interfere"]["new_association"], 1)
        self.assertEqual(result["repeated_unit_pair_beta_half_errors"], [-0.5, -0.25, -0.125, -0.0625])

    def test_cli_sample_exact(self):
        run = subprocess.run([sys.executable, "-m", "examples.kronecker_delta.demo"], cwd=ROOT,
                             capture_output=True, check=True)
        self.assertEqual(run.stdout, Path(__file__).with_name("sample_output.json").read_bytes())
        self.assertEqual(run.stderr, b"")

    def test_cli_help(self):
        run = subprocess.run([sys.executable, "-m", "examples.kronecker_delta.demo", "--help"],
                             cwd=ROOT, capture_output=True, check=True)
        self.assertIn(b"usage:", run.stdout)

    def test_cli_unknown_flag_rejected(self):
        run = subprocess.run([sys.executable, "-m", "examples.kronecker_delta.demo", "--typo"],
                             cwd=ROOT, capture_output=True)
        self.assertNotEqual(run.returncode, 0)


if __name__ == "__main__":
    unittest.main()
