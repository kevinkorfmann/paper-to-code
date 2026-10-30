"""Mathematical controls plus real-executable integration (no network in tests)."""

from decimal import Decimal, localcontext
from fractions import Fraction
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import unittest

from examples.inference_showcase.admixture_projection import demo, install


class ProjectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = json.loads(demo.FIXTURE.read_text())
        cls.g = [[int(x) for x in row] for row in cls.fixture["allele2_dosages"]]
        cls.p = cls.fixture["frequency_patterns"] * cls.fixture["repeats"]
        cls.result = demo.summarize()
        cls.q = cls.result["correct_references"]["q_source_A"]

    def assert_q_close(self, observed, expected, tolerance=1.1e-6):
        self.assertEqual(len(observed), len(expected))
        for x, y in zip(observed, expected):
            self.assertAlmostEqual(x, y, delta=tolerance)

    def test_fixture_regenerates_exactly(self):
        self.assertEqual(demo.generate_input(), self.fixture)

    def test_fixture_shape_and_counts(self):
        self.assertEqual(len(self.g), 5)
        self.assertEqual({len(row) for row in self.g}, {2000})
        self.assertEqual({x for row in self.g for x in row}, {0, 1, 2})
        self.assertEqual(self.fixture["true_q_source_A"], [0, .25, .5, .75, 1])

    def test_reference_versions(self):
        self.assertEqual(platform.python_version(), "3.14.5")
        self.assertEqual(self.result["versions"]["admixture"], "1.4.0")
        install.verify(install.BINARY.read_bytes(), install.BINARY_SHA256)

    def test_checksum_rejects_altered_bytes(self):
        with self.assertRaises(ValueError):
            install.verify(b"not the official executable", install.BINARY_SHA256)

    def test_exact_bed_bytes_and_padding(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            demo.write_inputs(path, [[0, 2], [1, 1], [2, 0], [0, 2], [2, 1]], [[.2, .8]] * 2)
            self.assertEqual((path / "study.bed").read_bytes(), b"\x6c\x1b\x01\x38\x03\xcb\x02")
            self.assertEqual(len((path / "study.fam").read_text().splitlines()), 5)
            self.assertEqual((path / "study.2.P.in").read_text(), "0.2 0.8\n" * 2)

    def test_independent_bed_decoder_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            demo.write_inputs(path, self.g, self.p)
            data = (path / "study.bed").read_bytes()[3:]
            decoded = [[] for _ in self.g]
            table = {0: 0, 2: 1, 3: 2}
            for marker in range(2000):
                for i in range(5):
                    bits = (data[2 * marker + i // 4] >> (2 * (i % 4))) & 3
                    decoded[i].append(table[bits])
            self.assertEqual(decoded, self.g)

    def test_invalid_genotypes(self):
        for value in ([], [[]], [[True]], [[-1]], [[3]], [[.5]], [[None]], [[0, 1]]):
            with self.subTest(value=value), self.assertRaises(ValueError):
                demo.validate(value, [[.2, .8]])

    def test_invalid_frequencies(self):
        for value in ([], [[.2]], [[.2, .8, .5]], [[0, .8]], [[.2, 1]],
                      [[float("nan"), .8]], [[.2, float("inf")]], [[True, .8]]):
            with self.subTest(value=value), self.assertRaises(ValueError):
                demo.validate([[0]], value)

    def test_invalid_output(self):
        for value in ("", "0.5", "0.5 0.5 0", "nan nan", "-0.1 1.1", "0.2 0.2", "0 1\n0 1"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                demo.parse_q(value, 1)
        self.assertEqual(demo.parse_q("0.3 0.7\n", 1), [.3])

    def test_invalid_optimizer_seed(self):
        for seed in (-1, 2**31, 1.5, True):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                demo.run_projection([[0]], [[.2, .8]], seed)

    def test_oracle_closed_form_exact_rational(self):
        calls = [0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
        # Constant pA=1/5, pB=4/5: q=(mean dosage/2-pB)/(pA-pB).
        expected = (Fraction(sum(calls), 2 * len(calls)) - Fraction(4, 5)) / Fraction(-3, 5)
        self.assertAlmostEqual(demo.oracle_q(calls, [[.2, .8]] * 10), float(expected), places=14)

    def test_oracle_60_digit_decimal_reference(self):
        calls = [0, 1, 2, 1, 0, 2, 1]
        p = [[.1, .8], [.2, .7], [.3, .9], [.7, .2], [.4, .8], [.6, .2], [.25, .75]]
        with localcontext() as context:
            context.prec = 60
            rows = [(Decimal(str(a)), Decimal(str(b))) for a, b in p]
            lo, hi = Decimal(0), Decimal(1)
            for _ in range(210):
                mid = (lo + hi) / 2
                score = Decimal(0)
                for g, (a, b) in zip(calls, rows):
                    prob = mid * a + (1 - mid) * b
                    score += (a - b) * (Decimal(g) / prob - Decimal(2 - g) / (1 - prob))
                if score > 0:
                    lo = mid
                else:
                    hi = mid
            self.assertAlmostEqual(demo.oracle_q(calls, p), float((lo + hi) / 2), places=14)

    def test_oracle_is_global_grid_maximum(self):
        for calls in self.g:
            q = demo.oracle_q(calls, self.p)
            maximum = demo.log_likelihood(calls, self.p, q)
            self.assertTrue(all(maximum + 1e-9 >= demo.log_likelihood(calls, self.p, k / 20)
                                for k in range(21)))

    def test_real_estimates_match_independent_oracle(self):
        expected = [demo.oracle_q(g, self.p) for g in self.g]
        self.assert_q_close(self.q, expected, 1.01e-5)  # Includes numerical boundary offsets.
        self.assert_q_close(self.q[1:4], expected[1:4], 5.1e-7)

    def test_real_log_likelihood_independent_calculation(self):
        expected = math.fsum(demo.log_likelihood(g, self.p, q) for g, q in zip(self.g, self.q))
        self.assertAlmostEqual(self.result["correct_references"]["log_likelihood_without_binomial_constant"],
                               expected, delta=1.1e-6)

    def test_second_optimizer_seed_same_identifiable_solution(self):
        self.assert_q_close(demo.run_projection(self.g, self.p, 99)["q_source_A"], self.q)

    def test_halving_reference_contrast_doubles_interior_q(self):
        changed = self.result["halved_reference_contrast"]["q_source_A"]
        self.assert_q_close(changed[1:3], [2 * x for x in self.q[1:3]])
        self.assertEqual(changed[3:], [.99999, .99999])

    def test_changed_reference_matches_independent_oracle(self):
        p = [[(a + b) / 2, b] for a, b in self.p]
        expected = [demo.oracle_q(g, p) for g in self.g]
        self.assert_q_close(self.result["halved_reference_contrast"]["q_source_A"], expected, 1.01e-5)

    def test_identical_profiles_are_analytically_unidentifiable(self):
        p = [[.5, .5]] * 2000
        for g in self.g:
            self.assertIsNone(demo.oracle_q(g, p))
            for q in (0, .2, .5, .9, 1):
                self.assertAlmostEqual(demo.log_likelihood(g, p, q), -4000 * math.log(2), places=10)

    def test_flat_actual_output_changes_with_seed_not_likelihood(self):
        flat = self.result["identical_reference_profiles"]
        self.assertFalse(flat["identifiable"])
        self.assertGreater(max(abs(a - b) for a, b in zip(flat["seed_41"]["q_source_A"],
                                                        flat["seed_99"]["q_source_A"])), .3)
        for seed in ("seed_41", "seed_99"):
            self.assertEqual(flat[seed]["log_likelihood_without_binomial_constant"],
                             flat["analytical_log_likelihood"])

    def test_allele_complement_invariance(self):
        g = [[2 - x for x in row] for row in self.g]
        p = [[1 - a, 1 - b] for a, b in self.p]
        self.assert_q_close(demo.run_projection(g, p)["q_source_A"], self.q)

    def test_source_label_swap_complements_q(self):
        swapped = demo.run_projection(self.g, [[b, a] for a, b in self.p])
        self.assert_q_close(swapped["q_source_A"], [1 - x for x in self.q])

    def test_marker_order_invariance(self):
        reversed_markers = demo.run_projection([g[::-1] for g in self.g], self.p[::-1])
        self.assert_q_close(reversed_markers["q_source_A"], self.q)

    def test_individual_order_equivariance(self):
        permuted = demo.run_projection(self.g[::-1], self.p)
        self.assert_q_close(permuted["q_source_A"], self.q[::-1])

    def test_asymmetric_closed_form_real_tool_control(self):
        calls = [0] * 20 + [1] * 30 + [2] * 50  # allele2 frequency 0.65; q=0.25.
        actual = demo.run_projection([calls], [[.2, .8]] * 100)
        self.assert_q_close(actual["q_source_A"], [.25])

    def test_degenerate_boundary_control_exposes_false_convergence(self):
        control = self.result["degenerate_boundary_control"]
        actual = control["actual_tool"]
        self.assertEqual(actual["q_source_A"], [.998559, .001664])
        self.assertGreater(control["mathematical_max_log_likelihood"] -
                           actual["log_likelihood_without_binomial_constant"], .18)
        self.assertEqual(demo.oracle_q([0] * 40, [[.2, .8]] * 40), 1)
        self.assertEqual(demo.oracle_q([2] * 40, [[.2, .8]] * 40), 0)

    def test_sample_output_exact(self):
        self.assertEqual(json.dumps(self.result, indent=2, allow_nan=False) + "\n", demo.SAMPLE.read_text())

    def test_cli_check_sample_and_invalid_argument(self):
        name = "examples.inference_showcase.admixture_projection.demo"
        result = subprocess.run([sys.executable, "-m", name, "--check-sample"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("matches byte for byte", result.stdout)
        result = subprocess.run([sys.executable, "-m", name, "--not-an-option"], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
