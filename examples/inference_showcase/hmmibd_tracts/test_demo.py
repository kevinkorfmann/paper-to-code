"""Real CLI checks, exhaustive Decimal HMM oracle, controls, and reproducibility."""

import hashlib
import itertools
import json
import math
from decimal import Decimal, localcontext
import subprocess
import sys
import unittest
from unittest.mock import patch

from examples.inference_showcase.hmmibd_tracts import build, demo


def exhaustive_decimal(left, right, positions):
    """Enumerate every hidden path, independent of the sum-product implementation."""
    with localcontext() as context:
        context.prec = 50
        error = Decimal("0.001")
        emission = [((1-error)**2/2 + error**2/2 if a == b else error*(1-error),
                     Decimal("0.25")) for a, b in zip(left, right)]
        resets = [1-(-Decimal("0.00000074")*(b-a)).exp()
                  for a, b in zip(positions, positions[1:])]
        total = Decimal(0)
        weighted_ibd = Decimal(0)
        best_weight = Decimal(-1)
        best_path = None
        for states in itertools.product((0, 1), repeat=len(left)):
            weight = Decimal("0.5")
            for i, state in enumerate(states):
                weight *= emission[i][state]
                if i:
                    switch = resets[i-1]/2
                    weight *= switch if state != states[i-1] else 1-switch
            total += weight
            weighted_ibd += weight*states.count(0)/len(states)
            if weight > best_weight:
                best_path, best_weight = states, weight
        return float(weighted_ibd/total), best_path


class HmmIbdTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = json.loads(demo.FIXTURE.read_text())
        cls.report = demo.report()
        cls.inputs = demo.case_inputs(cls.fixture, "blocks")

    def test_source_checksum(self):
        build.verify((build.BUILD / "hmmIBD.c").read_bytes(), build.SOURCE_SHA256)
        build.verify((build.BUILD / "LICENSE").read_bytes(), build.LICENSE_SHA256)

    def test_modified_source_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "checksum"):
            build.verify(b"changed", build.SOURCE_SHA256)

    def test_fixture_regenerates(self):
        self.assertEqual(demo.generate_fixture(), self.fixture)
        self.assertEqual(self.fixture["truth_for_evaluation_only"]["blocks_ibd_indices_half_open"], [400, 800])

    def test_new_seed_changes_genotypes(self):
        other = demo.generate_fixture(demo.SEED + 1)
        self.assertNotEqual(other["cases"], self.fixture["cases"])
        self.assertEqual(other["truth_for_evaluation_only"], self.fixture["truth_for_evaluation_only"])

    def test_block_tract_recovered_for_this_fixture(self):
        fit = self.report["fits"]["blocks"]
        shared = [s for s in fit["viterbi_segments"] if s["state"] == 0]
        self.assertEqual(shared, [{"start_bp": 400001, "end_bp": 799001, "state": 0, "markers": 400}])
        self.assertAlmostEqual(fit["posterior_mean_ibd"], 1/3, delta=0.005)
        self.assertNotEqual(fit["posterior_mean_ibd"], fit["viterbi_marker_fraction_ibd"])

    def test_positive_and_negative_controls(self):
        self.assertGreater(self.report["fits"]["identical"]["posterior_mean_ibd"], 0.999)
        self.assertLess(self.report["fits"]["independent"]["posterior_mean_ibd"], 0.001)
        self.assertEqual(self.report["fits"]["identical"]["viterbi_segments"][0]["state"], 0)
        self.assertEqual(self.report["fits"]["independent"]["viterbi_segments"][0]["state"], 1)

    def test_more_em_iterations_preserve_results(self):
        for name in self.fixture["cases"]:
            self.assertEqual(demo.run_tool(*demo.case_inputs(self.fixture, name), iterations=100),
                             self.report["fits"][name])

    def test_conditional_posterior_matches_independent_sum_product(self):
        check = self.report["fixed_parameter_check"]
        self.assertAlmostEqual(check["tool_posterior_mean_ibd"], check["oracle_posterior_mean_ibd"], delta=5.1e-6)

    def test_exhaustive_decimal_oracle_and_viterbi(self):
        left = [1]*12
        right = [1]*5 + [0]*3 + [1]*4
        positions = [1+i*50_000 for i in range(12)]
        mean, path = exhaustive_decimal(left, right, positions)
        posterior = demo.posterior_oracle(left, right, positions)
        self.assertAlmostEqual(sum(posterior)/len(posterior), mean, places=13)
        actual = demo.run_tool(left, right, positions, iterations=1, prior=0.5)
        self.assertAlmostEqual(actual["posterior_mean_ibd"], mean, delta=5.1e-6)
        expanded = tuple(state for segment in actual["viterbi_segments"]
                         for state in [segment["state"]]*segment["markers"])
        self.assertEqual(expanded, path)

    def test_emission_probabilities_sum_to_one(self):
        for f in (0.01, 0.2, 0.5, 0.9):
            values = [demo.emission(a, b, f) for a, b in itertools.product((0, 1), repeat=2)]
            for state in (0, 1):
                self.assertAlmostEqual(sum(x[state] for x in values), 1, places=14)

    def test_zero_error_bayes_factor(self):
        for f in (0.1, 0.5, 0.9):
            shared, independent = demo.emission(1, 1, f, error=0)
            self.assertAlmostEqual(shared / independent, 1/f)
            self.assertEqual(demo.emission(0, 1, f, error=0)[0], 0)

    def test_transition_is_stochastic_with_stationary_prior(self):
        for distance in (0, 5, 1000, 100_000_000):
            matrix = demo.transition(distance, prior=0.3)
            for row in matrix:
                self.assertAlmostEqual(sum(row), 1)
                self.assertTrue(all(0 <= x <= 1 for x in row))
            self.assertAlmostEqual(0.3*matrix[0][0] + 0.7*matrix[1][0], 0.3)
        self.assertEqual(demo.transition(0), ((1.0, 0.0), (0.0, 1.0)))

    def test_transition_semigroup(self):
        first, second, combined = [demo.transition(d, 0.3) for d in (1100, 2500, 3600)]
        for i, j in itertools.product((0, 1), repeat=2):
            self.assertAlmostEqual(sum(first[i][k]*second[k][j] for k in (0, 1)), combined[i][j], places=14)

    def test_frequency_control_matches_bayes_oracle(self):
        control = self.report["matching_allele_frequency_control"]
        for item in control.values():
            self.assertAlmostEqual(item["tool_posterior_mean_ibd"], item["independent_marker_bayes_oracle"], delta=5.1e-6)
        self.assertGreater(control["0.1"]["tool_posterior_mean_ibd"], control["0.5"]["tool_posterior_mean_ibd"] + 0.2)

    def test_missing_emission_is_uninformative(self):
        self.assertEqual(demo.emission(-1, 0, 0.2), (1.0, 1.0))
        positions = [1 + i*1000 for i in range(15)]
        self.assertTrue(all(abs(x-0.3) < 1e-14 for x in demo.posterior_oracle([-1]*15, [1]*15, positions, prior=0.3)))

    def test_missing_markers_match_cli_oracle(self):
        left, right, positions = [x.copy() for x in self.inputs]
        for i in range(100, 1100, 7):
            left[i] = -1
        result = demo.run_tool(left, right, positions, iterations=1, prior=0.5)
        oracle = sum(demo.posterior_oracle(left, right, positions))/len(left)
        self.assertAlmostEqual(result["posterior_mean_ibd"], oracle, delta=5.1e-6)

    def test_sample_order_invariance(self):
        left, right, positions = self.inputs
        self.assertEqual(demo.run_tool(right, left, positions), self.report["fits"]["blocks"])

    def test_allele_label_invariance(self):
        left, right, positions = self.inputs
        original = demo.run_tool(left, right, positions, 0.3, iterations=1, prior=0.5)
        flipped = demo.run_tool([1-x for x in left], [1-x for x in right], positions, 0.7, iterations=1, prior=0.5)
        self.assertEqual(original, flipped)

    def test_coordinate_origin_invariance(self):
        left, right, positions = self.inputs
        result = demo.run_tool(left, right, [p+123 for p in positions])
        for segment in result["viterbi_segments"]:
            segment["start_bp"] -= 123
            segment["end_bp"] -= 123
        self.assertEqual(result, self.report["fits"]["blocks"])

    def test_invalid_input_is_rejected_before_cli(self):
        for left, right, positions, freq in [([], [], [], .5), ([0], [0,1], [1], .5),
                ([2], [0], [1], .5), ([0,1], [0,1], [1,4], .5), ([0], [1], [0], .5),
                ([0], [1], [1.0], .5), ([0], [1], [1], 0), ([0], [1], [1], math.nan)]:
            with self.subTest((left, positions, freq)), self.assertRaises(ValueError):
                demo.run_tool(left, right, positions, freq)

    def test_invalid_iteration_and_prior_settings(self):
        for kwargs in ({"iterations": 0}, {"iterations": 101}, {"iterations": 1.5}, {"prior": 0}, {"prior": math.nan}):
            with self.subTest(kwargs), self.assertRaises(ValueError):
                demo.run_tool(*self.inputs, **kwargs)

    def test_insufficient_informative_markers_is_explicit(self):
        with self.assertRaisesRegex(ValueError, "informative"):
            demo.run_tool([1]*9, [1]*9, [1+i*1000 for i in range(9)])

    def test_missing_binary_has_build_instruction(self):
        with patch.object(demo, "BINARY", demo.HERE / "nonexistent-binary"), self.assertRaisesRegex(FileNotFoundError, "build"):
            demo.run_tool(*self.inputs)

    def test_checked_output(self):
        self.assertEqual(self.report, json.loads(demo.SAMPLE.read_text()))

    def test_cli_checked_sample(self):
        proc = subprocess.run([sys.executable, "-m", "examples.inference_showcase.hmmibd_tracts.demo", "--check-sample"],
                              capture_output=True, text=True, check=True, timeout=30)
        self.assertEqual(proc.stdout, demo.SAMPLE.read_text())

    def test_cli_rejects_unknown_option(self):
        proc = subprocess.run([sys.executable, "-m", "examples.inference_showcase.hmmibd_tracts.demo", "--unknown"], capture_output=True, text=True)
        self.assertNotEqual(proc.returncode, 0)


if __name__ == "__main__":
    unittest.main()
