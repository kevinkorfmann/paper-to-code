import copy
import json
from math import fsum, lgamma, log
from pathlib import Path
import subprocess
import sys
import unittest

import dadi
import numpy as np

from examples.inference_showcase.dadi_size_change.demo import (
    FIXTURE, GRIDS, TRUE_NU, TRUE_THETA, TRUE_TIME,
    build_report, check_sample, fit, generate_input, load_sfs, spectrum,
)


class DadiSizeChangeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = json.loads(FIXTURE.read_text())
        cls.data = load_sfs(cls.fixture)
        cls.report = build_report(cls.fixture)

    def test_frozen_synthetic_input_and_seed(self):
        self.assertEqual(generate_input(), self.fixture)
        self.assertNotEqual(generate_input(17), self.fixture)
        self.assertEqual(self.data.sum(), 2307)

    def test_truth_and_seed_are_not_inference_inputs(self):
        minimal = {k: v for k, v in self.fixture.items()
                   if k in ("sample_size_haploid", "unfolded_counts_including_masked_corners")}
        np.testing.assert_array_equal(load_sfs(minimal), self.data)
        self.assertNotAlmostEqual(self.report["fit"]["nu"], TRUE_NU, places=1)

    def test_monomorphic_corners_are_ignored(self):
        changed = copy.deepcopy(self.fixture)
        changed["unfolded_counts_including_masked_corners"][0] = 123456
        changed["unfolded_counts_including_masked_corners"][-1] = 987654
        alternative = load_sfs(changed)
        np.testing.assert_array_equal(alternative.compressed(), self.data.compressed())
        np.testing.assert_array_equal(np.flatnonzero(alternative.mask), [0, 20])
        self.assertFalse(alternative.folded)

    def test_constant_population_matches_neutral_inverse_frequency_spectrum(self):
        # Independent analytical equilibrium: E[S_i] = theta/i.
        model = spectrum(1, TRUE_TIME)
        np.testing.assert_allclose(model[1:-1], 1/np.arange(1, 20), rtol=0.005)

    def test_profiled_theta_and_poisson_likelihood_against_independent_formula(self):
        model = spectrum(2.5, TRUE_TIME)
        observed, expected = list(self.data.compressed()), list(model.compressed())
        theta = fsum(observed)/fsum(expected)
        independent_ll = fsum(k*log(theta*m)-theta*m-lgamma(k+1)
                              for k, m in zip(observed, expected))
        self.assertAlmostEqual(dadi.Inference.optimal_sfs_scaling(model, self.data), theta, places=10)
        self.assertAlmostEqual(dadi.Inference.ll_multinom(model, self.data), independent_ll, places=9)
        for factor in (0.8, 1.2):
            self.assertLess(dadi.Inference.ll(theta*factor*model, self.data), independent_ll)

    def test_multistart_optimum_and_neighboring_likelihood(self):
        result = self.report["fit"]
        self.assertLess(np.ptp(result["nu_from_each_start"]), 1e-4)
        self.assertGreater(result["profile_poisson_log_likelihood"], result["constant_size_profile_log_likelihood"])
        for delta in (-0.05, 0.05):
            neighbor = spectrum(result["nu"]+delta, TRUE_TIME)
            self.assertLess(dadi.Inference.ll_multinom(neighbor, self.data), result["profile_poisson_log_likelihood"])

    def test_noiseless_same_solver_recovers_generator(self):
        control = self.report["same_solver_noiseless_control"]
        self.assertAlmostEqual(control["nu"], TRUE_NU, places=4)
        self.assertAlmostEqual(control["theta"], TRUE_THETA, places=3)

    def test_profile_fit_invariant_to_count_scale(self):
        doubled = fit(2*self.data)
        self.assertAlmostEqual(doubled["nu"], self.report["fit"]["nu"], places=4)
        self.assertAlmostEqual(doubled["theta"], 2*self.report["fit"]["theta"], places=3)

    def test_wrong_time_changes_estimate_despite_higher_likelihood(self):
        correct, wrong = self.report["fit"], self.report["assumed_time_0_8_control"]
        self.assertGreater(abs(correct["nu"]-wrong["nu"]), 0.2)
        self.assertGreater(wrong["profile_poisson_log_likelihood"], correct["profile_poisson_log_likelihood"])
        # This is a property of the fixed noisy fixture, not a universal claim.

    def test_grid_refinement_quantifies_discretization(self):
        self.assertGreater(self.report["grid_refinement_relative_nu_change"], 0)
        self.assertLess(self.report["grid_refinement_relative_nu_change"], 0.005)

    def test_invalid_observed_data(self):
        for counts in ([0]*21, [0, 1, 0], [0, -1]+[0]*19,
                       [0, 1.5]+[0]*19, [0, True]+[0]*19):
            changed = dict(self.fixture, unfolded_counts_including_masked_corners=counts)
            with self.subTest(counts=counts), self.assertRaises(ValueError):
                load_sfs(changed)
        with self.assertRaises(ValueError):
            fit(self.data.fold())
        missing_bin = self.data.copy()
        missing_bin.mask[5] = True
        with self.assertRaises(ValueError):
            fit(missing_bin)

    def test_invalid_model_parameters_and_starts(self):
        for value in (0, -1, True, float("inf"), float("nan")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                spectrum(value, TRUE_TIME)
            with self.subTest(time=value), self.assertRaises(ValueError):
                fit(self.data, time=value)
        for grids in ((40, 50), (50, 40, 60), (20, 30, 40), (40.0, 50, 60)):
            with self.subTest(grids=grids), self.assertRaises(ValueError):
                spectrum(3, 0.2, grids=grids)
        for starts in ((), (0.2,), (10,), (True,)):
            with self.subTest(starts=starts), self.assertRaises(ValueError):
                fit(self.data, starts=starts)
        with self.assertRaises(ValueError):
            fit(self.data, time=3)

    def test_observations_are_not_mutated(self):
        original = copy.deepcopy(self.fixture)
        fit(self.data)
        self.assertEqual(self.fixture, original)
        np.testing.assert_array_equal(self.data.compressed(), load_sfs(original).compressed())
        self.assertFalse(dadi.cuda_enabled())

    def test_checked_output_and_reject_changed_result(self):
        check_sample(self.report, json.loads(Path(__file__).with_name("sample_summary.json").read_text()))
        with self.assertRaises(AssertionError):
            check_sample({"nu": 2.0}, {"nu": 3.0})

    def test_cli_and_invalid_time(self):
        command = [sys.executable, "-m", "examples.inference_showcase.dadi_size_change.demo"]
        output = subprocess.run(command+["--check-sample"], capture_output=True, text=True, check=True, timeout=30)
        check_sample(json.loads(output.stdout), self.report)
        invalid = subprocess.run(command+["--time", "-1"], capture_output=True, text=True, timeout=30)
        self.assertNotEqual(invalid.returncode, 0)
        self.assertIn("time must be positive", invalid.stderr)


if __name__ == "__main__":
    unittest.main()
