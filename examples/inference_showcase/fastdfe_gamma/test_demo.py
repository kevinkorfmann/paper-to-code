import copy
import importlib.metadata
import json
from math import erfc, exp, fsum, lgamma, log, sqrt
from pathlib import Path
import subprocess
import sys
import unittest

import fastdfe as fd
import numpy as np

from examples.inference_showcase.fastdfe_gamma.demo import (
    FIXED, FIXTURE, N, NEUTRAL_TARGET, THETA, TRUTH,
    build_report, check_sample, dfe_masses, discretization, expected_spectra,
    fit, generate_input, observed_spectra,
)


class FastdfeGammaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = json.loads(FIXTURE.read_text())
        cls.neutral, cls.selected = observed_spectra(cls.fixture)
        cls.report = build_report(cls.fixture)

    def test_pinned_tool_and_frozen_generator(self):
        self.assertEqual(importlib.metadata.version("fastdfe"), "1.4.1")
        self.assertEqual(generate_input(), self.fixture)
        self.assertNotEqual(generate_input(17)["selected_polymorphic_counts"],
                            self.fixture["selected_polymorphic_counts"])
        self.assertFalse(fd.Settings.parallelize)

    def test_truth_seed_and_generator_metadata_are_not_fit_inputs(self):
        minimal = {key: value for key, value in self.fixture.items()
                   if key in ("haploid_sample_size", "neutral_target_sites", "selected_target_sites",
                              "neutral_polymorphic_counts", "selected_polymorphic_counts")}
        neutral, selected = observed_spectra(minimal)
        np.testing.assert_array_equal(neutral.data, self.neutral.data)
        np.testing.assert_array_equal(selected.data, self.selected.data)
        self.assertNotAlmostEqual(self.report["fit"]["S_d"], TRUTH["S_d"], places=0)

    def test_independent_neutral_expectation_and_theta(self):
        expected, _ = expected_spectra()
        np.testing.assert_allclose(expected.polymorphic, [THETA*NEUTRAL_TARGET/i for i in range(1, N)],
                                   rtol=1e-13, atol=0)
        harmonic = fsum(1/i for i in range(1, N))
        theta = fsum(self.fixture["neutral_polymorphic_counts"])/(harmonic*NEUTRAL_TARGET)
        self.assertAlmostEqual(self.neutral.theta, theta, places=14)
        self.assertEqual(expected.n_sites, NEUTRAL_TARGET)

    def test_gamma_cdf_against_closed_form_half_and_unit_shapes(self):
        # For shape 1, |S| is exponential; for 1/2, use the erfc identity.
        # Neither reference uses scipy's gamma CDF or fastDFE's discretization.
        x = np.array([0., 0.01, 1., 10., 100., 1000.])
        for shape in (0.5, 1.0):
            reference = [erfc(sqrt(shape*v/20)) if shape == 0.5 else exp(-v/20) for v in x]
            cdf = fd.GammaExpParametrization().get_cdf(S_d=-20, b=shape, p_b=0, S_b=1)
            np.testing.assert_allclose(cdf(-x), reference, rtol=1e-11, atol=1e-14)
            tails = [reference[i] for i in (2, 3, 4)]
            masses = [1-tails[0], tails[0]-tails[1], tails[1]-tails[2], tails[2]]
            np.testing.assert_allclose(dfe_masses(-20, shape), masses, rtol=1e-11, atol=1e-14)

    def test_independent_poisson_likelihood(self):
        row = self.report["fit"]
        observed = self.fixture["selected_polymorphic_counts"]
        ll = fsum(k*log(mean)-mean-lgamma(k+1)
                  for k, mean in zip(observed, row["expected_selected_counts"]))
        self.assertAlmostEqual(ll, row["log_likelihood"], places=5)

    def test_integration_range_contains_gamma_tail_at_fit_bound_corners(self):
        grid = discretization()
        for mean in (-1e4, -0.1):
            for shape in (0.1, 5.0):
                cdf = fd.GammaExpParametrization().get_cdf(S_d=mean, b=shape, p_b=0, S_b=1)
                tail = cdf(np.array([grid.bins[0], grid.bins[-1]]))
                self.assertLess(float(tail[0]), 1e-10)
                self.assertEqual(float(tail[1]), 1.0)

    def test_multistart_fit_beats_neighboring_parameters(self):
        best = self.report["fit"]
        for key in ("S_d", "b"):
            np.testing.assert_allclose([row[key] for row in best["runs"]], best[key], rtol=2e-4)
            for factor in (0.98, 1.02):
                params = dict(FIXED, S_d=best["S_d"], b=best["b"])
                params[key] *= factor
                expected = fd.Simulation(sfs_neut=self.neutral, params=params,
                                         discretization=discretization(), parallelize=False).run()
                means = expected.polymorphic*self.selected.n_sites/self.neutral.n_sites
                ll = fsum(k*log(m)-m-lgamma(k+1) for k, m in zip(self.selected.polymorphic, means))
                self.assertLess(ll, best["log_likelihood"])

    def test_noiseless_same_solver_recovers_mean_and_shape(self):
        result = self.report["same_solver_noiseless_control"]
        self.assertAlmostEqual(result["S_d"], TRUTH["S_d"], places=3)
        self.assertAlmostEqual(result["b"], TRUTH["b"], places=5)

    def test_target_perturbation_preserves_all_polymorphic_counts(self):
        neutral, selected = observed_spectra(self.fixture, 2.0)
        np.testing.assert_array_equal(neutral.data, self.neutral.data)
        np.testing.assert_array_equal(selected.polymorphic, self.selected.polymorphic)
        self.assertEqual(selected.n_sites, 2*self.selected.n_sites)
        self.assertEqual(selected.data[-1], 0)
        baseline, wrong = self.report["fit"], self.report["double_selected_target_control"]
        self.assertEqual(baseline["theta_from_neutral"], wrong["theta_from_neutral"])
        self.assertGreater(abs(wrong["S_d"]), 10*abs(baseline["S_d"]))
        self.assertGreater(abs(wrong["b"]-baseline["b"]), 0.1)
        # These inequalities describe the fixed fixture, not a universal DFE response.

    def test_numerical_resolution_sensitivity(self):
        for field in ("relative_resolution_change_in_mean_magnitude", "relative_resolution_change_in_shape"):
            self.assertGreater(self.report[field], 0)
            self.assertLess(self.report[field], 0.001)

    def test_count_and_target_scaling_preserves_inferred_dfe(self):
        doubled = fit(fd.Spectrum(2*self.neutral.data), fd.Spectrum(2*self.selected.data))
        for key in ("S_d", "b", "theta_from_neutral"):
            np.testing.assert_allclose(doubled[key], self.report["fit"][key], rtol=2e-4)

    def test_probabilities_and_fixed_model_assumptions(self):
        self.assertEqual(self.report["fixed_parameters"], {"p_b": 0, "S_b": 1, "eps": 0, "h": 0.5})
        for name in ("fit", "double_selected_target_control", "refined_resolution_fit"):
            masses = self.report[name]["magnitude_bin_probabilities"]
            self.assertTrue(all(0 <= p <= 1 for p in masses))
            self.assertAlmostEqual(sum(masses), 1.0, places=8)

    def test_invalid_observed_counts_and_targets(self):
        cases = [dict(haploid_sample_size=3), dict(neutral_target_sites=0),
                 dict(selected_target_sites=2), dict(selected_polymorphic_counts=[0]*19),
                 dict(neutral_polymorphic_counts=[0]*19),
                 dict(selected_polymorphic_counts=[1]*18),
                 dict(selected_polymorphic_counts=[-1]+[1]*18),
                 dict(selected_polymorphic_counts=[True]+[1]*18),
                 dict(selected_polymorphic_counts=[1.5]+[1]*18)]
        for changes in cases:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                observed_spectra(self.fixture | changes)
        for factor in (0, -1, True, float("nan"), float("inf")):
            with self.subTest(factor=factor), self.assertRaises(ValueError):
                observed_spectra(self.fixture, factor)

    def test_invalid_fit_inputs_and_starts(self):
        for starts in ((), ((0, 1),), ((-20, float("nan")),), ((True, 1),)):
            with self.subTest(starts=starts), self.assertRaises(ValueError):
                fit(self.neutral, self.selected, starts=starts)
        for values in ([1, 2, 3, 4, 0], [0, *self.selected.polymorphic, 0],
                       [1, *self.selected.polymorphic, 2]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                fit(self.neutral, fd.Spectrum(values))
        for resolution in (0, 99, 1601, True):
            with self.subTest(resolution=resolution), self.assertRaises(ValueError):
                discretization(resolution=resolution)
        for mean, shape in ((0, 1), (-20, 0), (-20, float("nan"))):
            with self.assertRaises(ValueError):
                dfe_masses(mean, shape)

    def test_inference_does_not_modify_inputs(self):
        original = copy.deepcopy(self.fixture)
        arrays = [s.data.copy() for s in (self.neutral, self.selected)]
        fit(self.neutral, self.selected)
        self.assertEqual(self.fixture, original)
        for sfs, array in zip((self.neutral, self.selected), arrays):
            np.testing.assert_array_equal(sfs.data, array)

    def test_checked_output_and_reject_changed_result(self):
        check_sample(self.report, json.loads(Path(__file__).with_name("sample_summary.json").read_text()))
        with self.assertRaises(AssertionError):
            check_sample({"S_d": -15.0}, {"S_d": -25.0})

    def test_cli_reproduces_report(self):
        result = subprocess.run([sys.executable, "-m", "examples.inference_showcase.fastdfe_gamma.demo", "--check-sample"],
                                capture_output=True, text=True, check=True, timeout=45)
        check_sample(json.loads(result.stdout), self.report)

    def test_cli_rejects_invalid_target_factor(self):
        result = subprocess.run([sys.executable, "-m", "examples.inference_showcase.fastdfe_gamma.demo",
                                 "--target-factor", "-1"], capture_output=True, text=True, timeout=45)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("selected_target_factor must be positive", result.stderr)


if __name__ == "__main__":
    unittest.main()
