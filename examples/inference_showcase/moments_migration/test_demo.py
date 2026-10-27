"""Analytical, inference, numerical, fixture, and CLI checks for the real tool."""

import copy
import json
import math
import subprocess
import sys
import unittest
from unittest.mock import patch

import moments
import numpy as np

from examples.inference_showcase.moments_migration import demo


class MomentsMigrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = json.loads(demo.FIXTURE.read_text())
        cls.counts = np.asarray(cls.fixture["counts"])
        cls.result = demo.report()
        cls.fit = cls.result["fits"]["observed"]

    def test_neutral_equilibrium_independent_one_over_i(self):
        fs = moments.Demographics1D.snm([24])
        np.testing.assert_allclose(fs.compressed(), 1/np.arange(1, 24), atol=1e-12, rtol=1e-12)

    def test_instantaneous_split_hypergeometric_oracle(self):
        n1, n2 = 8, 12
        fs = demo.expected_sfs(0.7, (n1, n2), time=0)
        for i in range(n1+1):
            for j in range(n2+1):
                if 0 < i+j < n1+n2:
                    expected = math.comb(n1, i)*math.comb(n2, j)/math.comb(n1+n2, i+j)/(i+j)
                    self.assertAlmostEqual(float(fs[i, j]), expected, places=12)

    def test_zero_time_has_no_information_about_migration(self):
        a = demo.expected_sfs(0, time=0)
        b = demo.expected_sfs(10, time=0)
        np.testing.assert_array_equal(a.compressed(), b.compressed())
        with self.assertRaises(ValueError):
            demo.fit(self.counts, time=0)

    def test_isolated_population_marginal_equilibrium(self):
        fs = demo.expected_sfs(0, time=1)
        for removed_axis in [0, 1]:
            np.testing.assert_allclose(fs.marginalize([removed_axis]).compressed(),
                                       1/np.arange(1, 12), atol=1e-12, rtol=1e-12)

    def test_symmetric_migration_preserves_exchange_symmetry(self):
        fs = demo.expected_sfs(1)
        np.testing.assert_allclose(fs.compressed(), fs.T.compressed(), atol=1e-12, rtol=1e-12)

    def test_only_globally_monomorphic_corners_are_masked(self):
        fs = demo.observed_sfs(self.counts)
        self.assertEqual(int(fs.mask.sum()), 2)
        self.assertTrue(fs.mask[0, 0] and fs.mask[-1, -1])
        self.assertFalse(fs.mask[0, -1] or fs.mask[-1, 0])
        self.assertEqual(int(fs.S()), int(self.counts.sum()))

    def test_fixture_regenerates_with_exact_integer_counts(self):
        self.assertEqual(demo.generate_fixture(), self.fixture)
        self.assertEqual(self.counts.dtype.kind, "i")
        self.assertEqual(self.fixture["sample_sizes"], [n-1 for n in self.counts.shape])
        self.assertEqual(self.counts.sum(), 2665)

    def test_seed_changes_data_without_changing_model(self):
        other = demo.generate_fixture(demo.SEED+1)
        self.assertNotEqual(other["counts"], self.fixture["counts"])
        self.assertEqual(other["generation_for_evaluation_only"]["M"], 1)

    def test_poisson_likelihood_and_scale_independent_oracle(self):
        model = demo.expected_sfs(self.fit["M"])
        theta, ll = demo.independent_profile_ll(model, self.counts)
        data = demo.observed_sfs(self.counts)
        self.assertAlmostEqual(theta, moments.Inference.optimal_sfs_scaling(model, data), places=10)
        self.assertAlmostEqual(ll, moments.Inference.ll_multinom(model, data), places=9)
        self.assertAlmostEqual(ll, moments.Inference.ll(theta*model, data), places=9)

    def test_all_starts_converge_to_same_interior_optimum(self):
        for fit in self.result["fits"].values():
            self.assertEqual([r["start"] for r in fit["starts"]], list(demo.STARTS))
            self.assertTrue(all(row["optimizer_warnflag"] == 0 for row in fit["starts"]))
            estimates = [row["M"] for row in fit["starts"]]
            self.assertLess(max(estimates)-min(estimates), 0.001)
            self.assertTrue(demo.BOUNDS[0] < fit["M"] < demo.BOUNDS[1])

    def test_likelihood_neighbors_and_log_spaced_grid(self):
        data = demo.observed_sfs(self.counts)
        candidates = list(np.geomspace(0.01, 10, 15))+[self.fit["M"]*0.98, self.fit["M"]*1.02]
        for m in candidates:
            ll = moments.Inference.ll_multinom(demo.expected_sfs(float(m)), data)
            self.assertLess(ll, self.fit["profile_log_likelihood"]+1e-6)

    def test_no_migration_control_is_worse_for_this_fixture(self):
        self.assertLess(self.result["checks"]["zero_migration_profile_log_likelihood"],
                        self.fit["profile_log_likelihood"]-100)

    def test_noiseless_same_solver_recovery(self):
        fit = self.result["fits"]["noiseless_same_solver"]
        self.assertAlmostEqual(fit["M"], 1, delta=1e-4)
        self.assertAlmostEqual(fit["theta"], 500, delta=0.01)

    def test_doubled_counts_preserve_shape_and_double_scale(self):
        fit = demo.fit(self.counts*2, starts=(1.0,))
        self.assertAlmostEqual(fit["M"], self.fit["M"], delta=2e-4)
        self.assertAlmostEqual(fit["theta"], self.fit["theta"]*2, delta=0.01)

    def test_changing_time_changes_the_conditional_estimate(self):
        wrong = self.result["fits"]["assumed_T_1"]
        self.assertGreater(abs(wrong["M"]-self.fit["M"]), 0.04)
        self.assertLess(wrong["profile_log_likelihood"], self.fit["profile_log_likelihood"])

    def test_time_step_refinement_is_small(self):
        self.assertLess(self.result["checks"]["halved_time_step_relative_M_change"], 0.001)

    def test_larger_internal_spectrum_diagnostic(self):
        self.assertLess(self.result["checks"]["larger_spectrum_relative_M_change"], 0.005)
        self.assertEqual(self.result["fits"]["larger_integrated_spectrum"]["padding_per_population"], 16)

    def test_invalid_data_and_model_inputs_rejected(self):
        for counts in [np.zeros((13, 13)), [[1]], np.ones(13), np.full((13, 13), -1),
                       np.full((13, 13), np.nan), np.full((13, 13), np.inf),
                       np.full((13, 13), 1e10)]:
            with self.subTest(shape=np.shape(counts)), self.assertRaises(ValueError):
                demo.observed_sfs(counts)
        for corner in [(0, 0), (-1, -1)]:
            invalid = self.counts.copy()
            invalid[corner] = 1
            with self.assertRaises(ValueError):
                demo.observed_sfs(invalid)
        for kwargs in [{"migration": -1}, {"migration": np.nan}, {"time": -1},
                       {"time": np.inf}, {"dt": 0}, {"padding": -1}, {"padding": 1.5},
                       {"sample_sizes": [12]}, {"sample_sizes": [12.5, 12]}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                demo.expected_sfs(**({"migration": 1}|kwargs))
        with self.assertRaises(ValueError):
            demo.fit(self.counts, starts=())

    def test_optimizer_failure_is_not_reported_as_success(self):
        with patch.object(moments.Inference, "optimize_log_fmin", return_value=([1], 0, 1, 1, 1)):
            with self.assertRaises(RuntimeError):
                demo.fit(self.counts)

    def test_fit_input_does_not_include_truth(self):
        metadata = copy.deepcopy(self.fixture)
        metadata["generation_for_evaluation_only"] = {"M": 99, "T": 99, "theta": 99}
        actual = demo.fit(metadata["counts"], starts=(1.0,))
        self.assertAlmostEqual(actual["M"], self.fit["M"], delta=2e-4)

    def test_checked_sample_and_strict_structure_versions(self):
        demo.assert_close(self.result, json.loads(demo.SAMPLE.read_text()))
        bad = copy.deepcopy(self.result)
        bad["versions"]["moments-popgen"] = "wrong"
        with self.assertRaises(AssertionError):
            demo.assert_close(bad, self.result)
        bad = copy.deepcopy(self.result)
        bad["fits"]["observed"]["M"] += 0.01
        with self.assertRaises(AssertionError):
            demo.assert_close(bad, self.result)

    def test_cli_generation_and_invalid_time(self):
        cmd = [sys.executable, "-m", "examples.inference_showcase.moments_migration.demo"]
        generated = subprocess.run(cmd+["--generate-input"], check=True, capture_output=True, text=True)
        self.assertEqual(json.loads(generated.stdout), self.fixture)
        for args in [["--time", "0"], ["--time", "1", "--check-sample"]]:
            proc = subprocess.run(cmd+args, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 2)
            self.assertIn("error:", proc.stderr)


if __name__ == "__main__":
    unittest.main()
