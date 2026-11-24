import json
import math
import unittest

import numpy as np

from . import demo


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = demo.report()
        cls.full, cls.masked, cls.trimmed = cls.result["runs"]

    def test_same_markers_and_genotypes(self):
        full, *masked = demo.fixture()
        expected = [(m.position, m.ancestral_state) for m in full.sites()]
        for ts in masked:
            self.assertEqual([(m.position, m.ancestral_state) for m in ts.sites()], expected)
            np.testing.assert_array_equal(full.genotype_matrix(), ts.genotype_matrix())
        self.assertTrue(all(r["genotype_calls_unchanged_by_dating"] for r in self.result["runs"]))
        self.assertTrue(all(r["tip_dates_unchanged"] for r in self.result["runs"]))

    def test_nominal_length_is_not_exposure(self):
        self.assertEqual(self.full["nominal_coordinate_length_bp"], 2_000_000)
        self.assertEqual(self.masked["nominal_coordinate_length_bp"], 2_000_000)
        self.assertEqual(self.full["per_edge_spans_bp"], [2_000_000]*2)
        self.assertEqual(self.masked["per_edge_spans_bp"], [1_000_000]*2)
        self.assertEqual(self.trimmed["nominal_coordinate_length_bp"], 1_000_000)
        self.assertEqual(self.trimmed["per_edge_spans_bp"], [1_000_000]*2)
        self.assertTrue(all(r["per_edge_mutation_counts"] == [10,10] for r in self.result["runs"]))

    def test_age_direction_and_trim_control(self):
        self.assertAlmostEqual(self.full["time_estimate_generations"], 50, places=6)
        self.assertAlmostEqual(self.masked["time_estimate_generations"], 100, places=6)
        self.assertEqual(self.masked["time_estimate_generations"], self.trimmed["time_estimate_generations"])
        self.assertEqual(self.masked["posterior_sd_generations"], self.trimmed["posterior_sd_generations"])

    def test_independent_likelihood_maximum(self):
        # Direct Poisson probability for two equally long edges; no tsdate
        # internals or fitted values used to construct the likelihood.
        for r in self.result["runs"]:
            exposures = [demo.MUTATION_RATE * span for span in r["per_edge_spans_bp"]]
            counts = r["per_edge_mutation_counts"]
            def log_likelihood(time):
                return sum(k*math.log(e*time)-e*time-math.lgamma(k+1)
                           for e,k in zip(exposures,counts))
            optimum = max(range(1, 251), key=log_likelihood)
            self.assertEqual(r["independent_poisson_mle_generations"], optimum)
            self.assertAlmostEqual(r["time_estimate_generations"], optimum, places=6)

    def test_sample_and_nontrivial_failure(self):
        expected = json.loads((demo.HERE / "sample_output.json").read_text())
        self.assertTrue(demo.equal_report(self.result, expected))
        expected["runs"][1]["time_estimate_generations"] = 51
        self.assertFalse(demo.equal_report(self.result, expected))
