import copy
import json
from pathlib import Path
import unittest

import numpy as np
import tskit

from examples.inference_showcase.tsdate_clock.demo import (
    FIXTURE, RATE, build_report, build_undated, check_sample, generate_input, infer_dates, run_analysis,
)


class TsdateClockTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = json.loads(FIXTURE.read_text())
        cls.undated, cls.dated, cls.fit, cls.double, cls.fit_double = run_analysis(cls.data)

    def test_synthetic_generator_and_fixed_input(self):
        generated = generate_input()
        self.assertEqual(len(generated["mutations"]), 144)
        self.assertEqual(generated["edges"], self.data["edges"])
        self.assertEqual([m["node"] for m in generated["mutations"]], [m["node"] for m in self.data["mutations"]])
        np.testing.assert_allclose([m["position"] for m in generated["mutations"]],
                                   [m["position"] for m in self.data["mutations"]], rtol=0, atol=1e-8)
        self.assertNotEqual(generate_input(17)["mutations"], self.data["mutations"])

    def test_truth_ages_and_rate_are_not_inputs(self):
        stripped = copy.deepcopy(self.data)
        del stripped["true_node_times"]
        del stripped["simulation_mutation_rate"]
        self.assertTrue(build_undated(stripped).equals(self.undated))

    def test_placeholder_times_and_unknown_mutation_times(self):
        self.assertEqual(self.undated.time_units, tskit.TIME_UNITS_UNCALIBRATED)
        np.testing.assert_array_equal(self.undated.nodes_time, [0, 0, 0, 0, 1, 1, 2])
        self.assertTrue(np.all(tskit.is_unknown_time(self.undated.mutations_time)))
        self.assertEqual(self.undated.num_provenances, 0)

    def test_placeholder_scale_does_not_set_estimated_ages(self):
        _, alternative = infer_dates(build_undated(self.data, placeholder_scale=10000), RATE)
        np.testing.assert_allclose(alternative.node_posteriors()["mean"], self.fit.node_posteriors()["mean"], rtol=1e-8)

    def test_topology_and_genotypes_preserved(self):
        for dated in (self.dated, self.double):
            self.assertTrue(dated.tables.edges.equals(self.undated.tables.edges))
            np.testing.assert_array_equal(dated.genotype_matrix(), self.undated.genotype_matrix())
            np.testing.assert_array_equal(dated.sites_position, self.undated.sites_position)

    def test_parent_ordering_and_fixed_sample_ages(self):
        for dated in (self.dated, self.double):
            self.assertEqual(dated.time_units, "generations")
            np.testing.assert_array_equal(dated.nodes_time[dated.samples()], 0)
            for edge in dated.edges():
                self.assertGreater(dated.nodes_time[edge.parent], dated.nodes_time[edge.child])

    def test_posteriors_are_finite_with_uncertainty(self):
        posterior = self.fit.node_posteriors()
        for field in ("mean", "variance"):
            self.assertTrue(np.all(np.isfinite(posterior[field][4:])))
            self.assertTrue(np.all(posterior[field][4:] > 0))

    def test_double_clock_halves_mean_and_quarters_variance(self):
        p, q = self.fit.node_posteriors(), self.fit_double.node_posteriors()
        np.testing.assert_allclose(q["mean"][4:]*2, p["mean"][4:], rtol=1e-6)
        np.testing.assert_allclose(q["variance"][4:]*4, p["variance"][4:], rtol=1e-6)

    def test_more_mutations_change_the_inferred_clock(self):
        data = copy.deepcopy(self.data)
        # Add a nearby second independent site for every original event.
        data["mutations"] += [dict(position=m["position"]+1e-5, node=m["node"]) for m in self.data["mutations"]]
        data["mutations"].sort(key=lambda m: m["position"])
        _, fit = infer_dates(build_undated(data), RATE)
        self.assertGreater(fit.node_posteriors()["mean"][6], 1.5*self.fit.node_posteriors()["mean"][6])

    def test_estimates_are_not_replayed_truth(self):
        self.assertFalse(np.allclose(self.fit.node_posteriors()["mean"][4:], self.data["true_node_times"][4:]))

    def test_invalid_clock(self):
        for rate in (0, -1, float("nan"), float("inf"), True):
            with self.subTest(rate=rate), self.assertRaises(ValueError):
                infer_dates(self.undated, rate)

    def test_checked_output(self):
        check_sample(build_report(self.data), json.loads(Path(__file__).with_name("sample_summary.json").read_text()))
        with self.assertRaises(AssertionError):
            check_sample({"age": 2.0}, {"age": 1.0})


if __name__ == "__main__":
    unittest.main()
