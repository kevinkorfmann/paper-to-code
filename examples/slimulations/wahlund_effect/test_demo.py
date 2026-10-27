import copy
from fractions import Fraction as F
import json
from math import comb, fsum, sqrt
from pathlib import Path
import subprocess
import sys
import unittest

from examples.slimulations.wahlund_effect.demo import (
    SCENARIOS, SCRIPT, build_report, founder_counts, metrics,
    one_generation_expectation, parse_counts, run_counts, summarize,
)


def binomial_distribution(n, p):
    return [(k, F(comb(n, k))*p**k*(1-p)**(n-k)) for k in range(n+1)]


class WahlundTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = build_report()

    def test_exact_founder_counts(self):
        self.assertEqual(founder_counts(100, .2), (100, 4, 32))
        self.assertEqual(founder_counts(100, .8), (100, 64, 32))
        self.assertEqual(founder_counts(300, .8), (300, 192, 96))

    def test_founder_variance_identity_with_exact_rationals(self):
        groups = [founder_counts(100, .2), founder_counts(300, .8)]
        result = metrics(groups)
        pbar = F(1, 4)*F(1, 5)+F(3, 4)*F(4, 5)
        hs = F(8, 25)
        ht = 2*pbar*(1-pbar)
        variance = F(1, 4)*(F(1, 5)-pbar)**2+F(3, 4)*(F(4, 5)-pbar)**2
        self.assertEqual(ht-hs, 2*variance)
        self.assertEqual(ht-hs, F(27, 200))
        self.assertAlmostEqual(result["pooled_gap"], float(ht-hs), places=14)

    def test_realized_identity_including_within_group_residual(self):
        # Observed genotypes need not be in HWE for this decomposition to hold.
        result = metrics([(30, 4, 7), (70, 40, 9)])
        self.assertAlmostEqual(result["pooled_gap"], result["twice_between_variance"]+result["within_residual"], places=14)
        self.assertNotAlmostEqual(result["within_residual"], 0)

    def test_merely_pooling_counts_preserves_observed_heterozygosity(self):
        first, second = founder_counts(100, .2), founder_counts(100, .8)
        separate = metrics([first, second])
        pooled = metrics([(200, first[1]+second[1], first[2]+second[2])])
        for key in ("p", "H_observed", "H_pooled", "pooled_gap"):
            self.assertEqual(separate[key], pooled[key])
        self.assertAlmostEqual(separate["pooled_gap"], .18)
        self.assertEqual(pooled["twice_between_variance"], 0)
        self.assertAlmostEqual(pooled["within_residual"], .18)

    def test_unequal_sizes_are_individual_weighted(self):
        result = metrics([founder_counts(100, .2), founder_counts(300, .8)])
        self.assertAlmostEqual(result["p"], .65)
        self.assertAlmostEqual(result["H_pooled"], .455)
        self.assertNotEqual(result["p"], .5)

    def test_population_swap_and_allele_relabeling(self):
        groups = [(30, 4, 7), (70, 40, 9)]
        original = metrics(groups)
        swapped = metrics(groups[::-1])
        flipped = metrics([(n, n-AA-het, het) for n, AA, het in groups])
        for key, value in original.items():
            self.assertAlmostEqual(value, swapped[key], places=14)
            self.assertAlmostEqual(1-value if key == "p" else value, flipped[key], places=14)

    def test_one_step_expectation_from_independent_binomial_enumeration(self):
        n1, n2, p1, p2 = 4, 8, F(1, 2), F(1)
        total = n1+n2
        pbar = (n1*p1+n2*p2)/total
        for pooled in (False, True):
            expected = one_generation_expectation(n1, n2, float(p1), float(p2), pooled)
            if pooled:
                distribution = binomial_distribution(2*total, pbar)
            else:
                distribution = [(a+b, pa*pb) for a, pa in binomial_distribution(2*n1, p1)
                                for b, pb in binomial_distribution(2*n2, p2)]
            exact_ht = sum(prob*2*F(k, 2*total)*(1-F(k, 2*total)) for k, prob in distribution)
            if pooled:
                exact_h = sum(prob*F(k, total) for k, prob in binomial_distribution(total, 2*pbar*(1-pbar)))
            else:
                exact_h = sum(prob*F(k, total) for n, p in ((n1, p1), (n2, p2))
                              for k, prob in binomial_distribution(n, 2*p*(1-p)))
            self.assertAlmostEqual(expected["H_observed"], float(exact_h), places=14)
            self.assertAlmostEqual(expected["H_pooled"], float(exact_ht), places=14)
            self.assertAlmostEqual(expected["pooled_gap"], float(exact_ht-exact_h), places=14)

    def test_finite_population_correction_and_negative_expected_gap(self):
        separate = one_generation_expectation()
        common = one_generation_expectation(pooled=True)
        self.assertAlmostEqual(separate["H_pooled"], .4992)
        self.assertAlmostEqual(separate["pooled_gap"], .1792)
        self.assertAlmostEqual(common["H_pooled"], .49875)
        self.assertAlmostEqual(common["pooled_gap"], -.00125)

    def test_actual_engine_initialization_and_zero_duration(self):
        for _, n1, n2, p1, p2 in SCENARIOS:
            first, second = founder_counts(n1, p1), founder_counts(n2, p2)
            for pooled in (False, True):
                expected = ((n1+n2, first[1]+second[1], first[2]+second[2]),) if pooled else (first, second)
                actual = run_counts(n1=n1, n2=n2, p1=p1, p2=p2, pooled=pooled, replicates=3, generations=0)
                self.assertEqual(actual, {0: [expected]*3})

    def test_all_treatments_against_exact_one_generation_moments(self):
        # Theoretical SEs use Bernoulli cumulants, not measured sample SEs.
        for t in self.report["treatments"]:
            n1, n2 = t["founder_group_sizes"]
            p1, p2 = t["founder_frequencies"]
            total = n1+n2
            pbar = (n1*p1+n2*p2)/total
            populations = [(total, pbar)] if t["mating"] == "common_pool" else [(n1, p1), (n2, p2)]
            m = 2*total
            k2 = fsum(2*n*p*(1-p) for n, p in populations)
            k3 = fsum(2*n*p*(1-p)*(1-2*p) for n, p in populations)
            k4 = fsum(2*n*p*(1-p)*(1-6*p*(1-p)) for n, p in populations)
            v, mu3, mu4 = k2/m**2, k3/m**3, (k4+3*k2*k2)/m**4
            a = 2*(1-2*pbar)
            hvar = fsum(n*(2*p*(1-p))*(1-2*p*(1-p)) for n, p in populations)/total**2
            variances = {"p": v, "H_observed": hvar,
                         "H_pooled": max(0, a*a*v-4*a*mu3+4*(mu4-v*v))}
            observed = t["checkpoints"][-1]["means"]
            for key, variance in variances.items():
                with self.subTest(scenario=t["scenario"], mating=t["mating"], metric=key):
                    tolerance = 6*sqrt(variance/self.report["replicates_per_treatment"])+1e-9
                    self.assertLessEqual(abs(observed[key]-t["one_generation_expectation"][key]), tolerance)

    def test_mating_changes_genotypes_but_pooling_founders_does_not(self):
        separate, common = self.report["treatments"][:2]
        self.assertEqual(separate["checkpoints"][0]["means"]["H_observed"], .32)
        self.assertEqual(common["checkpoints"][0]["means"]["H_observed"], .32)
        self.assertLess(separate["checkpoints"][1]["means"]["H_observed"], .34)
        self.assertGreater(common["checkpoints"][1]["means"]["H_observed"], .48)

    def test_equal_frequency_and_opposite_fixation_controls(self):
        for t in self.report["treatments"]:
            row = t["checkpoints"][1]["means"]
            if t["scenario"] == "equal_frequency_control":
                self.assertLess(abs(row["pooled_gap"]), .02)
            if t["scenario"] == "opposite_fixation_control" and t["mating"] == "separate_demes":
                self.assertEqual(row["H_observed"], 0)
                self.assertEqual(row["pooled_gap"], .5)

    def test_monomorphic_boundaries_are_absorbing(self):
        for p in (0, 1):
            for pooled in (False, True):
                counts = run_counts(n1=4, n2=4, p1=p, p2=p, pooled=pooled, replicates=2)
                for groups in counts[1]:
                    result = metrics(groups)
                    self.assertEqual(result["p"], p)
                    self.assertEqual(result["H_observed"], 0)
                    self.assertEqual(result["pooled_gap"], 0)

    def test_seed_repeatability_and_variation(self):
        args = dict(replicates=8)
        first = run_counts(seed=401, **args)
        self.assertEqual(first, run_counts(seed=401, **args))
        self.assertNotEqual(first, run_counts(seed=402, **args))

    def test_summary_and_metrics_preserve_inputs_and_report_uncertainty(self):
        counts = {0: [((4, 1, 2), (4, 1, 2)), ((4, 0, 0), (4, 4, 0))]}
        original = copy.deepcopy(counts)
        row = summarize(counts)[0]
        self.assertEqual(row["means"]["H_observed"], .25)
        self.assertEqual(row["monte_carlo_se"]["H_observed"], .25)
        self.assertEqual(counts, original)
        self.assertIsNone(summarize({0: [counts[0][0]]})[0]["monte_carlo_se"]["H_observed"])

    def test_invalid_inputs_before_engine_execution(self):
        for patch in [dict(n1=1), dict(n1=26), dict(p1=.3, n1=25), dict(p1=True), dict(p1=float("nan")),
                      dict(pooled=1), dict(replicates=0), dict(generations=2), dict(seed=0)]:
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                run_counts(**patch, executable="nonexistent-executable")
        for groups in ([], [(4, 4, 1)], [(4, -1, 0)], [(4, 1.0, 2)]):
            with self.assertRaises(ValueError):
                metrics(groups)

    def test_malformed_engine_output_is_rejected(self):
        valid = 'DATA,0,0,0,100,4,32\nDATA,0,0,1,100,64,32\n'
        self.assertEqual(len(parse_counts(valid, 100, 100, False, 1, 0)[0]), 1)
        for bad in (valid+valid, '', valid.replace(',100,4,32', ',101,4,32'),
                    valid.replace(',100,4,32', ',100,90,32'), valid.splitlines()[0],
                    valid.replace('DATA,0', 'DATA,2'), 'DATA,0,0\n'):
            with self.assertRaises(ValueError):
                parse_counts(bad, 100, 100, False, 1, 0)

    def test_direct_eidos_validation(self):
        for definition in ('POOLED=2', 'AA1=1000', 'GENERATIONS=2'):
            result = subprocess.run(['slim', '-d', definition, str(SCRIPT)], capture_output=True, text=True, timeout=10)
            self.assertNotEqual(result.returncode, 0)

    def test_checked_sample(self):
        expected = json.loads(Path(__file__).with_name('sample_summary.json').read_text())
        self.assertEqual(self.report, expected)

    def test_cli_and_invalid_replicates(self):
        command = [sys.executable, '-m', 'examples.slimulations.wahlund_effect.demo']
        output = subprocess.run(command+['--replicates', '2'], capture_output=True, text=True, check=True, timeout=30)
        self.assertEqual(len(json.loads(output.stdout)['treatments']), 8)
        bad = subprocess.run(command+['--replicates', '0'], capture_output=True, text=True, timeout=10)
        self.assertNotEqual(bad.returncode, 0)


if __name__ == '__main__':
    unittest.main()
