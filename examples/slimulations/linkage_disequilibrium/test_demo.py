import itertools
import json
from fractions import Fraction
from math import log, sqrt
from pathlib import Path
import statistics
import subprocess
import sys
import unittest
from unittest.mock import patch

from examples.slimulations.linkage_disequilibrium.demo import (
    FOUNDER_MODES, SCRIPT, build_report, conditional_gametes, deterministic_path,
    initial_counts, linkage_disequilibrium, parse_counts, run_counts, summarize,
)

HAPS = ((1, 1), (1, 0), (0, 1), (0, 0))


def enumerate_meiosis(diplotypes, c):
    """Independent exact-rational oracle: choose homolog and crossover outcome."""
    result = [Fraction(0)]*4
    total = sum(diplotypes.values())
    for (first, second), count in diplotypes.items():
        for left, right in ((first, second), (second, first)):
            for crossover, probability in ((False, 1-c), (True, c)):
                gamete = (left[0], right[1] if crossover else left[1])
                result[HAPS.index(gamete)] += count/total*probability/2
    return result


class LinkageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = build_report()

    def test_signed_D_and_equivalent_determinant(self):
        for f in ((.5, 0, 0, .5), (0, .5, .5, 0), (.1, .2, .3, .4)):
            self.assertAlmostEqual(linkage_disequilibrium(f), f[0]*f[3]-f[1]*f[2])
        self.assertEqual(linkage_disequilibrium((.5, 0, 0, .5)), .25)
        self.assertEqual(linkage_disequilibrium((0, .5, .5, 0)), -.25)

    def test_conditional_gametes_against_exact_diplotype_enumeration(self):
        diplotypes = {(a, b): Fraction(1+i+2*j) for i, a in enumerate(HAPS) for j, b in enumerate(HAPS)}
        counts = [sum(int(w)*(int(a == h)+int(b == h)) for (a, b), w in diplotypes.items()) for h in HAPS]
        coupling = sum(int(w) for (a, b), w in diplotypes.items() if {a, b} == {HAPS[0], HAPS[3]})
        repulsion = sum(int(w) for (a, b), w in diplotypes.items() if {a, b} == {HAPS[1], HAPS[2]})
        for c in (Fraction(0), Fraction(1, 10), Fraction(1, 2)):
            expected = enumerate_meiosis(diplotypes, c)
            self.assertEqual(sum(expected), 1)
            for a, b in zip(conditional_gametes((*counts, coupling, repulsion), float(c)), expected):
                self.assertAlmostEqual(a, float(b), places=14)

    def test_random_pairing_recursion_against_exact_meiosis(self):
        f = tuple(map(Fraction, (".4", ".1", ".2", ".3")))
        diplotypes = {(a, b): f[i]*f[j] for i, a in enumerate(HAPS) for j, b in enumerate(HAPS)}
        for c in (Fraction(0), Fraction(1, 10), Fraction(1, 2)):
            expected = enumerate_meiosis(diplotypes, c)
            actual = deterministic_path(tuple(map(float, f)), float(c), 1)[1]
            for a, b in zip(actual, expected):
                self.assertAlmostEqual(a, float(b), places=14)

    def test_geometric_decay_preserves_marginal_allele_frequencies(self):
        for initial in ((.5, 0, 0, .5), (0, .5, .5, 0), (.4, .1, .2, .3)):
            for c in (0, .1, .5):
                for t, f in enumerate(deterministic_path(initial, c, 30)):
                    self.assertAlmostEqual(sum(f), 1, places=14)
                    self.assertGreaterEqual(min(f), 0)
                    self.assertAlmostEqual(f[0]+f[1], initial[0]+initial[1], places=14)
                    self.assertAlmostEqual(f[0]+f[2], initial[0]+initial[2], places=14)
                    self.assertAlmostEqual(linkage_disequilibrium(f), linkage_disequilibrium(initial)*(1-c)**t, places=14)

    def test_linkage_equilibrium_and_no_recombination_controls(self):
        for f in ((.1, .2, .3, .4), (.5, 0, 0, .5)):
            self.assertEqual(deterministic_path(f, 0, 10), [f]*11)
        # Independent alleles with p(A)=.25 and p(B)=.5.
        f = (.125, .125, .375, .375)
        self.assertEqual(deterministic_path(f, .5, 10), [f]*11)

    def test_finite_sample_D_bias_by_exhaustive_enumeration(self):
        q = tuple(map(Fraction, (".4", ".1", ".2", ".3")))
        for copies in (2, 4):
            expected = Fraction(0)
            for draw in itertools.product(range(4), repeat=copies):
                probability = Fraction(1)
                for h in draw:
                    probability *= q[h]
                f = [Fraction(draw.count(h), copies) for h in range(4)]
                expected += probability*(f[0]-(f[0]+f[1])*(f[0]+f[2]))
            Dq = q[0]-(q[0]+q[1])*(q[0]+q[2])
            self.assertEqual(expected, (1-Fraction(1, copies))*Dq)

    def test_identical_haplotype_counts_do_not_determine_meiotic_output(self):
        random = initial_counts(1000, "random_pairing")
        double = initial_counts(1000, "all_double_heterozygotes")
        self.assertEqual(random[:4], double[:4])
        self.assertEqual(conditional_gametes(random, .5), (.375, .125, .125, .375))
        self.assertEqual(conditional_gametes(double, .5), (.25, .25, .25, .25))

    def test_actual_founder_counts_and_tick_zero(self):
        for founders in FOUNDER_MODES:
            for n in (4, 1000):
                expected = initial_counts(n, founders)
                self.assertEqual(run_counts(n=n, founders=founders, replicates=3, generations=0), {0: [expected]*3})

    def test_first_generation_haplotypes_with_simultaneous_hoeffding_bound(self):
        n, replicates, comparisons, alpha = 100, 512, 24, 1e-8
        # These founder populations are identical, so all 2*N*R gametes have
        # common probabilities q. Union bound covers 4 haplotypes x 6 runs.
        epsilon = sqrt(log(2*comparisons/alpha)/(2*2*n*replicates))
        for i, founders in enumerate(FOUNDER_MODES):
            for j, c in enumerate((0, .1, .5)):
                rows = run_counts(c, n=n, replicates=replicates, generations=1,
                                  founders=founders, seed=202610270+3*i+j)[1]
                q = conditional_gametes(initial_counts(n, founders), c)
                for h in range(4):
                    observed = sum(row[h] for row in rows)/(2*n*replicates)
                    self.assertLessEqual(abs(observed-q[h]), epsilon)
                # A separate six-MC-SE diagnostic, not a calibrated familywise test.
                Ds = [linkage_disequilibrium(tuple(x/(2*n) for x in row[:4])) for row in rows]
                expected_D = (1-1/(2*n))*linkage_disequilibrium(q)
                self.assertLess(abs(statistics.mean(Ds)-expected_D), 6*statistics.stdev(Ds)/sqrt(replicates))

    def test_conditional_oracle_with_stochastic_second_generation_parents(self):
        n, replicates = 64, 512
        # Conditional on the realized t=1 parents, every t=2 gamete is still an
        # independent bounded trial; the q differs across populations.
        epsilon = sqrt(log(2*12/1e-8)/(2*2*n*replicates))
        for j, c in enumerate((0, .1, .5)):
            counts = run_counts(c, n=n, replicates=replicates, generations=2, seed=202610280+j)
            predictions = [conditional_gametes(row, c) for row in counts[1]]
            for h in range(4):
                error = statistics.mean(child[h]/(2*n)-q[h] for child, q in zip(counts[2], predictions))
                self.assertLessEqual(abs(error), epsilon)

    def test_no_recombination_never_creates_missing_haplotypes(self):
        counts = run_counts(0, n=40, replicates=64, generations=20, seed=202610290)
        for rows in counts.values():
            self.assertTrue(all(row[1] == row[2] == row[5] == 0 for row in rows))

    def test_no_recombination_exact_finite_drift_expectation(self):
        # With only AB and ab, D=p*(1-p); its finite WF expectation is known.
        for row in self.report["treatments"][0]["checkpoints"][1:]:
            expected = .25*(1-1/2000)**row["generations_elapsed"]
            self.assertLess(abs(row["mean_D"]-expected), 6*row["monte_carlo_se_D"])

    def test_fixed_experiment_effects_and_alternative_founder_control(self):
        zero, linked, free, alternative = self.report["treatments"]
        self.assertGreater(zero["checkpoints"][-1]["mean_D"], .24)
        self.assertTrue(.08 < linked["checkpoints"][-1]["mean_D"] < .13)
        self.assertLess(abs(free["checkpoints"][-1]["mean_D"]), .01)
        self.assertTrue(.1 < free["checkpoints"][1]["mean_D"] < .15)
        self.assertLess(abs(alternative["checkpoints"][-1]["mean_D"]), .01)
        self.assertEqual(alternative["exact_expected_D_at_t1"], 0)
        self.assertIsNone(alternative["checkpoints"][-1]["infinite_random_pairing_D"])

    def test_average_within_population_D_is_not_D_of_pooled_haplotypes(self):
        # Two absorbed populations each have D=0; pooling invents D=.25.
        counts = {0: [(8, 0, 0, 0, 0, 0), (0, 0, 0, 8, 0, 0)]}
        row = summarize(counts, 4, 0, "random_pairing")[0]
        self.assertEqual(row["mean_D"], 0)
        self.assertEqual(row["mean_p_A"], .5)
        self.assertEqual(linkage_disequilibrium((.5, 0, 0, .5)), .25)
        single = summarize({0: [initial_counts(4, "random_pairing")]}, 4, 0, "random_pairing")[0]
        self.assertIsNone(single["monte_carlo_se_D"])

    def test_seed_repeatability_and_variation(self):
        kwargs = dict(n=40, replicates=8, generations=4)
        first = run_counts(seed=202610291, **kwargs)
        self.assertEqual(first, run_counts(seed=202610291, **kwargs))
        self.assertNotEqual(first, run_counts(seed=202610292, **kwargs))

    def test_invalid_inputs_before_engine_execution(self):
        invalid = [dict(c=-.1), dict(c=.51), dict(c=float("nan")), dict(c=True), dict(n=5),
                   dict(n=0), dict(n=4004), dict(founders="unknown"), dict(replicates=0),
                   dict(generations=-1), dict(generations=101), dict(seed=0)]
        for change in invalid:
            with self.subTest(change=change), self.assertRaises(ValueError):
                run_counts(**change, executable="nonexistent-executable")
        for f in ((1, 0, 0), (1, 1, 0, 0), (float("nan"), 0, 0, 1), (-.1, .1, .5, .5)):
            with self.assertRaises(ValueError):
                linkage_disequilibrium(f)
        with self.assertRaises(ValueError):
            conditional_gametes((0, 0, 0, 0, 0, 0), .5)
        with self.assertRaises(ValueError):
            conditional_gametes((4, 0, 0, 4, 5, 0), .5)

    def test_parser_rejects_missing_duplicate_or_impossible_counts(self):
        good = "DATA,0,0,4,0,0,4,2,0\n"
        self.assertEqual(parse_counts(good, 4, 1, 0), {0: [(4, 0, 0, 4, 2, 0)]})
        bad = ("", good+good, good.replace("0,0,4,0", "0,1,4,0"),
               "DATA,0,0,4,0,0,3,2,0\n", "DATA,0,0,4,0,0,4,5,0\n", "DATA,0,0,4\n")
        for output in bad:
            with self.subTest(output=output), self.assertRaises(ValueError):
                parse_counts(output, 4, 1, 0)

    def test_version_rejection_and_direct_eidos_validation(self):
        with patch("examples.slimulations.neutral_drift.demo.subprocess.run",
                   return_value=subprocess.CompletedProcess([], 0, "SLiM version 5.0", "")):
            with self.assertRaisesRegex(ValueError, "requires SLiM 4.0.1"):
                run_counts()
        for definition in ("N=5", "C=0.51", "FOUNDERS=2", "GENERATIONS=-1", "REPLICATES=0"):
            result = subprocess.run(["slim", "-d", definition, str(SCRIPT)], capture_output=True, text=True, timeout=10)
            self.assertNotEqual(result.returncode, 0)

    def test_checked_sample(self):
        expected = json.loads(Path(__file__).with_name("sample_summary.json").read_text())
        self.assertEqual(json.loads(json.dumps(self.report)), expected)

    def test_cli_and_fixed_control_duration(self):
        command = [sys.executable, "-m", "examples.slimulations.linkage_disequilibrium.demo"]
        result = subprocess.run(command+["--replicates", "2", "--generations", "0"],
                                capture_output=True, text=True, check=True, timeout=30)
        report = json.loads(result.stdout)
        self.assertTrue(all(len(t["checkpoints"]) == 1 for t in report["treatments"][:3]))
        self.assertEqual(report["treatments"][-1]["checkpoints"][-1]["generations_elapsed"], 1)
        bad = subprocess.run(command+["--replicates", "0"], capture_output=True, text=True, timeout=10)
        self.assertNotEqual(bad.returncode, 0)


if __name__ == "__main__":
    unittest.main()
