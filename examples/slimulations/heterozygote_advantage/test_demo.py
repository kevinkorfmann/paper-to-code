import json
from fractions import Fraction
from math import log, sqrt
from pathlib import Path
import statistics
import subprocess
import sys
import unittest

from examples.slimulations.heterozygote_advantage.demo import (
    MODES, SCRIPT, build_report, deterministic_path, fitnesses,
    gamete_frequency, initial_genotypes, run_counts, summarize,
)


class SelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = build_report()

    def test_fitness_convention(self):
        self.assertEqual(fitnesses("neutral"), (1, 1, 1))
        self.assertEqual(fitnesses("additive"), (1, 1.05, 1.1))
        self.assertEqual(fitnesses("overdominant"), (1, 1.2, 1.1))

    def test_recursion_against_exact_mendelian_parent_enumeration(self):
        # Enumerate weighted parental genotypes and gametes with exact rationals.
        p = Fraction(1, 5)
        genotypes = [(1-p)**2, 2*p*(1-p), p*p]
        for mode, fitness in [("neutral", [1, 1, 1]),
                              ("additive", [1, Fraction(21, 20), Fraction(11, 10)]),
                              ("overdominant", [1, Fraction(6, 5), Fraction(11, 10)])]:
            weighted = [g*w for g, w in zip(genotypes, fitness)]
            parents = [x/sum(weighted) for x in weighted]
            offspring = [Fraction(0)]*3
            for first, a in enumerate(parents):
                for second, b in enumerate(parents):
                    for x in (0, 1):
                        for y in (0, 1):
                            px = Fraction(first, 2) if x else 1-Fraction(first, 2)
                            py = Fraction(second, 2) if y else 1-Fraction(second, 2)
                            offspring[x+y] += a*b*px*py
            exact = offspring[2]+offspring[1]/2
            actual = deterministic_path(float(p), mode, 1)[-1]
            self.assertAlmostEqual(actual, float(exact), places=14)
            self.assertEqual(sum(offspring), 1)

    def test_exact_equilibrium_and_attracting_direction(self):
        wa, wh, wA = map(Fraction, ("1", "1.2", "1.1"))
        equilibrium = (wh-wa)/(2*wh-wa-wA)
        self.assertEqual(equilibrium, Fraction(2, 3))
        self.assertAlmostEqual(deterministic_path(float(equilibrium), "overdominant", 100)[-1], 2/3, places=14)
        for p in (0.05, 0.2, 0.5, 0.8, 0.95):
            next_p = deterministic_path(p, "overdominant", 1)[-1]
            self.assertLess(abs(next_p-2/3), abs(p-2/3))
        step = 1e-5
        slope = (deterministic_path(2/3+step, "overdominant", 1)[-1]
                 - deterministic_path(2/3-step, "overdominant", 1)[-1])/(2*step)
        self.assertTrue(0 < slope < 1)

    def test_deterministic_neutral_and_additive_controls(self):
        for p in (0.2, 0.8):
            self.assertTrue(all(abs(x-p) < 1e-14 for x in deterministic_path(p, "neutral", 80)))
            additive = deterministic_path(p, "additive", 80)
            self.assertTrue(all(a < b for a, b in zip(additive, additive[1:])))
            self.assertGreater(additive[-1], 0.9)

    def test_conditional_expectation_depends_on_genotypes_not_only_p(self):
        # Both parental populations have p=.5, but different heterozygosity.
        all_hetero = gamete_frequency(0, 100, 0, "overdominant")
        all_homo = gamete_frequency(50, 0, 50, "overdominant")
        self.assertEqual(all_hetero, 0.5)
        self.assertAlmostEqual(all_homo, 11/21)
        self.assertNotEqual(all_hetero, all_homo)

    def test_exact_initial_genotypes_and_engine_fitness_audit(self):
        for mode in MODES:
            for p, expected in ((0.2, (8, 64)), (0.8, (128, 64))):
                self.assertEqual(initial_genotypes(200, p), expected)
                self.assertEqual(run_counts(mode, p=p, generations=0, replicates=3), {0: [expected]*3})

    def test_first_generation_binomial_mean_variance_and_genotypes(self):
        n, replicates = 25, 1024
        for i, mode in enumerate(MODES):
            for j, p in enumerate((0.2, 0.8)):
                rows = run_counts(mode, n=n, p=p, generations=1, replicates=replicates, seed=811+2*i+j)[1]
                q = deterministic_path(p, mode, 1)[1]
                frequencies = [(2*AA+het)/(2*n) for AA, het in rows]
                variance = q*(1-q)/(2*n)
                self.assertLess(abs(statistics.mean(frequencies)-q), 6*sqrt(variance/replicates))
                m, v = 2*n, (2*n)*q*(1-q)
                fourth = (3*v*v+v*(1-6*q*(1-q)))/(m**4)
                observed_variance_about_truth = statistics.mean((x-q)**2 for x in frequencies)
                self.assertLess(abs(observed_variance_about_truth-variance), 6*sqrt((fourth-variance**2)/replicates))
                for observed, expected in ((sum(AA for AA, _ in rows)/(n*replicates), q*q),
                                           (sum(het for _, het in rows)/(n*replicates), 2*q*(1-q))):
                    self.assertLess(abs(observed-expected), 6*sqrt(expected*(1-expected)/(n*replicates)))

    def test_neutral_ensemble_mean_conservative_bound(self):
        treatments = [t for t in self.report["treatments"] if t["mode"] == "neutral"]
        comparisons = sum(len(t["checkpoints"]) for t in treatments)
        tolerance = sqrt(log(2*comparisons/1e-6)/(2*self.report["replicates_per_treatment"]))
        for t in treatments:
            for row in t["checkpoints"]:
                self.assertLess(abs(row["mean_p"]-t["p_initial"]), tolerance)

    def test_finite_runs_show_distinct_selection_directions(self):
        rows = {(t["mode"], t["p_initial"]): t["checkpoints"][-1] for t in self.report["treatments"]}
        for p in (0.2, 0.8):
            self.assertLess(abs(rows["overdominant", p]["mean_p"]-2/3), .05)
            self.assertGreater(rows["additive", p]["mean_p"], rows["overdominant", p]["mean_p"]+.15)
            self.assertGreater(rows["overdominant", p]["monte_carlo_se"], 0)

    def test_finite_overdominance_can_absorb(self):
        for t in self.report["small_population_control_200_generations"]:
            row = t["checkpoints"][-1]
            self.assertGreater(row["lost"], 0)
            self.assertGreater(row["fixed"], 0)
            self.assertEqual(row["lost"]+row["fixed"]+row["segregating"], 128)

    def test_summaries_include_absorbed_populations(self):
        counts = {0: [(1, 8)]*3, 1: [(0, 0), (25, 0), (1, 8)]}
        row = summarize(counts, 25, .2, "overdominant")[-1]
        self.assertEqual(row["mean_p"], .4)
        self.assertEqual((row["lost"], row["fixed"], row["segregating"]), (1, 1, 1))
        self.assertEqual(row["monte_carlo_se"], round(statistics.stdev([0, 1, .2])/sqrt(3), 9))

    def test_absorbing_boundaries_all_modes(self):
        for mode in MODES:
            for p, genotype in ((0, (0, 0)), (1, (25, 0))):
                for rows in run_counts(mode, n=25, p=p, replicates=3, generations=5).values():
                    self.assertEqual(rows, [genotype]*3)

    def test_seed_repeatability_and_variation(self):
        args = dict(n=25, replicates=8, generations=4)
        first = run_counts(seed=401, **args)
        self.assertEqual(first, run_counts(seed=401, **args))
        self.assertNotEqual(first, run_counts(seed=402, **args))

    def test_invalid_inputs_before_engine_execution(self):
        for patch in [dict(mode="bad"), dict(n=1), dict(n=26), dict(p=.3), dict(p=True),
                      dict(replicates=0), dict(generations=-1), dict(generations=301), dict(seed=0)]:
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                run_counts(**patch, executable="nonexistent-executable")
        for bad in (-.1, 1.1, True, float("nan")):
            with self.assertRaises(ValueError):
                deterministic_path(bad, "neutral", 1)
        with self.assertRaises(ValueError):
            gamete_frequency(0, 0, 0, "neutral")

    def test_direct_eidos_validation(self):
        for definition in ("MODE=3", "INITIAL_HET=300", "GENERATIONS=-1"):
            result = subprocess.run(["slim", "-d", definition, str(SCRIPT)], capture_output=True, text=True, timeout=10)
            self.assertNotEqual(result.returncode, 0)

    def test_checked_sample(self):
        expected = json.loads(Path(__file__).with_name("sample_summary.json").read_text())
        self.assertEqual(json.loads(json.dumps(self.report)), expected)

    def test_cli(self):
        command = [sys.executable, "-m", "examples.slimulations.heterozygote_advantage.demo"]
        result = subprocess.run(command+["--replicates", "2", "--generations", "0"],
                                capture_output=True, text=True, check=True, timeout=30)
        report = json.loads(result.stdout)
        self.assertTrue(all(len(t["checkpoints"]) == 1 for t in report["treatments"]))
        self.assertEqual(report["small_population_control_200_generations"][0]["checkpoints"][-1]["generations_elapsed"], 200)
        bad = subprocess.run(command+["--replicates", "0"], capture_output=True, text=True, timeout=10)
        self.assertNotEqual(bad.returncode, 0)


if __name__ == "__main__":
    unittest.main()
