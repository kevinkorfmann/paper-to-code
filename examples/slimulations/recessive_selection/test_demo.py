"""Exact Mendelian oracles, engine fitness audits, and finite sampling checks."""

from fractions import Fraction
import json
import math
import statistics
import subprocess
import sys
import unittest

from examples.slimulations.recessive_selection import demo


class RecessiveSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = demo.build_report()

    def test_fitnesses(self):
        self.assertEqual(demo.fitnesses("neutral"), (1, 1, 1))
        self.assertEqual(demo.fitnesses("recessive"), (1, 1, Fraction(1, 5)))
        self.assertEqual(demo.fitnesses("additive"), (1, Fraction(3, 5), Fraction(1, 5)))
        self.assertEqual(demo.fitnesses("dominant"), (1, Fraction(1, 5), Fraction(1, 5)))

    def test_founder_counts_and_engine_fitness_audits(self):
        for mode in demo.MODES:
            for p, expected in ((.05, (400, 1, 38)), (.5, (400, 100, 200))):
                self.assertEqual(demo.founder_counts(400, p), expected)
                n, AA, het = expected
                self.assertEqual(demo.run_counts(mode, n=n, AA=AA, het=het, replicates=2, generations=0), {0: [(AA, het)]*2})

    def test_exact_mendelian_parent_enumeration(self):
        # Select two independent parental genotypes, then enumerate both gametes.
        for n, AA, het in ((400, 1, 38), (400, 100, 200), (400, 0, 40), (25, 3, 7)):
            for mode in demo.MODES:
                counts = (n-AA-het, het, AA)
                weights = [count*fitness for count, fitness in zip(counts, demo.fitnesses(mode))]
                parents = [weight/sum(weights) for weight in weights]
                offspring = [Fraction(0)]*3
                for first, pa in enumerate(parents):
                    for second, pb in enumerate(parents):
                        for a in (0, 1):
                            for b in (0, 1):
                                qa = Fraction(first, 2) if a else 1-Fraction(first, 2)
                                qb = Fraction(second, 2) if b else 1-Fraction(second, 2)
                                offspring[a+b] += pa*pb*qa*qb
                actual = demo.exact_prediction(n, AA, het, mode)
                self.assertEqual(sum(offspring), 1)
                self.assertEqual(actual['gamete_frequency'], offspring[2]+offspring[1]/2)
                self.assertEqual(actual['offspring_AA_fraction'], offspring[2])
                self.assertEqual(actual['offspring_heterozygote_fraction'], offspring[1])

    def test_binomial_mean_variance_by_exact_enumeration(self):
        for mode in demo.MODES:
            predicted = demo.exact_prediction(4, 1, 2, mode)
            q = predicted['gamete_frequency']
            probabilities = [math.comb(8, k)*q**k*(1-q)**(8-k) for k in range(9)]
            self.assertEqual(sum(probabilities), 1)
            self.assertEqual(sum(Fraction(k, 8)*p for k, p in enumerate(probabilities)), q)
            variance = sum((Fraction(k, 8)-q)**2*p for k, p in enumerate(probabilities))
            self.assertEqual(variance, predicted['offspring_frequency_variance'])

    def test_recessive_frequency_change_formula(self):
        for frequency in (.05, .1, .2, .5, .8, 1):
            p = Fraction(str(frequency))
            n, AA, het = demo.founder_counts(400, frequency)
            q = demo.exact_prediction(n, AA, het, 'recessive')['gamete_frequency']
            cost = Fraction(4, 5)
            self.assertEqual(q-p, -cost*p*p*(1-p)/(1-cost*p*p))

    def test_rare_recessive_change_is_quadratic(self):
        cost = Fraction(4, 5)
        for p in (Fraction(1, 100), Fraction(1, 1000), Fraction(1, 10000)):
            decrement = cost*p*p*(1-p)/(1-cost*p*p)
            self.assertLess(abs(float(decrement/(cost*p*p))-1), 1.01*float(p))
        # The ratio of decrement to p vanishes as p approaches zero.
        self.assertLess(float(cost*Fraction(1, 10000)), 0.0001)

    def test_neutral_expectation_for_arbitrary_genotype_composition(self):
        for n, AA, het in ((400, 1, 38), (400, 0, 40), (20, 7, 6)):
            self.assertEqual(demo.exact_prediction(n, AA, het, 'neutral')['gamete_frequency'], Fraction(2*AA+het, 2*n))

    def test_all_heterozygote_carriers_remove_first_generation_selection(self):
        predicted = demo.exact_prediction(400, 0, 40, 'recessive')
        self.assertEqual(predicted['gamete_frequency'], Fraction(1, 20))
        self.assertEqual(predicted['mean_parent_fitness'], 1)
        self.assertLess(demo.exact_prediction(400, 1, 38, 'recessive')['gamete_frequency'], Fraction(1, 20))

    def test_first_generation_mean_matches_exact_expectation(self):
        for t in self.report['treatments']:
            q = float(Fraction(t['exact_expected_frequency_fraction']))
            se = math.sqrt(q*(1-q)/(800*512))
            self.assertLess(abs(t['observed_after_one_generation']['mean_frequency']-q), 6*se)

    def test_first_generation_variance_matches_binomial(self):
        for t in self.report['treatments']:
            q = float(Fraction(t['exact_expected_frequency_fraction']))
            variance = q*(1-q)/800
            fourth = 3*variance**2+q*(1-q)*(1-6*q*(1-q))/800**3
            variance_se = math.sqrt((fourth-(509/511)*variance**2)/512)
            self.assertLess(abs(t['observed_after_one_generation']['sample_frequency_variance']-variance), 6*variance_se+1e-9)

    def test_offspring_genotype_frequencies(self):
        for t in self.report['treatments']:
            q = float(Fraction(t['exact_expected_frequency_fraction']))
            for field, expected in [('mean_AA_fraction', q*q), ('mean_heterozygote_fraction', 2*q*(1-q))]:
                se = math.sqrt(expected*(1-expected)/(400*512))
                self.assertLess(abs(t['observed_after_one_generation'][field]-expected), 6*se)

    def test_rare_selection_order_and_common_recessive_response(self):
        rows = {(t['initial_frequency'], t['mode']): t['observed_after_one_generation'] for t in self.report['treatments'] if t['founders']=='hardy_weinberg'}
        self.assertGreater(rows[.05, 'recessive']['mean_frequency'], rows[.05, 'additive']['mean_frequency']+.01)
        self.assertGreater(rows[.05, 'additive']['mean_frequency'], rows[.05, 'dominant']['mean_frequency']+.01)
        self.assertGreater((.5-rows[.5, 'recessive']['mean_frequency'])/.5, 0.2)
        self.assertLess((.05-rows[.05, 'recessive']['mean_frequency'])/.05, 0.06)

    def test_some_deleterious_alleles_increase_by_chance(self):
        row = self.report['treatments'][1]['observed_after_one_generation']
        self.assertGreater(row['frequency_increased_replicates'], 0)
        self.assertLess(row['frequency_increased_replicates'], 512)

    def test_absorbing_boundaries(self):
        for mode in demo.MODES:
            for AA in (0, 20):
                counts = demo.run_counts(mode, n=20, AA=AA, het=0, replicates=2)
                self.assertEqual(counts, {0: [(AA, 0)]*2, 1: [(AA, 0)]*2})

    def test_summary_includes_absorbed_populations(self):
        rows = [(0, 0), (20, 0), (0, 20)]
        result = demo.summarize(rows, 20, .5)
        self.assertEqual(result['mean_frequency'], .5)
        self.assertEqual(result['lost_replicates'], 1)
        self.assertEqual(result['fixed_replicates'], 1)
        self.assertEqual(result['monte_carlo_se'], round(statistics.stdev([0, 1, .5])/math.sqrt(3), 9))

    def test_parser_rejects_incomplete_duplicate_and_wrong_fitness(self):
        valid = 'FIT,0,1\nFIT,1,1\nFIT,2,0.2\nDATA,0,0,400,1,38\n'
        for text in (valid+'DATA,0,0,400,1,38\n', valid.replace('FIT,2,0.2', 'FIT,2,0.3'), valid.replace('400,1,38', '400,2,38'), ''):
            with self.assertRaises(ValueError):
                demo.parse_output(text, 400, 1, 38, 'recessive', 1, 0)

    def test_invalid_wrapper_inputs(self):
        for kwargs in ({'n': True}, {'AA': 401}, {'het': -1}, {'AA': 399, 'het': 2}, {'replicates': 0}, {'generations': 2}, {'seed': 0}, {'mode': 'unknown'}):
            with self.subTest(kwargs), self.assertRaises(ValueError):
                demo.run_counts(**kwargs)

    def test_script_guards_direct_invalid_inputs(self):
        for definition in ('N=1', 'MODE=4', 'AA=400', 'GENERATIONS=2'):
            proc = subprocess.run(['slim', '-s', '1', '-d', definition, str(demo.SCRIPT)], capture_output=True, text=True, timeout=30)
            self.assertNotEqual(proc.returncode, 0)

    def test_seed_reproducibility(self):
        settings = {'replicates': 4, 'seed': 811}
        first = demo.run_counts(**settings)
        self.assertEqual(first, demo.run_counts(**settings))
        self.assertNotEqual(first[1], demo.run_counts(replicates=4, seed=812)[1])

    def test_one_replicate_has_no_estimated_standard_error(self):
        result = demo.summarize([(1, 38)], 400, .05)
        self.assertIsNone(result['monte_carlo_se'])
        self.assertIsNone(result['sample_frequency_variance'])

    def test_checked_output(self):
        self.assertEqual(self.report, json.loads(demo.SAMPLE.read_text()))

    def test_cli_checked_output_and_invalid_option(self):
        command = [sys.executable, '-m', 'examples.slimulations.recessive_selection.demo']
        actual = subprocess.run([*command, '--check-sample'], capture_output=True, text=True, check=True, timeout=60)
        self.assertEqual(actual.stdout, demo.SAMPLE.read_text())
        bad = subprocess.run([*command, '--replicates', '0'], capture_output=True, text=True)
        self.assertNotEqual(bad.returncode, 0)


if __name__ == '__main__':
    unittest.main()
