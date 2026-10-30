"""Independent finite Markov-chain enumeration and actual SLiM controls."""

from fractions import Fraction as F
import json
import math
import statistics
import subprocess
import sys
import unittest

from examples.slimulations.reversible_mutation import demo


def enumerate_chain(n, initial, u, v, generations):
    """Full rational transition matrix, independent of the moment recurrence."""
    size = 2*n
    transition = []
    for old in range(size+1):
        q = F(old, size)*(1-v)+F(size-old, size)*u
        transition.append([math.comb(size, k)*q**k*(1-q)**(size-k) for k in range(size+1)])
    distribution = [F(k == initial) for k in range(size+1)]
    result = [distribution]
    for _ in range(generations):
        distribution = [sum(distribution[j]*transition[j][k] for j in range(size+1))
                        for k in range(size+1)]
        result.append(distribution)
    return result


class MutationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = demo.build_report()

    def test_four_moments_against_full_rational_markov_chain(self):
        for u, v in ((F(0), F(0)), (F(1, 4), F(1, 4)), (F(1, 5), F(1, 10)),
                     (F(1), F(1)), (F(3, 4), F(1, 2))):
            for initial in (0, 2, 6):
                theory = demo.moments(3, initial, float(u), float(v), 5)
                for t, distribution in enumerate(enumerate_chain(3, initial, u, v, 5)):
                    self.assertEqual(sum(distribution), 1)
                    for r in range(5):
                        expected = sum(prob*F(k, 6)**r for k, prob in enumerate(distribution))
                        self.assertAlmostEqual(theory[t][r], float(expected), delta=2e-14)

    def test_diversity_variance_against_direct_distribution(self):
        distribution = enumerate_chain(3, 2, F(1, 5), F(1, 10), 4)[-1]
        values = [2*F(k, 6)*(1-F(k, 6)) for k in range(7)]
        mean = sum(p*h for p, h in zip(distribution, values))
        variance = sum(p*(h-mean)**2 for p, h in zip(distribution, values))
        row = demo.prediction(demo.moments(3, 2, .2, .1, 4)[-1])
        self.assertAlmostEqual(row['mean_gene_diversity'], float(mean), delta=1e-14)
        self.assertAlmostEqual(row['gene_diversity_variance'], float(variance), delta=1e-14)

    def test_mean_closed_form_and_mutation_direction(self):
        for u, v in ((.01, .01), (.02, .005), (.1, .7)):
            for n in (20, 80):
                for t, row in enumerate(demo.moments(n, 0, u, v, 80)):
                    self.assertAlmostEqual(row[1], u/(u+v)*(1-(1-u-v)**t), delta=1e-14)

    def test_no_mutation_recovers_exact_drift_diversity_decay(self):
        for n in (2, 20, 80):
            for t, raw in enumerate(demo.moments(n, n, 0, 0, 80)):
                row = demo.prediction(raw)
                self.assertAlmostEqual(row['mean_frequency'], .5, delta=1e-14)
                self.assertAlmostEqual(row['mean_gene_diversity'], .5*(1-1/(2*n))**t, delta=1e-14)

    def test_symmetric_stationary_and_finite_time_diversity(self):
        for n in (2, 20, 80):
            for mu in (.01, .5, .8):
                hstar = demo.symmetric_stationary_diversity(n, mu)
                b = 1-1/(2*n)
                self.assertAlmostEqual(hstar, b*(2*mu*(1-mu)+(1-2*mu)**2*hstar))
                for t, raw in enumerate(demo.moments(n, 0, mu, mu, 80)):
                    expected = hstar*(1-(b*(1-2*mu)**2)**t)
                    self.assertAlmostEqual(demo.prediction(raw)['mean_gene_diversity'], expected, delta=2e-14)
        self.assertLess(demo.symmetric_stationary_diversity(20, .01), demo.symmetric_stationary_diversity(80, .01))

    def test_diversity_of_mean_exceeds_mean_diversity_by_twice_variance(self):
        for raw in demo.moments(20, 0, .02, .005, 80):
            row = demo.prediction(raw)
            gap = 2*raw[1]*(1-raw[1])-row['mean_gene_diversity']
            self.assertAlmostEqual(gap, 2*row['frequency_variance'], delta=1e-14)

    def test_allele_complement_symmetry(self):
        left = demo.moments(20, 7, .02, .005, 80)
        right = demo.moments(20, 33, .005, .02, 80)
        for a, b in zip(left, right):
            self.assertAlmostEqual(a[1], 1-b[1], delta=2e-14)
            self.assertAlmostEqual(demo.prediction(a)['mean_gene_diversity'], demo.prediction(b)['mean_gene_diversity'], delta=2e-14)

    def test_actual_means_match_finite_population_theory(self):
        for treatment in self.report['treatments']:
            n, u, v = (treatment[k] for k in ('diploid_population_size', 'u_a_to_A', 'v_A_to_a'))
            raw = demo.moments(n, 0, u, v, 80)
            for checkpoint in treatment['checkpoints']:
                theory = demo.prediction(raw[checkpoint['generations_elapsed']])
                for mean, variance in (('mean_frequency', 'frequency_variance'), ('mean_gene_diversity', 'gene_diversity_variance')):
                    tolerance = 6*math.sqrt(theory[variance]/512)+1e-9
                    self.assertLessEqual(abs(checkpoint['observed'][mean]-theory[mean]), tolerance)

    def test_recovery_size_and_bias_contrasts(self):
        none, small, large, biased = [t['checkpoints'][-1]['observed'] for t in self.report['treatments']]
        self.assertEqual(none['zero_A_replicates'], 512)
        self.assertEqual(none['mean_gene_diversity'], 0)
        self.assertGreater(small['mean_gene_diversity'], .25)
        self.assertGreater(large['mean_gene_diversity'], small['mean_gene_diversity']+.08)
        self.assertGreater(biased['mean_frequency'], small['mean_frequency']+.2)
        self.assertGreater(small['zero_A_replicates']+small['all_A_replicates'], 0)

    def test_exact_founder_counts_and_generation_zero(self):
        for copies in (0, 1, 7, 16):
            self.assertEqual(demo.run_counts(n=8, initial_copies=copies, generations=0, replicates=2), {0: [copies]*2})

    def test_no_mutation_boundaries_are_absorbing(self):
        for copies in (0, 16):
            result = demo.run_counts(n=8, initial_copies=copies, u=0, v=0, generations=6, replicates=3)
            self.assertTrue(all(row == [copies]*3 for row in result.values()))

    def test_deterministic_forward_reverse_and_double_flip_guard(self):
        for u, v, initial, expected in ((1, 0, 0, 16), (0, 1, 16, 0)):
            rows = demo.run_counts(n=8, initial_copies=initial, u=u, v=v, generations=4, replicates=3)
            self.assertTrue(all(row == [expected]*3 for t, row in rows.items() if t))
        rows = demo.run_counts(n=8, initial_copies=0, u=1, v=1, generations=5, replicates=3)
        self.assertEqual(rows, {t: [16*(t % 2)]*3 for t in demo.checkpoints(5)})

    def test_mutation_can_leave_either_monomorphic_state(self):
        for initial in (0, 40):
            rows = demo.run_counts(n=20, initial_copies=initial, u=.5, v=.5, generations=2, replicates=16)
            self.assertTrue(all(0 < k < 40 for k in rows[1]))

    def test_parent_independent_reset_at_half_mutation(self):
        for initial in (0, 20, 40):
            rows = demo.run_counts(n=20, initial_copies=initial, u=.5, v=.5, generations=1, replicates=512, seed=111)
            values = [k/40 for k in rows[1]]
            self.assertLess(abs(statistics.mean(values)-.5), 6*math.sqrt(.25/40/512))
            # Independent binomial fourth moment gives the sample-variance SE.
            variance = .25/40
            fourth = 3*variance**2+(.25*(1-6*.25))/40**3
            se = math.sqrt((fourth-(509/511)*variance**2)/512)
            self.assertLess(abs(statistics.variance(values)-variance), 6*se)

    def test_one_generation_creation_probability_from_no_A(self):
        rows = demo.run_counts(n=20, initial_copies=0, u=.01, v=.6, generations=1, replicates=1024, seed=333)
        expected = 1-.99**40
        observed = sum(k > 0 for k in rows[1])/1024
        self.assertLess(abs(observed-expected), 6*math.sqrt(expected*(1-expected)/1024))

    def test_seed_reproducibility_and_different_seed(self):
        settings = dict(n=20, replicates=8, generations=4)
        first = demo.run_counts(seed=12, **settings)
        self.assertEqual(first, demo.run_counts(seed=12, **settings))
        self.assertNotEqual(first, demo.run_counts(seed=13, **settings))

    def test_all_replicates_including_boundaries_enter_summary(self):
        row = demo.summarize([0, 16, 8], 8)
        self.assertEqual(row['mean_frequency'], .5)
        self.assertEqual(row['mean_gene_diversity'], round(1/6, 9))
        self.assertEqual(row['zero_A_replicates'], 1)
        self.assertEqual(row['all_A_replicates'], 1)
        self.assertEqual(row['segregating_replicates'], 1)

    def test_single_replicate_has_no_monte_carlo_standard_error(self):
        row = demo.summarize([4], 8)
        self.assertIsNone(row['frequency_monte_carlo_se'])
        self.assertIsNone(row['diversity_monte_carlo_se'])

    def test_invalid_wrapper_and_stationary_inputs(self):
        for kwargs in ({'n': True}, {'n': 1}, {'initial_copies': 41}, {'initial_copies': -1},
                       {'u': True}, {'u': '0.1'}, {'u': float('nan')}, {'v': float('inf')},
                       {'u': -.1}, {'v': 1.1}, {'replicates': 0}, {'generations': 201}, {'seed': 0}):
            with self.subTest(kwargs), self.assertRaises(ValueError):
                demo.run_counts(**kwargs)
        for mu in (0, 1, -.1, True):
            with self.assertRaises(ValueError):
                demo.symmetric_stationary_diversity(20, mu)

    def test_invalid_direct_script_inputs(self):
        for definition in ('N=1', 'U=-0.1', 'V=1.1', 'INITIAL_COPIES=41', 'GENERATIONS=201', 'REPLICATES=0'):
            result = subprocess.run(['slim', '-s', '1', '-d', definition, str(demo.SCRIPT)], capture_output=True, text=True, timeout=20)
            self.assertNotEqual(result.returncode, 0)

    def test_parser_rejects_invalid_records_fitness_and_versions(self):
        valid = 'Eidos version 3.0.1\nSLiM version 4.0.1\nFIT,0,1.0,1.0\nDATA,0,0,20,0\n'
        self.assertEqual(demo.parse_output(valid, 20, 0, 1, 0), {0: [0]})
        for bad in ('', valid+'DATA,0,0,20,0\n', valid+'FIT,0,1.0,1.0\n',
                    valid.replace('1.0,1.0', '.9,1.0'), valid.replace('4.0.1', '5.0'),
                    valid.replace('20,0', '20,1'), valid.replace('20,0', '20,41'),
                    valid.replace('20,0', '21,0'), valid+'DATA,1,0,20,0\n',
                    valid.replace('FIT,0,1.0,1.0\n', '')):
            with self.subTest(bad), self.assertRaises(ValueError):
                demo.parse_output(bad, 20, 0, 1, 0)

    def test_checked_output(self):
        self.assertEqual(self.report, json.loads(demo.SAMPLE.read_text()))

    def test_cli_checked_output_and_invalid_option(self):
        command = [sys.executable, '-m', 'examples.slimulations.reversible_mutation.demo']
        result = subprocess.run([*command, '--check-sample'], capture_output=True, text=True, check=True, timeout=180)
        self.assertEqual(result.stdout, demo.SAMPLE.read_text())
        result = subprocess.run([*command, '--replicates', '0'], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)


if __name__ == '__main__':
    unittest.main()
