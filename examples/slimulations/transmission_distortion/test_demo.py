"""Independent enumeration, engine controls, and checked synthetic results."""

from fractions import Fraction as F
import json
import math
import statistics
import subprocess
import sys
import unittest

from examples.slimulations.transmission_distortion import demo


def binomial_distribution(copies, q):
    return [math.comb(copies, k)*q**k*(1-q)**(copies-k) for k in range(copies+1)]


def iid_mean_moments(n, m):
    """Raw moments of an iid sample mean, from distinct-index partitions."""
    a, b, c, d = m[1:]
    return [F(1), a, (n*b+n*(n-1)*a*a)/n**2,
            (n*c+3*n*(n-1)*b*a+n*(n-1)*(n-2)*a**3)/n**3,
            (n*d+4*n*(n-1)*c*a+3*n*(n-1)*b*b
             +6*n*(n-1)*(n-2)*b*a*a+n*(n-1)*(n-2)*(n-3)*a**4)/n**4]


def offspring_fourth_central(n, q_moments):
    # Conditional offspring count is binomial: factorial moments are (2N)_j*q^j.
    stirling = ((1,), (0, 1), (0, 1, 1), (0, 1, 3, 1), (0, 1, 7, 6, 1))
    raw = [F(1)]
    for r in range(1, 5):
        falling, moment = 1, F(0)
        for j in range(1, r+1):
            falling *= 2*n-j+1
            moment += stirling[r][j]*falling*q_moments[j]
        raw.append(moment/(2*n)**r)
    return raw[4]-4*raw[1]*raw[3]+6*raw[1]**2*raw[2]-3*raw[1]**4


class TransmissionDistortionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = demo.build_report()

    def test_exact_parental_genotype_and_gamete_enumeration(self):
        for n, AA, het in ((200, 50, 100), (200, 0, 200), (200, 100, 0), (7, 1, 3)):
            for tau in (F(0), F(1, 4), F(1, 2), F(3, 4), F(1)):
                parent_prob = [F(n-AA-het, n), F(het, n), F(AA, n)]
                transmission = [F(0), tau, F(1)]
                offspring = [F(0)]*3
                for i in range(3):
                    for j in range(3):
                        for a in (0, 1):
                            for b in (0, 1):
                                pa = transmission[i] if a else 1-transmission[i]
                                pb = transmission[j] if b else 1-transmission[j]
                                offspring[a+b] += parent_prob[i]*parent_prob[j]*pa*pb
                prediction = demo.exact_predictions(n, AA, het, tau)[1]
                self.assertEqual(sum(offspring), 1)
                self.assertEqual(prediction['AA_fraction'], offspring[2])
                self.assertEqual(prediction['heterozygote_fraction'], offspring[1])
                self.assertEqual(prediction['mean_frequency'], offspring[2]+offspring[1]/2)

    def test_two_generations_by_full_multinomial_and_binomial_enumeration(self):
        n = 3
        for tau in (F(0), F(1, 4), F(1, 2), F(3, 4), F(1)):
            q = (1+tau)/3  # one founder of each genotype, not HWE
            probs = ((1-q)**2, 2*q*(1-q), q*q)
            distribution = [F(0)]*(2*n+1)
            AA_mean = H_mean = F(0)
            for aa in range(n+1):
                for het in range(n-aa+1):
                    AA = n-aa-het
                    weight = math.comb(n, aa)*math.comb(n-aa, het)
                    weight *= probs[0]**aa*probs[1]**het*probs[2]**AA
                    gamete = (AA+tau*het)/n
                    for k, prob in enumerate(binomial_distribution(2*n, gamete)):
                        distribution[k] += weight*prob
                    AA_mean += weight*gamete**2
                    H_mean += weight*2*gamete*(1-gamete)
            mean = sum(F(k, 2*n)*p for k, p in enumerate(distribution))
            variance = sum((F(k, 2*n)-mean)**2*p for k, p in enumerate(distribution))
            expected = demo.exact_predictions(n, 1, 1, tau)[2]
            self.assertEqual(sum(distribution), 1)
            self.assertEqual(expected, {'mean_frequency': mean, 'frequency_variance': variance,
                                        'AA_fraction': AA_mean, 'heterozygote_fraction': H_mean})
            q_moments = iid_mean_moments(n, [F(1)]+[q*q+2*tau**r*q*(1-q) for r in range(1, 5)])
            fourth = sum((F(k, 2*n)-mean)**4*p for k, p in enumerate(distribution))
            self.assertEqual(offspring_fourth_central(n, q_moments), fourth)

    def test_mendelian_mean_is_a_martingale_for_arbitrary_pairing(self):
        for AA, het in ((50, 100), (0, 200), (100, 0), (17, 93)):
            for row in demo.exact_predictions(200, AA, het, .5).values():
                self.assertEqual(row['mean_frequency'], F(2*AA+het, 400))

    def test_frequency_change_requires_heterozygotes(self):
        for tau in (F(1, 4), F(1, 2), F(3, 4)):
            for AA, het in ((50, 100), (0, 200), (100, 0), (17, 93)):
                p = F(2*AA+het, 400)
                first = demo.exact_predictions(200, AA, het, tau)[1]['mean_frequency']
                self.assertEqual(first-p, (tau-F(1, 2))*F(het, 200))
        homo = demo.exact_predictions(200, 100, 0, .75)
        self.assertEqual(homo[1]['mean_frequency'], F(1, 2))
        self.assertEqual(homo[2]['mean_frequency'], F(5, 8))

    def test_reverse_bias_and_allele_relabeling(self):
        for AA, het in ((50, 100), (0, 200), (100, 0), (17, 93)):
            forward = demo.exact_predictions(200, AA, het, F(3, 4))
            reverse = demo.exact_predictions(200, 200-AA-het, het, F(1, 4))
            for t in (1, 2):
                self.assertEqual(forward[t]['mean_frequency'], 1-reverse[t]['mean_frequency'])
                self.assertEqual(forward[t]['frequency_variance'], reverse[t]['frequency_variance'])

    def test_second_generation_heterozygosity_has_finite_population_correction(self):
        row = demo.exact_predictions(200, 50, 100, .75)[2]
        p = row['mean_frequency']
        self.assertLess(row['heterozygote_fraction'], 2*p*(1-p))
        self.assertEqual(row['heterozygote_fraction']+2*row['AA_fraction'], 2*p)

    def test_actual_founders_and_fitness_audit(self):
        for AA, het in ((50, 100), (0, 200), (100, 0)):
            self.assertEqual(demo.run_counts(AA=AA, het=het, replicates=2, generations=0), {0: [(AA, het)]*2})

    def test_first_and_second_generation_means_match_exact_predictions(self):
        for treatment in self.report['treatments']:
            counts = treatment['founder_counts']
            predictions = demo.exact_predictions(200, counts['AA'], counts['Aa'], treatment['tau'])
            for row in treatment['checkpoints']:
                p = predictions[row['generations_elapsed']]
                tolerance = 6*math.sqrt(float(p['frequency_variance'])/512)+1e-9
                self.assertLess(abs(row['observed']['mean_frequency']-float(p['mean_frequency'])), tolerance)

    def test_both_generation_variances_with_independent_fourth_moment_tolerances(self):
        for treatment in self.report['treatments']:
            founders, tau = treatment['founder_counts'], F(str(treatment['tau']))
            predictions = demo.exact_predictions(200, founders['AA'], founders['Aa'], tau)
            q = predictions[1]['mean_frequency']
            moments = {1: [q**r for r in range(5)],
                       2: iid_mean_moments(200, [F(1)]+[q*q+2*tau**r*q*(1-q) for r in range(1, 5)])}
            for row in treatment['checkpoints']:
                t = row['generations_elapsed']
                variance = predictions[t]['frequency_variance']
                fourth = offspring_fourth_central(200, moments[t])
                variance_se = math.sqrt(float((fourth-F(509, 511)*variance**2)/512))
                self.assertLess(abs(row['observed']['frequency_variance']-float(variance)), 6*variance_se+1e-9)

    def test_genotype_fractions_match_expectations(self):
        for treatment in self.report['treatments']:
            for row in treatment['checkpoints']:
                # A conservative six-SE bound: conditional binomial variance plus
                # total-variance bound for the random gen-2 gamete probability.
                for key in ('AA_fraction', 'heterozygote_fraction'):
                    observed = row['observed']['mean_'+key]
                    expected = row['prediction'][key]
                    variance_bound = .25/200+4*row['prediction']['frequency_variance']
                    self.assertLess(abs(observed-expected), 6*math.sqrt(variance_bound/512)+1e-9)

    def test_bias_and_pairing_contrasts(self):
        rows = [t['checkpoints'][0]['observed']['mean_frequency'] for t in self.report['treatments']]
        self.assertGreater(rows[3], rows[1]+.1)
        self.assertGreater(rows[1], rows[0]+.1)
        self.assertGreater(rows[0], rows[2]+.1)
        self.assertLess(abs(rows[4]-.5), .01)
        self.assertGreater(self.report['treatments'][4]['checkpoints'][1]['observed']['mean_frequency'], .6)

    def test_complete_transmission_extremes_are_exact(self):
        for tau, expected in ((0, (0, 0)), (1, (8, 0))):
            result = demo.run_counts(n=8, AA=0, het=8, tau=tau, replicates=3)
            self.assertEqual(result[1], [expected]*3)
            self.assertEqual(result[2], [expected]*3)

    def test_no_heterozygotes_gives_identical_first_generation_across_biases(self):
        settings = dict(n=20, AA=10, het=0, replicates=5, generations=1, seed=721)
        reference = demo.run_counts(tau=.5, **settings)
        for tau in (0, .25, .75, 1):
            self.assertEqual(reference, demo.run_counts(tau=tau, **settings))

    def test_absorbing_boundaries_for_all_transmission_probabilities(self):
        for AA in (0, 8):
            for tau in (0, .25, .5, .75, 1):
                result = demo.run_counts(n=8, AA=AA, het=0, tau=tau, replicates=2)
                self.assertEqual(result, {t: [(AA, 0)]*2 for t in range(3)})

    def test_seed_reproducibility_and_different_seed(self):
        first = demo.run_counts(replicates=3, seed=821)
        self.assertEqual(first, demo.run_counts(replicates=3, seed=821))
        self.assertNotEqual(first[1], demo.run_counts(replicates=3, seed=822)[1])

    def test_all_replicates_including_absorption_enter_summary(self):
        row = demo.summarize([(0, 0), (8, 0), (0, 8)], 8)
        self.assertEqual(row['mean_frequency'], .5)
        self.assertEqual(row['lost_replicates'], 1)
        self.assertEqual(row['fixed_replicates'], 1)
        self.assertEqual(row['monte_carlo_se'], round(statistics.stdev([0, 1, .5])/math.sqrt(3), 9))

    def test_single_replicate_does_not_invent_a_standard_error(self):
        row = demo.summarize([(1, 2)], 8)
        self.assertIsNone(row['monte_carlo_se'])
        self.assertIsNone(row['frequency_variance'])

    def test_invalid_wrapper_inputs(self):
        for kwargs in ({'n': True}, {'n': 1}, {'AA': 201}, {'het': -1}, {'AA': 101, 'het': 100},
                       {'tau': True}, {'tau': '0.5'}, {'tau': float('nan')}, {'tau': float('inf')},
                       {'tau': -.1}, {'tau': 1.1}, {'replicates': 0}, {'generations': 3}, {'seed': 0}):
            with self.subTest(kwargs), self.assertRaises(ValueError):
                demo.run_counts(**kwargs)

    def test_invalid_direct_script_inputs(self):
        for definition in ('N=1', 'TAU=-0.1', 'TAU=1.1', 'AA=200', 'GENERATIONS=3', 'REPLICATES=0'):
            result = subprocess.run(['slim', '-s', '1', '-d', definition, str(demo.SCRIPT)], capture_output=True, text=True, timeout=20)
            self.assertNotEqual(result.returncode, 0)

    def test_parser_rejects_missing_duplicate_bad_fitness_and_bad_version(self):
        valid = 'Eidos version 3.0.1\nSLiM version 4.0.1\nFIT,1.0,1.0\nDATA,0,0,200,50,100\n'
        self.assertEqual(demo.parse_output(valid, 200, 50, 100, 1, 0), {0: [(50, 100)]})
        for bad in ('', valid+'DATA,0,0,200,50,100\n', valid.replace('1.0,1.0', '.9,1.0'),
                    valid.replace('4.0.1', '5.0'), valid.replace('200,50,100', '200,51,100'),
                    valid.replace('200,50,100', '200,150,100'), valid+'DATA,1,0,200,50,100\n'):
            with self.subTest(bad), self.assertRaises(ValueError):
                demo.parse_output(bad, 200, 50, 100, 1, 0)

    def test_checked_output(self):
        self.assertEqual(self.report, json.loads(demo.SAMPLE.read_text()))

    def test_cli_checked_output_and_invalid_option(self):
        command = [sys.executable, '-m', 'examples.slimulations.transmission_distortion.demo']
        result = subprocess.run([*command, '--check-sample'], capture_output=True, text=True, check=True, timeout=120)
        self.assertEqual(result.stdout, demo.SAMPLE.read_text())
        result = subprocess.run([*command, '--replicates', '0'], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)


if __name__ == '__main__':
    unittest.main()
