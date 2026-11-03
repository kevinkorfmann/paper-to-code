"""Independent rational crosses, finite states, parser failures and real SLiM."""
from collections import defaultdict
from fractions import Fraction as F
import itertools
import json
import math
import statistics
import subprocess
import sys
import unittest

from examples.slimulations.underdominance import demo


def crosses(counts, w):
    """Explicit ordered weighted parents and Mendelian gametes, without q()."""
    parents = [(0, 0), (0, 1), (1, 1)]
    weights = [F(counts[0]), F(counts[1])*w, F(counts[2])]
    total = sum(weights); offspring = [F(0)]*3
    for i, j, a, b in itertools.product(range(3), range(3), range(2), range(2)):
        offspring[parents[i][a]+parents[j][b]] += weights[i]*weights[j]/(4*total*total)
    return offspring


def enumerate_generation(counts, w):
    """Enumerate ordered offspring genotypes, no multinomial coefficient formula."""
    probs = crosses(counts, w); output = defaultdict(F)
    for offspring in itertools.product(range(3), repeat=sum(counts)):
        state = tuple(offspring.count(g) for g in range(3))
        output[state] += math.prod(probs[g] for g in offspring)
    return output


class UnderdominanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = {name: demo.run_counts(founders, w, seed=demo.SEED+i)
                       for i, (name, (founders, w)) in enumerate(demo.TREATMENTS.items())}
        cls.report = demo.build_report(results=cls.results)

    def test_gamete_probability_against_ordered_parental_crosses(self):
        for counts in demo.states(4):
            for w in (F(1), F(3, 5), F(1, 10)):
                exact = crosses(counts, w)
                q = demo.gamete_frequency(counts, float(w))
                for a, b in zip(((1-q)**2, 2*q*(1-q), q*q), exact):
                    self.assertAlmostEqual(a, float(b), places=14)

    def test_transition_against_exact_ordered_offspring_enumeration(self):
        for counts, row in zip(demo.states(3), demo.transition(3, .6)):
            exact = enumerate_generation(counts, F(3, 5))
            for state, got in zip(demo.states(3), row):
                self.assertAlmostEqual(got, float(exact[state]), places=14)

    def test_transition_probability_mass_and_conditional_moments(self):
        for counts, row in zip(demo.states(8), demo.transition(8, .6)):
            q = demo.gamete_frequency(counts, .6)
            p = [(h+2*d)/16 for a, h, d in demo.states(8)]
            self.assertAlmostEqual(math.fsum(row), 1., places=14)
            self.assertTrue(all(x >= 0 for x in row))
            self.assertAlmostEqual(math.fsum(x*y for x, y in zip(row, p)), q, places=14)
            self.assertAlmostEqual(math.fsum(x*(y-q)**2 for x, y in zip(row, p)), q*(1-q)/16, places=14)

    def test_multigeneration_chain_against_exact_rational_enumeration(self):
        founders = (1, 1, 1); dist = {founders: F(1)}
        expected = demo.exact_path(founders, .6, 3)
        for t in range(4):
            if t:
                updated = defaultdict(F)
                for state, mass in dist.items():
                    for target, p in enumerate_generation(state, F(3, 5)).items(): updated[target] += mass*p
                dist = updated
            exact_p = sum(m*F(h+2*d, 6) for (a, h, d), m in dist.items())
            exact_h = sum(m*2*F(h+2*d, 6)*(1-F(h+2*d, 6)) for (a, h, d), m in dist.items())
            self.assertAlmostEqual(expected[t]['mean_p'], float(exact_p), places=14)
            self.assertAlmostEqual(expected[t]['mean_gene_diversity'], float(exact_h), places=14)
            exact_p2 = sum(m*F(h+2*d, 6)**2 for (a, h, d), m in dist.items())
            exact_h2 = sum(m*(2*F(h+2*d, 6)*(1-F(h+2*d, 6)))**2 for (a, h, d), m in dist.items())
            self.assertAlmostEqual(expected[t]['second_moment_p'], float(exact_p2), places=14)
            self.assertAlmostEqual(expected[t]['second_moment_gene_diversity'], float(exact_h2), places=14)
            self.assertAlmostEqual(expected[t]['loss_probability'], float(dist.get((3, 0, 0), 0)), places=14)
            self.assertAlmostEqual(expected[t]['fixation_probability'], float(dist.get((0, 0, 3), 0)), places=14)

    def test_neutral_marginal_is_binomial(self):
        for counts, row in zip(demo.states(5), demo.transition(5, 1.)):
            k = counts[1]+2*counts[2]; p = F(k, 10)
            marginal = [0.]*11
            for (a, h, d), mass in zip(demo.states(5), row): marginal[h+2*d] += mass
            for j, observed in enumerate(marginal):
                self.assertAlmostEqual(observed, float(math.comb(10, j)*p**j*(1-p)**(10-j)), places=14)

    def test_exact_neutral_mean_and_diversity_formula(self):
        for t, row in enumerate(demo.exact_path((3, 2, 3), 1., 80)):
            self.assertAlmostEqual(row['mean_p'], .5, places=13)
            self.assertAlmostEqual(row['mean_gene_diversity'], .5*(15/16)**t, places=13)

    def test_exact_allele_label_symmetry(self):
        a = demo.exact_path((4, 3, 1), .6, 20)
        b = demo.exact_path((1, 3, 4), .6, 20)
        for x, y in zip(a, b):
            self.assertAlmostEqual(x['mean_p']+y['mean_p'], 1, places=13)
            self.assertAlmostEqual(x['mean_gene_diversity'], y['mean_gene_diversity'], places=13)
            self.assertAlmostEqual(x['loss_probability'], y['fixation_probability'], places=13)

    def test_symmetric_exact_mean_can_stay_half_as_diversity_disappears(self):
        path = demo.exact_path((2, 4, 2), .6, 80)
        for row in path:
            self.assertAlmostEqual(row['mean_p'], .5, places=13)
            self.assertAlmostEqual(row['probability_mass'], 1, places=13)
        self.assertLess(path[-1]['mean_gene_diversity'], 1e-7)
        self.assertGreater(path[-1]['loss_probability']+path[-1]['fixation_probability'], .99999)

    def test_exact_absorbing_boundaries(self):
        for founders in ((8, 0, 0), (0, 0, 8)):
            for row in demo.exact_path(founders, .6, 10):
                self.assertEqual(row['mean_p'], founders[2]/8)
                self.assertEqual(row['mean_gene_diversity'], 0)

    def test_deterministic_unstable_equilibrium_and_derivative(self):
        self.assertEqual(demo.deterministic_path(.5, .6, 80), [.5]*81)
        for p in (.1, .4, .49): self.assertLess(demo.deterministic_path(p, .6, 1)[1], p)
        for p in (.51, .6, .9): self.assertGreater(demo.deterministic_path(p, .6, 1)[1], p)
        eps = 1e-6
        slope = (demo.deterministic_path(.5+eps, .6, 1)[1]-demo.deterministic_path(.5-eps, .6, 1)[1])/(2*eps)
        self.assertAlmostEqual(slope, 1.25, places=9)

    def test_deterministic_neutral_and_boundary_reductions(self):
        for p in (0., .2, .5, .9, 1.):
            for value in demo.deterministic_path(p, 1., 80): self.assertAlmostEqual(value, p, places=13)
        for p in (0., 1.): self.assertEqual(demo.deterministic_path(p, .6, 80), [p]*81)

    def test_genotype_composition_is_needed_beyond_allele_count(self):
        self.assertEqual(demo.gamete_frequency((6, 0, 2), .6), .25)
        self.assertAlmostEqual(demo.gamete_frequency((4, 4, 0), .6), .1875, places=14)
        self.assertEqual(demo.gamete_frequency((6, 0, 2), 1.), demo.gamete_frequency((4, 4, 0), 1.))

    def test_realized_genotypes_sizes_founders_and_all_generations(self):
        for name, (founders, w) in demo.TREATMENTS.items():
            rows = self.results[name]
            self.assertEqual(set(rows), set(range(81)))
            self.assertEqual(rows[0], [founders]*512)
            self.assertTrue(all(len(values) == 512 for values in rows.values()))
            self.assertTrue(all(sum(counts) == sum(founders) for values in rows.values() for counts in values))

    def test_actual_first_generation_mean_variance_and_heterozygosity(self):
        for name, (founders, w) in demo.TREATMENTS.items():
            n = sum(founders); q = demo.gamete_frequency(founders, w)
            values = [(h+2*d)/(2*n) for a, h, d in self.results[name][1]]
            variance = q*(1-q)/(2*n)
            self.assertLess(abs(statistics.mean(values)-q), 6*math.sqrt(variance/512))
            # Fixed numerical sanity envelope, not a claim about a new random run.
            self.assertLess(abs(statistics.variance(values)/variance-1), .25)
            het = [h/n for a, h, d in self.results[name][1]]; target = 2*q*(1-q)
            self.assertLess(abs(statistics.mean(het)-target), 6*math.sqrt(target*(1-target)/(n*512)))

    def test_actual_small_population_matches_exact_multigeneration_distribution(self):
        for group in self.report['treatments'][-2:]:
            for row in group['checkpoints'][1:]:
                exact = row['exact_finite_genotype_model']; got = row['observed']
                for label, second in [('mean_p', 'second_moment_p'), ('mean_gene_diversity', 'second_moment_gene_diversity')]:
                    variance = max(0, exact[second]-exact[label]**2)
                    self.assertLessEqual(abs(got[label]-exact[label]), 6*math.sqrt(variance/512)+1/512)
                for count, probability in [('lost_replicates', 'loss_probability'), ('fixed_replicates', 'fixation_probability')]:
                    p = exact[probability]
                    self.assertLessEqual(abs(got[count]/512-p), 6*math.sqrt(p*(1-p)/512)+1/512)

    def test_actual_neutral_mean_and_diversity_have_correct_expectations(self):
        for name in ('neutral_at_threshold', 'small_neutral'):
            n = sum(demo.TREATMENTS[name][0])
            for t in (1, 10, 40, 80):
                observed = demo.summarize(self.results[name][t])
                self.assertLess(abs(observed['mean_p']-.5), 6*observed['mean_p_monte_carlo_se']+1/512)
                self.assertLess(abs(observed['mean_gene_diversity']-.5*(1-1/(2*n))**t), 6*observed['gene_diversity_monte_carlo_se']+1/512)

    def test_actual_threshold_populations_polarize_more_than_neutral_control(self):
        selected = demo.summarize(self.results['underdominant_at_threshold'][80])
        neutral = demo.summarize(self.results['neutral_at_threshold'][80])
        self.assertEqual(selected['segregating_replicates'], 0)
        self.assertGreater(selected['fixed_replicates'], 150)
        self.assertGreater(selected['lost_replicates'], 150)
        self.assertGreater(neutral['segregating_replicates'], 400)
        self.assertGreater(neutral['mean_gene_diversity']-selected['mean_gene_diversity'], .25)

    def test_actual_starting_frequency_predicts_majority_not_every_replicate(self):
        low = demo.summarize(self.results['underdominant_below'][80])
        high = demo.summarize(self.results['underdominant_above'][80])
        self.assertGreater(low['lost_replicates'], 450)
        self.assertGreater(high['fixed_replicates'], 450)
        self.assertGreater(low['fixed_replicates'], 0)
        self.assertGreater(high['lost_replicates'], 0)

    def test_summary_pooled_minus_within_is_twice_between_population_variance(self):
        for rows in self.results['underdominant_at_threshold'].values():
            summary = demo.summarize(rows)
            frequencies = [(h+2*d)/200 for a, h, d in rows]
            self.assertAlmostEqual(summary['gene_diversity_of_pooled_frequency']-summary['mean_gene_diversity'],
                                   2*statistics.pvariance(frequencies), places=14)

    def test_actual_absorption_is_permanent(self):
        for name, values in self.results.items():
            n = sum(demo.TREATMENTS[name][0])
            for t in range(80):
                for a, b in zip(values[t], values[t+1]):
                    if a[0] == n or a[2] == n: self.assertEqual(a, b)

    def test_actual_genotype_composition_control(self):
        for founders in ((6, 0, 2), (4, 4, 0)):
            rows = demo.run_counts(founders, .6, generations=1, seed=402)
            observed = demo.summarize(rows[1]); expected = demo.gamete_frequency(founders, .6)
            self.assertLess(abs(observed['mean_p']-expected), 6*observed['mean_p_monte_carlo_se'])

    def test_actual_tiny_whole_genotype_distribution(self):
        from collections import Counter
        founders = (1, 1, 0)
        counts = Counter(demo.run_counts(founders, .6, generations=1, replicates=2048, seed=406)[1])
        expected = enumerate_generation(founders, F(3, 5))
        for state in demo.states(2):
            p = float(expected[state])
            self.assertLessEqual(abs(counts[state]/2048-p), 6*math.sqrt(p*(1-p)/2048)+1/2048)

    def test_incidental_selfing_is_allowed_in_tiny_population(self):
        rows = demo.run_counts((1, 0, 1), .6, generations=1, replicates=2048, seed=407)[1]
        # Independent parental draws give Hobs=.5. Prohibiting selfing gives 1.
        heterozygosity = statistics.mean(h/2 for a, h, d in rows)
        self.assertLess(abs(heterozygosity-.5), 6*math.sqrt(.25/(2*2048)))

    def test_actual_monomorphic_controls(self):
        for founders in ((8, 0, 0), (0, 0, 8)):
            result = demo.run_counts(founders, generations=3, replicates=3, seed=403)
            self.assertTrue(all(rows == [founders]*3 for rows in result.values()))

    def test_repeat_seed_and_zero_generations(self):
        args = dict(founders=(1, 2, 1), generations=3, replicates=3, seed=404)
        self.assertEqual(demo.run_counts(**args), demo.run_counts(**args))
        self.assertEqual(demo.run_counts((1, 2, 1), generations=0, replicates=2), {0: [(1, 2, 1)]*2})

    def test_invalid_python_parameters(self):
        for founders in ((), (1, 1), (0, 0, 0), (-1, 1, 3), (True, 1, 1), (1.5, 1, 1), (201, 0, 0)):
            with self.assertRaises(ValueError): demo.run_counts(founders)
        for w in (0, -.1, 1.1, True, float('nan'), float('inf'), 'bad'):
            with self.assertRaises(ValueError): demo.run_counts(w_het=w)
        for kw in ({'generations': -1}, {'generations': 101}, {'replicates': 0}, {'seed': 0}, {'seed': True}):
            with self.assertRaises(ValueError): demo.run_counts(**kw)
        for p in (-.1, 1.1, float('nan'), True):
            with self.assertRaises(ValueError): demo.deterministic_path(p, .6, 1)
        with self.assertRaises(ValueError): demo.states(21)

    def test_invalid_eidos_parameters(self):
        for definition in ('N=1', 'INITIAL_AA=101', 'W_HET=0.0', 'W_HET=1.1', 'GENERATIONS=-1', 'REPLICATES=0'):
            result = subprocess.run(['slim', '-d', definition, str(demo.SCRIPT)], capture_output=True, text=True, timeout=10)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('invalid', result.stderr)

    def test_parser_rejects_corruption(self):
        command = ['slim', '-s', '405', '-d', 'N=4', '-d', 'INITIAL_AA=1', '-d', 'INITIAL_HET=2',
                   '-d', 'GENERATIONS=1', '-d', 'REPLICATES=1', str(demo.SCRIPT)]
        text = subprocess.run(command, capture_output=True, text=True, check=True, timeout=10).stdout
        parse = lambda value: demo.parse_output(value, (1, 2, 1), .6, 1, 1)
        self.assertEqual(parse(text)[0], [(1, 2, 1)])
        bad = [text.replace('DATA,0,0,1,2,1', 'DATA,0,0,0,2,2'),
               text.replace('DATA,0,0,1,2,1', 'DATA,0,0,-1,4,1'),
               text.replace('DATA,0,0,1,2,1', 'DATA,0,0,1,2,2'),
               text.replace('FIT,1,0.6', 'FIT,1,1'),
               text.replace('SLiM version 4.0.1', 'SLiM version 5.0'),
               text+'DATA,0,0,1,2,1\n',
               '\n'.join(line for line in text.splitlines() if not line.startswith('DATA,1,'))]
        for corrupted in bad:
            with self.assertRaises(ValueError): parse(corrupted)

    def test_checked_sample_is_actual_output(self):
        self.assertEqual(demo.render(self.report), demo.SAMPLE.read_text())
        self.assertEqual(json.loads(demo.SAMPLE.read_text())['python_version'], '3.14.5')

    def test_cli_reproduces_checked_sample(self):
        result = subprocess.run([sys.executable, '-m', 'examples.slimulations.underdominance.demo', '--check-sample'],
                                capture_output=True, text=True, check=True, timeout=120)
        self.assertEqual(result.stdout, demo.SAMPLE.read_text())

    def test_cli_rejects_invalid_and_mismatched_output(self):
        for args in (['--seed', '0'], ['--replicates', '1', '--check-sample']):
            result = subprocess.run([sys.executable, '-m', 'examples.slimulations.underdominance.demo', *args],
                                    capture_output=True, text=True, timeout=120)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('error:', result.stderr)

    def test_single_replicate_standard_error_is_unavailable(self):
        row = demo.summarize([(1, 2, 1)])
        self.assertIsNone(row['mean_p_monte_carlo_se'])
        self.assertIsNone(row['gene_diversity_monte_carlo_se'])


if __name__ == '__main__': unittest.main()
