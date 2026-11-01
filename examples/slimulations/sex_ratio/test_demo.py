"""Exact enumeration of all tiny-population gamete draws, plus real SLiM controls."""

from collections import defaultdict
from fractions import Fraction as F
from functools import lru_cache
import itertools
import json
import math
import platform
import statistics
import subprocess
import sys
import unittest

from examples.slimulations.sex_ratio import demo


@lru_cache(None)
def exact_transition(km, kf, nm=1, nf=2):
    """Enumerate ordered paternal/maternal gametes for every offspring (2^6 draws)."""
    pm, pf = F(km, 2*nm), F(kf, 2*nf)
    probabilities = [pm, pf]*(nm+nf)
    distribution = defaultdict(F)
    for draw in itertools.product((0, 1), repeat=2*(nm+nf)):
        mass = math.prod(p if allele else 1-p for allele, p in zip(draw, probabilities))
        distribution[sum(draw[:2*nm]), sum(draw[2*nm:])] += mass
    assert sum(distribution.values()) == 1
    return distribution


def enumerated_moments(distribution, nm=1, nf=2):
    moments = [F(0)]*5
    for (km, kf), mass in distribution.items():
        m, f = F(km, 2*nm), F(kf, 2*nf)
        for i, v in enumerate((m, f, m*m, f*f, m*f)):
            moments[i] += mass*v
    return tuple(moments)


class SexRatioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = {name: demo.run_counts(nm=nm, nf=nf, seed=demo.SEED+i)
                       for i, (name, nm, nf) in enumerate(demo.TREATMENTS)}
        cls.results['opposite_sex_fixation_control'] = demo.run_counts(
            nm=20, nf=180, generations=1, opposite=1, seed=demo.SEED+3)
        cls.report = demo.build_report(results=cls.results)

    def test_classical_benchmark_and_sex_symmetry(self):
        self.assertEqual(demo.classical_ne(100, 100), 200)
        self.assertEqual(demo.classical_ne(20, 180), 72)
        self.assertEqual(demo.classical_ne(180, 20), 72)

    def test_all_tiny_states_one_generation_match_enumeration(self):
        for km, kf in itertools.product(range(3), range(5)):
            predicted = demo.moment_path(1, 2, 1, F(km, 2), F(kf, 4))[-1]
            self.assertEqual(predicted, enumerated_moments(exact_transition(km, kf)))

    def test_multigeneration_full_markov_enumeration(self):
        distribution = {(1, 2): F(1)}
        path = demo.moment_path(1, 2, 3)
        for t in range(4):
            self.assertEqual(enumerated_moments(distribution), path[t])
            # Also compute diversity directly from every enumerated state.
            h = sum(mass*2*F(km+kf, 6)*(1-F(km+kf, 6))
                    for (km, kf), mass in distribution.items())
            self.assertEqual(h, demo.predicted(path[t], 1, 2)['mean_census_gene_diversity'])
            next_distribution = defaultdict(F)
            for old, mass in distribution.items():
                for new, transition in exact_transition(*old).items():
                    next_distribution[new] += mass*transition
            distribution = next_distribution

    def test_first_generation_census_variance_same_but_role_variance_differs(self):
        for _, nm, nf in demo.TREATMENTS:
            expectation = demo.predicted(demo.moment_path(nm, nf, 1)[1], nm, nf)
            self.assertEqual(expectation['census_p_variance'], F(1, 1600))
            self.assertEqual(expectation['parental_role_p_variance'], 1/(8*demo.classical_ne(nm, nf)))

    def test_opposite_fixation_exact_control_and_conditional_census_change(self):
        path = demo.moment_path(20, 180, 1, 0, 1)
        self.assertEqual(demo.predicted(path[0], 20, 180)['mean_census_p'], F(9, 10))
        self.assertEqual(demo.predicted(path[0], 20, 180)['mean_parental_role_p'], F(1, 2))
        self.assertEqual(path[1], (F(1,2), F(1,2), F(1,4), F(1,4), F(1,4)))

    def test_absorbing_states_stay_absorbed(self):
        for p in (0, 1):
            path = demo.moment_path(20, 180, 100, p, p)
            self.assertTrue(all(x == path[0] for x in path))
            self.assertEqual(demo.predicted(path[-1], 20, 180)['mean_census_gene_diversity'], 0)

    def test_mirror_sex_sizes_have_identical_ensemble_predictions(self):
        a, b = demo.moment_path(20, 180, 80), demo.moment_path(180, 20, 80)
        for x, y in zip(a, b):
            self.assertEqual(demo.predicted(x, 20, 180), demo.predicted(y, 180, 20))

    def test_role_weighted_mean_is_conserved(self):
        path = demo.moment_path(20, 180, 80, F(1,4), F(3,4))
        for state in path:
            self.assertEqual(demo.predicted(state, 20, 180)['mean_parental_role_p'], F(1,2))

    def test_finite_curve_is_not_simple_ne_substitution(self):
        for nm, nf in [(100, 100), (20, 180)]:
            exact = demo.predicted(demo.moment_path(nm, nf, 2)[2], nm, nf)['mean_census_gene_diversity']
            shortcut = F(1,2)*(1-1/(2*demo.classical_ne(nm,nf)))**2
            self.assertNotEqual(exact, shortcut)

    def test_all_replicates_and_exact_hardy_weinberg_founders(self):
        for name, nm, nf in demo.TREATMENTS:
            self.assertEqual(set(self.results[name]), {0,1,20,80})
            self.assertEqual(self.results[name][0], [[nm,nf,100]]*512)
            self.assertTrue(all(len(rows) == 512 for rows in self.results[name].values()))

    def test_actual_opposite_founders_produce_only_heterozygotes(self):
        control = self.results['opposite_sex_fixation_control']
        self.assertEqual(control[0], [[0,360,0]]*512)
        self.assertEqual(control[1], [[20,180,200]]*512)

    def test_actual_first_generation_variances(self):
        for group in self.report['treatments'][:3]:
            row = group['checkpoints'][1]
            for label in ('census_p', 'parental_role_p'):
                ratio = row['observed'][label+'_sample_variance']/row['exact_finite_expectation'][label+'_variance']
                self.assertLess(abs(ratio-1), .25)  # Numerical sanity bound for 512 replicates.

    def test_actual_means_match_exact_curve_with_monte_carlo_error(self):
        for group in self.report['treatments'][:3]:
            for row in group['checkpoints'][1:]:
                for key, se_key in [('mean_census_p','census_p_monte_carlo_se'),
                                    ('mean_parental_role_p','parental_role_p_monte_carlo_se'),
                                    ('mean_census_gene_diversity','census_gene_diversity_monte_carlo_se')]:
                    self.assertLess(abs(row['observed'][key]-row['exact_finite_expectation'][key]),
                                    5*row['observed'][se_key])

    def test_skewed_ratios_lose_more_diversity_and_mirror_control_agrees(self):
        end = [g['checkpoints'][-1]['observed'] for g in self.report['treatments'][:3]]
        for skew in end[1:]:
            difference = end[0]['mean_census_gene_diversity']-skew['mean_census_gene_diversity']
            se = math.hypot(end[0]['census_gene_diversity_monte_carlo_se'], skew['census_gene_diversity_monte_carlo_se'])
            self.assertGreater(difference, 5*se)
        self.assertLess(abs(end[1]['mean_census_gene_diversity']-end[2]['mean_census_gene_diversity']),
                        5*math.hypot(end[1]['census_gene_diversity_monte_carlo_se'],end[2]['census_gene_diversity_monte_carlo_se']))

    def test_replicate_variance_consistent_with_mean_diversity(self):
        for group in self.report['treatments']:
            for row in group['checkpoints']:
                o = row['observed']; p = o['mean_census_p']
                self.assertAlmostEqual(o['mean_census_gene_diversity'],
                    2*(p*(1-p)-o['census_p_sample_variance']*511/512), places=13)

    def test_diversity_distinct_from_observed_heterozygosity(self):
        o = self.report['treatments'][-1]['checkpoints'][1]['observed']
        self.assertEqual(o['mean_census_gene_diversity'], .5)
        self.assertEqual(o['mean_observed_heterozygosity'], 1.)

    def test_no_conditioning_on_absorption(self):
        for name, nm, nf in demo.TREATMENTS:
            rows = self.results[name][80]
            stats = demo.summarize(rows,nm,nf)
            self.assertEqual(len(rows),512)
            self.assertEqual(stats['allele_fixed_replicates'],sum(km+kf == 400 for km,kf,_ in rows))
            self.assertEqual(stats['allele_lost_replicates'],sum(km+kf == 0 for km,kf,_ in rows))
        self.assertGreater(self.report['treatments'][1]['checkpoints'][-1]['observed']['allele_fixed_replicates'],0)

    def test_versions(self):
        self.assertEqual(platform.python_version(),'3.14.5')
        self.assertEqual(demo.check_slim_version('slim'),'4.0.1')

    def test_small_seed_repeat_and_zero_generation(self):
        args = dict(nm=4,nf=12,replicates=2,generations=3,seed=123)
        self.assertEqual(demo.run_counts(**args),demo.run_counts(**args))
        self.assertEqual(demo.run_counts(nm=4,nf=12,replicates=1,generations=0)[0],[[4,12,8]])

    def test_python_invalid_parameters(self):
        for kwargs in [{'nm':5},{'nf':True},{'nm':0},{'replicates':0},{'generations':101},
                       {'opposite':2},{'seed':0},{'seed':True}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                demo.run_counts(**kwargs)
        for p in (-.1,1.1,float('nan'),float('inf'),True):
            with self.assertRaises(ValueError): demo.moment_path(1,2,1,p,.5)

    def test_eidos_invalid_sex_count(self):
        p = subprocess.run(['slim','-d','NM=5',str(demo.SCRIPT)],capture_output=True,text=True,timeout=10)
        self.assertNotEqual(p.returncode,0)
        self.assertIn('invalid sex count',p.stderr)

    def test_parser_rejects_corrupt_or_missing_audits(self):
        p = subprocess.run(['slim','-s','123','-d','NM=4','-d','NF=12','-d','REPLICATES=1',
                            '-d','GENERATIONS=0',str(demo.SCRIPT)],capture_output=True,text=True,check=True,timeout=10)
        text = p.stdout
        self.assertEqual(demo.parse_output(text,4,12,1,0,0)[0],[[4,12,8]])
        for bad in ['', text+'DATA,0,0,4,12,4,12,8\n', text.replace('INIT,0,M,1,2,1','INIT,0,M,0,4,0'),
                    text.replace('FIT,1','FIT,2'),text.replace('SLiM version 4.0.1','SLiM version 5.0'),
                    text.replace('DATA,0,0,4,12,4,12,8','DATA,0,0,4,12,4,12,9'),
                    text.replace('DATA,0,0,4,12,4,12,8','DATA,0,0,5,11,4,12,8')]:
            with self.assertRaises(ValueError):demo.parse_output(bad,4,12,1,0,0)

    def test_sample_output(self):
        self.assertEqual(demo.render(self.report),demo.SAMPLE.read_text())

    def test_single_replicate_uncertainty_is_not_zero(self):
        stats=demo.summarize([[4,12,8]],4,12)
        self.assertIsNone(stats['census_p_monte_carlo_se'])
        self.assertIsNone(stats['census_p_sample_variance'])

    def test_cli_and_invalid_flag(self):
        command = [sys.executable,'-m','examples.slimulations.sex_ratio.demo']
        p=subprocess.run(command+['--replicates','2','--generations','1'],capture_output=True,text=True,timeout=20)
        self.assertEqual(p.returncode,0,p.stderr)
        self.assertEqual(len(json.loads(p.stdout)['treatments']),4)
        bad=subprocess.run(command+['--unknown'],capture_output=True,text=True,timeout=10)
        self.assertEqual(bad.returncode,2)


if __name__ == '__main__':
    unittest.main()
