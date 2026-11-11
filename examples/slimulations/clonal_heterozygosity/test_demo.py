"""Independent exact enumeration, boundary checks, and actual SLiM integration."""
from collections import defaultdict
from fractions import Fraction as F
import itertools
import json
import math
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch
from examples.slimulations.clonal_heterozygosity import demo as d


def enumerate_offspring(counts, c):
    """Choose actual parents and homologs, rather than using the p/H recursion."""
    parents = [(0, 0)]*counts[0]+[(0, 1)]*counts[1]+[(1, 1)]*counts[2]
    n = len(parents)
    probs = [F(0) for _ in range(3)]
    for individual in parents:
        probs[sum(individual)] += c/n
    for first, second, a, b in itertools.product(parents, parents, range(2), range(2)):
        probs[first[a]+second[b]] += (1-c)/(4*n*n)
    return probs


def exact_distribution(initial, c, generations):
    n = sum(initial)
    states = {tuple(initial): F(1)}
    for _ in range(generations):
        updated = defaultdict(F)
        for counts, mass in states.items():
            probs = enumerate_offspring(counts, c)
            # Enumerate ordered offspring outcomes, including all multinomial terms.
            for offspring in itertools.product(range(3), repeat=n):
                weight = mass
                for genotype in offspring:
                    weight *= probs[genotype]
                updated[tuple(offspring.count(j) for j in range(3))] += weight
        states = dict(updated)
    return states


class ExactTests(unittest.TestCase):
    def test_independent_parent_and_homolog_probabilities(self):
        for n in (2, 3, 4):
            for het in range(n+1):
                for hom in range(n-het+1):
                    counts = (n-het-hom, het, hom)
                    for c in (F(0), F(1, 4), F(1)):
                        expected = enumerate_offspring(counts, c)
                        self.assertEqual(sum(expected), 1)
                        for a, b in zip(d.offspring_probabilities(counts, float(c)), expected):
                            self.assertAlmostEqual(a, float(b))

    def test_multigeneration_exact_rational_enumeration(self):
        for initial in ((1, 2, 0), (0, 3, 0), (1, 0, 2)):
            n = sum(initial)
            for c in (F(0), F(1, 4), F(1)):
                for t in (0, 1, 3):
                    dist = exact_distribution(initial, c, t)
                    self.assertEqual(sum(dist.values()), 1)
                    moments = defaultdict(F)
                    for counts, mass in dist.items():
                        p = F(counts[1]+2*counts[2], 2*n)
                        moments['frequency'] += mass*p
                        moments['p_squared'] += mass*p*p
                        moments['gene_diversity'] += mass*2*p*(1-p)
                        moments['observed_heterozygosity'] += mass*F(counts[1], n)
                        moments['genotype_diversity'] += mass*(1-sum(F(k, n)**2 for k in counts))
                    moments['variance_frequency'] = moments['p_squared']-moments['frequency']**2
                    for key, expected in d.expected_means(n, initial[1], initial[2], float(c), t).items():
                        self.assertAlmostEqual(expected, float(moments[key]), places=13)

    def test_clonal_closed_form(self):
        for t in (0, 1, 20, 320):
            expected = d.expected_means(40, 20, 0, 1, t)
            self.assertAlmostEqual(expected['gene_diversity'], .25+.125*(39/40)**t)
            self.assertEqual(expected['observed_heterozygosity'], .5)
            self.assertAlmostEqual(expected['genotype_diversity'], .5*(39/40)**t)

    def test_sexual_closed_form(self):
        for t in (1, 20, 320):
            expected = d.expected_means(40, 20, 0, 0, t)
            self.assertAlmostEqual(expected['gene_diversity'], .375*(79/80)**t)
            self.assertAlmostEqual(expected['observed_heterozygosity'], .375*(79/80)**(t-1))

    def test_all_heterozygote_clones_are_absorbing(self):
        self.assertEqual(d.offspring_probabilities((0, 40, 0), 1), [0, 1, 0])
        e = d.expected_means(40, 40, 0, 1, 400)
        self.assertEqual((e['frequency'], e['gene_diversity'], e['observed_heterozygosity'], e['variance_frequency']), (.5, .5, 1, 0))

    def test_all_heterozygotes_not_absorbing_with_sex(self):
        self.assertEqual(d.offspring_probabilities((0, 40, 0), 0), [.25, .5, .25])

    def test_homozygous_boundaries_are_absorbing(self):
        for c in (0, .5, 1):
            for hom in (0, 40):
                e = d.expected_means(40, 0, hom, c, 320)
                self.assertEqual(e['gene_diversity'], 0)
                self.assertEqual(e['observed_heterozygosity'], 0)

    def test_allele_relabeling_symmetry(self):
        a = d.expected_means(40, 12, 6, .75, 90)
        b = d.expected_means(40, 12, 22, .75, 90)
        self.assertAlmostEqual(a.pop('frequency')+b.pop('frequency'), 1)
        for key in a:
            self.assertAlmostEqual(a[key], b[key])


class ContractTests(unittest.TestCase):
    @staticmethod
    def output(counts=(2, 2, 0)):
        return '\n'.join('DATA,'+str(t)+',0,'+','.join(map(str, counts)) for t in (0, 1, 2, 4))

    def parse(self, text):
        return d.parse_output(text, 4, 2, 0, 1, 4, 1)

    def test_valid_parser_and_headers(self):
        self.assertEqual(self.parse('// seed\n'+self.output())[4], [[2, 2, 0]])

    def test_missing_extra_and_duplicate_output(self):
        for text in (self.output().split('\n', 1)[1], self.output()+'\nDATA,5,0,2,2,0',
                     self.output()+'\nDATA,4,0,2,2,0', self.output().replace(',0,2,2,0', ',1,2,2,0')):
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.parse(text)

    def test_bad_counts_founders_and_clonal_innovation(self):
        for bad in ('DATA,0,0,1,3,0', 'DATA,0,0,-1,5,0', 'DATA,0,0,2,1,0',
                    'DATA,0,0,2,2', 'DATA,0,0,2,NaN,0', 'DATA,0,0,2,2.0,0'):
            with self.subTest(row=bad), self.assertRaises(ValueError):
                self.parse(self.output().replace('DATA,0,0,2,2,0', bad))
        with self.assertRaisesRegex(ValueError, 'absent genotype'):
            self.parse(self.output().replace('DATA,4,0,2,2,0', 'DATA,4,0,1,2,1'))

    def test_python_parameters_before_subprocess(self):
        invalid = [{'n': True}, {'n': 1}, {'n': 201}, {'n': 3.5}, {'het': -1},
                   {'het': 41}, {'hom': 21}, {'hom': False}, {'clone_rate': True},
                   {'clone_rate': -1}, {'clone_rate': math.inf}, {'clone_rate': math.nan},
                   {'steps': 0}, {'steps': 401}, {'replicates': 0}, {'seed': 0}]
        with patch.object(d, 'check_slim_version') as check:
            for kwargs in invalid:
                with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                    d.run(**kwargs)
            check.assert_not_called()

    def test_summarizer_rejects_invalid_rows(self):
        for rows in ([], [[2, 2, 0]], [[2, 2, 0], [2, 1, 0]], [[2, 2, 0], [2, True, 1]]):
            with self.assertRaises(ValueError):
                d.summarize(rows, 4, 2, 0, 1, 0)

    def test_measured_discrepancy_is_not_silently_accepted(self):
        with self.assertRaises(AssertionError):
            d.summarize([[0, 0, 40]]*512, 40, 20, 0, 0, 1)

    def test_wrong_simulator_version_rejected(self):
        completed = subprocess.CompletedProcess([], 0, 'SLiM version 5.2', '')
        with patch('subprocess.run', return_value=completed), self.assertRaises(ValueError):
            d.run()

    def test_checkpoints_are_deduplicated(self):
        self.assertEqual(d.checkpoints(1), [0, 1])
        self.assertEqual(d.checkpoints(4), [0, 1, 2, 4])


class IntegrationTests(unittest.TestCase):
    def test_actual_version(self):
        self.assertEqual(d.check_slim_version('slim'), '4.0.1')

    def test_seed_replay(self):
        self.assertEqual(d.run(n=8, het=4, steps=8, replicates=4, clone_rate=.5),
                         d.run(n=8, het=4, steps=8, replicates=4, clone_rate=.5))

    def test_clonal_all_heterozygotes_exact(self):
        for rows in d.run(n=8, het=8, steps=40, replicates=12).values():
            self.assertEqual(rows, [[0, 8, 0]]*12)

    def test_clonal_homozygotes_never_create_heterozygotes(self):
        for rows in d.run(n=8, het=0, hom=4, steps=40, replicates=24).values():
            self.assertTrue(all(row[1] == 0 for row in rows))

    def test_absorbing_homozygote_controls(self):
        for c, hom in itertools.product((0, .5, 1), (0, 8)):
            for rows in d.run(n=8, het=0, hom=hom, steps=10, replicates=3, clone_rate=c).values():
                self.assertEqual(rows, [[8-hom, 0, hom]]*3)

    def test_sexual_segregation(self):
        rows = d.run(n=40, het=40, steps=1, replicates=512, clone_rate=0)[1]
        self.assertTrue(all(sum(row[i] for row in rows) > 0 for i in range(3)))
        d.summarize(rows, 40, 40, 0, 0, 1)

    def test_mixed_reproduction_finite_means(self):
        output = d.run(n=20, het=8, hom=4, steps=12, replicates=512, clone_rate=.25)
        for t, rows in output.items():
            d.summarize(rows, 20, 8, 4, .25, t)

    def test_eidos_rejects_bad_parameters(self):
        for value in ('N=2.5', 'HET=-1', 'HOM=21', 'C=1.1', 'STEPS=401', 'REPLICATES=0'):
            result = subprocess.run(['slim', '-d', value, str(d.HERE/'model.slim')],
                                    capture_output=True, text=True, timeout=30)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('invalid', result.stderr)

    def test_saved_output_replayed_exactly(self):
        self.assertEqual(d.summary(), json.loads((d.HERE/'sample_summary.json').read_text()))


if __name__ == '__main__':
    unittest.main()
