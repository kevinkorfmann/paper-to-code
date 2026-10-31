"""Real Beagle integration plus exact independent copying/probability controls."""

import copy
from fractions import Fraction
import json
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from examples.inference_showcase.beagle_imputation import demo, install


class BeagleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = json.loads(demo.FIXTURE.read_text())
        cls.cases = {m: demo.run_case(cls.fixture, m) for m in demo.MODES}
        cls.summary = demo.summarize(cls.fixture, cls.cases)
        cls.typed = set(cls.fixture['typed_indices'])
        cls.hidden = [i for i in range(200) if i not in cls.typed]

    def test_fixture_regenerates_exactly(self):
        self.assertEqual(self.fixture, demo.generate_input())

    def test_input_dimensions(self):
        self.assertEqual(len(self.fixture['positions']), 200)
        self.assertEqual(len(self.typed), 51)
        self.assertEqual(len(self.hidden), 149)
        self.assertEqual(self.summary['input']['hidden_genotypes'], 447)

    def test_hidden_truth_never_written_to_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp); demo.write_inputs(p, self.fixture, 'matched')
            records = [r.split('\t') for r in (p/'target.vcf').read_text().splitlines() if not r.startswith('#')]
            self.assertEqual({int(r[2][1:]) for r in records}, self.typed)
            self.assertEqual(len(records), 51)
            self.assertEqual(set(x.name for x in p.iterdir()), {'target.vcf', 'reference.vcf', 'map.txt'})

    def test_target_and_tag_reference_are_identical_between_controls(self):
        observed = []; tag_lines = []
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            for mode in demo.MODES:
                demo.write_inputs(p, self.fixture, mode)
                observed.append((p/'target.vcf').read_bytes())
                tag_lines.append([r for r in (p/'reference.vcf').read_text().splitlines()
                                  if not r.startswith('#') and int(r.split('\t')[2][1:]) in self.typed])
        self.assertEqual(observed, [observed[0]]*3)
        self.assertEqual(tag_lines, [tag_lines[0]]*3)

    def test_reference_frequencies_are_half_at_every_site(self):
        for mode in demo.MODES:
            ref, _ = demo.haplotypes(self.fixture, mode)
            self.assertEqual([sum(r[i] for r in ref) for i in range(200)], [40]*200)

    def test_reference_change_is_only_hidden_alleles(self):
        matched, _ = demo.haplotypes(self.fixture, 'matched')
        flipped, _ = demo.haplotypes(self.fixture, 'flipped')
        for a, b in zip(matched, flipped):
            for i in range(200): self.assertEqual(b[i], a[i] if i in self.typed else 1-a[i])

    def test_uninformative_exact_copy_dosage_is_one(self):
        ref, targets = demo.haplotypes(self.fixture, 'uninformative')
        oracle = demo.exact_copy_dosages(ref, targets, sorted(self.typed))
        self.assertEqual({oracle[s][i] for s in range(3) for i in self.hidden}, {Fraction(1)})

    def test_exact_copy_oracle_hand_enumeration(self):
        # Three compatible reference chromosomes for tag 0; one carries hidden ALT.
        ref = [[0, 0], [0, 0], [0, 1], [1, 1]]
        target = [[[0, 1], [1, 0]]]
        value = demo.exact_copy_dosages(ref, target, [0])
        self.assertEqual(value, [[Fraction(1), Fraction(4, 3)]])

    def test_copy_oracle_rejects_no_compatible_haplotype(self):
        with self.assertRaises(ValueError):
            demo.exact_copy_dosages([[0, 0]], [[[1, 0], [1, 0]]], [0])

    def test_frequency_only_baseline_exact_mse(self):
        self.assertEqual(demo.mse([1, 1, 1], [0, 1, 2]), float(Fraction(2, 3)))
        self.assertEqual(self.summary['frequency_only_baseline']['mse'], float(Fraction(2, 3)))

    def test_correlation_reversal_and_constant_control(self):
        self.assertEqual(demo.correlation([0, 1, 2], [2, 1, 0]), -1)
        self.assertIsNone(demo.correlation([1, 1, 1], [0, 1, 2]))
        for function in (demo.mse, demo.correlation):
            with self.assertRaises(ValueError): function([], [])
            with self.assertRaises(ValueError): function([0], [1, 2])

    def test_all_observed_genotypes_preserved(self):
        _, target = demo.haplotypes(self.fixture, 'matched')
        for mode in demo.MODES:
            for i in self.typed:
                for s, pair in enumerate(target):
                    self.assertEqual(self.cases[mode][i]['calls'][s]['gt'], [pair[0][i], pair[1][i]])

    def test_only_withheld_sites_flagged_imputed(self):
        for mode in demo.MODES:
            self.assertEqual([i for i, row in enumerate(self.cases[mode]) if row['imputed']], self.hidden)

    def test_executed_matched_and_flipped_errors(self):
        cases = self.summary['cases']
        self.assertEqual(cases['matched']['hidden_dosage_mse'], 0)
        self.assertEqual(cases['flipped']['hidden_dosage_mse'], float(Fraction(8, 3)))
        for mode in ('matched', 'flipped'):
            self.assertEqual(cases[mode]['max_dosage_distance_from_exact_copy'], 0)

    def test_signed_correlation_exposes_direction_that_r_squared_loses(self):
        cases = self.summary['cases']
        self.assertEqual(cases['matched']['signed_correlation'], 1)
        self.assertEqual(cases['flipped']['signed_correlation'], -1)
        for mode in ('matched', 'flipped'):
            self.assertEqual(cases[mode]['squared_correlation'], 1)
            self.assertEqual(cases[mode]['first_hidden_reported_dr2'], 1)

    def test_every_flipped_hidden_dosage_complements_matched(self):
        for i in self.hidden:
            for a, b in zip(self.cases['matched'][i]['calls'], self.cases['flipped'][i]['calls']):
                self.assertEqual(a['ds']+b['ds'], 2)

    def test_uninformative_panel_does_not_equal_ideal_copy_baseline(self):
        x = self.summary['cases']['uninformative']
        self.assertEqual(x['first_hidden_dosages'], [.92, .96, 1.])
        self.assertAlmostEqual(x['max_dosage_distance_from_exact_copy'], .08)
        self.assertAlmostEqual(x['hidden_dosage_mse'], .616)
        self.assertAlmostEqual(x['exact_copy_baseline_mse'], 2/3)

    def test_dosage_is_sum_of_allele_probabilities_at_printed_precision(self):
        for rows in self.cases.values():
            for row in rows:
                for call in row['calls']:
                    self.assertAlmostEqual(call['ds'], sum(call['ap']), delta=.021)

    def test_gp_matches_independent_bernoulli_calculation_at_printed_precision(self):
        for rows in self.cases.values():
            for row in rows:
                for call in row['calls']:
                    p, q = call['ap']; expected = [(1-p)*(1-q), p*(1-q)+(1-p)*q, p*q]
                    self.assertAlmostEqual(sum(call['gp']), 1, delta=.021)
                    self.assertAlmostEqual(call['gp'][1]+2*call['gp'][2], call['ds'], delta=.031)
                    for a, b in zip(call['gp'], expected): self.assertAlmostEqual(a, b, delta=.016)

    def test_versions_and_dependency_checksum(self):
        self.assertEqual(platform.python_version(), '3.14.5')
        self.assertTrue(install.runtime().is_file())
        self.assertEqual(self.summary['versions']['beagle'], '5.5 (27Feb25.75f)')
        self.assertEqual(self.summary['versions']['temurin_jre'], '17.0.21+9')
        install.verify(install.JAR.read_bytes(), install.JAR_SHA256)

    def test_runtime_pins_supported_platforms_only(self):
        for key in [('Darwin', 'arm64'), ('Linux', 'x86_64')]:
            name, digest, relative = install.package(*key)
            self.assertIn('17.0.21_9', name)
            self.assertEqual(len(digest), 64)
            self.assertTrue(relative.startswith('jdk-17.0.21+9-jre'))
        with self.assertRaises(RuntimeError): install.package('unsupported', 'unsupported')

    def test_checksum_rejects_altered_bytes(self):
        with self.assertRaises(ValueError): install.verify(b'changed', install.JAR_SHA256)

    def test_altered_cached_download_is_rejected_without_network(self):
        with tempfile.TemporaryDirectory() as tmp, patch('urllib.request.urlopen') as fetch:
            p = Path(tmp)/'artifact'; p.write_bytes(b'changed')
            with self.assertRaises(ValueError): install.fetch(install.JAR_URL, p, install.JAR_SHA256)
            fetch.assert_not_called()

    def test_invalid_fixture_values(self):
        for key, value in [('base_haplotype', '01x1'), ('base_haplotype', '01'),
                           ('reference_haplotypes', 3), ('reference_haplotypes', True),
                           ('seed', -1), ('target_classes', [[False, 0], [0, 1], [1, 1]])]:
            data = copy.deepcopy(self.fixture); data[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError): demo.validate(data)

    def test_invalid_positions_and_typed_indices(self):
        for key, value in [('positions', [1]*200), ('positions', list(reversed(self.fixture['positions']))),
                           ('typed_indices', [0, 0, 199]), ('typed_indices', [0, 200]),
                           ('typed_indices', [1, 199]), ('typed_indices', list(range(200)))]:
            data = copy.deepcopy(self.fixture); data[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): demo.validate(data)

    def test_invalid_mode_and_seed(self):
        with self.assertRaises(ValueError): demo.haplotypes(self.fixture, 'unknown')
        for seed in [-1, 2**31, True, .5]:
            with self.assertRaises(ValueError): demo.run_case(self.fixture, 'matched', seed)

    def test_output_parser_rejects_incomplete_records(self):
        for text in ['', 'not a VCF', '#CHROM\tPOS\n']:
            with self.assertRaises(ValueError): demo.parse_vcf(text, [2000])

    def test_output_parser_rejects_invalid_probabilities_and_order(self):
        head = '#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tAA\tAB\tBB\n'
        cell = '0|0:0:0:0:1,0,0'
        row = '1\t2000\tm1\tA\tG\t.\tPASS\tDR2=1;IMP\tGT:DS:AP1:AP2:GP\t'+'\t'.join([cell]*3)+'\n'
        self.assertEqual(demo.parse_vcf(head+row, [2000])[0]['calls'][0]['ds'], 0)
        for invalid in [row.replace(':0:0:0:', ':nan:0:0:'), row.replace('DR2=1', 'DR2=inf'),
                        row.replace('1,0,0', '2,-1,0'), row.replace('0|0', '0/0'), row+row]:
            with self.assertRaises(ValueError): demo.parse_vcf(head+invalid, [2000])

    def test_map_has_explicit_centimorgan_units(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp); demo.write_inputs(p, self.fixture, 'matched')
            self.assertEqual((p/'map.txt').read_text(), '1 m0 0 1000\n1 m199 0.199 200000\n')

    def test_sample_matches_executed_summary(self):
        self.assertEqual(demo.render(self.summary), demo.SAMPLE.read_text())

    def test_cli_and_invalid_flag(self):
        good = subprocess.run([sys.executable, '-m', 'examples.inference_showcase.beagle_imputation.demo', '--check-sample'], capture_output=True, text=True, timeout=30)
        self.assertEqual(good.returncode, 0, good.stderr)
        self.assertEqual(good.stdout, demo.SAMPLE.read_text())
        bad = subprocess.run([sys.executable, '-m', 'examples.inference_showcase.beagle_imputation.demo', '--unknown'], capture_output=True, text=True, timeout=10)
        self.assertEqual(bad.returncode, 2)

    def test_same_seed_repeats_exactly(self):
        self.assertEqual(demo.run_case(self.fixture, 'uninformative'), self.cases['uninformative'])


if __name__ == '__main__':
    unittest.main()
