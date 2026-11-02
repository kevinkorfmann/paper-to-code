"""Independent enumeration, tool integration, sensitivity and output checks."""

from collections import Counter
from fractions import Fraction
from itertools import combinations
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from examples.inference_showcase.vcftools_hwe import demo, install, reference as ref


class OracleTests(unittest.TestCase):
    def test_distribution_normalizes_exactly(self):
        for n in range(1, 31):
            for a in range(2 * n + 1):
                self.assertEqual(sum(ref.conditional_distribution(n, a).values()), 1)

    def test_labeled_allele_enumeration_independent_of_combinatorial_formula(self):
        for n in range(1, 7):
            for a in range(2 * n + 1):
                counts = Counter()
                for placement in combinations(range(2 * n), a):
                    occupied = set(placement)
                    h = sum(((2 * j in occupied) != (2 * j + 1 in occupied)) for j in range(n))
                    counts[h] += 1
                actual = {h: Fraction(v, sum(counts.values())) for h, v in counts.items()}
                self.assertEqual(ref.conditional_distribution(n, a), actual)

    def test_allele_complement_invariance(self):
        for a in range(41):
            self.assertEqual(ref.conditional_distribution(20, a), ref.conditional_distribution(20, 40 - a))

    def test_feasible_support_and_parity(self):
        self.assertEqual(list(ref.conditional_distribution(10, 17)), [1, 3])
        self.assertEqual(list(ref.conditional_distribution(10, 4)), [0, 2, 4])

    def test_conditional_mean_differs_from_plugin_expectation(self):
        for n, a in [(20, 4), (100, 21), (10, 10)]:
            expected = sum(h * p for h, p in ref.conditional_distribution(n, a).items())
            self.assertEqual(expected, Fraction(a * (2 * n - a), 2 * n - 1))
            self.assertNotEqual(expected, Fraction(a * (2 * n - a), 2 * n))

    def test_exact_tests_are_superuniform_under_conditional_null(self):
        for n in range(2, 21):
            for a in range(1, n + 1):
                law = ref.conditional_distribution(n, a)
                for alpha in [Fraction(1, 100), Fraction(1, 20), Fraction(1, 2)]:
                    for statistic in ['two_sided', 'deficit', 'excess']:
                        rejected = sum(p for h, p in law.items()
                                       if ref.exact_tests((n - (a + h) // 2, h, (a - h) // 2))[statistic] <= alpha)
                        self.assertLessEqual(rejected, alpha)

    def test_inclusive_tail_identity(self):
        result = ref.exact_tests((4, 2, 4))
        self.assertEqual(result['deficit'] + result['excess'], 1 + result['observed_probability'])

    def test_rare_homozygote_closed_form(self):
        for n in [2, 20, 100, 1000]:
            self.assertEqual(ref.exact_tests((n - 1, 0, 1))['two_sided'], Fraction(1, 2 * n - 1))

    def test_probability_ordering_is_not_doubled_tail(self):
        result = ref.exact_tests((4, 2, 4))
        self.assertNotEqual(result['two_sided'], min(1, 2 * result['deficit'], 2 * result['excess']))

    def test_monomorphic_degeneracy(self):
        for counts in [(5, 0, 0), (0, 0, 5)]:
            self.assertEqual(set(ref.exact_tests(counts).values()), {Fraction(1)})
            self.assertEqual(ref.pearson_test(counts), (None, None))

    def test_invalid_genotype_counts(self):
        for counts in [[], [1, 2], [0, 0, 0], [1, -1, 2], [1.0, 2, 3], [True, 2, 3], [1001, 0, 0]]:
            with self.subTest(counts=counts), self.assertRaises(ValueError):
                ref.exact_tests(counts)

    def test_invalid_conditional_parameters(self):
        for pair in [(0, 0), (1001, 2), (3, -1), (3, 7), (True, 1), (3, 1.0)]:
            with self.subTest(pair=pair), self.assertRaises(ValueError):
                ref.conditional_distribution(*pair)

    def test_invalid_calibration_threshold(self):
        for alpha in [0.05, Fraction(0), Fraction(1), Fraction(-1)]:
            with self.assertRaises(ValueError):
                ref.calibration(20, 4, alpha)

    def test_known_pearson_and_expected_counts(self):
        self.assertEqual(ref.plug_in_expectations((40, 20, 40)), (25, 50, 25))
        chi, p = ref.pearson_test((40, 20, 40))
        self.assertEqual(chi, 36)
        self.assertAlmostEqual(p, math.erfc(6 / math.sqrt(2)), places=22)


class ToolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        install.check_version()
        cls.output = demo.summary()

    def test_all_small_genotype_tables_against_exact_oracle(self):
        tables = [(n - h - a, h, a) for n in range(1, 13)
                  for h in range(n + 1) for a in range(n - h + 1)]
        records = demo.run_tool([demo.calls_from_counts(c) for c in tables])
        self.assertEqual(len(records), 454)
        for counts, record in zip(tables, records):
            with self.subTest(counts=counts):
                demo.verify_result(record, counts)

    def test_tool_allele_label_symmetry(self):
        records = demo.run_tool([demo.calls_from_counts(c) for c in [(17, 2, 1), (1, 2, 17)]])
        for field in ['p_two_sided', 'p_deficit', 'p_excess', 'chi_square']:
            self.assertEqual(records[0][field], records[1][field])

    def test_tool_sample_order_invariance(self):
        row = demo.calls_from_counts((17, 2, 1))
        self.assertEqual(*demo.run_tool([row, list(reversed(row))]))

    def test_tool_phase_and_heterozygote_order_invariance(self):
        row = demo.calls_from_counts((17, 2, 1))
        changed = [gt.replace('0/1', '1/0').replace('/', '|') for gt in row]
        self.assertEqual(*demo.run_tool([row, changed]))

    def test_missing_padding_does_not_add_called_samples(self):
        row = demo.calls_from_counts((17, 2, 1))
        self.assertEqual(*demo.run_tool([row, row + ['./.'] * 30]))

    def test_tool_monomorphic_edge_has_no_chisquare_pvalue(self):
        result = demo.run_tool([demo.calls_from_counts((20, 0, 0))])[0]
        demo.verify_result(result, (20, 0, 0))
        self.assertIsNone(result['chi_square'])
        self.assertEqual(result['p_two_sided'], 1)

    def test_invalid_or_all_missing_vcf_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            for rows in [[], [[]], [['./.']], [['0']], [['0/2']], [['0/.']], [['0/1'] * 1001]]:
                with self.subTest(rows=rows[:1]), self.assertRaises(ValueError):
                    demo.write_vcf(Path(temporary) / 'test.vcf', rows)

    def test_vcf_input_preserves_counts_and_is_deterministic(self):
        with tempfile.TemporaryDirectory() as temporary:
            p = Path(temporary) / 'input.vcf'
            row = demo.calls_from_counts((4, 2, 4))
            demo.write_vcf(p, [row]); before = p.read_bytes()
            demo.write_vcf(p, [row]); self.assertEqual(before, p.read_bytes())
            self.assertEqual(Counter(p.read_text().splitlines()[-1].split('\t')[9:]),
                             {'0/0': 4, '0/1': 2, '1/1': 4})

    def test_malformed_tool_output_rejected(self):
        header = '\t'.join(demo.HEADER) + '\n'
        for text in ['', header, header + '1\t2\t1/2/1\t1/2/1\t0\t1\t1\t1\n',
                     header + '1\t1\t1/2/1\t1/2/1\t0\tnan\t1\t1\n',
                     header + '1\t1\t1/2/1\t1/2/1\t-1\t1\t1\t1\n',
                     header + '1\t1\t0/2/1\t1/2/1\tnan\t1\t1\t1\n',
                     header + '1\t1\t5/0/0\t5/0/0\tinf\t1\t1\t1\n']:
            with self.assertRaises(ValueError):
                demo.parse_hwe(text, 1)

    def test_mutated_tool_result_fails_independent_check(self):
        result = dict(self.output['cases'][0])
        result['p_two_sided'] = 0.9
        with self.assertRaises(ValueError):
            demo.verify_result(result, (25, 50, 25))

    def test_exact_runtime_version_enforced(self):
        self.assertEqual(install.check_version(), '0.1.17')
        with patch.object(demo.platform, 'python_version', return_value='3.14.4'):
            with self.assertRaises(ValueError):
                demo.summary()

    def test_sample_output_matches_executed_results(self):
        self.assertEqual(self.output, json.loads((demo.HERE / 'sample_summary.json').read_text()))

    def test_cli_default_output_matches_bytes(self):
        p = subprocess.run([sys.executable, '-m', 'examples.inference_showcase.vcftools_hwe.demo'],
                           capture_output=True, check=True, timeout=30)
        self.assertEqual(p.stdout, (demo.HERE / 'sample_summary.json').read_bytes())

    def test_cli_check_sample(self):
        p = subprocess.run([sys.executable, '-m', 'examples.inference_showcase.vcftools_hwe.demo', '--check-sample'],
                           capture_output=True, timeout=30)
        self.assertEqual(p.returncode, 0, p.stderr)

    def test_cli_unknown_argument_fails(self):
        p = subprocess.run([sys.executable, '-m', 'examples.inference_showcase.vcftools_hwe.demo', '--unknown'],
                           capture_output=True, timeout=30)
        self.assertEqual(p.returncode, 2)

    def test_sample_size_and_direction_controls(self):
        rows = {row['name']: row for row in self.output['cases']}
        self.assertGreater(rows['deficit_n10']['p_two_sided'], .05)
        self.assertLess(rows['deficit_n100']['p_two_sided'], 1e-8)
        self.assertEqual(rows['deficit_n100']['chi_square'], rows['excess_n100']['chi_square'])
        self.assertNotEqual(rows['deficit_n100']['p_two_sided'], rows['excess_n100']['p_two_sided'])
        self.assertEqual(self.output['rare_homozygote_exact_p_fraction'], '1/199')

    def test_genotype_dependent_removal_is_not_evidence_of_cause(self):
        rows = {row['name']: row for row in self.output['cases']}
        self.assertEqual(rows['balanced']['p_two_sided'], 1)
        self.assertEqual(rows['remove_40_heterozygotes']['counts'], [25, 10, 25])
        self.assertLess(rows['remove_40_heterozygotes']['p_two_sided'], 2e-7)

    def test_exact_conditional_calibration_of_actual_tool_decisions(self):
        cal = self.output['calibration']
        self.assertEqual(len(cal['configurations']), 3)
        self.assertEqual(Fraction(cal['exact_null_rejection_fraction']), Fraction(1, 481))
        self.assertEqual(Fraction(cal['chi_square_null_rejection_fraction']), Fraction(73, 481))
        self.assertLess(cal['exact_null_rejection_probability'], .05)
        self.assertGreater(cal['chi_square_null_rejection_probability'], .15)


if __name__ == '__main__':
    unittest.main()
