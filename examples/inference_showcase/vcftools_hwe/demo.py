"""Run VCFtools --hardy and compare its output with an independent exact oracle."""

import argparse
from fractions import Fraction
import json
import math
from pathlib import Path
import platform
import subprocess
import tempfile

from .install import BINARY, check_version
from .reference import (calibration, exact_tests, pearson_test,
                        plug_in_expectations, validate_counts)

HERE = Path(__file__).resolve().parent
GENOTYPES = {'0/0', '0/1', '1/0', '1/1', '0|0', '0|1', '1|0', '1|1', './.'}
HEADER = ['CHR', 'POS', 'OBS(HOM1/HET/HOM2)', 'E(HOM1/HET/HOM2)',
          'ChiSq_HWE', 'P_HWE', 'P_HET_DEFICIT', 'P_HET_EXCESS']


def calls_from_counts(counts):
    return [gt for gt, n in zip(('0/0', '0/1', '1/1'), validate_counts(counts))
            for _ in range(n)]


def write_vcf(path, rows):
    rows = [list(row) for row in rows]
    if not rows or any(not row or any(gt not in GENOTYPES for gt in row)
                       or all(gt == './.' for gt in row) for row in rows):
        raise ValueError('Require nonempty rows of supported diploid GTs with called genotypes')
    n = max(map(len, rows))
    if n > 1000:
        raise ValueError('Educational input is limited to 1000 samples')
    lines = ['##fileformat=VCFv4.2', '##source=original_synthetic_hwe_example',
             '##contig=<ID=1,length=1000000>',
             '##FORMAT=<ID=GT,Number=1,Type=String,Description="Synthetic diploid genotype">',
             '\t'.join(['#CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL', 'FILTER',
                        'INFO', 'FORMAT'] + [f'synthetic_{i:04d}' for i in range(n)])]
    for i, row in enumerate(rows, 1):
        lines.append('\t'.join(['1', str(i), f'case_{i}', 'A', 'C', '.', 'PASS', '.', 'GT']
                               + row + ['./.'] * (n - len(row))))
    Path(path).write_text('\n'.join(lines) + '\n')


def parse_hwe(text, expected_rows):
    lines = text.splitlines()
    if not lines or lines[0].split('\t') != HEADER or len(lines) != expected_rows + 1:
        raise ValueError('Unexpected VCFtools HWE header or number of records')
    result = []
    for i, line in enumerate(lines[1:], 1):
        parts = line.split('\t')
        if len(parts) != 8 or parts[:2] != ['1', str(i)]:
            raise ValueError('Unexpected output site or column count')
        counts = tuple(map(int, parts[2].split('/')))
        validate_counts(counts)
        expected = list(map(float, parts[3].split('/')))
        chi = float(parts[4])
        probabilities = list(map(float, parts[5:]))
        if len(expected) != 3 or any(not math.isfinite(v) or v < 0 for v in expected):
            raise ValueError('Invalid expected counts')
        if any(not math.isfinite(v) or not 0 <= v <= 1 for v in probabilities):
            raise ValueError('Invalid p-value')
        monomorphic = counts[1] == 0 and (counts[0] == 0 or counts[2] == 0)
        if (not math.isfinite(chi) and not (math.isnan(chi) and monomorphic)) or chi < 0:
            raise ValueError('Unexpected chi-square statistic')
        result.append({'counts': list(counts), 'expected_counts_printed': expected,
                       'chi_square': chi if math.isfinite(chi) else None,
                       'p_two_sided': probabilities[0], 'p_deficit': probabilities[1],
                       'p_excess': probabilities[2]})
    return result


def run_tool(rows, binary=BINARY):
    check_version(binary)
    with tempfile.TemporaryDirectory(prefix='vcftools-hwe-') as temporary:
        directory = Path(temporary)
        vcf = directory / 'synthetic.vcf'
        write_vcf(vcf, rows)
        prefix = directory / 'result'
        subprocess.run([str(binary), '--vcf', str(vcf), '--hardy', '--out', str(prefix)],
                       capture_output=True, text=True, check=True, timeout=30)
        return parse_hwe(prefix.with_suffix('.hwe').read_text(), len(rows))


def verify_result(record, counts):
    counts = validate_counts(counts)
    if record['counts'] != list(counts):
        raise ValueError('Tool genotype counts differ from constructed input')
    exact = exact_tests(counts)
    for field, oracle in [('p_two_sided', 'two_sided'), ('p_deficit', 'deficit'),
                          ('p_excess', 'excess')]:
        # VCFtools prints seven significant digits; preserve small p-values.
        if not math.isclose(record[field], float(exact[oracle]), rel_tol=5.1e-7, abs_tol=1e-300):
            raise ValueError(f'Independent exact check failed: {field}, {counts}')
    for actual, expected in zip(record['expected_counts_printed'], plug_in_expectations(counts)):
        if abs(actual - float(expected)) > 0.00500001:
            raise ValueError('Expected genotype count check failed')
    chi, _ = pearson_test(counts)
    if chi is None:
        if record['chi_square'] is not None:
            raise ValueError('Expected undefined monomorphic chi-square')
    elif not math.isclose(record['chi_square'], chi, rel_tol=5.1e-7, abs_tol=1e-12):
        raise ValueError('Independent Pearson statistic check failed')


def rounded(value):
    return None if value is None else float(f'{float(value):.10g}')


def summary(binary=BINARY):
    if platform.python_implementation() != 'CPython' or platform.python_version() != '3.14.5':
        raise ValueError('Checked environment requires CPython 3.14.5')
    fixture = json.loads((HERE / 'synthetic_input.json').read_text())
    cases = fixture['cases']
    settings = fixture['calibration']
    null = calibration(settings['diploid_individuals'], settings['alt_copies'],
                       Fraction(settings['alpha']))
    all_counts = [case['counts'] for case in cases] + [row['counts'] for row in null['configurations']]
    actual = run_tool([calls_from_counts(c) for c in all_counts], binary)
    for counts, record in zip(all_counts, actual):
        verify_result(record, counts)
    reports = []
    for case, record in zip(cases, actual):
        reports.append({'name': case['name'], **record,
                        'asymptotic_chi_square_p': rounded(pearson_test(case['counts'])[1])})
    calibration_rows = []
    tool_rejection_mass = Fraction()
    for row, record in zip(null['configurations'], actual[len(cases):]):
        if record['p_two_sided'] <= float(Fraction(settings['alpha'])):
            tool_rejection_mass += row['null_probability']
        calibration_rows.append({'counts': list(row['counts']),
                                 'conditional_null_probability': rounded(row['null_probability']),
                                 'vcftools_p_two_sided': record['p_two_sided'],
                                 'asymptotic_chi_square_p': rounded(row['chi_square_p'])})
    if tool_rejection_mass != null['exact_rejection_mass']:
        raise ValueError('Tool rejection set disagrees with exact conditional calibration')
    return {'versions': {'python': platform.python_version(), 'vcftools': check_version(binary)},
            'seed': None, 'input': 'Constructed counts; unrelated synthetic diploids under the null',
            'inference_target': 'Conditional exact test of Hardy-Weinberg genotype proportions',
            'counts_order': ['0/0', '0/1', '1/1'], 'cases': reports,
            'calibration': {**settings, 'configurations': calibration_rows,
                            'exact_null_rejection_probability': rounded(tool_rejection_mass),
                            'exact_null_rejection_fraction': str(tool_rejection_mass),
                            'chi_square_null_rejection_probability': rounded(null['chi_square_rejection_mass']),
                            'chi_square_null_rejection_fraction': str(null['chi_square_rejection_mass'])},
            'rare_homozygote_exact_p_fraction': str(exact_tests((99, 0, 1))['two_sided']),
            'records_independently_checked': len(actual),
            'limitations': 'Per-site conditional hypothesis tests; no causal diagnosis, posterior HWE probability, '
                           'genome-wide error rate, fitted inbreeding coefficient or empirical accuracy claim.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vcftools', type=Path, default=BINARY)
    parser.add_argument('--check-sample', action='store_true')
    args = parser.parse_args()
    output = json.dumps(summary(args.vcftools), indent=2, allow_nan=False) + '\n'
    if args.check_sample and output != (HERE / 'sample_summary.json').read_text():
        raise SystemExit('Checked sample differs from actual output')
    print(output, end='')


if __name__ == '__main__':
    main()
