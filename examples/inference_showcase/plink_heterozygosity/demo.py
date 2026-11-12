"""Original synthetic PLINK --het frequency and missingness sensitivity example."""
import argparse
from fractions import Fraction
import json
import math
from pathlib import Path
import platform
import subprocess
import tempfile
from ..plink_king.install import BINARY, BINARY_SHA256, VERSION, verify
HERE = Path(__file__).resolve().parent


def validate(calls, frequencies):
    if not isinstance(calls, (list, tuple)) or not 1 <= len(calls) <= 1000:
        raise ValueError('require 1 to 1000 calls')
    if len(calls) != len(frequencies):
        raise ValueError('one frequency per call required')
    if any(g is not None and (type(g) is not int or g not in (0, 1, 2)) for g in calls):
        raise ValueError('calls must be diploid ALT counts 0, 1, 2, or None')
    if not any(g is not None for g in calls):
        raise ValueError('at least one observed call required')
    for p in frequencies:
        if isinstance(p, bool) or not math.isfinite(float(p)) or not 0 < p < 1:
            raise ValueError('frequencies must be finite and strictly between 0 and 1')


def oracle(calls, frequencies):
    validate(calls, frequencies)
    ps = [Fraction(str(p)) for g, p in zip(calls, frequencies) if g is not None]
    n = len(ps)
    observed = sum(g == 1 for g in calls)
    expected = sum(2*p*(1-p) for p in ps)
    return {'OBS_CT': n, 'O(HOM)': n-observed, 'E(HOM)': n-expected,
            'O(HET)': observed, 'E(HET)': expected, 'F': 1-Fraction(observed)/expected}


def inputs(calls, frequencies):
    validate(calls, frequencies)
    vcf = ['##fileformat=VCFv4.2', '##contig=<ID=1>',
           '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
           '#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ttarget']
    freq = ['#CHROM\tID\tREF\tALT\tALT_FREQS\tOBS_CT']
    for j, (g,p) in enumerate(zip(calls, frequencies), 1):
        gt = './.' if g is None else ('0/0', '0/1', '1/1')[g]
        vcf.append(f'1\t{j}\tv{j}\tA\tC\t.\tPASS\t.\tGT\t{gt}')
        freq.append(f'1\tv{j}\tA\tC\t{float(p):.17g}\t10000')
    return '\n'.join(vcf)+'\n', '\n'.join(freq)+'\n'


def tool(calls, frequencies):
    vcf, freq = inputs(calls, frequencies)
    verify(BINARY.read_bytes(), BINARY_SHA256)
    with tempfile.TemporaryDirectory(prefix='plink-het-') as directory:
        root = Path(directory)
        (root/'input.vcf').write_text(vcf)
        (root/'freq.afreq').write_text(freq)
        command = [str(BINARY), '--vcf', str(root/'input.vcf'), '--read-freq', str(root/'freq.afreq'),
                   '--het', 'cols=hom,het,nobs,f', '--threads', '1', '--memory', '640',
                   '--out', str(root/'result')]
        proc = subprocess.run(command, capture_output=True, text=True, timeout=30)
        if proc.returncode or VERSION not in proc.stdout:
            raise RuntimeError(proc.stdout+proc.stderr)
        rows = [line.split() for line in (root/'result.het').read_text().splitlines()]
        if len(rows) != 2 or rows[0] != ['#IID', 'O(HOM)', 'E(HOM)', 'O(HET)', 'E(HET)', 'OBS_CT', 'F'] or rows[1][0] != 'target':
            raise RuntimeError('unexpected PLINK output format')
        result = dict(zip(rows[0][1:], map(float, rows[1][1:])))
        if not all(math.isfinite(x) for x in result.values()):
            raise RuntimeError('nonfinite PLINK output')
        return result


def run():
    calls = [0,1,1,2]*5
    scenarios = {
        'balanced_reference': (calls, [.5]*20),
        'rare_ALT_reference': (calls, [.1]*20),
        'rare_REF_reference': (calls, [.9]*20),
        'mask_heterozygotes': ([None if g == 1 else g for g in calls], [.5]*20),
        'mask_homozygotes': ([g if g == 1 else None for g in calls], [.5]*20),
        'copy_all_sites': (calls*2, [.5]*40),
    }
    results = {}
    for name, (gs, ps) in scenarios.items():
        actual = tool(gs, ps)
        exact = oracle(gs, ps)
        if any(not math.isclose(actual[k], float(v), abs_tol=5e-5, rel_tol=1e-5) for k,v in exact.items()):
            raise AssertionError((name, actual, exact))
        results[name] = {'plink': actual, 'exact_F': str(exact['F'])}
    return {'python': platform.python_version(), 'plink': VERSION, 'seed': None,
            'design': 'Deterministic 20-marker one-sample fixture; supplied synthetic reference frequencies.',
            'cases': results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-sample', action='store_true')
    args = parser.parse_args()
    text = json.dumps(run(), indent=2, sort_keys=True)+'\n'
    if args.check_sample and text != (HERE/'sample_summary.json').read_text():
        raise SystemExit('sample output mismatch')
    print(text, end='')

if __name__ == '__main__':
    main()
