"""Impute withheld markers; keep truth fixed while changing only the reference."""

import argparse
from fractions import Fraction
import gzip
import json
import math
from pathlib import Path
import platform
import random
import subprocess
import tempfile

from . import install

SEED = 20261031
FOLDER = Path(__file__).parent
FIXTURE = FOLDER / 'synthetic_input.json'
SAMPLE = FOLDER / 'sample_summary.json'
MODES = ('matched', 'flipped', 'uninformative')
SAMPLES = ('AA', 'AB', 'BB')


def generate_input():
    rng = random.Random(SEED)
    return {'seed': SEED, 'base_haplotype': ''.join(str(rng.randrange(2)) for _ in range(200)),
            'positions': [1000+1000*i for i in range(200)],
            'typed_indices': [i for i in range(200) if i % 4 == 0 or i == 199],
            'reference_haplotypes': 80, 'target_classes': [[0, 0], [0, 1], [1, 1]]}


def validate(fixture):
    base = fixture['base_haplotype']; positions = fixture['positions']; typed = fixture['typed_indices']
    if type(fixture['seed']) is not int or not 0 <= fixture['seed'] < 2**31:
        raise ValueError('Expected a nonnegative 32-bit data seed')
    if not isinstance(base, str) or len(base) < 4 or set(base) - {'0', '1'}:
        raise ValueError('Expected a binary base haplotype')
    if len(positions) != len(base) or any(type(x) is not int or x < 1 for x in positions):
        raise ValueError('Expected one positive integer position per marker')
    if positions != sorted(set(positions)):
        raise ValueError('Positions must be unique and increasing')
    if any(type(x) is not int or not 0 <= x < len(base) for x in typed) or typed != sorted(set(typed)):
        raise ValueError('Typed indices must be unique, increasing, and in range')
    if len(typed) < 2 or len(typed) >= len(base) or typed[0] != 0 or typed[-1] != len(base)-1:
        raise ValueError('Require both endpoints typed and some interior markers hidden')
    n = fixture['reference_haplotypes']
    if type(n) is not int or n < 4 or n % 4:
        raise ValueError('Reference haplotype count must be a positive multiple of four')
    if (fixture['target_classes'] != [[0, 0], [0, 1], [1, 1]]
            or any(type(x) is not int for pair in fixture['target_classes'] for x in pair)):
        raise ValueError('This controlled example uses the AA, AB, BB target pairs')


def haplotypes(fixture, mode):
    validate(fixture)
    if mode not in MODES:
        raise ValueError('Unknown reference control')
    base = list(map(int, fixture['base_haplotype'])); typed = set(fixture['typed_indices'])
    reference = []
    for h in range(fixture['reference_haplotypes']):
        reference.append([b ^ (h % 2 if i in typed or mode == 'matched'
                               else (1-h % 2 if mode == 'flipped' else (h//2) % 2))
                          for i, b in enumerate(base)])
    targets = [[[b ^ c for b in base] for c in pair] for pair in fixture['target_classes']]
    return reference, targets


def write_vcf(path, positions, indices, names, pairs):
    lines = ['##fileformat=VCFv4.3', '##contig=<ID=1>',
             '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
             '\t'.join(['#CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL', 'FILTER', 'INFO', 'FORMAT', *names])]
    for i in indices:
        calls = [f'{a[i]}|{b[i]}' for a, b in pairs]
        lines.append('\t'.join(['1', str(positions[i]), f'm{i}', 'A', 'G', '.', 'PASS', '.', 'GT', *calls]))
    path.write_text('\n'.join(lines)+'\n')


def write_inputs(path, fixture, mode):
    reference, targets = haplotypes(fixture, mode)
    positions = fixture['positions']; n = len(positions)
    pairs = list(zip(reference[::2], reference[1::2]))
    write_vcf(path/'reference.vcf', positions, range(n), [f'R{i}' for i in range(len(pairs))], pairs)
    # Only typed sites reach Beagle; withheld truth is never a tool input.
    write_vcf(path/'target.vcf', positions, fixture['typed_indices'], SAMPLES, targets)
    length_cm = (positions[-1]-positions[0])/1_000_000
    (path/'map.txt').write_text(f'1 m0 0 {positions[0]}\n1 m{n-1} {length_cm} {positions[-1]}\n')


def parse_vcf(text, expected_positions):
    rows = []; names = None
    for line in text.splitlines():
        if line.startswith('##'):
            continue
        if line.startswith('#CHROM'):
            names = line.split('\t')[9:]
            continue
        fields = line.split('\t')
        if names != list(SAMPLES) or len(fields) != 12 or fields[0] != '1' or fields[3:5] != ['A', 'G']:
            raise ValueError('Unexpected VCF shape, samples, chromosome, or alleles')
        keys = fields[8].split(':')
        if not {'GT', 'DS', 'AP1', 'AP2', 'GP'} <= set(keys):
            raise ValueError('Missing requested inference fields')
        calls = []
        for cell in fields[9:]:
            values = dict(zip(keys, cell.split(':'), strict=True))
            gt = values['GT'].split('|')
            if len(gt) != 2 or any(x not in ('0', '1') for x in gt):
                raise ValueError('Expected complete phased biallelic calls')
            call = {'gt': list(map(int, gt)), 'ds': float(values['DS']),
                    'ap': [float(values['AP1']), float(values['AP2'])],
                    'gp': [float(x) for x in values['GP'].split(',')]}
            if (not math.isfinite(call['ds']) or not 0 <= call['ds'] <= 2 or len(call['gp']) != 3
                    or any(not math.isfinite(x) or not 0 <= x <= 1 for x in call['ap']+call['gp'])):
                raise ValueError('Invalid probabilities or dosage')
            calls.append(call)
        info = dict(x.split('=', 1) if '=' in x else (x, True) for x in fields[7].split(';'))
        dr2 = float(info['DR2']) if 'DR2' in info else None
        if dr2 is not None and (not math.isfinite(dr2) or not 0 <= dr2 <= 1):
            raise ValueError('Invalid DR2')
        rows.append({'position': int(fields[1]), 'imputed': 'IMP' in info,
                     'dr2': dr2, 'calls': calls})
    if [x['position'] for x in rows] != expected_positions:
        raise ValueError('Missing, duplicated, or reordered output markers')
    return rows


def run_case(fixture, mode, seed=SEED):
    validate(fixture)
    if type(seed) is not int or not 0 <= seed < 2**31:
        raise ValueError('Expected a nonnegative 32-bit integer seed')
    java = install.runtime()
    with tempfile.TemporaryDirectory(prefix='beagle-imputation-') as tmp:
        path = Path(tmp); write_inputs(path, fixture, mode)
        command = [str(java), '-Xmx256m', '-jar', str(install.JAR),
                   f'gt={path}/target.vcf', f'ref={path}/reference.vcf', f'map={path}/map.txt',
                   f'out={path}/inferred', 'ne=10000', 'err=0.0001', 'em=false',
                   f'seed={seed}', 'nthreads=1', 'ap=true', 'gp=true', 'impute=true',
                   'cluster=0.005', 'imp-states=1600', 'window=40', 'overlap=2']
        process = subprocess.run(command, capture_output=True, text=True, timeout=60)
        if process.returncode:
            raise RuntimeError('Beagle failed: '+process.stdout[-2000:]+process.stderr[-2000:])
        if 'beagle.27Feb25.75f.jar (version 5.5)' not in process.stdout:
            raise RuntimeError('Unexpected Beagle version')
        with gzip.open(path/'inferred.vcf.gz', 'rt') as stream:
            return parse_vcf(stream.read(), fixture['positions'])


def exact_copy_dosages(reference, targets, typed):
    """Independent zero-recombination/error copying baseline, NOT Beagle's HMM."""
    means = []
    for pair in targets:
        expected = [Fraction(0) for _ in pair[0]]
        for hap in pair:
            compatible = [r for r in reference if all(r[i] == hap[i] for i in typed)]
            if not compatible:
                raise ValueError('No compatible reference haplotype')
            for i in range(len(hap)):
                expected[i] += Fraction(sum(r[i] for r in compatible), len(compatible))
        means.append(expected)
    return means


def mse(x, y):
    if not x or len(x) != len(y):
        raise ValueError('MSE needs equal nonempty vectors')
    return math.fsum((float(a)-float(b))**2 for a, b in zip(x, y))/len(x)


def correlation(x, y):
    if not x or len(x) != len(y):
        raise ValueError('Correlation needs equal nonempty vectors')
    mx, my = math.fsum(x)/len(x), math.fsum(y)/len(y)
    xx, yy = math.fsum((a-mx)**2 for a in x), math.fsum((b-my)**2 for b in y)
    if xx == 0 or yy == 0:
        return None
    return math.fsum((a-mx)*(b-my) for a, b in zip(x, y))/math.sqrt(xx*yy)


def summarize(fixture=None, cases=None):
    fixture = json.loads(FIXTURE.read_text()) if fixture is None else fixture
    validate(fixture)
    cases = {mode: run_case(fixture, mode) for mode in MODES} if cases is None else cases
    typed = set(fixture['typed_indices']); hidden = [i for i in range(len(fixture['positions'])) if i not in typed]
    _, targets = haplotypes(fixture, 'matched')
    truth = [sum(pair[j][i] for j in (0, 1)) for i in hidden for pair in targets]
    result = {'versions': {'python': platform.python_version(), 'beagle': install.BEAGLE_VERSION,
                          'temurin_jre': install.JAVA_VERSION},
              'data_seed': fixture['seed'], 'beagle_seed': SEED, 'threads': 1,
              'input': {'reference_haplotypes': fixture['reference_haplotypes'], 'target_diploids': 3,
                        'markers': len(fixture['positions']), 'typed_markers': len(typed),
                        'hidden_markers': len(hidden), 'hidden_genotypes': len(truth)},
              'frequency_only_baseline': {'dosage': 1.0, 'mse': mse([1.]*len(truth), truth)},
              'cases': {}}
    for mode in MODES:
        rows = cases[mode]; reference, _ = haplotypes(fixture, mode)
        oracle = exact_copy_dosages(reference, targets, sorted(typed))
        values = [call['ds'] for i in hidden for call in rows[i]['calls']]
        expected = [float(oracle[s][i]) for i in hidden for s in range(3)]
        corr = correlation(values, truth)
        first = rows[hidden[0]]
        result['cases'][mode] = {
            'hidden_dosage_mse': mse(values, truth), 'signed_correlation': corr,
            'squared_correlation': None if corr is None else corr*corr,
            'exact_copy_baseline_mse': mse(expected, truth),
            'max_dosage_distance_from_exact_copy': max(abs(a-b) for a, b in zip(values, expected)),
            'first_hidden_position': first['position'],
            'first_hidden_truth': [sum(pair[j][hidden[0]] for j in (0, 1)) for pair in targets],
            'first_hidden_dosages': [x['ds'] for x in first['calls']],
            'first_hidden_reported_dr2': first['dr2'],
            'typed_genotypes_preserved': all(sorted(rows[i]['calls'][s]['gt']) == sorted([pair[0][i], pair[1][i]])
                                             for i in typed for s, pair in enumerate(targets))}
    return result


def render(result):
    def rounded(x):
        if isinstance(x, float): return round(x, 9)
        if isinstance(x, dict): return {k: rounded(v) for k, v in x.items()}
        if isinstance(x, list): return [rounded(v) for v in x]
        return x
    return json.dumps(rounded(result), indent=2, allow_nan=False)+'\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-sample', action='store_true')
    args = parser.parse_args()
    text = render(summarize())
    if args.check_sample and text != SAMPLE.read_text():
        parser.error('Output differs from the checked sample')
    print(text, end='')


if __name__ == '__main__':
    main()
