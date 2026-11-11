"""Original deterministic genotype panels, actual PLINK PCA, exact matrix oracle."""
import argparse
from fractions import Fraction
import itertools
import json
import math
from pathlib import Path
import platform
import struct
import subprocess
import tempfile

from ..plink_king.install import BINARY, BINARY_SHA256, VERSION, verify

HERE = Path(__file__).resolve().parent


def panel(a=2, b=1):
    """Factorial HWE dosage marginals; a and b are counts of copied marker patterns."""
    if any(type(w) is not int or not 1 <= w <= 100 for w in (a, b)):
        raise ValueError('pattern counts must be integers from 1 to 100')
    return {f's{i:02d}': [x]*a + [y]*b
            for i, (x, y) in enumerate(itertools.product((0, 1, 1, 2), repeat=2))}


def validate(samples):
    if not isinstance(samples, dict) or not 3 <= len(samples) <= 128:
        raise ValueError('require 3 to 128 named samples')
    lengths = set()
    for name, calls in samples.items():
        if not isinstance(name, str) or not name or not name.isascii() or not name.isalnum():
            raise ValueError('sample IDs must be nonempty ASCII letters/digits')
        if not isinstance(calls, (list, tuple)) or not 2 <= len(calls) <= 1000:
            raise ValueError('require 2 to 1000 marker calls per sample')
        if any(type(g) is not int or g not in (0, 1, 2) for g in calls):
            raise ValueError('require complete integer diploid biallelic calls 0, 1, 2')
        lengths.add(len(calls))
    if len(lengths) != 1:
        raise ValueError('all samples must have the same markers')
    if any(sum(col) in (0, 2*len(samples)) for col in zip(*samples.values())):
        raise ValueError('monomorphic sites cannot be variance-standardized')


def frequencies(samples):
    validate(samples)
    return [Fraction(sum(col), 2*len(samples)) for col in zip(*samples.values())]


def vcf_text(samples):
    validate(samples)
    rows = ['##fileformat=VCFv4.2', '##contig=<ID=1>',
            '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
            '#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t' + '\t'.join(samples)]
    for j, col in enumerate(zip(*samples.values()), 1):
        rows.append(f'1\t{j}\tv{j}\tA\tC\t.\tPASS\t.\tGT\t' +
                    '\t'.join(('0/0', '0/1', '1/1')[g] for g in col))
    return '\n'.join(rows) + '\n'


def freq_text(samples):
    # Explicit toy sample frequencies. This is not a large external reference panel.
    rows = ['#CHROM\tID\tREF\tALT\tALT_FREQS\tOBS_CT']
    for j, p in enumerate(frequencies(samples), 1):
        rows.append(f'1\tv{j}\tA\tC\t{float(p):.17g}\t{2*len(samples)}')
    return '\n'.join(rows) + '\n'


def exact_grm(samples):
    """R_ij = mean_k (g_ik-2p_k)(g_jk-2p_k)/(2p_k(1-p_k))."""
    ps = frequencies(samples)
    rows = list(samples.values())
    return [[sum((Fraction(x)-2*p)*(Fraction(y)-2*p)/(2*p*(1-p))
                 for x, y, p in zip(a, b, ps))/len(ps)
             for b in rows] for a in rows]


def normalized(v):
    norm = math.sqrt(sum(x*x for x in v))
    if not math.isfinite(norm) or norm == 0:
        raise ValueError('nonzero finite vector required')
    return [x/norm for x in v]


def dot(a, b):
    if len(a) != len(b):
        raise ValueError('equal vector lengths required')
    return sum(x*y for x, y in zip(a, b))


def projector(columns):
    """Projector for mutually orthogonal, nonzero columns; callers test orthogonality."""
    vectors = [normalized(v) for v in columns]
    return [[sum(v[i]*v[j] for v in vectors) for j in range(len(vectors[0]))]
            for i in range(len(vectors[0]))]


def matrix_error(a, b):
    if len(a) != len(b) or any(len(x) != len(y) for x, y in zip(a, b)):
        raise ValueError('equal matrix shapes required')
    return max(abs(float(x)-float(y)) for ar, br in zip(a, b) for x, y in zip(ar, br))


def axes():
    pairs = list(itertools.product((0, 1, 1, 2), repeat=2))
    return [normalized([row[k]-1 for row in pairs]) for k in (0, 1)]


def tool(samples):
    validate(samples)
    verify(BINARY.read_bytes(), BINARY_SHA256)
    with tempfile.TemporaryDirectory(prefix='plink-pca-weighting-') as directory:
        root = Path(directory)
        (root/'input.vcf').write_text(vcf_text(samples))
        (root/'freq.afreq').write_text(freq_text(samples))
        command = [str(BINARY), '--vcf', str(root/'input.vcf'), '--read-freq', str(root/'freq.afreq'),
                   '--pca', '2', '--make-rel', 'square', 'bin', '--threads', '1',
                   '--memory', '640', '--out', str(root/'result')]
        proc = subprocess.run(command, text=True, capture_output=True, timeout=30)
        if proc.returncode:
            raise RuntimeError(proc.stdout + proc.stderr)
        if VERSION not in proc.stdout:
            raise RuntimeError('unexpected PLINK version')
        eigenvalues = [float(x) for x in (root/'result.eigenval').read_text().split()]
        lines = [line.split() for line in (root/'result.eigenvec').read_text().splitlines()]
        if (len(eigenvalues) != 2 or not lines or lines[0] != ['#IID', 'PC1', 'PC2']
                or any(len(row) != 3 for row in lines[1:])
                or [r[0] for r in lines[1:]] != list(samples)):
            raise RuntimeError('unexpected eigenvector layout or sample order')
        columns = [normalized([float(row[j]) for row in lines[1:]]) for j in (1, 2)]
        data = (root/'result.rel.bin').read_bytes()
        n = len(samples)
        if len(data) != 8*n*n:
            raise RuntimeError('unexpected binary GRM size')
        flat = struct.unpack('<'+'d'*n*n, data)
        grm = [list(flat[i*n:(i+1)*n]) for i in range(n)]
        if not all(math.isfinite(v) for v in eigenvalues + list(flat) + sum(columns, [])):
            raise RuntimeError('nonfinite PLINK output')
        return {'eigenvalues': eigenvalues, 'columns': columns, 'grm': grm}


def clean(x):
    value = round(float(x), 6)
    return 0.0 if value == 0 else value


def run():
    a_axis, b_axis = axes()
    expected_projector = projector([a_axis, b_axis])
    cases = {}
    raw = {}
    for name, a, b in [('baseline', 2, 1), ('copy_B_three_more_times', 2, 4),
                       ('one_per_pattern', 1, 1), ('copy_every_marker', 4, 2)]:
        samples = panel(a, b)
        result = tool(samples)
        raw[name] = result
        cases[name] = {
            'pattern_counts': {'A': a, 'B': b}, 'markers': a+b,
            'plink_eigenvalues_printed': result['eigenvalues'],
            'exact_nonzero_eigenvalues': [str(v) for v in sorted(
                (Fraction(16*a, a+b), Fraction(16*b, a+b)), reverse=True)],
            'max_grm_error': clean(matrix_error(result['grm'], exact_grm(samples))),
            'max_two_PC_projector_error': clean(matrix_error(projector(result['columns']), expected_projector)),
        }
        # Individual vectors are not identifiable when eigenvalues tie.
        if a != b:
            cases[name]['PC1_absolute_alignment_A_B'] = [
                clean(abs(dot(result['columns'][0], v))) for v in (a_axis, b_axis)]
    return {'versions': {'python': platform.python_version(), 'plink': VERSION},
            'randomness': 'none; deterministic PCA, no --approx',
            'samples': 16, 'input': 'synthetic_genotypes.json',
            'allele_frequencies': '1/2 at every marker; supplied from this exact toy construction',
            'baseline_versus_B_copied_PC1_absolute_alignment': clean(abs(dot(
                raw['baseline']['columns'][0], raw['copy_B_three_more_times']['columns'][0]))),
            'cases': cases,
            'interpretation': 'Changing relative marker multiplicity swaps PC1, not the two-dimensional subspace.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-sample', action='store_true')
    args = parser.parse_args()
    if json.loads((HERE/'synthetic_genotypes.json').read_text()) != panel():
        raise AssertionError('checked synthetic input differs from deterministic generator')
    summary = run()
    if args.check_sample:
        if summary != json.loads((HERE/'sample_summary.json').read_text()):
            raise AssertionError('checked output changed')
        print('PLINK PCA checked input/output match')
    else:
        print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
