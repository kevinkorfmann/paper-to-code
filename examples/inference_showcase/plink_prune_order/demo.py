"""Original three-marker LD chain: actual PLINK pruning vs exact subset search."""
import argparse
from fractions import Fraction
import itertools
import json
import math
from pathlib import Path
import platform
import subprocess
import tempfile

from ..plink_king.install import BINARY, BINARY_SHA256, VERSION, verify


def panel(repeats=16):
    """A,C are balanced independent 0/2 calls; B=(A+C)/2. No random sampling."""
    if type(repeats) is not int or not 13 <= repeats <= 100:
        raise ValueError('repeats must be an integer from 13 to 100 (at least 52 founders)')
    return [(a, (a+c)//2, c) for a, c in itertools.product((0, 2), repeat=2)
            for _ in range(repeats)]


def validate(rows, order):
    if (not isinstance(rows, (list, tuple)) or not 50 <= len(rows) <= 400
            or any(not isinstance(row, (list, tuple)) or len(row) != 3
                   or any(type(g) is not int or g not in (0, 1, 2) for g in row) for row in rows)):
        raise ValueError('require 50 to 400 complete diploid three-marker rows')
    if not isinstance(order, (list, tuple)) or sorted(order) != ['A', 'B', 'C']:
        raise ValueError('order must be a permutation of A,B,C')
    if any(len(set(col)) < 2 for col in zip(*rows)):
        raise ValueError('all markers must vary')


def exact_r2(rows):
    """Squared centered dosage correlation, with rational arithmetic."""
    validate(rows, ('A', 'B', 'C'))
    columns = list(zip(*rows))
    n = len(rows)
    centered = [[Fraction(x) - Fraction(sum(col), n) for x in col] for col in columns]
    return [[sum(x*y for x, y in zip(a, b))**2 /
             (sum(x*x for x in a)*sum(y*y for y in b))
             for b in centered] for a in centered]


def maximum_subsets(rows, threshold=Fraction(1, 5)):
    """Exhaustive mathematical comparator, not a replacement inference pipeline."""
    matrix = exact_r2(rows)
    valid = [subset for k in range(4) for subset in itertools.combinations(range(3), k)
             if all(matrix[i][j] <= threshold for i, j in itertools.combinations(subset, 2))]
    largest = max(map(len, valid))
    return [['ABC'[i] for i in subset] for subset in valid if len(subset) == largest]


def vcf_text(rows, order=('A', 'B', 'C')):
    validate(rows, order)
    lines = ['##fileformat=VCFv4.2', '##contig=<ID=1>',
             '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
             '#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t' +
             '\t'.join(f's{i:03d}' for i in range(len(rows)))]
    for pos, marker in enumerate(order, 1):
        k = 'ABC'.index(marker)
        lines.append(f'1\t{pos}\t{marker}\tA\tC\t.\tPASS\t.\tGT\t' +
                     '\t'.join(('0/0', '0/1', '1/1')[row[k]] for row in rows))
    return '\n'.join(lines) + '\n'


def tool(rows, order=('A', 'B', 'C'), mode=2, threshold=0.2):
    validate(rows, order)
    if type(mode) is not int or mode not in (1, 2):
        raise ValueError('mode must be 1 or 2')
    if isinstance(threshold, bool) or not math.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError('threshold must be finite and strictly between 0 and 1')
    verify(BINARY.read_bytes(), BINARY_SHA256)
    with tempfile.TemporaryDirectory(prefix='plink-prune-order-') as directory:
        root = Path(directory)
        (root/'input.vcf').write_text(vcf_text(rows, order))
        cmd = [str(BINARY), '--vcf', str(root/'input.vcf'), '--indep-pairwise', '3', '1',
               str(threshold), '--indep-order', str(mode), '--r2-unphased', 'square',
               '--threads', '1', '--memory', '640', '--out', str(root/'result')]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if proc.returncode or VERSION not in proc.stdout:
            raise RuntimeError('PLINK failed or unexpected version: ' + proc.stdout + proc.stderr)
        keep = (root/'result.prune.in').read_text().split()
        removed = (root/'result.prune.out').read_text().split()
        variant_order = (root/'result.unphased.vcor2.vars').read_text().split()
        matrix = [[float(x) for x in line.split()] for line in
                  (root/'result.unphased.vcor2').read_text().splitlines()]
        if (len(keep)+len(removed) != 3 or set(keep) & set(removed)
                or set(keep+removed) != set(order) or variant_order != list(order)
                or len(matrix) != 3 or any(len(row) != 3 for row in matrix)
                or any(not math.isfinite(x) for row in matrix for x in row)):
            raise RuntimeError('unexpected PLINK output layout')
        return {'retained': keep, 'removed': removed, 'r2_matrix': matrix}


def demo():
    rows = panel()
    return {'python': platform.python_version(), 'plink': VERSION,
            'construction': '16 repeats of each (A,C) in {0,2} x {0,2}; B=(A+C)/2',
            'founders': len(rows), 'positions_bp': [1, 2, 3], 'variant_order': ['A', 'B', 'C'],
            'allele_frequencies': [0.5, 0.5, 0.5], 'window_variants': 3, 'step_variants': 1,
            'r2_threshold': 0.2,
            'exact_r2_matrix': [[str(x) for x in row] for row in exact_r2(rows)],
            'order_1': tool(rows, mode=1), 'order_2': tool(rows, mode=2),
            'maximum_cardinality_subsets': maximum_subsets(rows),
            'threshold_0_6_order_2': tool(rows, mode=2, threshold=0.6),
            'note': 'A pruning-list diagnostic, not ancestry or demographic inference; no uncertainty claim.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check-sample', action='store_true')
    args = parser.parse_args()
    output = json.dumps(demo(), indent=2, sort_keys=True) + '\n'
    if args.check_sample:
        if output != Path(__file__).with_name('sample_output.json').read_text():
            raise SystemExit('sample output mismatch')
        print('sample output matches')
    else:
        print(output, end='')


if __name__ == '__main__':
    main()
