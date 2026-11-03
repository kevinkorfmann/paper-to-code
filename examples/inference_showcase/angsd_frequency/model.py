"""Original likelihood controls; ANGSD itself performs the reported inference."""
import csv
import gzip
import math
from pathlib import Path
import subprocess
import tempfile

from .install import BINARY, check_version


def validate(rows):
    rows = tuple(tuple(row) for row in rows)
    if not rows:
        raise ValueError('At least one diploid likelihood triple is required')
    for row in rows:
        if len(row) != 3 or any(not math.isfinite(x) or x <= 0 for x in row):
            raise ValueError('Likelihood triples must contain finite positive numbers')
    return rows


def log_likelihood(rows, p):
    rows = validate(rows)
    if not math.isfinite(p) or not 0 <= p <= 1:
        raise ValueError('Frequency must be in [0, 1]')
    return math.fsum(math.log(a * (1-p)**2 + b * 2*p*(1-p) + c * p*p)
                     for a, b, c in rows)


def hard_call_frequency(rows):
    """Frequency after unique per-individual maximum-likelihood genotype calls."""
    rows = validate(rows)
    if any(row.count(max(row)) != 1 for row in rows):
        raise ValueError('Hard-call baseline is undefined for tied likelihoods')
    return sum(row.index(max(row)) for row in rows) / (2 * len(rows))


def normalized_likelihood_dosage(rows):
    """Deliberately wrong as a population MLE; assumes uniform genotype prior."""
    rows = validate(rows)
    return math.fsum((b+2*c)/(a+b+c) for a, b, c in rows) / (2*len(rows))


def repeated_heterozygote_mle(a, b):
    """Exact maximizer for repeated (a, 1, b), 0<a,b<1."""
    if not (0 < a < 1 and 0 < b < 1):
        raise ValueError('Both homozygote likelihoods must be between zero and one')
    return (1-a) / (2-a-b)


def write_vcf(sites, path):
    """Write positive linear likelihoods as VCF log10 GL, preserving REF/ALT."""
    if not sites:
        raise ValueError('At least one site is required')
    normalized = [(name, validate(rows)) for name, rows in sites]
    n = len(normalized[0][1])
    if len({name for name, _ in normalized}) != len(normalized):
        raise ValueError('Site names must be unique')
    if any(not name or any(ch.isspace() for ch in name) for name, _ in normalized):
        raise ValueError('Site names must be nonempty without whitespace')
    if any(len(rows) != n for _, rows in normalized):
        raise ValueError('All sites must have the same diploid sample count')
    header = ['##fileformat=VCFv4.2', '##contig=<ID=1,length=1000000>',
              '##FORMAT=<ID=GL,Number=G,Type=Float,Description="Log10 genotype likelihoods">',
              '#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t' +
              '\t'.join('sample' + str(i) for i in range(n))]
    records = []
    for pos, (name, rows) in enumerate(normalized, 1):
        values = [','.join(format(math.log10(x), '.12g') for x in row) for row in rows]
        records.append('\t'.join(['1', str(pos), name, 'A', 'C', '.', 'PASS', '.', 'GL', *values]))
    Path(path).write_text('\n'.join(header + records) + '\n')
    return normalized


def run_angsd(sites, binary=BINARY):
    """No silent fallback: run the actual fixed-version CLI and parse its output."""
    check_version(binary)
    with tempfile.TemporaryDirectory(prefix='angsd-frequency-') as temporary:
        directory = Path(temporary)
        checked = write_vcf(sites, directory / 'input.vcf')
        command = [str(binary), '-vcf-gl', str(directory / 'input.vcf'),
                   '-doMaf', '1', '-nThreads', '1', '-out', str(directory / 'result')]
        completed = subprocess.run(command, capture_output=True, text=True, timeout=30)
        output = directory / 'result.mafs.gz'
        if completed.returncode != 0 or not output.exists():
            raise RuntimeError('ANGSD failed: ' + completed.stderr[-3000:])
        with gzip.open(output, 'rt') as handle:
            records = list(csv.DictReader(handle, delimiter='\t'))
        result = {}
        for record in records:
            pos = int(record['position'])
            if not 1 <= pos <= len(checked) or record['chromo'] != '1':
                raise ValueError('Unexpected ANGSD output location')
            if (record['major'], record['minor']) != ('A', 'C'):
                raise ValueError('Unexpected ANGSD allele orientation')
            name = checked[pos-1][0]
            if name in result:
                raise ValueError('Duplicate ANGSD output site')
            estimate = float(record['knownEM'])
            if not math.isfinite(estimate) or not 0 <= estimate <= 1:
                raise ValueError('Invalid ANGSD frequency estimate')
            result[name] = {'alt_frequency': estimate, 'informative_diploids': int(record['nInd'])}
        expected = {name for name, rows in checked if any(len(set(row)) > 1 for row in rows)}
        if set(result) != expected:
            raise ValueError('Unexpected retained or omitted ANGSD sites')
        return result


def constructed_sites(n=20):
    if not isinstance(n, int) or isinstance(n, bool) or n < 2 or n % 2:
        raise ValueError('Use an even sample count of at least two')
    symmetric = [(0.2, 1.0, 0.2)] * n
    asymmetric = [(0.8, 1.0, 0.2)] * n
    return [('symmetric', symmetric), ('asymmetric', asymmetric),
            ('alleles_reversed', [row[::-1] for row in asymmetric]),
            ('rescaled', [tuple(x * (i+1) / n for x in row) for i, row in enumerate(asymmetric)]),
            ('half_uninformative', asymmetric[:n//2] + [(1.0, 1.0, 1.0)] * (n//2)),
            ('all_uninformative', [(1.0, 1.0, 1.0)] * n),
            ('all_reference', [(1.0, 1e-12, 1e-12)] * n),
            ('all_alternate', [(1e-12, 1e-12, 1.0)] * n)]
