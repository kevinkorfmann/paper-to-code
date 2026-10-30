"""Original synthetic projection experiment; no third-party Python packages."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import re
import subprocess
import tempfile

from .install import BINARY, BINARY_SHA256, VERSION, verify

HERE = Path(__file__).resolve().parent
FIXTURE = HERE / "synthetic_genotypes.json"
SAMPLE = HERE / "sample_summary.json"
DATA_SEED = 20261030
OPTIMIZER_SEEDS = (41, 99)
PATTERNS = ((0.1, 0.8), (0.2, 0.7), (0.15, 0.65), (0.35, 0.9))
REPEATS = 500
TRUE_Q = (0.0, 0.25, 0.5, 0.75, 1.0)


def generate_input():
    """Independent Bernoulli alleles; q is a probability, not a pedigree fraction."""
    rng = random.Random(DATA_SEED)
    frequencies = PATTERNS * REPEATS
    genotypes = []
    for q in TRUE_Q:
        calls = []
        for a, b in frequencies:
            p = q * a + (1 - q) * b
            calls.append(str(int(rng.random() < p) + int(rng.random() < p)))
        genotypes.append("".join(calls))
    return {"data_seed": DATA_SEED, "frequency_patterns": [list(x) for x in PATTERNS],
            "repeats": REPEATS, "true_q_source_A": list(TRUE_Q),
            "allele2_dosages": genotypes}


def validate(genotypes, frequencies):
    if not frequencies or not genotypes:
        raise ValueError("Need at least one marker and one individual")
    for row in frequencies:
        if len(row) != 2 or any(type(x) not in (int, float) or not math.isfinite(x)
                                or not 0 < x < 1 for x in row):
            raise ValueError("Need two finite, strictly interior frequencies per marker")
    for row in genotypes:
        if len(row) != len(frequencies) or any(type(g) is not int or g not in (0, 1, 2)
                                               for g in row):
            raise ValueError("Genotypes must be complete diploid allele-2 counts 0/1/2")


def write_inputs(directory, genotypes, frequencies):
    """Write SNP-major BED: 00=A1/A1, 10=A1/A2, 11=A2/A2, low bits first.

    ADMIXTURE's .P uses allele-2 frequencies for this BED encoding, verified
    against asymmetric analytical controls in the integration tests.
    """
    validate(genotypes, frequencies)
    n = len(genotypes)
    data = bytearray((0x6C, 0x1B, 0x01))
    codes = (0, 2, 3)
    for j in range(len(frequencies)):
        for start in range(0, n, 4):
            data.append(sum(codes[genotypes[i][j]] << (2 * (i - start))
                            for i in range(start, min(start + 4, n))))
    (directory / "study.bed").write_bytes(data)
    (directory / "study.bim").write_text("".join(
        f"1 synthetic_{j + 1} 0 {j + 1} A C\n" for j in range(len(frequencies))))
    (directory / "study.fam").write_text("".join(
        f"SYN individual_{i + 1} 0 0 0 -9\n" for i in range(n)))
    (directory / "study.2.P.in").write_text("".join(
        f"{a:.12g} {b:.12g}\n" for a, b in frequencies))


def parse_q(text, n):
    rows = [list(map(float, line.split())) for line in text.splitlines()]
    if len(rows) != n or any(len(row) != 2 or any(not math.isfinite(x) or not 0 <= x <= 1
                                               for x in row)
                              or abs(sum(row) - 1) > 1.1e-6 for row in rows):
        raise ValueError("Malformed ADMIXTURE Q output")
    return [row[0] for row in rows]


def run_projection(genotypes, frequencies, seed=41):
    """Infer Q with real ADMIXTURE; truth is not an argument or an input file."""
    validate(genotypes, frequencies)
    if type(seed) is not int or not 0 <= seed < 2**31:
        raise ValueError("Optimizer seed must be an integer in [0, 2**31)")
    if not BINARY.is_file():
        raise RuntimeError("Run python -m examples.inference_showcase.admixture_projection.install")
    verify(BINARY.read_bytes(), BINARY_SHA256)
    with tempfile.TemporaryDirectory(prefix="synthetic-admixture-") as tmp:
        directory = Path(tmp)
        write_inputs(directory, genotypes, frequencies)
        process = subprocess.run(
            [str(BINARY), "-P", f"--seed={seed}", "-C=1e-10", "-j1", "study.bed", "2"],
            cwd=directory, capture_output=True, text=True, timeout=60, check=True)
        if f"ADMIXTURE Version {VERSION}" not in process.stdout or "Converged in " not in process.stdout:
            raise RuntimeError("Unexpected executable version or no convergence report")
        q = parse_q((directory / "study.2.Q").read_text(), len(genotypes))
        p = [list(map(float, row.split())) for row in
             (directory / "study.2.P").read_text().splitlines()]
        if len(p) != len(frequencies) or any(len(row) != 2 for row in p):
            raise ValueError("Malformed ADMIXTURE P output")
        if any(not math.isfinite(x) or abs(x - y) > 5.1e-7
               for row, expected in zip(p, frequencies) for x, y in zip(row, expected)):
            raise RuntimeError("Projection changed the supplied reference frequencies")
        match = re.search(r"^Loglikelihood: ([-+0-9.eE]+)$", process.stdout, re.MULTILINE)
        if match is None or not math.isfinite(float(match[1])):
            raise ValueError("No finite final log likelihood")
        return {"q_source_A": q, "log_likelihood_without_binomial_constant": float(match[1])}


def log_likelihood(calls, frequencies, q):
    """Independent binomial log likelihood, omitting q-independent log choose(2,g)."""
    return math.fsum(g * math.log(p) + (2 - g) * math.log1p(-p)
                     for g, (a, b) in zip(calls, frequencies)
                     for p in (q * a + (1 - q) * b,))


def oracle_q(calls, frequencies):
    """Independent concave 1D maximization, not ADMIXTURE's block/EM solver.

    Return None when the supplied source profiles are identical: every q is
    a maximum. Mathematical bounds are [0,1], unlike the tool's small offsets.
    """
    validate([calls], frequencies)
    if all(a == b for a, b in frequencies):
        return None
    def score(q):
        return math.fsum((a - b) * (g / p - (2 - g) / (1 - p))
                         for g, (a, b) in zip(calls, frequencies)
                         for p in (q * a + (1 - q) * b,))
    if score(0) <= 0:
        return 0.0
    if score(1) >= 0:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(70):
        mid = (lo + hi) / 2
        if score(mid) > 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def summarize():
    raw = FIXTURE.read_bytes()
    fixture = json.loads(raw)
    genotypes = [[int(g) for g in row] for row in fixture["allele2_dosages"]]
    frequencies = fixture["frequency_patterns"] * fixture["repeats"]
    base = run_projection(genotypes, frequencies)
    narrower = [[(a + b) / 2, b] for a, b in frequencies]
    changed = run_projection(genotypes, narrower)
    flat = [[0.5, 0.5] for _ in frequencies]
    flat_runs = [run_projection(genotypes, flat, seed) for seed in OPTIMIZER_SEEDS]
    boundary = run_projection([[0] * 40, [2] * 40], [[0.2, 0.8]] * 40)
    return {
        "versions": {"python": platform.python_version(), "admixture": VERSION,
                     "admixture_binary_sha256": BINARY_SHA256},
        "data_seed": fixture["data_seed"], "optimizer_seeds": list(OPTIMIZER_SEEDS),
        "fixture_sha256": hashlib.sha256(raw).hexdigest(),
        "individuals": len(genotypes), "independent_markers": len(frequencies),
        "true_q_source_A": fixture["true_q_source_A"],
        "correct_references": base,
        "independent_mathematical_mle": [round(oracle_q(g, frequencies), 9) for g in genotypes],
        "halved_reference_contrast": changed,
        "degenerate_boundary_control": {
            "actual_tool": boundary, "mathematical_mle": [1.0, 0.0],
            "mathematical_max_log_likelihood": round(160 * math.log(0.8), 6),
            "note": "Convergence report is not an optimality certificate on this stress case",
        },
        "identical_reference_profiles": {
            "identifiable": False, "seed_41": flat_runs[0], "seed_99": flat_runs[1],
            "analytical_log_likelihood": round(-2 * len(genotypes) * len(frequencies) * math.log(2), 6),
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-sample", action="store_true")
    args = parser.parse_args()
    output = json.dumps(summarize(), indent=2, allow_nan=False) + "\n"
    if args.check_sample:
        if output != SAMPLE.read_text():
            raise SystemExit("Output differs from sample_summary.json")
        print("Checked ADMIXTURE projection output matches byte for byte")
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
