"""Actual SLiM reversible mutation and exact finite-population moments."""

import argparse
import csv
import io
import json
import math
from pathlib import Path
import statistics
import subprocess

from examples.slimulations.neutral_drift.demo import _integer, check_slim_version

SCRIPT = Path(__file__).with_name("mutation.slim")
SAMPLE = Path(__file__).with_name("sample_summary.json")
SEED = 202610300


def validate(n, initial_copies, u, v, generations):
    _integer(n, "n", 2, 1000)
    _integer(initial_copies, "initial_copies", 0, 2*n)
    _integer(generations, "generations", 0, 200)
    for name, value in (("u", u), ("v", v)):
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"{name} must be a finite probability")


def moments(n, initial_copies, u, v, generations):
    """Exact-model raw moments E[p**r], r=0..4, evaluated in binary64."""
    validate(n, initial_copies, u, v, generations)
    copies, a = 2*n, 1-u-v
    result = [[(initial_copies/copies)**r for r in range(5)]]
    # Stirling numbers of the second kind convert factorial to raw moments.
    stirling = ((1,), (0, 1), (0, 1, 1), (0, 1, 3, 1), (0, 1, 7, 6, 1))
    for _ in range(generations):
        q = [sum(math.comb(r, j)*u**(r-j)*a**j*result[-1][j]
                 for j in range(r+1)) for r in range(5)]
        next_row = [1.0]
        for r in range(1, 5):
            falling = 1
            total = 0.0
            for j in range(1, r+1):
                falling *= copies-j+1
                total += stirling[r][j]*falling*q[j]
            next_row.append(total/copies**r)
        result.append(next_row)
    return result


def prediction(raw):
    _, m1, m2, m3, m4 = raw
    h = 2*(m1-m2)
    return {"mean_frequency": m1, "frequency_variance": max(0.0, m2-m1*m1),
            "mean_gene_diversity": h,
            "gene_diversity_variance": max(0.0, 4*(m2-2*m3+m4)-h*h)}


def symmetric_stationary_diversity(n, mu):
    """Stationary E[2p(1-p)] for 0 < mu < 1; not observed heterozygosity."""
    validate(n, 0, mu, mu, 0)
    if not 0 < mu < 1:
        raise ValueError("stationary formula requires 0 < mu < 1")
    b = 1-1/(2*n)
    return 2*b*mu*(1-mu)/(1-b*(1-2*mu)**2)


def checkpoints(generations):
    return {0, generations//2, generations} | ({1} if generations else set())


def parse_output(stdout, n, initial_copies, replicates, generations):
    if "Eidos version 3.0.1\n" not in stdout or "SLiM version 4.0.1\n" not in stdout:
        raise ValueError("Missing or unexpected engine versions")
    records, fitness = {}, {}
    for row in csv.reader(io.StringIO(stdout)):
        if not row:
            continue
        if row[0] == "FIT":
            if len(row) != 4:
                raise ValueError("Malformed fitness audit")
            t = int(row[1])
            if t in fitness or tuple(map(float, row[2:])) != (1.0, 1.0):
                raise ValueError("Duplicate or nonneutral fitness audit")
            fitness[t] = True
        elif row[0] == "DATA":
            if len(row) != 5:
                raise ValueError("Malformed allele record")
            t, rep, size, copies = map(int, row[1:])
            if size != n or not 0 <= copies <= 2*n or (t, rep) in records:
                raise ValueError("Invalid size, allele count, or duplicate record")
            records[t, rep] = copies
    times = checkpoints(generations)
    if set(fitness) != times or set(records) != {(t, r) for t in times for r in range(replicates)}:
        raise ValueError("Missing or unexpected SLiM records")
    if any(records[0, r] != initial_copies for r in range(replicates)):
        raise ValueError("Incorrect founder allele count")
    return {t: [records[t, r] for r in range(replicates)] for t in sorted(times)}


def run_counts(*, n=20, initial_copies=0, u=.01, v=.01, replicates=512,
               generations=80, seed=SEED, executable="slim"):
    validate(n, initial_copies, u, v, generations)
    _integer(replicates, "replicates", 1, 2048)
    _integer(seed, "seed", 1, 2**62-1)
    check_slim_version(executable)
    command = [executable, "-s", str(seed)]
    for name, value in (("N", n), ("INITIAL_COPIES", initial_copies), ("U", u), ("V", v),
                        ("REPLICATES", replicates), ("GENERATIONS", generations)):
        command.extend(["-d", f"{name}={value}"])
    result = subprocess.run([*command, str(SCRIPT)], capture_output=True, text=True,
                            check=True, timeout=180)
    return parse_output(result.stdout, n, initial_copies, replicates, generations)


def summarize(counts, n):
    p = [k/(2*n) for k in counts]
    h = [2*x*(1-x) for x in p]
    se = lambda x: round(statistics.stdev(x)/math.sqrt(len(x)), 9) if len(x)>1 else None
    return {"mean_frequency": round(statistics.mean(p), 9),
            "frequency_monte_carlo_se": se(p),
            "mean_gene_diversity": round(statistics.mean(h), 9),
            "diversity_monte_carlo_se": se(h),
            "diversity_of_mean_frequency": round(2*statistics.mean(p)*(1-statistics.mean(p)), 9),
            "zero_A_replicates": counts.count(0), "all_A_replicates": counts.count(2*n),
            "segregating_replicates": sum(0 < k < 2*n for k in counts)}


def build_report(executable="slim", replicates=512, seed=SEED):
    _integer(seed, "seed", 1, 2**62-4)
    scenarios = [("no_mutation", 20, 0.0, 0.0), ("symmetric_small", 20, .01, .01),
                 ("symmetric_large", 80, .01, .01), ("biased_mutation", 20, .02, .005)]
    treatments = []
    for offset, (label, n, u, v) in enumerate(scenarios):
        counts = run_counts(n=n, u=u, v=v, replicates=replicates, seed=seed+offset, executable=executable)
        theory = moments(n, 0, u, v, 80)
        treatments.append({"treatment": label, "diploid_population_size": n,
                           "u_a_to_A": u, "v_A_to_a": v, "seed": seed+offset,
                           "stationary_mean_frequency": round(u/(u+v), 9) if u+v else None,
                           "symmetric_stationary_gene_diversity": round(symmetric_stationary_diversity(n, u), 9) if u == v and u else None,
                           "checkpoints": [{"generations_elapsed": t,
                                            "prediction": {k: round(value, 9) for k, value in prediction(theory[t]).items()},
                                            "observed": summarize(values, n)} for t, values in counts.items()]})
    return {"example": "Reversible mutation restores variation after fixation",
            "slim_version": "4.0.1", "eidos_version": "3.0.1", "reference_python_version": "3.14.5",
            "replicates_per_treatment": replicates, "initial_A_frequency": 0,
            "treatments": treatments,
            "interpretation": "H=2p(1-p) is within-population gene diversity, not the observed heterozygote fraction. "
                              "All replicates count, including temporarily monomorphic populations. "
                              "A stationary ensemble does not imply permanent polymorphism in every population. "
                              "Mutation probabilities are illustrative, not estimates for real organisms."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slim", default="slim")
    parser.add_argument("--replicates", type=int, default=512)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--check-sample", action="store_true")
    args = parser.parse_args()
    try:
        output = json.dumps(build_report(args.slim, args.replicates, args.seed), indent=2, allow_nan=False)+"\n"
        if args.check_sample and output != SAMPLE.read_text():
            raise ValueError("Output differs from checked sample")
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        parser.error(str(error))
    print(output, end="")


if __name__ == "__main__":
    main()
