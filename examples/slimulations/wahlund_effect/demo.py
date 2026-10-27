"""Measure the Wahlund effect with actual SLiM genotypes and exact one-step controls."""

import argparse
from collections import defaultdict
import csv
from fractions import Fraction
import io
import json
from math import fsum, isfinite, sqrt
from pathlib import Path
import statistics
import subprocess

from examples.slimulations.neutral_drift.demo import _integer, check_slim_version, SLIM_VERSION

SCRIPT = Path(__file__).with_name("pooling.slim")
SCENARIOS = (("different_frequencies", 100, 100, .2, .8),
             ("equal_frequency_control", 100, 100, .5, .5),
             ("unequal_size_control", 100, 300, .2, .8),
             ("opposite_fixation_control", 100, 100, 0., 1.))


def founder_counts(n, p):
    """Exact HWE founder counts; reject rounding rather than change the frequency."""
    _integer(n, "n", 2, 1000)
    if isinstance(p, bool) or not isinstance(p, (int, float)) or not isfinite(p) or not 0 <= p <= 1:
        raise ValueError("p must be finite and in [0,1]")
    p = Fraction(str(p))
    AA, het = n*p*p, 2*n*p*(1-p)
    if AA.denominator != 1 or het.denominator != 1:
        raise ValueError("N and p must give integral Hardy-Weinberg founder counts")
    return n, int(AA), int(het)


def metrics(groups):
    """Whole-census quantities for one replicate; groups are (N, n_AA, n_Aa)."""
    if not groups:
        raise ValueError("at least one mating group is required")
    for n, AA, het in groups:
        if any(type(x) is not int for x in (n, AA, het)) or n <= 0 or min(AA, het) < 0 or AA+het > n:
            raise ValueError("invalid genotype counts")
    total = sum(n for n, _, _ in groups)
    p = sum(2*AA+het for _, AA, het in groups)/(2*total)
    observed = sum(het for _, _, het in groups)/total
    within = fsum((n/total)*2*((2*AA+het)/(2*n))*(1-(2*AA+het)/(2*n)) for n, AA, het in groups)
    between = fsum((n/total)*((2*AA+het)/(2*n)-p)**2 for n, AA, het in groups)
    pooled = 2*p*(1-p)
    return {"p": p, "H_observed": observed, "H_pooled": pooled, "H_within": within,
            "twice_between_variance": 2*between, "pooled_gap": pooled-observed,
            "within_residual": within-observed}


def one_generation_expectation(n1=100, n2=100, p1=.2, p2=.8, pooled=False):
    """Finite-census E[H_observed] and E[2*p_hat*(1-p_hat)] after one generation."""
    founder_counts(n1, p1); founder_counts(n2, p2)
    if type(pooled) is not bool:
        raise ValueError("pooled must be bool")
    total = n1+n2
    weights = (n1/total, n2/total)
    pbar = weights[0]*p1+weights[1]*p2
    H_T = 2*pbar*(1-pbar)
    H_S = fsum(w*2*p*(1-p) for w, p in zip(weights, (p1, p2)))
    expected_observed = H_T if pooled else H_S
    # Independent offspring gene draws within each actual mating population.
    variance_pbar = expected_observed/(4*total)
    expected_plugin = H_T-2*variance_pbar
    return {"p": pbar, "H_observed": expected_observed,
            "H_pooled": expected_plugin, "pooled_gap": expected_plugin-expected_observed,
            "variance_of_p": variance_pbar,
            "founder_H_pooled": H_T, "founder_H_observed": H_S,
            "founder_pooling_gap": H_T-H_S}


def parse_counts(stdout, n1, n2, pooled, replicates, generations):
    rows = defaultdict(lambda: defaultdict(dict))
    sizes = (n1+n2,) if pooled else (n1, n2)
    for row in csv.reader(io.StringIO(stdout)):
        if row and row[0] == "DATA":
            if len(row) != 7:
                raise ValueError("malformed SLiM DATA row")
            t, rep, group, n, AA, het = map(int, row[1:])
            if not 0 <= group < len(sizes) or n != sizes[group] or group in rows[t][rep]:
                raise ValueError("unexpected group size or duplicate record")
            metrics([(n, AA, het)])
            rows[t][rep][group] = (n, AA, het)
    if set(rows) != set(range(generations+1)):
        raise ValueError("missing or unexpected generation")
    for generation in rows.values():
        if set(generation) != set(range(replicates)) or any(set(groups) != set(range(len(sizes))) for groups in generation.values()):
            raise ValueError("missing or unexpected replicate/group")
    return {t: [tuple(rows[t][r][g] for g in range(len(sizes))) for r in range(replicates)] for t in sorted(rows)}


def run_counts(*, n1=100, n2=100, p1=.2, p2=.8, pooled=False,
               replicates=256, generations=1, seed=202610270, executable="slim"):
    first, second = founder_counts(n1, p1), founder_counts(n2, p2)
    if type(pooled) is not bool:
        raise ValueError("pooled must be bool")
    for value, name, low, high in [(replicates, "replicates", 1, 1024),
                                  (generations, "generations", 0, 1), (seed, "seed", 1, 2**62-1)]:
        _integer(value, name, low, high)
    check_slim_version(executable)
    command = [executable, "-s", str(seed)]
    for name, value in [("N1", n1), ("N2", n2), ("AA1", first[1]), ("HET1", first[2]),
                        ("AA2", second[1]), ("HET2", second[2]), ("POOLED", int(pooled)),
                        ("REPLICATES", replicates), ("GENERATIONS", generations)]:
        command.extend(["-d", f"{name}={value}"])
    result = subprocess.run([*command, str(SCRIPT)], text=True, capture_output=True, check=True, timeout=60)
    return parse_counts(result.stdout, n1, n2, pooled, replicates, generations)


def summarize(counts):
    checkpoints = []
    for t, replicates in counts.items():
        observations = [metrics(groups) for groups in replicates]
        means, errors = {}, {}
        for key in observations[0]:
            values = [row[key] for row in observations]
            means[key] = round(statistics.mean(values), 9)
            errors[key] = round(statistics.stdev(values)/sqrt(len(values)), 9) if len(values) > 1 else None
        checkpoints.append({"generations_elapsed": t, "means": means, "monte_carlo_se": errors})
    return checkpoints


def build_report(executable="slim", replicates=256, seed=202610270):
    _integer(seed, "seed", 1, 2**62-8)
    treatments = []
    for name, n1, n2, p1, p2 in SCENARIOS:
        for pooled in (False, True):
            run_seed = seed+len(treatments)
            counts = run_counts(n1=n1, n2=n2, p1=p1, p2=p2, pooled=pooled,
                                replicates=replicates, seed=run_seed, executable=executable)
            expected = one_generation_expectation(n1, n2, p1, p2, pooled)
            treatments.append({"scenario": name, "mating": "common_pool" if pooled else "separate_demes",
                               "founder_group_sizes": [n1, n2], "founder_frequencies": [p1, p2],
                               "seed": run_seed, "checkpoints": summarize(counts),
                               "one_generation_expectation": {k: round(v, 9) for k, v in expected.items()}})
    return {"example": "Wahlund effect: pooling samples versus pooling parents",
            "slim_version": SLIM_VERSION, "eidos_version": "3.0.1", "replicates_per_treatment": replicates,
            "treatments": treatments,
            "interpretation": "Founder genotypes are identical across mating treatments. Pooling counts does not change observed heterozygosity. "
                              "One generation of common-pool random mating changes genotype frequencies. H_pooled uses the realized census "
                              "frequency and has a finite-population correction in expectation. Pooled gaps are not estimates of pedigree inbreeding."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slim", default="slim")
    parser.add_argument("--replicates", type=int, default=256)
    parser.add_argument("--seed", type=int, default=202610270)
    args = parser.parse_args()
    try:
        report = build_report(args.slim, args.replicates, args.seed)
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
