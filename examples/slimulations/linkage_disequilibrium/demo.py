"""Show that free recombination need not erase population LD in one generation."""

import argparse
from collections import defaultdict
import csv
import io
import json
from math import isclose, isfinite, sqrt
from pathlib import Path
import statistics
import subprocess

from examples.slimulations.neutral_drift.demo import _integer, check_slim_version, SLIM_VERSION

SCRIPT = Path(__file__).with_name("recombination.slim")
HAPLOTYPES = ("AB", "Ab", "aB", "ab")
FOUNDER_MODES = {"random_pairing": 0, "all_double_heterozygotes": 1}


def _fraction(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value) or not 0 <= value <= .5:
        raise ValueError("recombination fraction must be in [0,0.5]")


def _frequencies(values):
    if len(values) != 4 or any(isinstance(x, bool) or not isinstance(x, (int, float))
                              or not isfinite(x) or x < 0 for x in values):
        raise ValueError("four finite nonnegative haplotype frequencies required")
    if not isclose(sum(values), 1, rel_tol=0, abs_tol=1e-12):
        raise ValueError("haplotype frequencies must sum to one")


def linkage_disequilibrium(frequencies):
    """Signed gametic D; not D-prime, r-squared, or a diploid association measure."""
    _frequencies(frequencies)
    AB, Ab, aB, _ = frequencies
    return AB - (AB+Ab)*(AB+aB)


def deterministic_path(frequencies, c, generations):
    """Infinite-population, random-haplotype-pairing recursion; no drift."""
    _frequencies(frequencies)
    _fraction(c)
    _integer(generations, "generations", 0, 100)
    path = [tuple(frequencies)]
    for _ in range(generations):
        AB, Ab, aB, ab = path[-1]
        change = c*linkage_disequilibrium(path[-1])
        path.append((AB-change, Ab+change, aB+change, ab-change))
    return path


def _counts(row, n):
    if len(row) != 6 or any(type(x) is not int or x < 0 for x in row):
        raise ValueError("six nonnegative integer counts required")
    AB, Ab, aB, ab, coupling, repulsion = row
    if sum(row[:4]) != 2*n or coupling+repulsion > n or coupling > min(AB, ab) or repulsion > min(Ab, aB):
        raise ValueError("inconsistent haplotype or double-heterozygote counts")


def conditional_gametes(row, c):
    """Exact gamete probabilities conditional on actual parental diplotypes.

    row is (n_AB,n_Ab,n_aB,n_ab,n_AB/ab,n_Ab/aB). Only the two
    double-heterozygote classes change haplotype frequencies in meiosis.
    """
    _fraction(c)
    n = sum(row[:4])/2
    if n < 1 or n != int(n):
        raise ValueError("a nonempty diploid population is required")
    _counts(row, n)
    change = c*(row[4]-row[5])/(2*n)
    gametes = tuple(x/(2*n)+sign*change for x, sign in zip(row[:4], (-1, 1, 1, -1)))
    _frequencies(gametes)
    return gametes


def initial_counts(n, founders):
    _integer(n, "n", 4, 4000)
    if n % 4 or founders not in FOUNDER_MODES:
        raise ValueError("N must be divisible by four and founders must name a supported mode")
    return n, 0, 0, n, n//2 if founders == "random_pairing" else n, 0


def parse_counts(output, n, replicates, generations):
    """Validate the tagged engine output without silently dropping missing data."""
    observations = defaultdict(dict)
    for row in csv.reader(io.StringIO(output)):
        if row and row[0] == "DATA":
            if len(row) != 9:
                raise ValueError("malformed SLiM observation")
            t, rep, *counts = map(int, row[1:])
            _counts(counts, n)
            if rep in observations[t]:
                raise ValueError("duplicate replicate observation")
            observations[t][rep] = tuple(counts)
    times = {0, generations//4, generations//2, generations}
    if generations:
        times.add(1)
    if set(observations) != times or any(set(rows) != set(range(replicates)) for rows in observations.values()):
        raise ValueError("missing or unexpected observations")
    return {t: [rows[i] for i in range(replicates)] for t, rows in sorted(observations.items())}


def run_counts(c=.5, *, n=1000, replicates=128, generations=8, seed=202610260,
               founders="random_pairing", executable="slim"):
    _fraction(c)
    initial_counts(n, founders)
    for value, name, low, high in ((replicates, "replicates", 1, 2048),
                                    (generations, "generations", 0, 100), (seed, "seed", 1, 2**62-1)):
        _integer(value, name, low, high)
    check_slim_version(executable)
    command = [executable, "-s", str(seed)]
    for name, value in (("N", n), ("REPLICATES", replicates), ("GENERATIONS", generations),
                        ("C", c), ("FOUNDERS", FOUNDER_MODES[founders])):
        command.extend(["-d", f"{name}={value}"])
    result = subprocess.run([*command, str(SCRIPT)], capture_output=True, text=True, check=True, timeout=90)
    observations = parse_counts(result.stdout, n, replicates, generations)
    if observations[0] != [initial_counts(n, founders)]*replicates:
        raise ValueError("engine founder counts disagree with intended initialization")
    return observations


def summarize(counts, n, c, founders):
    path = deterministic_path((.5, 0, 0, .5), c, max(counts)) if founders == "random_pairing" else None
    result = []
    for t, rows in counts.items():
        frequencies = [tuple(x/(2*n) for x in row[:4]) for row in rows]
        values = [linkage_disequilibrium(x) for x in frequencies]
        result.append({"generations_elapsed": t,
                       "mean_D": round(statistics.mean(values), 9),
                       "monte_carlo_se_D": round(statistics.stdev(values)/sqrt(len(rows)), 9) if len(rows) > 1 else None,
                       "infinite_random_pairing_D": round(linkage_disequilibrium(path[t]), 9) if path else None,
                       "mean_p_A": round(statistics.mean(x[0]+x[1] for x in frequencies), 9),
                       "mean_p_B": round(statistics.mean(x[0]+x[2] for x in frequencies), 9),
                       "mean_Ab_plus_aB": round(statistics.mean(x[1]+x[2] for x in frequencies), 9)})
    return result


def build_report(executable="slim", replicates=128, generations=8, seed=202610260):
    treatments = []
    specifications = [(c, "random_pairing", generations) for c in (0, .1, .5)]
    specifications.append((.5, "all_double_heterozygotes", 1))
    for offset, (c, founders, duration) in enumerate(specifications):
        n, run_seed = 1000, seed+offset
        counts = run_counts(c, n=n, replicates=replicates, generations=duration, seed=run_seed,
                            founders=founders, executable=executable)
        q = conditional_gametes(initial_counts(n, founders), c)
        treatments.append({"recombination_fraction": c, "founders": founders, "seed": run_seed,
                           "initial_counts_AB_Ab_aB_ab_coupling_repulsion": initial_counts(n, founders),
                           "first_generation_gamete_probabilities_AB_Ab_aB_ab": q,
                           "exact_expected_D_at_t1": round((1-1/(2*n))*linkage_disequilibrium(q), 9),
                           "checkpoints": summarize(counts, n, c, founders)})
    return {"example": "slimulations: recombination erodes linkage disequilibrium, not instantly",
            "slim_version": SLIM_VERSION, "diploid_N": 1000, "replicates_per_treatment": replicates,
            "formula": "D_t = D_0*(1-c)^t in an infinite randomly paired population",
            "treatments": treatments,
            "interpretation": "D is computed within each population and then averaged, not from pooled haplotypes. "
                              "The trajectory curve is an infinite-population comparator, not an exact finite-population mean. "
                              "The t=1 expectation conditions on the specified founder diplotypes and includes finite-sampling bias. "
                              "Monte Carlo SE describes replicate variation, not uncertainty about biological parameters."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slim", default="slim")
    parser.add_argument("--replicates", type=int, default=128)
    parser.add_argument("--generations", type=int, default=8,
                        help="duration of random-pairing treatments; the alternative-founder control always runs one generation")
    parser.add_argument("--seed", type=int, default=202610260)
    args = parser.parse_args()
    try:
        report = build_report(args.slim, args.replicates, args.generations, args.seed)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
