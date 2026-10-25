"""Compare heterozygote advantage, additive selection, and neutrality in SLiM."""

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

SCRIPT = Path(__file__).with_name("selection.slim")
MODES = {"neutral": (0, 0.0, 0.5), "additive": (1, 0.1, 0.5),
         "overdominant": (2, 0.1, 2.0)}


def fitnesses(mode):
    """Return (w_aa, w_Aa, w_AA) in the SLiM 1, 1+h*s, 1+s convention."""
    if mode not in MODES:
        raise ValueError("unknown selection mode")
    _, s, h = MODES[mode]
    return 1.0, 1+h*s, 1+s


def gamete_frequency(aa, hetero, AA, mode):
    """Exact fitness-weighted probability that a sampled parental gamete is A."""
    for count in (aa, hetero, AA):
        if isinstance(count, bool) or not isinstance(count, (int, float)) or not isfinite(count) or count < 0:
            raise ValueError("genotype weights must be finite and nonnegative")
    wa, wh, wA = fitnesses(mode)
    total = aa*wa + hetero*wh + AA*wA
    if total <= 0:
        raise ValueError("genotype weights cannot all be zero")
    return (AA*wA + hetero*wh/2)/total


def deterministic_path(p, mode, generations):
    """Infinite-population random-mating recursion, not a finite-population mean."""
    if isinstance(p, bool) or not isinstance(p, (int, float)) or not isfinite(p) or not 0 <= p <= 1:
        raise ValueError("p must be in [0,1]")
    _integer(generations, "generations", 0, 300)
    fitnesses(mode)
    path = [p]
    for _ in range(generations):
        p = gamete_frequency((1-p)**2, 2*p*(1-p), p*p, mode)
        path.append(p)
    return path


def initial_genotypes(n, p):
    """Exact Hardy-Weinberg counts for the explicitly supported starting states."""
    _integer(n, "n", 2, 1000)
    if p not in (0, 0.2, 0.8, 1) or isinstance(p, bool):
        raise ValueError("p must be 0, 0.2, 0.8, or 1")
    AA, het = n*p*p, 2*n*p*(1-p)
    if not isclose(AA, round(AA), abs_tol=1e-10) or not isclose(het, round(het), abs_tol=1e-10):
        raise ValueError("N must give integral Hardy-Weinberg counts (use a multiple of 25)")
    return round(AA), round(het)


def run_counts(mode="overdominant", *, n=200, p=0.2, replicates=128,
               generations=80, seed=20261025, executable="slim"):
    fitnesses(mode)
    AA, het = initial_genotypes(n, p)
    for value, name, lo, hi in [(replicates, "replicates", 1, 2048),
                               (generations, "generations", 0, 300), (seed, "seed", 1, 2**62-1)]:
        _integer(value, name, lo, hi)
    check_slim_version(executable)
    command = [executable, "-s", str(seed)]
    for name, value in [("N", n), ("MODE", MODES[mode][0]), ("INITIAL_AA", AA),
                        ("INITIAL_HET", het), ("REPLICATES", replicates), ("GENERATIONS", generations)]:
        command.extend(["-d", f"{name}={value}"])
    result = subprocess.run([*command, str(SCRIPT)], capture_output=True, text=True, check=True, timeout=90)
    observations, engine_fitness = defaultdict(dict), {}
    for row in csv.reader(io.StringIO(result.stdout)):
        if row and row[0] == "FIT":
            if len(row) != 3 or int(row[1]) in engine_fitness:
                raise ValueError("malformed fitness audit")
            engine_fitness[int(row[1])] = float(row[2])
        if row and row[0] == "DATA":
            if len(row) != 5:
                raise ValueError("malformed genotype output")
            t, rep, homo, hetero = map(int, row[1:])
            if homo < 0 or hetero < 0 or homo+hetero > n or rep in observations[t]:
                raise ValueError("impossible genotype counts or duplicate replicate")
            observations[t][rep] = (homo, hetero)
    expected_fitness = {i: w for i, (count, w) in enumerate(zip((n-AA-het, het, AA), fitnesses(mode))) if count}
    if engine_fitness != expected_fitness:
        raise ValueError("SLiM genotype fitness audit disagrees with intended model")
    times = {0, generations//4, generations//2, generations}
    if generations > 0:
        times.add(1)
    if set(observations) != times or any(set(rows) != set(range(replicates)) for rows in observations.values()):
        raise ValueError("missing or unexpected observations")
    return {t: [rows[i] for i in range(replicates)] for t, rows in sorted(observations.items())}


def summarize(counts, n, p, mode):
    path = deterministic_path(p, mode, max(counts))
    result = []
    for t, rows in counts.items():
        frequencies = [(2*AA+het)/(2*n) for AA, het in rows]
        lost, fixed = frequencies.count(0), frequencies.count(1)
        result.append({"generations_elapsed": t,
                       "mean_p": round(statistics.mean(frequencies), 9),
                       "monte_carlo_se": round(statistics.stdev(frequencies)/sqrt(len(rows)), 9) if len(rows) > 1 else None,
                       "infinite_population_p": round(path[t], 9),
                       "mean_observed_heterozygosity": round(statistics.mean(het/n for _, het in rows), 9),
                       "lost": lost, "fixed": fixed, "segregating": len(rows)-lost-fixed})
    return result


def build_report(executable="slim", replicates=128, generations=80, seed=20261025):
    treatments = []
    for mode in MODES:
        for p in (0.2, 0.8):
            run_seed = seed + len(treatments)
            counts = run_counts(mode, p=p, replicates=replicates, generations=generations,
                                seed=run_seed, executable=executable)
            treatments.append({"mode": mode, "diploid_N": 200, "p_initial": p, "seed": run_seed,
                               "fitness_aa_Aa_AA": fitnesses(mode),
                               "checkpoints": summarize(counts, 200, p, mode)})
    stress = []
    for offset, p in enumerate((0.2, 0.8)):
        run_seed = seed+6+offset
        counts = run_counts("overdominant", n=25, p=p, replicates=replicates, generations=200,
                            seed=run_seed, executable=executable)
        stress.append({"diploid_N": 25, "p_initial": p, "seed": run_seed,
                       "checkpoints": summarize(counts, 25, p, "overdominant")})
    return {"example": "slimulations: heterozygote advantage with finite-population loss",
            "slim_version": SLIM_VERSION, "replicates_per_treatment": replicates,
            "initialization": "Exact Hardy-Weinberg genotype counts installed after tick 1 reproduction",
            "infinite_population_overdominant_equilibrium": round(2/3, 9),
            "treatments": treatments, "small_population_control_200_generations": stress,
            "interpretation": "All means include fixed and lost populations. The deterministic path is an infinite-population comparator, "
                              "not the exact finite-population expectation. Finite populations can absorb even with heterozygote advantage. "
                              "Monte Carlo SE describes replicate uncertainty in this specified model, not uncertainty in biological parameters."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slim", default="slim")
    parser.add_argument("--replicates", type=int, default=128)
    parser.add_argument("--generations", type=int, default=80,
                        help="duration of N=200 treatments; the N=25 control always runs 200 generations")
    parser.add_argument("--seed", type=int, default=20261025)
    args = parser.parse_args()
    try:
        report = build_report(args.slim, args.replicates, args.generations, args.seed)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
