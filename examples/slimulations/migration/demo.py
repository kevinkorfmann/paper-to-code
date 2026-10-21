"""Run real SLiM to test expected homogenization by symmetric migration."""

import argparse
from collections import defaultdict
import csv
import io
import json
from math import isfinite, log, sqrt
from pathlib import Path
import statistics
import subprocess

from examples.slimulations.neutral_drift.demo import _integer, check_slim_version, SLIM_VERSION

SCRIPT = Path(__file__).with_name("migration.slim")


def check_rate(rate):
    if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not isfinite(rate) or not 0 <= rate <= 1:
        raise ValueError("migration must be finite and in [0,1]")


def expected_difference(migration, generations):
    check_rate(migration)
    _integer(generations, "generations", 0, 100)
    return (1 - 2*migration)**generations


def run_counts(migration=0.05, *, n=50, replicates=256, generations=20,
               seed=20261021, executable="slim"):
    check_rate(migration)
    for value, name, low, high in [(n, "n", 2, 1000), (replicates, "replicates", 1, 1024),
                                  (generations, "generations", 0, 100), (seed, "seed", 1, 2**62-1)]:
        _integer(value, name, low, high)
    check_slim_version(executable)
    command = [executable, "-s", str(seed)]
    for name, value in [("N", n), ("REPLICATES", replicates), ("GENERATIONS", generations), ("MIGRATION", migration)]:
        command.extend(["-d", f"{name}={value}"])
    result = subprocess.run([*command, str(SCRIPT)], capture_output=True, text=True, check=True, timeout=180)
    observations = defaultdict(dict)
    for row in csv.reader(io.StringIO(result.stdout)):
        if row and row[0] == "DATA":
            if len(row) != 5:
                raise ValueError("Malformed SLiM row")
            time, replicate, a, b = map(int, row[1:])
            if not (0 <= a <= 2*n and 0 <= b <= 2*n) or replicate in observations[time]:
                raise ValueError("Invalid count or duplicate replicate")
            observations[time][replicate] = (a, b)
    times = {0, generations//4, generations//2, generations}
    if generations >= 1:
        times.add(1)
    if set(observations) != times or any(set(rows) != set(range(replicates)) for rows in observations.values()):
        raise ValueError("Missing or unexpected SLiM observations")
    return {t: [rows[i] for i in range(replicates)] for t, rows in sorted(observations.items())}


def build_report(executable="slim", replicates=256, generations=20, seed=20261021):
    n = 50
    groups = []
    for offset, migration in enumerate((0.0, 0.05)):
        counts = run_counts(migration, n=n, replicates=replicates, generations=generations,
                            seed=seed+offset, executable=executable)
        checkpoints = []
        for time, rows in counts.items():
            differences = [(a-b)/(2*n) for a,b in rows]
            pooled = [(a+b)/(4*n) for a,b in rows]
            checkpoints.append({
                "generations_elapsed": time,
                "mean_p1_minus_p2": round(statistics.mean(differences), 9),
                "difference_standard_error": round(statistics.stdev(differences)/sqrt(replicates), 9) if replicates > 1 else None,
                "expected_p1_minus_p2": round(expected_difference(migration, time), 9),
                "mean_pooled_frequency": round(statistics.mean(pooled), 9),
                "min_realized_difference": min(differences),
                "max_realized_difference": max(differences),
            })
        groups.append({"migration_fraction": migration, "seed": seed+offset, "checkpoints": checkpoints})
    comparisons = 2*sum(len(g["checkpoints"]) for g in groups)
    # Difference is bounded [-1,1]. Pooled frequency is [0,1], half this range.
    tolerance = 2*sqrt(log(2*comparisons/1e-6)/(2*replicates))
    return {
        "example": "slimulations: migration erodes divergence between two demes",
        "slim_version": SLIM_VERSION,
        "diploid_individuals_per_deme": n, "independent_pairs_per_treatment": replicates,
        "initial_p1": 1.0, "initial_p2": 0.0,
        "formula": "E[p1_t - p2_t] = (1 - 2*m)^t; E[(p1_t+p2_t)/2] = 0.5",
        "bounded_mean_comparisons": comparisons,
        "simultaneous_difference_tolerance_alpha_1e_minus_6": round(tolerance, 9),
        "treatments": groups,
        "interpretation": (
            "The expectation concerns signed frequency difference, not absolute "
            "difference or F_ST. Drift leaves finite replicate scatter, including "
            "reversals; pooling preserves frequency only in expectation."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slim", default="slim")
    parser.add_argument("--replicates", type=int, default=256)
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261021)
    args = parser.parse_args()
    try:
        report = build_report(args.slim, args.replicates, args.generations, args.seed)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
