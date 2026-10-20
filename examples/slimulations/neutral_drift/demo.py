"""Run actual SLiM, then compare replicate means with Wright–Fisher expectations."""

import argparse
from collections import defaultdict
import csv
import io
import json
from math import log, sqrt
from pathlib import Path
import re
import statistics
import subprocess

SCRIPT = Path(__file__).with_name("neutral_drift.slim")
SLIM_VERSION = "4.0.1"


def _integer(value, name, lower, upper):
    if type(value) is not int or not lower <= value <= upper:
        raise ValueError(f"{name} must be an integer in [{lower}, {upper}]")


def expected_diversity(n: int, initial_copies: int, generations: int) -> float:
    _integer(n, "n", 2, 1000)
    _integer(initial_copies, "initial_copies", 0, 2 * n)
    _integer(generations, "generations", 0, 500)
    p = initial_copies / (2 * n)
    return 2 * p * (1 - p) * (1 - 1 / (2 * n)) ** generations


def check_slim_version(executable: str) -> str:
    result = subprocess.run(
        [executable, "-v"], capture_output=True, text=True, check=True, timeout=30
    )
    match = re.search(r"SLiM version ([0-9.]+)", result.stdout + result.stderr)
    if not match or match[1] != SLIM_VERSION:
        raise ValueError(f"This example requires SLiM {SLIM_VERSION}")
    return match[1]


def run_counts(
    n=20, replicates=256, generations=80, seed=20261020,
    initial_copies=None, executable="slim",
) -> dict[int, list[int]]:
    _integer(n, "n", 2, 1000)
    _integer(replicates, "replicates", 1, 4096)
    _integer(generations, "generations", 0, 500)
    _integer(seed, "seed", 1, 2**62 - 1)
    initial_copies = n if initial_copies is None else initial_copies
    _integer(initial_copies, "initial_copies", 0, 2 * n)
    check_slim_version(executable)
    command = [executable, "-s", str(seed)]
    for name, value in (
        ("N", n), ("REPLICATES", replicates), ("GENERATIONS", generations),
        ("INITIAL_COPIES", initial_copies),
    ):
        command.extend(["-d", f"{name}={value}"])
    result = subprocess.run(
        [*command, str(SCRIPT)], capture_output=True, text=True, check=True, timeout=180
    )
    observations = defaultdict(dict)
    for row in csv.reader(io.StringIO(result.stdout)):
        if row and row[0] == "DATA":
            _, time, replicate, size, count = row
            time, replicate, size, count = map(int, (time, replicate, size, count))
            if size != n or not 0 <= count <= 2 * n:
                raise ValueError("SLiM emitted an invalid population count")
            if replicate in observations[time]:
                raise ValueError("SLiM emitted a duplicate replicate")
            observations[time][replicate] = count
    expected_times = {0, generations // 4, generations // 2, generations}
    if generations >= 1:
        expected_times.add(1)
    if set(observations) != expected_times:
        raise ValueError("SLiM output is missing or has unexpected checkpoints")
    if any(set(counts) != set(range(replicates)) for counts in observations.values()):
        raise ValueError("SLiM output has missing replicate observations")
    return {time: [counts[i] for i in range(replicates)]
            for time, counts in sorted(observations.items())}


def summarize(counts, n, initial_copies):
    rows = []
    for time, values in sorted(counts.items()):
        frequencies = [count / (2 * n) for count in values]
        diversity = [2 * p * (1 - p) for p in frequencies]
        se = statistics.stdev(diversity) / sqrt(len(diversity)) if len(diversity) > 1 else 0.0
        rows.append({
            "generations_elapsed": time,
            "mean_allele_frequency": round(statistics.mean(frequencies), 9),
            "mean_gene_diversity": round(statistics.mean(diversity), 9),
            "gene_diversity_standard_error": round(se, 9) if len(diversity) > 1 else None,
            "expected_gene_diversity": round(expected_diversity(n, initial_copies, time), 9),
            "fixed_replicates": sum(count == 2 * n for count in values),
            "lost_replicates": sum(count == 0 for count in values),
            "segregating_replicates": sum(0 < count < 2 * n for count in values),
        })
    return rows


def build_report(executable="slim", replicates=256, generations=80, seed=20261020):
    groups = []
    for offset, n in enumerate((20, 100)):
        counts = run_counts(n, replicates, generations, seed + offset, executable=executable)
        groups.append({
            "diploid_population_size": n, "gene_copies": 2 * n, "seed": seed + offset,
            "checkpoints": summarize(counts, n, n),
        })
    comparisons = 2 * sum(len(group["checkpoints"]) for group in groups)
    # H lies in [0, 0.5]; p lies in [0,1] and uses twice this tolerance.
    # Count both families in one union bound.
    epsilon = 0.5 * sqrt(log(2 * comparisons / 1e-6) / (2 * replicates))
    return {
        "example": "slimulations: neutral drift removes gene diversity faster in small populations",
        "slim_version": SLIM_VERSION,
        "replicates_per_population_size": replicates,
        "initial_allele_frequency": 0.5,
        "formula": "E[2*p_t*(1-p_t)] = 2*p_0*(1-p_0)*(1-1/(2*N))^t",
        "bounded_mean_comparisons": comparisons,
        "simultaneous_hoeffding_tolerance_alpha_1e_minus_6": round(epsilon, 9),
        "populations": groups,
        "interpretation": (
            "H=2p(1-p) is gene diversity from population allele frequency, not the "
            "observed fraction of heterozygous individuals. Theory describes an ensemble "
            "mean; individual finite trajectories need not decrease monotonically."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slim", default="slim")
    parser.add_argument("--replicates", type=int, default=256)
    parser.add_argument("--generations", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20261020)
    args = parser.parse_args()
    try:
        report = build_report(args.slim, args.replicates, args.generations, args.seed)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
