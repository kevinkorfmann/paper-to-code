"""Compare actual SLiM genotype counts with exact mating-system expectations."""

import argparse
from collections import defaultdict
import csv
import io
import json
from math import log, sqrt
from pathlib import Path
import statistics
import subprocess

from examples.slimulations.neutral_drift.demo import _integer, check_slim_version, SLIM_VERSION

SCRIPT = Path(__file__).with_name("selfing.slim")


def validate_mode(selfing):
    if isinstance(selfing, bool) or not isinstance(selfing, (int, float)) or selfing not in (0, 1):
        raise ValueError("selfing must be 0 (random mating) or 1 (complete selfing)")


def expected_heterozygosity(selfing, n, generations):
    """All-Aa initial state: exact ensemble mean under the two modeled modes."""
    validate_mode(selfing)
    _integer(n, "n", 2, 1000)
    _integer(generations, "generations", 0, 50)
    if generations == 0:
        return 1.0
    if selfing == 1:
        return 0.5**generations
    return 0.5*(1-1/(2*n))**(generations-1)


def run_counts(selfing=1, *, n=100, replicates=256, generations=8,
               seed=20261023, initial_state=1, executable="slim"):
    validate_mode(selfing)
    for value, name, lo, hi in [(n, "n", 2, 1000), (replicates, "replicates", 1, 1024),
                               (generations, "generations", 0, 50), (seed, "seed", 1, 2**62-1),
                               (initial_state, "initial_state", 0, 2)]:
        _integer(value, name, lo, hi)
    check_slim_version(executable)
    command = [executable, "-s", str(seed)]
    for name, value in [("SELFING", selfing), ("N", n), ("REPLICATES", replicates),
                        ("GENERATIONS", generations), ("INITIAL_STATE", initial_state)]:
        command.extend(["-d", f"{name}={value}"])
    result = subprocess.run([*command, str(SCRIPT)], capture_output=True, text=True, check=True, timeout=180)
    observations = defaultdict(dict)
    for row in csv.reader(io.StringIO(result.stdout)):
        if row and row[0] == "DATA":
            if len(row) != 5:
                raise ValueError("Malformed SLiM output")
            time, replicate, copies, heterozygotes = map(int, row[1:])
            if (not 0 <= copies <= 2*n or not 0 <= heterozygotes <= min(copies, 2*n-copies)
                    or (copies-heterozygotes) % 2 or replicate in observations[time]):
                raise ValueError("Impossible genotype counts or duplicate replicate")
            observations[time][replicate] = (copies, heterozygotes)
    times = {0, generations//4, generations//2, generations}
    if generations > 0:
        times.add(1)
    if set(observations) != times or any(set(rows) != set(range(replicates)) for rows in observations.values()):
        raise ValueError("Missing or unexpected observations")
    return {t: [rows[i] for i in range(replicates)] for t, rows in sorted(observations.items())}


def build_report(executable="slim", replicates=256, generations=8, seed=20261023):
    n = 100
    treatments = []
    for offset, selfing in enumerate((0, 1)):
        counts = run_counts(selfing, n=n, replicates=replicates, generations=generations,
                            seed=seed+offset, executable=executable)
        checkpoints = []
        for time, rows in counts.items():
            p = [copies/(2*n) for copies, _ in rows]
            h = [heterozygotes/n for _, heterozygotes in rows]
            checkpoints.append({
                "generations_elapsed": time,
                "mean_observed_heterozygosity": round(statistics.mean(h), 9),
                "heterozygosity_standard_error": round(statistics.stdev(h)/sqrt(replicates), 9) if replicates > 1 else None,
                "expected_observed_heterozygosity": round(expected_heterozygosity(selfing, n, time), 9),
                "mean_allele_frequency": round(statistics.mean(p), 9),
                "mean_gene_diversity_2p1_minus_p": round(statistics.mean(2*x*(1-x) for x in p), 9),
            })
        treatments.append({"mode": "random_mating" if selfing == 0 else "complete_selfing",
                           "explicit_selfing_fraction": selfing, "seed": seed+offset, "checkpoints": checkpoints})
    comparisons = 2*sum(len(group["checkpoints"]) for group in treatments)
    tolerance = sqrt(log(2*comparisons/1e-6)/(2*replicates))
    return {
        "example": "slimulations: selfing removes heterozygotes without selection",
        "slim_version": SLIM_VERSION, "diploid_population_size": n,
        "replicates_per_treatment": replicates, "initial_state": "all individuals Aa; p=0.5; H_obs=1",
        "complete_selfing_formula": "E[H_obs(t)] = 2^(-t)",
        "random_mating_formula_for_t_ge_1": "E[H_obs(t)] = 0.5*(1-1/(2*N))^(t-1)",
        "bounded_mean_comparisons": comparisons,
        "simultaneous_mean_tolerance_alpha_1e_minus_6": round(tolerance, 9),
        "treatments": treatments,
        "interpretation": (
            "H_obs counts heterozygous individuals; 2p(1-p) is gene diversity. "
            "They differ under selfing. Ensemble allele frequency remains 0.5, "
            "but individual populations drift. No fitness or inbreeding-depression effect is modeled."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slim", default="slim")
    parser.add_argument("--replicates", type=int, default=256)
    parser.add_argument("--generations", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261023)
    args = parser.parse_args()
    try:
        report = build_report(args.slim, args.replicates, args.generations, args.seed)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
