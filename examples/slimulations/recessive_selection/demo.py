"""Compare actual SLiM offspring with exact selection and sampling predictions."""

import argparse
import csv
from fractions import Fraction
import io
import json
import math
from pathlib import Path
import statistics
import subprocess

from examples.slimulations.neutral_drift.demo import _integer, check_slim_version
from examples.slimulations.wahlund_effect.demo import founder_counts

SCRIPT = Path(__file__).with_name("selection.slim")
SAMPLE = Path(__file__).with_name("sample_summary.json")
MODES = ("neutral", "recessive", "additive", "dominant")
SEED = 202610280


def fitnesses(mode):
    if mode not in MODES:
        raise ValueError("Unknown selection mode")
    # aa, Aa, AA; decimal rationals retain exact arithmetic for the oracle.
    return {"neutral": (Fraction(1), Fraction(1), Fraction(1)),
            "recessive": (Fraction(1), Fraction(1), Fraction(1, 5)),
            "additive": (Fraction(1), Fraction(3, 5), Fraction(1, 5)),
            "dominant": (Fraction(1), Fraction(1, 5), Fraction(1, 5))}[mode]


def validate_counts(n, AA, het):
    _integer(n, "n", 2, 1000)
    _integer(AA, "AA", 0, n)
    _integer(het, "het", 0, n)
    if AA+het > n:
        raise ValueError("Genotype counts exceed the population size")


def exact_prediction(n, AA, het, mode):
    """Exact conditional expectation from actual parental genotype counts."""
    validate_counts(n, AA, het)
    wa, wh, wA = fitnesses(mode)
    mean_fitness = ((n-AA-het)*wa + het*wh + AA*wA)/n
    q = (AA*wA + Fraction(het, 2)*wh)/(n*mean_fitness)
    return {"gamete_frequency": q, "mean_parent_fitness": mean_fitness,
            "offspring_frequency_variance": q*(1-q)/(2*n),
            "offspring_AA_fraction": q*q, "offspring_heterozygote_fraction": 2*q*(1-q)}


def parse_output(stdout, n, AA, het, mode, replicates, generations):
    records, observed_fitness = {}, {}
    for row in csv.reader(io.StringIO(stdout)):
        if not row:
            continue
        if row[0] == "FIT":
            if len(row) != 3:
                raise ValueError("Malformed fitness audit")
            genotype, value = int(row[1]), float(row[2])
            if genotype in observed_fitness or genotype not in (0, 1, 2):
                raise ValueError("Duplicate or invalid fitness audit")
            observed_fitness[genotype] = value
        elif row[0] == "DATA":
            if len(row) != 6:
                raise ValueError("Malformed genotype record")
            t, rep, size, a, h = map(int, row[1:])
            if size != n or (t, rep) in records:
                raise ValueError("Invalid size or duplicate genotype record")
            validate_counts(n, a, h)
            records[t, rep] = (a, h)
    if set(records) != {(t, r) for t in range(generations+1) for r in range(replicates)}:
        raise ValueError("Missing or unexpected SLiM record")
    expected_genotypes = {g for g, count in enumerate((n-AA-het, het, AA)) if count}
    if set(observed_fitness) != expected_genotypes or any(
            not math.isclose(value, float(fitnesses(mode)[g]), abs_tol=1e-12)
            for g, value in observed_fitness.items()):
        raise ValueError("SLiM fitness audit does not match the analytical model")
    if any(records[0, r] != (AA, het) for r in range(replicates)):
        raise ValueError("SLiM founder counts differ from the requested counts")
    return {t: [records[t, r] for r in range(replicates)] for t in range(generations+1)}


def run_counts(mode="recessive", *, n=400, AA=1, het=38, replicates=512,
               generations=1, seed=SEED, executable="slim"):
    validate_counts(n, AA, het)
    fitnesses(mode)
    _integer(replicates, "replicates", 1, 2048)
    _integer(generations, "generations", 0, 1)
    _integer(seed, "seed", 1, 2**62-1)
    check_slim_version(executable)
    command = [executable, "-s", str(seed)]
    for name, value in [("MODE", MODES.index(mode)), ("N", n), ("AA", AA), ("HET", het),
                        ("REPLICATES", replicates), ("GENERATIONS", generations)]:
        command.extend(["-d", f"{name}={value}"])
    result = subprocess.run([*command, str(SCRIPT)], capture_output=True, text=True, check=True, timeout=60)
    return parse_output(result.stdout, n, AA, het, mode, replicates, generations)


def summarize(rows, n, initial_frequency):
    values = [(2*AA+het)/(2*n) for AA, het in rows]
    replicates = len(rows)
    return {"mean_frequency": round(statistics.mean(values), 9),
            "monte_carlo_se": round(statistics.stdev(values)/math.sqrt(replicates), 9) if replicates > 1 else None,
            "sample_frequency_variance": round(statistics.variance(values), 9) if replicates > 1 else None,
            "mean_AA_fraction": round(statistics.mean(AA/n for AA, _ in rows), 9),
            "mean_heterozygote_fraction": round(statistics.mean(het/n for _, het in rows), 9),
            "frequency_increased_replicates": sum(p > initial_frequency for p in values),
            "lost_replicates": sum(p == 0 for p in values),
            "fixed_replicates": sum(p == 1 for p in values)}


def build_report(executable="slim", replicates=512, seed=SEED):
    _integer(seed, "seed", 1, 2**62-9)
    scenarios = [("hardy_weinberg", p, mode) for p in (0.05, 0.5) for mode in MODES]
    scenarios.append(("all_copies_in_heterozygotes", 0.05, "recessive"))
    treatments = []
    for i, (founders, p, mode) in enumerate(scenarios):
        n, AA, het = founder_counts(400, p) if founders == "hardy_weinberg" else (400, 0, 40)
        counts = run_counts(mode, n=n, AA=AA, het=het, replicates=replicates, seed=seed+i, executable=executable)
        expected = exact_prediction(n, AA, het, mode)
        q = expected["gamete_frequency"]
        treatments.append({"founders": founders, "initial_frequency": p, "mode": mode,
                           "seed": seed+i, "founder_counts": {"aa": n-AA-het, "Aa": het, "AA": AA},
                           "fitness_aa_Aa_AA": [float(w) for w in fitnesses(mode)],
                           "exact_expected_frequency_fraction": str(q),
                           "prediction": {key: round(float(value), 9) for key, value in expected.items()},
                           "observed_after_one_generation": summarize(counts[1], n, p)})
    return {"example": "Recessive deleterious alleles are weakly selected when rare",
            "slim_version": "4.0.1", "eidos_version": "3.0.1", "python_version": "3.14.5",
            "diploid_population_size": 400, "replicates_per_treatment": replicates,
            "generations_elapsed": 1, "treatments": treatments,
            "interpretation": "Exact expectations condition on the founder genotype counts. Replicate means fluctuate; "
                              "some deleterious alleles increase by chance. All replicates are included. "
                              "This one-generation experiment does not establish long-term persistence or a mutation-selection balance."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slim", default="slim")
    parser.add_argument("--replicates", type=int, default=512)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--check-sample", action="store_true")
    args = parser.parse_args()
    try:
        text = json.dumps(build_report(args.slim, args.replicates, args.seed), indent=2, allow_nan=False)+"\n"
        if args.check_sample and text != SAMPLE.read_text():
            raise ValueError("Output differs from checked sample")
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        parser.error(str(error))
    print(text, end="")


if __name__ == "__main__":
    main()
