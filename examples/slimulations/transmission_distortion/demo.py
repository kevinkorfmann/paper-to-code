"""Actual SLiM transmission distortion with exact two-generation predictions."""

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

SCRIPT = Path(__file__).with_name("transmission.slim")
SAMPLE = Path(__file__).with_name("sample_summary.json")
SEED = 202610290


def validate(n, AA, het, tau):
    _integer(n, "n", 2, 1000)
    _integer(AA, "AA", 0, n)
    _integer(het, "het", 0, n)
    if AA+het > n:
        raise ValueError("Genotype counts exceed population size")
    if type(tau) not in (int, float, Fraction) or not math.isfinite(tau) or not 0 <= tau <= 1:
        raise ValueError("tau must be a finite probability")
    return Fraction(str(tau))


def exact_predictions(n, AA, het, tau):
    """Rational conditional means/variances, including second-generation drift."""
    tau = validate(n, AA, het, tau)
    q = (AA+tau*het)/n
    second_mean = q*q+2*tau*q*(1-q)
    # A gen-1 individual transmits with probability Y in {0,tau,1}.
    # Gen-2 gamete probability is the sample mean of N iid such Y values.
    variance_q2 = (q*q+2*tau*tau*q*(1-q)-second_mean**2)/n
    expected_q2_product = second_mean*(1-second_mean)-variance_q2
    return {
        1: {"mean_frequency": q, "frequency_variance": q*(1-q)/(2*n),
            "AA_fraction": q*q, "heterozygote_fraction": 2*q*(1-q)},
        2: {"mean_frequency": second_mean,
            "frequency_variance": variance_q2+expected_q2_product/(2*n),
            "AA_fraction": second_mean**2+variance_q2,
            "heterozygote_fraction": 2*expected_q2_product},
    }


def parse_output(stdout, n, AA, het, replicates, generations):
    records, fitness = {}, []
    if "Eidos version 3.0.1\n" not in stdout or "SLiM version 4.0.1\n" not in stdout:
        raise ValueError("Missing or unexpected engine versions")
    for row in csv.reader(io.StringIO(stdout)):
        if not row:
            continue
        if row[0] == "FIT":
            if len(row) != 3:
                raise ValueError("Malformed fitness audit")
            fitness.append(tuple(map(float, row[1:])))
        elif row[0] == "DATA":
            if len(row) != 6:
                raise ValueError("Malformed genotype record")
            t, rep, size, a, h = map(int, row[1:])
            if size != n or (t, rep) in records:
                raise ValueError("Invalid size or duplicate record")
            validate(n, a, h, 0.5)
            records[t, rep] = (a, h)
    if fitness != [(1.0, 1.0)]:
        raise ValueError("Founder fitness is not uniformly one")
    if set(records) != {(t, r) for t in range(generations+1) for r in range(replicates)}:
        raise ValueError("Missing or unexpected SLiM record")
    if any(records[0, r] != (AA, het) for r in range(replicates)):
        raise ValueError("Incorrect founder genotypes")
    return {t: [records[t, r] for r in range(replicates)] for t in range(generations+1)}


def run_counts(*, n=200, AA=50, het=100, tau=0.75, replicates=512,
               generations=2, seed=SEED, executable="slim"):
    validate(n, AA, het, tau)
    _integer(replicates, "replicates", 1, 2048)
    _integer(generations, "generations", 0, 2)
    _integer(seed, "seed", 1, 2**62-1)
    check_slim_version(executable)
    command = [executable, "-s", str(seed)]
    for name, value in (("N", n), ("AA", AA), ("HET", het), ("TAU", float(tau)),
                        ("REPLICATES", replicates), ("GENERATIONS", generations)):
        command.extend(["-d", f"{name}={value}"])
    result = subprocess.run([*command, str(SCRIPT)], capture_output=True, text=True,
                            check=True, timeout=120)
    return parse_output(result.stdout, n, AA, het, replicates, generations)


def summarize(rows, n):
    p = [(2*a+h)/(2*n) for a, h in rows]
    return {"mean_frequency": round(statistics.mean(p), 9),
            "frequency_variance": round(statistics.variance(p), 9) if len(p)>1 else None,
            "monte_carlo_se": round(statistics.stdev(p)/math.sqrt(len(p)), 9) if len(p)>1 else None,
            "mean_AA_fraction": round(statistics.mean(a/n for a, _ in rows), 9),
            "mean_heterozygote_fraction": round(statistics.mean(h/n for _, h in rows), 9),
            "lost_replicates": sum(x == 0 for x in p),
            "fixed_replicates": sum(x == 1 for x in p)}


def build_report(executable="slim", replicates=512, seed=SEED):
    _integer(seed, "seed", 1, 2**62-5)
    scenarios = [("hardy_weinberg", 50, 100, .5),
                 ("hardy_weinberg", 50, 100, .75),
                 ("hardy_weinberg", 50, 100, .25),
                 ("all_heterozygotes", 0, 200, .75),
                 ("only_homozygotes", 100, 0, .75)]
    treatments = []
    for offset, (label, AA, het, tau) in enumerate(scenarios):
        counts = run_counts(AA=AA, het=het, tau=tau, replicates=replicates,
                            seed=seed+offset, executable=executable)
        prediction = exact_predictions(200, AA, het, tau)
        treatments.append({"founders": label, "tau": tau, "seed": seed+offset,
                           "founder_counts": {"aa": 200-AA-het, "Aa": het, "AA": AA},
                           "checkpoints": [{"generations_elapsed": t,
                                            "exact_expected_frequency_fraction": str(prediction[t]["mean_frequency"]),
                                            "prediction": {key: round(float(value), 9) for key, value in prediction[t].items()},
                                            "observed": summarize(counts[t], 200)} for t in (1, 2)]})
    return {"example": "Transmission distortion can change frequency without a parental fitness advantage",
            "slim_version": "4.0.1", "eidos_version": "3.0.1", "reference_python_version": "3.14.5",
            "diploid_population_size": 200, "replicates_per_treatment": replicates,
            "initial_frequency": 0.5, "fitness_aa_Aa_AA": [1, 1, 1], "treatments": treatments,
            "interpretation": "The same transmission bias acts in both parental roles. Only heterozygotes express it. "
                              "All finite replicates are included. This is a generic inheritance-probability model, "
                              "not a molecular mechanism, an R2d2 model, or a prediction for a real population."}


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
