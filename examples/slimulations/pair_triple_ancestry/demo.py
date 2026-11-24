"""Matched pairwise ancestry, different three-way sharing in one SLiM generation."""
from argparse import ArgumentParser
from fractions import Fraction
import json
from math import comb
from pathlib import Path
import re
from statistics import mean, stdev
from subprocess import run

HERE = Path(__file__).resolve().parent
SEED = 20261124
WEIGHTS = ((1, 1, 7, 7), (1, 3, 3, 9))


def expectation(weights, k):
    if len(weights) != 4 or min(weights) < 0 or sum(weights) == 0 or k not in (2, 3):
        raise ValueError("need four nonnegative weights and order 2 or 3")
    return sum(Fraction(w, sum(weights)) ** k for w in weights)


def sharing(counts, k):
    n = sum(counts)
    if len(counts) != 4 or min(counts) < 0 or not 2 <= k <= n:
        raise ValueError("bad marker counts or sharing order")
    return Fraction(sum(comb(c, k) for c in counts), comb(n, k))


def parse_counts(stdout, reps, n):
    rows = {}
    for line in stdout.splitlines():
        if line.startswith("DATA,"):
            _, i, *counts = line.split(",")
            i, counts = int(i), tuple(map(int, counts))
            if len(counts) != 4 or min(counts) < 0 or sum(counts) != n or i in rows:
                raise ValueError("bad offspring marker census")
            rows[i] = counts
    if set(rows) != set(range(reps)):
        raise ValueError("missing or extra replicate rows")
    return [rows[i] for i in range(reps)]


def simulate(slim, mode, seed, reps, n, model=None):
    version = run([slim, "-v"], check=True, capture_output=True, text=True).stdout
    if not re.search(r"^SLiM version 4\.0\.1(?:,|\s|$)", version, re.MULTILINE):
        raise ValueError("reference version is SLiM 4.0.1")
    if model is None:
        model = (HERE / "model.slim").read_text()
    # Streaming also lets a read-only runner exercise the exact model bytes.
    out = run([slim, "-s", str(seed), "-d", f"REPS={reps}", "-d", f"MODE={mode}",
               "-d", f"OFFSPRING={n}", "/dev/stdin"], input=model, check=True,
              capture_output=True, text=True)
    return parse_counts(out.stdout, reps, n)


def report(slim="slim", seed=SEED, reps=512, n=32, model=None):
    if not (2 <= reps <= 2048 and 3 <= n <= 64 and seed > 0):
        raise ValueError("invalid settings")
    results = []
    for mode, w in enumerate(WEIGHTS):
        rows = simulate(slim, mode, seed+mode, reps, n, model)
        result = {"weights": list(w), "seed": seed+mode}
        for k, label in ((2, "pair"), (3, "triple")):
            measurements = [float(sharing(row, k)) for row in rows]
            result[label] = {
                "mean": round(mean(measurements), 9),
                "monte_carlo_se": round(stdev(measurements)/reps**0.5, 9),
                "exact_expectation": float(expectation(w, k)),
            }
        results.append(result)
    return {"slim": "4.0.1", "cohorts_per_lottery": reps, "offspring_per_cohort": n,
            "replicate_aggregation": "mean of within-cohort distinct-offspring sharing fractions",
            "results": results}


def main(argv=None):
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--slim", default="slim")
    parser.add_argument("--check-sample", action="store_true")
    args = parser.parse_args(argv)
    output = json.dumps(report(args.slim), indent=2, allow_nan=False) + "\n"
    if args.check_sample:
        if output.encode() != (HERE / "sample_summary.json").read_bytes():
            raise SystemExit("sample output mismatch")
        print("sample output matches")
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
