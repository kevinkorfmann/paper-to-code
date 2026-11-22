"""Neutral covariance of allele-frequency levels and increments."""
from argparse import ArgumentParser
import json
from pathlib import Path
import re
from statistics import mean, stdev
from subprocess import run

HERE = Path(__file__).resolve().parent
TIMES = (0, 6, 12, 18)
REFERENCE_SEED = 20261122


def model_covariance(n, p0, first, second):
    """Exact covariance of two intervals' changes in the neutral WF null."""
    a, b = first
    c, d = second
    if not (0 <= a <= b and 0 <= c <= d and n >= 2 and 0 <= p0 <= 1):
        raise ValueError("bad interval or population parameters")
    lo, hi = max(a, c), min(b, d)
    if lo >= hi:
        return 0.0
    retention = 1 - 1 / (2 * n)
    return p0 * (1 - p0) * (retention ** lo - retention ** hi)


def parse_counts(stdout, n, reps, initial_copies):
    expected = {(rep, time) for rep in range(1, reps + 1) for time in TIMES}
    counts = {}
    for line in stdout.splitlines():
        if line.startswith("DATA,"):
            _, rep, t, copies = line.split(",")
            key = int(rep), int(t)
            copies = int(copies)
            if key not in expected or key in counts or not 0 <= copies <= 2 * n:
                raise ValueError(f"unexpected count row: {line}")
            counts[key] = copies
    if counts.keys() != expected:
        raise ValueError("missing count rows; retain every population and time")
    rows = [[counts[rep, time] for time in TIMES] for rep in range(1, reps + 1)]
    if any(row[0] != initial_copies for row in rows):
        raise ValueError("initial copy count did not match fixture")
    return rows


def simulate(slim, n, reps, initial_copies, seed):
    if not (2 <= n <= 200 and 2 <= reps <= 4096 and 0 <= initial_copies <= 2*n):
        raise ValueError("invalid population size, replicates, or initial count")
    version = run([slim, "-v"], check=True, capture_output=True, text=True).stdout
    if re.search(r"^SLiM version 4\.0\.1(?:,|\s|$)", version, re.MULTILINE) is None:
        raise ValueError("reference replay requires SLiM 4.0.1")
    result = run([slim, "-s", str(seed), "-d", f"N={n}", "-d", f"REPS={reps}",
                  "-d", f"COPIES={initial_copies}", str(HERE / "model.slim")],
                 check=True, capture_output=True, text=True)
    return parse_counts(result.stdout, n, reps, initial_copies)


def summarize(rows, n, initial_copies, seed):
    cases = {
        "levels_generations_6_and_12": ((0, 6), (0, 12)),
        "disjoint_increments_0_6_and_6_12": ((0, 6), (6, 12)),
        "overlapping_increments_0_12_and_6_18": ((0, 12), (6, 18)),
    }
    p0, denom = initial_copies / (2 * n), (2 * n) ** 2
    positions = {t: i for i, t in enumerate(TIMES)}
    covariance = {}
    for label, (first, second) in cases.items():
        a, b, c, d = (positions[t] for t in (*first, *second))
        # E[p_b - p_a] = 0; for the "levels" case p_a is known and fixed.
        products = [(r[b] - r[a]) * (r[d] - r[c]) / denom for r in rows]
        covariance[label] = {
            "estimate": round(mean(products), 9),
            "monte_carlo_se": round(stdev(products) / len(rows) ** 0.5, 9),
            "wf_expectation": round(model_covariance(n, p0, first, second), 9),
        }
    return {
        "slim_version": "4.0.1",
        "seed": seed,
        "diploids_per_population": n,
        "replicates": len(rows),
        "initial_copies": initial_copies,
        "initial_frequency": p0,
        "census_generations": list(TIMES),
        "covariance_units": "allele frequency squared",
        "covariances": covariance,
        "lost_at_final_census": sum(row[-1] == 0 for row in rows),
        "fixed_at_final_census": sum(row[-1] == 2*n for row in rows),
    }


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--slim", default="slim")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--replicates", type=int, default=2048)
    parser.add_argument("--initial-copies", type=int, default=10)
    parser.add_argument("--seed", type=int, default=REFERENCE_SEED)
    parser.add_argument("--check-sample", action="store_true")
    args = parser.parse_args()
    rows = simulate(args.slim, args.n, args.replicates, args.initial_copies, args.seed)
    output = json.dumps(summarize(rows, args.n, args.initial_copies, args.seed), indent=2,
                        allow_nan=False) + "\n"
    if args.check_sample:
        if output.encode() != (HERE / "sample_summary.json").read_bytes():
            raise SystemExit("output differs from checked sample")
        print("sample output matches")
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
