"""One-generation variance of signed haplotype LD under neutral drift."""
from argparse import ArgumentParser
import json
from pathlib import Path
import re
from statistics import mean, stdev
from subprocess import run

HERE = Path(__file__).resolve().parent
SEED = 20261123


def ld(counts):
    """With the haplotype order AB, Ab, aB, ab, return D."""
    aboth, aonly, bonly, neither = counts
    n = sum(counts)
    if n <= 0 or min(counts) < 0:
        raise ValueError("haplotype counts must be nonnegative and nonempty")
    # Integer numerator preserves the sign and exact zeros.
    return (aboth * neither - aonly * bonly) / n**2


def second_moment(n):
    if n < 1:
        raise ValueError("need at least one haplotype")
    return (n - 1) / (16 * n**2)


def parse_counts(stdout, n, replicates):
    expected = {(r, t) for r in range(1, replicates + 1) for t in (0, 1)}
    observed = {}
    for line in stdout.splitlines():
        if line.startswith("DATA,"):
            _, rep, generation, *values = line.split(",")
            key = int(rep), int(generation)
            counts = tuple(map(int, values))
            if (len(counts) != 4 or key not in expected or key in observed
                    or min(counts) < 0 or sum(counts) != 2*n):
                raise ValueError(f"invalid count record: {line}")
            observed[key] = counts
    if observed.keys() != expected:
        raise ValueError("missing population records")
    if any(observed[r, 0] != (n//2,) * 4 for r in range(1, replicates+1)):
        raise ValueError("founder haplotypes are not exactly balanced")
    return [observed[r, 1] for r in range(1, replicates+1)]


def simulate(binary, n, replicates, seed):
    if not (2 <= n <= 200 and n % 2 == 0 and 2 <= replicates <= 4096 and seed > 0):
        raise ValueError("need even N in [2,200], replicates in [2,4096] and seed > 0")
    version = run([binary, "-v"], check=True, capture_output=True, text=True).stdout
    if not re.search(r"^SLiM version 4\.0\.1(?:,|\s|$)", version, re.MULTILINE):
        raise ValueError("reference version is SLiM 4.0.1")
    result = run([binary, "-s", str(seed), "-d", f"N={n}", "-d", f"REPS={replicates}",
                  str(HERE / "model.slim")], check=True, capture_output=True, text=True)
    rows = parse_counts(result.stdout, n, replicates)
    ds = [ld(row) for row in rows]
    squares = [d*d for d in ds]
    return {
        "diploids": n,
        "haps": 2*n,
        "seed": seed,
        "replicates": len(rows),
        "mean_D": round(mean(ds), 9),
        "mean_D_expected": 0.0,
        "mean_D_mcse": round(stdev(ds) / len(rows)**0.5, 9),
        "mean_D_squared": round(mean(squares), 9),
        "mean_D_squared_expected": round(second_moment(2*n), 9),
        "mean_D_squared_mcse": round(stdev(squares) / len(rows)**0.5, 9),
        "positive_D": sum(d > 0 for d in ds),
        "negative_D": sum(d < 0 for d in ds),
        "zero_D": sum(d == 0 for d in ds),
    }


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--slim", default="slim")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--replicates", type=int, default=2048)
    parser.add_argument("--check-sample", action="store_true")
    args = parser.parse_args()
    results = [simulate(args.slim, n, args.replicates, args.seed+i)
               for i, n in enumerate((10, 40))]
    output = json.dumps({"slim_version": "4.0.1", "generation": 1, "results": results},
                        indent=2, allow_nan=False) + "\n"
    if args.check_sample:
        if output.encode() != (HERE / "sample_summary.json").read_bytes():
            raise SystemExit("output differs from checked sample")
        print("sample output matches")
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
