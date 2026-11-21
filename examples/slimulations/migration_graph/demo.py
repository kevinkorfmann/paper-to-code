"""Reproduce a neutral marker's short-term response to migration topology."""

import argparse
import json
import math
import statistics
import subprocess
from pathlib import Path

N = 40
REPLICATES = 256
M = 0.3
SEEDS = {"ring": 202611211, "complete": 202611212}


def prediction(graph, generations=2):
    """Predict E[p] by applying backward parent-source probabilities."""
    rows = [[1.0, 0.0, 0.0, 0.0]]
    for _ in range(generations):
        last = rows[-1]
        next_row = []
        for i in range(4):
            sources = (
                [(i + 1) % 4, (i - 1) % 4]
                if graph == "ring"
                else [j for j in range(4) if j != i]
            )
            next_row.append(
                (1 - M) * last[i] + M * statistics.mean(last[j] for j in sources)
            )
        rows.append(next_row)
    return rows


def run(graph, slim="slim"):
    completed = subprocess.run(
        [slim, "-s", str(SEEDS[graph]), "-d", f"N={N}", "-d", f"R={REPLICATES}",
         "-d", f"M={M}", "-d", f"RING={'T' if graph == 'ring' else 'F'}",
         str(Path(__file__).with_name("model.slim"))],
        text=True, capture_output=True, check=True, timeout=90,
    )
    rows = {}
    for line in completed.stdout.splitlines():
        if line.startswith("DATA,"):
            tick, replicate, *counts = map(int, line.split(",")[1:])
            key = (tick, replicate)
            if (key in rows or len(counts) != 4
                    or not all(0 <= x <= 2 * N for x in counts)):
                raise ValueError(f"invalid output row: {line}")
            rows[key] = [x / (2 * N) for x in counts]
    expected_keys = {(g, r) for g in range(3) for r in range(REPLICATES)}
    if rows.keys() != expected_keys:
        raise ValueError("incomplete SLiM output")
    if graph == "ring" and any(rows[1, r][2] != 0 for r in range(REPLICATES)):
        raise ValueError("opposite deme received impossible one-step migrant")
    out = []
    for g, expected in enumerate(prediction(graph)):
        values = [rows[g, r] for r in range(REPLICATES)]
        cols = list(zip(*values))
        means = list(map(statistics.mean, cols))
        ses = [statistics.stdev(col) / math.sqrt(REPLICATES) for col in cols]
        if not all(abs(a-b) <= 6*s + .002 for a, b, s in zip(means, expected, ses)):
            raise ValueError("observed means do not agree with matrix expectation")
        out.append({
            "generation": g,
            "expected": [round(x, 6) for x in expected],
            "observed": [round(x, 6) for x in means],
            "mc_se": [round(x, 6) for x in ses],
            "observed_pooled": round(statistics.mean(means), 6),
        })
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slim", default="slim")
    parser.add_argument("--check-sample", action="store_true")
    args = parser.parse_args()
    rendered = json.dumps({g: run(g, args.slim) for g in SEEDS}, indent=2) + "\n"
    if args.check_sample:
        if rendered != Path(__file__).with_name("sample_output.json").read_text():
            raise SystemExit("sample output does not match")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
