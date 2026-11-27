"""Run the seeded SLiM experiment and report allele-frequency precision."""
import argparse
import json
from pathlib import Path
import statistics
import subprocess

ROOT = Path(__file__).resolve().parent
SEED = 20261127
REPS = 2048


def parse(raw):
    rows = [tuple(map(int, line.split(",")[1:]))
            for line in raw.splitlines() if line.startswith("DATA,")]
    if len(rows) != REPS or any(i != row[0] or len(row) != 3
                                or not all(0 <= n <= 20 for n in row[1:])
                                for i, row in enumerate(rows)):
        raise ValueError("unexpected SLiM output")
    return rows


def run(slim):
    version = subprocess.check_output([slim, "-v"], text=True).splitlines()[0]
    if not version.startswith("SLiM version 4.0.1,"):
        raise ValueError("the checked replay requires SLiM 4.0.1")
    raw = subprocess.check_output([slim, "-s", str(SEED), str(ROOT / "model.slim")],
                                  text=True)
    return parse(raw)


def summary(rows):
    stats = {}
    for index, label, theory in ((1, "unrelated", 1 / 80),
                                 (2, "siblings", 11 / 160)):
        frequencies = [row[index] / 20 for row in rows]
        stats[label] = {
            "mean_frequency": round(statistics.mean(frequencies), 6),
            "variance": round(statistics.variance(frequencies), 6),
            "theory_variance": theory,
        }
    return {"replicates": REPS, "seed": SEED, "groups": stats}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--slim", default="slim")
    parser.add_argument("--check-sample", action="store_true")
    args = parser.parse_args()
    output = json.dumps(summary(run(args.slim)), indent=2, sort_keys=True) + "\n"
    if args.check_sample and output != (ROOT / "sample_output.json").read_text():
        raise SystemExit("checked output differs")
    print(output, end="")
