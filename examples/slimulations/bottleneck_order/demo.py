"""Compare two orders of the same neutral Wright–Fisher census sizes."""
import argparse
import json
from math import comb, fsum, prod
from pathlib import Path
import statistics
import subprocess

HERE = Path(__file__).resolve().parent
REPLICATES = 4096
MODELS = {"small_first": ((4, 12, 40), 202611281),
          "small_second": ((12, 4, 40), 202611282)}


def transition(prior, copies):
    """Propagate a probability vector on allele counts for one generation."""
    n = len(prior) - 1
    return [
        fsum(w * comb(copies, k) * (i / n)**k * (1 - i / n)**(copies-k)
             for i, w in enumerate(prior) if w)
        for k in range(copies + 1)
    ]


def exact(path):
    probabilities = [0.0, 1.0, 0.0]  # Point mass at frequency 1/2.
    for copies in path:
        probabilities = transition(probabilities, copies)
    return probabilities


def run(path, seed, reps=REPLICATES):
    out = subprocess.run(
        ["slim", "-s", str(seed), "-d", f"REPS={reps}",
         "-d", f"FIRST={path[0]//2}", "-d", f"SECOND={path[1]//2}",
         str(HERE / "model.slim")],
        check=True, capture_output=True, text=True, timeout=90,
    ).stdout
    counts = [int(line.split()[1]) for line in out.splitlines()
              if line.startswith("COUNT ")]
    if len(counts) != reps or not all(0 <= k <= 40 for k in counts):
        raise ValueError("Unexpected number or range of allele counts")
    return counts


def report():
    result = {}
    for name, (path, seed) in MODELS.items():
        counts = run(path, seed)
        p = [k / path[-1] for k in counts]
        theory = exact(path)
        result[name] = dict(
            diploid_sizes=[m // 2 for m in path], seed=seed,
            replicates=len(counts), mean_frequency=statistics.mean(p),
            loss_count=counts.count(0), fixation_count=counts.count(path[-1]),
            observed_monomorphic=sum(k in (0, path[-1]) for k in counts) / len(counts),
            exact_monomorphic=theory[0] + theory[-1],
            observed_gene_diversity=statistics.mean(2 * x * (1-x) for x in p),
            exact_gene_diversity=0.5 * prod(1 - 1/m for m in path),
        )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-sample", action="store_true")
    rendered = json.dumps(report(), indent=2, sort_keys=True) + "\n"
    if parser.parse_args().check_sample:
        if rendered != (HERE / "sample_output.json").read_text():
            raise SystemExit("Seeded output differs from saved output")
    print(rendered, end="")
