"""Run real hmmIBD inference on synthetic haploid pairs and independent checks."""

import argparse
import csv
import io
import json
import math
from pathlib import Path
import platform
import random
import subprocess
import tempfile

from .build import BINARY, COMMIT, SOURCE_SHA256

HERE = Path(__file__).resolve().parent
FIXTURE = HERE / "synthetic_pairs.json"
SAMPLE = HERE / "sample_output.json"
SEED = 20261028
ERROR = 0.001
RECOMBINATION = 7.4e-7  # per bp per outcrossed meiosis, upstream P. falciparum default


def generate_fixture(seed=SEED):
    rng = random.Random(seed)
    cases = {}
    for name in ["blocks", "identical", "independent"]:
        left, right = [], []
        for i in range(1200):
            shared = name == "identical" or name == "blocks" and 400 <= i < 800
            a = int(rng.random() < 0.5)
            b = a if shared else int(rng.random() < 0.5)
            a ^= int(rng.random() < ERROR)
            b ^= int(rng.random() < ERROR)
            left.append(str(a))
            right.append(str(b))
        cases[name] = {"left": "".join(left), "right": "".join(right)}
    return {"seed": seed, "spacing_bp": 1000, "allele1_frequency": 0.5,
            "genotype_error_probability": ERROR, "cases": cases,
            "truth_for_evaluation_only": {"blocks_ibd_indices_half_open": [400, 800],
                                         "identical_ibd_sites": 1200, "independent_ibd_sites": 0}}


def validate(left, right, positions, frequency):
    if not left or len(left) != len(right) or len(left) != len(positions):
        raise ValueError("Genotype and position lengths must match and be nonempty")
    if any(x not in (0, 1, -1) for x in [*left, *right]):
        raise ValueError("Only biallelic haploid calls 0, 1 or missing -1 are supported")
    if any(type(p) is not int or not 1 <= p <= 2_000_000_000 for p in positions):
        raise ValueError("Positions must be positive bounded integers")
    if any(b - a < 5 for a, b in zip(positions, positions[1:])):
        raise ValueError("Positions must increase by at least the upstream 5 bp thinning limit")
    if not math.isfinite(frequency) or not 0 < frequency < 1:
        raise ValueError("Allele-1 frequency must be strictly between zero and one")


def emission(a, b, frequency, error=ERROR):
    """Sum over unobserved alleles; state 0 shares one ancestral allele."""
    if a == -1 or b == -1:
        return (1.0, 1.0)
    p = (1 - frequency, frequency)
    def observation(observed, latent):
        return 1 - error if observed == latent else error
    shared = sum(p[x] * observation(a, x) * observation(b, x) for x in (0, 1))
    independent = math.prod(sum(p[x] * observation(observed, x) for x in (0, 1))
                            for observed in (a, b))
    return shared, independent


def transition(distance, prior=0.5, generations=1.0):
    """Poisson no-reset probability plus a fresh draw from the stationary prior."""
    stay = math.exp(-RECOMBINATION * generations * distance)
    p = (prior, 1 - prior)
    return tuple(tuple(stay * (i == j) + (1 - stay) * p[j] for j in (0, 1)) for i in (0, 1))


def posterior_oracle(left, right, positions, frequency=0.5, prior=0.5):
    """Independent scaled sum-product calculation, conditional on k=1 and prior."""
    validate(left, right, positions, frequency)
    if not 0 < prior < 1:
        raise ValueError("Prior must be strictly between zero and one")
    observations = [emission(a, b, frequency) for a, b in zip(left, right)]
    edges = [transition(b - a, prior) for a, b in zip(positions, positions[1:])]
    forward = []
    current = [prior, 1 - prior]
    for i, obs in enumerate(observations):
        if i:
            current = [sum(current[s] * edges[i - 1][s][t] for s in (0, 1)) for t in (0, 1)]
        current = [current[s] * obs[s] for s in (0, 1)]
        total = sum(current)
        current = [x / total for x in current]
        forward.append(current)
    backward = [1.0, 1.0]
    posterior = [0.0] * len(left)
    for i in range(len(left) - 1, -1, -1):
        joint = [forward[i][s] * backward[s] for s in (0, 1)]
        posterior[i] = joint[0] / sum(joint)
        if i:
            backward = [sum(edges[i - 1][s][t] * observations[i][t] * backward[t]
                            for t in (0, 1)) for s in (0, 1)]
            scale = max(backward)
            backward = [x / scale for x in backward]
    return posterior


def run_tool(left, right, positions, frequency=0.5, *, iterations=50, prior=None):
    """Infer using the external CLI. Truth is never an input to this function."""
    validate(left, right, positions, frequency)
    if type(iterations) is not int or not 1 <= iterations <= 100:
        raise ValueError("Iterations must be an integer from 1 to 100")
    if prior is not None and (not math.isfinite(prior) or not 0 < prior < 1):
        raise ValueError("Prior must be strictly between zero and one")
    if not BINARY.is_file():
        raise FileNotFoundError("Build the pinned dependency: python -m examples.inference_showcase.hmmibd_tracts.build")
    # Short relative names avoid upstream fixed-size path buffers; never use shell=True.
    with tempfile.TemporaryDirectory(prefix="hmmibd-") as directory:
        path = Path(directory)
        path.joinpath("data.tsv").write_text("chrom\tpos\tleft\tright\n" + "".join(
            f"1\t{pos}\t{a}\t{b}\n" for pos, a, b in zip(positions, left, right)))
        path.joinpath("freq.tsv").write_text("".join(
            f"1\t{pos}\t{1-frequency:.17g}\t{frequency:.17g}\n" for pos in positions))
        args = [str(BINARY), "-i", "data.tsv", "-f", "freq.tsv", "-o", "result", "-m", str(iterations)]
        if prior is not None:
            args += ["-r", str(prior)]
        subprocess.run(args, cwd=directory, capture_output=True, text=True, check=True, timeout=30)
        summaries = list(csv.DictReader(io.StringIO(path.joinpath("result.hmm_fract.txt").read_text()), delimiter="\t"))
        if len(summaries) != 1:
            raise ValueError("hmmIBD did not report exactly one pair; at least 10 informative markers are required")
        summary = summaries[0]
        segments = [{"start_bp": int(r["start"]), "end_bp": int(r["end"]),
                     "state": int(r["different"]), "markers": int(r["Nsnp"])}
                    for r in csv.DictReader(io.StringIO(path.joinpath("result.hmm.txt").read_text()), delimiter="\t")]
    result = {"posterior_mean_ibd": float(summary["fract_sites_IBD"]),
              "viterbi_marker_fraction_ibd": float(summary["fract_vit_sites_IBD"]),
              "viterbi_segments": segments}
    if not all(math.isfinite(result[key]) and 0 <= result[key] <= 1
               for key in ("posterior_mean_ibd", "viterbi_marker_fraction_ibd")):
        raise ValueError("Non-finite or invalid tool result")
    if sum(s["markers"] for s in segments) != len(left):
        raise ValueError("Unexpected marker thinning or incomplete output")
    return result


def case_inputs(fixture, name):
    pair = fixture["cases"][name]
    return ([int(x) for x in pair["left"]], [int(x) for x in pair["right"]],
            [1 + fixture["spacing_bp"] * i for i in range(len(pair["left"]))])


def report():
    fixture = json.loads(FIXTURE.read_text())
    fits = {name: run_tool(*case_inputs(fixture, name)) for name in fixture["cases"]}
    left, right, positions = case_inputs(fixture, "blocks")
    conditional = run_tool(left, right, positions, iterations=1, prior=0.5)
    oracle = sum(posterior_oracle(left, right, positions)) / len(left)
    # Widely separated markers make the transition matrix approach its stationary rows.
    independent_positions = [1 + 100_000_000 * i for i in range(12)]
    sensitivity = {}
    for frequency in (0.5, 0.1):
        result = run_tool([1] * 12, [1] * 12, independent_positions, frequency, iterations=1, prior=0.5)
        ibd, nonibd = emission(1, 1, frequency)
        sensitivity[str(frequency)] = {"tool_posterior_mean_ibd": result["posterior_mean_ibd"],
                                       "independent_marker_bayes_oracle": round(ibd / (ibd + nonibd), 9)}
    return {"seed": SEED, "python": platform.python_version(), "hmmibd_commit": COMMIT,
            "source_sha256": SOURCE_SHA256, "markers_per_pair": len(left),
            "truth_ibd_marker_fraction_blocks": round(400 / 1200, 9), "fits": fits,
            "fixed_parameter_check": {"prior": 0.5, "generations_used_for_decoding": 1,
                                      "tool_posterior_mean_ibd": conditional["posterior_mean_ibd"],
                                      "oracle_posterior_mean_ibd": round(oracle, 9)},
            "matching_allele_frequency_control": sensitivity}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-sample", action="store_true")
    args = parser.parse_args()
    text = json.dumps(report(), indent=2, sort_keys=True) + "\n"
    if args.check_sample and text != SAMPLE.read_text():
        raise SystemExit("Output differs from checked sample")
    print(text, end="")


if __name__ == "__main__":
    main()
