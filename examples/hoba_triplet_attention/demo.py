"""Show joint dependence that identical one-axis marginals cannot preserve."""

import argparse
import json
import math
from pathlib import Path
import platform

from .model import block_attention, triplet_read

SAMPLE = Path(__file__).with_name("sample_output.json")


def describe(query):
    signs = [[-1], [1]]
    result = triplet_read([query], signs, signs, signs, signs)
    rows = [math.fsum(row) for row in result.weights]
    cols = [math.fsum(row[k] for row in result.weights) for k in range(2)]
    independent_output = math.prod((math.fsum(p * v[0] for p, v in zip(rows, signs)),
                                    math.fsum(p * v[0] for p, v in zip(cols, signs))))
    return {"query": query, "joint_weights": result.weights,
            "first_marginal": rows, "second_marginal": cols,
            "joint_output": result.output[0],
            "product_of_marginals_output": independent_output}


def build_report():
    q = [[0], [0], [0]]
    keys = [[1], [1], [1]]
    values = [[1], [1], [10]]
    local = block_attention(q, keys, keys, values, values, 2)
    dense = block_attention(q, keys, keys, values, values, 3)
    return {
        "example": "HOBA joint triplet attention: marginals can hide dependence",
        "python_version": platform.python_version(),
        "randomness": "None; fixed supplied projections.",
        "sign_pairs": [describe(1), describe(-1), describe(0)],
        "block_boundary_control": {
            "queries": [0, 0, 0], "values_both_branches": [1, 1, 10],
            "block_size_two_outputs": [x.output[0] for x in local],
            "one_full_block_outputs": [x.output[0] for x in dense],
            "local_weight_shapes": [[len(x.weights), len(x.weights[0])] for x in local],
        },
        "scope": "Single-head supplied-input kernel and nonoverlapping blocks only. Bidirectional within blocks; no training, full HOBA, speed or model-quality claims.",
    }


def render(report):
    def rounded(value):
        if isinstance(value, float):
            return round(value, 12)
        if isinstance(value, (list, tuple)):
            return [rounded(x) for x in value]
        if isinstance(value, dict):
            return {key: rounded(val) for key, val in value.items()}
        return value
    return json.dumps(rounded(report), indent=2, allow_nan=False) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-sample", action="store_true")
    args = parser.parse_args()
    output = render(build_report())
    if args.check_sample:
        if output != SAMPLE.read_text(encoding="utf-8"):
            parser.error("Output differs from checked sample, including interpreter version")
        print("sample output matches")
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
