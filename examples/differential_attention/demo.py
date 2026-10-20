"""A four-token algebra experiment, not a trained model or a benchmark."""

import argparse
import json
from math import log

from .model import differential_attention, lambda_from_vectors, lambda_initial


def build_report(lambda_value: float = 1.0, background: float = 10.0) -> dict:
    # d=1; log probabilities are keys, and every query is 1.
    # The last causal query sees all four keys, realizing these two maps.
    queries = ((1.0,),) * 4
    keys1 = tuple((log(p),) for p in (0.4, 0.2, 0.2, 0.2))
    keys2 = tuple((log(p),) for p in (0.1, 0.3, 0.3, 0.3))
    values = tuple((background, float(i == 0)) for i in range(4))

    def run(k2, vals, coefficient):
        return differential_attention(
            queries, queries, keys1, k2, vals, coefficient, causal=True
        )

    result = run(keys2, values, lambda_value)
    standard = run(keys2, values, 0.0)
    identical = run(keys1, values, lambda_value)
    varying_values = tuple((background + offset, float(i == 0))
                           for i, offset in enumerate((0, -1, 1, 2)))
    varying = run(keys2, varying_values, lambda_value)
    rounded = lambda row: [round(x, 9) + 0.0 for x in row]
    return {
        "example": "Differential Transformer: one attention head before normalization",
        "paper": "https://arxiv.org/abs/2410.05258v2",
        "parameters": {
            "tokens": 4, "head_dimension_d": 1, "value_dimension": 2,
            "causal": True, "shown_query_index_zero_based": 3,
            "lambda": lambda_value, "shared_value_component": background,
        },
        "last_query": {
            "softmax_map_1": rounded(result.map1[-1]),
            "softmax_map_2": rounded(result.map2[-1]),
            "signed_difference_weights": rounded(result.weights[-1]),
            "weight_sum": round(sum(result.weights[-1]), 9) + 0.0,
            "standard_attention_output": rounded(standard.output[-1]),
            "differential_attention_output": rounded(result.output[-1]),
        },
        "counterexamples_last_query": {
            "identical_maps_output": rounded(identical.output[-1]),
            "nonconstant_background_output": rounded(varying.output[-1]),
        },
        "paper_lambda_parameterization_example": {
            "layer_one_initial": round(lambda_initial(1), 9),
            "zero_vectors_layer_one": round(lambda_from_vectors([0], [0], [0], [0], 1), 9),
        },
        "interpretation": (
            "Values have two channels: constructed background and a token-0 marker. "
            "At lambda=1, a constant value channel cancels; unequal maps retain a contrast. "
            "Identical maps also cancel useful signal, and a nonconstant background need not cancel. "
            "These identities are not evidence of improved LLM performance."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lambda-value", type=float, default=1.0)
    parser.add_argument("--background", type=float, default=10.0)
    args = parser.parse_args()
    try:
        report = build_report(args.lambda_value, args.background)
    except (TypeError, ValueError, OverflowError) as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
