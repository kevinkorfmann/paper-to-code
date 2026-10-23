"""Feature groups can choose different depths without adding query parameters."""

import argparse
import json
from math import log, sqrt

from .model import route_depth

EPSILON = 1e-6
SOURCES = ((2.0, 0.0, 0.0, 2.0), (0.0, 2.0, 2.0, 0.0))


def build_report(strength=1.0):
    # Choose finite logits giving a 9:1 depth odds ratio in each of two heads.
    # The full-row RMS is sqrt(2+epsilon), and opposite logits differ by 4*b/RMS.
    coefficient = log(9)*sqrt(2+EPSILON)/4
    query = [strength*coefficient*sign for sign in (1, -1, 1, -1)]
    single = route_depth(SOURCES, query, heads=1, epsilon=EPSILON)
    multi = route_depth(SOURCES, query, heads=2, epsilon=EPSILON)
    uniform = route_depth(SOURCES, [0]*4, heads=2, epsilon=EPSILON)
    identical = route_depth([SOURCES[0]]*2, query, heads=2, epsilon=EPSILON)
    rounded = lambda values: [round(value, 9) for value in values]

    def result(read):
        return {"weights_by_head": [rounded(row) for row in read.weights], "output": rounded(read.output)}

    return {
        "example": "multi-head attention residuals: independent depth reads for feature groups",
        "randomness": "none; source vectors and pseudo-query are constructed, not trained",
        "sources": [list(row) for row in SOURCES],
        "query_strength": strength, "query": rounded(query),
        "epsilon": EPSILON, "key_normalization": "full source row; unit RMSNorm gain",
        "query_scalar_count_for_both_head_counts": len(query),
        "single_head": result(single), "two_heads": result(multi),
        "counterexamples": {"zero_query": result(uniform), "identical_sources": result(identical)},
        "convex_hull_witness": {
            "coordinate_0_plus_2_in_either_source": 2.0,
            "coordinate_0_plus_2_in_two_head_output": round(multi.output[0]+multi.output[2], 9),
        },
        "interpretation": (
            "Each head forms a convex mixture of its source slices. Their concatenation "
            "need not be a convex mixture of the full source vectors. This constructed "
            "routing example establishes no trained-model quality or speed improvement."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strength", type=float, default=1.0)
    args = parser.parse_args()
    try:
        report = build_report(args.strength)
    except (ValueError, TypeError, OverflowError) as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
