"""A hand-designed rollback walk suppresses middle payloads, not the oldest."""

import argparse
import json

from .model import IncrementalAttention, causal_attention, walk_from_increments


def rounded(value):
    if isinstance(value, dict):
        return {key: rounded(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [rounded(item) for item in value]
    return round(value, 9) if type(value) is float else value


def build_report(beta=4.0):
    labels = ["anchor", "draft", "detail", "rollback", "replacement", "query"]
    walk = walk_from_increments([1, 1, 1, -2, 1, 0])
    queries = [[1]] * len(labels)
    keys = [[2], [2], [2], [-2], [2], [-2]]
    # One-hot payload values let output coordinates identify their contributions.
    values = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0],
              [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]]
    rows = causal_attention(queries, keys, values, walk, walk, beta)
    # A constant stack walk has zero biases: ordinary causal attention.
    plain = causal_attention(queries, keys, values, [0]*6, [0]*6, beta)
    decoder = IncrementalAttention(beta)
    decoded = tuple(decoder.step(q, k, v, w, w)
                    for q, k, v, w in zip(queries, keys, values, walk, strict=True))
    error = max(abs(a-b) for direct, online in zip(rows, decoded, strict=True)
                for a, b in zip(direct.output, online.output, strict=True))

    def summary(row):
        return {"biases": row.biases, "weights": row.weights,
                "payload_output": row.output,
                "draft_and_detail_mass": row.weights[1] + row.weights[2]}

    return rounded({
        "example": "First Passage Attention: hand-designed rollback",
        "beta": float(beta), "tokens": labels, "walk_field_and_levels": walk,
        "content_logits_at_final_query": [k[0] for k in keys],
        "payload_coordinates": ["anchor", "draft", "detail", "replacement"],
        "ordinary_causal_attention": summary(plain[-1]),
        "first_passage_attention": summary(rows[-1]),
        "incremental_vs_full_max_output_error": error,
        "interpretation": "The supplied walk gives middle payloads negative biases while "
                          "preserving the oldest anchor and the replacement. Finite beta "
                          "attenuates rather than hard-deletes them. These fields are "
                          "hand-designed, not learned relevance or rollback capability.",
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--beta", type=float, default=4.0,
                        help="strictly positive bias strength (default: 4)")
    args = parser.parse_args()
    try:
        report = build_report(args.beta)
    except (TypeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
