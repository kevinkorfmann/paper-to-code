"""One shared top-k route can preserve average mass but miss a layer."""
import argparse
from itertools import combinations
import json
from math import exp, fsum, log
from pathlib import Path


def softmax(logits):
    high = max(logits)
    weights = [exp(x - high) for x in logits]
    return tuple(x / fsum(weights) for x in weights)


def route(scores, k):
    if not 1 <= k <= len(scores):
        raise ValueError("k must be between one and the number of visible keys")
    return tuple(sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:k])


def evaluate(weights, values, selected):
    mass = fsum(weights[i] for i in selected)
    return mass, fsum(weights[i] * values[i] for i in selected) / mass


def teacher_consensus(rows):
    """Average of layer probabilities, not softmax of average logits."""
    return tuple(fsum(col) / len(rows) for col in zip(*rows, strict=True))


def maximin(rows, k):
    """Exact tiny diagnostic, not part of CLSA or a scalable router."""
    return max(combinations(range(len(rows[0])), k),
               key=lambda ids: min(fsum(w[i] for i in ids) for w in rows))


def run(k):
    # Three visible positions in one causal prefix. One head per layer.
    rows = tuple(softmax(tuple(log(p) for p in w))
                 for w in ((0.65, 0.25, 0.10), (0.10, 0.30, 0.60)))
    values = (0.0, 1.0, 4.0)
    # The oracle teacher target stands in for a trained indexer's ordering.
    consensus = route(teacher_consensus(rows), k)
    worst_layer = maximin(rows, k)
    per_layer = tuple(route(w, k) for w in rows)

    def metrics(routes, selections):
        pairs = [evaluate(w, values, ids) for w, ids in zip(rows, routes, strict=True)]
        return dict(selections=selections,
                    retained_mass=[round(p[0], 6) for p in pairs],
                    sparse_output=[round(p[1], 6) for p in pairs])

    return {
        "visible_keys": len(values),
        "k": k,
        "dense_output": [round(evaluate(w, values, range(len(values)))[1], 6)
                         for w in rows],
        "consensus": metrics([consensus] * len(rows), consensus),
        "maximin_diagnostic": metrics([worst_layer] * len(rows), worst_layer),
        "per_layer_oracle": metrics(per_layer, per_layer),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-sample", action="store_true")
    rendered = json.dumps([run(k) for k in (1, 2, 3)], indent=2) + "\n"
    if parser.parse_args().check_sample:
        if rendered != Path(__file__).with_name("sample_output.txt").read_text():
            raise SystemExit("Output differs from saved sample")
    print(rendered, end="")
