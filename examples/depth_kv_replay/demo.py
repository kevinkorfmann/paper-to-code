"""Stride-based KV replay for precomputed (synthetic) depth activations."""
import json
import math


def scalar_attention(q, kv, visible):
    """One head, one query; key/value entries have scalar coordinates."""
    if not math.isfinite(q) or not 1 <= visible <= len(kv):
        raise ValueError("invalid query/prefix")
    rows = kv[:visible]
    scores = [q * k for k, _ in rows]
    if not all(map(math.isfinite, scores)) or not all(math.isfinite(v) for _, v in rows):
        raise ValueError("nonfinite visible cache")
    peak = max(scores)
    weights = [math.exp(s - peak) for s in scores]
    return math.fsum(w * v for w, (_, v) in zip(weights, rows, strict=True)) / math.fsum(weights)


def replay(queries, project_kv, stride, visible):
    """One forward trace. project_kv(iteration, layer) is called only at writers."""
    if type(stride) is not int or stride < 1 or type(visible) is not int or visible < 1:
        raise ValueError("expected positive integer stride/prefix")
    if not queries or not queries[0] or any(len(x) != len(queries[0]) for x in queries):
        raise ValueError("expected a rectangular, nonempty query grid")
    cache = {}
    trace = []
    for iteration, row in enumerate(queries):
        group = iteration // stride
        outputs = []
        for layer, q in enumerate(row):
            slot = (group, layer)
            if iteration % stride == 0:
                cache[slot] = tuple(project_kv(iteration, layer))
            outputs.append(scalar_attention(q, cache[slot], visible))
        trace.append({"iteration": iteration, "group": group,
                      "writer": group * stride, "outputs": outputs})
    return trace, len(cache)


def demo():
    # Reader queries reverse sign. A deliberately huge suffix token must never enter.
    queries = [[2], [-2], [2], [-2], [0]]
    layers = [
        [(-1, 0), (1, 1)],
        [(-1, 0), (1, 10)],
        [(-1, 0), (1, 2)],
        [(-1, 0), (1, 20)],
        [(-1, 0), (1, 4)],
    ]
    result = {}
    for stride in (1, 2, 5):
        calls = []
        def project(r, layer):
            calls.append([r, layer])
            return layers[r] + [(1000, 1000)]
        trace, slots = replay(queries, project, stride, 2)
        for row in trace:
            row["outputs"] = [round(x, 12) for x in row["outputs"]]
        result[str(stride)] = {"trace": trace, "cache_groups": slots,
                               "projection_calls": calls}
    return {"visible_tokens": 2, "strides": result}


if __name__ == "__main__":
    print(json.dumps(demo(), indent=2, sort_keys=True))
