"""Joint ordered-pair attention from Rahman & Qian (2026), Eqs. 2--5.

Only supplied projections and nonoverlapping local blocks are implemented.
This is bidirectional within each block, not an autoregressive attention layer.
"""

from dataclasses import dataclass
import math


def _vector(values, name):
    try:
        values = tuple(values)
        if any(isinstance(x, bool) for x in values):
            raise ValueError(f"{name} must contain finite real numbers, not booleans")
        result = tuple(float(x) for x in values)
    except (TypeError, OverflowError, ValueError) as exc:
        raise ValueError(f"{name} must contain finite real numbers") from exc
    if not result or not all(math.isfinite(x) for x in result):
        raise ValueError(f"{name} must be nonempty and finite")
    return result


def _matrix(rows, name):
    try:
        result = tuple(_vector(row, name) for row in rows)
    except TypeError as exc:
        raise ValueError(f"{name} must be a nonempty rectangular matrix") from exc
    if not result or any(len(row) != len(result[0]) for row in result):
        raise ValueError(f"{name} must be a nonempty rectangular matrix")
    return result


def _finite_sum(values):
    try:
        result = math.fsum(values)
    except (OverflowError, ValueError) as exc:
        raise ValueError("Intermediate arithmetic overflow; use smaller inputs") from exc
    if not math.isfinite(result):
        raise ValueError("Intermediate arithmetic overflow; use smaller inputs")
    return result


@dataclass(frozen=True)
class Readout:
    """Joint probabilities (first key axis, second key axis) and mixed values."""

    weights: tuple[tuple[float, ...], ...]
    output: tuple[float, ...]


def triplet_read(query, keys1, keys2, values1, values2):
    """Compute one query's joint softmax over every ordered pair, including j=k.

    Each branch has the same number of tokens. Keys share the query dimension.
    Value branches share a value dimension, which may differ from key dimension.
    Inputs must have finite entries and finite intermediate products/sums.
    """
    q = _vector(query, "query")
    k1, k2 = _matrix(keys1, "keys1"), _matrix(keys2, "keys2")
    v1, v2 = _matrix(values1, "values1"), _matrix(values2, "values2")
    n, d = len(k1), len(q)
    if not (len(k2) == len(v1) == len(v2) == n):
        raise ValueError("Both key/value branches must have the same token count")
    if len(k1[0]) != d or len(k2[0]) != d:
        raise ValueError("Query and both key dimensions must match")
    if len(v1[0]) != len(v2[0]):
        raise ValueError("Both value dimensions must match")
    scores = tuple(tuple(_finite_sum(q[r] * a[r] * b[r] for r in range(d))
                         / math.sqrt(d) for b in k2) for a in k1)
    peak = max(max(row) for row in scores)
    # Subtraction can produce -inf for opposite finite extremes. exp(-inf)=0
    # is the correct underflow limit; at least one maximum has mass exactly 1.
    masses = tuple(tuple(math.exp(score - peak) for score in row) for row in scores)
    total = math.fsum(mass for row in masses for mass in row)
    weights = tuple(tuple(mass / total for mass in row) for row in masses)
    output = tuple(_finite_sum(weights[j][k] * _finite_sum((v1[j][r] * v2[k][r],))
                               for j in range(n) for k in range(n))
                   for r in range(len(v1[0])))
    return Readout(weights, output)


def block_attention(queries, keys1, keys2, values1, values2, block_size):
    """Return local readouts for disjoint contiguous blocks, including a short tail.

    Every query and both key indices belong to the same block. No overlapping
    blocks, overlap averaging, cross-block module, masks, or trained projections.
    Each returned weight matrix uses local indices within that query's block.
    """
    arrays = tuple(_matrix(x, name) for x, name in zip(
        (queries, keys1, keys2, values1, values2),
        ("queries", "keys1", "keys2", "values1", "values2")))
    n = len(arrays[0])
    if any(len(x) != n for x in arrays):
        raise ValueError("All projections must have the same token count")
    if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size < 1:
        raise ValueError("block_size must be a positive integer")
    out = []
    for start in range(0, n, block_size):
        q, k1, k2, v1, v2 = (x[start:start + block_size] for x in arrays)
        out.extend(triplet_read(row, k1, k2, v1, v2) for row in q)
    return tuple(out)
