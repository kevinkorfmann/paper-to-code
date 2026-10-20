"""Single-head core of Ye et al., Differential Transformer, equations (1)–(2).

Inputs are already-projected Q1/Q2/K1/K2 (N x d) and V (N x 2d).
No learned projections, normalization, multi-head block, or training is included.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from math import exp, fsum, isfinite, sqrt

Matrix = tuple[tuple[float, ...], ...]


def _finite(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must contain real int/float values, not bools")
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} exceeds float range") from error
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _matrix(values: Sequence[Sequence[float]], name: str) -> Matrix:
    rows = tuple(tuple(_finite(x, name) for x in row) for row in values)
    if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError(f"{name} must be a nonempty rectangular matrix")
    return rows


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    try:
        result = fsum(x * y for x, y in zip(a, b, strict=True))
    except (OverflowError, ValueError) as error:
        raise ValueError("dot product exceeds finite float arithmetic") from error
    if not isfinite(result):
        raise ValueError("dot product exceeds finite float arithmetic")
    return result


def softmax(logits: Sequence[float]) -> tuple[float, ...]:
    """Stable row softmax; -inf masks a key, and all-masked rows are invalid."""
    if not logits:
        raise ValueError("softmax requires at least one logit")
    values = tuple(
        float("-inf") if x == float("-inf") else _finite(x, "logits") for x in logits
    )
    largest = max(values)
    if largest == float("-inf"):
        raise ValueError("softmax requires at least one unmasked key")
    weights = tuple(exp(x - largest) for x in values)
    total = fsum(weights)
    return tuple(x / total for x in weights)


def lambda_initial(layer: int) -> float:
    """Paper's default schedule, using a ONE-based layer index."""
    if type(layer) is not int:
        raise TypeError("layer must be an integer")
    if layer < 1:
        raise ValueError("layer must be >= 1")
    return 0.8 - 0.6 * exp(-0.3 * (layer - 1))


def lambda_from_vectors(
    q1: Sequence[float], k1: Sequence[float],
    q2: Sequence[float], k2: Sequence[float], layer: int,
) -> float:
    """Evaluate equation (2); the resulting scalar is NOT clipped to [0,1]."""
    vectors = tuple(tuple(_finite(x, "lambda vectors") for x in v) for v in (q1, k1, q2, k2))
    if not vectors[0] or any(len(v) != len(vectors[0]) for v in vectors):
        raise ValueError("lambda vectors must be nonempty and have equal lengths")
    initial = lambda_initial(layer)
    try:
        result = exp(_dot(vectors[0], vectors[1])) - exp(_dot(vectors[2], vectors[3])) + initial
    except OverflowError as error:
        raise ValueError("lambda exponent exceeds finite float arithmetic") from error
    return _finite(result, "lambda")


@dataclass(frozen=True)
class AttentionResult:
    map1: Matrix
    map2: Matrix
    weights: Matrix
    output: Matrix


def differential_attention(
    q1: Sequence[Sequence[float]],
    q2: Sequence[Sequence[float]],
    k1: Sequence[Sequence[float]],
    k2: Sequence[Sequence[float]],
    values: Sequence[Sequence[float]],
    lambda_value: float,
    *,
    causal: bool = False,
) -> AttentionResult:
    """Compute (softmax(Q1 K1^T / sqrt(d)) - lambda softmax(Q2 K2^T / sqrt(d))) V.

    Causal mode permits key j iff j <= query i, in both branches BEFORE softmax.
    Weights are signed coefficients, not a probability distribution.
    """
    if type(causal) is not bool:
        raise TypeError("causal must be a bool")
    coefficient = _finite(lambda_value, "lambda")
    queries1, queries2, keys1, keys2, vals = (
        _matrix(matrix, name)
        for matrix, name in ((q1, "q1"), (q2, "q2"), (k1, "k1"), (k2, "k2"), (values, "values"))
    )
    length, width = len(queries1), len(queries1[0])
    if any(len(m) != length or len(m[0]) != width for m in (queries2, keys1, keys2)):
        raise ValueError("Q1, Q2, K1, K2 must all have shape N x d")
    if len(vals) != length or len(vals[0]) != 2 * width:
        raise ValueError("values must have shape N x 2d, as in the paper's single head")

    def attention_map(queries: Matrix, keys: Matrix) -> Matrix:
        return tuple(
            softmax(tuple(
                float("-inf") if causal and j > i else _dot(query, key) / sqrt(width)
                for j, key in enumerate(keys)
            ))
            for i, query in enumerate(queries)
        )

    first = attention_map(queries1, keys1)
    second = attention_map(queries2, keys2)
    weights = tuple(
        tuple(_finite(a - coefficient * b, "differential weight") for a, b in zip(row1, row2))
        for row1, row2 in zip(first, second)
    )
    output = tuple(
        tuple(_dot(row, column) for column in zip(*vals)) for row in weights
    )
    return AttentionResult(first, second, weights, output)
