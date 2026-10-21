"""Original scalar headwise G1 gate from Qiu et al. (2025), equation 5.

One self-attention head, already projected Q/K/V, and supplied gate projection.
The gate is query-dependent: sigmoid(X W_gate), before any output projection.
"""

from dataclasses import dataclass
from math import exp, sqrt

# Reuse only the repository's original numerical primitives, not its differential map.
from examples.differential_attention.model import Matrix, _dot, _finite, _matrix, softmax


def sigmoid(value):
    """Avoid overflow for large finite logits of either sign."""
    value = _finite(value, "gate logit")
    if value >= 0:
        return 1 / (1 + exp(-value))
    exponential = exp(value)
    return exponential / (1 + exponential)


@dataclass(frozen=True)
class GatedResult:
    attention: Matrix
    gates: tuple[float, ...]
    ordinary_output: Matrix
    gated_output: Matrix


def gated_attention(queries, keys, values, hidden, gate_weights, *, causal=False):
    """Return A, g, AV and diag(g)AV, where g_i=sigmoid(x_i dot w).

    Q/K have shape Nxd, V has shape Nxv, X has shape Nxh, w has length h.
    A scalar per query is broadcast over all v value channels in this head.
    """
    if type(causal) is not bool:
        raise TypeError("causal must be bool")
    q, k, v, x = (_matrix(m, name) for m, name in (
        (queries, "queries"), (keys, "keys"), (values, "values"), (hidden, "hidden")
    ))
    w = tuple(_finite(a, "gate_weights") for a in gate_weights)
    n, d = len(q), len(q[0])
    if any(len(m) != n for m in (k, v, x)) or len(k[0]) != d:
        raise ValueError("Q, K, V, X need equal row counts; Q/K need equal widths")
    if len(w) != len(x[0]):
        raise ValueError("gate_weights must match hidden width")
    attention = tuple(softmax(tuple(
        float("-inf") if causal and j > i else _dot(qi, kj) / sqrt(d)
        for j, kj in enumerate(k)
    )) for i, qi in enumerate(q))
    ordinary = tuple(tuple(_dot(row, col) for col in zip(*v)) for row in attention)
    gates = tuple(sigmoid(_dot(row, w)) for row in x)
    gated = tuple(tuple(g * a for a in row) for g, row in zip(gates, ordinary))
    return GatedResult(attention, gates, ordinary, gated)
