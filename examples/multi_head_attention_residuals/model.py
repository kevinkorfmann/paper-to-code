"""Multi-head depth routing: Luo et al. (2026), v2 Eq. 3 / Appendix E.

One token position at one routing site. Sources are available raw sublayer
outputs, not cumulative residual sums. RMS statistics span the full width;
softmax spans the depth sources separately for each contiguous feature head.
"""

from dataclasses import dataclass
from math import fsum, sqrt

from examples.differential_attention.model import Matrix, _dot, _finite, _matrix, softmax


def rms_keys(sources, gain=None, *, epsilon=1e-6):
    """Full-row RMSNorm with optional per-coordinate gain, used only for keys.

    Scaling before squaring avoids overflow for large finite source values.
    Epsilon is inside the square root and must be strictly positive.
    """
    rows = _matrix(sources, "sources")
    width = len(rows[0])
    epsilon = _finite(epsilon, "epsilon")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    gains = (1.0,)*width if gain is None else tuple(_finite(g, "gain") for g in gain)
    if len(gains) != width:
        raise ValueError("gain must match the full source width")
    root_epsilon = sqrt(epsilon)
    keys = []
    for row in rows:
        scale = max(root_epsilon, *(abs(x) for x in row))
        scaled = tuple(x/scale for x in row)
        denominator = sqrt(fsum(x*x for x in scaled)/width + (root_epsilon/scale)**2)
        keys.append(tuple(_finite(x/denominator*g, "normalized key") for x, g in zip(scaled, gains)))
    return tuple(keys)


@dataclass(frozen=True)
class DepthRead:
    keys: Matrix
    weights: Matrix  # H x number of sources; each row sums to 1.
    output: tuple[float, ...]


def route_depth(sources, query, *, heads=1, gain=None, epsilon=1e-6):
    """Return H depth distributions and their concatenated raw-value mixtures.

    S has shape N x D; q and optional RMS gain have D entries; H divides D.
    The q vector is split, not copied: every head uses its own D/H entries.
    There is no sqrt(head_width) scale in the paper's depth-routing equation.
    The caller must supply only sources already available at this routing site.
    """
    rows = _matrix(sources, "sources")
    width = len(rows[0])
    q = tuple(_finite(x, "query") for x in query)
    if len(q) != width:
        raise ValueError("query must match the source width")
    if type(heads) is not int or heads < 1 or width % heads != 0:
        raise ValueError("heads must be a positive integer divisor of the source width")
    keys = rms_keys(rows, gain, epsilon=epsilon)
    head_width = width//heads
    weights, output = [], []
    for start in range(0, width, head_width):
        end = start+head_width
        probabilities = softmax(tuple(_dot(q[start:end], key[start:end]) for key in keys))
        weights.append(probabilities)
        output.extend(_dot(probabilities, [row[c] for row in rows]) for c in range(start, end))
    return DepthRead(keys, tuple(weights), tuple(output))
