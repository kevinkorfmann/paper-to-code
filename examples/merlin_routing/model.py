"""MERLIN routing, Hwang et al. (2026), v1 Eqs. 7--9 and aggregation Eq. 4.

Only routing moments and supplied readouts: no trained backbone or KV state.
The covariance is population (divide by n), not unbiased sample covariance.
"""

from dataclasses import dataclass
import math


def _finite(value, label):
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite numbers, not bools")
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be finite numbers") from exc
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite numbers")
    return value


def _vector(values, label):
    try:
        result = tuple(_finite(v, label) for v in values)
    except TypeError as exc:
        raise ValueError(f"{label} must be a nonempty numeric vector") from exc
    if not result:
        raise ValueError(f"{label} must be nonempty")
    return result


def _sum(terms, label):
    try:
        return _finite(math.fsum(terms), label)
    except OverflowError as exc:
        raise ValueError(f"{label} exceeds supported floating-point range") from exc


@dataclass(frozen=True, init=False)
class SegmentMoments:
    """Immutable compressed summary, constructed from a nonempty key prefix.

    A local coordinate shift avoids subtracting two huge raw second moments.
    Keys are not retained. Prefix length counts visible tokens in this segment;
    omit entirely invisible segments instead of constructing an empty summary.
    """

    count: int
    mean: tuple[float, ...]
    covariance: tuple[tuple[float, ...], ...]

    def __init__(self, keys, *, prefix_length=None):
        try:
            rows = tuple(_vector(row, "keys") for row in keys)
        except TypeError as exc:
            raise ValueError("keys must be a nonempty matrix") from exc
        if not rows or any(len(row) != len(rows[0]) for row in rows):
            raise ValueError("keys must be nonempty with equal-width rows")
        n = len(rows) if prefix_length is None else prefix_length
        if type(n) is not int or not 1 <= n <= len(rows):
            raise ValueError("prefix_length must be an integer in 1..number of keys")
        rows = rows[:n]
        d = len(rows[0])
        origin = rows[0]
        offsets = tuple(tuple(_finite(x-o, "key offset") for x, o in zip(row, origin, strict=True))
                        for row in rows)
        center = tuple(_sum((row[a]/n for row in offsets), "center") for a in range(d))
        mean = tuple(_finite(o+c, "mean") for o, c in zip(origin, center, strict=True))
        centered = tuple(tuple(_finite(x-c, "centered key") for x, c in zip(row, center, strict=True))
                         for row in offsets)
        covariance = tuple(tuple(_sum((row[a]*row[b]/n for row in centered), "covariance")
                                 for b in range(d)) for a in range(d))
        object.__setattr__(self, "count", n)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "covariance", covariance)

    def scores(self, query):
        """Return E[q dot k] and Var[q dot k] from the summary, not the keys."""
        q = _vector(query, "query")
        if len(q) != len(self.mean):
            raise ValueError("query dimension must match keys")
        mean = _sum((x*y for x, y in zip(q, self.mean, strict=True)), "mean score")
        terms = [q[a]*self.covariance[a][b]*q[b]
                 for a in range(len(q)) for b in range(len(q))]
        variance = _sum(terms, "variance score")
        magnitude = _sum((abs(x) for x in terms), "variance magnitude")
        # A PSD covariance can acquire a tiny negative quadratic form by roundoff.
        if variance < -64*math.ulp(magnitude):
            raise ValueError("variance is negative beyond floating-point roundoff")
        return mean, max(0.0, variance)


def _softmax(scores, temperature):
    # Temperatures here multiply scores (not their reciprocals). Center BEFORE
    # multiplying so equal enormous logits do not overflow unnecessarily.
    if temperature == 0:
        return (1.0/len(scores),)*len(scores)
    peak = max(scores)
    weights = []
    for score in scores:
        gap = score-peak
        # Opposite-sign near-max floats may overflow on subtraction. A small
        # temperature can bring that difference back into range; scale first.
        log_weight = (score*temperature-peak*temperature
                      if not math.isfinite(gap) and temperature < 1
                      else gap*temperature)
        weights.append(math.exp(log_weight))
    total = math.fsum(weights)
    return tuple(w/total for w in weights)


@dataclass(frozen=True)
class Routing:
    mean_scores: tuple[float, ...]
    variance_scores: tuple[float, ...]
    mean_weights: tuple[float, ...]
    variance_weights: tuple[float, ...]
    weights: tuple[float, ...]
    output: tuple[float, ...]


def route(segments, query, readouts, *, gamma=0.5, tau_mean=1.0, tau_variance=1.0):
    """Mix two normalized branches, then aggregate caller-supplied readouts.

    gamma is the MEAN branch gate. Scalar gamma and nonnegative temperatures
    are explicit constants here, not learned projections. Only visible segment
    summaries and already-causal readouts may be supplied. No token-count factor
    is applied: repeating tokens inside a segment does not change its moments.
    """
    gamma = _finite(gamma, "gamma")
    tm, tv = _finite(tau_mean, "tau_mean"), _finite(tau_variance, "tau_variance")
    if not 0 <= gamma <= 1 or tm < 0 or tv < 0:
        raise ValueError("gamma must be in [0,1]; temperatures must be nonnegative")
    q = _vector(query, "query")
    try:
        segments = tuple(segments)
        rows = tuple(_vector(row, "readouts") for row in readouts)
    except TypeError as exc:
        raise ValueError("provide segment summaries and readout rows") from exc
    if not segments or len(segments) != len(rows):
        raise ValueError("provide one readout for each nonempty segment summary")
    if any(not isinstance(s, SegmentMoments) for s in segments):
        raise ValueError("segments must contain SegmentMoments objects")
    if any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("readout rows must have equal width")
    scores = tuple(s.scores(q) for s in segments)
    means, variances = tuple(s[0] for s in scores), tuple(s[1] for s in scores)
    pm, pv = _softmax(means, tm), _softmax(variances, tv)
    weights = tuple(gamma*a+(1-gamma)*b for a, b in zip(pm, pv, strict=True))
    output = []
    for column in zip(*rows, strict=True):
        scale = max(abs(x) for x in column)
        if scale == 0:
            value = 0.0
        else:
            normalized = tuple(x/scale for x in column)
            value = _sum((w*x for w, x in zip(weights, normalized, strict=True)), "output")
            # Enforce the convex hull against last-bit sum-to-one roundoff;
            # in particular, a constant max-float readout must not overflow.
            value = min(max(normalized), max(min(normalized), value))*scale
        output.append(_finite(value, "output"))
    return Routing(means, variances, pm, pv, weights, tuple(output))
