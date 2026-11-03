"""DAMI scalar GDN calibration (Feng et al., arXiv:2610.41994v1, eqs. 2, 8–11).

Original standard-library implementation. This is not a model initializer:
activations are supplied, and no forward pass or training is performed.
"""

from dataclasses import dataclass
import math


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite real number")
    try:
        value = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} is outside floating-point range") from exc
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _sample(values, *, allow_empty=False):
    try:
        values = tuple(values)
    except TypeError as exc:
        raise ValueError("activations must be an iterable") from exc
    if not values and not allow_empty:
        raise ValueError("activations must not be empty")
    return tuple(_number(z, "activation") for z in values)


def softplus(x):
    """Stable log(1 + exp(x)); neither branch exponentiates a positive number."""
    x = _number(x, "x")
    return max(x, 0.0) + math.log1p(math.exp(-abs(x)))


def sigmoid(x):
    x = _number(x, "x")
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def _mean(values):
    return math.fsum(x / len(values) for x in values)


@dataclass(frozen=True)
class Gate:
    bias: float
    a_log: float

    def __post_init__(self):
        object.__setattr__(self, "bias", _number(self.bias, "bias"))
        object.__setattr__(self, "a_log", _number(self.a_log, "a_log"))
        # Keep exp(a_log) representable and nonzero in binary64.
        if not -700.0 <= self.a_log <= 700.0:
            raise ValueError("a_log must lie in [-700, 700]")

    def log_retention(self, activation):
        """Log of the explicit one-step decay, not the full state transition."""
        z = _number(activation, "activation")
        x = _number(z + self.bias, "shifted activation")
        # log(softplus(x)) approaches x in the far negative tail. Combining
        # in log space avoids underflowing softplus before a large multiplier.
        try:
            result = (-math.exp(self.a_log + x) if x < -36.0
                      else -math.exp(self.a_log) * softplus(x))
        except OverflowError as exc:
            raise ValueError("log retention overflow") from exc
        if not math.isfinite(result):
            raise ValueError("log retention overflow")
        return result

    def retention(self, activation):
        """Underflow to zero is allowed for extremely strong decay."""
        return math.exp(self.log_retention(activation))


def calibrate(activations, timescale, sensitivity):
    """Match mean sigmoid and mean log decay on a supplied calibration sample.

    Deliberately bounded educational domain: abs(z) <= 100,
    1e-6 <= sensitivity <= 1-1e-6, and 1e-6 <= timescale <= 1e12.
    Bisection brackets follow from monotonicity, rather than guessed bounds.
    """
    z = _sample(activations)
    tau = _number(timescale, "timescale")
    target = _number(sensitivity, "sensitivity")
    if any(abs(x) > 100 for x in z):
        raise ValueError("calibration activations must lie in [-100, 100]")
    if not 1e-6 <= tau <= 1e12:
        raise ValueError("timescale must lie in [1e-6, 1e12]")
    if not 1e-6 <= target <= 1 - 1e-6:
        raise ValueError("sensitivity must lie in [1e-6, 1-1e-6]")
    logit = math.log(target) - math.log1p(-target)
    lo, hi = logit - max(z), logit - min(z)
    for _ in range(200):
        mid = lo + (hi - lo) / 2
        if mid == lo or mid == hi:
            break
        observed = _mean([sigmoid(x + mid) for x in z])
        if observed == target:
            lo = hi = mid
            break
        if observed < target:
            lo = mid
        else:
            hi = mid
    bias = lo + (hi - lo) / 2
    slopes = [sigmoid(x + bias) for x in z]
    if _mean([s * (1.0 - s) for s in slopes]) < 1e-10:
        raise ValueError("calibration root is too flat for reliable binary64 bias")
    if abs(_mean(slopes) - target) > 2e-13:
        raise ValueError("sensitivity calibration did not converge")
    factor = _mean([softplus(x + bias) for x in z])
    return Gate(bias, -math.log(tau) - math.log(factor))


def diagnose(gate, activations):
    """Separate Softplus sensitivity, full log-decay slope, and retention means."""
    z = _sample(activations)
    slopes = [sigmoid(x + gate.bias) for x in z]
    logs = [gate.log_retention(x) for x in z]
    mean_log = _mean(logs)
    return {
        "mean_softplus_sensitivity": _mean(slopes),
        "mean_log_decay_slope_magnitude": math.exp(gate.a_log) * _mean(slopes),
        "mean_log_retention": mean_log,
        "geometric_mean_retention": math.exp(mean_log),
        "arithmetic_mean_retention": _mean([math.exp(x) for x in logs]),
    }


def geometric_timescales(heads, shortest, longest):
    """Return descending timescales, matching increasing rates in paper eq. 8."""
    if isinstance(heads, bool) or not isinstance(heads, int) or heads < 2:
        raise ValueError("heads must be an integer >= 2")
    shortest = _number(shortest, "shortest")
    longest = _number(longest, "longest")
    if not 1e-6 <= shortest <= longest <= 1e12:
        raise ValueError("require 1e-6 <= shortest <= longest <= 1e12")
    if shortest == longest:
        return (shortest,) * heads
    low, high = math.log(shortest), math.log(longest)
    result = [math.exp(high + (low - high) * h / (heads - 1)) for h in range(heads)]
    result[0], result[-1] = longest, shortest
    return tuple(result)


def retention_path(gate, activations):
    """Explicit decay envelope, including the empty-prefix value 1.

    The returned log path remains informative when exponentiation underflows.
    No writes, key projections, delta-rule erasure or readout are included.
    """
    z = _sample(activations, allow_empty=True)
    logs = [0.0]
    total = 0.0
    for x in z:
        total += gate.log_retention(x)
        if not math.isfinite(total):
            raise ValueError("cumulative log retention overflow")
        logs.append(total)
    return {"log_retention": tuple(logs), "retention": tuple(math.exp(x) for x in logs)}
