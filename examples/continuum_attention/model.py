"""Causal integration over piecewise-linear memory, Khadangi (2026), v2 §2.

The caller supplies already-scaled logits for one query. This is a forward
reference, not the paper's backward pass, trained decoder, or GPU kernel.
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


@dataclass(frozen=True)
class Segment:
    left_coefficient: float
    right_coefficient: float
    mass: float
    mean_coordinate: float


def segment_integrals(left_logit, right_logit, *, shift=None):
    """Integrate the two linear basis functions times the exponential score.

    Returned coefficients include exp(-shift); shift defaults to the larger
    endpoint. The mean coordinate is local u in [0,1] and does not depend on it.
    """
    a, b = _finite(left_logit, "logits"), _finite(right_logit, "logits")
    peak = max(a, b)
    shift = peak if shift is None else _finite(shift, "shift")
    if shift < peak:
        raise ValueError("shift must be at least the larger endpoint logit")
    gap = abs(b-a)
    if not math.isfinite(gap):
        raise ValueError("adjacent logit differences must fit in a finite float")
    # Reverse the segment if needed so the local exponent always decreases:
    # A = integral exp(-gap*u) du; B = integral u*exp(-gap*u) du.
    if gap == 0:
        mass_at_peak, low_coefficient = 1.0, 0.5
    else:
        mass_at_peak = -math.expm1(-gap)/gap
        if gap <= 0.5:
            term, series = 1.0, [0.5]
            for k in range(1, 25):
                term *= -gap/k
                series.append(term/(k+2))
            low_coefficient = math.fsum(series)
        else:
            low_coefficient = (mass_at_peak-math.exp(-gap))/gap
    high_coefficient = mass_at_peak-low_coefficient
    left, right = ((high_coefficient, low_coefficient) if a >= b
                   else (low_coefficient, high_coefficient))
    scale = math.exp(peak-shift)
    return Segment(scale*left, scale*right, scale*mass_at_peak,
                   right/mass_at_peak)


@dataclass(frozen=True)
class Read:
    output: tuple[float, ...]
    token_weights: tuple[float, ...]
    segment_probability_masses: tuple[float, ...]
    segment_mean_coordinates: tuple[float, ...]
    boundary_probability_mass: float
    shifted_normalizer: float
    logit_shift: float
    prefix_length: int


def attend(logits, values, *, prefix_length=None, boundary_caps=True):
    """Integrate the visible prefix, including constant half-token end caps.

    Positions are unit spaced. There is no arbitrary mask with holes: changing
    adjacency changes this operator. Values can have any common nonzero width.
    boundary_caps=False is an explicit diagnostic ablation, not the paper's CA.
    """
    try:
        scores = tuple(_finite(s, "logits") for s in logits)
        rows = tuple(tuple(_finite(v, "values") for v in row) for row in values)
    except TypeError as exc:
        raise ValueError("provide iterable logits and value rows") from exc
    if not scores or len(scores) != len(rows) or not rows[0]:
        raise ValueError("provide one nonempty value row per logit")
    if any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("value rows must have equal width")
    n = len(scores)
    count = n if prefix_length is None else prefix_length
    if type(count) is not int or not 1 <= count <= n:
        raise ValueError("prefix_length must be an integer in 1..number of tokens")
    if type(boundary_caps) is not bool:
        raise ValueError("boundary_caps must be a bool")
    if count == 1 and not boundary_caps:
        raise ValueError("one token without boundary caps has zero integration measure")
    shift = max(scores[:count])
    coefficients = [0.0]*count
    cap_mass = 0.0
    if boundary_caps:
        first, last = 0.5*math.exp(scores[0]-shift), 0.5*math.exp(scores[count-1]-shift)
        coefficients[0] += first
        coefficients[-1] += last
        cap_mass = first+last
    segments = []
    for j in range(1, count):
        segment = segment_integrals(scores[j-1], scores[j], shift=shift)
        coefficients[j-1] += segment.left_coefficient
        coefficients[j] += segment.right_coefficient
        segments.append(segment)
    normalizer = math.fsum(coefficients)
    if normalizer <= 0 or not math.isfinite(normalizer):
        raise ValueError("normalizer is outside the supported floating-point range")
    weights = tuple(c/normalizer for c in coefficients)
    output = []
    for column in zip(*rows[:count], strict=True):
        scale = max(abs(x) for x in column)
        value = 0.0 if scale == 0 else math.fsum(w*(x/scale) for w, x in zip(weights, column, strict=True))*scale
        if not math.isfinite(value):
            raise ValueError("output is outside finite floating-point range")
        output.append(value)
    return Read(tuple(output), weights+(0.0,)*(n-count),
                tuple(s.mass/normalizer for s in segments),
                tuple(s.mean_coordinate for s in segments), cap_mass/normalizer,
                normalizer, shift, count)
