"""Original scalar reference for shared attention normalization, not GPU code."""
import math
from dataclasses import dataclass


def finite(x):
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
        raise ValueError('expected a finite real number')
    return float(x)


def action(query_block, key_block, w8, w4):
    """TriAttention equation (3), with protected first and diagonal blocks."""
    if any(type(x) is not int for x in (query_block, key_block, w8, w4)):
        raise ValueError('block indices and boundaries must be integers')
    if min(query_block, key_block) < 0 or not 1 <= w8 <= w4:
        raise ValueError('invalid indices or boundaries')
    if key_block > query_block:
        return 'masked'
    if key_block == 0 or key_block == query_block:
        return '8-bit'
    distance = query_block - key_block
    return '8-bit' if distance < w8 else '4-bit' if distance < w4 else 'skip'


@dataclass(frozen=True)
class State:
    maximum: float
    mass: float
    numerator: float


def partial(logits, values):
    """Stable partial state for one row and a nonempty key subset."""
    logits, values = tuple(map(finite, logits)), tuple(map(finite, values))
    if not logits or len(logits) != len(values):
        raise ValueError('nonempty equal lengths required')
    maximum = max(logits)
    weights = [math.exp(x - maximum) for x in logits]
    numerator = math.fsum(w * v for w, v in zip(weights, values))
    if not math.isfinite(numerator):
        raise ValueError('weighted sum overflow')
    return State(maximum, math.fsum(weights), numerator)


def merge(states):
    """Merge disjoint partials in common score/value units (paper equation 2)."""
    states = tuple(states)
    if not states:
        raise ValueError('at least one state required')
    for s in states:
        if not isinstance(s, State):
            raise ValueError('State required')
        for x in (s.maximum, s.mass, s.numerator):
            finite(x)
        if s.mass <= 0:
            raise ValueError('positive mass required')
    maximum = max(s.maximum for s in states)
    factors = [math.exp(s.maximum - maximum) for s in states]
    mass = math.fsum(f * s.mass for f, s in zip(factors, states))
    numerator = math.fsum(f * s.numerator for f, s in zip(factors, states))
    if not math.isfinite(mass) or not math.isfinite(numerator):
        raise ValueError('merged state overflow')
    return State(maximum, mass, numerator)


def output(state):
    checked = merge([state])
    value = checked.numerator / checked.mass
    if not math.isfinite(value):
        raise ValueError('output overflow')
    return value
