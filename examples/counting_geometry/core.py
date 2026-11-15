"""Original scalar counting-head reference; no trained model or dependencies."""
import math


def count_coordinate(n, gamma=2.0, distractors=0):
    """Equal unit content weights, boundary weight gamma; only target values are 1."""
    for x in (n, distractors):
        if isinstance(x, bool) or not isinstance(x, int) or x < 0:
            raise ValueError('counts must be nonnegative integers')
    if not math.isfinite(gamma) or gamma <= 0:
        raise ValueError('gamma must be positive and finite')
    return n / (n + distractors + gamma)


def prototypes(limit, gamma=2.0):
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError('limit must be a positive integer')
    return [count_coordinate(c, gamma) for c in range(limit + 1)]


def classify(x, centers):
    """Nearest-center classifier; ties resolved toward the lower class index.

    Maximizing 2*t*x-t*t is algebraically equivalent. Squared distances avoid
    cancellation of nearly equal large logits, but cannot undo quantized inputs.
    """
    if not centers or not math.isfinite(x) or any(not math.isfinite(t) for t in centers):
        raise ValueError('finite input and nonempty finite centers required')
    distances = [(x-t)**2 for t in centers]
    winner = min(range(len(centers)), key=distances.__getitem__)
    ordered = sorted(distances)
    return winner, ordered[1]-ordered[0] if len(ordered)>1 else math.inf


def quantize(x, bits):
    """Uniform fixed-point grid on [0,1], ties-to-even; not IEEE float emulation."""
    if isinstance(bits, bool) or not isinstance(bits, int) or not 1 <= bits <= 52:
        raise ValueError('bits must be between 1 and 52')
    if not math.isfinite(x) or not 0 <= x <= 1:
        raise ValueError('coordinate must lie in [0,1]')
    return round(x * (2**bits)) / (2**bits)


def regression(n, gamma):
    """One explicit small-coordinate regressor, not a minimax solution."""
    return gamma * count_coordinate(n, gamma)
