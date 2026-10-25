"""One attention row: Wang (2026), v2 equations 2–4 and Lemma 3.1.

Scores are already computed/scaled. A finite phantom logit has a zero value.
Value gates are per read; the optional routing-only result renormalizes their
coefficients. No projection weights are trained here.
"""

from dataclasses import dataclass
from math import fsum

from examples.differential_attention.model import _dot, _finite, _matrix, softmax
from examples.gated_attention.model import sigmoid


def projection_gates(values, weight, bias=0.0):
    """Paper Eq. 6: sigmoid(w dot value + bias), one scalar per value read."""
    rows = _matrix(values, "values")
    w = tuple(_finite(x, "projection weight") for x in weight)
    b = _finite(bias, "projection bias")
    if len(w) != len(rows[0]):
        raise ValueError("projection weight must match the value width")
    return tuple(sigmoid(_finite(_dot(w, row)+b, "gate logit")) for row in rows)


@dataclass(frozen=True)
class Read:
    routing_weights: tuple[float, ...]
    phantom_mass: float
    gates: tuple[float, ...]
    retained_weights: tuple[float, ...]
    retained_mass: float
    output: tuple[float, ...]
    renormalized_weights: tuple[float, ...] | None
    renormalized_output: tuple[float, ...] | None


def attend(logits, values, *, sink_logit=None, gates=None, mask=None):
    """Compute sum_j p_j*g_j*v_j for one query and one head.

    None disables the phantom. mask[j]=True permits a real key; the phantom
    always remains available. Gates default to 1 and may include the limiting
    values 0 and 1. All-zero retained mass makes conditional routing undefined,
    represented by None. All real keys masked requires a phantom.
    """
    scores = tuple(_finite(s, "logits") for s in logits)
    rows = _matrix(values, "values")
    if not scores or len(scores) != len(rows):
        raise ValueError("one finite score is required per value row")
    allowed = (True,)*len(scores) if mask is None else tuple(mask)
    if len(allowed) != len(scores) or any(type(x) is not bool for x in allowed):
        raise ValueError("mask must have one bool per real key")
    gate = (1.0,)*len(scores) if gates is None else tuple(_finite(g, "gates") for g in gates)
    if len(gate) != len(scores) or any(not 0 <= g <= 1 for g in gate):
        raise ValueError("one gate in [0,1] is required per real key")
    available_scores = tuple(s if ok else float("-inf") for s, ok in zip(scores, allowed))
    if sink_logit is None:
        probabilities, phantom = softmax(available_scores), 0.0
    else:
        joint = softmax((*available_scores, _finite(sink_logit, "sink_logit")))
        probabilities, phantom = joint[:-1], joint[-1]
    retained = tuple(p*g for p, g in zip(probabilities, gate))
    mass = fsum(retained)
    output = tuple(_dot(retained, column) for column in zip(*rows))
    # Preserve attenuation above; this is a separate counterfactual read.
    normalized = tuple(p/mass for p in retained) if mass > 0 else None
    normalized_output = (tuple(_dot(normalized, column) for column in zip(*rows))
                         if normalized is not None else None)
    return Read(probabilities, phantom, gate, retained, mass, output, normalized, normalized_output)
