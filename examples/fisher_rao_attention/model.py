"""Exact constant/affine-score flow, not a trained or Hamiltonian model.

The state is logits, not rounded probabilities. See README.md for the source,
derivation and floating-point limits. No third-party implementation is copied.
"""
import math


def _scalar(x, name, positive=False, nonnegative=False):
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    try:
        x = float(x)
    except OverflowError:
        raise ValueError(f"{name} must be finite") from None
    if not math.isfinite(x) or (positive and x <= 0) or (nonnegative and x < 0):
        raise ValueError(f"invalid {name}")
    return x


def _vector(x, name):
    if not isinstance(x, (list, tuple)) or len(x) < 2:
        raise ValueError(f"{name} needs at least two entries")
    return [_scalar(v, name) for v in x]


def _finite(x):
    if not math.isfinite(x):
        raise ValueError("arithmetic range exceeded; rescale inputs")
    return x


def _center(x):
    m = max(x)
    return [_finite(v-m) for v in x]


def logits_from_probabilities(p):
    """Initialize from a strictly positive probability row (sum tolerance 1e-12)."""
    p = _vector(p, "probabilities")
    if any(v <= 0 or v > 1 for v in p) or not math.isclose(
            math.fsum(p), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("initial probabilities must be positive and sum to one")
    return _center([math.log(v) for v in p])


def log_probabilities(logits):
    """Stable log-softmax; keeps finite log masses even when exp underflows."""
    u = _center(_vector(logits, "logits"))
    log_z = math.log(math.fsum(math.exp(x) for x in u))
    return [_finite(x-log_z) for x in u]


def probabilities(logits):
    """Binary64 probabilities; extreme tails can round to zero."""
    return [math.exp(x) for x in log_probabilities(logits)]


def _ramp_fraction(x):
    # 1 - (1-exp(-x))/x. Series avoids cancellation for small x.
    if x < 1e-4:
        return x*(0.5+x*(-1/6+x*(1/24+x*(-1/120+x/720))))
    return 1.0 + math.expm1(-x)/x


def advance(logits, scores, dt, *, temperature=1.0, mobility=1.0, slope=None):
    """Exact (up to binary64) step for scores(t)=scores+slope*t on [0, dt].

    u' = mobility * (scores(t) - temperature*u).
    Returns a new max-centered logit row. None slope means constant scores.
    """
    u = _center(_vector(logits, "logits"))
    s = _center(_vector(scores, "scores"))
    d = [0.0]*len(u) if slope is None else _center(_vector(slope, "slope"))
    if len(u) != len(s) or len(u) != len(d):
        raise ValueError("logits, scores and slope must have equal lengths")
    dt = _scalar(dt, "dt", nonnegative=True)
    tau = _scalar(temperature, "temperature", positive=True)
    gamma = _scalar(mobility, "mobility", positive=True)
    kappa = _finite(gamma*tau)
    if kappa == 0:
        raise ValueError("relaxation rate underflow; rescale parameters")
    x = _finite(kappa*dt)
    if dt == 0:
        return u
    if x == 0:
        raise ValueError("time-rate product underflow; rescale parameters")
    r = math.exp(-x)
    w = -math.expm1(-x)
    constant = _finite(w/tau)
    ramp = _finite((dt/tau)*_ramp_fraction(x))
    out = [_finite(r*a+constant*b+ramp*c) for a,b,c in zip(u,s,d)]
    return _center(out)


def diagnostics(logits, scores, *, temperature=1.0, mobility=1.0, slope=None):
    """Energy, KL, row velocity and driven energy balance for supplied scores."""
    logs = log_probabilities(logits)
    p = [math.exp(x) for x in logs]
    s = _vector(scores, "scores")
    d = [0.0]*len(p) if slope is None else _vector(slope, "slope")
    if len(p) != len(s) or len(p) != len(d):
        raise ValueError("logits, scores and slope must have equal lengths")
    tau = _scalar(temperature, "temperature", positive=True)
    gamma = _scalar(mobility, "mobility", positive=True)
    target = log_probabilities([_finite(x/tau) for x in s])
    # The omitted +tau in the free-energy gradient is a common constant.
    f = [_finite(tau*l-x) for l,x in zip(logs,s)]
    try:
        mean = _finite(math.fsum(a*b for a,b in zip(p,f)))
        centered = [_finite(x-mean) for x in f]
        variance = _finite(math.fsum(a*b*b for a,b in zip(p,centered)))
        dissipation = _finite(gamma*variance)
        work = _finite(-math.fsum(a*b for a,b in zip(p,d)))
        kl = _finite(math.fsum(a*(b-c) for a,b,c in zip(p,logs,target)))
        velocity = [_finite(-gamma*a*b) for a,b in zip(p,centered)]
        gap = _finite(tau*kl)
        derivative = _finite(work-dissipation)
    except (OverflowError, ValueError):
        raise ValueError("arithmetic range exceeded; rescale inputs") from None
    return {"probabilities": p, "free_energy": mean, "kl_to_target": kl,
            "free_energy_gap": gap, "dissipation": dissipation,
            "score_work": work, "energy_derivative": derivative,
            "probability_derivative": velocity}
