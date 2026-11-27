"""Scalar demonstration of the decay combiner in Universal Attention."""
import argparse
import math
from pathlib import Path


def step_log_survival(gates, *, geometric=True):
    """Input gates are decay fractions, not survival fractions."""
    if not gates or any(not 0 <= g <= 1 for g in gates):
        raise ValueError("expected nonempty finite decay fractions in [0, 1]")
    if geometric:
        if 0 in gates:
            return 0.0
        log_decay = math.fsum(math.log(g) for g in gates) / len(gates)
        survival = -math.expm1(log_decay)
        return math.log(survival) if survival else -math.inf
    decay = math.fsum(gates) / len(gates)
    return math.log1p(-decay) if decay < 1 else -math.inf


def causal_bias(step, t, j, *, geometric=True):
    if j > t:
        return -math.inf
    return sum(step_log_survival(step(i, j), geometric=geometric)
               for i in range(j + 1, t + 1))


def normalized_weights(logits):
    peak = max(logits)
    if not math.isfinite(peak):
        raise ValueError("need a finite unmasked logit")
    masses = [math.exp(x - peak) for x in logits]
    total = sum(masses)
    return [x / total for x in masses]


def fixture_step(_query, key):
    return (0.5, 0.0 if key == 0 else 0.5, 0.5)


def report():
    lines = ["All content logits equal; readout equals key 0's weight."]
    for label, geometric in (("geometric", True), ("arithmetic control", False)):
        biases = [causal_bias(fixture_step, 3, j, geometric=geometric)
                  for j in range(4)]
        w = normalized_weights(biases)
        lines.append(f"{label}: key 0 survival={math.exp(biases[0]):.6f},"
                     f" readout={w[0]:.6f}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--check-sample", action="store_true")
    args = p.parse_args()
    output = report()
    if args.check_sample and Path(__file__).with_name("sample_output.txt").read_text() != output:
        raise SystemExit("sample output mismatch")
    print(output, end="")
