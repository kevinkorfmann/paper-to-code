"""Deterministic phase invariance and norm-sensitivity counterexample."""
import argparse
import json
from pathlib import Path
import platform
from .model import attention, born_readout

SAMPLE = Path(__file__).with_name("sample_output.json")


def build_report():
    query = [[1, 1j]]
    keys = [[1, 0], [0, 2j]]
    values = [[1, 0], [0, 1]]
    weights, mixed = attention(query, keys, values)
    rotated, _ = attention([[1j * x for x in query[0]]],
                           [[-1j * x for x in keys[0]], [-x for x in keys[1]]], values)
    unit_keys, _ = attention(query, [[1, 0], [0, 1j]], values)
    # Identity readout matrix with a single mixed slot: phi = mixed[0].
    probabilities, baseline = born_readout(mixed[0], [[0], [1]])
    slots = [[1, 1], [-1, 1]]
    coherent_phi = [sum(slot[j] for slot in slots) for j in range(2)]
    coherent, _ = born_readout(coherent_phi, [[0], [1]])
    # Incoherent detection: square each slot separately, then sum.
    incoherent_intensity = [sum(abs(slot[j])**2 for slot in slots) for j in range(2)]
    incoherent = [x / sum(incoherent_intensity) for x in incoherent_intensity]
    return {
        "example": "Quartic Born attention and Born readout",
        "python_version": platform.python_version(),
        "randomness": "None; fixed complex vectors.",
        "quartic_weights": weights[0],
        "mixed_real_values": [x.real for x in mixed[0]],
        "independent_query_and_key_phases": rotated[0],
        "unit_length_keys_counterexample": unit_keys[0],
        "normalized_fidelity_squared_weights": [0.5, 0.5],
        "born_readout_probabilities": probabilities,
        "pixel_count_baseline": baseline,
        "coherent_pooling_control": {
            "slots": slots,
            "summed_amplitudes_identity_R": coherent_phi,
            "coherent_probabilities": coherent,
            "incoherent_probabilities": incoherent,
        },
        "explanation": "Doubling only the second key multiplies its score by 16. Per-vector normalization erases this effect.",
        "scope": "Eqs. 2 and 4 kernels only; no training, full transformer, quantum execution, evidence integration or performance claims."
    }


def render(report):
    def rounded(x):
        if isinstance(x, float):
            return round(x, 12)
        if isinstance(x, dict):
            return {k: rounded(v) for k, v in x.items()}
        if isinstance(x, list):
            return [rounded(v) for v in x]
        return x
    return json.dumps(rounded(report), indent=2, allow_nan=False) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-sample", action="store_true")
    args = parser.parse_args()
    text = render(build_report())
    if args.check_sample and text != SAMPLE.read_text():
        parser.error("Output differs from checked sample (including Python version)")
    print(text, end="")


if __name__ == "__main__":
    main()
