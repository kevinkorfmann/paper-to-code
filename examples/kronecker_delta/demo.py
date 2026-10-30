"""Deterministic controls: preservation is not always the desired behavior."""

import argparse
import json

from .model import association, delta_update, kron_update, read


def report():
    key = (1.0, 0.0)
    old_value, new_value = (1.0, 0.0), (0.0, 1.0)
    zero = ((0.0, 0.0), (0.0, 0.0))
    initial = kron_update(zero, key, old_value)
    kron = kron_update(initial, key, new_value)
    delta = delta_update(initial, key, new_value)
    erased = kron_update(kron, key, old_value, target=0)
    # Both coordinates of this unit vector overlap the stored value basis.
    overlap = (0.6, 0.8)
    correlated = kron_update(initial, key, overlap)
    # Same direction, but nonunit features: beta=1 is outside contraction.
    scaled_k, scaled_v = (2.0,), (3.0,)
    naive = kron_update(((0.0,),), scaled_k, scaled_v)
    projection = kron_update(((0.0,),), scaled_k, scaled_v, beta=1/36)
    errors = []
    state = ((0.0,),)
    for _ in range(4):
        state = kron_update(state, (1.0,), (1.0,), beta=0.5)
        errors.append(association(state, (1.0,), (1.0,))-1)
    return {
        "same_key_orthogonal_values": {
            "initial_readout": read(initial, key),
            "kronecker_readout": read(kron, key),
            "vector_delta_readout": read(delta, key),
            "kronecker_old_association": association(kron, key, old_value),
            "vector_delta_old_association": association(delta, key, old_value),
            "new_association_both": association(kron, key, new_value),
        },
        "overwrite_counterexample": {
            "desired_readout": new_value,
            "kronecker_squared_error": sum((a-b)**2 for a, b in zip(read(kron, key), new_value)),
            "vector_delta_squared_error": sum((a-b)**2 for a, b in zip(read(delta, key), new_value)),
        },
        "explicit_scalar_erasure": {
            "readout": read(erased, key),
            "old_association": association(erased, key, old_value),
            "new_association": association(erased, key, new_value),
        },
        "overlapping_values_interfere": {
            "new_value": overlap,
            "readout": read(correlated, key),
            "new_association": association(correlated, key, overlap),
            "old_association_before": association(initial, key, old_value),
            "old_association_after": association(correlated, key, old_value),
        },
        "nonunit_feature_control": {
            "key_norm_squared_times_value_norm_squared": 36,
            "beta_1_association": association(naive, scaled_k, scaled_v),
            "beta_1_over_36_association": association(projection, scaled_k, scaled_v),
        },
        "repeated_unit_pair_beta_half_errors": errors,
    }


def _rounded(value):
    if isinstance(value, dict):
        return {k: _rounded(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_rounded(v) for v in value]
    return round(value, 9) if isinstance(value, float) else value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print(json.dumps(_rounded(report()), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
