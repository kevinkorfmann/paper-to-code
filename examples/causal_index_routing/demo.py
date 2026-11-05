"""Deterministic routing, truncation, and future-contamination controls."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform

from .model import PrefixIndex, exhaustive_top_k, native_scores, select

SAMPLE = Path(__file__).with_name("sample_output.json")


def build_report():
    heads, weights = [[1]], [1]
    keys = [[9], [-9], [4], [4]]
    contiguous = PrefixIndex(keys, [[0, 1], [2, 3]])
    similar = PrefixIndex(keys, [[0, 2], [1, 3]])
    routed = select(contiguous, keys, heads, weights, budget=2, top_k=1)
    regrouped = select(similar, keys, heads, weights, budget=2, top_k=1)
    full = select(contiguous, keys, heads, weights, budget=4, top_k=1)
    exact = exhaustive_top_k(heads, weights, keys, 1)
    scores = native_scores(heads, weights, keys)

    prefix = [[2], [1]]
    safe = PrefixIndex(prefix, [[0], [1]])
    before = select(safe, prefix, heads, weights, budget=1, top_k=1)
    complete = prefix + [[-100]]
    future_index = PrefixIndex(complete, [[0, 2], [1]])
    rejected = False
    try:
        select(future_index, prefix, heads, weights, budget=1, top_k=1)
    except ValueError:
        rejected = True
    # Deliberately WRONG comparator: use future-derived means to rank clusters,
    # then mask future IDs. Kept separate from the safe production selector.
    bad_scores = native_scores(heads, weights, future_index.representatives)
    bad_order = sorted(range(len(bad_scores)), key=lambda i: (-bad_scores[i], i))
    bad_candidates = [i for group in bad_order for i in future_index.groups[group]
                      if i < len(prefix)][:1]
    after = select(safe, complete[:len(prefix)], heads, weights, budget=1, top_k=1)

    old = PrefixIndex([[2], [1], [3]], [[1, 2]], sinks=[0])
    window = select(old, [[2], [1], [3], [10]], heads, weights, budget=3, top_k=1)
    trunc_keys = [[1], [10]]
    truncated = select(PrefixIndex(trunc_keys, [[0, 1]]), trunc_keys,
                       heads, weights, budget=1, top_k=1)
    return {
        "environment": {"implementation": platform.python_implementation(),
                        "python": platform.python_version(), "dependencies": []},
        "scope": "Supplied-cluster mean routing only; no training or speed claims",
        "cluster_cancellation": {
            "keys": keys, "native_scores": scores,
            "contiguous_means": contiguous.representatives,
            "mean_of_member_scores": [sum(scores[i] for i in group) / len(group)
                                      for group in contiguous.groups],
            "contiguous_selection": asdict(routed),
            "similarity_regrouping": asdict(regrouped),
            "full_budget_selection": asdict(full),
            "exhaustive_top_1": exact,
            "top_1_recall_contiguous": len(set(routed.selected_ids) & set(exact)) / len(exact),
            "top_1_recall_regrouped": len(set(regrouped.selected_ids) & set(exact)) / len(exact)},
        "causality": {
            "visible_prefix": prefix, "future_key": [-100],
            "correct_before": before.selected_ids,
            "correct_after_future_append": after.selected_ids,
            "future_derived_means": future_index.representatives,
            "wrong_global_route_then_mask_candidates": bad_candidates,
            "future_snapshot_rejected": rejected},
        "reserved_candidate_slots": asdict(window),
        "truncation_counterexample": {
            "keys": trunc_keys, "selection": asdict(truncated),
            "exhaustive_top_1": exhaustive_top_k(heads, weights, trunc_keys, 1)},
    }


def render(report):
    return json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-sample", action="store_true")
    args = parser.parse_args()
    output = render(build_report())
    if args.check_sample:
        if not SAMPLE.exists() or SAMPLE.read_text(encoding="utf-8") != output:
            raise SystemExit("sample differs; use the pinned CPython interpreter")
        print("sample output matches")
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
