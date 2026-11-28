"""Show what a covariate removes from a tskit site statistic."""
import argparse
import json
from pathlib import Path

import numpy as np
import tskit

GROUP = np.array([0, 0, 0, 0, 1, 1, 1, 1])
WITHIN = np.array([0, 0, 1, 1, 0, 0, 1, 1])
TRAIT = 10 * GROUP + 2 * WITHIN
WINDOWS = [0, 10, 20]


def build_ts():
    t = tskit.TableCollection(sequence_length=20)
    for _ in GROUP:
        t.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
    for left, pattern in zip(WINDOWS, (GROUP, WITHIN)):
        clade = t.nodes.add_row(time=1)
        root = t.nodes.add_row(time=2)
        t.edges.add_row(left, left + 10, root, clade)
        for child, present in enumerate(pattern):
            t.edges.add_row(left, left + 10, clade if present else root, child)
        site = t.sites.add_row(left + 5, ancestral_state="0")
        t.mutations.add_row(site, clade, derived_state="1")
    t.sort()
    return t.tree_sequence()


def statistic(ts, covariates=None, span_normalise=False, trait=TRAIT):
    return ts.trait_linear_model(
        np.asarray(trait)[:, None], Z=covariates, mode="site",
        windows=WINDOWS, span_normalise=span_normalise,
    ).ravel()


def report():
    ts = build_ts()
    return {
        "windows": WINDOWS,
        "site_order": ["group_tag", "within_group_marker"],
        "unadjusted_beta_squared": statistic(ts).round(10).tolist(),
        "group_adjusted_beta_squared": statistic(ts, GROUP[:, None]).round(10).tolist(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-sample", action="store_true")
    rendered = json.dumps(report(), indent=2) + "\n"
    if parser.parse_args().check_sample:
        if rendered != Path(__file__).with_name("sample_output.json").read_text():
            raise SystemExit("Output does not match the saved sample")
    print(rendered, end="")
