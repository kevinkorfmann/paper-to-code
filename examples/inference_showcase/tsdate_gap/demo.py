"""Conditional tsdate root age with the same variants and different exposure."""
from argparse import ArgumentParser
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np
import tsdate
import tskit

HERE = Path(__file__).resolve().parent
END = 2_000_000
CALLABLE_END = 1_000_000
MUTATION_RATE = 1e-7
ROOT = 2


def fixture():
    tables = tskit.TableCollection(END)
    tables.time_units = tskit.TIME_UNITS_UNCALIBRATED
    for _ in range(2):
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
    # This is a topological placeholder, not the generated or assumed age.
    tables.nodes.add_row(time=1)
    for child in range(2):
        tables.edges.add_row(0, END, ROOT, child)
    for index in range(20):
        site = tables.sites.add_row(25_000 + index * 50_000, ancestral_state="0")
        tables.mutations.add_row(site, index % 2, derived_state="1")
    tables.sort()
    included = tables.tree_sequence()
    masked = included.delete_intervals([[CALLABLE_END, END]], simplify=False)
    return included, masked, masked.trim()


def audit_counts(ts):
    """Independent span and mutation counts using edges and site records."""
    span = {tip: 0 for tip in ts.samples()}
    mutations = dict.fromkeys(span, 0)
    for edge in ts.edges():
        if edge.parent != ROOT or edge.child not in span:
            raise ValueError("unexpected topology")
        span[edge.child] += edge.right - edge.left
    for mutation in ts.mutations():
        if mutation.node not in mutations:
            raise ValueError("unexpected mutation assignment")
        mutations[mutation.node] += 1
    return list(span.values()), list(mutations.values())


def run_date(ts):
    dated, fit = tsdate.date(
        ts, mutation_rate=MUTATION_RATE, method="variational_gamma",
        max_iterations=25, rescaling_iterations=0, match_segregating_sites=False,
        return_fit=True, record_provenance=False, time_units="generations",
    )
    estimate = fit.node_posteriors()[ROOT]
    return dated, float(estimate["mean"]), float(estimate["variance"] ** 0.5)


def summarize(ts, dated, mean, sd, label):
    spans, counts = audit_counts(ts)
    return {
        "treatment": label,
        "nominal_coordinate_length_bp": int(ts.sequence_length),
        "per_edge_spans_bp": [int(x) for x in spans],
        "per_edge_mutation_counts": counts,
        "time_estimate_generations": round(mean, 8),
        "posterior_sd_generations": round(sd, 8),
        "independent_poisson_mle_generations": round(sum(counts)/(MUTATION_RATE*sum(spans)), 8),
        "tip_dates_unchanged": bool(np.array_equal(dated.nodes_time[:ROOT], [0, 0])),
        "genotype_calls_unchanged_by_dating": bool(
            np.array_equal(ts.genotype_matrix(), dated.genotype_matrix())
        ),
    }


def report():
    runs = zip(("gap_retained_as_exposure", "gap_masked_coordinates_preserved",
                "gap_masked_then_trimmed"), fixture(), strict=True)
    return {
        "versions": {k: version(k) for k in ("tsdate", "tskit", "numpy")},
        "mutation_rate_per_bp_generation": MUTATION_RATE,
        "seed": None,
        "variational_gamma_iterations": 25,
        "rescaling_iterations": 0,
        "runs": [summarize(ts, *run_date(ts), label) for label, ts in runs],
    }


def equal_report(actual, expected):
    if isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            equal_report(actual[k], v) for k, v in expected.items()
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            equal_report(a, e) for a, e in zip(actual, expected)
        )
    if type(expected) is float:
        return bool(np.isclose(actual, expected, atol=1e-7, rtol=1e-7))
    return actual == expected


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--check-sample", action="store_true")
    args = parser.parse_args()
    actual = report()
    if args.check_sample:
        expected = json.loads((HERE / "sample_output.json").read_text())
        if not equal_report(actual, expected):
            raise SystemExit("sample output mismatch")
        print("sample output matches within documented tolerance")
    else:
        print(json.dumps(actual, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
