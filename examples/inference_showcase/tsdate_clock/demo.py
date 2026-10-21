"""Date a known synthetic genealogy from mutations using the real tsdate engine."""

import argparse
import copy
import importlib.metadata
import json
from math import isfinite
from pathlib import Path

import numpy as np
import tsdate
import tskit

FIXTURE = Path(__file__).with_name("synthetic_input.json")
SEED = 20261021
RATE = 1e-7
LENGTH = 1_000_000
EDGES = ((4, 0), (4, 1), (5, 2), (5, 3), (6, 4), (6, 5))
TRUE_TIMES = (0, 0, 0, 0, 100, 200, 500)


def generate_input(seed=SEED):
    """Independent Poisson mutations along six fixed branches; no coalescent draw."""
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.Generator(np.random.PCG64(seed))
    mutations = []
    for parent, child in EDGES:
        mean = RATE*LENGTH*(TRUE_TIMES[parent]-TRUE_TIMES[child])
        count = rng.poisson(mean)
        for position in rng.uniform(0, LENGTH, count):
            mutations.append({"position": float(position), "node": child})
    return {
        "generator": "numpy PCG64; independent Poisson edge counts and uniform positions",
        "seed": seed, "sequence_length": LENGTH, "simulation_mutation_rate": RATE,
        "sample_count": 4, "true_node_times": list(TRUE_TIMES),
        "edges": [list(edge) for edge in EDGES],
        "mutations": sorted(mutations, key=lambda row: row["position"]),
    }


def build_undated(data, placeholder_scale=1.0):
    """Use topology and mutation locations only; never read true_node_times/rate.

    Valid tskit tables need increasing parent times. Set temporary topological
    heights, and explicitly label them uncalibrated. Mutation times are unknown.
    """
    if not isfinite(placeholder_scale) or placeholder_scale <= 0:
        raise ValueError("placeholder_scale must be positive and finite")
    edges = data["edges"]
    count = 1+max(max(edge) for edge in edges)
    heights = [0]*count
    children = {node: [] for node in range(count)}
    for parent, child in edges:
        children[parent].append(child)
    visiting, visited = set(), set()

    def height(node):
        if node in visiting:
            raise ValueError("Topology has a cycle")
        if node not in visited:
            visiting.add(node)
            heights[node] = 1+max(height(c) for c in children[node]) if children[node] else 0
            visiting.remove(node)
            visited.add(node)
        return heights[node]

    tables = tskit.TableCollection(data["sequence_length"])
    tables.time_units = tskit.TIME_UNITS_UNCALIBRATED
    for node in range(count):
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE if node < data["sample_count"] else 0,
                             time=height(node)*placeholder_scale)
    for parent, child in edges:
        tables.edges.add_row(0, data["sequence_length"], parent, child)
    for mutation in data["mutations"]:
        site = tables.sites.add_row(mutation["position"], ancestral_state="0")
        tables.mutations.add_row(site, mutation["node"], derived_state="1", time=tskit.UNKNOWN_TIME)
    tables.sort()
    return tables.tree_sequence()


def infer_dates(undated, mutation_rate):
    if isinstance(mutation_rate, bool) or not isfinite(mutation_rate) or mutation_rate <= 0:
        raise ValueError("mutation_rate must be positive and finite")
    return tsdate.date(
        undated, mutation_rate=mutation_rate, method="variational_gamma",
        max_iterations=25, rescaling_intervals=10, rescaling_iterations=0,
        match_segregating_sites=False, return_fit=True, progress=False,
        record_provenance=False, time_units="generations",
    )


def run_analysis(data, rate=RATE):
    undated = build_undated(data)
    dated, fit = infer_dates(undated, rate)
    twice_rate, fit_twice = infer_dates(undated, 2*rate)
    return undated, dated, fit, twice_rate, fit_twice


def build_report(data=None, rate=RATE):
    data = json.loads(FIXTURE.read_text()) if data is None else copy.deepcopy(data)
    undated, dated, fit, twice_rate, fit_twice = run_analysis(data, rate)
    posterior, posterior_twice = fit.node_posteriors(), fit_twice.node_posteriors()
    rows = []
    for node in range(data["sample_count"], dated.num_nodes):
        rows.append({
            "node": node, "true_time_generations_for_evaluation_only": data["true_node_times"][node],
            "posterior_mean_generations": round(float(posterior["mean"][node]), 6),
            "posterior_sd_generations": round(float(np.sqrt(posterior["variance"][node])), 6),
            "posterior_mean_at_double_assumed_rate": round(float(posterior_twice["mean"][node]), 6),
        })
    return {
        "example": "tsdate: mutation-clock dating conditional on a known topology",
        "versions": {name: importlib.metadata.version(name) for name in ["tsdate", "tskit", "numpy", "scipy", "numba"]},
        "seed": data["seed"], "sequence_length": undated.sequence_length,
        "samples": undated.num_samples, "internal_nodes": dated.num_nodes-dated.num_samples,
        "sites": undated.num_sites, "mutations": undated.num_mutations,
        "assumed_mutation_rate_per_base_per_generation": rate,
        "method": "variational_gamma; 25 EP iterations; rescaling disabled",
        "genotype_calls_preserved": bool(np.array_equal(undated.genotype_matrix(), dated.genotype_matrix())),
        "node_results": rows,
        "interpretation": (
            "Ages are inferred from mutation counts and an assumed clock conditional "
            "on a supplied topology. True ages are withheld from inference. Doubling "
            "the assumed mutation rate approximately halves ages; this is a clock "
            "sensitivity example, not a benchmark or demographic inference."
        ),
    }


def check_sample(actual, expected):
    """Compare floats with a documented cross-platform numerical tolerance."""
    if isinstance(expected, dict):
        if actual.keys() != expected.keys():
            raise AssertionError("Report keys changed")
        for key in expected:
            check_sample(actual[key], expected[key])
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            raise AssertionError("Report list length changed")
        for a, b in zip(actual, expected):
            check_sample(a, b)
    elif type(expected) is float:
        if not np.isclose(actual, expected, rtol=1e-6, atol=1e-6):
            raise AssertionError(f"Numerical sample mismatch: {actual} versus {expected}")
    elif actual != expected:
        raise AssertionError(f"Sample mismatch: {actual!r} versus {expected!r}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mutation-rate", type=float, default=RATE)
    parser.add_argument("--check-sample", action="store_true")
    args = parser.parse_args()
    try:
        report = build_report(rate=args.mutation_rate)
        if args.check_sample:
            expected = json.loads(Path(__file__).with_name("sample_summary.json").read_text())
            check_sample(report, expected)
    except (ValueError, AssertionError) as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
