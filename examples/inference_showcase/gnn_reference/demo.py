"""Original deterministic tskit GNN reference-panel example (no inference run)."""
from __future__ import annotations

import argparse
import json
import math
import platform
from pathlib import Path

import numpy as np
import tskit

HERE = Path(__file__).resolve().parent


def build_tree_sequence(*, split=None, length=100, time_scale=1):
    """Six samples: focal 0, A candidates 1--3, B candidates 4--5.

    In tree 1, 0--4 form a polytomy at time 1, joining 5 at time 2.
    If split is supplied, tree 2 instead pairs 0 and 4 at time 0.5.
    Labels are supplied to GNN as reference sets, not biological populations.
    """
    if not math.isfinite(length) or length <= 0:
        raise ValueError("length must be finite and positive")
    if not math.isfinite(time_scale) or time_scale <= 0:
        raise ValueError("time_scale must be finite and positive")
    if split is not None and (not math.isfinite(split) or not 0 < split < length):
        raise ValueError("split must be strictly inside the sequence")
    tables = tskit.TableCollection(length)
    tables.time_units = "arbitrary"
    for _ in range(6):
        tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
    tables.nodes.add_row(time=time_scale)  # node 6
    tables.nodes.add_row(time=2 * time_scale)  # node 7
    if split is not None:
        tables.nodes.add_row(time=0.5 * time_scale)  # node 8
    end = length if split is None else split
    for child in range(5):
        tables.edges.add_row(0, end, 6, child)
    for child in (6, 5):
        tables.edges.add_row(0, length, 7, child)
    if split is not None:
        for child in (0, 4):
            tables.edges.add_row(split, length, 8, child)
        for child in (1, 2, 3, 8):
            tables.edges.add_row(split, length, 6, child)
    tables.sort()
    return tables.tree_sequence()


def gnn(ts, groups, focal=0):
    """Bounded tutorial wrapper: two nonempty disjoint sample-node groups.

    Require every tree to connect focal to another reference sample. This avoids
    silently interpreting unsupported genomic intervals as zero ancestry.
    """
    samples = set(map(int, ts.samples()))
    if type(focal) is not int or focal not in samples:
        raise ValueError("focal must be an integer sample node")
    if len(groups) != 2 or any(not group for group in groups):
        raise ValueError("supply two nonempty reference groups")
    flat = [node for group in groups for node in group]
    if any(type(node) is not int or node not in samples for node in flat):
        raise ValueError("references must be integer sample nodes")
    if len(set(flat)) != len(flat):
        raise ValueError("references must be unique and disjoint")
    for tree in ts.trees():
        if not any(node != focal and tree.mrca(focal, node) != tskit.NULL for node in flat):
            raise ValueError("every tree must connect focal to another reference")
    return ts.genealogical_nearest_neighbours([focal], groups)[0]


def report():
    ts = build_tree_sequence()
    panels = {
        "one_near_A_one_near_B": [[1], [4, 5]],
        "three_near_A_one_near_B": [[1, 2, 3], [4, 5]],
        "only_distant_B": [[1, 2, 3], [5]],
    }
    return {
        "versions": {"python": platform.python_version(), "tskit": tskit.__version__, "numpy": np.__version__},
        "seed": None,
        "input": {"sample_nodes": list(range(6)), "focal": 0, "sequence_length": ts.sequence_length,
                  "edges": [[e.left, e.right, e.parent, e.child] for e in ts.edges()],
                  "node_times": ts.nodes_time.tolist(), "time_units": ts.time_units},
        "single_tree": {name: {"reference_groups": groups, "gnn_A_B": gnn(ts, groups).tolist()} for name, groups in panels.items()},
        "two_tree_control": {
            "breakpoint": 75,
            "description": "Second tree pairs focal 0 with B reference 4; spans are 75 and 25.",
            "gnn_A_B": {name: gnn(build_tree_sequence(split=75), groups).tolist() for name, groups in panels.items()},
        },
        "interpretation": "Conditional reference-neighbour fractions, not admixture proportions or inferred history.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-sample", action="store_true")
    args = parser.parse_args()
    rendered = json.dumps(report(), indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.check_sample:
        if rendered != (HERE / "sample_output.json").read_text():
            raise SystemExit("Checked sample differs")
        print("Checked sample matches")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
