"""tskit: one ancestor, two IBD tracts.

Estimand: edgewise genealogical IBD for one pair of haploid samples in a
supplied tree sequence, not IBD detection from genotypes. This deterministic
100-bp fixture has no mutations and no seed. All local MRCAs are node 2 at
time 10. Squashing adjacent matching edge rows preserves the local trees
but can change inferred tract lengths. An intervening node is a control:
the path through it remains distinct after squashing. A 70-bp cutoff is
illustrative, not a recommended empirical threshold. Don't squash blindly
where edges are meant to encode biological breakpoints.

Reproduce in a fresh environment with CPython 3.12.11:
    python -m pip install tskit==1.0.4 numpy==2.5.5
    python ibd_edge_rows.py --test
    python ibd_edge_rows.py

Actual output, macOS arm64:
split rows: [(0, 40, 2), (40, 100, 2)]; span=100; min_span=70: []
squashed rows: [(0, 100, 2)]; span=100; min_span=70: [(0, 100, 2)]
changed path: [(0, 40, 2), (40, 100, 2)]; span=100; min_span=70: []

Sources:
    https://tskit.dev/tskit/docs/stable/ibd.html
    https://tskit.dev/tskit/docs/stable/python-api.html#tskit.TreeSequence.ibd_segments
    https://doi.org/10.1371/journal.pcbi.1006581 (tree-sequence representation)
"""
import argparse
import unittest

import tskit



def fixture(change_parent=False):
    # Two haploid samples, one MRCA at time 10, span [0, 100).
    tables = tskit.TableCollection(100)
    a, b = (tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
            for _ in range(2))
    root = tables.nodes.add_row(time=10)
    tables.edges.add_row(0, 100, root, b)
    tables.edges.add_row(0, 40, root, a)
    if change_parent:
        unary = tables.nodes.add_row(time=3)
        tables.edges.add_row(40, 100, root, unary)
        tables.edges.add_row(40, 100, unary, a)
    else:
        tables.edges.add_row(40, 100, root, a)
    tables.sort()
    return tables


def ibd(ts, min_span=0):
    result = ts.ibd_segments(within=[0, 1], min_span=min_span, store_segments=True)
    if not result.num_segments:
        return []
    return sorted((int(s.left), int(s.right), int(s.node)) for s in result[0, 1])


def variants():
    split = fixture()
    squashed = split.copy()
    squashed.edges.squash()
    changed = fixture(change_parent=True)
    changed.edges.squash()
    changed.sort()  # Squash orders by node ID, not by parent time.
    squashed.sort()
    return {name: tb.tree_sequence() for name, tb in
            [("split rows", split), ("squashed rows", squashed), ("changed path", changed)]}


def main():
    for name, ts in variants().items():
        segments = ibd(ts)
        print(f"{name}: {segments}; span={sum(r-l for l,r,_ in segments)}; "
              f"min_span=70: {ibd(ts, 70)}")


class EdgeRowsTest(unittest.TestCase):
    def setUp(self):
        self.ts = variants()

    def test_local_tree_parent_maps_match_after_squashing(self):
        for x in (0, 39.5, 40, 99.5):
            with self.subTest(x=x):
                self.assertEqual(self.ts["split rows"].at(x).parent_dict,
                                 self.ts["squashed rows"].at(x).parent_dict)

    def test_mrca_and_time_constant_all_encodings(self):
        for name, ts in self.ts.items():
            for tree in ts.trees():
                with self.subTest(name=name, interval=tree.interval):
                    self.assertEqual(tree.mrca(0, 1), 2)
                    self.assertEqual(tree.tmrca(0, 1), 10)

    def test_distinct_stored_edges_split(self):
        self.assertEqual(ibd(self.ts["split rows"]), [(0, 40, 2), (40, 100, 2)])

    def test_squash_joins_matching_edges_only(self):
        self.assertEqual(ibd(self.ts["squashed rows"]), [(0, 100, 2)])
        self.assertEqual(ibd(self.ts["changed path"]), [(0, 40, 2), (40, 100, 2)])

    def test_unfiltered_span_invariant(self):
        for name, ts in self.ts.items():
            with self.subTest(name=name):
                self.assertEqual(sum(r-l for l, r, _ in ibd(ts)), 100)

    def test_threshold_changes_retained_span(self):
        self.assertEqual(ibd(self.ts["split rows"], 70), [])
        self.assertEqual(ibd(self.ts["squashed rows"], 70), [(0, 100, 2)])
        self.assertEqual(ibd(self.ts["changed path"], 70), [])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--test", action="store_true", help="run six focused checks")
    if parser.parse_args().test:
        unittest.main(argv=["ibd_edge_rows.py"], verbosity=2)
    else:
        main()
