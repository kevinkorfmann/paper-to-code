"""Show the denominator of tskit's mean descendant counts."""
import numpy as np
import tskit


def make_example():
    tables = tskit.TableCollection(sequence_length=100)
    a0, a1, b0 = [tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
                  for _ in range(3)]
    focal = tables.nodes.add_row(time=1)
    root = tables.nodes.add_row(time=2)
    for left, right, below_focal in ((0, 20, [a0, a1]),
                                      (20, 50, [b0]), (50, 100, [])):
        for sample in (a0, a1, b0):
            tables.edges.add_row(left, right,
                                 focal if sample in below_focal else root,
                                 sample)
        if below_focal:
            tables.edges.add_row(left, right, root, focal)
    tables.sort()
    return tables.tree_sequence(), focal, [[a0, a1], [b0]]


def estimate(ts, focal, sample_sets):
    conditional = ts.mean_descendants(sample_sets)[focal]
    weighted_counts = np.zeros(len(sample_sets))
    active_span = 0
    for tree in ts.trees():
        descendants = set(tree.samples(focal))
        active_span += tree.span if descendants else 0
        for j, samples in enumerate(sample_sets):
            weighted_counts[j] += tree.span * len(descendants.intersection(samples))
    return conditional, active_span, weighted_counts / ts.sequence_length


def main():
    ts, focal, sets = make_example()
    conditional, active_span, whole_span = estimate(ts, focal, sets)
    print(f"tree intervals: {ts.num_trees}")
    print(f"focal ancestral span: {active_span:g}/{ts.sequence_length:g}")
    print("mean_descendants (A, B):", ", ".join(f"{x:.2f}" for x in conditional))
    print("including absent intervals (A, B):",
          ", ".join(f"{x:.2f}" for x in whole_span))


if __name__ == "__main__":
    main()
