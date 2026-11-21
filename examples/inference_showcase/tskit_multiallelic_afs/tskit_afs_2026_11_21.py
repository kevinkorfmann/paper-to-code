"""One multiallelic site can contribute more than one unit to a folded AFS.

Run: python tskit_afs_2026_11_21.py
Tested with CPython 3.12.11, tskit 1.0.3, and NumPy 2.5.4.
Deterministic synthetic haploid samples; no random seed or external data.
Output below is unnormalised (site weights or branch-generation x bp).

tskit statistic definition:
https://tskit.dev/tskit/docs/stable/stats.html#allele-frequency-spectrum
Site/branch framework: https://doi.org/10.1534/genetics.120.303253

Site mode groups distinct observed allelic states by count. In folded mode a
state of count k contributes half to bin min(k, n-k); this sums to one for
biallelic sites but not for all multiallelic sites. In polarised mode the
ancestral state is excluded. Branch mode instead measures the lengths of
branches subtending each number of samples, integrated across genomic span.
It does not count observed variants or depend on their allelic states.
Inputs are complete, connected, dated genealogies and known ancestral states
for the polarised comparisons. No mutation rates or demography are inferred;
this is a hand-built counterexample, not a fit or validation on genomic data.
"""

from collections import Counter

import numpy as np
import tskit


def fixture(triallelic):
    """Six tips: the first two coalesce at 1, then all join at 3."""
    tables = tskit.TableCollection(10)
    tables.time_units = "generations"
    tips = [tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
            for _ in range(6)]
    pair = tables.nodes.add_row(time=1)
    root = tables.nodes.add_row(time=3)
    for tip in tips[:2]:
        tables.edges.add_row(0, 10, pair, tip)
    for child in tips[2:] + [pair]:
        tables.edges.add_row(0, 10, root, child)
    site = tables.sites.add_row(4, ancestral_state="A")
    first = tables.mutations.add_row(site=site, node=pair, derived_state="C")
    if triallelic:
        tables.mutations.add_row(
            site=site, node=tips[0], derived_state="G", parent=first
        )
    tables.sort()
    return tables.tree_sequence()


def main():
    for label, triallelic in (("biallelic", False), ("triallelic", True)):
        ts = fixture(triallelic)
        variant = next(ts.variants())
        counts = Counter(variant.alleles[g] for g in variant.genotypes)
        expected_counts = {"A": 4, "C": 1, "G": 1} if triallelic else {"A": 4, "C": 2}
        if counts != expected_counts or ts.num_sites != 1:
            raise AssertionError((counts, ts.num_sites))
        folded = ts.allele_frequency_spectrum(span_normalise=False)
        polarised = ts.allele_frequency_spectrum(
            span_normalise=False, polarised=True
        )
        branch = ts.allele_frequency_spectrum(
            span_normalise=False, mode="branch", polarised=True
        )
        np.testing.assert_array_equal(
            folded, [0, 1, 0.5, 0, 0, 0, 0] if triallelic
            else [0, 0, 1, 0, 0, 0, 0]
        )
        np.testing.assert_array_equal(
            polarised, [0, 2, 0, 0, 0, 0, 0] if triallelic
            else [0, 0, 1, 0, 0, 0, 0]
        )
        # Two one-generation and four three-generation terminal branches;
        # one two-generation branch subtends two tips. All span 10 bp.
        np.testing.assert_array_equal(branch, [0, 140, 20, 0, 0, 0, 0])
        np.testing.assert_array_equal(
            ts.allele_frequency_spectrum(mode="branch", polarised=True),
            [0, 14, 2, 0, 0, 0, 0],
        )
        print(f"{label}: {dict(sorted(counts.items()))}; {ts.num_sites} site")
        print(f"  folded site bins: {folded.tolist()}, mass={folded.sum()}")
        print(f"  polarised site bins: {polarised.tolist()}")
        print(f"  polarised branch bins: {branch.tolist()}")
    print("10 exact checks passed")


if __name__ == "__main__":
    main()
