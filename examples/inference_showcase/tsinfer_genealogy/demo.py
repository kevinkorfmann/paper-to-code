"""Infer genealogy topology from synthetic phased variants with real tsinfer."""

import argparse
from collections import Counter
import csv
from importlib.metadata import version
import json
from pathlib import Path
import platform
import tempfile

import bio2zarr.tskit
import msprime
import numpy as np
import tsinfer
import tskit
import zarr

ANCESTRY_SEED = 104729
MUTATION_SEED = 104743
SEQUENCE_LENGTH = 100000


def simulate_source(ancestry_seed=ANCESTRY_SEED, mutation_seed=MUTATION_SEED):
    for seed in (ancestry_seed, mutation_seed):
        if type(seed) is not int or not 1 <= seed < 2**32:
            raise ValueError("Seeds must be integers in [1, 2^32-1]")
    ancestry = msprime.sim_ancestry(
        samples=4, ploidy=2, population_size=10000,
        sequence_length=SEQUENCE_LENGTH, recombination_rate=0,
        random_seed=ancestry_seed, record_provenance=False,
    )
    return msprime.sim_mutations(
        ancestry, rate=5e-9, model=msprime.BinaryMutationModel(),
        random_seed=mutation_seed, record_provenance=False,
    )


def infer_from_variants(path, sequence_length):
    """Only variant arrays and the genomic length enter the inference stage."""
    data = zarr.open_group(str(path), mode="r")
    variants = tsinfer.VariantData(
        data, ancestral_state=data["variant_allele"][:, 0],
        sequence_length=sequence_length,
    )
    # No true topology, mutation age, coalescence time, or demographic parameter.
    # num_threads=0 uses the package's sequential implementation.
    inferred = tsinfer.infer(
        variants, num_threads=0, recombination_rate=None, record_provenance=False
    )
    return variants, inferred


def compare_genotypes(source, inferred):
    if source.num_samples != inferred.num_samples or source.num_sites != inferred.num_sites:
        raise ValueError("Sample/site counts do not agree")
    matches = total = 0
    for original, reconstructed in zip(source.variants(), inferred.variants(), strict=True):
        if original.site.position != reconstructed.site.position:
            raise ValueError("Variant positions do not agree")
        for a, b in zip(original.genotypes, reconstructed.genotypes, strict=True):
            if a < 0 or b < 0:
                raise ValueError("This example expects complete genotypes")
            # Compare decoded states: allele index labels need not be identical.
            matches += original.alleles[a] == reconstructed.alleles[b]
            total += 1
    return {"matching_allele_calls": int(matches), "total_allele_calls": total}


def clades(tree):
    """Unique nontrivial rooted clades as sets of sample IDs; ignore unary nodes."""
    n = tree.tree_sequence.num_samples
    return {
        frozenset(tree.samples(node))
        for node in tree.nodes()
        if 1 < tree.num_samples(node) < n
    }


def topology_score(source, inferred):
    """Span-weighted clade precision/recall on intervals with a single root.

    Unknown flanks and forests are explicitly excluded and their span reported.
    Sample identity/order must agree; internal node IDs and time scales are ignored.
    """
    if source.sequence_length != inferred.sequence_length:
        raise ValueError("Genomic lengths do not agree")
    if not np.array_equal(source.samples(), inferred.samples()):
        raise ValueError("Sample node IDs/order do not agree")
    boundaries = sorted(set(source.breakpoints()) | set(inferred.breakpoints()))
    scored = shared_mass = true_mass = inferred_mass = 0.0
    for left, right in zip(boundaries, boundaries[1:]):
        truth = source.at((left + right) / 2)
        estimate = inferred.at((left + right) / 2)
        if truth.num_roots != 1 or estimate.num_roots != 1:
            continue
        length = right - left
        true_clades, inferred_clades = clades(truth), clades(estimate)
        scored += length
        shared_mass += length * len(true_clades & inferred_clades)
        true_mass += length * len(true_clades)
        inferred_mass += length * len(inferred_clades)
    return {
        "scored_span": scored,
        "excluded_span": source.sequence_length - scored,
        "scored_fraction": round(scored / source.sequence_length, 9),
        "clade_precision": round(shared_mass / inferred_mass, 9) if inferred_mass else None,
        "clade_recall": round(shared_mass / true_mass, 9) if true_mass else None,
        "shared_clade_span": shared_mass,
        "true_clade_span": true_mass,
        "inferred_clade_span": inferred_mass,
    }


def write_variants(source, path):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["position", "ancestral_state", *[f"haplotype_{i}" for i in range(source.num_samples)]])
        for variant in source.variants():
            writer.writerow([
                int(variant.site.position), variant.site.ancestral_state,
                *[variant.alleles[index] for index in variant.genotypes],
            ])


def run_pipeline(output_dir, ancestry_seed=ANCESTRY_SEED, mutation_seed=MUTATION_SEED):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    source = simulate_source(ancestry_seed, mutation_seed)
    if source.num_sites == 0:
        raise ValueError("Synthetic input has no variant sites")
    source.dump(output_dir / "source.trees")
    write_variants(source, output_dir / "input_variants.csv")
    variant_path = output_dir / "input.vcz"
    bio2zarr.tskit.convert(source, variant_path, worker_processes=0, show_progress=False)
    variants, inferred = infer_from_variants(variant_path, source.sequence_length)
    inferred.dump(output_dir / "inferred.trees")
    report = {
        "example": "inference-showcase: tsinfer genealogy from synthetic phased variants",
        "versions": {
            "python": platform.python_version(),
            **{name: version(name) for name in ("tsinfer", "tskit", "msprime", "bio2zarr", "numpy", "zarr")},
        },
        "simulation_parameters": {
            "diploid_individuals": 4, "haploid_genomes": 8,
            "population_size": 10000, "sequence_length": SEQUENCE_LENGTH,
            "recombination_rate": 0, "mutation_rate": 5e-9,
            "mutation_model": "BinaryMutationModel", "discrete_genome": True,
            "ancestry_seed": ancestry_seed, "mutation_seed": mutation_seed,
        },
        "inference_settings": {
            "num_threads": 0, "recombination_rate": None,
            "ancestral_states": "known synthetic ancestral alleles",
            "site_ages_supplied": False, "true_topology_supplied": False,
        },
        "observed": {
            "variant_sites": source.num_sites,
            "true_local_trees": source.num_trees,
            "inferred_local_trees_including_unknown_flanks": inferred.num_trees,
            "inferred_time_units": inferred.time_units,
            "site_inference_types": dict(sorted(Counter(
                site.metadata["inference_type"] for site in inferred.sites()
            ).items())),
            "genotype_reconstruction": compare_genotypes(source, inferred),
            "topology_comparison": topology_score(source, inferred),
        },
        "interpretation": (
            "Genotype preservation tests the data representation, not ancestry accuracy. "
            "Topology scores compare this one synthetic genealogy on explicitly covered "
            "intervals; they are not general tool-performance estimates. Internal times "
            "are uncalibrated, not inferred dates or demographic parameters."
        ),
    }
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return source, variants, inferred, report


def build_report():
    with tempfile.TemporaryDirectory(prefix="tsinfer-example-") as temporary:
        return run_pipeline(Path(temporary) / "run")[-1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, help="new directory for variants, trees, and summary")
    args = parser.parse_args()
    try:
        report = build_report() if args.output_dir is None else run_pipeline(args.output_dir)[-1]
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
