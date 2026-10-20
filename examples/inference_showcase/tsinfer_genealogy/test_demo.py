import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import tskit

from examples.inference_showcase.tsinfer_genealogy.demo import (
    clades, compare_genotypes, infer_from_variants, run_pipeline, simulate_source, topology_score,
)


class InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="test-tsinfer-")
        cls.directory = Path(cls.temporary.name) / "run"
        cls.source, cls.variants, cls.inferred, cls.report = run_pipeline(cls.directory)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_actual_pinned_tools_and_sample_counts(self):
        self.assertEqual(self.report["versions"]["tsinfer"], "0.5.1")
        self.assertEqual(self.report["versions"]["msprime"], "1.4.4")
        self.assertEqual(self.source.num_samples, 8)
        self.assertEqual(self.inferred.num_samples, 8)
        self.assertEqual(self.source.num_sites, 37)

    def test_seeded_generator_is_available_without_assuming_cross_platform_byte_identity(self):
        regenerated = simulate_source()
        self.assertEqual(regenerated.num_samples, 8)
        self.assertEqual(regenerated.num_trees, 1)
        self.assertGreater(regenerated.num_sites, 0)
        self.assertEqual(regenerated.num_provenances, 0)

    def test_no_truth_times_in_variant_input(self):
        self.assertTrue(all(tskit.is_unknown_time(x) for x in self.variants.sites_time[:]))
        self.assertTrue(all(tskit.is_unknown_time(x) for x in self.variants.individuals_time[:]))
        self.assertEqual(self.inferred.time_units, "uncalibrated")

    def test_every_genotype_preserved_but_topology_incomplete(self):
        scores = self.report["observed"]
        self.assertEqual(scores["genotype_reconstruction"], {"matching_allele_calls": 296, "total_allele_calls": 296})
        self.assertEqual(scores["topology_comparison"]["clade_recall"], 0.5)
        self.assertEqual(scores["topology_comparison"]["clade_precision"], 1.0)
        self.assertGreater(scores["topology_comparison"]["excluded_span"], 0)
        self.assertEqual(scores["site_inference_types"], {"full": 13, "parsimony": 24})

    def test_score_is_identity_for_truth(self):
        score = topology_score(self.source, self.source)
        self.assertEqual(score["clade_recall"], 1)
        self.assertEqual(score["clade_precision"], 1)
        self.assertEqual(score["scored_fraction"], 1)
        self.assertEqual(len(clades(self.source.first())), 6)

    def test_unresolved_star_has_zero_recall_and_undefined_precision(self):
        tables = tskit.TableCollection(self.source.sequence_length)
        for _ in range(8):
            tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=0)
        root = tables.nodes.add_row(time=1)
        for sample in range(8):
            tables.edges.add_row(0, self.source.sequence_length, root, sample)
        tables.sort()
        star = tables.tree_sequence()
        self.assertEqual(clades(star.first()), set())
        score = topology_score(self.source, star)
        self.assertEqual(score["clade_recall"], 0)
        self.assertIsNone(score["clade_precision"])

    def test_sample_order_and_genomic_length_are_checked(self):
        subset = self.source.simplify([0, 1, 2, 3])
        with self.assertRaises(ValueError):
            compare_genotypes(self.source, subset)
        with self.assertRaises(ValueError):
            topology_score(self.source, subset)
        shorter = self.source.keep_intervals([[0, 50000]]).trim()
        with self.assertRaises(ValueError):
            topology_score(self.source, shorter)

    def test_genotype_comparison_detects_changed_input(self):
        tables = self.source.dump_tables()
        original = tables.mutations[0]
        tables.mutations[0] = original.replace(derived_state="X")
        changed = tables.tree_sequence()
        score = compare_genotypes(changed, self.inferred)
        self.assertLess(score["matching_allele_calls"], score["total_allele_calls"])

    def test_checked_summary_and_synthetic_input(self):
        root = Path(__file__).parent
        self.assertEqual(self.report, json.loads((root / "sample_summary.json").read_text()))
        self.assertEqual((self.directory / "input_variants.csv").read_bytes(), (root / "sample_input.csv").read_bytes())

    def test_repeat_inference_is_deterministic(self):
        _, repeated = infer_from_variants(self.directory / "input.vcz", self.source.sequence_length)
        self.assertTrue(self.inferred.tables.equals(repeated.tables, ignore_provenance=True))

    def test_invalid_seeds_and_existing_output_directory(self):
        for bad in (0, -1, 2**32, True):
            with self.assertRaises(ValueError):
                simulate_source(ancestry_seed=bad)
        with self.assertRaises(FileExistsError):
            run_pipeline(self.directory)
        result = subprocess.run(
            [sys.executable, "-m", "examples.inference_showcase.tsinfer_genealogy.demo",
             "--output-dir", str(self.directory)],
            text=True, capture_output=True,
        )
        self.assertEqual(result.returncode, 2)
        self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
