"""Mathematical, scientific, integration, reproducibility and CLI checks."""

from fractions import Fraction
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from examples.inference_showcase.plink_king import demo
from examples.inference_showcase.plink_king.install import BINARY, BINARY_SHA256, verify


class KingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.samples = demo.synthetic()
        cls.records = demo.run_plink(cls.samples)
        cls.result = demo.summary()

    def test_all_36_pair_counts_and_exact_rational_scores(self):
        self.assertEqual(len(self.records), 36)
        self.assertLessEqual(demo.compare_all(self.samples, self.records), 5e-6)

    def test_actual_sample_output(self):
        self.assertEqual(self.result, json.loads((demo.HERE / "sample_output.json").read_text()))

    def test_synthetic_reproduction_and_seed_sensitivity(self):
        self.assertEqual(self.samples, demo.synthetic())
        self.assertNotEqual(self.samples, demo.synthetic(seed=20261030))

    def test_vcf_input_fingerprint(self):
        self.assertEqual(hashlib.sha256(demo.vcf_text(self.samples).encode()).hexdigest(),
                         self.result["synthetic_vcf_sha256"])

    def test_duplicate_exactly_one_half(self):
        self.assertEqual(self.records[("duplicate", "parentA")]["kinship"], 0.5)
        self.assertEqual(self.samples["duplicate"], self.samples["parentA"])

    def test_first_degree_values_and_unrelated_control(self):
        selected = self.result["selected_pairs"]
        for label in ("parent_child", "full_siblings"):
            self.assertLess(abs(selected[label]["plink_kinship"] - 0.25), 0.02)
        self.assertLess(abs(selected["unrelated_same_population"]["plink_kinship"]), 0.02)

    def test_parent_child_has_no_opposite_homozygotes(self):
        for child in ("childA", "childB"):
            for parent in ("parentA", "parentB"):
                self.assertEqual(self.records[tuple(sorted((child, parent)))]["ibs0"], 0)
        self.assertGreater(self.records[("childA", "childB")]["ibs0"], 0)

    def test_extreme_ancestry_counterexample_retains_negative_scores(self):
        selected = self.result["selected_pairs"]
        for label in ("mixed_parent_child_low", "mixed_parent_child_high"):
            self.assertLess(selected[label]["plink_kinship"], -0.6)
            self.assertEqual(selected[label]["ibs0_count"], 0)
        self.assertLess(selected["unrelated_different_populations"]["plink_kinship"], -3)

    def test_balanced_mendelian_enumeration(self):
        expected = {"parent_child": 0.25, "full_siblings": 0.25, "unrelated_parents": 0.0}
        self.assertEqual(self.result["balanced_256_locus_control"], expected)
        self.assertEqual(len(demo.balanced_pedigree()["parentA"]), 256)

    def test_independent_weighted_mendelian_probability_limits(self):
        # A separate exact probabilistic calculation, not the synthetic generator
        # and not the average of finite-sample ratio estimates.
        for p, q, expected in [(Fraction(3, 10), Fraction(3, 10), Fraction(1, 4)),
                               (Fraction(1, 10), Fraction(9, 10), Fraction(-23, 36))]:
            hp = hc = distance = opposite = total = Fraction(0)
            for a0, a1, b0, b1, i, j in itertools.product((0, 1), repeat=6):
                mass = Fraction(1, 4)
                for allele, frequency in zip((a0, a1, b0, b1), (p, p, q, q)):
                    mass *= frequency if allele else 1 - frequency
                father = a0 + a1
                child = (a0, a1)[i] + (b0, b1)[j]
                hp += mass * (father == 1)
                hc += mass * (child == 1)
                distance += mass * (father - child) ** 2
                opposite += mass * (abs(father - child) == 2)
                total += mass
            self.assertEqual(total, 1)
            self.assertEqual(opposite, 0)
            self.assertEqual(Fraction(1, 2) - distance / (4 * min(hp, hc)), expected)

    def test_between_family_not_average_heterozygosity(self):
        samples = {"x": [0, 1, 1, 2], "y": [0, 1, 2, 2]}
        self.assertEqual(demo.run_plink(samples)[("x", "y")]["kinship"], 0.25)
        self.assertEqual(demo.reference_pair(*samples.values())["kinship"], Fraction(1, 4))
        self.assertNotEqual(Fraction(1, 4), Fraction(1, 3))

    def test_large_negative_score_uses_significant_digit_tolerance(self):
        samples = {"x": [1] * 3 + [0] * 100 + [1],
                   "y": [1] * 3 + [2] * 100 + [0]}
        records = demo.run_plink(samples)
        self.assertEqual(records[("x", "y")]["kinship"], -32.9167)
        self.assertEqual(demo.reference_pair(*samples.values())["kinship"], Fraction(-395, 12))
        self.assertLess(demo.compare_all(samples, records), 0.00005)

    def test_joint_missingness(self):
        samples = {"x": [0, 1, 1, 2, None, 1], "y": [0, 1, 2, 2, 1, None]}
        records = demo.run_plink(samples)
        self.assertEqual(records[("x", "y")]["n_sites"], 4)
        self.assertEqual(records[("x", "y")]["kinship"], 0.25)
        self.assertEqual(demo.compare_all(samples, records), 0)

    def test_pair_sample_composition_invariance(self):
        samples = {name: self.samples[name] for name in ("parentA", "childA")}
        self.assertEqual(demo.run_plink(samples)[("childA", "parentA")],
                         self.records[("childA", "parentA")])

    def test_sample_order_invariance(self):
        reversed_samples = dict(reversed(list(self.samples.items())))
        self.assertEqual(demo.run_plink(reversed_samples), self.records)

    def test_locus_order_invariance(self):
        reversed_loci = {name: values[::-1] for name, values in self.samples.items()}
        self.assertEqual(demo.run_plink(reversed_loci), self.records)

    def test_simultaneous_reference_alternate_flip_invariance(self):
        flipped = {name: [2 - g for g in values] for name, values in self.samples.items()}
        self.assertEqual(demo.run_plink(flipped), self.records)

    def test_replicating_loci_changes_counts_not_scores(self):
        samples = {"x": [0, 1, 1, 2] * 10, "y": [0, 1, 2, 2] * 10}
        row = demo.run_plink(samples)[("x", "y")]
        self.assertEqual(row["kinship"], 0.25)
        self.assertEqual(row["n_sites"], 40)
        self.assertEqual(row["het_het"], 10)

    def test_undefined_score_is_not_zero(self):
        with self.assertRaisesRegex(ValueError, "undefined"):
            demo.reference_pair([0, 2, 0], [0, 2, 1])

    def test_invalid_genotypes(self):
        for invalid in (-1, 3, 1.0, True, "1"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                demo.vcf_text({"x": [0, invalid], "y": [1, 2]})

    def test_invalid_sample_shapes_and_names(self):
        for samples in ({}, {"x": [1]}, {"x": [], "y": []},
                        {"x": [0], "y": [1, 2]}, {"bad id": [1], "y": [1]},
                        {"x": "1", "y": [1]}, {"é": [1], "y": [1]}):
            with self.subTest(samples=samples), self.assertRaises(ValueError):
                demo.vcf_text(samples)

    def test_invalid_random_controls(self):
        for seed, loci in ((-1, 100), (True, 100), (1, 0), (1, 2.5)):
            with self.assertRaises(ValueError):
                demo.synthetic(seed, loci)

    def test_dependency_checksum_and_tamper_detection(self):
        verify(BINARY.read_bytes(), BINARY_SHA256)
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            verify(b"not the pinned binary", BINARY_SHA256)

    def test_cli_default_and_saved_input(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "input.vcf"
            run = subprocess.run([sys.executable, "-m", demo.__name__, "--write-vcf", str(path)],
                                 capture_output=True, text=True, check=True, timeout=60)
            self.assertEqual(run.stdout, (demo.HERE / "sample_output.json").read_text())
            self.assertEqual(path.read_text(), demo.vcf_text(self.samples))

    def test_cli_check_sample_and_reject_mismatched_options(self):
        run = subprocess.run([sys.executable, "-m", demo.__name__, "--check-sample"],
                             capture_output=True, text=True, timeout=60)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(run.stdout, "")
        for options in (("--loci", "0"), ("--seed", "-1"),
                        ("--check-sample", "--seed", "1")):
            run = subprocess.run([sys.executable, "-m", demo.__name__, *options],
                                 capture_output=True, text=True, timeout=60)
            self.assertNotEqual(run.returncode, 0)


if __name__ == "__main__":
    unittest.main()
