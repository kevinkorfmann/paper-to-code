"""Run actual PLINK 2 on synthetic diploid pedigrees, with independent controls."""

import argparse
import csv
from fractions import Fraction
import hashlib
import itertools
import json
import math
from pathlib import Path
import platform
import random
import subprocess
import tempfile

from .install import BINARY, BINARY_SHA256, VERSION, verify

HERE = Path(__file__).resolve().parent
SEED = 20261029
LOCI = 10000
NAMES = ("parentA", "parentB", "childA", "childB", "unrelated", "duplicate",
         "low", "high", "admixed")
PAIRS = {
    "parent_child": ("parentA", "childA"),
    "full_siblings": ("childA", "childB"),
    "duplicate": ("parentA", "duplicate"),
    "unrelated_same_population": ("parentA", "unrelated"),
    "unrelated_different_populations": ("low", "high"),
    "mixed_parent_child_low": ("low", "admixed"),
    "mixed_parent_child_high": ("high", "admixed"),
}


def synthetic(seed=SEED, loci=LOCI):
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if type(loci) is not int or loci < 1:
        raise ValueError("loci must be a positive integer")
    rng = random.Random(seed)
    samples = {name: [] for name in NAMES}
    for _ in range(loci):
        def founder(p):
            return (int(rng.random() < p), int(rng.random() < p))
        def offspring(a, b):
            return (a[rng.randrange(2)], b[rng.randrange(2)])
        a, b = founder(0.3), founder(0.3)
        ca, cb = offspring(a, b), offspring(a, b)
        unrelated = founder(0.3)
        low, high = founder(0.1), founder(0.9)
        admixed = offspring(low, high)
        for name, alleles in zip(NAMES, (a, b, ca, cb, unrelated, a, low, high, admixed)):
            samples[name].append(sum(alleles))
    return samples


def balanced_pedigree():
    """All 256 equally likely p=1/2 founder/transmission states, not a simulation."""
    samples = {name: [] for name in ("parentA", "parentB", "childA", "childB")}
    for a0, a1, b0, b1, t0, t1, t2, t3 in itertools.product((0, 1), repeat=8):
        a, b = (a0, a1), (b0, b1)
        row = (sum(a), sum(b), a[t0] + b[t1], a[t2] + b[t3])
        for name, value in zip(samples, row):
            samples[name].append(value)
    return samples


def validate(samples):
    if not isinstance(samples, dict) or len(samples) < 2:
        raise ValueError("at least two named samples are required")
    lengths = set()
    for name, values in samples.items():
        if not isinstance(name, str) or not name or not name.isascii() or not name.isalnum():
            raise ValueError("sample IDs must be nonempty ASCII letters/digits")
        if not isinstance(values, (list, tuple)):
            raise ValueError("genotypes must be lists or tuples")
        lengths.add(len(values))
        if any(value is not None and (type(value) is not int or value not in (0, 1, 2))
               for value in values):
            raise ValueError("genotypes must be 0, 1, 2, or None")
    if len(lengths) != 1 or not next(iter(lengths)):
        raise ValueError("samples need equal, nonzero numbers of loci")


def vcf_text(samples):
    validate(samples)
    lines = ["##fileformat=VCFv4.2", "##contig=<ID=1>",
             '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
             "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + "\t".join(samples)]
    codes = {0: "0/0", 1: "0/1", 2: "1/1", None: "./."}
    for k, genotypes in enumerate(zip(*samples.values()), start=1):
        lines.append(f"1\t{k}\ts{k}\tA\tC\t.\tPASS\t.\tGT\t" +
                     "\t".join(codes[g] for g in genotypes))
    return "\n".join(lines) + "\n"


def reference_pair(x, y):
    """Exact rational Eq. 11, using squared genotype distances, not tool output."""
    validate({"x": x, "y": y})
    called = [(a, b) for a, b in zip(x, y) if a is not None and b is not None]
    hx = sum(a == 1 for a, _ in called)
    hy = sum(b == 1 for _, b in called)
    if min(hx, hy) == 0:
        raise ValueError("KING score undefined without heterozygotes in both samples")
    distance = sum((a - b) ** 2 for a, b in called)
    return {
        "n_sites": len(called), "het_x": hx, "het_y": hy,
        "het_het": sum(a == b == 1 for a, b in called),
        "ibs0": sum(abs(a - b) == 2 for a, b in called),
        "squared_distance": distance,
        "kinship": Fraction(1, 2) - Fraction(distance, 4 * min(hx, hy)),
    }


def run_plink(samples):
    validate(samples)
    if not BINARY.is_file():
        raise FileNotFoundError("Run python -m examples.inference_showcase.plink_king.install first")
    verify(BINARY.read_bytes(), BINARY_SHA256)
    version = subprocess.run([str(BINARY), "--version"], check=True, capture_output=True,
                             text=True, timeout=30).stdout.strip()
    if version != VERSION:
        raise RuntimeError(f"Unexpected executable version: {version}")
    with tempfile.TemporaryDirectory(prefix="synthetic-king-") as directory:
        root = Path(directory)
        vcf = root / "synthetic.vcf"
        vcf.write_text(vcf_text(samples), encoding="utf-8")
        command = [str(BINARY), "--vcf", str(vcf), "--make-king-table", "counts",
                   "cols=id,nsnp,hethet,ibs0,ibs1,kinship", "--threads", "1",
                   "--memory", "640", "--out", str(root / "result")]
        completed = subprocess.run(command, capture_output=True, text=True, timeout=60)
        if completed.returncode:
            raise RuntimeError(completed.stdout + completed.stderr)
        table = (root / "result.kin0").read_text(encoding="utf-8")
    records = {}
    for row in csv.DictReader(table.lstrip("#").splitlines(), delimiter="\t"):
        x, y = row["IID1"], row["IID2"]
        hh = int(row["HETHET"])
        records[tuple(sorted((x, y)))] = {
            "n_sites": int(row["NSNP"]), "het_het": hh, "ibs0": int(row["IBS0"]),
            "hets": {x: hh + int(row["HET1_HOM2"]), y: hh + int(row["HET2_HOM1"])},
            "kinship": float(row["KINSHIP"]),
        }
    if any(not math.isfinite(row["kinship"]) for row in records.values()):
        raise ValueError("Nonfinite KING score; check jointly called heterozygotes")
    if len(records) != len(samples) * (len(samples) - 1) // 2:
        raise RuntimeError("Unexpected number of PLINK pair records")
    return records


def compare_all(samples, records):
    errors = []
    for x, y in itertools.combinations(samples, 2):
        expected = reference_pair(samples[x], samples[y])
        actual = records[tuple(sorted((x, y)))]
        for key in ("n_sites", "het_het", "ibs0"):
            if actual[key] != expected[key]:
                raise AssertionError(f"PLINK/reference {key} disagreement for {x}, {y}")
        if actual["hets"] != {x: expected["het_x"], y: expected["het_y"]}:
            raise AssertionError("PLINK/reference heterozygosity disagreement")
        error = abs(actual["kinship"] - float(expected["kinship"]))
        # PLINK tables have six significant digits, not six decimal places.
        magnitude = max(abs(actual["kinship"]), abs(float(expected["kinship"])))
        tolerance = (0.500001 * 10 ** (math.floor(math.log10(magnitude)) - 5)
                     if magnitude else 1e-12)
        if error > tolerance:
            raise AssertionError("PLINK/reference kinship disagreement")
        errors.append(error)
    return max(errors)


def summary(seed=SEED, loci=LOCI):
    samples = synthetic(seed, loci)
    records = run_plink(samples)
    maximum_error = compare_all(samples, records)
    selected = {}
    for label, (x, y) in PAIRS.items():
        actual = records[tuple(sorted((x, y)))]
        expected = reference_pair(samples[x], samples[y])
        selected[label] = {
            "samples": [x, y], "n_sites": actual["n_sites"],
            "heterozygotes": [actual["hets"][x], actual["hets"][y]],
            "both_heterozygous": actual["het_het"], "ibs0_count": actual["ibs0"],
            "plink_kinship": actual["kinship"],
            "independent_exact_score": str(expected["kinship"]),
        }
    balanced = balanced_pedigree()
    control_records = run_plink(balanced)
    compare_all(balanced, control_records)
    return {
        "python": platform.python_version(), "tool": VERSION, "seed": seed,
        "loci": loci, "samples": len(samples),
        "synthetic_vcf_sha256": hashlib.sha256(vcf_text(samples).encode()).hexdigest(),
        "generating_frequencies": {"main_founders": 0.3, "low": 0.1, "high": 0.9},
        "all_pairs_checked": len(records), "max_abs_tool_reference_error": round(maximum_error, 9),
        "selected_pairs": selected,
        "balanced_256_locus_control": {
            "parent_child": control_records[tuple(sorted(("parentA", "childA")))]["kinship"],
            "full_siblings": control_records[tuple(sorted(("childA", "childB")))]["kinship"],
            "unrelated_parents": control_records[("parentA", "parentB")]["kinship"],
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--loci", type=int, default=LOCI)
    parser.add_argument("--write-vcf", type=Path, help="also save the generated synthetic input")
    parser.add_argument("--check-sample", action="store_true")
    args = parser.parse_args()
    if args.check_sample and (args.seed != SEED or args.loci != LOCI):
        parser.error("--check-sample requires the default seed and locus count")
    try:
        result = summary(args.seed, args.loci)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    if args.write_vcf:
        args.write_vcf.write_text(vcf_text(synthetic(args.seed, args.loci)), encoding="utf-8")
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.check_sample:
        if text != (HERE / "sample_output.json").read_text(encoding="utf-8"):
            raise AssertionError("Output differs from the checked reference")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
