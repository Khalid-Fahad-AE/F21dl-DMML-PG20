"""Integrity checks for the shared split, data/splits.csv (run: python tests/test_splits.py).

The whole project relies on this file: every model is trained and tested on it. These checks confirm
the properties the report claims, without needing the images:
  * 18,133 images after de-duplication, split 12,696 / 2,718 / 2,719 (train / val / test);
  * no leaf ID and no duplicate group appears in more than one split (no leakage between splits);
  * no exact duplicate (same MD5) is left;
  * the split is stratified: each class's share in every split is within 2 points of its overall share.
Uses only the standard library, so it runs before any requirements are installed.
"""
import csv
from collections import Counter, defaultdict
from pathlib import Path

SPLITS_CSV = Path(__file__).resolve().parents[1] / "data" / "splits.csv"
EXPECTED = {"train": 12696, "val": 2718, "test": 2719}


def load():
    with open(SPLITS_CSV, newline="") as f:
        return list(csv.DictReader(f))


def check_sizes(rows):
    counts = Counter(r["split"] for r in rows)
    assert len(rows) == sum(EXPECTED.values()), f"expected 18,133 rows, got {len(rows)}"
    assert dict(counts) == EXPECTED, f"split sizes {dict(counts)} != {EXPECTED}"


def check_no_leakage(rows, key):
    splits_of = defaultdict(set)
    for r in rows:
        splits_of[r[key]].add(r["split"])
    leaked = {k: s for k, s in splits_of.items() if len(s) > 1}
    assert not leaked, f"{len(leaked)} {key} values appear in several splits, e.g. {next(iter(leaked.items()))}"


def check_no_exact_duplicates(rows):
    dupes = [m for m, n in Counter(r["md5"] for r in rows).items() if n > 1]
    assert not dupes, f"{len(dupes)} MD5 hashes occur more than once"


def check_stratified(rows, tolerance=0.02):
    overall = Counter(r["label"] for r in rows)
    for split in EXPECTED:
        part = [r for r in rows if r["split"] == split]
        here = Counter(r["label"] for r in part)
        for label, n in overall.items():
            diff = abs(here[label] / len(part) - n / len(rows))
            assert diff <= tolerance, f"class {label} share in {split} is off by {diff:.3f}"


def main():
    rows = load()
    checks = [
        ("split sizes", lambda: check_sizes(rows)),
        ("no leaf in two splits", lambda: check_no_leakage(rows, "leaf_id")),
        ("no duplicate group in two splits", lambda: check_no_leakage(rows, "group")),
        ("no exact duplicates", lambda: check_no_exact_duplicates(rows)),
        ("stratified by class", lambda: check_stratified(rows)),
    ]
    for name, fn in checks:
        fn()
        print(f"ok  {name}")
    print("splits.csv: all checks passed")


if __name__ == "__main__":
    main()
