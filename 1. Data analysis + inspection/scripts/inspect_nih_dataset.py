#!/usr/bin/env python3
"""
Safely inspect the NIH ChestX-ray14 dataset area without opening image files.

This script deliberately avoids directories whose names clearly indicate test,
testing, heldout, or held-out data. It only counts image filenames and reports
likely metadata/split files so you can choose the correct arguments for the
main preprocessing script.
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
from collections import Counter, defaultdict
from pathlib import Path


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
TEXT_EXTENSIONS = {".csv", ".txt", ".lst"}
BLOCKED_DIR_MARKERS = ("test", "testing", "heldout", "held-out")


def is_blocked_dir(path: Path) -> bool:
    """Return True for directories that should not be entered."""
    name = path.name.lower().replace("_", "-")
    return any(marker in name for marker in BLOCKED_DIR_MARKERS)


def read_csv_header(path: Path) -> list[str]:
    """Read only the first row of a CSV file."""
    try:
        with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
            return next(csv.reader(f), [])
    except Exception:
        return []


def score_metadata_csv(path: Path, header: list[str]) -> int:
    text = " ".join([path.name, *header]).lower()
    score = 0
    for token in ("image index", "finding labels", "patient id", "view position"):
        if token in text:
            score += 3
    for token in ("data_entry", "metadata", "label", "nih"):
        if token in text:
            score += 1
    if "test" in path.name.lower():
        score -= 5
    return score


def looks_like_training_split(path: Path) -> bool:
    name = path.name.lower()
    return (
        ("train" in name or "valid" in name or "val" in name)
        and "test" not in name
        and path.suffix.lower() in TEXT_EXTENSIONS
    )


def looks_like_test_exclusion(path: Path) -> bool:
    name = path.name.lower()
    return "test" in name and path.suffix.lower() in TEXT_EXTENSIONS


def inspect_dataset(dataset_root: Path, max_depth: int, top_n: int) -> None:
    if not dataset_root.exists():
        raise SystemExit(f"Dataset root does not exist: {dataset_root}")

    image_counts_by_dir: Counter[Path] = Counter()
    image_ext_counts: Counter[str] = Counter()
    csv_candidates: list[tuple[int, Path, list[str]]] = []
    train_split_candidates: list[Path] = []
    test_exclusion_candidates: list[Path] = []
    skipped_dirs: list[Path] = []
    visited_dirs = 0

    root_depth = len(dataset_root.resolve().parts)

    for current, dirnames, filenames in os.walk(dataset_root, topdown=True, followlinks=False):
        current_path = Path(current)
        rel_depth = len(current_path.resolve().parts) - root_depth
        visited_dirs += 1

        kept_dirs = []
        for dirname in sorted(dirnames):
            child = current_path / dirname
            if is_blocked_dir(child):
                skipped_dirs.append(child)
            elif rel_depth < max_depth:
                kept_dirs.append(dirname)
        dirnames[:] = kept_dirs

        for filename in filenames:
            path = current_path / filename
            suffix = path.suffix.lower()
            if suffix in IMAGE_EXTENSIONS:
                image_counts_by_dir[current_path] += 1
                image_ext_counts[suffix] += 1
            elif suffix == ".csv":
                header = read_csv_header(path)
                csv_candidates.append((score_metadata_csv(path, header), path, header))
                if looks_like_training_split(path):
                    train_split_candidates.append(path)
                if looks_like_test_exclusion(path):
                    test_exclusion_candidates.append(path)
            elif suffix in TEXT_EXTENSIONS:
                if looks_like_training_split(path):
                    train_split_candidates.append(path)
                if looks_like_test_exclusion(path):
                    test_exclusion_candidates.append(path)

    print("\nSafe NIH dataset inspection")
    print("=" * 32)
    print(f"Dataset root: {dataset_root}")
    print(f"Visited directories: {visited_dirs}")
    print(f"Skipped test-like directories: {len(skipped_dirs)}")
    for path in skipped_dirs[:top_n]:
        print(f"  skipped: {path}")
    if len(skipped_dirs) > top_n:
        print(f"  ... {len(skipped_dirs) - top_n} more skipped directories")

    print("\nImage file extensions counted without opening images:")
    if image_ext_counts:
        for ext, count in image_ext_counts.most_common():
            print(f"  {ext}: {count}")
    else:
        print("  No PNG/JPG/JPEG files found outside skipped directories.")

    print(f"\nTop directories containing images (showing up to {top_n}):")
    if image_counts_by_dir:
        for directory, count in image_counts_by_dir.most_common(top_n):
            print(f"  {count:8d}  {directory}")
    else:
        print("  None found.")

    print(f"\nLikely NIH metadata CSV files (showing up to {top_n}):")
    ranked_csvs = sorted(csv_candidates, key=lambda item: (-item[0], str(item[1])))
    if ranked_csvs:
        for score, path, header in ranked_csvs[:top_n]:
            header_preview = ", ".join(header[:8]) if header else "header not readable"
            print(f"  score={score:2d}  {path}")
            print(f"             header: {header_preview}")
    else:
        print("  None found.")

    print(f"\nLikely training/validation split or list files (showing up to {top_n}):")
    for path in sorted(set(train_split_candidates))[:top_n]:
        print(f"  {path}")
    if not train_split_candidates:
        print("  None found.")

    print(f"\nLikely test filename exclusion files (filenames only; do not analyse labels):")
    for path in sorted(set(test_exclusion_candidates))[:top_n]:
        print(f"  {path}")
    if not test_exclusion_candidates:
        print("  None found.")

    print("\nSuggested next step:")
    print("  Use the best metadata CSV and any train/validation/test filename list paths")
    print("  as arguments to scripts/preprocess_and_analyse.py.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Safely inspect NIH dataset folders.")
    parser.add_argument("--dataset-root", type=Path, default=Path("/datasets/pgohil"))
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=args.log_level, format="%(levelname)s: %(message)s")
    inspect_dataset(args.dataset_root, args.max_depth, args.top_n)


if __name__ == "__main__":
    main()
