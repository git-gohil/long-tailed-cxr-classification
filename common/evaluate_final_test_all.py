#!/usr/bin/env python3
"""Evaluate all seed runs through the same validated evaluator used by sweeps."""
import argparse
from pathlib import Path
import subprocess
import sys

METHODS = ["bce_baseline", "focal_loss", "db_loss", "sqrt_sampling", "crt", "balanced_mixup"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--runs-dir", type=Path, default=None)
    parser.add_argument("--dataset-root", type=Path, default=Path("/datasets/pgohil"))
    parser.add_argument("--metadata-csv", type=Path, default=None)
    parser.add_argument("--test-list", type=Path, default=None)
    parser.add_argument("--methods", nargs="+", choices=["all", *METHODS], default=["all"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 2024])
    parser.add_argument("--device", choices=["cuda", "cpu", "auto"], default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    runs = args.runs_dir or args.project_root / "runs"
    methods = METHODS if args.methods == ["all"] else args.methods
    if "all" in methods:
        parser.error("Use --methods all alone")
    directories = [(method, seed, runs / method / f"seed_{seed}") for method in methods for seed in args.seeds]
    for _, _, directory in directories:
        if not (directory / "checkpoint_best.pt").is_file():
            raise ValueError(f"Missing checkpoint: {directory}")
    for method, seed, directory in directories:
        for split in ("val", "test"):
            subprocess.run([sys.executable, str(args.project_root / "evaluate.py"),
                            "--project-root", str(args.project_root), "--dataset-root", str(args.dataset_root),
                            "--metadata-csv", str(args.metadata_csv or args.dataset_root / "Data_Entry_2017.csv"),
                            "--test-list", str(args.test_list or args.dataset_root / "test_list.txt"),
                            "--method", method, "--seed", str(seed), "--run-dir", str(directory),
                            "--split", split, "--device", args.device, "--batch-size", str(args.batch_size),
                            "--num-workers", str(args.num_workers), "--skip-existing"], check=True)


if __name__ == "__main__":
    main()
