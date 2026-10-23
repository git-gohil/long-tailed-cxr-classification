#!/usr/bin/env python3
"""Aggregate test mAP with patient-cluster bootstrap confidence intervals.

This is intentionally separate from aggregate_results.py. It never reads from
or writes to analysis/results; its default output directory is
analysis/results_patient_bootstrap.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.metrics import average_precision, precision_recall_f1
from common.nih_longtail import PATHOLOGIES, write_json  # noqa: E402
from common.hyperparameter_sweep import validate_final_run


METHODS = ["bce_baseline", "focal_loss", "db_loss", "sqrt_sampling", "crt", "balanced_mixup"]
GROUPS = ("head", "medium", "tail")
DISPLAY_NAMES = {
    "bce_baseline": "BCE",
    "focal_loss": "Focal Loss",
    "db_loss": "DB Loss",
    "sqrt_sampling": "Sqrt Sampling",
    "crt": "cRT",
    "balanced_mixup": "Balanced MixUp",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Patient-cluster bootstrap aggregation for NIH test predictions.")
    parser.add_argument("--project-root", type=Path, default=Path("/home-mscluster/pgohil/cxr-research"))
    parser.add_argument("--runs-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--class-groups", type=Path, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2024])
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def safe_mean(values) -> float:
    arr = np.asarray(list(values), dtype=float)
    finite = np.isfinite(arr)
    return float(arr[finite].mean()) if finite.any() else float("nan")




def map_metrics(labels: np.ndarray, probs: np.ndarray, class_to_group: dict[str, str]) -> tuple[float, dict[str, float]]:
    per_class = {pathology: average_precision(labels[:, i], probs[:, i]) for i, pathology in enumerate(PATHOLOGIES)}
    group_map = {
        group: safe_mean(per_class[p] for p in PATHOLOGIES if class_to_group[p] == group)
        for group in GROUPS
    }
    return safe_mean(per_class.values()), group_map


def load_array(path: Path) -> np.ndarray:
    if not path.exists():
        raise SystemExit(f"Missing required evaluation output: {path}")
    return np.load(path, allow_pickle=False)


def load_predictions(runs_dir: Path, methods: list[str], seeds: list[int], class_groups_path: Path | None = None) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    data: dict[str, dict[int, np.ndarray]] = {}
    reference_labels = reference_image_ids = reference_patient_ids = None

    for method in methods:
        data[method] = {}
        for seed in seeds:
            validate_final_run(runs_dir / method / f"seed_{seed}", method, seed, class_groups_path)
            eval_dir = runs_dir / method / f"seed_{seed}" / "eval_test"
            probs = load_array(eval_dir / "predictions.npy")
            labels = load_array(eval_dir / "labels.npy")
            image_ids = load_array(eval_dir / "image_ids.npy").astype(str)
            patient_ids = load_array(eval_dir / "patient_ids.npy").astype(str)

            if probs.shape != labels.shape or probs.ndim != 2 or probs.shape[1] != len(PATHOLOGIES):
                raise SystemExit(f"Unexpected prediction/label shape in {eval_dir}: {probs.shape} vs {labels.shape}")
            if not len(labels) or not np.isfinite(probs).all() or not np.isin(labels, [0, 1]).all() or np.any((probs < 0) | (probs > 1)):
                raise ValueError(f"Nonfinite/invalid labels or probabilities in {eval_dir}")
            if image_ids.ndim != 1 or patient_ids.ndim != 1 or len(np.unique(image_ids)) != len(image_ids):
                raise ValueError(f"Duplicate or invalid image identifiers in {eval_dir}")
            if len(image_ids) != len(labels) or len(patient_ids) != len(labels):
                raise SystemExit(f"Image/patient ID length mismatch in {eval_dir}")
            if np.any(np.isin(patient_ids, ["", "nan", "None"])):
                raise SystemExit(f"Missing patient IDs in {eval_dir}; patient bootstrap cannot continue.")

            if reference_labels is None:
                reference_labels = labels
                reference_image_ids = image_ids
                reference_patient_ids = patient_ids
            elif not (
                np.array_equal(labels, reference_labels)
                and np.array_equal(image_ids, reference_image_ids)
                and np.array_equal(patient_ids, reference_patient_ids)
            ):
                raise SystemExit(
                    f"Test rows are not identically aligned in {eval_dir}. "
                    "All methods and seeds must use the same labels, image order, and patient IDs."
                )
            data[method][seed] = probs

    assert reference_labels is not None and reference_image_ids is not None and reference_patient_ids is not None
    return data, reference_labels, reference_image_ids, reference_patient_ids


def indices_from_patient_draw(patient_codes: np.ndarray, patient_draw: np.ndarray, num_patients: int) -> np.ndarray:
    cluster_counts = np.bincount(patient_draw, minlength=num_patients)
    return np.repeat(np.arange(len(patient_codes)), cluster_counts[patient_codes])


def percentile_interval(values: np.ndarray) -> tuple[float, float]:
    return float(np.nanpercentile(values, 2.5)), float(np.nanpercentile(values, 97.5))


def write_output_readme(output_dir: Path, resamples: int, seed: int, patient_count: int) -> None:
    text = f"""PATIENT-LEVEL BOOTSTRAP RESULTS

These files are separate from the old image-level bootstrap in analysis/results.
Do not copy these files into analysis/results and do not rename them as the old results.

Method:
- Bootstrap unit: unique patient, preserving all images belonging to each sampled patient.
- Shared patient draws are used for every method and seed, enabling paired comparisons.
- Point estimate: mean test mAP across the configured training seeds.
- Confidence interval: percentile interval of mean mAP across those same seed models after patient resampling.
- Training-seed variability is reported separately as the standard deviation across seeds.
- Resamples: {resamples}
- Bootstrap RNG seed: {seed}
- Unique test patients: {patient_count}

Files:
- patient_bootstrap_summary.csv: overall mAP mean/SD across seeds and patient-bootstrap 95% CI.
- patient_bootstrap_group_summary.csv: head/medium/tail mAP and patient-bootstrap 95% CI.
- patient_bootstrap_seed_metrics.csv: recomputed mAP for every method and training seed.
- patient_bootstrap_vs_bce.csv: paired mAP differences versus BCE with patient-bootstrap 95% CI.
- patient_bootstrap_distributions.npz: raw bootstrap distributions for audit or new plots.
- patient_bootstrap_details.json: configuration, method, counts, and output descriptions.
- patient_bootstrap_map_ci.png: overall mean test mAP with patient-bootstrap intervals.
"""
    (output_dir / "PATIENT_BOOTSTRAP_README.txt").write_text(text, encoding="utf-8")


def write_plot(output_dir: Path, summary: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ordered = summary.sort_values("mean_mAP_across_seeds", ascending=False).reset_index(drop=True)
    x = np.arange(len(ordered))
    centers = ordered["mean_mAP_across_seeds"].to_numpy(dtype=float)
    lower = ordered["patient_bootstrap_CI_lower"].to_numpy(dtype=float)
    upper = ordered["patient_bootstrap_CI_upper"].to_numpy(dtype=float)
    plt.figure(figsize=(9, 5))
    plt.vlines(x, lower, upper, color="#333333", linewidth=1.5)
    plt.scatter(x, centers, color="#4c78a8", s=55, zorder=3)
    plt.xticks(x, [DISPLAY_NAMES.get(m, m) for m in ordered["method"]], rotation=25, ha="right")
    plt.ylabel("Mean test mAP across seeds")
    plt.title("Patient-Cluster Bootstrap 95% Confidence Intervals")
    plt.tight_layout()
    plt.savefig(output_dir / "patient_bootstrap_map_ci.png", dpi=200)
    plt.close()


def self_test() -> None:
    patient_codes = np.array([0, 0, 1])
    idx = indices_from_patient_draw(patient_codes, np.array([0, 0]), num_patients=2)
    assert len(idx) == 4 and np.count_nonzero(idx == 0) == 2 and np.count_nonzero(idx == 1) == 2
    labels = np.zeros((3, len(PATHOLOGIES)), dtype=int)
    probs = np.zeros_like(labels, dtype=float)
    labels[[0, 2], 0] = 1
    probs[:, 0] = [0.9, 0.1, 0.8]
    assert math.isclose(average_precision(labels[:, 0], probs[:, 0]), 1.0)
    print("Patient-bootstrap self-test passed.")


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    if args.bootstrap_resamples <= 0:
        raise SystemExit("--bootstrap-resamples must be greater than zero")

    runs_dir = args.runs_dir or args.project_root / "runs"
    output_dir = args.output_dir or args.project_root / "analysis" / "results_patient_bootstrap"
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"Use a new, empty patient-bootstrap output directory: {output_dir}")
    legacy_dir = args.project_root / "analysis" / "results"
    if output_dir.resolve() == legacy_dir.resolve():
        raise SystemExit(
            f"Refusing to write patient-bootstrap files into the old results directory: {legacy_dir}. "
            "Use analysis/results_patient_bootstrap instead."
        )
    class_groups_path = args.class_groups or args.project_root / "1. Data analysis + inspection" / "results" / "data" / "class_groups.csv"
    groups_df = pd.read_csv(class_groups_path)
    class_to_group = dict(zip(groups_df["pathology"], groups_df["group"]))
    missing_groups = sorted(set(PATHOLOGIES) - set(class_to_group))
    invalid_groups = sorted(set(class_to_group.values()) - set(GROUPS))
    if missing_groups or invalid_groups:
        raise SystemExit(f"Invalid class group file: missing={missing_groups}, invalid groups={invalid_groups}")

    data, labels, image_ids, patient_ids = load_predictions(runs_dir, args.methods, args.seeds, class_groups_path)
    unique_patients, patient_codes = np.unique(patient_ids, return_inverse=True)
    num_patients = len(unique_patients)
    rng = np.random.default_rng(args.bootstrap_seed)

    seed_rows = []
    point_metrics: dict[str, dict[int, tuple[float, dict[str, float]]]] = {}
    for method in args.methods:
        point_metrics[method] = {}
        for seed in args.seeds:
            overall, group_map = map_metrics(labels, data[method][seed], class_to_group)
            point_metrics[method][seed] = (overall, group_map)
            seed_rows.append({"method": method, "seed": seed, "mAP": overall, **{f"{g}_mAP": group_map[g] for g in GROUPS}})

    boot_overall = {method: np.empty(args.bootstrap_resamples, dtype=float) for method in args.methods}
    boot_groups = {
        method: {group: np.empty(args.bootstrap_resamples, dtype=float) for group in GROUPS}
        for method in args.methods
    }
    progress_every = max(1, args.bootstrap_resamples // 10)
    for resample in range(args.bootstrap_resamples):
        patient_draw = rng.integers(0, num_patients, size=num_patients)
        idx = indices_from_patient_draw(patient_codes, patient_draw, num_patients)
        for method in args.methods:
            seed_overall = []
            seed_groups = {group: [] for group in GROUPS}
            for seed in args.seeds:
                overall, group_map = map_metrics(labels[idx], data[method][seed][idx], class_to_group)
                seed_overall.append(overall)
                for group in GROUPS:
                    seed_groups[group].append(group_map[group])
            boot_overall[method][resample] = safe_mean(seed_overall)
            for group in GROUPS:
                boot_groups[method][group][resample] = safe_mean(seed_groups[group])
        if (resample + 1) % progress_every == 0 or resample + 1 == args.bootstrap_resamples:
            print(f"Completed {resample + 1}/{args.bootstrap_resamples} patient bootstrap resamples.", flush=True)

    summary_rows = []
    group_rows = []
    for method in args.methods:
        seed_maps = [point_metrics[method][seed][0] for seed in args.seeds]
        lower, upper = percentile_interval(boot_overall[method])
        summary_rows.append({
            "method": method,
            "mean_mAP_across_seeds": safe_mean(seed_maps),
            "std_mAP_across_seeds": float(np.std(seed_maps, ddof=1)) if len(seed_maps) > 1 else float("nan"),
            "patient_bootstrap_CI_lower": lower,
            "patient_bootstrap_CI_upper": upper,
            "bootstrap_unit": "patient",
            "bootstrap_resamples": args.bootstrap_resamples,
            "training_seeds": "|".join(map(str, args.seeds)),
        })
        for group in GROUPS:
            seed_group_maps = [point_metrics[method][seed][1][group] for seed in args.seeds]
            group_lower, group_upper = percentile_interval(boot_groups[method][group])
            group_rows.append({
                "method": method,
                "group": group,
                "mean_group_mAP_across_seeds": safe_mean(seed_group_maps),
                "std_group_mAP_across_seeds": float(np.std(seed_group_maps, ddof=1)) if len(seed_group_maps) > 1 else float("nan"),
                "patient_bootstrap_CI_lower": group_lower,
                "patient_bootstrap_CI_upper": group_upper,
            })

    comparison_rows = []
    if "bce_baseline" in args.methods:
        baseline_point = next(row["mean_mAP_across_seeds"] for row in summary_rows if row["method"] == "bce_baseline")
        for row in summary_rows:
            if row["method"] == "bce_baseline":
                continue
            differences = boot_overall[row["method"]] - boot_overall["bce_baseline"]
            lower, upper = percentile_interval(differences)
            comparison_rows.append({
                "method": row["method"],
                "reference_method": "bce_baseline",
                "mean_mAP_difference": row["mean_mAP_across_seeds"] - baseline_point,
                "patient_bootstrap_difference_CI_lower": lower,
                "patient_bootstrap_difference_CI_upper": upper,
                "CI_excludes_zero": bool(lower > 0 or upper < 0),
            })

    output_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(summary_rows).sort_values("mean_mAP_across_seeds", ascending=False)
    summary.to_csv(output_dir / "patient_bootstrap_summary.csv", index=False)
    pd.DataFrame(group_rows).to_csv(output_dir / "patient_bootstrap_group_summary.csv", index=False)
    pd.DataFrame(seed_rows).to_csv(output_dir / "patient_bootstrap_seed_metrics.csv", index=False)
    pd.DataFrame(comparison_rows).to_csv(output_dir / "patient_bootstrap_vs_bce.csv", index=False)
    np.savez_compressed(
        output_dir / "patient_bootstrap_distributions.npz",
        methods=np.asarray(args.methods),
        groups=np.asarray(GROUPS),
        overall_mAP=np.stack([boot_overall[m] for m in args.methods]),
        group_mAP=np.stack([[boot_groups[m][g] for g in GROUPS] for m in args.methods]),
    )
    details = {
        "analysis_name": "patient_cluster_bootstrap",
        "old_results_directory_touched": False,
        "runs_dir": str(runs_dir),
        "output_dir": str(output_dir),
        "methods": args.methods,
        "seeds": args.seeds,
        "bootstrap_resamples": args.bootstrap_resamples,
        "bootstrap_seed": args.bootstrap_seed,
        "bootstrap_unit": "unique patient with all of that patient's images retained",
        "shared_draws_across_methods": True,
        "point_estimate": "mean test mAP across training seeds",
        "interval_estimand": "patient-sampling uncertainty conditional on the configured trained seed models",
        "test_images": int(len(image_ids)),
        "unique_test_patients": int(num_patients),
        "outputs": [
            "patient_bootstrap_summary.csv",
            "patient_bootstrap_group_summary.csv",
            "patient_bootstrap_seed_metrics.csv",
            "patient_bootstrap_vs_bce.csv",
            "patient_bootstrap_distributions.npz",
            "patient_bootstrap_map_ci.png",
            "PATIENT_BOOTSTRAP_README.txt",
        ],
    }
    write_json(details, output_dir / "patient_bootstrap_details.json")
    write_output_readme(output_dir, args.bootstrap_resamples, args.bootstrap_seed, num_patients)
    write_plot(output_dir, summary)
    print(f"Patient-bootstrap aggregation complete: {output_dir}")
    print("The old analysis/results directory was not read or modified.")


if __name__ == "__main__":
    main()
