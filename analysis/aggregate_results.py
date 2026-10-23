#!/usr/bin/env python3
"""Aggregate seed-repeated CXR results and bootstrap test-set uncertainty."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.metrics import average_precision, precision_recall_f1
from common.nih_longtail import PATHOLOGIES  # noqa: E402
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
    parser = argparse.ArgumentParser(description="Aggregate multi-seed CXR experiment results.")
    parser.add_argument("--project-root", type=Path, default=Path("/home-mscluster/pgohil/cxr-research"))
    parser.add_argument("--runs-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--class-groups", type=Path, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2024])
    parser.add_argument("--methods", nargs="+", default=METHODS)
    parser.add_argument("--bootstrap-mode", choices=["none", "representative", "pooled"], default="none",
                        help="Default: summary tables only. Use the separate patient-bootstrap script for final CIs.")
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_mean(values) -> float:
    arr = np.asarray(list(values), dtype=float)
    finite = np.isfinite(arr)
    return float(arr[finite].mean()) if finite.any() else float("nan")


def safe_std(values) -> float:
    arr = np.asarray(list(values), dtype=float)
    finite = np.isfinite(arr)
    return float(arr[finite].std(ddof=1)) if finite.sum() > 1 else float("nan")




def map_by_group(labels: np.ndarray, probs: np.ndarray, class_to_group: dict[str, str]) -> tuple[float, dict[str, float]]:
    per_class = {p: average_precision(labels[:, i], probs[:, i]) for i, p in enumerate(PATHOLOGIES)}
    group_map = {}
    for group in GROUPS:
        classes = [p for p in PATHOLOGIES if class_to_group.get(p) == group]
        group_map[group] = safe_mean(per_class[p] for p in classes)
    return safe_mean(per_class.values()), group_map


def bootstrap_ci(labels: np.ndarray, probs: np.ndarray, class_to_group: dict[str, str], resamples: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    n = len(labels)
    mAPs = []
    group_values = {group: [] for group in GROUPS}
    for _ in range(resamples):
        idx = rng.integers(0, n, size=n)
        mAP, group_map = map_by_group(labels[idx], probs[idx], class_to_group)
        mAPs.append(mAP)
        for group in GROUPS:
            group_values[group].append(group_map[group])
    out = {
        "mAP": {
            "lower": float(np.nanpercentile(mAPs, 2.5)),
            "upper": float(np.nanpercentile(mAPs, 97.5)),
        },
        "group_mAP": {},
    }
    for group in GROUPS:
        out["group_mAP"][group] = {
            "lower": float(np.nanpercentile(group_values[group], 2.5)),
            "upper": float(np.nanpercentile(group_values[group], 97.5)),
        }
    return out


def wilcoxon_exact(x: list[float], y: list[float]) -> float:
    diffs = [a - b for a, b in zip(x, y) if math.isfinite(a) and math.isfinite(b) and a != b]
    n = len(diffs)
    if n == 0:
        return 1.0
    abs_diffs = np.asarray([abs(d) for d in diffs], dtype=float)
    order = np.argsort(abs_diffs, kind="mergesort")
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i + 1
        while j < n and abs_diffs[order[j]] == abs_diffs[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    observed = min(ranks[np.asarray(diffs) > 0].sum(), ranks[np.asarray(diffs) < 0].sum())
    possible = []
    for signs in itertools.product([-1, 1], repeat=n):
        signed = ranks * np.asarray(signs)
        possible.append(min(ranks[signed > 0].sum(), ranks[signed < 0].sum()))
    return float(np.mean(np.asarray(possible) <= observed + 1e-12))


def collect_method(runs_dir: Path, method: str, seeds: list[int], class_groups_path: Path | None = None,
                   reference_test_rows: dict | None = None) -> list[dict]:
    rows = []
    for seed in seeds:
        run_dir = runs_dir / method / f"seed_{seed}"
        reference_test_rows = validate_final_run(run_dir, method, seed, class_groups_path, reference_test_rows)
        metrics_path = run_dir / "eval_test" / "metrics.json"
        if not metrics_path.exists():
            raise SystemExit(f"Missing {metrics_path}")
        metrics = load_json(metrics_path)
        compute_path = run_dir / "compute_log.json"
        compute = load_json(compute_path) if compute_path.exists() else {}
        tuned = load_json(run_dir / "eval_test/metrics_tuned_threshold.json")
        rows.append({"method": method, "seed": seed, "run_dir": run_dir, "metrics": metrics, "tuned": tuned,
                     "compute": compute, "test_rows": reference_test_rows})
    return rows


def load_bootstrap_arrays(method_rows: list[dict], mode: str) -> tuple[np.ndarray, np.ndarray]:
    selected = method_rows if mode == "pooled" else [method_rows[0]]
    labels, probs = [], []
    for row in selected:
        eval_dir = row["run_dir"] / "eval_test"
        labels.append(np.load(eval_dir / "labels.npy"))
        probs.append(np.load(eval_dir / "predictions.npy"))
    return np.concatenate(labels, axis=0), np.concatenate(probs, axis=0)


def write_tables(output_dir: Path, summary_rows: list[dict], all_rows: dict[str, list[dict]], details: dict) -> None:
    seed_rows = []
    group_rows = []
    per_class_rows = []
    compute_rows = []
    threshold_rows = []
    significance_rows = []

    for method, method_rows in all_rows.items():
        for row in method_rows:
            metrics = row["metrics"]
            seed_rows.append({
                "method": method,
                "seed": row["seed"],
                "mAP": metrics["mAP"],
                "macro_f1_at_0.5": metrics["macro_f1_at_0.5"],
                "macro_f1_tuned": row["tuned"]["macro_f1"],
                "macro_roc_auc": safe_mean(metrics["per_class_roc_auc"].values()),
                "gpu_hours": float(row["compute"].get("gpu_hours", 0.0) or 0.0),
            })
            compute_rows.append({"method": method, "seed": row["seed"], **row["compute"]})
            for group in GROUPS:
                group_rows.append({
                    "method": method,
                    "seed": row["seed"],
                    "group": group,
                    "mAP": metrics["group_mAP"][group],
                    "f1_at_0.5": metrics["group_f1_at_0.5"][group],
                    "f1_tuned": row["tuned"]["group_f1"][group],
                })
            for pathology in PATHOLOGIES:
                prf = metrics["per_class_precision_recall_f1_at_0.5"][pathology]
                per_class_rows.append({
                    "method": method,
                    "seed": row["seed"],
                    "pathology": pathology,
                    "ap": metrics["per_class_ap"][pathology],
                    "roc_auc": metrics["per_class_roc_auc"][pathology],
                    "precision_at_0.5": prf["p"],
                    "recall_at_0.5": prf["r"],
                    "f1_at_0.5": prf["f1"],
                    "f1_tuned": row["tuned"]["per_class_precision_recall_f1"][pathology]["f1"],
                })
            threshold_path = row["run_dir"] / "eval_val" / "tuned_thresholds.json"
            if threshold_path.exists():
                thresholds = load_json(threshold_path)
                for pathology, threshold in thresholds.items():
                    threshold_rows.append({"method": method, "seed": row["seed"], "pathology": pathology, "threshold": threshold})

    for item in details["pairwise_top_methods"]:
        significance_rows.append(item)
    for row in summary_rows:
        if row["method"] != "bce_baseline":
            significance_rows.append({"method_a": row["method"], "method_b": "bce_baseline", "p_value": row["p_value_vs_baseline"]})

    pd.DataFrame(seed_rows).to_csv(output_dir / "seed_level_metrics.csv", index=False)
    pd.DataFrame(group_rows).to_csv(output_dir / "group_metrics_by_seed.csv", index=False)
    pd.DataFrame(per_class_rows).to_csv(output_dir / "per_class_metrics_by_seed.csv", index=False)
    pd.DataFrame(compute_rows).to_csv(output_dir / "compute_by_seed.csv", index=False)
    pd.DataFrame(threshold_rows).to_csv(output_dir / "val_tuned_thresholds_by_seed.csv", index=False)
    pd.DataFrame(significance_rows).to_csv(output_dir / "significance_tests.csv", index=False)

    per_class_summary = (
        pd.DataFrame(per_class_rows)
        .groupby(["method", "pathology"], as_index=False)
        .agg(
            mean_ap=("ap", "mean"),
            std_ap=("ap", "std"),
            mean_roc_auc=("roc_auc", "mean"),
            mean_f1_at_0_5=("f1_at_0.5", "mean"),
            std_f1_at_0_5=("f1_at_0.5", "std"),
            mean_f1_tuned=("f1_tuned", "mean"),
            std_f1_tuned=("f1_tuned", "std"),
        )
    )
    per_class_summary.to_csv(output_dir / "per_class_metrics_summary.csv", index=False)


def plot_method_map(output_dir: Path, summary: pd.DataFrame) -> None:
    df = summary.sort_values("mean_mAP", ascending=False)
    x = np.arange(len(df))
    plt.figure(figsize=(9, 5))
    plt.bar(x, df["mean_mAP"], yerr=df["std_mAP"], capsize=4, color="#4c78a8")
    plt.xticks(x, [DISPLAY_NAMES.get(m, m) for m in df["method"]], rotation=25, ha="right")
    plt.ylabel("Test mAP")
    plt.title("Mean Test mAP Across Seeds")
    plt.tight_layout()
    plt.savefig(output_dir / "method_map_mean_std.png", dpi=200)
    plt.close()


def plot_group_map(output_dir: Path, summary: pd.DataFrame) -> None:
    df = summary.set_index("method").loc[[m for m in METHODS if m in set(summary["method"])]]
    x = np.arange(len(df))
    width = 0.25
    colors = {"head": "#4c78a8", "medium": "#59a14f", "tail": "#e15759"}
    plt.figure(figsize=(10, 5))
    for i, group in enumerate(GROUPS):
        plt.bar(x + (i - 1) * width, df[f"{group}_mAP"], width, label=group, color=colors[group])
    plt.xticks(x, [DISPLAY_NAMES.get(m, m) for m in df.index], rotation=25, ha="right")
    plt.ylabel("Test group mAP")
    plt.title("Head, Medium, and Tail mAP by Method")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "group_map_by_method.png", dpi=200)
    plt.close()


def plot_per_class_heatmap(output_dir: Path, per_class_summary: pd.DataFrame) -> None:
    pivot = per_class_summary.pivot(index="pathology", columns="method", values="mean_ap").reindex(PATHOLOGIES)
    pivot = pivot[[m for m in METHODS if m in pivot.columns]]
    plt.figure(figsize=(10, 7))
    image = plt.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="viridis", vmin=0, vmax=np.nanmax(pivot.to_numpy(dtype=float)))
    plt.colorbar(image, label="Mean AP")
    plt.yticks(np.arange(len(pivot.index)), pivot.index)
    plt.xticks(np.arange(len(pivot.columns)), [DISPLAY_NAMES.get(m, m) for m in pivot.columns], rotation=30, ha="right")
    plt.title("Per-Class AP by Method")
    plt.tight_layout()
    plt.savefig(output_dir / "per_class_ap_heatmap.png", dpi=200)
    plt.close()


def plot_bootstrap_ci(output_dir: Path, summary: pd.DataFrame) -> None:
    df = summary.sort_values("mean_mAP", ascending=False)
    x = np.arange(len(df))
    plt.figure(figsize=(9, 5))
    plt.vlines(x, df["bootstrap_CI_lower"], df["bootstrap_CI_upper"], color="#222222")
    plt.scatter(x, df["bootstrap_reference_mAP"], color="#222222")
    plt.xticks(x, [DISPLAY_NAMES.get(m, m) for m in df["method"]], rotation=25, ha="right")
    plt.ylabel("Test mAP")
    plt.title("Bootstrap 95% CI for Test mAP")
    plt.tight_layout()
    plt.savefig(output_dir / "bootstrap_map_ci.png", dpi=200)
    plt.close()


def plot_seed_variability(output_dir: Path, seed_df: pd.DataFrame) -> None:
    methods = [m for m in METHODS if m in set(seed_df["method"])]
    plt.figure(figsize=(9, 5))
    for method in methods:
        rows = seed_df[seed_df["method"] == method].sort_values("seed")
        plt.plot(rows["seed"].astype(str), rows["mAP"], marker="o", label=DISPLAY_NAMES.get(method, method))
    plt.xlabel("Seed")
    plt.ylabel("Test mAP")
    plt.title("Seed-to-Seed Test mAP Variability")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "seed_variability_map.png", dpi=200)
    plt.close()


def plot_gpu_tradeoff(output_dir: Path, summary: pd.DataFrame) -> None:
    plt.figure(figsize=(7, 5))
    for _, row in summary.iterrows():
        plt.scatter(row["gpu_hours"], row["mean_mAP"], s=70)
        plt.text(row["gpu_hours"], row["mean_mAP"], DISPLAY_NAMES.get(row["method"], row["method"]), fontsize=8)
    plt.xlabel("GPU hours across seeds")
    plt.ylabel("Mean test mAP")
    plt.title("Compute vs Performance")
    plt.tight_layout()
    plt.savefig(output_dir / "gpu_hours_vs_map.png", dpi=200)
    plt.close()


def write_plots(output_dir: Path, summary_rows: list[dict], bootstrap: bool = True) -> None:
    if plt is None:
        raise SystemExit("Matplotlib is required to write PNG plots. Run aggregation in the same environment used for training, or install matplotlib.")
    summary = pd.DataFrame(summary_rows)
    seed_df = pd.read_csv(output_dir / "seed_level_metrics.csv")
    per_class_summary = pd.read_csv(output_dir / "per_class_metrics_summary.csv")
    plot_method_map(output_dir, summary)
    plot_group_map(output_dir, summary)
    plot_per_class_heatmap(output_dir, per_class_summary)
    if bootstrap:
        plot_bootstrap_ci(output_dir, summary)
    plot_seed_variability(output_dir, seed_df)
    plot_gpu_tradeoff(output_dir, summary)


def main() -> None:
    args = parse_args()
    runs_dir = args.runs_dir or args.project_root / "runs"
    output_dir = args.output_dir or args.project_root / "analysis" / "results"
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"Use a new, empty aggregation output directory: {output_dir}")
    class_groups_path = args.class_groups or args.project_root / "1. Data analysis + inspection" / "results" / "data" / "class_groups.csv"
    groups_df = pd.read_csv(class_groups_path)
    class_to_group = dict(zip(groups_df["pathology"], groups_df["group"]))

    all_rows = {}
    reference_test_rows = None
    for method in args.methods:
        all_rows[method] = collect_method(runs_dir, method, args.seeds, class_groups_path, reference_test_rows)
        reference_test_rows = all_rows[method][0]["test_rows"]
    baseline = [r["metrics"]["mAP"] for r in all_rows.get("bce_baseline", [])]
    rows = []
    details = {"bootstrap_mode": args.bootstrap_mode, "seeds": args.seeds, "methods": {}, "pairwise_top_methods": []}

    for method, method_rows in all_rows.items():
        metrics = [r["metrics"] for r in method_rows]
        reference_map = float("nan")
        ci = {"mAP": {"lower": float("nan"), "upper": float("nan")}, "group_mAP": {}}
        if args.bootstrap_mode != "none":
            labels, probs = load_bootstrap_arrays(method_rows, args.bootstrap_mode)
            reference_map = map_by_group(labels, probs, class_to_group)[0]
            ci = bootstrap_ci(labels, probs, class_to_group, args.bootstrap_resamples, args.bootstrap_seed)
        seed_maps = [m["mAP"] for m in metrics]
        group_means = {g: safe_mean(m["group_mAP"][g] for m in metrics) for g in GROUPS}
        group_stds = {g: safe_std(m["group_mAP"][g] for m in metrics) for g in GROUPS}
        group_f1_means = {g: safe_mean(m["group_f1_at_0.5"][g] for m in metrics) for g in GROUPS}
        per_class_ap = {p: {"mean": safe_mean(m["per_class_ap"][p] for m in metrics), "std": safe_std(m["per_class_ap"][p] for m in metrics)} for p in PATHOLOGIES}
        p_value = None if method == "bce_baseline" or not baseline else wilcoxon_exact(seed_maps, baseline)
        gpu_hours = sum(float(r["compute"].get("gpu_hours", 0.0) or 0.0) for r in method_rows)
        row = {
            "method": method,
            "mean_mAP": safe_mean(seed_maps),
            "std_mAP": safe_std(seed_maps),
            "head_mAP": group_means["head"],
            "medium_mAP": group_means["medium"],
            "tail_mAP": group_means["tail"],
            "head_mAP_std": group_stds["head"],
            "medium_mAP_std": group_stds["medium"],
            "tail_mAP_std": group_stds["tail"],
            "macro_f1": safe_mean(m["macro_f1_at_0.5"] for m in metrics),
            "macro_f1_std": safe_std(m["macro_f1_at_0.5"] for m in metrics),
            "macro_f1_tuned": safe_mean(r["tuned"]["macro_f1"] for r in method_rows),
            "macro_f1_tuned_std": safe_std(r["tuned"]["macro_f1"] for r in method_rows),
            "macro_roc_auc": safe_mean(safe_mean(m["per_class_roc_auc"].values()) for m in metrics),
            "head_f1": group_f1_means["head"],
            "medium_f1": group_f1_means["medium"],
            "tail_f1": group_f1_means["tail"],
            "bootstrap_CI_lower": ci["mAP"]["lower"],
            "bootstrap_CI_upper": ci["mAP"]["upper"],
            "bootstrap_reference_mAP": reference_map,
            "p_value_vs_baseline": p_value,
            "gpu_hours": gpu_hours,
        }
        rows.append(row)
        details["methods"][method] = {**row, "per_class_ap": per_class_ap, "bootstrap_group_mAP_CI": ci["group_mAP"], "seed_mAP": seed_maps}

    ordered = sorted(rows, key=lambda r: r["mean_mAP"], reverse=True)
    top = [r["method"] for r in ordered[:3]]
    for a, b in itertools.combinations(top, 2):
        p = wilcoxon_exact([r["metrics"]["mAP"] for r in all_rows[a]], [r["metrics"]["mAP"] for r in all_rows[b]])
        details["pairwise_top_methods"].append({"method_a": a, "method_b": b, "p_value": p})

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values("mean_mAP", ascending=False).to_csv(output_dir / "results_summary.csv", index=False)
    write_tables(output_dir, rows, all_rows, details)
    write_plots(output_dir, rows, bootstrap=args.bootstrap_mode != "none")
    from common.nih_longtail import write_json
    write_json(details, output_dir / "results_summary.json")
    print(f"Wrote {output_dir / 'results_summary.csv'}")
    print(f"Wrote {output_dir / 'results_summary.json'}")


if __name__ == "__main__":
    main()
