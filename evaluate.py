#!/usr/bin/env python3
"""Evaluate one method/seed checkpoint on validation or final test data."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.metrics import average_precision, precision_recall_f1
from common.nih_longtail import (  # noqa: E402
    PATHOLOGIES,
    build_criterion,
    load_class_groups,
    load_manifest,
    make_loader,
    make_model,
    select_device,
    sigmoid_np,
    split_integrity_report,
    sha256_file,
    write_json,
)
from common.run_state import PROTOCOL


FILENAME_RE = re.compile(r"[^,\s;|]+?\.(?:png|jpg|jpeg)", re.IGNORECASE)
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
METHOD_FOLDERS = {
    "bce_baseline": "bce_baseline",
    "focal_loss": "focal_loss",
    "db_loss": "db_loss",
    "sqrt_sampling": "sqrt_sampling",
    "crt": "crt",
    "balanced_mixup": "balanced_mixup",
}
DEFAULTS = {
    "gamma": 2.0,
    "alpha": None,
    "db_map_alpha": 10.0,
    "db_map_beta": 0.2,
    "db_map_gamma": 0.1,
    "db_neg_scale": 2.0,
    "db_init_bias": 0.05,
    "db_bias_version": None,
    "sampling_exponent": 0.5,
    "no_positive_weight": None,
    "mixup_alpha": 0.4,
    "disable_mixup": False,
    "retain_classifier": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved method/seed run.")
    parser.add_argument("--project-root", type=Path, default=Path("/home-mscluster/pgohil/cxr-research"))
    parser.add_argument("--dataset-root", type=Path, default=Path("/datasets/pgohil"))
    parser.add_argument("--metadata-csv", type=Path, default=Path("/datasets/pgohil/Data_Entry_2017.csv"))
    parser.add_argument("--test-list", type=Path, default=Path("/datasets/pgohil/test_list.txt"))
    parser.add_argument("--method", required=True, choices=sorted(METHOD_FOLDERS))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--split", required=True, choices=["val", "test"])
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--train-manifest", type=Path, default=None)
    parser.add_argument("--val-manifest", type=Path, default=None)
    parser.add_argument("--class-groups", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true", help="Reuse an evaluation only when its completion record and input hashes match.")
    return parser.parse_args()


def fill_paths(args: argparse.Namespace) -> None:
    data_dir = args.project_root / "1. Data analysis + inspection" / "results" / "data"
    args.train_manifest = args.train_manifest or data_dir / "train_manifest.csv"
    args.val_manifest = args.val_manifest or data_dir / "val_manifest.csv"
    args.class_groups = args.class_groups or data_dir / "class_groups.csv"
    args.run_dir = args.run_dir or args.project_root / "runs" / args.method / f"seed_{args.seed}"
    args.checkpoint = args.checkpoint or args.run_dir / "checkpoint_best.pt"


def load_filename_list(path: Path) -> set[str]:
    names: set[str] = set()
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            for match in FILENAME_RE.findall(line):
                names.add(Path(match).name)
    if not names:
        raise SystemExit(f"No test filenames found in {path}")
    return names


def metadata_for_filenames(metadata_csv: Path, filenames: set[str]) -> pd.DataFrame:
    rows = []
    with metadata_csv.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        norm = {re.sub(r"[^a-z0-9]+", "", c.lower()): c for c in reader.fieldnames or []}
        image_col = norm.get("imageindex")
        labels_col = norm.get("findinglabels")
        patient_col = norm.get("patientid")
        if not image_col or not labels_col:
            raise SystemExit(f"Metadata CSV lacks Image Index or Finding Labels: {metadata_csv}")
        for row in reader:
            filename = Path(row.get(image_col, "")).name
            if filename not in filenames:
                continue
            labels = {p.strip() for p in str(row.get(labels_col, "")).split("|") if p.strip()}
            record = {
                "image_filename": filename,
                "patient_id": row.get(patient_col, ""),
                "original_finding_labels": row.get(labels_col, ""),
            }
            for pathology in PATHOLOGIES:
                record[pathology] = int(pathology in labels)
            rows.append(record)
    missing = sorted(filenames - set(r["image_filename"] for r in rows))
    if missing:
        raise SystemExit(f"Metadata missing {len(missing)} test filename(s), e.g. {missing[:5]}")
    return pd.DataFrame(rows)


def scan_image_paths(dataset_root: Path, filenames: set[str]) -> dict[str, Path]:
    found: dict[str, Path] = {}
    remaining = set(filenames)
    for current, dirnames, file_names in os.walk(dataset_root):
        dirnames[:] = [d for d in dirnames if not Path(current, d).is_symlink()]
        for name in file_names:
            if name in remaining and Path(name).suffix.lower() in IMAGE_EXTENSIONS:
                found[name] = Path(current) / name
                remaining.remove(name)
        if not remaining:
            break
    if remaining:
        raise SystemExit(f"Could not find {len(remaining)} test image file(s), e.g. {sorted(remaining)[:5]}")
    return found


def build_test_manifest(args: argparse.Namespace) -> pd.DataFrame:
    names = load_filename_list(args.test_list)
    meta = metadata_for_filenames(args.metadata_csv, names)
    paths = scan_image_paths(args.dataset_root, names)
    meta["image_path"] = meta["image_filename"].map(lambda name: str(paths[name]))
    if args.max_test_samples:
        meta = meta.sort_values("image_filename").head(args.max_test_samples).copy()
    return meta[["image_filename", "image_path", "patient_id", "original_finding_labels", *PATHOLOGIES]].sort_values("image_filename").reset_index(drop=True)




def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(int)
    n_pos = int(y_true.sum())
    n_neg = int(len(y_true) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty(len(y_score), dtype=float)
    sorted_scores = y_score[order]
    i = 0
    while i < len(sorted_scores):
        j = i + 1
        while j < len(sorted_scores) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    return float((ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def safe_mean(values) -> float:
    arr = np.asarray(list(values), dtype=float)
    finite = np.isfinite(arr)
    return float(arr[finite].mean()) if finite.any() else float("nan")


def prf(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict[str, float]:
    y_true = y_true.astype(int)
    pred = (y_score >= threshold).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    precision, recall, f1 = precision_recall_f1(tp, fp, fn)
    return {"p": precision, "r": recall, "f1": f1}


def metrics_from_probs(labels: np.ndarray, probs: np.ndarray, groups_df: pd.DataFrame, thresholds: dict[str, float] | None = None) -> dict:
    groups = groups_df.set_index("pathology")["group"].to_dict()
    per_class_ap = {}
    per_class_auc = {}
    per_class_prf = {}
    for idx, pathology in enumerate(PATHOLOGIES):
        y = labels[:, idx]
        if int(y.sum()) == 0:
            warnings.warn(f"{pathology} has zero positives; AP is undefined; F1 uses zero-division=0.")
        if int(y.sum()) == 0 or int(y.sum()) == len(y):
            warnings.warn(f"{pathology} lacks both classes; ROC-AUC is undefined.")
        threshold = 0.5 if thresholds is None else float(thresholds[pathology])
        per_class_ap[pathology] = average_precision(y, probs[:, idx])
        per_class_auc[pathology] = roc_auc(y, probs[:, idx])
        per_class_prf[pathology] = prf(y, probs[:, idx], threshold)
    group_mAP = {}
    group_f1 = {}
    for group in ("head", "medium", "tail"):
        classes = [p for p in PATHOLOGIES if groups.get(p) == group]
        group_mAP[group] = safe_mean(per_class_ap[p] for p in classes)
        group_f1[group] = safe_mean(per_class_prf[p]["f1"] for p in classes)
    return {
        "per_class_ap": per_class_ap,
        "mAP": safe_mean(per_class_ap.values()),
        "group_mAP": group_mAP,
        "per_class_precision_recall_f1_at_0.5": per_class_prf if thresholds is None else None,
        "macro_f1_at_0.5": safe_mean(v["f1"] for v in per_class_prf.values()) if thresholds is None else None,
        "group_f1_at_0.5": group_f1 if thresholds is None else None,
        "per_class_roc_auc": per_class_auc,
    }


def tuned_f1_metrics(labels: np.ndarray, probs: np.ndarray, groups_df: pd.DataFrame, thresholds: dict[str, float]) -> dict:
    base = metrics_from_probs(labels, probs, groups_df, thresholds)
    per_class = {}
    for idx, pathology in enumerate(PATHOLOGIES):
        per_class[pathology] = prf(labels[:, idx], probs[:, idx], thresholds[pathology])
    groups = groups_df.set_index("pathology")["group"].to_dict()
    group_f1 = {}
    for group in ("head", "medium", "tail"):
        classes = [p for p in PATHOLOGIES if groups.get(p) == group]
        group_f1[group] = safe_mean(per_class[p]["f1"] for p in classes)
    return {
        "thresholds": thresholds,
        "per_class_precision_recall_f1": per_class,
        "macro_f1": safe_mean(v["f1"] for v in per_class.values()),
        "group_f1": group_f1,
        "per_class_ap": base["per_class_ap"],
        "mAP": base["mAP"],
        "group_mAP": base["group_mAP"],
        "per_class_roc_auc": base["per_class_roc_auc"],
    }


def tune_thresholds(labels: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    thresholds = {}
    grid = np.linspace(0.0, 1.0, 101)
    for idx, pathology in enumerate(PATHOLOGIES):
        best_threshold = 0.5
        best_f1 = -1.0
        for threshold in grid:
            f1 = prf(labels[:, idx], probs[:, idx], float(threshold))["f1"]
            score = f1 if math.isfinite(f1) else -1.0
            if score > best_f1:
                best_f1 = score
                best_threshold = float(threshold)
        thresholds[pathology] = best_threshold
    return thresholds


def criterion_args(method: str, checkpoint: dict, args: argparse.Namespace) -> SimpleNamespace:
    hyper = checkpoint.get("method_hyperparameters") or {}
    training_args = checkpoint.get("training_args") or {}
    values = {key: hyper.get(key, training_args.get(key, default)) for key, default in DEFAULTS.items()}
    return SimpleNamespace(method=method, batch_size=args.batch_size, num_workers=args.num_workers, threshold=0.5, seed=args.seed, **values)


def main() -> None:
    args = parse_args()
    fill_paths(args)
    if not args.checkpoint.exists():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")
    out_dir = args.run_dir / f"eval_{args.split}"
    train_df = load_manifest(args.train_manifest)
    val_df = load_manifest(args.val_manifest)
    report = split_integrity_report(train_df, val_df, args.test_list, args.metadata_csv)
    if not report["passed"]:
        raise ValueError(f"Patient/image split integrity failed: {report}")
    eval_df = val_df if args.split == "val" else build_test_manifest(args)
    groups_df, class_groups = load_class_groups(args.class_groups, train_df)
    device = select_device(args)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if checkpoint.get("training_protocol") != PROTOCOL or checkpoint.get("random_seed") != args.seed:
        raise ValueError("Evaluation requires a corrected checkpoint with the requested seed")
    if checkpoint.get("method") != args.method or checkpoint.get("class_names", checkpoint.get("pathology_order")) != PATHOLOGIES:
        raise ValueError("Checkpoint method or pathology order mismatch")
    if class_groups != checkpoint.get("class_groups"):
        raise ValueError("Evaluation class groups differ from training")
    identity = {"training_protocol": PROTOCOL, "method": args.method, "seed": args.seed, "split": args.split,
                "checkpoint_sha256": sha256_file(args.checkpoint), "max_test_samples": args.max_test_samples,
                "input_sha256": {str(p.resolve()): sha256_file(p) for p in
                                 (args.train_manifest, args.val_manifest, args.class_groups, args.test_list, args.metadata_csv)}}
    for path in (args.train_manifest, args.val_manifest, args.test_list, args.metadata_csv):
        key = str(path.resolve())
        if checkpoint.get("input_sha256", {}).get(key) != identity["input_sha256"][key]:
            raise ValueError(f"Evaluation input differs from training: {path}")
    # BCE derives its groups from training counts; other methods also hash the CSV.
    if args.method != "bce_baseline":
        key = str(args.class_groups.resolve())
        if checkpoint.get("input_sha256", {}).get(key) != identity["input_sha256"][key]:
            raise ValueError("Evaluation class groups input differs from training")
    threshold_path = args.run_dir / "eval_val" / "tuned_thresholds.json"
    if args.split == "test":
        val_marker = args.run_dir / "eval_val/evaluation_complete.json"
        if not val_marker.is_file() or not threshold_path.is_file():
            raise ValueError("Run validation evaluation successfully before test evaluation")
        val_record = json.loads(val_marker.read_text())
        if val_record["identity"]["checkpoint_sha256"] != identity["checkpoint_sha256"]:
            raise ValueError("Validation thresholds came from a different checkpoint")
        if val_record["identity"].get("input_sha256") != identity["input_sha256"]:
            raise ValueError("Validation/test inputs differ")
        identity["validation_thresholds_sha256"] = sha256_file(threshold_path)
        if val_record["output_sha256"]["tuned_thresholds.json"] != identity["validation_thresholds_sha256"]:
            raise ValueError("Validation thresholds changed after evaluation")
    marker = out_dir / "evaluation_complete.json"
    if args.skip_existing and marker.is_file():
        saved = json.loads(marker.read_text())
        if saved.get("identity") == identity and all((out_dir / name).is_file() and sha256_file(out_dir / name) == digest
                                                    for name, digest in saved.get("output_sha256", {}).items()) and saved.get("output_sha256"):
            print(f"Verified existing evaluation: {out_dir}")
            return
        raise ValueError(f"Existing evaluation no longer matches its inputs/outputs: {out_dir}")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise ValueError(f"Refusing to overwrite evaluation: {out_dir}")
    model = make_model(pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    eval_args = criterion_args(args.method, checkpoint, args)
    criterion = build_criterion(eval_args, train_df).to(device)
    loader = make_loader(eval_df, training=False, args=eval_args)

    model.eval()
    logits_all, labels_all, names = [], [], []
    with torch.no_grad():
        for images, labels, batch_names in loader:
            logits = model(images.to(device, non_blocking=True))
            _ = criterion(logits, labels.to(device, non_blocking=True))
            logits_all.append(logits.detach().cpu().numpy())
            labels_all.append(labels.numpy())
            names.extend(batch_names)
    logits = np.concatenate(logits_all, axis=0)
    labels = np.concatenate(labels_all, axis=0).astype(np.int64)
    probs = sigmoid_np(logits).astype(np.float32)
    patient_lookup = eval_df.set_index("image_filename")["patient_id"].astype(str).to_dict()
    patient_ids = np.asarray([patient_lookup[name] for name in names])
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "predictions.npy", probs)
    np.save(out_dir / "logits.npy", logits.astype(np.float32))
    np.save(out_dir / "labels.npy", labels)
    np.save(out_dir / "image_ids.npy", np.asarray(names))
    np.save(out_dir / "patient_ids.npy", patient_ids)
    metrics = metrics_from_probs(labels, probs, groups_df)
    write_json(metrics, out_dir / "metrics.json")

    if args.split == "val":
        thresholds = tune_thresholds(labels, probs)
        write_json(thresholds, out_dir / "tuned_thresholds.json")
    else:
        threshold_path = args.run_dir / "eval_val" / "tuned_thresholds.json"
        if threshold_path.exists():
            thresholds = json.loads(threshold_path.read_text(encoding="utf-8"))
            write_json(tuned_f1_metrics(labels, probs, groups_df, thresholds), out_dir / "metrics_tuned_threshold.json")
        else:
            raise ValueError(f"Run validation evaluation first: missing {threshold_path}")

    write_json({"identity": identity, "output_sha256": {p.name: sha256_file(p) for p in sorted(out_dir.iterdir()) if p.is_file()}}, marker)

    print(f"Saved {args.split} evaluation outputs to {out_dir}")


if __name__ == "__main__":
    main()
