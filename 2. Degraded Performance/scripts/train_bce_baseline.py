#!/usr/bin/env python3
"""ResNet-152 + ordinary BCE baseline for NIH ChestX-ray14.

This is deliberately a plain imbalanced baseline:
- no class weights
- no pos_weight
- no sampler
- no balancing or long-tailed learning method
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import shutil
import sys
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image, ImageOps
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.transforms import InterpolationMode

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.metrics import average_precision, precision_recall_f1
from common.run_state import PROTOCOL, validate_run, require_mutable_run, input_hashes, capture_rng, restore_rng, atomic_checkpoint
from common.nih_longtail import load_manifest, split_integrity_report as shared_split_report, load_resume


PATHOLOGIES = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Effusion",
    "Emphysema",
    "Fibrosis",
    "Hernia",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pleural_Thickening",
    "Pneumonia",
    "Pneumothorax",
]

GROUP_COLORS = {"head": "#2f6fbb", "medium": "#4f9f5f", "tail": "#c96b3c"}
FILENAME_RE = re.compile(r"[^,\s;|]+?\.(?:png|jpg|jpeg)", re.IGNORECASE)

PREPROCESSING_DESCRIPTION = {
    "image_source": "Original NIH image paths from the preprocessing manifests; no processed image files are overwritten or reused.",
    "training": [
        "open image from manifest image_path",
        "convert to grayscale, then RGB/three channels",
        "resize shorter image representation to 256 x 256",
        "random crop 224 x 224",
        "random horizontal flip p=0.5",
        "random rotation in [-10, 10] degrees",
        "brightness and contrast jitter 0.2",
        "convert to tensor",
        "ImageNet normalization",
    ],
    "validation_test": [
        "open image from manifest image_path",
        "convert to grayscale, then RGB/three channels",
        "resize to 256 x 256",
        "center crop 224 x 224",
        "convert to tensor",
        "ImageNet normalization",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/evaluate ResNet-152 ordinary BCE baseline.")
    parser.add_argument("--mode", choices=["train", "check-splits", "evaluate-val", "plot-only"], default="train")
    parser.add_argument("--project-root", type=Path, default=Path("/home-mscluster/pgohil/cxr-research"))
    parser.add_argument("--dataset-root", type=Path, default=Path("/datasets/pgohil"))
    parser.add_argument("--metadata-csv", type=Path, default=Path("/datasets/pgohil/Data_Entry_2017.csv"))
    parser.add_argument("--test-list", type=Path, default=Path("/datasets/pgohil/test_list.txt"))
    parser.add_argument("--train-manifest", type=Path, default=None)
    parser.add_argument("--val-manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--gradient-clip", type=float, default=1.0, help="0 disables gradient clipping.")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--no-amp", action="store_true", help="Disable CUDA mixed precision.")
    parser.add_argument("--resume", action="store_true", help="Resume a matching, unevaluated corrected run from checkpoint_final.pt.")
    parser.add_argument("--dry-run", action="store_true", help="Use small limits and one epoch unless explicit limits are set.")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--check-batches", type=int, default=0, help="Verify this many train batches before training.")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.set_defaults(method="bce_baseline")
    return parser.parse_args()


def default_paths(args: argparse.Namespace) -> None:
    if args.train_manifest is None:
        args.train_manifest = args.project_root / "1. Data analysis + inspection" / "results" / "data" / "train_manifest.csv"
    if args.val_manifest is None:
        args.val_manifest = args.project_root / "1. Data analysis + inspection" / "results" / "data" / "val_manifest.csv"
    if args.output_dir is None:
        args.output_dir = args.project_root / "runs" / "bce_baseline" / f"seed_{args.seed}"
    if args.dry_run:
        args.epochs = min(args.epochs, 1)
        args.max_train_samples = args.max_train_samples or 64
        args.max_val_samples = args.max_val_samples or 64
        args.check_batches = max(args.check_batches, 2)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def atomic_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(data), f, indent=2, sort_keys=True)
        f.write("\n")
    tmp.replace(path)


def to_jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value





def load_filename_list(path: Path) -> set[str]:
    if not path.exists():
        raise SystemExit(f"Split file not found: {path}")
    names: set[str] = set()
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            for match in FILENAME_RE.findall(line):
                names.add(Path(match).name)
    return names


def metadata_for_filenames(metadata_csv: Path, filenames: set[str]) -> pd.DataFrame:
    rows = []
    with metadata_csv.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        norm = {re.sub(r"[^a-z0-9]+", "", c.lower()): c for c in fieldnames}
        image_col = norm.get("imageindex")
        labels_col = norm.get("findinglabels")
        patient_col = norm.get("patientid")
        if not image_col or not labels_col or not patient_col:
            raise SystemExit(f"Metadata CSV lacks Image Index, Finding Labels, or Patient ID: {metadata_csv}")
        for row in reader:
            filename = Path(row.get(image_col, "")).name
            if filename in filenames:
                record = {
                    "image_filename": filename,
                    "patient_id": row.get(patient_col, ""),
                    "original_finding_labels": row.get(labels_col, ""),
                }
                labels = {p.strip() for p in str(record["original_finding_labels"]).split("|") if p.strip()}
                for pathology in PATHOLOGIES:
                    record[pathology] = int(pathology in labels)
                record["no_finding"] = int("No Finding" in labels)
                record["num_positive_pathologies"] = int(sum(record[p] for p in PATHOLOGIES))
                rows.append(record)
    return pd.DataFrame(rows)


def split_integrity_report(train, val, args):
    return shared_split_report(train, val, args.test_list, args.metadata_csv)



def assert_split_integrity(report: dict) -> None:
    if not report["passed"]:
        raise SystemExit(f"Split integrity failed: {report.get('failures', {})}")


def class_frequencies_and_groups(train: pd.DataFrame, val: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
    rows = []
    groups = {"head": [], "medium": [], "tail": []}
    for pathology in PATHOLOGIES:
        count = int(train[pathology].sum())
        if count > 5000:
            group = "head"
        elif count >= 1000:
            group = "medium"
        else:
            group = "tail"
        groups[group].append(pathology)
        rows.append({
            "pathology": pathology,
            "train_positive_count": count,
            "validation_positive_count": int(val[pathology].sum()) if val is not None else None,
            "group": group,
        })
    return pd.DataFrame(rows), groups


def make_transform(training: bool):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if training:
        return transforms.Compose([
            transforms.Resize((256, 256), interpolation=InterpolationMode.BILINEAR),
            transforms.RandomCrop((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10, interpolation=InterpolationMode.BILINEAR),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            normalize,
        ])
    return transforms.Compose([
        transforms.Resize((256, 256), interpolation=InterpolationMode.BILINEAR),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])


class NIHManifestDataset(Dataset):
    def __init__(self, manifest: pd.DataFrame, training: bool):
        self.manifest = manifest.reset_index(drop=True)
        self.transform = make_transform(training)

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int):
        row = self.manifest.iloc[index]
        with Image.open(row["image_path"]) as img:
            image = ImageOps.grayscale(img).convert("RGB")
            image = self.transform(image)
        labels = torch.tensor(row[PATHOLOGIES].to_numpy(dtype=np.float32), dtype=torch.float32)
        return image, labels, row["image_filename"]


def make_loader(df: pd.DataFrame, training: bool, args: argparse.Namespace) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(args.seed + (1 if training else 1001))
    return DataLoader(
        NIHManifestDataset(df, training=training),
        batch_size=args.batch_size,
        shuffle=training,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def make_model(pretrained: bool = True) -> nn.Module:
    weights = models.ResNet152_Weights.IMAGENET1K_V2 if pretrained else None
    model = models.resnet152(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, len(PATHOLOGIES))
    return model



def select_device(args: argparse.Namespace) -> torch.device:
    if args.device == "cpu":
        return torch.device("cpu")
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA was requested but is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")





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
    pos_rank_sum = ranks[y_true == 1].sum()
    return float((pos_rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def safe_mean(values: list[float]) -> float:
    arr = np.array(values, dtype=float)
    return float(np.nanmean(arr)) if np.isfinite(arr).any() else float("nan")


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    xr = pd.Series(x[mask]).rank(method="average").to_numpy(dtype=float)
    yr = pd.Series(y[mask]).rank(method="average").to_numpy(dtype=float)
    if np.std(xr) == 0 or np.std(yr) == 0:
        return float("nan")
    return float(np.corrcoef(xr, yr)[0, 1])


def calculate_metrics(
    labels: np.ndarray,
    logits: np.ndarray,
    class_freq: pd.DataFrame,
    threshold: float,
    loss: float,
) -> tuple[dict, pd.DataFrame, dict]:
    probs = 1 / (1 + np.exp(-logits))
    pred = (probs >= threshold).astype(int)
    rows = []
    for idx, pathology in enumerate(PATHOLOGIES):
        y = labels[:, idx].astype(int)
        p = pred[:, idx].astype(int)
        tp = int(((p == 1) & (y == 1)).sum())
        fp = int(((p == 1) & (y == 0)).sum())
        fn = int(((p == 0) & (y == 1)).sum())
        precision, recall, f1 = precision_recall_f1(tp, fp, fn)
        if y.sum() == 0:
            warnings.warn(f"{pathology} has no positive examples in this split; AP is undefined; F1 uses zero-division=0.")
        if y.sum() == 0 or y.sum() == len(y):
            warnings.warn(f"{pathology} lacks both classes in this split; ROC-AUC set to NaN.")
        rows.append({
            "pathology": pathology,
            "training_positive_count": int(class_freq.set_index("pathology").loc[pathology, "train_positive_count"]),
            "group": str(class_freq.set_index("pathology").loc[pathology, "group"]),
            "positive_count": int(y.sum()),
            "ap": average_precision(y, probs[:, idx]),
            "roc_auc": roc_auc(y, probs[:, idx]),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })
    per_class = pd.DataFrame(rows)
    metrics = {
        "loss": float(loss),
        "mAP": safe_mean(per_class["ap"].tolist()),
        "macro_roc_auc": safe_mean(per_class["roc_auc"].tolist()),
        "macro_f1": safe_mean(per_class["f1"].tolist()),
        "threshold": threshold,
    }
    for group in ("head", "medium", "tail"):
        group_rows = per_class[per_class["group"] == group]
        metrics[f"{group}_mAP"] = safe_mean(group_rows["ap"].tolist())
        metrics[f"{group}_macro_f1"] = safe_mean(group_rows["f1"].tolist())
    degradation = degradation_summary(per_class, metrics)
    return metrics, per_class, degradation


def degradation_summary(per_class: pd.DataFrame, metrics: dict) -> dict:
    head_map = metrics.get("head_mAP", float("nan"))
    tail_map = metrics.get("tail_mAP", float("nan"))
    head_f1 = metrics.get("head_macro_f1", float("nan"))
    tail_f1 = metrics.get("tail_macro_f1", float("nan"))
    counts = per_class["training_positive_count"].to_numpy(dtype=float)
    log_counts = np.log10(np.maximum(counts, 1))
    ap = per_class["ap"].to_numpy(dtype=float)
    f1 = per_class["f1"].to_numpy(dtype=float)
    return {
        "head_tail_map_gap": head_map - tail_map if math.isfinite(head_map) and math.isfinite(tail_map) else float("nan"),
        "head_tail_f1_gap": head_f1 - tail_f1 if math.isfinite(head_f1) and math.isfinite(tail_f1) else float("nan"),
        "tail_head_map_ratio": tail_map / head_map if math.isfinite(tail_map) and math.isfinite(head_map) and head_map != 0 else float("nan"),
        "spearman_log_train_frequency_vs_ap": spearman(log_counts, ap),
        "spearman_log_train_frequency_vs_f1": spearman(log_counts, f1),
    }


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module, class_freq: pd.DataFrame, args: argparse.Namespace):
    model.eval()
    loss_sum, sample_count = 0.0, 0
    logits_all: list[np.ndarray] = []
    labels_all: list[np.ndarray] = []
    names: list[str] = []
    with torch.no_grad():
        for images, labels, batch_names in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss_sum += float(loss.item()) * labels.shape[0]
            sample_count += labels.shape[0]
            logits_all.append(logits.detach().cpu().numpy())
            labels_all.append(labels.detach().cpu().numpy())
            names.extend(batch_names)
    logits_np = np.concatenate(logits_all, axis=0)
    labels_np = np.concatenate(labels_all, axis=0)
    metrics, per_class, degradation = calculate_metrics(labels_np, logits_np, class_freq, args.threshold, loss_sum / sample_count if sample_count else float("nan"))
    return metrics, per_class, degradation, logits_np, labels_np, names


def train_one_epoch(model, loader, device, criterion, optimizer, scaler, args):
    model.train()
    loss_sum, sample_count = 0.0, 0
    use_amp = scaler is not None and scaler.is_enabled()
    for images, labels, _ in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        try:
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise SystemExit("Training loss became NaN or infinite.")
            if use_amp:
                scaler.scale(loss).backward()
                if args.gradient_clip and args.gradient_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if args.gradient_clip and args.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
                optimizer.step()
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower() and device.type == "cuda":
                torch.cuda.empty_cache()
                raise SystemExit("CUDA out of memory. Reduce --batch-size, for example to 8 or 4, and rerun. The script did not change the setup silently.") from exc
            raise
        loss_sum += float(loss.item()) * labels.shape[0]
        sample_count += labels.shape[0]
    return loss_sum / sample_count if sample_count else float("nan")


def save_checkpoint(path: Path, model, optimizer, scheduler, epoch: int, best_map: float, class_groups: dict, args: argparse.Namespace, scaler=None, resume_state=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_checkpoint({
        "training_protocol": PROTOCOL,
        "input_sha256": json.loads((args.output_dir / "config.json").read_text())["input_sha256"],
        "method": "bce_baseline",
        "amp_scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "resume_state": resume_state,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "best_validation_mAP": best_map,
        "pathology_order": PATHOLOGIES,
        "class_groups": class_groups,
        "training_args": vars(args),
        "random_seed": args.seed,
        "preprocessing_description": PREPROCESSING_DESCRIPTION,
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
    }, path)





def check_batches(model, loader, device, criterion, args) -> None:
    model.eval()
    checked = 0
    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.to(device)
            labels = labels.to(device)
            if list(images.shape[1:]) != [3, 224, 224]:
                raise SystemExit(f"Unexpected tensor shape: {list(images.shape)}")
            label_values = set(labels.detach().cpu().numpy().ravel().tolist())
            if not label_values <= {0.0, 1.0}:
                raise SystemExit(f"Labels contain values outside 0/1: {label_values}")
            logits = model(images)
            if list(logits.shape) != [labels.shape[0], len(PATHOLOGIES)]:
                raise SystemExit(f"Unexpected model output shape: {list(logits.shape)}")
            loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise SystemExit("Smoke-test loss is not finite.")
            checked += 1
            if checked >= args.check_batches:
                break
    print(f"Batch check passed for {checked} batch(es).")


def save_metric_outputs(prefix: str, out_dir: Path, metrics: dict, per_class: pd.DataFrame, degradation: dict, logits: np.ndarray, labels: np.ndarray, names: list[str]) -> None:
    if prefix != "validation":
        raise ValueError(f"Unsupported BCE baseline metric prefix: {prefix}")
    atomic_json(metrics, out_dir / "validation_metrics.json")
    per_class.to_csv(out_dir / "validation_per_class_metrics.csv", index=False)
    np.savez_compressed(out_dir / "validation_predictions_labels.npz", logits=logits, labels=labels, image_filenames=np.array(names))
    atomic_json(degradation, out_dir / "baseline_degradation_summary.json")
    write_report(out_dir / "baseline_degradation_report.txt", metrics, per_class, degradation)
    make_plots(out_dir, per_class)


def write_report(path: Path, metrics: dict, per_class: pd.DataFrame, degradation: dict) -> None:
    lowest = per_class.sort_values("ap", na_position="last").head(5)
    evidence = (
        "Yes, tail-class performance is worse than head-class performance by the saved mAP/F1 gaps."
        if degradation.get("head_tail_map_gap", float("nan")) > 0 or degradation.get("head_tail_f1_gap", float("nan")) > 0
        else "No clear head-to-tail degradation is shown by the saved gaps."
    )
    lines = [
        "Baseline ResNet-152 BCE Degradation Report",
        "=" * 44,
        f"Selected checkpoint epoch: {metrics.get('selected_epoch', metrics.get('epoch', 'unknown'))}",
        f"Overall mAP: {metrics.get('mAP')}",
        f"Overall macro-F1 at threshold {metrics.get('threshold')}: {metrics.get('macro_f1')}",
        f"Head mAP: {metrics.get('head_mAP')}",
        f"Medium mAP: {metrics.get('medium_mAP')}",
        f"Tail mAP: {metrics.get('tail_mAP')}",
        f"Head macro-F1: {metrics.get('head_macro_f1')}",
        f"Medium macro-F1: {metrics.get('medium_macro_f1')}",
        f"Tail macro-F1: {metrics.get('tail_macro_f1')}",
        f"Head-tail mAP gap: {degradation.get('head_tail_map_gap')}",
        f"Head-tail macro-F1 gap: {degradation.get('head_tail_f1_gap')}",
        f"Tail/head mAP ratio: {degradation.get('tail_head_map_ratio')}",
        f"Spearman log frequency vs AP: {degradation.get('spearman_log_train_frequency_vs_ap')}",
        f"Spearman log frequency vs F1: {degradation.get('spearman_log_train_frequency_vs_f1')}",
        "",
        "Lowest-performing pathologies by AP:",
    ]
    for _, row in lowest.iterrows():
        lines.append(f"- {row['pathology']}: AP={row['ap']}, F1={row['f1']}, group={row['group']}, train positives={row['training_positive_count']}")
    lines += ["", f"Evidence of degraded tail-class performance: {evidence}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def make_plots(out_dir: Path, per_class: pd.DataFrame) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ordered = per_class.sort_values("training_positive_count", ascending=False)
    colors = [GROUP_COLORS.get(g, "gray") for g in ordered["group"]]

    def bar_plot(column: str, ylabel: str, filename: str) -> None:
        plt.figure(figsize=(12, 5))
        plt.bar(ordered["pathology"], ordered[column], color=colors)
        plt.ylabel(ylabel)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(out_dir / filename, dpi=200)
        plt.close()

    bar_plot("ap", "Average Precision", "per_class_ap.png")
    bar_plot("f1", "F1 at threshold 0.5", "per_class_f1.png")

    for metric, filename, ylabel in [
        ("ap", "class_frequency_vs_ap.png", "Average Precision"),
        ("f1", "class_frequency_vs_f1.png", "F1 at threshold 0.5"),
    ]:
        plt.figure(figsize=(7, 5))
        for group, group_df in per_class.groupby("group"):
            plt.scatter(group_df["training_positive_count"], group_df[metric], label=group, color=GROUP_COLORS.get(group, "gray"), s=60)
            for _, row in group_df.iterrows():
                plt.text(row["training_positive_count"], row[metric], row["pathology"], fontsize=8)
        plt.xscale("log")
        plt.xlabel("Positive training images, log scale")
        plt.ylabel(ylabel)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / filename, dpi=200)
        plt.close()

    group_metrics = per_class.groupby("group", sort=False)[["ap", "f1"]].mean(numeric_only=True).reindex(["head", "medium", "tail"])
    plt.figure(figsize=(6, 4))
    plt.bar(group_metrics.index, group_metrics["ap"], color=[GROUP_COLORS.get(g, "gray") for g in group_metrics.index])
    plt.ylabel("mAP")
    plt.tight_layout()
    plt.savefig(out_dir / "group_map_comparison.png", dpi=200)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.bar(group_metrics.index, group_metrics["f1"], color=[GROUP_COLORS.get(g, "gray") for g in group_metrics.index])
    plt.ylabel("Macro-F1 at threshold 0.5")
    plt.tight_layout()
    plt.savefig(out_dir / "group_f1_comparison.png", dpi=200)
    plt.close()


def plot_training_history(out_dir: Path) -> None:
    path = out_dir / "training_history.csv"
    if not path.exists():
        return
    history = pd.read_csv(path)
    if history.empty:
        return
    plt.figure(figsize=(8, 5))
    plt.plot(history["epoch"], history["train_loss"], label="train loss")
    plt.plot(history["epoch"], history["val_loss"], label="validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "training_curves.png", dpi=200)
    plt.close()


def train(args: argparse.Namespace) -> None:
    if args.mode == "train":
        validate_run(args)
    set_seed(args.seed)
    train_df = load_manifest(args.train_manifest, args.max_train_samples)
    val_df = load_manifest(args.val_manifest, args.max_val_samples)
    report = split_integrity_report(train_df, val_df, args)
    assert_split_integrity(report)
    if args.mode == "check-splits":
        print("Patient and image split integrity passed; no run files written.")
        return

    class_freq, class_groups = class_frequencies_and_groups(train_df, val_df)
    if not args.resume:
        provenance = {"training_protocol": PROTOCOL, "input_sha256": input_hashes(args)}
        atomic_json(report, args.output_dir / "split_integrity_report.json")
        class_freq.to_csv(args.output_dir / "class_frequencies.csv", index=False)
        atomic_json(class_groups, args.output_dir / "class_groups.json")
        config = {**vars(args), **provenance, "method": "bce_baseline", "method_hyperparameters": {},
                  "preprocessing_description": PREPROCESSING_DESCRIPTION, "pathology_order": PATHOLOGIES}
        atomic_json(config, args.output_dir / "run_config.json")
        atomic_json(config, args.output_dir / "config.json")

    device = select_device(args)
    print(f"Using device: {device}")
    train_loader = make_loader(train_df, training=True, args=args)
    val_loader = make_loader(val_df, training=False, args=args)
    model = make_model(pretrained=not args.resume).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and not args.no_amp))

    if args.check_batches and not args.resume:
        check_batches(model, train_loader, device, criterion, args)

    start_epoch = 1
    best_map = -float("inf")
    best_epoch = 0
    last_path = args.output_dir / "checkpoint_final.pt"
    best_path = args.output_dir / "checkpoint_best.pt"
    legacy_last_path = args.output_dir / "last_model.pt"
    legacy_best_path = args.output_dir / "best_model.pt"
    history_path = args.output_dir / "training_history.csv"
    training_log_path = args.output_dir / "training_log.csv"
    history_rows = []
    stale_epochs = 0
    best_metrics = {}
    loaders = [train_loader, val_loader]
    if args.resume:
        checkpoint = load_resume(last_path, model, optimizer, scheduler, scaler, device)
        start_epoch = int(checkpoint["epoch"]) + 1
        best_map = float(checkpoint["best_validation_mAP"])
        state = checkpoint["resume_state"]
        best_epoch, best_metrics = state["best_epoch"], state["best_metrics"]
        stale_epochs, history_rows = state["stale_epochs"], state["history_rows"]
        restore_rng(state["rng"], loaders)
        print(f"Resumed from {last_path} at epoch {start_epoch}.")
    for epoch in range(start_epoch, args.epochs + 1):
        if stale_epochs >= args.patience:
            break
        tic = time.time()
        train_loss = train_one_epoch(model, train_loader, device, criterion, optimizer, scaler, args)
        val_metrics, val_per_class, degradation, logits, labels, names = evaluate(model, val_loader, device, criterion, class_freq, args)
        scheduler.step(val_metrics["mAP"] if math.isfinite(val_metrics["mAP"]) else -1.0)
        val_metrics["epoch"] = epoch
        improved = val_metrics["mAP"] > best_map
        if improved:
            best_map = val_metrics["mAP"]
            best_epoch = epoch
            val_metrics["selected_epoch"] = epoch
            best_metrics = dict(val_metrics)
            save_checkpoint(best_path, model, optimizer, scheduler, epoch, best_map, class_groups, args, scaler)
            shutil.copy2(best_path, legacy_best_path)
            save_metric_outputs("validation", args.output_dir, val_metrics, val_per_class, degradation, logits, labels, names)
            stale_epochs = 0
        else:
            stale_epochs += 1
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_mAP": val_metrics["mAP"],
            "val_macro_f1": val_metrics["macro_f1"],
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": time.time() - tic,
            "best_epoch": best_epoch,
        }
        history_rows.append(row)
        state = {"best_epoch": best_epoch, "best_metrics": best_metrics, "stale_epochs": stale_epochs,
                 "history_rows": history_rows, "rng": capture_rng(loaders)}
        save_checkpoint(last_path, model, optimizer, scheduler, epoch, best_map, class_groups, args, scaler, state)
        shutil.copy2(last_path, legacy_last_path)
        pd.DataFrame(history_rows).to_csv(history_path, index=False)
        pd.DataFrame(history_rows)[["epoch", "train_loss", "val_loss", "val_mAP"]].to_csv(training_log_path, index=False)
        plot_training_history(args.output_dir)
        print(f"Epoch {epoch:03d}: train_loss={train_loss:.4f} val_loss={val_metrics['loss']:.4f} val_mAP={val_metrics['mAP']:.4f} val_macro_f1={val_metrics['macro_f1']:.4f}")
        if stale_epochs >= args.patience:
            print(f"Early stopping after {stale_epochs} epoch(s) without validation mAP improvement.")
            break
    print(f"Best validation mAP: {best_map} at epoch {best_epoch}.")


def evaluate_validation(args: argparse.Namespace) -> None:
    require_mutable_run(args.output_dir)
    set_seed(args.seed)
    train_df = load_manifest(args.train_manifest, args.max_train_samples)
    val_df = load_manifest(args.val_manifest, args.max_val_samples)
    report = split_integrity_report(train_df, val_df, args)
    assert_split_integrity(report)
    class_freq, _ = class_frequencies_and_groups(train_df, val_df)
    device = select_device(args)
    model = make_model(pretrained=False).to(device)
    checkpoint_path = args.output_dir / "checkpoint_best.pt"
    if not checkpoint_path.exists():
        checkpoint_path = args.output_dir / "best_model.pt"
    if not checkpoint_path.exists():
        raise SystemExit(f"Best checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if (checkpoint.get("training_protocol"), checkpoint.get("method"), checkpoint.get("random_seed")) != (PROTOCOL, "bce_baseline", args.seed):
        raise ValueError("Validation evaluation requires a corrected BCE checkpoint with the requested seed")
    if checkpoint.get("pathology_order") != PATHOLOGIES or checkpoint.get("input_sha256") != input_hashes(args):
        raise ValueError("Checkpoint data inputs or pathology order differ from this evaluation")
    model.load_state_dict(checkpoint["model_state_dict"])
    loader = make_loader(val_df, training=False, args=args)
    criterion = nn.BCEWithLogitsLoss()
    metrics, per_class, degradation, logits, labels, names = evaluate(model, loader, device, criterion, class_freq, args)
    metrics["selected_epoch"] = checkpoint.get("epoch")
    save_metric_outputs("validation", args.output_dir, metrics, per_class, degradation, logits, labels, names)
    print(f"Validation metrics saved to {args.output_dir}")


def main() -> None:
    args = parse_args()
    default_paths(args)
    if args.mode in {"train", "check-splits"}:
        train(args)
    elif args.mode == "evaluate-val":
        evaluate_validation(args)
    elif args.mode == "plot-only":
        per_class_path = args.output_dir / "validation_per_class_metrics.csv"
        if not per_class_path.exists():
            raise SystemExit(f"Missing validation metrics: {per_class_path}")
        make_plots(args.output_dir, pd.read_csv(per_class_path))
        plot_training_history(args.output_dir)
        print(f"Plots updated in {args.output_dir}")


if __name__ == "__main__":
    main()
