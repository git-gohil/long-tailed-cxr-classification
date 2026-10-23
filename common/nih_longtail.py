#!/usr/bin/env python3
"""Shared training utilities for NIH ChestX-ray14 long-tail experiments.

The baseline in ``2. Degraded Performance`` is the source of truth for the
model, manifests, augmentations, optimiser, scheduler, early stopping and
metrics.  This module keeps those choices in one reusable place while allowing
each experiment to change only its loss, sampler, or classifier-training stage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import shutil
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler, WeightedRandomSampler
from common.metrics import average_precision, precision_recall_f1
from common.run_state import PROTOCOL, validate_run, require_mutable_run, input_hashes, capture_rng, restore_rng, atomic_checkpoint

DB_BIAS_VERSION = 2


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

PREPROCESSING_DESCRIPTION = {
    "image_source": "Original NIH image paths from preprocessing manifests; images are not copied or modified.",
    "training": [
        "open image from manifest image_path",
        "convert to grayscale, then RGB/three channels",
        "resize to 256 x 256",
        "random crop 224 x 224",
        "random horizontal flip p=0.5",
        "random rotation in [-10, 10] degrees",
        "brightness and contrast jitter 0.2",
        "convert to tensor",
        "ImageNet normalization",
    ],
    "validation": [
        "open image from manifest image_path",
        "convert to grayscale, then RGB/three channels",
        "resize to 256 x 256",
        "center crop 224 x 224",
        "convert to tensor",
        "ImageNet normalization",
    ],
}

GROUP_COLORS = {"head": "#2f6fbb", "medium": "#4f9f5f", "tail": "#c96b3c"}


def project_default() -> Path:
    return Path("/home-mscluster/pgohil/cxr-research")


def add_common_args(parser: argparse.ArgumentParser, method: str, folder: str) -> None:
    parser.add_argument("--mode", choices=["train", "evaluate-val", "check-splits"], default="train")
    parser.add_argument("--project-root", type=Path, default=project_default())
    parser.add_argument("--dataset-root", type=Path, default=Path("/datasets/pgohil"))
    parser.add_argument("--metadata-csv", type=Path, default=None)
    parser.add_argument("--test-list", type=Path, default=Path("/datasets/pgohil/test_list.txt"))
    parser.add_argument("--train-manifest", type=Path, default=None)
    parser.add_argument("--val-manifest", type=Path, default=None)
    parser.add_argument("--class-groups", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--experiment-folder", default=folder)
    parser.add_argument("--method", default=method)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-overwrite", action="store_true", help="Deprecated; occupied run directories are always protected.")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--check-batches", type=int, default=0)


def fill_default_paths(args: argparse.Namespace) -> None:
    data_dir = args.project_root / "1. Data analysis + inspection" / "results" / "data"
    args.train_manifest = args.train_manifest or data_dir / "train_manifest.csv"
    args.val_manifest = args.val_manifest or data_dir / "val_manifest.csv"
    args.class_groups = args.class_groups or data_dir / "class_groups.csv"
    args.metadata_csv = args.metadata_csv or args.dataset_root / "Data_Entry_2017.csv"
    args.output_dir = args.output_dir or args.project_root / "runs" / args.method / f"seed_{args.seed}"
    if args.smoke_test:
        args.output_dir = args.project_root / args.experiment_folder / "results" / "smoke_test"
        args.epochs = min(args.epochs, 2)
        args.max_train_samples = args.max_train_samples or 64
        args.max_val_samples = args.max_val_samples or 64
        args.check_batches = max(args.check_batches, 1)


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


def to_jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


def write_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(to_jsonable(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def print_config(args: argparse.Namespace, extras: dict | None = None) -> None:
    config = {**vars(args), **(extras or {})}
    print("Resolved configuration:")
    print(json.dumps(to_jsonable(config), indent=2, sort_keys=True))


def load_manifest(path: Path, limit: int | None = None) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Manifest not found: {path}")
    df = pd.read_csv(path)
    required = {"image_filename", "image_path", "patient_id", *PATHOLOGIES}
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f"{path} is missing columns: {missing}")
    for pathology in PATHOLOGIES:
        values = set(pd.Series(df[pathology]).dropna().unique())
        if df[pathology].isna().any() or not values <= {0, 1}:
            raise SystemExit(f"{path} column {pathology} contains non-binary labels: {values}")
    if df.empty or df["image_filename"].duplicated().any() or df["patient_id"].isna().any():
        raise ValueError(f"Empty manifest, duplicate images, or missing patient IDs: {path}")
    if limit:
        df = df.head(limit).copy()
    return df.reset_index(drop=True)


def load_test_filenames(path: Path) -> set[str]:
    if not path.exists():
        raise SystemExit(f"Test list not found: {path}")
    return {Path(line.strip()).name for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()}


def split_integrity_report(train: pd.DataFrame, val: pd.DataFrame, test_list: Path, metadata_csv: Path) -> dict:
    test_names = load_test_filenames(test_list)
    if not test_names:
        raise ValueError(f"Empty test exclusion list: {test_list}")
    # Read identifiers only; held-out labels are not used for training decisions.
    metadata = pd.read_csv(metadata_csv, usecols=lambda col: re.sub(r"[^a-z0-9]", "", col.lower()) in {"imageindex", "patientid"})
    metadata.columns = [re.sub(r"[^a-z0-9]", "", col.lower()) for col in metadata.columns]
    if not {"imageindex", "patientid"} <= set(metadata.columns):
        raise ValueError(f"Missing image/patient identifiers in {metadata_csv}")
    test = metadata[metadata["imageindex"].isin(test_names)]
    if set(test["imageindex"]) != test_names or test["imageindex"].duplicated().any():
        raise ValueError("Missing or duplicate test identifiers in metadata")

    def patients(values):
        numeric = pd.to_numeric(values, errors="raise")
        if numeric.isna().any() or not np.isfinite(numeric).all() or (numeric < 0).any() or (numeric % 1 != 0).any():
            raise ValueError("Missing or invalid NIH patient IDs")
        return set(numeric.astype(np.int64))

    train_patients, val_patients, test_patients = patients(train["patient_id"]), patients(val["patient_id"]), patients(test["patientid"])
    train_files = set(train["image_filename"])
    val_files = set(val["image_filename"])
    report = {
        "train_images": len(train),
        "validation_images": len(val),
        "test_filenames_in_official_list": len(test_names),
        "train_validation_image_overlap": sorted(train_files & val_files),
        "train_test_image_overlap": sorted(train_files & test_names),
        "validation_test_image_overlap": sorted(val_files & test_names),
        "train_validation_patient_overlap": sorted(train_patients & val_patients),
        "train_test_patient_overlap": sorted(train_patients & test_patients),
        "validation_test_patient_overlap": sorted(val_patients & test_patients),
    }
    report["passed"] = not any(report[k] for k in report if k.endswith("_overlap"))
    return report


def load_class_groups(path: Path, train_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    if path.exists():
        groups_df = pd.read_csv(path)
        required = {"pathology", "train_positive_count", "group"}
        missing = sorted(required - set(groups_df.columns))
        if missing:
            raise SystemExit(f"{path} is missing columns: {missing}")
        groups_df = groups_df[["pathology", "train_positive_count", "group"]].copy()
    else:
        warnings.warn(f"Class group file not found: {path}; deriving groups from training counts only.")
        rows = []
        for p in PATHOLOGIES:
            count = int(train_df[p].sum())
            group = "head" if count > 5000 else "medium" if count >= 1000 else "tail"
            rows.append({"pathology": p, "train_positive_count": count, "group": group})
        groups_df = pd.DataFrame(rows)
    missing_classes = sorted(set(PATHOLOGIES) - set(groups_df["pathology"]))
    if missing_classes:
        raise SystemExit(f"Class group file is missing pathologies: {missing_classes}")
    if groups_df["pathology"].duplicated().any() or not set(groups_df["group"]) <= {"head", "medium", "tail"}:
        raise ValueError("Class group file contains duplicate classes or invalid groups")
    groups_df = groups_df.set_index("pathology").loc[PATHOLOGIES].reset_index()
    groups = {g: groups_df.loc[groups_df["group"] == g, "pathology"].tolist() for g in ("head", "medium", "tail")}
    return groups_df, groups


def make_transform(training: bool):
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode

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
        from PIL import Image, ImageOps

        row = self.manifest.iloc[index]
        with Image.open(row["image_path"]) as img:
            image = ImageOps.grayscale(img).convert("RGB")
            image = self.transform(image)
        labels = torch.tensor(row[PATHOLOGIES].to_numpy(dtype=np.float32), dtype=torch.float32)
        return image, labels, row["image_filename"]


def make_model(pretrained: bool = True) -> nn.Module:
    from torchvision import models

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


def make_loader(df: pd.DataFrame, training: bool, args: argparse.Namespace, sampler: Sampler | None = None) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(args.seed + (1 if training else 1001))
    return DataLoader(
        NIHManifestDataset(df, training=training),
        batch_size=args.batch_size,
        shuffle=bool(training and sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def sigmoid_np(logits: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(logits, -80, 80)))


def safe_mean(values) -> float:
    arr = np.array(list(values), dtype=float)
    return float(np.nanmean(arr)) if np.isfinite(arr).any() else float("nan")


def calculate_metrics(labels: np.ndarray, logits: np.ndarray, class_freq: pd.DataFrame, threshold: float, loss: float) -> tuple[dict, pd.DataFrame]:
    probs = sigmoid_np(logits)
    pred = (probs >= threshold).astype(int)
    counts = class_freq.set_index("pathology")
    rows = []
    for idx, pathology in enumerate(PATHOLOGIES):
        y = labels[:, idx].astype(int)
        p = pred[:, idx].astype(int)
        tp = int(((p == 1) & (y == 1)).sum())
        fp = int(((p == 1) & (y == 0)).sum())
        fn = int(((p == 0) & (y == 1)).sum())
        precision, recall, f1 = precision_recall_f1(tp, fp, fn)
        if y.sum() == 0:
            warnings.warn(f"{pathology} has no positives in this split; AP is undefined, F1 uses zero-division=0.")
        rows.append({
            "pathology": pathology,
            "training_positive_count": int(counts.loc[pathology, "train_positive_count"]),
            "group": str(counts.loc[pathology, "group"]),
            "positive_count": int(y.sum()),
            "ap": average_precision(y, probs[:, idx]),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })
    per_class = pd.DataFrame(rows)
    metrics = {"loss": float(loss), "mAP": safe_mean(per_class["ap"]), "macro_f1": safe_mean(per_class["f1"]), "threshold": threshold}
    for group in ("head", "medium", "tail"):
        group_rows = per_class[per_class["group"] == group]
        metrics[f"{group}_mAP"] = safe_mean(group_rows["ap"])
        metrics[f"{group}_macro_f1"] = safe_mean(group_rows["f1"])
    return metrics, per_class


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
    metrics, per_class = calculate_metrics(labels_np, logits_np, class_freq, args.threshold, loss_sum / sample_count if sample_count else float("nan"))
    return metrics, per_class, logits_np, labels_np, names


class SigmoidFocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float | None = None):
        super().__init__()
        if gamma < 0:
            raise ValueError("gamma must be >= 0")
        if alpha is not None and not 0 <= alpha <= 1:
            raise ValueError("alpha must be in [0, 1]")
        self.gamma = float(gamma)
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        loss = bce * (1 - p_t).pow(self.gamma)
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = loss * alpha_t
        return loss.mean()


class DistributionBalancedLoss(nn.Module):
    """Distribution-Balanced Loss for multi-label imbalance.

    Implements Eq. 10 from Wu et al. (ECCV 2020): mapped co-occurrence-aware
    weights are applied to every class term, while negative-tolerant
    regularisation shifts all logits, scales negative logits, and compensates
    negative weights by the same scale factor.
    """

    def __init__(
        self,
        class_counts: np.ndarray,
        total_samples: int,
        map_alpha: float = 10.0,
        map_beta: float = 0.2,
        map_gamma: float = 0.1,
        neg_scale: float = 2.0,
        init_bias: float = 0.05,
        eps: float = 1e-6,
    ):
        super().__init__()
        if neg_scale <= 0:
            raise ValueError("neg_scale must be > 0")
        counts = torch.tensor(np.maximum(class_counts.astype(np.float32), 1.0))
        freq_inv = 1.0 / counts
        prior = torch.clamp(counts / max(float(total_samples), 1.0), eps, 1 - eps)
        self.neg_scale = float(neg_scale)
        self.register_buffer("freq_inv", freq_inv)
        self.register_buffer(
            "init_bias",
            # Keep prior bias strength fixed across negative-scale candidates.
            -torch.log((1 - prior) / prior) * float(init_bias),
        )
        self.map_alpha = float(map_alpha)
        self.map_beta = float(map_beta)
        self.map_gamma = float(map_gamma)
        self.eps = float(eps)

    def rebalance_weights(self, targets: torch.Tensor) -> torch.Tensor:
        # Per-class frequencies are calculated from the training manifest only.
        # For image k, positive_mass is sum_j y_j^k / n_j, so co-occurring
        # positive labels share the expected sampling frequency denominator.
        positive_mass = (targets * self.freq_inv).sum(dim=1, keepdim=True).clamp_min(self.eps)
        # These project argument names predate the paper-aligned names: alpha is
        # the sigmoid slope, beta is the midpoint, and gamma is the lift.
        ratios = self.freq_inv.view(1, -1) / positive_mass
        mapped = torch.sigmoid(self.map_alpha * (ratios - self.map_beta)) + self.map_gamma
        # Eq. 5 applies the mapped weight to positive and negative class terms.
        return mapped

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        adjusted_logits = logits + self.init_bias.view(1, -1)
        weights = self.rebalance_weights(targets)
        negative_mask = 1.0 - targets
        adjusted_logits = adjusted_logits * targets + adjusted_logits * negative_mask * self.neg_scale
        weights = weights * targets + weights * negative_mask / self.neg_scale
        raw = nn.functional.binary_cross_entropy_with_logits(adjusted_logits, targets, reduction="none")
        return (raw * weights).mean()


def labels_array(df: pd.DataFrame) -> np.ndarray:
    return df[PATHOLOGIES].to_numpy(dtype=np.float32)


def class_counts(df: pd.DataFrame) -> np.ndarray:
    return labels_array(df).sum(axis=0).astype(np.float32)


def square_root_weights(df: pd.DataFrame, exponent: float = 0.5, no_positive_weight: float | None = None) -> np.ndarray:
    if not math.isfinite(exponent) or exponent < 0:
        raise ValueError("sampling exponent must be finite and nonnegative")
    counts = np.maximum(class_counts(df), 1.0)
    # No-positive images are common in NIH. The median-count default keeps them
    # present without letting them dominate the sampler.
    default = no_positive_weight if no_positive_weight is not None else float(np.median(counts) ** (-exponent))
    if not math.isfinite(default) or default <= 0:
        raise ValueError("no-positive weight must be finite and positive")
    weights = []
    for row in labels_array(df):
        pos = np.where(row > 0)[0]
        if len(pos) == 0:
            weights.append(default)
        else:
            weights.append(float(1.0 / (counts[pos].min() ** exponent)))
    weights = np.array(weights, dtype=np.float64)
    return weights / weights.mean()


def save_sampling_summary(df: pd.DataFrame, weights: np.ndarray, out_dir: Path, exponent: float) -> None:
    labels = labels_array(df)
    expected = (labels * weights[:, None]).sum(axis=0) / weights.sum()
    summary = {
        "exponent": exponent,
        "num_images": int(len(weights)),
        "min": float(weights.min()),
        "max": float(weights.max()),
        "mean": float(weights.mean()),
        "percentiles": {str(p): float(np.percentile(weights, p)) for p in [1, 5, 25, 50, 75, 95, 99]},
        "expected_sampled_positive_fraction": dict(zip(PATHOLOGIES, expected.tolist())),
        "expected_all_negative_fraction": float(weights[labels.sum(axis=1) == 0].sum() / weights.sum()),
    }
    write_json(summary, out_dir / "sampling_weights_summary.json")
    write_json(dict(zip(PATHOLOGIES, class_counts(df).astype(int).tolist())), out_dir / "class_counts_used.json")


class ClassAwareSampler(Sampler[int]):
    """DB class-aware sampling: shuffled cycles, three draws per class visit.

    Follows the authors' sampler with a floor(C * largest_class_count / 4)
    epoch budget. Empty classes are skipped for subsets; tiny subsets get at
    least one draw. All-negative images belong to no class pool.
    """

    def __init__(self, df: pd.DataFrame, seed: int = 42):
        labels = labels_array(df)
        self.buckets = {
            c: np.flatnonzero(labels[:, c] == 1).tolist()
            for c in range(len(PATHOLOGIES)) if np.any(labels[:, c] == 1)
        }
        if not self.buckets:
            raise ValueError("DB class-aware sampling requires at least one positive label.")
        self.length = max(1, max(map(len, self.buckets.values())) * len(self.buckets) // 4)
        self.no_positive_count = int((labels.sum(axis=1) == 0).sum())
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        return self.length

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)

        def shuffled_cycle(values):
            values = list(values)
            while True:
                rng.shuffle(values)
                yield from values

        classes = shuffled_cycle(self.buckets)
        images = {c: shuffled_cycle(indices) for c, indices in self.buckets.items()}
        for start in range(0, self.length, 3):
            selected = next(classes)
            for _ in range(min(3, self.length - start)):
                yield next(images[selected])


class RarestClassBalancedSampler(Sampler[int]):
    def __init__(self, df: pd.DataFrame, seed: int = 42, include_no_positive: bool = True):
        self.rng = random.Random(seed)
        counts = np.maximum(class_counts(df), 1.0)
        buckets: dict[str, list[int]] = {p: [] for p in PATHOLOGIES}
        self.no_positive: list[int] = []
        for idx, row in enumerate(labels_array(df)):
            pos = np.where(row > 0)[0]
            if len(pos) == 0:
                self.no_positive.append(idx)
                continue
            rarest = int(pos[np.argmin(counts[pos])])
            buckets[PATHOLOGIES[rarest]].append(idx)
        self.buckets = {k: v for k, v in buckets.items() if v}
        if include_no_positive and self.no_positive:
            self.buckets["no_positive"] = self.no_positive
        self.keys = sorted(self.buckets)
        self.length = len(df)
        if not self.keys:
            raise ValueError("Balanced sampler has no non-empty buckets.")

    def __len__(self) -> int:
        return self.length

    def __iter__(self):
        for _ in range(self.length):
            key = self.rng.choice(self.keys)
            yield self.rng.choice(self.buckets[key])


class PartnerCycle:
    def __init__(self, loader: DataLoader):
        self.loader = loader
        self.iterator = iter(loader)

    def next(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            return next(self.iterator)


def trainable_report(model: nn.Module) -> tuple[list[str], list[str]]:
    trainable, frozen = [], []
    for name, param in model.named_parameters():
        (trainable if param.requires_grad else frozen).append(name)
    return trainable, frozen


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save_checkpoint(path: Path, model, optimizer, scheduler, scaler, epoch: int, best_map: float, class_groups: dict, args: argparse.Namespace, extras: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_checkpoint({
        "training_protocol": PROTOCOL,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "amp_scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "best_validation_mAP": best_map,
        "random_seed": args.seed,
        "method": args.method,
        "method_hyperparameters": method_hyperparameters(args),
        "class_names": PATHOLOGIES,
        "class_groups": class_groups,
        "training_args": vars(args),
        "preprocessing_description": PREPROCESSING_DESCRIPTION,
        **(extras or {}),
    }, path)


def load_resume(path: Path, model, optimizer, scheduler, scaler, device) -> dict:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("training_protocol") != PROTOCOL or "resume_state" not in checkpoint:
        raise ValueError("Checkpoint lacks corrected resume state; start a fresh run")
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and checkpoint.get("optimizer_state_dict"):
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint.get("scheduler_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if scaler is not None and checkpoint.get("amp_scaler_state_dict"):
        scaler.load_state_dict(checkpoint["amp_scaler_state_dict"])
    return checkpoint


def method_hyperparameters(args: argparse.Namespace) -> dict:
    keys = [
        "gamma", "alpha", "db_map_alpha", "db_map_beta", "db_map_gamma", "db_neg_scale",
        "db_init_bias", "db_bias_version", "sampling_exponent", "no_positive_weight", "retain_classifier",
        "mixup_alpha", "disable_mixup",
    ]
    return {k: getattr(args, k) for k in keys if hasattr(args, k)}


def build_criterion(args: argparse.Namespace, train_df: pd.DataFrame) -> nn.Module:
    if args.method == "focal_loss":
        return SigmoidFocalLoss(gamma=args.gamma, alpha=args.alpha)
    if args.method == "db_loss":
        if getattr(args, "db_bias_version", None) != DB_BIAS_VERSION:
            raise ValueError("DB checkpoint uses the old bias formula; train a fresh DB run")
        return DistributionBalancedLoss(
            class_counts(train_df),
            len(train_df),
            map_alpha=args.db_map_alpha,
            map_beta=args.db_map_beta,
            map_gamma=args.db_map_gamma,
            neg_scale=args.db_neg_scale,
            init_bias=args.db_init_bias,
        )
    return nn.BCEWithLogitsLoss()


def make_train_sampler(args: argparse.Namespace, train_df: pd.DataFrame, out_dir: Path) -> Sampler | None:
    if args.method == "db_loss":
        sampler = ClassAwareSampler(train_df, seed=args.seed)
        write_json({
            "sampler": "class_aware",
            "class_pool_sizes": {PATHOLOGIES[c]: len(pool) for c, pool in sampler.buckets.items()},
            "samples_per_epoch": len(sampler),
            "samples_per_class_visit": 3,
            "epoch_budget": "max(1, floor(nonempty_classes * largest_class_count / 4))",
            "excluded_all_negative_images": sampler.no_positive_count,
            "seed": args.seed,
            "epoch_seed": "seed + epoch - 1 (training epochs start at 1)",
        }, out_dir / "class_aware_sampling.json")
        print(f"DB class-aware sampling: {len(sampler)} draws/epoch; {sampler.no_positive_count} all-negative images excluded.")
        return sampler
    if args.method == "sqrt_sampling":
        weights = square_root_weights(train_df, args.sampling_exponent, args.no_positive_weight)
        save_sampling_summary(train_df, weights, out_dir, args.sampling_exponent)
        print("Sampling weight summary saved.")
        return WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), num_samples=len(weights), replacement=True)
    if args.method == "crt" and getattr(args, "stage", "stage2") == "stage2":
        return RarestClassBalancedSampler(train_df, seed=args.seed)
    return None


def configure_crt_model(model: nn.Module, args: argparse.Namespace, device: torch.device) -> dict:
    source = args.source_checkpoint
    if not source.exists():
        raise SystemExit(f"BCE source checkpoint not found: {source}. Train Folder 2 baseline first or pass --source-checkpoint.")
    checkpoint = torch.load(source, map_location=device, weights_only=False)
    if checkpoint.get("training_protocol") != PROTOCOL or checkpoint.get("random_seed") != args.seed:
        raise ValueError("cRT requires a corrected, seed-matched BCE checkpoint")
    if checkpoint.get("method", "bce_baseline") != "bce_baseline":
        raise ValueError("cRT source must be a BCE baseline checkpoint")
    if checkpoint.get("class_names", checkpoint.get("pathology_order")) != PATHOLOGIES:
        raise ValueError("cRT source has a different pathology order")
    source_inputs = checkpoint.get("input_sha256", {})
    for key in ("train_manifest", "val_manifest", "test_list", "metadata_csv"):
        path = Path(getattr(args, key)).resolve()
        if source_inputs.get(str(path)) != sha256_file(path):
            raise ValueError(f"cRT source was trained with different data inputs: {path}")
    model.load_state_dict(checkpoint["model_state_dict"])
    if len(PATHOLOGIES) != model.fc.out_features:
        raise SystemExit(f"Expected 14 output classes, got {model.fc.out_features}.")
    if not args.retain_classifier:
        model.fc.reset_parameters()
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("fc.")
    trainable, frozen = trainable_report(model)
    if any(not name.startswith("fc.") for name in trainable):
        raise SystemExit(f"Backbone parameters are trainable: {trainable}")
    return {
        "source_checkpoint": str(source),
        "source_checkpoint_sha256": sha256_file(source),
        "retained_classifier": bool(args.retain_classifier),
        "trainable_parameters": trainable,
        "frozen_parameters": frozen,
    }


def check_batches(model, loader, device, criterion, args) -> None:
    model.eval()
    checked = 0
    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            if list(logits.shape) != [labels.shape[0], len(PATHOLOGIES)]:
                raise SystemExit(f"Unexpected logits shape: {list(logits.shape)}")
            loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise SystemExit("Smoke-test loss is not finite.")
            checked += 1
            if checked >= args.check_batches:
                break
    print(f"Batch check passed for {checked} batch(es).")


def optimizer_for(model: nn.Module, args: argparse.Namespace):
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)


def balanced_mixup_batch(images, labels, balanced_images, balanced_labels, alpha: float):
    balanced_weight = float(np.random.beta(alpha, 1.0))
    ordinary_weight = 1.0 - balanced_weight
    mixed_images = ordinary_weight * images + balanced_weight * balanced_images
    mixed_labels = torch.clamp(ordinary_weight * labels + balanced_weight * balanced_labels, 0.0, 1.0)
    return mixed_images, mixed_labels, balanced_weight


def train_one_epoch(model, loader, device, criterion, optimizer, scaler, args, partner_loader: DataLoader | None = None, mixup_stats: dict | None = None):
    model.train()
    if args.method == "crt" and getattr(args, "stage", "stage2") == "stage2":
        # cRT Stage 2 is classifier-only: keep backbone state, including
        # BatchNorm running statistics, fixed while the classifier trains.
        model.eval()
        model.fc.train()
    loss_sum, sample_count = 0.0, 0
    use_amp = scaler is not None and scaler.is_enabled()
    partners = PartnerCycle(partner_loader) if partner_loader is not None else None
    for images, labels, _ in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if partners is not None and not args.disable_mixup:
            other_images, other_labels, _ = partners.next()
            other_images = other_images.to(device, non_blocking=True)
            other_labels = other_labels.to(device, non_blocking=True)
            if len(other_images) != len(images):
                n = min(len(images), len(other_images))
                images = images[:n]
                labels = labels[:n]
                other_images = other_images[:n]
                other_labels = other_labels[:n]
            images, labels, lam = balanced_mixup_batch(
                images, labels, other_images, other_labels, args.mixup_alpha
            )
            if mixup_stats is not None:
                mixup_stats.setdefault("lambda_values", []).append(lam)
                mixup_stats["partner_class_counts"] = (np.array(mixup_stats.get("partner_class_counts", np.zeros(len(PATHOLOGIES)))) + other_labels.detach().cpu().numpy().sum(axis=0)).tolist()
        optimizer.zero_grad(set_to_none=True)
        try:
            with torch.amp.autocast("cuda", enabled=(use_amp and device.type == "cuda")):
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
                raise SystemExit("CUDA out of memory. Reduce --batch-size and resume.") from exc
            raise
        loss_sum += float(loss.item()) * labels.shape[0]
        sample_count += labels.shape[0]
    return loss_sum / sample_count if sample_count else float("nan")


def save_metric_outputs(out_dir: Path, metrics: dict, per_class: pd.DataFrame, logits: np.ndarray, labels: np.ndarray, names: list[str]) -> None:
    write_json(metrics, out_dir / "validation_metrics.json")
    per_class.to_csv(out_dir / "validation_per_class_metrics.csv", index=False)
    np.savez_compressed(out_dir / "validation_predictions.npz", logits=logits, labels=labels, image_filenames=np.array(names))


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
    plt.savefig(out_dir / "training_validation_loss_curves.png", dpi=200)
    plt.close()
    plt.figure(figsize=(8, 5))
    plt.plot(history["epoch"], history["val_mAP"], label="validation mAP")
    plt.xlabel("Epoch")
    plt.ylabel("mAP")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "validation_map_curve.png", dpi=200)
    plt.close()


def write_final_summary(out_dir: Path, metrics: dict, best_epoch: int) -> None:
    lines = [
        f"Best epoch: {best_epoch}",
        f"Validation mAP: {metrics.get('mAP')}",
        f"Macro-F1 @ {metrics.get('threshold')}: {metrics.get('macro_f1')}",
        f"Head/medium/tail mAP: {metrics.get('head_mAP')} / {metrics.get('medium_mAP')} / {metrics.get('tail_mAP')}",
        f"Head/medium/tail macro-F1: {metrics.get('head_macro_f1')} / {metrics.get('medium_macro_f1')} / {metrics.get('tail_macro_f1')}",
    ]
    (out_dir / "final_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Final concise summary:")
    print("\n".join(lines))


def load_data_and_prepare(args: argparse.Namespace):
    fill_default_paths(args)
    set_seed(args.seed)
    train_df = load_manifest(args.train_manifest, args.max_train_samples)
    val_df = load_manifest(args.val_manifest, args.max_val_samples)
    report = split_integrity_report(train_df, val_df, args.test_list, args.metadata_csv)
    if not report["passed"]:
        raise SystemExit(f"Split integrity failed: {report}")
    class_freq, class_groups = load_class_groups(args.class_groups, train_df)
    return train_df, val_df, class_freq, class_groups


def run_training(args: argparse.Namespace) -> None:
    fill_default_paths(args)
    if args.mode == "train":
        validate_run(args)
    train_df, val_df, class_freq, class_groups = load_data_and_prepare(args)
    if args.mode == "check-splits":
        print("Patient and image split integrity passed; no run files written.")
        return
    extras = {"preprocessing_description": PREPROCESSING_DESCRIPTION, "class_names": PATHOLOGIES,
              "training_protocol": PROTOCOL, "input_sha256": input_hashes(args)}
    print_config(args, extras)
    if not args.resume:
        write_json({**vars(args), **extras, "method_hyperparameters": method_hyperparameters(args)}, args.output_dir / "run_config.json")
        write_json({**vars(args), **extras, "method_hyperparameters": method_hyperparameters(args)}, args.output_dir / "config.json")
        write_json(split_integrity_report(train_df, val_df, args.test_list, args.metadata_csv), args.output_dir / "split_integrity_report.json")
        class_freq.to_csv(args.output_dir / "class_frequencies.csv", index=False)
        write_json(class_groups, args.output_dir / "class_groups.json")
    device = select_device(args)
    model = make_model(pretrained=not args.resume and args.method != "crt").to(device)
    checkpoint_extras: dict = {"input_sha256": extras["input_sha256"]}
    if args.method == "crt":
        checkpoint_extras.update(configure_crt_model(model, args, device))
        write_json(checkpoint_extras, args.output_dir / "source_bce_checkpoint_info.json")
        write_json({"trainable": checkpoint_extras["trainable_parameters"], "frozen": checkpoint_extras["frozen_parameters"]}, args.output_dir / "parameter_report.json")
    criterion = build_criterion(args, train_df).to(device)
    sampler = make_train_sampler(args, train_df, args.output_dir)
    train_loader = make_loader(train_df, training=True, args=args, sampler=sampler)
    val_loader = make_loader(val_df, training=False, args=args)
    partner_loader = None
    if args.method == "balanced_mixup":
        partner_sampler = RarestClassBalancedSampler(train_df, seed=args.seed + 17)
        partner_loader = make_loader(train_df, training=True, args=args, sampler=partner_sampler)
    optimizer = optimizer_for(model, args)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and not args.no_amp))
    if args.check_batches and not args.resume:
        check_batches(model, train_loader, device, criterion, args)
    start_epoch, best_map, best_epoch = 1, -float("inf"), 0
    last_path = args.output_dir / "checkpoint_final.pt"
    best_path = args.output_dir / "checkpoint_best.pt"
    legacy_last_path = args.output_dir / "last_checkpoint.pt"
    legacy_best_path = args.output_dir / "best_model.pt"
    history_path = args.output_dir / "training_history.csv"
    training_log_path = args.output_dir / "training_log.csv"
    history_rows = []
    stale_epochs = 0
    best_metrics: dict = {}
    mixup_stats: dict = {}
    loaders = [train_loader, val_loader, partner_loader]
    if args.resume:
        checkpoint = load_resume(last_path, model, optimizer, scheduler, scaler, device)
        start_epoch = int(checkpoint["epoch"]) + 1
        best_map = float(checkpoint["best_validation_mAP"])
        state = checkpoint["resume_state"]
        best_epoch, best_metrics = state["best_epoch"], state["best_metrics"]
        stale_epochs, history_rows = state["stale_epochs"], state["history_rows"]
        mixup_stats = state["mixup_stats"]
        restore_rng(state["rng"], loaders)
        print(f"Resumed from {last_path} at epoch {start_epoch}.")
    for epoch in range(start_epoch, args.epochs + 1):
        if stale_epochs >= args.patience:
            break
        if isinstance(sampler, ClassAwareSampler):
            sampler.set_epoch(epoch - 1)
        tic = time.time()
        train_loss = train_one_epoch(model, train_loader, device, criterion, optimizer, scaler, args, partner_loader, mixup_stats)
        val_metrics, val_per_class, logits, labels, names = evaluate(model, val_loader, device, criterion, class_freq, args)
        scheduler.step(val_metrics["mAP"] if math.isfinite(val_metrics["mAP"]) else -1.0)
        val_metrics["epoch"] = epoch
        improved = val_metrics["mAP"] > best_map
        if improved:
            best_map = val_metrics["mAP"]
            best_epoch = epoch
            best_metrics = dict(val_metrics)
            best_metrics["selected_epoch"] = epoch
            save_checkpoint(best_path, model, optimizer, scheduler, scaler, epoch, best_map, class_groups, args, checkpoint_extras)
            shutil.copy2(best_path, legacy_best_path)
            save_metric_outputs(args.output_dir, best_metrics, val_per_class, logits, labels, names)
            stale_epochs = 0
        else:
            stale_epochs += 1
        history_rows.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_mAP": val_metrics["mAP"],
            "val_macro_f1": val_metrics["macro_f1"],
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": time.time() - tic,
            "best_epoch": best_epoch,
        })
        resume_state = {"best_epoch": best_epoch, "best_metrics": best_metrics, "stale_epochs": stale_epochs,
                        "history_rows": history_rows, "mixup_stats": mixup_stats, "rng": capture_rng(loaders)}
        save_checkpoint(last_path, model, optimizer, scheduler, scaler, epoch, best_map, class_groups, args,
                        {**checkpoint_extras, "resume_state": resume_state})
        shutil.copy2(last_path, legacy_last_path)
        pd.DataFrame(history_rows).to_csv(history_path, index=False)
        pd.DataFrame(history_rows)[["epoch", "train_loss", "val_loss", "val_mAP"]].to_csv(training_log_path, index=False)
        plot_training_history(args.output_dir)
        print(f"Epoch {epoch:03d}: train_loss={train_loss:.4f} val_loss={val_metrics['loss']:.4f} val_mAP={val_metrics['mAP']:.4f} val_macro_f1={val_metrics['macro_f1']:.4f}")
        if stale_epochs >= args.patience:
            print(f"Early stopping after {stale_epochs} epoch(s) without validation mAP improvement.")
            break
    if args.method == "balanced_mixup":
        lambdas = mixup_stats.get("lambda_values", [])
        write_json({
            "num_mixed_batches": len(lambdas),
            "lambda_min": min(lambdas) if lambdas else None,
            "lambda_max": max(lambdas) if lambdas else None,
            "lambda_mean": float(np.mean(lambdas)) if lambdas else None,
            "partner_class_counts": dict(zip(PATHOLOGIES, np.asarray(mixup_stats.get("partner_class_counts", np.zeros(len(PATHOLOGIES)))).tolist())),
        }, args.output_dir / "balanced_mixup_sampling_stats.json")
    write_final_summary(args.output_dir, best_metrics, best_epoch)


def run_evaluation(args: argparse.Namespace) -> None:
    fill_default_paths(args)
    require_mutable_run(args.output_dir)
    train_df, val_df, class_freq, _ = load_data_and_prepare(args)
    print_config(args)
    device = select_device(args)
    model = make_model(pretrained=False).to(device)
    checkpoint_path = args.output_dir / "checkpoint_best.pt"
    if not checkpoint_path.exists():
        checkpoint_path = args.output_dir / "best_model.pt"
    if not checkpoint_path.exists():
        raise SystemExit(f"Best checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if (checkpoint.get("training_protocol"), checkpoint.get("method"), checkpoint.get("random_seed")) != (PROTOCOL, args.method, args.seed):
        raise ValueError("Validation evaluation requires a corrected checkpoint with the requested method and seed")
    if checkpoint.get("class_names") != PATHOLOGIES or checkpoint.get("input_sha256") != input_hashes(args):
        raise ValueError("Checkpoint data inputs or pathology order differ from this evaluation")
    model.load_state_dict(checkpoint["model_state_dict"])
    for key, value in checkpoint.get("method_hyperparameters", {}).items():
        setattr(args, key, value)
    if args.method == "db_loss":
        args.db_bias_version = checkpoint.get("method_hyperparameters", {}).get("db_bias_version")
    criterion = build_criterion(args, train_df).to(device)
    loader = make_loader(val_df, training=False, args=args)
    metrics, per_class, logits, labels, names = evaluate(model, loader, device, criterion, class_freq, args)
    metrics["selected_epoch"] = checkpoint.get("epoch")
    save_metric_outputs(args.output_dir, metrics, per_class, logits, labels, names)
    print(f"Validation metrics saved to {args.output_dir}")


def main_with_args(args: argparse.Namespace) -> None:
    if args.mode == "evaluate-val":
        run_evaluation(args)
    else:
        run_training(args)
