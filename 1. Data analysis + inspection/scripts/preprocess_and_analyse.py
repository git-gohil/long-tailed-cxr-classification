#!/usr/bin/env python3
"""
Training EDA and train/validation image verification for NIH ChestX-ray14.

This script never creates a test manifest and never opens test images. Test
split files are used only as filename exclusion lists.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import random
import re
import tempfile
import warnings
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps


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
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
BLOCKED_DIR_MARKERS = ("test", "testing", "heldout", "held-out")
FILENAME_RE = re.compile(r"[^,\s;|]+?\.(?:png|jpg|jpeg)", re.IGNORECASE)


try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - optional dependency
    tqdm = None


def progress(iterable: Iterable, total: int | None = None, desc: str = "") -> Iterable:
    if tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc=desc)


def normalise_column(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.strip().lower())


def atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=path.parent, suffix=".tmp", newline="") as tmp:
        tmp_path = Path(tmp.name)
        df.to_csv(tmp, index=False)
    tmp_path.replace(path)


def atomic_write_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=path.parent, suffix=".tmp") as tmp:
        tmp_path = Path(tmp.name)
        json.dump(data, tmp, indent=2, sort_keys=True)
        tmp.write("\n")
    tmp_path.replace(path)


def is_blocked_dir(path: Path) -> bool:
    name = path.name.lower().replace("_", "-")
    return any(marker in name for marker in BLOCKED_DIR_MARKERS)


def find_files(root: Path, suffixes: set[str], include_test_named_files: bool = False) -> list[Path]:
    found: list[Path] = []
    for current, dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        dirnames[:] = [
            d for d in dirnames
            if not is_blocked_dir(current_path / d) and not (current_path / d).is_symlink()
        ]
        for filename in filenames:
            path = current_path / filename
            if path.is_symlink():
                continue
            if path.suffix.lower() in suffixes:
                if include_test_named_files or "test" not in path.name.lower():
                    found.append(path)
    return sorted(found)


def auto_find_metadata_csv(dataset_root: Path) -> Path:
    candidates = []
    for path in find_files(dataset_root, {".csv"}, include_test_named_files=False):
        try:
            with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
                header = next(csv.reader(f), [])
        except Exception:
            header = []
        header_text = " ".join(header).lower()
        score = 0
        for token in ("image index", "finding labels", "patient id", "view position"):
            if token in header_text:
                score += 3
        if "data_entry" in path.name.lower() or "metadata" in path.name.lower():
            score += 2
        if score:
            candidates.append((score, path))
    if not candidates:
        raise SystemExit("Could not auto-detect metadata CSV. Re-run with --metadata-csv PATH.")
    candidates.sort(key=lambda item: (-item[0], str(item[1])))
    logging.info("Auto-selected metadata CSV: %s", candidates[0][1])
    return candidates[0][1]


def auto_find_split_file(dataset_root: Path, kind: str) -> Path | None:
    files = find_files(dataset_root, {".txt", ".lst", ".csv"}, include_test_named_files=True)
    scored = []
    for path in files:
        name = path.name.lower()
        score = 0
        if kind == "test" and "test" in name:
            score += 5
        elif kind == "val" and "val" in name and "train" not in name and "test" not in name:
            score += 5
        elif kind == "train" and "train" in name and "test" not in name and "val" not in name:
            score += 5
        elif kind == "trainval" and "train" in name and ("val" in name or "train_val" in name) and "test" not in name:
            score += 6
        if score:
            scored.append((score, path))
    if not scored:
        return None
    scored.sort(key=lambda item: (-item[0], str(item[1])))
    logging.info("Auto-selected %s split/list file: %s", kind, scored[0][1])
    return scored[0][1]


def load_filename_list(path: Path | None) -> set[str]:
    """Extract image filenames from a split/list file without interpreting labels."""
    if path is None:
        return set()
    filenames: set[str] = set()
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            for match in FILENAME_RE.findall(line):
                filenames.add(Path(match).name)
    if not filenames:
        raise SystemExit(f"No image filenames found in split/list file: {path}")
    logging.info("Loaded %d filenames from %s", len(filenames), path)
    return filenames


def scan_image_paths(dataset_root: Path) -> dict[str, list[Path]]:
    image_paths = find_files(dataset_root, IMAGE_EXTENSIONS, include_test_named_files=False)
    by_name: dict[str, list[Path]] = defaultdict(list)
    for path in image_paths:
        by_name[path.name].append(path)
    logging.info("Found %d unique image filenames outside test-like directories.", len(by_name))
    return by_name


def read_metadata_rows(metadata_csv: Path, needed_filenames: set[str] | None, test_filenames: set[str]) -> pd.DataFrame:
    """Read metadata rows for permitted filenames only."""
    with metadata_csv.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise SystemExit(f"Metadata CSV has no header: {metadata_csv}")
        column_map = {normalise_column(c): c for c in reader.fieldnames}

        required = {
            "imageindex": "Image Index",
            "findinglabels": "Finding Labels",
        }
        missing = [pretty for key, pretty in required.items() if key not in column_map]
        if missing:
            raise SystemExit(f"Metadata CSV is missing required columns: {missing}")

        optional = {
            "patientid": "Patient ID",
            "patientage": "Patient Age",
            "patientgender": "Patient Gender",
            "viewposition": "View Position",
        }

        rows = []
        skipped_test_rows = 0
        for row in reader:
            filename = Path(row.get(column_map["imageindex"], "")).name
            if not filename:
                continue
            if filename in test_filenames:
                skipped_test_rows += 1
                continue
            if needed_filenames is not None and filename not in needed_filenames:
                continue
            item = {
                "image_filename": filename,
                "finding_labels": row.get(column_map["findinglabels"], ""),
            }
            for key, pretty in optional.items():
                source = column_map.get(key)
                item[pretty.lower().replace(" ", "_")] = row.get(source, "") if source else ""
            rows.append(item)

    logging.info("Loaded %d permitted metadata rows; skipped %d test-exclusion rows.", len(rows), skipped_test_rows)
    return pd.DataFrame(rows)


def labels_to_binary(label_string: str) -> tuple[dict[str, int], int, int]:
    labels = {part.strip() for part in str(label_string).split("|") if part.strip()}
    binary = {pathology: int(pathology in labels) for pathology in PATHOLOGIES}
    no_finding = int("No Finding" in labels)
    positive_count = sum(binary.values())
    return binary, no_finding, positive_count


def build_manifest(metadata: pd.DataFrame, image_paths_by_name: dict[str, list[Path]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    records = []
    issues = []
    duplicate_names = {name: paths for name, paths in image_paths_by_name.items() if len(paths) > 1}

    if metadata.empty:
        raise SystemExit("No permitted metadata rows were found. Check --metadata-csv and split/list arguments.")

    for _, row in metadata.iterrows():
        filename = row["image_filename"]
        paths = image_paths_by_name.get(filename, [])
        if not paths:
            issues.append({"issue_type": "metadata_image_not_found", "detail": filename, "count": 1})
            continue
        if len(paths) > 1:
            issues.append({"issue_type": "duplicate_filename_to_paths", "detail": filename, "count": len(paths)})
            continue
        binary, no_finding, positive_count = labels_to_binary(row["finding_labels"])
        record = {
            "image_filename": filename,
            "image_path": str(paths[0]),
            "patient_id": row.get("patient_id", ""),
            "original_finding_labels": row["finding_labels"],
            **binary,
            "no_finding": no_finding,
            "num_positive_pathologies": positive_count,
            "patient_age": row.get("patient_age", ""),
            "patient_gender": row.get("patient_gender", ""),
            "view_position": row.get("view_position", ""),
        }
        records.append(record)

    if issues:
        raise SystemExit(f"Required images are missing or have ambiguous duplicate paths: {issues[:5]}. "
                         "No manifests were written. Repair the dataset and rerun preprocessing.")
    manifest = pd.DataFrame(records)
    if manifest.empty:
        raise SystemExit("No manifest rows could be built. Check duplicate filenames and missing images.")

    for name, paths in duplicate_names.items():
        issues.append({"issue_type": "duplicate_filename_in_image_tree", "detail": name, "count": len(paths)})

    issues_df = pd.DataFrame(issues, columns=["issue_type", "detail", "count"])
    return manifest, issues_df


def split_by_existing_lists(manifest: pd.DataFrame, train_names: set[str], val_names: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = manifest[manifest["image_filename"].isin(train_names)].copy()
    val = manifest[manifest["image_filename"].isin(val_names)].copy()
    missing_train = train_names - set(train["image_filename"])
    missing_val = val_names - set(val["image_filename"])
    if missing_train or missing_val:
        raise SystemExit(f"Split/list filenames missing from manifest: "
                         f"train={sorted(missing_train)[:5]}, validation={sorted(missing_val)[:5]}")
    return train, val


def patient_level_split(manifest: pd.DataFrame, val_fraction: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    patient_col = "patient_id"
    if patient_col not in manifest or manifest[patient_col].astype(str).str.len().eq(0).all():
        raise SystemExit("Patient ID is required for a safe patient-level validation split.")

    patient_labels = manifest.groupby(patient_col)[PATHOLOGIES].max().reset_index()
    patients = patient_labels[patient_col].to_numpy()
    y = patient_labels[PATHOLOGIES].to_numpy(dtype=int)

    try:
        from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

        splitter = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
        train_idx, val_idx = next(splitter.split(np.zeros(len(patients)), y))
        method = "patient-level iterative multilabel stratification"
    except Exception:
        rng = np.random.default_rng(seed)
        target_val_count = max(1, round(len(patients) * val_fraction))
        best_val_idx = None
        best_score = float("inf")
        overall_prev = y.mean(axis=0)
        for _ in range(200):
            perm = rng.permutation(len(patients))
            val_idx = perm[:target_val_count]
            val_prev = y[val_idx].mean(axis=0) if len(val_idx) else np.zeros(y.shape[1])
            score = np.abs(val_prev - overall_prev).sum()
            if score < best_score:
                best_score = score
                best_val_idx = val_idx
        val_idx = np.array(best_val_idx, dtype=int)
        train_idx = np.setdiff1d(np.arange(len(patients)), val_idx)
        method = "patient-level random fallback with 200 prevalence-matching shuffles; less exact than iterative stratification"

    train_patients = set(patients[train_idx])
    val_patients = set(patients[val_idx])
    train = manifest[manifest[patient_col].isin(train_patients)].copy()
    val = manifest[manifest[patient_col].isin(val_patients)].copy()
    return train, val, method


def validate_manifests(train: pd.DataFrame, val: pd.DataFrame, test_filenames: set[str]) -> None:
    if train.empty or val.empty:
        raise SystemExit(f"Train/validation split is empty: train={len(train)}, val={len(val)}")

    train_patients = set(train["patient_id"].astype(str))
    val_patients = set(val["patient_id"].astype(str))
    patient_overlap = train_patients & val_patients
    if patient_overlap:
        raise SystemExit(f"Patient overlap between train and validation: {len(patient_overlap)}")

    train_files = set(train["image_filename"])
    val_files = set(val["image_filename"])
    if train_files & val_files:
        raise SystemExit("Filename overlap detected between train and validation.")
    if (train_files | val_files) & test_filenames:
        raise SystemExit("Detected overlap with the test filename exclusion list.")

    for name, df in (("train", train), ("validation", val)):
        missing_paths = [p for p in df["image_path"] if not Path(p).exists()]
        if missing_paths:
            raise SystemExit(f"{name} manifest has {len(missing_paths)} missing image paths.")
        for pathology in PATHOLOGIES:
            values = set(df[pathology].dropna().unique())
            if not values <= {0, 1}:
                raise SystemExit(f"{name} column {pathology} contains values other than 0/1: {values}")


def inspect_one_image(row: dict, calculate_hash: bool) -> dict:
    path = Path(row["image_path"])
    result = {
        "image_filename": row["image_filename"],
        "image_path": str(path),
        "exists": path.exists(),
        "readable": False,
        "width": None,
        "height": None,
        "mode": None,
        "extension": path.suffix.lower(),
        "file_size_bytes": None,
        "sha256": None,
        "error": "",
    }
    try:
        result["file_size_bytes"] = path.stat().st_size
        with Image.open(path) as img:
            result["mode"] = img.mode
            result["width"], result["height"] = img.size
            img.load()
        if calculate_hash:
            hasher = hashlib.sha256()
            with path.open("rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    hasher.update(chunk)
            result["sha256"] = hasher.hexdigest()
        result["readable"] = True
    except Exception as exc:
        result["error"] = str(exc)
    return result


def inspect_images(manifest: pd.DataFrame, skip_hashes: bool, num_workers: int) -> pd.DataFrame:
    rows = manifest[["image_filename", "image_path"]].to_dict("records")
    calculate_hash = not skip_hashes
    if num_workers <= 1:
        return pd.DataFrame([inspect_one_image(row, calculate_hash) for row in progress(rows, desc="Inspecting images")])
    results = []
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = [pool.submit(inspect_one_image, row, calculate_hash) for row in rows]
        for future in progress(as_completed(futures), total=len(futures), desc="Inspecting images"):
            results.append(future.result())
    return pd.DataFrame(results)


def parse_age(series: pd.Series) -> pd.Series:
    ages = pd.to_numeric(series.astype(str).str.extract(r"(\d+)", expand=False), errors="coerce")
    ages = ages.where((ages >= 0) & (ages <= 120))
    return ages


def label_combination(row: pd.Series) -> str:
    labels = [p for p in PATHOLOGIES if row[p] == 1]
    return "|".join(labels) if labels else "No Finding/zero pathology"


def calculate_label_outputs(train: pd.DataFrame, val: pd.DataFrame) -> dict[str, pd.DataFrame | dict]:
    n_train = len(train)
    y_train = train[PATHOLOGIES].to_numpy(dtype=int)
    y_val = val[PATHOLOGIES].to_numpy(dtype=int)
    positive = y_train.sum(axis=0)
    val_positive = y_val.sum(axis=0)
    negative = n_train - positive

    freq_df = pd.DataFrame({
        "pathology": PATHOLOGIES,
        "train_positive_count": positive,
        "validation_positive_count": val_positive,
        "train_negative_count": negative,
        "train_prevalence_percent": np.where(n_train > 0, positive / n_train * 100, np.nan),
        "validation_prevalence_percent": np.where(len(val) > 0, val_positive / len(val) * 100, np.nan),
    }).sort_values("train_positive_count", ascending=False)

    def class_group(count: int) -> str:
        if count > 5000:
            return "head"
        if count >= 1000:
            return "medium"
        return "tail"

    groups = [class_group(int(c)) for c in positive]
    group_df = pd.DataFrame({"pathology": PATHOLOGIES, "train_positive_count": positive, "group": groups})

    class_weights = pd.DataFrame({"pathology": PATHOLOGIES, "train_positive_count": positive, "train_negative_count": negative})
    max_freq = positive.max() if positive.size else 0
    for beta in (0.9, 0.99, 0.999, 0.9999):
        effective = np.array([(1 - beta ** c) / (1 - beta) if c > 0 else np.nan for c in positive], dtype=float)
        class_weights[f"effective_number_beta_{beta}"] = effective
        class_weights[f"class_balanced_weight_beta_{beta}"] = np.where(effective > 0, 1 / effective, np.nan)
    inverse = np.full_like(positive, np.nan, dtype=float)
    pos_weight = np.full_like(positive, np.nan, dtype=float)
    sqrt_inverse = np.full_like(positive, np.nan, dtype=float)
    nonzero_mask = positive > 0
    inverse[nonzero_mask] = n_train / positive[nonzero_mask]
    pos_weight[nonzero_mask] = negative[nonzero_mask] / positive[nonzero_mask]
    sqrt_inverse[nonzero_mask] = np.sqrt(max_freq / positive[nonzero_mask])
    class_weights["inverse_frequency_weight"] = inverse
    class_weights["bce_pos_weight_negative_over_positive"] = pos_weight
    class_weights["sqrt_frequency"] = np.sqrt(positive)
    class_weights["sqrt_inverse_sampling_weight"] = sqrt_inverse

    combinations = train.apply(label_combination, axis=1).value_counts().reset_index()
    combinations.columns = ["label_combination", "count"]
    combinations["percentage"] = combinations["count"] / n_train * 100

    cooc = y_train.T @ y_train
    cooc_df = pd.DataFrame(cooc, index=PATHOLOGIES, columns=PATHOLOGIES)
    cooc_pct_df = cooc_df / n_train * 100
    conditional = cooc_df.copy().astype(float)
    for pathology in PATHOLOGIES:
        denom = cooc_df.loc[pathology, pathology]
        conditional.loc[pathology] = cooc_df.loc[pathology] / denom if denom else np.nan
    jaccard = pd.DataFrame(index=PATHOLOGIES, columns=PATHOLOGIES, dtype=float)
    for a in PATHOLOGIES:
        for b in PATHOLOGIES:
            intersection = cooc_df.loc[a, b]
            union = cooc_df.loc[a, a] + cooc_df.loc[b, b] - intersection
            jaccard.loc[a, b] = intersection / union if union else np.nan

    total_positive_labels = int(positive.sum())
    group_positive = group_df.groupby("group")["train_positive_count"].sum().to_dict()
    group_percentages = {
        group: (count / total_positive_labels * 100 if total_positive_labels else 0.0)
        for group, count in group_positive.items()
    }

    zero_positive = int((train["num_positive_pathologies"] == 0).sum())
    label_cardinality = float(train["num_positive_pathologies"].mean())
    summary = {
        "label_cardinality": label_cardinality,
        "label_density": label_cardinality / len(PATHOLOGIES),
        "unique_label_combinations": int(len(combinations)),
        "no_finding_count": int(train["no_finding"].sum()),
        "no_finding_percent": float(train["no_finding"].mean() * 100),
        "zero_positive_pathology_images": zero_positive,
        "group_positive_label_percentages": group_percentages,
    }

    return {
        "class_frequencies": freq_df,
        "class_groups": group_df,
        "class_weights": class_weights,
        "label_combinations": combinations,
        "cooccurrence_counts": cooc_df,
        "cooccurrence_percentages": cooc_pct_df,
        "conditional_probabilities": conditional,
        "label_jaccard": jaccard,
        "summary": summary,
    }


def build_sampling_weights(train: pd.DataFrame, class_weights: pd.DataFrame) -> pd.DataFrame:
    weight_by_class = dict(zip(class_weights["pathology"], class_weights["sqrt_inverse_sampling_weight"]))
    no_finding_weight = 1.0
    rows = []
    for _, row in train.iterrows():
        positives = [p for p in PATHOLOGIES if row[p] == 1]
        if positives:
            rarest = max(positives, key=lambda p: weight_by_class.get(p, 0))
            weight = float(weight_by_class[rarest])
        else:
            rarest = "No Finding/zero pathology"
            weight = no_finding_weight
        rows.append({
            "image_filename": row["image_filename"],
            "image_path": row["image_path"],
            "sampling_weight": weight,
            "rarest_positive_pathology": rarest,
            "no_finding": int(row["no_finding"]),
        })
    return pd.DataFrame(rows)


def save_plots(train: pd.DataFrame, val: pd.DataFrame, image_stats: pd.DataFrame, outputs: dict, figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    freq = outputs["class_frequencies"].copy()
    group_df = outputs["class_groups"].copy()

    plt.figure(figsize=(11, 5))
    plt.bar(freq["pathology"], freq["train_positive_count"])
    plt.title("Training Class Frequencies")
    plt.ylabel("Positive training images")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(figures_dir / "class_frequency_bar.png", dpi=200)
    plt.close()

    plt.figure(figsize=(11, 5))
    plt.bar(freq["pathology"], freq["train_positive_count"])
    plt.yscale("log")
    plt.title("Training Class Frequencies (Log Scale)")
    plt.ylabel("Positive training images, log scale")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(figures_dir / "class_frequency_log.png", dpi=200)
    plt.close()

    plt.figure(figsize=(11, 5))
    plt.bar(freq["pathology"], freq["train_prevalence_percent"])
    plt.title("Training Pathology Prevalence")
    plt.ylabel("Training images (%)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(figures_dir / "pathology_prevalence.png", dpi=200)
    plt.close()

    colors = {"head": "#2f6fbb", "medium": "#4f9f5f", "tail": "#c96b3c"}
    ordered_groups = group_df.sort_values("train_positive_count", ascending=False)
    plt.figure(figsize=(11, 5))
    plt.bar(
        ordered_groups["pathology"],
        ordered_groups["train_positive_count"],
        color=[colors.get(g, "gray") for g in ordered_groups["group"]],
    )
    plt.title("Head, Medium, and Tail Classes")
    plt.ylabel("Positive training images")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(figures_dir / "head_medium_tail_groups.png", dpi=200)
    plt.close()

    cooc = outputs["cooccurrence_counts"]
    plt.figure(figsize=(9, 8))
    plt.imshow(cooc, cmap="viridis")
    plt.colorbar(label="Co-occurring training images")
    plt.title("Pathology Co-occurrence Counts")
    plt.xticks(range(len(PATHOLOGIES)), PATHOLOGIES, rotation=45, ha="right")
    plt.yticks(range(len(PATHOLOGIES)), PATHOLOGIES)
    plt.tight_layout()
    plt.savefig(figures_dir / "cooccurrence_heatmap.png", dpi=200)
    plt.close()

    counts = train["num_positive_pathologies"].clip(upper=4).value_counts().sort_index()
    labels = ["0", "1", "2", "3", ">3"]
    values = [int(counts.get(i, 0)) for i in [0, 1, 2, 3, 4]]
    plt.figure(figsize=(7, 4))
    plt.bar(labels, values)
    plt.title("Number of Positive Pathology Labels per Training Image")
    plt.xlabel("Positive pathology labels")
    plt.ylabel("Training images")
    plt.tight_layout()
    plt.savefig(figures_dir / "label_count_per_image_distribution.png", dpi=200)
    plt.close()

    images_per_patient = train.groupby("patient_id").size()
    plt.figure(figsize=(7, 4))
    plt.hist(images_per_patient, bins=50)
    plt.title("Training Images per Patient")
    plt.xlabel("Images per patient")
    plt.ylabel("Patients")
    plt.tight_layout()
    plt.savefig(figures_dir / "images_per_patient_distribution.png", dpi=200)
    plt.close()

    readable = image_stats[image_stats["readable"] == True]
    if not readable.empty:
        plt.figure(figsize=(6, 5))
        plt.hexbin(readable["width"], readable["height"], gridsize=40, cmap="magma")
        plt.colorbar(label="Image count")
        plt.title("Original Training Image Dimensions")
        plt.xlabel("Width")
        plt.ylabel("Height")
        plt.tight_layout()
        plt.savefig(figures_dir / "original_image_dimensions.png", dpi=200)
        plt.close()

    ages = parse_age(train.get("patient_age", pd.Series(dtype=str))).dropna()
    if not ages.empty:
        plt.figure(figsize=(7, 4))
        plt.hist(ages, bins=30)
        plt.title("Training Patient Age Distribution")
        plt.xlabel("Age")
        plt.ylabel("Training images")
        plt.tight_layout()
        plt.savefig(figures_dir / "age_distribution.png", dpi=200)
        plt.close()

    train_prev = train[PATHOLOGIES].mean() * 100
    val_prev = val[PATHOLOGIES].mean() * 100
    x = np.arange(len(PATHOLOGIES))
    plt.figure(figsize=(12, 5))
    plt.bar(x - 0.2, train_prev, width=0.4, label="Train")
    plt.bar(x + 0.2, val_prev, width=0.4, label="Validation")
    plt.title("Train vs Validation Pathology Prevalence")
    plt.ylabel("Images (%)")
    plt.xticks(x, PATHOLOGIES, rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "train_vs_validation_prevalence.png", dpi=200)
    plt.close()


def get_torch_transforms(training: bool = False):
    try:
        from torchvision import transforms
        from torchvision.transforms import InterpolationMode
    except Exception as exc:
        raise SystemExit(f"PyTorch/torchvision is required for transform verification: {exc}")

    common = [
        transforms.Resize((256, 256), interpolation=InterpolationMode.BILINEAR),
        transforms.CenterCrop((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    if not training:
        return transforms.Compose(common)
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        *common,
    ])


class NIHChestXrayDataset:
    """Minimal reusable PyTorch-style dataset for later training code."""

    def __init__(self, manifest_csv: str | Path, transform=None):
        self.manifest = pd.read_csv(manifest_csv)
        self.transform = transform or get_torch_transforms(training=False)

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int):
        row = self.manifest.iloc[index]
        with Image.open(row["image_path"]) as img:
            image = ImageOps.grayscale(img)
            tensor = self.transform(image)
        labels = row[PATHOLOGIES].to_numpy(dtype=np.float32)
        return tensor, labels, row["image_filename"]


def verify_preprocessing(train_manifest_csv: Path, train: pd.DataFrame, output_dir: Path, sample_count: int) -> dict:
    if sample_count <= 0:
        return {"checked_images": 0}
    dataset = NIHChestXrayDataset(train_manifest_csv, transform=get_torch_transforms(training=False))
    sample_count = min(sample_count, len(dataset))
    stats = []
    before_images = []
    after_images = []
    for idx in range(sample_count):
        tensor, labels, filename = dataset[idx]
        if list(tensor.shape) != [3, 224, 224]:
            raise SystemExit(f"Unexpected tensor shape for {filename}: {list(tensor.shape)}")
        if not bool(np.isfinite(tensor.numpy()).all()):
            raise SystemExit(f"NaN or infinite values found after preprocessing: {filename}")
        stats.append({
            "filename": filename,
            "tensor_min": float(tensor.min()),
            "tensor_max": float(tensor.max()),
            "tensor_mean": float(tensor.mean()),
            "tensor_std": float(tensor.std()),
        })

        if idx < 8:
            row = train.iloc[idx]
            with Image.open(row["image_path"]) as img:
                before = ImageOps.grayscale(img).resize((112, 112))
            arr = tensor.numpy().transpose(1, 2, 0)
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            arr = np.clip(arr * std + mean, 0, 1)
            after = Image.fromarray((arr * 255).astype(np.uint8)).resize((112, 112))
            before_images.append((before.convert("RGB"), filename))
            after_images.append((after.convert("RGB"), filename))

    stats_df = pd.DataFrame(stats)
    atomic_write_csv(stats_df, output_dir / "preprocessing_verification.csv")

    if before_images:
        sheet = Image.new("RGB", (224 * len(before_images), 150), "white")
        draw = ImageDraw.Draw(sheet)
        for i, ((before, filename), (after, _)) in enumerate(zip(before_images, after_images)):
            x = i * 224
            sheet.paste(before, (x, 0))
            sheet.paste(after, (x + 112, 0))
            draw.text((x + 2, 116), filename[:24], fill=(0, 0, 0))
        sheet.save(output_dir / "preprocessing_contact_sheet.png")

    return {
        "checked_images": sample_count,
        "tensor_min": float(stats_df["tensor_min"].min()),
        "tensor_max": float(stats_df["tensor_max"].max()),
        "tensor_mean": float(stats_df["tensor_mean"].mean()),
        "tensor_std": float(stats_df["tensor_std"].mean()),
    }


def save_outputs(
    train: pd.DataFrame,
    val: pd.DataFrame,
    image_stats: pd.DataFrame,
    issues_df: pd.DataFrame,
    outputs: dict,
    sampling_weights: pd.DataFrame,
    data_dir: Path,
) -> dict:
    atomic_write_csv(train, data_dir / "train_manifest.csv")
    atomic_write_csv(val, data_dir / "val_manifest.csv")
    atomic_write_csv(outputs["class_frequencies"], data_dir / "class_frequencies.csv")
    atomic_write_csv(outputs["class_groups"], data_dir / "class_groups.csv")
    atomic_write_csv(outputs["class_weights"], data_dir / "class_weights.csv")
    atomic_write_csv(outputs["label_combinations"], data_dir / "label_combination_counts.csv")
    atomic_write_csv(outputs["cooccurrence_counts"].rename_axis("pathology").reset_index(), data_dir / "cooccurrence_counts.csv")
    atomic_write_csv(outputs["cooccurrence_percentages"].rename_axis("pathology").reset_index(), data_dir / "cooccurrence_percentages.csv")
    atomic_write_csv(outputs["conditional_probabilities"].rename_axis("pathology").reset_index(), data_dir / "conditional_probabilities.csv")
    atomic_write_csv(outputs["label_jaccard"].rename_axis("pathology").reset_index(), data_dir / "label_jaccard.csv")
    atomic_write_csv(issues_df, data_dir / "data_quality_issues.csv")
    atomic_write_csv(sampling_weights, data_dir / "training_image_sampling_weights.csv")
    atomic_write_csv(image_stats, data_dir / "training_image_integrity.csv")

    readable = image_stats[image_stats["readable"] == True]
    duplicate_hash_count = 0
    if "sha256" in readable and readable["sha256"].notna().any():
        duplicate_hash_count = int(readable["sha256"].duplicated(keep=False).sum())

    ages = parse_age(train.get("patient_age", pd.Series(dtype=str)))
    label_counts = train["num_positive_pathologies"]
    positive_counts = outputs["class_frequencies"].set_index("pathology")["train_positive_count"]
    nonzero = positive_counts[positive_counts > 0]
    summary = {
        "train_images": int(len(train)),
        "validation_images": int(len(val)),
        "unique_training_patients": int(train["patient_id"].nunique()),
        "unique_validation_patients": int(val["patient_id"].nunique()),
        "images_per_training_patient_summary": train.groupby("patient_id").size().describe().to_dict(),
        "missing_images": int((image_stats["exists"] == False).sum()),
        "unreadable_or_corrupted_training_images": int((image_stats["readable"] == False).sum()),
        "duplicate_filenames_in_train_manifest": int(train["image_filename"].duplicated().sum()),
        "duplicate_full_paths_in_train_manifest": int(train["image_path"].duplicated().sum()),
        "exact_duplicate_image_hash_rows": duplicate_hash_count,
        "image_file_extensions": readable["extension"].value_counts().to_dict(),
        "original_width_summary": readable["width"].dropna().describe().to_dict(),
        "original_height_summary": readable["height"].dropna().describe().to_dict(),
        "image_mode_distribution": readable["mode"].value_counts().to_dict(),
        "file_size_bytes_summary": readable["file_size_bytes"].dropna().describe().to_dict(),
        "patient_age_summary": ages.dropna().describe().to_dict() if ages.notna().any() else {},
        "patient_gender_counts": train.get("patient_gender", pd.Series(dtype=str)).value_counts(dropna=False).to_dict(),
        "view_position_counts": train.get("view_position", pd.Series(dtype=str)).value_counts(dropna=False).to_dict(),
        "positive_training_counts": positive_counts.to_dict(),
        "positive_validation_counts": outputs["class_frequencies"].set_index("pathology")["validation_positive_count"].to_dict(),
        "negative_training_counts": outputs["class_frequencies"].set_index("pathology")["train_negative_count"].to_dict(),
        "prevalence_percent": outputs["class_frequencies"].set_index("pathology")["train_prevalence_percent"].to_dict(),
        "no_finding_count": int(train["no_finding"].sum()),
        "no_finding_percent": float(train["no_finding"].mean() * 100),
        "positive_pathology_labels_per_image_counts": {
            "zero": int((label_counts == 0).sum()),
            "one": int((label_counts == 1).sum()),
            "two": int((label_counts == 2).sum()),
            "three": int((label_counts == 3).sum()),
            "more_than_three": int((label_counts > 3).sum()),
        },
        **outputs["summary"],
        "max_class_frequency": int(nonzero.max()) if not nonzero.empty else 0,
        "minimum_nonzero_class_frequency": int(nonzero.min()) if not nonzero.empty else 0,
        "imbalance_ratio": float(nonzero.max() / nonzero.min()) if len(nonzero) else None,
        "head_classes": outputs["class_groups"].query("group == 'head'")["pathology"].tolist(),
        "medium_classes": outputs["class_groups"].query("group == 'medium'")["pathology"].tolist(),
        "tail_classes": outputs["class_groups"].query("group == 'tail'")["pathology"].tolist(),
        "no_finding_sampling_weight": 1.0,
        "no_finding_sampling_weight_note": "No Finding images have no disease class; weight 1.0 keeps them neutral while rare pathology images receive higher sqrt-inverse weights.",
    }
    atomic_write_json(summary, data_dir / "dataset_summary.json")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build training-only NIH manifests and EDA outputs.")
    parser.add_argument("--dataset-root", type=Path, default=Path("/datasets/pgohil"))
    parser.add_argument("--project-root", type=Path, default=Path("/home-mscluster/pgohil/cxr-research/1. Data analysis + inspection"))
    parser.add_argument("--metadata-csv", type=Path, default=None, help="Path to NIH metadata CSV, e.g. Data_Entry_2017.csv")
    parser.add_argument("--train-list", type=Path, default=None, help="Optional file containing training image filenames")
    parser.add_argument("--val-list", type=Path, default=None, help="Optional file containing validation image filenames")
    parser.add_argument("--train-val-list", type=Path, default=None, help="Optional combined train/validation filename list")
    parser.add_argument("--test-list", type=Path, default=None, help="Optional test filename list used only for exclusion")
    parser.add_argument("--val-fraction", type=float, default=0.125)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true", help="Process a small permitted subset for debugging")
    parser.add_argument("--max-images", type=int, default=None, help="Limit permitted images for debugging")
    parser.add_argument("--skip-hashes", action="store_true", help="Skip slow SHA256 duplicate-image hashing")
    parser.add_argument("--num-workers", type=int, default=2, help="Conservative image-inspection workers")
    parser.add_argument("--verify-samples", type=int, default=8)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s: %(message)s")
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.max_images is not None and args.max_images < 1:
        raise ValueError("--max-images must be positive")
    if args.dry_run:
        args.max_images = args.max_images or 500
        args.verify_samples = min(args.verify_samples, 4)
        args.skip_hashes = True
        logging.info("Dry run enabled: limiting to %d images and skipping hashes.", args.max_images)

    # Debug subsets must never replace the manifests consumed by training.
    results_dir = args.project_root / ("results_debug" if args.dry_run or args.max_images is not None else "results")
    data_dir = results_dir / "data"
    figures_dir = results_dir / "figures"
    metadata_csv = args.metadata_csv or auto_find_metadata_csv(args.dataset_root)
    if args.train_val_list and (args.train_list or args.val_list):
        raise SystemExit("Use --train-val-list alone or separate --train-list/--val-list arguments.")
    if args.train_val_list:
        train_list_path = val_list_path = None
        train_val_list_path = args.train_val_list
    else:
        train_list_path = args.train_list or auto_find_split_file(args.dataset_root, "train")
        val_list_path = args.val_list or auto_find_split_file(args.dataset_root, "val")
        # Explicit separate lists take precedence over any discovered combined list.
        train_val_list_path = (None if args.train_list or args.val_list or (train_list_path and val_list_path)
                               else auto_find_split_file(args.dataset_root, "trainval"))
    if val_list_path and not train_list_path and not train_val_list_path:
        raise SystemExit("A validation list requires a training list; pass --train-list as well.")
    test_list_path = args.test_list or auto_find_split_file(args.dataset_root, "test")

    test_filenames = load_filename_list(test_list_path)
    train_filenames = load_filename_list(train_list_path)
    val_filenames = load_filename_list(val_list_path)
    train_val_filenames = load_filename_list(train_val_list_path)

    if train_filenames and val_filenames:
        needed_filenames = train_filenames | val_filenames
        split_mode = "existing train and validation lists"
    elif train_val_filenames:
        needed_filenames = train_val_filenames
        split_mode = "patient-level split from existing train/validation combined list"
    elif train_filenames:
        needed_filenames = train_filenames
        split_mode = "patient-level split from training list"
    else:
        needed_filenames = None
        split_mode = "patient-level split from all non-test image files under dataset root"
        logging.warning("No training list found. Using all image files outside test-like directories, excluding test-list filenames.")

    if needed_filenames is not None:
        overlap = needed_filenames & test_filenames
        if overlap:
            raise SystemExit(f"Training/validation filename list overlaps with test exclusion list: {len(overlap)} files")

    image_paths_by_name = scan_image_paths(args.dataset_root)
    if test_filenames:
        image_paths_by_name = {k: v for k, v in image_paths_by_name.items() if k not in test_filenames}

    if needed_filenames is not None and args.max_images:
        needed_filenames = set(sorted(needed_filenames)[: args.max_images])

    metadata = read_metadata_rows(metadata_csv, needed_filenames, test_filenames)
    expected_filenames = needed_filenames if needed_filenames is not None else set(image_paths_by_name)
    metadata_filenames = set(metadata["image_filename"]) if not metadata.empty else set()
    missing_metadata = sorted(expected_filenames - metadata_filenames)
    if missing_metadata:
        raise SystemExit(f"Missing metadata for {len(missing_metadata)} required image(s), e.g. {missing_metadata[:5]}. "
                         "No manifests were written. Repair the metadata and rerun preprocessing.")
    manifest, issues_df = build_manifest(metadata, image_paths_by_name)

    duplicate_manifest_filenames = manifest["image_filename"][manifest["image_filename"].duplicated()].unique()
    if len(duplicate_manifest_filenames):
        raise SystemExit(
            "Duplicate filenames were produced in the manifest. Check metadata duplicates, for example: "
            + ", ".join(map(str, duplicate_manifest_filenames[:10]))
        )

    if args.max_images:
        manifest = manifest.sort_values("image_filename").head(args.max_images).copy()

    if train_filenames and val_filenames:
        if args.max_images:
            # Only an explicitly requested debug cap may omit listed images.
            selected = set(manifest["image_filename"])
            train_filenames &= selected
            val_filenames &= selected
        train, val = split_by_existing_lists(manifest, train_filenames, val_filenames)
        split_method = split_mode
    else:
        train, val, split_method = patient_level_split(manifest, args.val_fraction, args.seed)

    train = train.sort_values("image_filename").reset_index(drop=True)
    val = val.sort_values("image_filename").reset_index(drop=True)
    validate_manifests(train, val, test_filenames)

    train_manifest_csv = data_dir / "train_manifest.csv"

    zero_positive = train[PATHOLOGIES].sum()
    zero_classes = zero_positive[zero_positive == 0].index.tolist()
    if zero_classes:
        logging.warning("These pathologies have zero positive training examples: %s", ", ".join(zero_classes))

    for split, frame in (("training", train), ("validation", val)):
        logging.info("Checking %s image readability.", split)
        stats = inspect_images(frame, skip_hashes=args.skip_hashes or split == "validation",
                               num_workers=max(1, args.num_workers))
        unreadable = stats[stats["readable"] == False]
        if not unreadable.empty:
            examples = unreadable["image_path"].head(5).tolist()
            raise SystemExit(f"Found {len(unreadable)} unreadable or corrupted {split} image(s), e.g. {examples}. "
                             "No manifests were written. Repair the images and rerun preprocessing.")
        if split == "training":
            image_stats = stats

    outputs = calculate_label_outputs(train, val)
    sampling_weights = build_sampling_weights(train, outputs["class_weights"])
    summary = save_outputs(train, val, image_stats, issues_df, outputs, sampling_weights, data_dir)
    summary["split_method"] = split_method
    atomic_write_json(summary, data_dir / "dataset_summary.json")
    save_plots(train, val, image_stats, outputs, figures_dir)
    verification = verify_preprocessing(train_manifest_csv, train, data_dir, args.verify_samples)
    summary["preprocessing_verification"] = verification
    atomic_write_json(summary, data_dir / "dataset_summary.json")

    freq = outputs["class_frequencies"]
    most = freq.iloc[0]
    nonzero_freq = freq[freq["train_positive_count"] > 0].sort_values("train_positive_count")
    least = nonzero_freq.iloc[0] if not nonzero_freq.empty else None

    print("\nTraining-only NIH preprocessing/EDA summary")
    print("=" * 46)
    print(f"Split method: {split_method}")
    print(f"Train images: {len(train)}")
    print(f"Validation images: {len(val)}")
    print(f"Unique training patients: {train['patient_id'].nunique()}")
    print(f"Unique validation patients: {val['patient_id'].nunique()}")
    print(f"Most frequent pathology: {most['pathology']} ({int(most['train_positive_count'])})")
    if least is not None:
        print(f"Least frequent non-zero pathology: {least['pathology']} ({int(least['train_positive_count'])})")
    print(f"Imbalance ratio: {summary['imbalance_ratio']}")
    print(f"Head classes: {', '.join(summary['head_classes']) or 'none'}")
    print(f"Medium classes: {', '.join(summary['medium_classes']) or 'none'}")
    print(f"Tail classes: {', '.join(summary['tail_classes']) or 'none'}")
    print(f"No Finding training images: {summary['no_finding_count']} ({summary['no_finding_percent']:.2f}%)")
    print(f"Label cardinality: {summary['label_cardinality']:.4f}")
    print(f"Preprocessing checked images: {verification.get('checked_images', 0)}")
    print(f"Reports: {data_dir}")
    print(f"Figures: {figures_dir}")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("default")
        main()
