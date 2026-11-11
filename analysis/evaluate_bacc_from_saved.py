#!/usr/bin/env python3
"""Tune thresholds on validation bACC and score saved test predictions on CPU."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.hyperparameter_sweep import PROTOCOL, SEEDS, SCRIPTS, file_hash, read_json, write_json


DEFAULT_METHODS = ["bce_baseline", "focal_loss", "db_loss_natural", "sqrt_sampling", "crt", "balanced_mixup"]
ARRAYS = ("predictions", "labels", "image_ids", "patient_ids")
ROW_ARRAYS = ("labels", "image_ids", "patient_ids")
GROUPS = ("head", "medium", "tail")


def tune_thresholds(labels, probs):
    """Exact search; maximize TP*N - FP*P, breaking ties by highest threshold."""
    thresholds = []
    for y, scores in zip(labels.T, np.asarray(probs, dtype=np.float64).T):
        positives, negatives = int(y.sum()), int(len(y) - y.sum())
        if not positives or not negatives:
            raise ValueError("bACC tuning requires both positive and negative validation labels")
        order = np.argsort(-scores, kind="stable")
        sorted_scores = scores[order]
        # Evaluate entire tied-score groups; the last group predicts all positive.
        ends = np.r_[np.flatnonzero(np.diff(sorted_scores)), len(y) - 1]
        tp = np.cumsum(y[order], dtype=np.int64)[ends]
        fp = ends + 1 - tp
        # The first candidate predicts all negative, including when max(score)=1.
        candidates = np.r_[np.nextafter(sorted_scores[0], np.inf), sorted_scores[ends]]
        gain = np.r_[0, tp * negatives - fp * positives]
        thresholds.append(float(candidates[np.argmax(gain)]))
    return np.asarray(thresholds, dtype=np.float64)


def metrics_at(labels, probs, thresholds):
    # Float64 preserves an all-negative threshold just above a float32 score.
    predicted = np.asarray(probs, dtype=np.float64) >= np.asarray(thresholds, dtype=np.float64)
    positive = labels.astype(bool)
    tp = (predicted & positive).sum(axis=0)
    tn = (~predicted & ~positive).sum(axis=0)
    positives, negatives = positive.sum(axis=0), (~positive).sum(axis=0)
    if np.any(positives == 0) or np.any(negatives == 0):
        raise ValueError("bACC requires both positive and negative labels for every disease")
    sensitivity, specificity = tp / positives, tn / negatives
    return {"bacc": (sensitivity + specificity) / 2,
            "sensitivity": sensitivity, "specificity": specificity}


def load_split(run_dir, split, config, class_names, group_hash):
    directory = run_dir / f"eval_{split}"
    marker_path = directory / "evaluation_complete.json"
    marker = read_json(marker_path)
    identity, hashes = marker["identity"], marker["output_sha256"]
    expected = (PROTOCOL, config["method"], config["seed"], split)
    if tuple(identity.get(k) for k in ("training_protocol", "method", "seed", "split")) != expected:
        raise ValueError(f"Evaluation identity mismatch: {directory}")
    if identity.get("max_test_samples") is not None:
        raise ValueError(f"Subset evaluation cannot be used for final bACC: {directory}")
    inputs = identity.get("input_sha256", {})
    training_inputs = config.get("input_sha256", {})
    shared_inputs = inputs.keys() & training_inputs.keys()
    if not shared_inputs or any(inputs[p] != training_inputs[p] for p in shared_inputs):
        raise ValueError(f"Evaluation inputs differ from training: {directory}")
    if group_hash is not None and group_hash not in inputs.values():
        raise ValueError(f"Class groups do not match the recorded evaluation inputs: {directory}")
    checkpoint_hash = identity.get("checkpoint_sha256")
    if not isinstance(checkpoint_hash, str) or len(checkpoint_hash) != 64:
        raise ValueError(f"Missing checkpoint fingerprint: {marker_path}")

    # Only consume recorded arrays: no checkpoint, logits, or F1 outputs required.
    arrays, consumed_hashes = {}, {"evaluation_complete.json": file_hash(marker_path)}
    for name in ARRAYS:
        path = directory / f"{name}.npy"
        digest = file_hash(path)
        if hashes.get(path.name) != digest:
            raise ValueError(f"Missing or changed evaluation hash: {path}")
        arrays[name] = np.load(path, allow_pickle=False)
        consumed_hashes[path.name] = digest
    labels, probs = arrays["labels"], arrays["predictions"]
    if labels.ndim != 2 or labels.shape != probs.shape or labels.shape[1] != len(class_names) or not len(labels):
        raise ValueError(f"Expected matching nonempty (N, {len(class_names)}) arrays: {directory}")
    if labels.dtype.kind not in "biuf" or not np.isin(labels, [0, 1]).all():
        raise ValueError(f"Labels must be binary: {directory}")
    if probs.dtype.kind not in "fiu" or not np.isfinite(probs).all() or np.any((probs < 0) | (probs > 1)):
        raise ValueError(f"Predictions must be finite probabilities in [0, 1]: {directory}")
    invalid = (labels.sum(axis=0) == 0) | (labels.sum(axis=0) == len(labels))
    if invalid.any():
        diseases = ", ".join(np.asarray(class_names)[invalid])
        raise ValueError(f"bACC undefined: {diseases} lack positives or negatives in {directory}")
    for name in ("image_ids", "patient_ids"):
        values = arrays[name].astype(str)
        if values.ndim != 1 or len(values) != len(labels) or np.isin(np.char.lower(np.char.strip(values)), ["", "nan", "none"]).any():
            raise ValueError(f"Invalid {name} in {directory}")
        arrays[name] = values
    if len(np.unique(arrays["image_ids"])) != len(labels):
        raise ValueError(f"Duplicate image IDs in {directory}")
    return arrays, identity, consumed_hashes


def read_groups(path):
    if path is None:
        return {}, None
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    groups = {row["pathology"]: row["group"] for row in rows}
    if len(groups) != len(rows) or not groups or not set(groups.values()) <= set(GROUPS):
        raise ValueError(f"Duplicate diseases or invalid class groups: {path}")
    return groups, file_hash(path)


def write_csv(path, rows):
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_saved(runs_dir, methods, seeds, output_dir, class_groups=None):
    if not methods or len(set(methods)) != len(methods) or not set(methods) <= SCRIPTS.keys():
        raise ValueError("Specify distinct supported methods")
    if not seeds or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise ValueError("Specify distinct non-negative seeds")
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise ValueError(f"Output directory already exists; choose a new name: {output_dir}")
    groups, group_hash = read_groups(class_groups)
    reference_names, reference_rows = None, {}
    per_class_rows, seed_rows, run_records = [], [], []

    for method in methods:
        for seed in seeds:
            run_dir = (runs_dir / method / f"seed_{seed}").resolve()
            if output_dir == run_dir or run_dir in output_dir.parents:
                raise ValueError("Write bACC results outside the evaluated method/seed directories")
            config_path = run_dir / "config.json"
            config = read_json(config_path)
            if (config.get("training_protocol"), config.get("method"), config.get("seed")) != (PROTOCOL, method, seed):
                raise ValueError(f"Training protocol/method/seed mismatch: {config_path}")
            if config.get("smoke_test") or config.get("dry_run") or any(config.get(k) is not None for k in ("max_train_samples", "max_val_samples")):
                raise ValueError(f"Subset/debug training cannot be used for final bACC: {run_dir}")
            if method in ("db_loss", "db_loss_natural") and (config.get("db_bias_version") != 2 or config.get("method_hyperparameters", {}).get("db_bias_version") != 2):
                raise ValueError(f"DB bias version 2 is required for final analysis: {run_dir}")
            names = config.get("class_names", config.get("pathology_order"))
            if not isinstance(names, list) or len(names) != 14 or not all(isinstance(p, str) and p for p in names) or len(set(names)) != 14:
                raise ValueError(f"Expected 14 distinct disease names in column order: {config_path}")
            if config.get("pathology_order", names) != names or (reference_names is not None and names != reference_names):
                raise ValueError(f"Disease column order differs between runs: {config_path}")
            reference_names = names
            scopes = {"overall": list(range(len(names)))}
            if groups:
                if set(groups) != set(names):
                    raise ValueError("Class groups must contain exactly the 14 configured diseases")
                for group in GROUPS:
                    scopes[group] = [i for i, name in enumerate(names) if groups[name] == group]
                    if not scopes[group]:
                        raise ValueError(f"Class group has no diseases: {group}")

            data, identities, input_hashes = {}, {}, {}
            for split in ("val", "test"):
                data[split], identities[split], input_hashes[split] = load_split(run_dir, split, config, names, group_hash)
                rows = {key: data[split][key] for key in ROW_ARRAYS}
                if split in reference_rows and any(not np.array_equal(rows[key], reference_rows[split][key]) for key in ROW_ARRAYS):
                    raise ValueError(f"{split} rows/labels are not identically aligned across runs: {run_dir}")
                reference_rows.setdefault(split, rows)
            for key in ("checkpoint_sha256", "input_sha256"):
                if identities["val"][key] != identities["test"][key]:
                    raise ValueError(f"Validation/test {key} mismatch: {run_dir}")
            for key in ("image_ids", "patient_ids"):
                if np.intersect1d(data["val"][key], data["test"][key]).size:
                    raise ValueError(f"Validation/test overlap in {key}: {run_dir}")

            val, test = data["val"], data["test"]
            thresholds = tune_thresholds(val["labels"], val["predictions"])
            validation = metrics_at(val["labels"], val["predictions"], thresholds)
            fixed = metrics_at(test["labels"], test["predictions"], 0.5)
            tuned = metrics_at(test["labels"], test["predictions"], thresholds)
            for i, name in enumerate(names):
                row = {"method": method, "seed": seed, "pathology": name, "group": groups.get(name, ""),
                       "threshold_bacc_tuned": float(thresholds[i]), "validation_bacc_tuned": float(validation["bacc"][i]),
                       "test_positives": int(test["labels"][:, i].sum()), "test_negatives": int(len(test["labels"]) - test["labels"][:, i].sum())}
                for suffix, values in (("at_0_5", fixed), ("tuned", tuned)):
                    row.update({f"test_{metric}_{suffix}": float(values[metric][i]) for metric in values})
                per_class_rows.append(row)
            for group, indices in scopes.items():
                row = {"method": method, "seed": seed, "group": group, "n_classes": len(indices)}
                for suffix, values in (("at_0_5", fixed), ("tuned", tuned)):
                    row.update({f"test_{metric}_{suffix}": float(values[metric][indices].mean()) for metric in values})
                seed_rows.append(row)
            run_records.append({"method": method, "seed": seed, "run_dir": str(run_dir), "class_names": names,
                                "checkpoint_sha256": identities["val"]["checkpoint_sha256"],
                                "config_sha256": file_hash(config_path), "evaluation_files_sha256": input_hashes,
                                "thresholds": dict(zip(names, thresholds.tolist())),
                                "validation_bacc": dict(zip(names, validation["bacc"].tolist()))})

    summary = []
    for method in methods:
        for group in scopes:
            rows = [r for r in seed_rows if r["method"] == method and r["group"] == group]
            row = {"method": method, "group": group, "n_seeds": len(rows), "n_classes": rows[0]["n_classes"]}
            for suffix in ("at_0_5", "tuned"):
                values = [r[f"test_bacc_{suffix}"] for r in rows]
                row[f"test_bacc_{suffix}_mean"] = float(np.mean(values))
                row[f"test_bacc_{suffix}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
            summary.append(row)

    # Validate every run before creating outputs; never write into saved evaluations.
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json(output_dir / "bacc_thresholds.json", {
        "metric": "(sensitivity + specificity) / 2", "tuning_split": "val", "scoring_split": "test",
        "positive_rule": "probability >= threshold (float64 comparison)",
        "threshold_search": "Distinct validation probabilities plus the next float64 above the maximum; tied scores stay together",
        "tie_break": "Highest threshold among equal validation bACC maxima (exact integer comparison)",
        "aggregation": "Equal-weight disease means, then mean and sample standard deviation across seeds",
        "metric_units": "Fractions in [0, 1]; one-seed standard deviations are blank in CSV",
        "class_groups": {"path": str(class_groups.resolve()), "sha256": group_hash} if class_groups is not None else None,
        "runs": run_records,
    })
    write_csv(output_dir / "bacc_per_class.csv", per_class_rows)
    write_csv(output_dir / "bacc_by_seed.csv", seed_rows)
    write_csv(output_dir / "bacc_summary.csv", summary)
    return summary


def self_test():
    """Check exact tuning, float32 boundaries, saved-file checks and all four outputs."""
    import subprocess
    import tempfile

    def must_fail(action, message):
        try:
            action()
        except (ValueError, OSError) as exc:
            assert message in str(exc), str(exc)
        else:
            raise AssertionError(f"Expected failure containing: {message}")

    rng = np.random.default_rng(42)
    for size in (4, 17, 31):
        for _ in range(10):
            labels = rng.integers(0, 2, size=(size, 1))
            labels[:2, 0] = [0, 1]
            probs = rng.choice([0, 0.2, 0.5, 0.9, 1], size=(size, 1)).astype(np.float32)
            candidates = [np.nextafter(float(probs.max()), np.inf), *np.unique(probs).astype(float)]
            y = labels[:, 0].astype(bool)
            def objective(threshold):
                prediction = probs[:, 0].astype(float) >= threshold
                return (int((prediction & y).sum()) * int((~y).sum()) - int((prediction & ~y).sum()) * int(y.sum()), threshold)
            expected = max(candidates, key=objective)
            assert tune_thresholds(labels, probs)[0] == expected
    for score in (0.0, 0.3, 1.0):
        labels = np.array([[1], [0]])
        probs = np.full((2, 1), score, dtype=np.float32)
        thresholds = tune_thresholds(labels, probs)
        result = metrics_at(labels, probs, thresholds)
        assert thresholds[0] > float(probs.max())
        assert result["bacc"][0] == 0.5 and result["sensitivity"][0] == 0 and result["specificity"][0] == 1
    labels = np.array([1, *([0] * 6), 1, *([0] * 12)])[:, None]
    probs = np.linspace(1, 0, 20)[:, None]
    assert tune_thresholds(labels, probs)[0] == probs[7, 0]  # F1 instead prefers the top score.
    # Unequal disease prevalence must still give equal weight to disease bACCs.
    labels = np.array([[1, 1], [1, 0], [1, 0], [0, 0]])
    probs = np.array([[1, 1], [1, 1], [1, 1], [1, 0]])
    np.testing.assert_allclose(metrics_at(labels, probs, 0.5)["bacc"], [0.5, 2 / 3])
    must_fail(lambda: tune_thresholds(np.zeros((2, 1)), np.zeros((2, 1))), "both positive and negative")

    with tempfile.TemporaryDirectory(prefix="cxr-bacc-test-") as temporary:
        root = Path(temporary)
        names = [f"disease_{i}" for i in range(14)]
        groups_path = root / "class_groups.csv"
        write_csv(groups_path, [{"pathology": name, "group": GROUPS[i % 3]} for i, name in enumerate(names)])
        original_inputs = {"/original/data/val_manifest.csv": "a" * 64, "/original/data/class_groups.csv": file_hash(groups_path)}
        for method in ("bce_baseline", "focal_loss"):
            for seed in (42, 123):
                run = root / "runs" / method / f"seed_{seed}"
                run.mkdir(parents=True)
                order_key = "pathology_order" if method == "bce_baseline" else "class_names"
                config = {"training_protocol": PROTOCOL, "method": method, "seed": seed, order_key: names, "input_sha256": original_inputs}
                write_json(run / "config.json", config)
                for split in ("val", "test"):
                    directory = run / f"eval_{split}"
                    directory.mkdir()
                    scores = [0.4, 0.3, 0.2, 0.1] if split == "val" else [0.9, 0.25 if seed == 42 else 0.35, 0.2, 0.1]
                    np.save(directory / "labels.npy", np.tile([[1], [1], [0], [0]], (1, 14)))
                    np.save(directory / "predictions.npy", np.tile(np.array(scores, dtype=np.float32)[:, None], (1, 14)))
                    for key in ("image_ids", "patient_ids"):
                        np.save(directory / f"{key}.npy", np.array([f"{split}_{key}_{i}" for i in range(4)]))
                    identity = {"training_protocol": PROTOCOL, "method": method, "seed": seed, "split": split,
                                "checkpoint_sha256": "b" * 64, "input_sha256": original_inputs, "max_test_samples": None}
                    write_json(directory / "evaluation_complete.json", {"identity": identity,
                               "output_sha256": {f"{name}.npy": file_hash(directory / f"{name}.npy") for name in ARRAYS}})
        source_hashes = {p: file_hash(p) for p in (root / "runs").rglob("*") if p.is_file()}
        output = root / "results"
        summary = evaluate_saved(root / "runs", ["bce_baseline", "focal_loss"], [42, 123], output, groups_path)
        assert len(summary) == 8
        for row in summary:
            assert row["n_seeds"] == 2 and row["test_bacc_at_0_5_mean"] == 0.75
            assert row["test_bacc_tuned_mean"] == 0.875
            assert np.isclose(row["test_bacc_tuned_std"], np.std([0.75, 1.0], ddof=1))
        assert {p.name for p in output.iterdir()} == {"bacc_thresholds.json", "bacc_per_class.csv", "bacc_by_seed.csv", "bacc_summary.csv"}
        with (output / "bacc_per_class.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 56 and float(rows[0]["test_bacc_tuned"]) == 0.75  # Test-optimal would be 1.0.
        for record in read_json(output / "bacc_thresholds.json")["runs"]:
            assert all(t == float(np.float32(0.3)) for t in record["thresholds"].values())
        assert all(file_hash(path) == digest for path, digest in source_hashes.items())
        single = evaluate_saved(root / "runs", ["bce_baseline"], [42], root / "single")
        assert len(single) == 1 and single[0]["test_bacc_tuned_std"] is None
        must_fail(lambda: evaluate_saved(root / "runs", ["bce_baseline"], [42], output), "already exists")

        run = root / "runs/bce_baseline/seed_42"
        def invalid_run():
            return evaluate_saved(root / "runs", ["bce_baseline"], [42], root / "invalid")
        path = run / "eval_test/predictions.npy"
        original = path.read_bytes()
        np.save(path, np.zeros((4, 14)))
        must_fail(invalid_run, "changed evaluation hash")
        path.write_bytes(original)
        marker_path = run / "eval_test/evaluation_complete.json"
        original_marker = read_json(marker_path)
        changed = read_json(marker_path)
        changed["identity"]["checkpoint_sha256"] = "c" * 64
        write_json(marker_path, changed)
        must_fail(invalid_run, "checkpoint_sha256 mismatch")
        write_json(marker_path, original_marker)
        path = run / "eval_test/patient_ids.npy"
        original = path.read_bytes()
        path.write_bytes((run / "eval_val/patient_ids.npy").read_bytes())
        changed = read_json(marker_path)
        changed["output_sha256"]["patient_ids.npy"] = file_hash(path)
        write_json(marker_path, changed)
        must_fail(invalid_run, "overlap in patient_ids")
        path.write_bytes(original)
        write_json(marker_path, original_marker)
        path = root / "runs/focal_loss/seed_42/eval_test/labels.npy"
        original = path.read_bytes()
        marker_path = path.parent / "evaluation_complete.json"
        original_marker = read_json(marker_path)
        np.save(path, np.tile([[1], [0], [1], [0]], (1, 14)))
        changed = read_json(marker_path)
        changed["output_sha256"]["labels.npy"] = file_hash(path)
        write_json(marker_path, changed)
        must_fail(lambda: evaluate_saved(root / "runs", ["bce_baseline", "focal_loss"], [42], root / "invalid"), "not identically aligned")
        path.write_bytes(original)
        write_json(marker_path, original_marker)
        assert not (root / "invalid").exists()
        cli = subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()),
                              "--runs-dir", str(root / "runs"), "--output-dir", str(root / "cli"),
                              "--methods", "bce_baseline", "--seeds", "42"], capture_output=True, text=True, check=True)
        assert "75.00 (SD n/a)" in cli.stdout and (root / "cli/bacc_summary.csv").is_file()
    print("Self-test passed: exact bACC tuning, ties, endpoints, no test tuning, seed/group summaries, four output files, input integrity checks, and the CLI.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, help="Directory containing <method>/seed_<seed>/")
    parser.add_argument("--output-dir", type=Path, help="New directory for the four bACC output files")
    parser.add_argument("--methods", nargs="+", choices=sorted(SCRIPTS), default=DEFAULT_METHODS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--class-groups", type=Path, help="Optional original class_groups.csv for head/medium/tail scores")
    parser.add_argument("--self-test", action="store_true", help="Run synthetic checks in a temporary directory")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.runs_dir is None or args.output_dir is None:
        parser.error("--runs-dir and --output-dir are required unless using --self-test")
    try:
        summary = evaluate_saved(args.runs_dir, args.methods, args.seeds, args.output_dir, args.class_groups)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print("Test bACC (%): mean +/- sample SD across seeds; thresholds tuned on validation bACC")
    print(f"{'Method':<20} {'Group':<8} {'Seeds':>5} {'At 0.5':>19} {'bACC-tuned':>19}")
    for row in summary:
        scores = []
        for suffix in ("at_0_5", "tuned"):
            mean, std = row[f"test_bacc_{suffix}_mean"], row[f"test_bacc_{suffix}_std"]
            scores.append(f"{100 * mean:.2f} +/- {100 * std:.2f}" if std is not None else f"{100 * mean:.2f} (SD n/a)")
        print(f"{row['method']:<20} {row['group']:<8} {row['n_seeds']:>5} {scores[0]:>19} {scores[1]:>19}")
    print(f"Saved bACC results to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
