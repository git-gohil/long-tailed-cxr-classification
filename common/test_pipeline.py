"""CPU regression checks. All images, checkpoints and results use temporary directories.

Run from the project root: python -B common/test_pipeline.py
No NIH data, pretrained weight download, GPU or Slurm is needed.
"""
import argparse
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch
import warnings

_cache = tempfile.TemporaryDirectory(prefix="cxr-test-cache-")
os.environ["MPLCONFIGDIR"] = _cache.name
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn
try:
    from sklearn.metrics import average_precision_score, f1_score
except ImportError:
    average_precision_score = f1_score = None

from common import nih_longtail as nt
from common import hyperparameter_sweep as sweep
from common.metrics import average_precision, precision_recall_f1
from common.run_state import PROTOCOL
import evaluate as evaluator
from analysis import aggregate_results as aggregate
from analysis import aggregate_results_patient_bootstrap as bootstrap

torch.set_num_threads(1)


def module_from_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


baseline = module_from_path("review_bce", ROOT / sweep.SCRIPTS["bce_baseline"])


class TinyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(3, 8), nn.BatchNorm1d(8), nn.ReLU(), nn.Dropout(0.2))
        self.fc = nn.Linear(8, len(nt.PATHOLOGIES))

    def forward(self, images):
        return self.fc(self.backbone(images.mean(dim=(2, 3))))


class TinyTransform:
    def __init__(self, training):
        self.training = training

    def __call__(self, image):
        result = torch.from_numpy(np.array(image, dtype=np.float32).transpose(2, 0, 1)) / 255
        if self.training:
            # Exercise all three RNG families, in addition to loader/sampler RNGs.
            result = result + (random.random() + np.random.random() + torch.rand(())) * 0.01
        return result


class MetricTests(unittest.TestCase):
    def test_ties_permutations_and_bootstrap_duplicates(self):
        rng = np.random.default_rng(42)
        for _ in range(30):
            y = rng.integers(0, 2, 50)
            scores = rng.integers(0, 5, 50) / 4
            index = rng.integers(0, 50, 80)
            if average_precision_score is not None:
                self.assertAlmostEqual(average_precision(y[index], scores[index]), average_precision_score(y[index], scores[index]))
            permutation = rng.permutation(50)
            self.assertAlmostEqual(average_precision(y, scores), average_precision(y[permutation], scores[permutation]))
        self.assertEqual(average_precision(np.array([1, 0]), np.array([0.5, 0.5])), 0.5)
        self.assertAlmostEqual(average_precision(np.array([0, 1, 1]), np.array([0.9, 0.8, 0.8])), 2 / 3)

    def test_missed_classes_remain_in_macro_f1(self):
        y = np.tile(np.array([[1], [0]]), (1, 14))
        scores = np.tile(np.array([[0.1], [0.2]]), (1, 14))
        scores[:, 0] = [0.9, 0.1]
        groups = pd.DataFrame({"pathology": nt.PATHOLOGIES, "group": ["tail"] * 14})
        self.assertEqual(precision_recall_f1(0, 0, 4)[2], 0)
        self.assertEqual(precision_recall_f1(0, 4, 4)[2], 0)
        self.assertEqual(precision_recall_f1(0, 0, 0)[2], 0)
        measured = evaluator.metrics_from_probs(y, scores, groups)["macro_f1_at_0.5"]
        self.assertAlmostEqual(measured, 1 / 14)
        if f1_score is not None:
            self.assertAlmostEqual(measured, f1_score(y, scores >= 0.5, average="macro", zero_division=0))

    def test_all_metric_callers_share_implementation(self):
        for module in (nt, baseline, evaluator, aggregate, bootstrap):
            self.assertIs(module.average_precision, average_precision)

    def test_sampling_zero_exponent_with_realistic_counts(self):
        labels = np.zeros((80, 14))
        labels[:40] = 1
        labels[40:60, :3] = 1
        df = pd.DataFrame(labels, columns=nt.PATHOLOGIES)
        np.testing.assert_array_equal(nt.square_root_weights(df, 0), np.ones(80))
        for exponent in (0.25, 0.5, 0.75):
            counts = labels.sum(0)
            expected = np.array([counts[row > 0].min() ** (-exponent) if row.any()
                                 else np.median(counts) ** (-exponent) for row in labels])
            np.testing.assert_allclose(nt.square_root_weights(df, exponent), expected / expected.mean())


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="cxr-pipeline-")
        self.addCleanup(self.temp.cleanup)
        self.project = Path(self.temp.name)
        self.dataset = self.project / "dataset"
        self.dataset.mkdir()
        self.data = self.project / "1. Data analysis + inspection/results/data"
        self.data.mkdir(parents=True)
        metadata, splits = [], {}
        rng = np.random.default_rng(123)
        for split, first_patient, count in (("train", 1, 12), ("val", 20, 6), ("test", 40, 6)):
            rows = []
            for index in range(count):
                patient = first_patient + index // 2
                name = f"{patient:08d}_{index % 2:03d}.png"
                image = self.dataset / name
                Image.fromarray(rng.integers(0, 256, (8, 8, 3), dtype=np.uint8)).save(image)
                labels = {p: int((index + j) % 3 == 0 and index != count - 1) for j, p in enumerate(nt.PATHOLOGIES)}
                text = "|".join(p for p in nt.PATHOLOGIES if labels[p]) or "No Finding"
                rows.append({"image_filename": name, "image_path": str(image), "patient_id": patient, **labels})
                metadata.append({"Image Index": name, "Patient ID": patient, "Finding Labels": text})
            splits[split] = pd.DataFrame(rows)
        self.train, self.val = splits["train"], splits["val"]
        self.train.to_csv(self.data / "train_manifest.csv", index=False)
        self.val.to_csv(self.data / "val_manifest.csv", index=False)
        pd.DataFrame(metadata).to_csv(self.dataset / "Data_Entry_2017.csv", index=False)
        (self.dataset / "test_list.txt").write_text("\n".join(splits["test"].image_filename) + "\n")
        pd.DataFrame({"pathology": nt.PATHOLOGIES, "train_positive_count": self.train[nt.PATHOLOGIES].sum(),
                      "group": ["tail"] * 14}).to_csv(self.data / "class_groups.csv", index=False)
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        for module in (nt, baseline, evaluator):
            self.stack.enter_context(patch.object(module, "make_model", side_effect=lambda pretrained=True: TinyNetwork()))
        for module in (nt, baseline):
            self.stack.enter_context(patch.object(module, "make_transform", side_effect=TinyTransform))
            self.stack.enter_context(patch.object(module, "plot_training_history"))
        self.stack.enter_context(patch.object(baseline, "make_plots"))
        self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        self.stack.enter_context(warnings.catch_warnings())
        warnings.simplefilter("ignore")

    def args(self, method, output, epochs=2, resume=False, seed=42):
        module = baseline if method == "bce_baseline" else module_from_path("review_" + method, ROOT / sweep.SCRIPTS[method])
        flags = ["train", "--project-root", str(self.project), "--dataset-root", str(self.dataset),
                 "--metadata-csv", str(self.dataset / "Data_Entry_2017.csv"),
                 "--test-list", str(self.dataset / "test_list.txt"), "--output-dir", str(output),
                 "--epochs", str(epochs), "--seed", str(seed), "--batch-size", "4", "--num-workers", "0", "--device", "cpu", "--no-amp"]
        if resume:
            flags.append("--resume")
        if method == "focal_loss":
            flags.append("--disable-alpha")
        if method == "crt":
            flags += ["--source-checkpoint", str(self.project / f"full/bce_baseline/seed_{seed}/checkpoint_best.pt")]
        with patch.object(sys, "argv", flags):
            args = module.parse_args()
        if method == "bce_baseline":
            baseline.default_paths(args)
        else:
            nt.fill_default_paths(args)
        return args

    def train_method(self, method, output, epochs=2, resume=False, seed=42):
        args = self.args(method, output, epochs, resume, seed)
        (baseline.train if method == "bce_baseline" else nt.run_training)(args)
        return torch.load(output / "checkpoint_final.pt", map_location="cpu", weights_only=False)

    def evaluate_method(self, method, output, split, skip=False, seed=42):
        flags = ["evaluate", "--project-root", str(self.project), "--dataset-root", str(self.dataset),
                 "--metadata-csv", str(self.dataset / "Data_Entry_2017.csv"), "--test-list", str(self.dataset / "test_list.txt"),
                 "--method", method, "--seed", str(seed), "--run-dir", str(output), "--split", split,
                 "--device", "cpu", "--num-workers", "0", "--batch-size", "4"]
        if skip:
            flags.append("--skip-existing")
        with patch.object(sys, "argv", flags):
            evaluator.main()

    def test_all_six_train_resume_evaluate_and_aggregate(self):
        experiment = self.project / "experiment"
        runs = self.project / "full"
        experiment.mkdir()
        (experiment / "final").symlink_to(runs, target_is_directory=True)
        sweep.write_json(experiment / "manifest.json", {
            "training_protocol": PROTOCOL, "seeds": [42], "grids": {method: [{}] for method in sweep.SCRIPTS}})
        for method in sweep.SCRIPTS:
            with self.subTest(method=method):
                full = runs / method / "seed_42"
                interrupted = self.project / "resumed" / method / "seed_42"
                continuous = self.train_method(method, full)
                self.train_method(method, interrupted, epochs=1)
                resumed = self.train_method(method, interrupted, epochs=2, resume=True)
                for key, tensor in continuous["model_state_dict"].items():
                    self.assertTrue(torch.equal(tensor, resumed["model_state_dict"][key]), (method, key))
                for key in ("best_epoch", "best_metrics", "stale_epochs"):
                    # JSON normalises NaN group metrics for the comparison.
                    self.assertEqual(nt.to_jsonable(continuous["resume_state"][key]), nt.to_jsonable(resumed["resume_state"][key]))
                for a, b in zip(continuous["resume_state"]["history_rows"], resumed["resume_state"]["history_rows"]):
                    for key in ("epoch", "train_loss", "val_loss", "val_mAP", "lr", "best_epoch"):
                        self.assertEqual(a[key], b[key], (method, key))
                self.evaluate_method(method, full, "val")
                self.evaluate_method(method, full, "test")
                self.evaluate_method(method, full, "test", skip=True)
                sweep.validate_final_run(full, method, 42)
                selection = experiment / "selection" / method
                selection.mkdir(parents=True)
                config = sweep.read_json(full / "config.json")
                settings = {key: config[key] for key in sweep.FIXED}
                for key in sweep.GRIDS[method][0]:
                    settings[key] = True if key == "disable_alpha" else config[key]
                sweep.write_json(selection / "best_config.json", {"settings": settings})
                sweep.write_json(full / "completed.json", sweep.verified_completion(
                    full, {"method": method, "seed": 42, "settings": settings}))
                pred = np.load(full / "eval_test/predictions.npy")
                labels = np.load(full / "eval_test/labels.npy")
                metrics = json.loads((full / "eval_test/metrics.json").read_text())
                if average_precision_score is not None:
                    self.assertAlmostEqual(metrics["mAP"], average_precision_score(labels, pred, average="macro"))
                    self.assertAlmostEqual(metrics["macro_f1_at_0.5"], f1_score(labels, pred >= .5, average="macro", zero_division=0))
        sweep.check_experiment(experiment)
        with patch.object(sys, "argv", ["aggregate", "--project-root", str(self.project), "--runs-dir", str(runs),
                                         "--output-dir", str(self.project / "summary"), "--seeds", "42", "--bootstrap-mode", "none"]):
            aggregate.main()
        with patch.object(sys, "argv", ["bootstrap", "--project-root", str(self.project), "--runs-dir", str(runs),
                                         "--output-dir", str(self.project / "bootstrap"), "--seeds", "42", "--bootstrap-resamples", "8"]):
            bootstrap.main()
        self.assertEqual(len(pd.read_csv(self.project / "summary/results_summary.csv")), 6)
        self.assertEqual(len(pd.read_csv(self.project / "bootstrap/patient_bootstrap_vs_bce.csv")), 5)
        marker = runs / "bce_baseline/seed_42/eval_test/metrics.json"
        marker.write_text("{}")
        with self.assertRaisesRegex(ValueError, "changed"):
            sweep.validate_final_run(runs / "bce_baseline/seed_42", "bce_baseline", 42)

    def test_completion_verifies_real_checkpoint_settings_and_score(self):
        output = self.project / "completion_audit"
        self.train_method("bce_baseline", output, epochs=1)
        expected = {"method": "bce_baseline", "seed": 42, "settings": {"lr": 1e-4}}
        record = sweep.verified_completion(output, expected)
        checkpoint_path = output / "checkpoint_best.pt"
        original = checkpoint_path.read_bytes()
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        checkpoint["training_args"]["lr"] = 3e-4
        torch.save(checkpoint, checkpoint_path)
        with self.assertRaisesRegex(ValueError, "configuration differs"):
            sweep.verified_completion(output, expected)
        checkpoint_path.write_bytes(original)
        metrics_path = output / "validation_metrics.json"
        metrics = sweep.read_json(metrics_path)
        metrics["mAP"] = 0.99 if metrics["mAP"] != 0.99 else 0.01
        sweep.write_json(metrics_path, metrics)
        with self.assertRaisesRegex(ValueError, "score/epoch differs"):
            sweep.verified_completion(output, expected)
        sweep.write_json(output / "completed.json", record)
        with self.assertRaisesRegex(ValueError, "files changed"):
            sweep.completed_score(output, expected)

    def test_evaluation_rejects_wrong_seed_and_test_before_validation(self):
        output = self.project / "evaluation"
        checkpoint = self.train_method("bce_baseline", output, epochs=1)
        with self.assertRaises((ValueError, FileNotFoundError)):
            self.evaluate_method("bce_baseline", output, "test")
        self.assertFalse((output / "eval_test").exists())
        checkpoint["random_seed"] = 123
        torch.save(checkpoint, output / "checkpoint_best.pt")
        with self.assertRaisesRegex(ValueError, "seed"):
            self.evaluate_method("bce_baseline", output, "val")
        self.assertFalse((output / "eval_val").exists())

    def test_final_verification_rejects_replaced_checkpoint(self):
        output = self.project / "checkpoint_audit"
        self.train_method("bce_baseline", output, epochs=1)
        self.evaluate_method("bce_baseline", output, "val")
        self.evaluate_method("bce_baseline", output, "test")
        sweep.validate_final_run(output, "bce_baseline", 42)
        checkpoint = output / "checkpoint_best.pt"
        original = checkpoint.read_bytes()
        for replacement in (b"not a checkpoint", original[:-1] + bytes([original[-1] ^ 1])):
            checkpoint.write_bytes(replacement)
            with self.assertRaisesRegex(ValueError, "checkpoint changed"):
                sweep.validate_final_run(output, "bce_baseline", 42)

    def test_evaluation_rejects_changed_groups_for_bce_and_focal(self):
        groups = self.data / "class_groups.csv"
        original = groups.read_text()
        for method in ("bce_baseline", "focal_loss"):
            output = self.project / method
            self.train_method(method, output, epochs=1)
            groups.write_text(original.replace("tail", "head"))
            with self.assertRaisesRegex(ValueError, "class groups"):
                self.evaluate_method(method, output, "val")
            self.assertFalse((output / "eval_val").exists())
            if method != "bce_baseline":
                groups.write_text(original + "\n")
                with self.assertRaisesRegex(ValueError, "class groups input"):
                    self.evaluate_method(method, output, "val")
            groups.write_text(original)
            self.evaluate_method(method, output, "val")
            groups.write_text(original.replace("tail", "head"))
            with self.assertRaisesRegex(ValueError, "class groups"):
                self.evaluate_method(method, output, "test")
            self.assertFalse((output / "eval_test").exists())
            groups.write_text(original)
            self.evaluate_method(method, output, "test")

    def test_aggregation_rejects_changed_groups(self):
        runs = self.project / "runs"
        output = runs / "bce_baseline/seed_42"
        self.train_method("bce_baseline", output, epochs=1)
        self.evaluate_method("bce_baseline", output, "val")
        self.evaluate_method("bce_baseline", output, "test")
        groups = self.data / "class_groups.csv"
        groups.write_text(groups.read_text().replace("tail", "head"))
        for module in (aggregate, bootstrap):
            destination = self.project / module.__name__
            flags = ["analysis", "--project-root", str(self.project), "--runs-dir", str(runs),
                     "--output-dir", str(destination), "--methods", "bce_baseline", "--seeds", "42"]
            with patch.object(sys, "argv", flags), self.assertRaisesRegex(ValueError, "class groups"):
                module.main()
            self.assertFalse(destination.exists())

    def test_validation_and_test_require_identical_group_inputs(self):
        output = self.project / "group_inputs"
        self.train_method("bce_baseline", output, epochs=1)
        self.evaluate_method("bce_baseline", output, "val")
        groups = self.data / "class_groups.csv"
        original = groups.read_text()
        # Same mapping, different file: BCE has no CSV hash in its training checkpoint.
        groups.write_text(original + "\n")
        with self.assertRaisesRegex(ValueError, "Validation/test inputs differ"):
            self.evaluate_method("bce_baseline", output, "test")
        groups.write_text(original)
        self.evaluate_method("bce_baseline", output, "test")
        marker = output / "eval_test/evaluation_complete.json"
        saved = json.loads(marker.read_text())
        saved["identity"]["input_sha256"][str(groups.resolve())] = "changed"
        marker.write_text(json.dumps(saved))
        with self.assertRaisesRegex(ValueError, "Validation/test input mismatch"):
            sweep.validate_final_run(output, "bce_baseline", 42)

    def test_aggregation_rejects_different_test_populations(self):
        test_list = self.dataset / "test_list.txt"
        original = test_list.read_text()
        cases = {
            "methods": [("bce_baseline", 42), ("focal_loss", 42)],
            "seeds": [("bce_baseline", 42), ("bce_baseline", 123)],
        }
        for name, identities in cases.items():
            runs = self.project / name
            experiment = self.project / f"{name}_experiment"
            experiment.mkdir()
            (experiment / "final").symlink_to(runs, target_is_directory=True)
            sweep.write_json(experiment / "manifest.json", {
                "training_protocol": PROTOCOL, "seeds": list(dict.fromkeys(s for _, s in identities)),
                "grids": {method: [{}] for method, _ in identities}})
            for index, (method, seed) in enumerate(identities):
                test_list.write_text(original if index == 0 else "\n".join(original.splitlines()[:4]) + "\n")
                output = runs / method / f"seed_{seed}"
                self.train_method(method, output, epochs=1, seed=seed)
                for split in ("val", "test"):
                    self.evaluate_method(method, output, split, seed=seed)
                selection = experiment / "selection" / method
                selection.mkdir(parents=True, exist_ok=True)
                sweep.write_json(selection / "best_config.json", {"settings": {}})
                sweep.write_json(output / "completed.json", sweep.verified_completion(
                    output, {"method": method, "seed": seed, "settings": {}}))
            with self.assertRaisesRegex(ValueError, "Test rows"):
                sweep.check_experiment(experiment)
            for module in (aggregate, bootstrap):
                destination = self.project / f"{name}_{module.__name__}"
                flags = ["analysis", "--project-root", str(self.project), "--runs-dir", str(runs),
                         "--output-dir", str(destination), "--methods", *dict.fromkeys(m for m, _ in identities),
                         "--seeds", *map(str, dict.fromkeys(s for _, s in identities))]
                with self.subTest(case=name, module=module.__name__), patch.object(sys, "argv", flags):
                    with self.assertRaisesRegex((ValueError, SystemExit), "Test rows"):
                        module.main()
                    self.assertFalse(destination.exists())
        test_list.write_text(original)

    def test_aggregation_rejects_old_db_bias_version(self):
        runs = self.project / "runs"
        output = runs / "db_loss/seed_42"
        self.train_method("db_loss", output, epochs=1)
        for split in ("val", "test"):
            self.evaluate_method("db_loss", output, split)
        original = (output / "config.json").read_text()
        for version in (None, 1):
            config = json.loads(original)
            for values in (config, config["method_hyperparameters"]):
                if version is None:
                    values.pop("db_bias_version")
                else:
                    values["db_bias_version"] = version
            sweep.write_json(output / "config.json", config)
            for module in (aggregate, bootstrap):
                destination = self.project / f"old_db_{version}_{module.__name__}"
                flags = ["analysis", "--project-root", str(self.project), "--runs-dir", str(runs),
                         "--output-dir", str(destination), "--methods", "db_loss", "--seeds", "42"]
                with self.subTest(version=version, module=module.__name__), patch.object(sys, "argv", flags):
                    with self.assertRaisesRegex(ValueError, "DB bias"):
                        module.main()
                    self.assertFalse(destination.exists())
        (output / "config.json").write_text(original)
        sweep.validate_final_run(output, "db_loss", 42)

    def test_final_analysis_rejects_subset_and_debug_training(self):
        runs = self.project / "subset_runs"
        for key, value in (("max_train_samples", 4), ("max_val_samples", 3)):
            output = runs / key / "focal_loss/seed_42"
            args = self.args("focal_loss", output, epochs=1)
            setattr(args, key, value)
            nt.run_training(args)
            # Full evaluation must not turn subset training into a full-data run.
            for split in ("val", "test"):
                self.evaluate_method("focal_loss", output, split)
            experiment = self.project / f"subset_{key}"
            experiment.mkdir()
            (experiment / "final").symlink_to(runs / key, target_is_directory=True)
            selection = experiment / "selection/focal_loss"
            selection.mkdir(parents=True)
            sweep.write_json(selection / "best_config.json", {"settings": {}})
            sweep.write_json(experiment / "manifest.json", {
                "training_protocol": PROTOCOL, "seeds": [42], "grids": {"focal_loss": [{}]}})
            sweep.write_json(output / "completed.json", sweep.verified_completion(
                output, {"method": "focal_loss", "seed": 42, "settings": {}}))
            with self.assertRaisesRegex(ValueError, "Subset/debug training"):
                sweep.check_experiment(experiment)
            for module in (aggregate, bootstrap):
                destination = self.project / f"subset_{key}_{module.__name__}"
                flags = ["analysis", "--project-root", str(self.project), "--runs-dir", str(runs / key),
                         "--output-dir", str(destination), "--methods", "focal_loss", "--seeds", "42"]
                with self.subTest(key=key, module=module.__name__), patch.object(sys, "argv", flags):
                    with self.assertRaisesRegex(ValueError, "Subset/debug training"):
                        module.main()
                    self.assertFalse(destination.exists())

        # Debug flags must also be rejected when a tiny fixture fits inside the cap.
        output = self.project / "debug_flags"
        self.train_method("focal_loss", output, epochs=1)
        for split in ("val", "test"):
            self.evaluate_method("focal_loss", output, split)
        original = sweep.read_json(output / "config.json")
        for key in ("smoke_test", "dry_run"):
            sweep.write_json(output / "config.json", {**original, key: True})
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "Subset/debug training"):
                sweep.validate_final_run(output, "focal_loss", 42)
        sweep.write_json(output / "config.json", original)
        sweep.validate_final_run(output, "focal_loss", 42)

    def test_missing_and_ambiguous_inputs_preserve_preprocessing_outputs(self):
        preprocessing = module_from_path("review_preprocess", ROOT / "1. Data analysis + inspection/scripts/preprocess_and_analyse.py")
        train_list, val_list = self.project / "train.txt", self.project / "val.txt"
        train_list.write_text("\n".join(self.train.image_filename) + "\n")
        val_list.write_text("\n".join(self.val.image_filename) + "\n")
        metadata_path = self.dataset / "Data_Entry_2017.csv"
        metadata_bytes = metadata_path.read_bytes()
        for split, frame in (("training", self.train), ("validation", self.val)):
            image_path = Path(frame.iloc[-1].image_path)
            image_bytes = image_path.read_bytes()
            for issue in ("missing_image", "duplicate_image", "missing_metadata", "empty_list"):
                duplicate = self.dataset / "duplicates" / image_path.name
                selected_list = train_list if split == "training" else val_list
                list_bytes = selected_list.read_bytes()
                try:
                    if issue == "missing_image":
                        image_path.unlink()
                    elif issue == "duplicate_image":
                        duplicate.parent.mkdir(exist_ok=True)
                        duplicate.write_bytes(image_bytes)
                    elif issue == "missing_metadata":
                        metadata = pd.read_csv(metadata_path)
                        metadata[metadata["Image Index"] != image_path.name].to_csv(metadata_path, index=False)
                    else:
                        selected_list.write_text("")
                    for folder in (self.data.parents[1], self.project / "new_preprocessing"):
                        before = {p.relative_to(folder): p.read_bytes() for p in folder.rglob("*") if p.is_file()}
                        flags = ["preprocess", "--project-root", str(folder), "--dataset-root", str(self.dataset),
                                 "--metadata-csv", str(metadata_path), "--train-list", str(train_list),
                                 "--val-list", str(val_list), "--test-list", str(self.dataset / "test_list.txt")]
                        with self.subTest(split=split, issue=issue, folder=folder), patch.object(sys, "argv", flags):
                            with self.assertRaisesRegex(SystemExit, "missing|Missing metadata|No image filenames"):
                                preprocessing.main()
                            self.assertEqual(before, {p.relative_to(folder): p.read_bytes() for p in folder.rglob("*") if p.is_file()})
                finally:
                    image_path.write_bytes(image_bytes)
                    duplicate.unlink(missing_ok=True)
                    metadata_path.write_bytes(metadata_bytes)
                    selected_list.write_bytes(list_bytes)

    def test_split_discovery_and_explicit_list_precedence(self):
        preprocessing = module_from_path("review_preprocess", ROOT / "1. Data analysis + inspection/scripts/preprocess_and_analyse.py")
        train_list, val_list = self.dataset / "train_list.txt", self.dataset / "val_list.txt"
        combined = self.dataset / "train_val_list.txt"
        train_list.write_text("\n".join(self.train.image_filename) + "\n")
        val_list.write_text("\n".join(self.val.image_filename) + "\n")
        combined_text = train_list.read_text() + val_list.read_text()
        combined.write_text(combined_text)
        self.assertEqual(preprocessing.auto_find_split_file(self.dataset, "val"), val_list)
        self.assertEqual(preprocessing.auto_find_split_file(self.dataset, "trainval"), combined)
        for mode in ("auto", "combined", "separate", "debug"):
            folder = self.project / f"preprocessing_{mode}"
            flags = ["preprocess", "--project-root", str(folder), "--dataset-root", str(self.dataset),
                     "--metadata-csv", str(self.dataset / "Data_Entry_2017.csv"),
                     "--test-list", str(self.dataset / "test_list.txt"), "--verify-samples", "0",
                     "--num-workers", "1", "--skip-hashes"]
            train_list.write_text("\n".join(self.train.image_filename) + "\n")
            combined.write_text(combined_text)
            if mode == "combined":
                # Discovered separate lists would overlap; they must be ignored.
                train_list.write_text(combined_text)
                flags += ["--train-val-list", str(combined)]
            elif mode in ("separate", "debug"):
                combined.write_text("")  # Unselected combined lists must not be read.
                flags += ["--train-list", str(train_list), "--val-list", str(val_list)]
                if mode == "debug":
                    flags += ["--max-images", "16"]
            with self.subTest(mode=mode), patch.object(sys, "argv", flags), patch.object(preprocessing, "save_plots"):
                preprocessing.main()
            data = folder / ("results_debug" if mode == "debug" else "results") / "data"
            train = pd.read_csv(data / "train_manifest.csv")
            val = pd.read_csv(data / "val_manifest.csv")
            self.assertFalse(set(train.patient_id) & set(val.patient_id))
            self.assertEqual(len(train) + len(val), 16 if mode == "debug" else 18)
            if mode in ("auto", "separate"):
                self.assertEqual(set(train.image_filename), set(self.train.image_filename))
                self.assertEqual(set(val.image_filename), set(self.val.image_filename))
        with patch.object(sys, "argv", [*flags, "--train-val-list", str(combined)]):
            with self.assertRaisesRegex(SystemExit, "Use --train-val-list alone"):
                preprocessing.main()

    def test_corrupt_images_abort_before_manifests_are_written(self):
        preprocessing = module_from_path("review_preprocess", ROOT / "1. Data analysis + inspection/scripts/preprocess_and_analyse.py")
        train_list, val_list = self.project / "train.txt", self.project / "val.txt"
        train_list.write_text("\n".join(self.train.image_filename) + "\n")
        val_list.write_text("\n".join(self.val.image_filename) + "\n")
        for split, frame in (("training", self.train), ("validation", self.val)):
            # Includes a training image beyond the default eight-image transform check.
            image_path = Path(frame.iloc[-1].image_path)
            original = image_path.read_bytes()
            image_path.write_bytes(b"corrupted image")
            try:
                for folder in (self.data.parents[1], self.project / "new_preprocessing"):
                    before = {p.relative_to(folder): p.read_bytes() for p in folder.rglob("*") if p.is_file()}
                    flags = ["preprocess", "--project-root", str(folder), "--dataset-root", str(self.dataset),
                             "--metadata-csv", str(self.dataset / "Data_Entry_2017.csv"),
                             "--train-list", str(train_list), "--val-list", str(val_list),
                             "--test-list", str(self.dataset / "test_list.txt"), "--num-workers", "1"]
                    with self.subTest(split=split, folder=folder), patch.object(sys, "argv", flags):
                        with self.assertRaisesRegex(SystemExit, f"unreadable or corrupted {split}"):
                            preprocessing.main()
                        self.assertEqual(before, {p.relative_to(folder): p.read_bytes() for p in folder.rglob("*") if p.is_file()})
            finally:
                image_path.write_bytes(original)

    def test_debug_preprocessing_preserves_production_outputs(self):
        preprocessing = module_from_path("review_preprocess", ROOT / "1. Data analysis + inspection/scripts/preprocess_and_analyse.py")
        folder = self.data.parents[1]
        figures = folder / "results/figures"
        figures.mkdir()
        (figures / "existing.png").write_bytes(b"existing figure")
        before = {p: p.read_bytes() for p in (folder / "results").rglob("*") if p.is_file()}
        for debug_flags in (["--dry-run"], ["--dry-run", "--max-images", "8"], ["--max-images", "8"]):
            flags = ["preprocess", "--project-root", str(folder), "--dataset-root", str(self.dataset),
                     "--metadata-csv", str(self.dataset / "Data_Entry_2017.csv"),
                     "--test-list", str(self.dataset / "test_list.txt"), "--verify-samples", "0",
                     "--num-workers", "1", "--skip-hashes", *debug_flags]
            with patch.object(sys, "argv", flags), patch.object(preprocessing, "save_plots"):
                preprocessing.main()
            after = {p: p.read_bytes() for p in (folder / "results").rglob("*") if p.is_file()}
            self.assertEqual(before, after)
            debug = folder / "results_debug/data"
            count = sum(len(pd.read_csv(debug / name)) for name in ("train_manifest.csv", "val_manifest.csv"))
            self.assertEqual(count, 8 if "--max-images" in debug_flags else 18)

    def test_epoch_losses_weight_unequal_batches(self):
        from torch.utils.data import DataLoader, TensorDataset

        images = torch.tensor([[-2.], [-1.], [0.], [1.], [8.]])
        labels = torch.tensor([[1.], [0.], [1.], [0.], [1.]]).repeat(1, len(nt.PATHOLOGIES))
        dataset = TensorDataset(images, labels, torch.arange(5))
        frequencies, _ = nt.load_class_groups(self.data / "class_groups.csv", self.train)
        for method in sweep.SCRIPTS:
            module = baseline if method == "bce_baseline" else nt
            args = self.args(method, self.project / "loss_check")
            model = nn.Sequential()
            model.add_module("fc", nn.Linear(1, len(nt.PATHOLOGIES)))
            with torch.no_grad():
                model.fc.weight.fill_(0.7)
                model.fc.bias.fill_(0.2)
            criterion = nt.build_criterion(args, self.train)
            expected = criterion(model(images), labels).item()
            # lr=0 keeps the predictions fixed so the exact sample mean is known.
            optimizer = torch.optim.SGD(model.parameters(), lr=0)
            for batch_size in (4, 5):
                loader = DataLoader(dataset, batch_size=batch_size)
                with self.subTest(method=method, batch_size=batch_size):
                    loss = module.train_one_epoch(model, loader, torch.device("cpu"), criterion, optimizer, None, args)
                    self.assertAlmostEqual(loss, expected, places=6)
                    result = module.evaluate(model, loader, torch.device("cpu"), criterion, frequencies, args)
                    self.assertAlmostEqual(result[0]["loss"], expected, places=6)

    def test_crt_rejects_source_from_different_inputs(self):
        output = self.project / "full/bce_baseline/seed_42"
        self.train_method("bce_baseline", output, epochs=1)
        path = self.data / "train_manifest.csv"
        path.write_text(path.read_text() + "\n")
        args = self.args("crt", self.project / "crt")
        with self.assertRaisesRegex(ValueError, "different data inputs"):
            nt.configure_crt_model(TinyNetwork(), args, torch.device("cpu"))

    def test_every_sweep_candidate_parses(self):
        for method, grid in sweep.GRIDS.items():
            module = baseline if method == "bce_baseline" else module_from_path("cli_" + method, ROOT / sweep.SCRIPTS[method])
            for candidate in grid:
                with self.subTest(method=method, candidate=candidate):
                    with patch.object(sys, "argv", ["train", *sweep.cli_options({**sweep.FIXED, **candidate})]):
                        args = module.parse_args()
                    for key, value in {**sweep.FIXED, **candidate}.items():
                        self.assertEqual(getattr(args, "alpha" if key == "disable_alpha" else key),
                                         None if key == "disable_alpha" else value)

    def test_existing_runs_are_untouched_on_refusal(self):
        for method in ("bce_baseline", "focal_loss"):
            output = self.project / method
            output.mkdir()
            (output / "config.json").write_text('{"original": true}')
            before = {p.name: p.read_bytes() for p in output.iterdir()}
            with self.assertRaisesRegex(ValueError, "overwrite"):
                self.train_method(method, output)
            self.assertEqual(before, {p.name: p.read_bytes() for p in output.iterdir()})

    def test_legacy_validation_preserves_protected_runs(self):
        for method in ("bce_baseline", "focal_loss"):
            for marker in ("completed.json", "eval_val", "eval_test"):
                with self.subTest(method=method, marker=marker):
                    output = self.project / method / marker
                    self.train_method(method, output, epochs=1)
                    args = self.args(method, output, epochs=1)
                    args.mode = "evaluate-val"
                    expected = {"method": method, "seed": 42, "settings": {"lr": args.lr}}
                    if marker == "completed.json":
                        sweep.write_json(output / marker, sweep.verified_completion(output, expected))
                    else:
                        (output / marker).mkdir()
                        (output / marker / "existing.txt").write_text("preserve this evaluation")
                    before = {p.relative_to(output): p.read_bytes() for p in output.rglob("*") if p.is_file()}
                    with self.assertRaisesRegex(ValueError, "immutable"):
                        (baseline.evaluate_validation if method == "bce_baseline" else nt.run_evaluation)(args)
                    self.assertEqual(before, {p.relative_to(output): p.read_bytes() for p in output.rglob("*") if p.is_file()})
                    if marker == "completed.json":
                        sweep.completed_score(output, expected)

    def test_legacy_validation_still_evaluates_unprotected_runs(self):
        for method in ("bce_baseline", "focal_loss", "db_loss"):
            output = self.project / method
            self.train_method(method, output, epochs=1)
            expected = sweep.read_json(output / "validation_metrics.json")
            args = self.args(method, output, epochs=1)
            args.mode = "evaluate-val"
            (baseline.evaluate_validation if method == "bce_baseline" else nt.run_evaluation)(args)
            actual = sweep.read_json(output / "validation_metrics.json")
            self.assertEqual(actual["mAP"], expected["mAP"])
            self.assertEqual(actual["selected_epoch"], expected["selected_epoch"])

    def test_old_db_bias_version_cannot_resume_or_be_reevaluated(self):
        output = self.project / "old_db"
        self.train_method("db_loss", output, epochs=1)
        config = sweep.read_json(output / "config.json")
        self.assertEqual(config.pop("db_bias_version"), nt.DB_BIAS_VERSION)
        config["method_hyperparameters"].pop("db_bias_version")
        sweep.write_json(output / "config.json", config)
        for name in ("checkpoint_best.pt", "checkpoint_final.pt"):
            checkpoint = torch.load(output / name, map_location="cpu", weights_only=False)
            checkpoint["training_args"].pop("db_bias_version")
            checkpoint["method_hyperparameters"].pop("db_bias_version")
            torch.save(checkpoint, output / name)
        before = {p.relative_to(output): p.read_bytes() for p in output.rglob("*") if p.is_file()}
        with self.assertRaisesRegex(ValueError, "db_bias_version"):
            self.train_method("db_loss", output, epochs=2, resume=True)
        args = self.args("db_loss", output, epochs=1)
        args.mode = "evaluate-val"
        with self.assertRaisesRegex(ValueError, "old bias formula"):
            nt.run_evaluation(args)
        with self.assertRaisesRegex(ValueError, "old bias formula"):
            self.evaluate_method("db_loss", output, "val")
        self.assertEqual(before, {p.relative_to(output): p.read_bytes() for p in output.rglob("*") if p.is_file()})

    def test_patient_overlap_is_rejected(self):
        bad = self.val.copy()
        bad.loc[0, "patient_id"] = self.train.patient_id.iloc[0]
        report = nt.split_integrity_report(self.train, bad, self.dataset / "test_list.txt", self.dataset / "Data_Entry_2017.csv")
        self.assertFalse(report["passed"])
        self.assertTrue(report["train_validation_patient_overlap"])
        bad.loc[0, "patient_id"] = 40
        report = nt.split_integrity_report(self.train, bad, self.dataset / "test_list.txt", self.dataset / "Data_Entry_2017.csv")
        self.assertTrue(report["validation_test_patient_overlap"])

    def test_resume_preserves_early_stop_and_rejects_changed_inputs(self):
        for method in ("bce_baseline", "focal_loss"):
            output = self.project / method
            checkpoint = self.train_method(method, output, epochs=1)
            checkpoint["resume_state"]["stale_epochs"] = 7
            torch.save(checkpoint, output / "checkpoint_final.pt")
            resumed = self.train_method(method, output, epochs=3, resume=True)
            self.assertEqual(resumed["epoch"], 1)
            args = self.args(method, output, epochs=3, resume=True)
            args.lr *= 2
            with self.assertRaisesRegex(ValueError, "configuration differs"):
                (baseline.train if method == "bce_baseline" else nt.run_training)(args)
        path = self.data / "train_manifest.csv"
        path.write_text(path.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "inputs changed"):
            self.train_method("focal_loss", self.project / "focal_loss", epochs=3, resume=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
