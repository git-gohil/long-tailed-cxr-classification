"""No GPU, training dependencies, or Slurm required: python3 -B common/test_hyperparameter_sweep.py."""
import argparse
import ast
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import hyperparameter_sweep as sweep
import evaluate_final_test_all as evaluate_all


class SweepTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.project = Path(__file__).resolve().parents[1]
        self.experiment = self.root / "experiment"
        self.experiment.mkdir()
        self.manifest = {
            "training_protocol": sweep.PROTOCOL, "seeds": sweep.SEEDS,
            "project_root": str(self.project), "dataset_root": "/datasets/pgohil",
            "bce_runs": str(self.experiment / "final/bce_baseline"), "scripts": sweep.SCRIPTS,
            "grids": {m: [{**sweep.FIXED, **c} for c in rows] for m, rows in sweep.GRIDS.items()},
        }
        sweep.write_json(self.experiment / "manifest.json", self.manifest)
        # Keep this suite runnable on login nodes without PyTorch. Pipeline tests
        # exercise the same verification with real torch.save checkpoints.
        fake_torch = SimpleNamespace(load=lambda path, **kwargs: sweep.read_json(path))
        torch_patch = patch.dict(sys.modules, {"torch": fake_torch})
        torch_patch.start()
        self.addCleanup(torch_patch.stop)

    def training_files(self, directory, method, seed, settings, score):
        resolved = dict(settings)
        if resolved.pop("disable_alpha", False):
            resolved["alpha"] = None
        config = {**resolved, "method": method, "seed": seed, "training_protocol": sweep.PROTOCOL,
                  "input_sha256": {"fixture.csv": "fixture-hash"}}
        sweep.write_json(directory / "config.json", config)
        sweep.write_json(directory / "checkpoint_best.pt", {
            "training_protocol": sweep.PROTOCOL, "method": method, "random_seed": seed,
            "training_args": config, "input_sha256": config["input_sha256"],
            "epoch": 1, "best_validation_mAP": score})
        sweep.write_json(directory / "validation_metrics.json", {"mAP": score, "selected_epoch": 1})

    def candidate(self, method, index, score):
        directory = sweep.candidate_dir(self.experiment, method, index)
        directory.mkdir(parents=True)
        settings = self.manifest["grids"][method][index]
        self.training_files(directory, method, 42, settings, score)
        expected = {"method": method, "seed": 42, "settings": settings}
        sweep.write_json(directory / "completed.json", sweep.verified_completion(directory, expected))
        return directory

    def test_exact_grids(self):
        self.assertEqual([c["lr"] for c in sweep.GRIDS["bce_baseline"]], [3e-5, 1e-4, 3e-4])
        self.assertEqual(sweep.GRIDS["focal_loss"], [{"gamma": g, "disable_alpha": True} for g in (1, 2, 5)])
        self.assertEqual({(c["db_neg_scale"], c["db_map_beta"]) for c in sweep.GRIDS["db_loss"]},
                         {(s, b) for s in (1, 2, 5) for b in (0.2, 0.3)})
        self.assertEqual(sweep.GRIDS["sqrt_sampling"], [{"sampling_exponent": x} for x in (0.25, 0.5, 0.75)])
        self.assertEqual(sweep.GRIDS["crt"], [{"lr": x, "epochs": 10} for x in (1e-4, 3e-4, 1e-3)])
        self.assertEqual(sweep.GRIDS["balanced_mixup"], [{"mixup_alpha": x} for x in (0.1, 0.2, 0.3)])
        self.assertEqual(sum(map(len, sweep.GRIDS.values())), 21)

    def test_selection_and_tie_break(self):
        for i, score in enumerate((0.2, 0.8, 0.8)):
            self.candidate("focal_loss", i, score)
        sweep.select(self.experiment, "focal_loss")
        winner = sweep.read_json(self.experiment / "selection/focal_loss/best_config.json")
        self.assertEqual(winner["candidate"], 1)
        self.assertEqual(winner["settings"]["gamma"], 2)

    def test_incomplete_candidate_blocks_selection(self):
        self.candidate("focal_loss", 0, 0.9)
        with self.assertRaises(FileNotFoundError):
            sweep.select(self.experiment, "focal_loss")
        self.assertFalse((self.experiment / "selection/focal_loss/best_config.json").exists())

    def test_nonfinite_score_blocks_selection(self):
        directory = self.candidate("focal_loss", 0, 0.9)
        (directory / "validation_metrics.json").write_text('{"mAP": NaN}')
        with self.assertRaises(ValueError):
            sweep.select(self.experiment, "focal_loss")

    def test_training_and_final_seed_commands(self):
        commands = []

        def fake_training(command, **kwargs):
            commands.append(command)
            if "--output-dir" in command:
                output = Path(command[command.index("--output-dir") + 1])
                settings = {**sweep.FIXED, "gamma": int(command[command.index("--gamma") + 1]), "disable_alpha": True}
                self.training_files(output, "focal_loss", int(command[command.index("--seed") + 1]), settings, 0.7)

        with patch.object(sweep.subprocess, "run", side_effect=fake_training):
            for index in range(3):
                sweep.run(self.experiment, "focal_loss", "sweep", index)
            self.assertEqual(len(commands), 3)  # No test evaluation during search.
            sweep.select(self.experiment, "focal_loss")
            for index in range(3):
                sweep.run(self.experiment, "focal_loss", "final", index)
        final_training = [c for c in commands[3:] if "--output-dir" in c]
        self.assertEqual([int(c[c.index("--seed") + 1]) for c in final_training], sweep.SEEDS)
        for command in final_training:
            self.assertIn("--disable-alpha", command)
            self.assertNotIn("--resume", command)
            self.assertEqual(command[command.index("--gamma") + 1], "1")
        self.assertEqual([c[c.index("--split") + 1] for c in commands if "--split" in c], ["val", "test"] * 3)
        with self.assertRaises(FileExistsError):
            sweep.run(self.experiment, "focal_loss", "final", 0)

    def test_crt_uses_seed_matched_source_and_ten_epochs(self):
        for index in range(3):
            self.candidate("crt", index, 0.7)
        sweep.select(self.experiment, "crt")
        source = self.experiment / "final/bce_baseline/seed_123/checkpoint_best.pt"
        source.parent.mkdir(parents=True)
        source.touch()

        def fake_training(command, **kwargs):
            if "--output-dir" in command:
                self.assertEqual(command[command.index("--source-checkpoint") + 1], str(source))
                self.assertEqual(command[command.index("--epochs") + 1], "10")
                self.assertNotIn("--retain-classifier", command)
                output = Path(command[command.index("--output-dir") + 1])
                self.training_files(output, "crt", 123, self.manifest["grids"]["crt"][0], 0.7)
        with patch.object(sweep.subprocess, "run", side_effect=fake_training):
            sweep.run(self.experiment, "crt", "final", 1)

    def test_submission_dependencies(self):
        project = self.root / "project with spaces"
        data = project / "1. Data analysis + inspection/results/data"
        data.mkdir(parents=True)
        for filename in ("train_manifest.csv", "val_manifest.csv", "class_groups.csv"):
            (data / filename).touch()
        (project / "slurm").mkdir()
        (project / "slurm/hyperparameter_job.sh").write_text("#!/bin/bash\n")
        for name in ["common/metrics.py", "common/run_state.py", "common/nih_longtail.py", "evaluate.py", *sweep.SCRIPTS.values()]:
            path = project / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# fixture\n")
        dataset = project / "dataset"
        dataset.mkdir()
        for name in ("Data_Entry_2017.csv", "test_list.txt"):
            (dataset / name).write_text("fixture\n")
        commands = []

        def fake_sbatch(command, **kwargs):
            commands.append(command)
            return SimpleNamespace(stdout=f"{100 + len(commands)}\n")

        args = SimpleNamespace(project_root=project, name="test", method="all", bce_runs=None,
                               dataset_root=dataset, partition="bigbatch", max_parallel=2, exclude="mscluster51",
                               gres="gpu:1", container="image.sif", env_python="python", dry_run=False)
        with patch.object(sweep.subprocess, "run", side_effect=fake_sbatch), contextlib.redirect_stdout(io.StringIO()):
            sweep.submit(args)
        self.assertEqual(len(commands), 18)
        for start in range(0, 18, 3):
            self.assertIn(f"--dependency=afterok:{101 + start}", commands[start + 1])
            self.assertIn(f"--dependency=afterok:{102 + start}", commands[start + 2])
            self.assertNotIn("--gres=gpu:1", commands[start + 1])
            self.assertIn("--array=0-2%2", commands[start + 2])
        self.assertIn("--dependency=afterok:103", commands[15])  # cRT waits for BCE final.
        self.assertIn("--array=0-5%2", commands[6])  # Six DB candidates.
        self.assertIn("--exclude=mscluster51", commands[0])

        # Standalone cRT must pin every external seed checkpoint, for previews
        # and real submissions, while all-method submission needs no BCE yet.
        bce_runs = project / "external BCE"
        for seed in sweep.SEEDS:
            source = bce_runs / f"seed_{seed}/checkpoint_best.pt"
            source.parent.mkdir(parents=True)
            source.write_bytes(f"BCE {seed}".encode())
        args.method, args.bce_runs = "crt", bce_runs
        for dry_run in (False, True):
            args.dry_run, args.name = dry_run, f"crt_{dry_run}"
            with patch.object(sweep.subprocess, "run", side_effect=fake_sbatch), contextlib.redirect_stdout(io.StringIO()):
                sweep.submit(args)
            experiment = project / "sweeps" / args.name
            manifest = sweep.read_json(experiment / "manifest.json")
            sweep.verify_manifest(manifest)
            for seed in sweep.SEEDS:
                source = bce_runs / f"seed_{seed}/checkpoint_best.pt"
                self.assertEqual(manifest["data_sha256"][str(source.resolve())], sweep.file_hash(source))
                original = source.read_bytes()
                source.write_bytes(b"replaced BCE checkpoint")
                with self.assertRaisesRegex(ValueError, "input changed"):
                    sweep.run(experiment, "crt", "sweep", 0)
                self.assertFalse(sweep.candidate_dir(experiment, "crt", 0).exists())
                source.write_bytes(original)

    def test_modified_candidate_artifacts_and_old_records_block_selection(self):
        directory = self.candidate("focal_loss", 0, 0.7)
        for name in sweep.TRAINING_FILES:
            path = directory / name
            original = path.read_bytes()
            path.write_bytes(original + b"\n")
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "files changed"):
                sweep.select(self.experiment, "focal_loss")
            path.write_bytes(original)
        record = sweep.read_json(directory / "completed.json")
        del record["output_sha256"]
        sweep.write_json(directory / "completed.json", record)
        with self.assertRaisesRegex(ValueError, "lacks verified training hashes"):
            sweep.select(self.experiment, "focal_loss")

    def test_completion_rejects_mismatched_checkpoint_and_settings(self):
        directory = self.candidate("focal_loss", 0, 0.7)
        expected = {"method": "focal_loss", "seed": 42, "settings": self.manifest["grids"]["focal_loss"][0]}
        changes = [
            ("config.json", {"lr": 0.5}),
            ("config.json", {"alpha": 0.25}),
            ("checkpoint_best.pt", {"random_seed": 123}),
            ("checkpoint_best.pt", {"training_args": {}}),
            ("checkpoint_best.pt", {"input_sha256": {"other.csv": "different"}}),
            ("validation_metrics.json", {"mAP": 1.0}),
            ("validation_metrics.json", {"selected_epoch": 2}),
        ]
        for name, changeset in changes:
            path = directory / name
            original = path.read_text()
            sweep.write_json(path, {**json.loads(original), **changeset})
            with self.subTest(name=name, changes=changeset), self.assertRaises(ValueError):
                sweep.verified_completion(directory, expected)
            path.write_text(original)

    def test_final_check_rejects_settings_different_from_training(self):
        directory = self.candidate("focal_loss", 0, 0.7)
        destination = self.experiment / "final/focal_loss/seed_42"
        destination.parent.mkdir(parents=True)
        directory.rename(destination)
        settings = {**self.manifest["grids"]["focal_loss"][0], "lr": 3e-4}
        selection = self.experiment / "selection/focal_loss"
        selection.mkdir(parents=True)
        sweep.write_json(selection / "best_config.json", {"settings": settings})
        sweep.write_json(self.experiment / "manifest.json", {
            **self.manifest, "seeds": [42], "grids": {"focal_loss": [settings]}})
        record = sweep.read_json(destination / "completed.json")
        record["settings"] = settings
        sweep.write_json(destination / "completed.json", record)
        with self.assertRaisesRegex(ValueError, "configuration differs"):
            sweep.check_experiment(self.experiment)

    def test_changed_source_blocks_worker(self):
        source = self.root / "source.py"
        source.write_text("original")
        self.manifest["source_sha256"] = {str(source): sweep.file_hash(source)}
        sweep.write_json(self.experiment / "manifest.json", self.manifest)
        source.write_text("changed")
        with self.assertRaisesRegex(ValueError, "input changed"):
            sweep.run(self.experiment, "focal_loss", "sweep", 0)
        self.assertFalse(sweep.candidate_dir(self.experiment, "focal_loss", 0).exists())

    def test_incomplete_final_blocks_analysis(self):
        with self.assertRaises(FileNotFoundError):
            sweep.check_experiment(self.experiment)

    def test_manual_analysis_gates_both_aggregators_and_preserves_paths(self):
        script = self.project / "slurm/analyse_sweep.sh"
        shell = '''
python3() { printf 'CHECK\\n'; test "$FAIL_CHECK" = 0; }
module() { :; }
apptainer() { printf 'ANALYSIS'; printf '<%s>' "$@"; printf '\\n'; }
export -f python3 module apptainer
bash "$1" "$2"
'''
        for fail in ("0", "1"):
            result = subprocess.run(["bash", "-c", shell, "test", str(script), str(self.experiment)],
                                    env={**os.environ, "PROJECT_ROOT": str(self.project), "FAIL_CHECK": fail},
                                    text=True, capture_output=True)
            self.assertEqual(result.returncode == 0, fail == "0", result.stderr)
            self.assertEqual(result.stdout.count("ANALYSIS"), 2 if fail == "0" else 0)
            if fail == "0":
                self.assertIn(f"<{self.project}>", result.stdout)
                self.assertIn(f"<{self.experiment}/final>", result.stdout)
                self.assertIn("<--bootstrap-mode><none>", result.stdout)
                self.assertIn("<--bootstrap-resamples><1000>", result.stdout)

    def test_all_method_evaluator_uses_each_seed_directory(self):
        runs = self.root / "runs with spaces"
        for method in evaluate_all.METHODS:
            for seed in sweep.SEEDS:
                directory = runs / method / f"seed_{seed}"
                directory.mkdir(parents=True)
                (directory / "checkpoint_best.pt").touch()
        with patch.object(sys, "argv", ["evaluate_all", "--project-root", str(self.project), "--runs-dir", str(runs)]):
            with patch.object(evaluate_all.subprocess, "run") as calls:
                evaluate_all.main()
        self.assertEqual(calls.call_count, 36)
        identities = []
        for call in calls.call_args_list:
            command = call.args[0]
            method, seed, split = [command[command.index(flag) + 1] for flag in ("--method", "--seed", "--split")]
            self.assertEqual(command[command.index("--run-dir") + 1], str(runs / method / f"seed_{seed}"))
            self.assertIn("--skip-existing", command)
            identities.append((method, seed, split))
        self.assertEqual(len(set(identities)), 36)

    def test_focal_cli_disables_alpha_without_changing_default(self):
        # Exercise the actual parser without importing torch/pandas on the login node.
        source = ast.parse((self.project / sweep.SCRIPTS["focal_loss"]).read_text())
        parse = next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == "parse_args")
        module = ast.Module(body=[parse], type_ignores=[])
        namespace = {"argparse": argparse, "add_common_args": lambda *args, **kwargs: None}
        exec(compile(module, "focal_parser", "exec"), namespace)
        for flags, expected in (([], 0.25), (["--disable-alpha"], None), (["--alpha", "0.75"], 0.75)):
            with patch.object(sys, "argv", ["train_focal_loss.py", *flags]):
                self.assertEqual(namespace["parse_args"]().alpha, expected)


if __name__ == "__main__":
    unittest.main()
