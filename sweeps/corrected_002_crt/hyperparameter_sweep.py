#!/usr/bin/env python3
"""Small fixed sweeps; submission, selection and final checks use only stdlib."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys


SEEDS = [42, 123, 2024]
SCRIPTS = {
    "bce_baseline": "2. Degraded Performance/scripts/train_bce_baseline.py",
    "focal_loss": "3. Focal Loss/scripts/train_focal_loss.py",
    "db_loss": "4. DB Loss/scripts/train_db_loss.py",
    "sqrt_sampling": "5. Square-root Sampling/scripts/train_sqrt_sampling.py",
    "balanced_mixup": "7. Balanced MixUp/scripts/train_balanced_mixup.py",
    "crt": "6. Decoupled Training cRT/scripts/train_crt_stage2.py",
}
GRIDS = {
    "bce_baseline": [{"lr": x} for x in (3e-5, 1e-4, 3e-4)],
    "focal_loss": [{"gamma": x, "disable_alpha": True} for x in (1, 2, 5)],
    "db_loss": [{"db_neg_scale": x, "db_map_beta": y}
                for x in (1, 2, 5) for y in (0.2, 0.3)],
    "sqrt_sampling": [{"sampling_exponent": x} for x in (0.25, 0.5, 0.75)],
    "balanced_mixup": [{"mixup_alpha": x} for x in (0.1, 0.2, 0.3)],
    "crt": [{"lr": x, "epochs": 10} for x in (1e-4, 3e-4, 1e-3)],
}
FIXED = {"lr": 1e-4, "weight_decay": 1e-4, "batch_size": 16,
         "epochs": 30, "patience": 7, "gradient_clip": 1.0, "num_workers": 4}
PROTOCOL = "corrected_v2"
TRAINING_FILES = ("config.json", "checkpoint_best.pt", "validation_metrics.json")


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_manifest(manifest):
    if manifest.get("training_protocol") != PROTOCOL:
        raise ValueError("This workflow requires a fresh corrected_v2 experiment")
    for key in ("source_sha256", "data_sha256"):
        for path, expected in manifest.get(key, {}).items():
            if file_hash(path) != expected:
                raise ValueError(f"Queued experiment input changed: {path}")


def validate_final_run(directory, method, seed, class_groups_path=None, reference_test_rows=None):
    """Reject partial, old, mixed or modified evaluation products before analysis."""
    directory = Path(directory)
    config = read_json(directory / "config.json")
    if (config.get("training_protocol"), config.get("method"), config.get("seed")) != (PROTOCOL, method, seed):
        raise ValueError(f"Incorrect protocol/method/seed: {directory}")
    if (config.get("smoke_test") or config.get("dry_run") or
            any(config.get(key) is not None for key in ("max_train_samples", "max_val_samples"))):
        raise ValueError(f"Subset/debug training cannot be used for final analysis: {directory}")
    if method == "db_loss" and (config.get("db_bias_version") != 2 or
                               config.get("method_hyperparameters", {}).get("db_bias_version") != 2):
        raise ValueError(f"DB bias version 2 is required for final analysis; train a fresh DB run: {directory}")
    checkpoint_hashes = []
    evaluation_inputs = []
    for split in ("val", "test"):
        output = directory / f"eval_{split}"
        marker = read_json(output / "evaluation_complete.json")
        identity = marker["identity"]
        if (identity.get("training_protocol"), identity.get("method"), identity.get("seed"), identity.get("split")) != (PROTOCOL, method, seed, split):
            raise ValueError(f"Evaluation identity mismatch: {output}")
        if identity.get("max_test_samples") is not None:
            raise ValueError(f"Subset evaluation cannot be used for final analysis: {output}")
        inputs = identity.get("input_sha256")
        if not inputs or any(inputs[path] != digest for path, digest in config.get("input_sha256", {}).items() if path in inputs):
            raise ValueError(f"Evaluation inputs differ from training: {output}")
        if class_groups_path is not None:
            key = str(Path(class_groups_path).resolve())
            if inputs.get(key) != file_hash(class_groups_path):
                raise ValueError(f"Analysis class groups differ from evaluation: {output}")
        evaluation_inputs.append(inputs)
        required = {"predictions.npy", "logits.npy", "labels.npy", "image_ids.npy", "patient_ids.npy", "metrics.json"}
        required.add("tuned_thresholds.json" if split == "val" else "metrics_tuned_threshold.json")
        hashes = marker["output_sha256"]
        if not required <= hashes.keys() or any(file_hash(output / name) != digest for name, digest in hashes.items()):
            raise ValueError(f"Evaluation files are incomplete or changed: {output}")
        checkpoint_hashes.append(identity["checkpoint_sha256"])
    if len(set(checkpoint_hashes)) != 1 or not (directory / "checkpoint_best.pt").is_file():
        raise ValueError(f"Validation/test checkpoint mismatch: {directory}")
    if file_hash(directory / "checkpoint_best.pt") != checkpoint_hashes[0]:
        raise ValueError(f"Best checkpoint changed after evaluation: {directory}")
    if evaluation_inputs[0] != evaluation_inputs[1]:
        raise ValueError(f"Validation/test input mismatch: {directory}")
    test_identity = read_json(directory / "eval_test/evaluation_complete.json")["identity"]
    if test_identity.get("validation_thresholds_sha256") != file_hash(directory / "eval_val/tuned_thresholds.json"):
        raise ValueError(f"Test thresholds do not match validation: {directory}")
    # These canonical arrays identify the test population and row order. Their
    # hashes were checked against the actual files above, so this stays stdlib-only.
    test_rows = {name: hashes[name] for name in ("labels.npy", "image_ids.npy", "patient_ids.npy")}
    if reference_test_rows is not None and test_rows != reference_test_rows:
        raise ValueError(f"Test rows are not identically aligned in {directory}; "
                         "all methods and seeds must use the same labels, image order, and patient IDs")
    return test_rows


def check_experiment(experiment):
    manifest = read_json(experiment / "manifest.json")
    verify_manifest(manifest)
    count = 0
    reference_test_rows = None
    for method in manifest["grids"]:
        settings = read_json(experiment / "selection" / method / "best_config.json")["settings"]
        for seed in manifest["seeds"]:
            directory = experiment / "final" / method / f"seed_{seed}"
            completed_score(directory, {"method": method, "seed": seed, "settings": settings})
            reference_test_rows = validate_final_run(directory, method, seed, reference_test_rows=reference_test_rows)
            count += 1
    print(f"Verified {count} complete corrected final runs. Ready for analysis.")


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def cli_options(settings):
    result = []
    for key, value in settings.items():
        if value is False or value is None:
            continue
        result.append("--" + key.replace("_", "-"))
        if value is not True:
            result.append(str(value))
    return result


def candidate_dir(experiment, method, index):
    return experiment / "candidates" / method / f"candidate_{index:02d}" / "seed_42"


def validate_training_settings(values, expected):
    required = {"method": expected["method"], "seed": expected["seed"]}
    for key, value in expected["settings"].items():
        # The focal CLI resolves --disable-alpha into alpha=None.
        if key == "disable_alpha" and value is True:
            required["alpha"] = None
        else:
            required[key] = value
    changed = [key for key, value in required.items() if key not in values or values[key] != value]
    if changed:
        raise ValueError(f"Training configuration differs from selected settings: {changed}")


def verified_completion(directory, expected):
    """Verify real checkpoint metadata in the worker's training environment."""
    import torch

    hashes = {name: file_hash(directory / name) for name in TRAINING_FILES}
    config = read_json(directory / "config.json")
    validate_training_settings(config, expected)
    checkpoint = torch.load(directory / "checkpoint_best.pt", map_location="cpu", weights_only=False, mmap=True)
    if (config.get("training_protocol"), checkpoint.get("training_protocol"),
            checkpoint.get("method"), checkpoint.get("random_seed")) != (PROTOCOL, PROTOCOL, expected["method"], expected["seed"]):
        raise ValueError(f"Checkpoint protocol/method/seed mismatch: {directory}")
    validate_training_settings(checkpoint.get("training_args", {}), expected)
    if not config.get("input_sha256") or checkpoint.get("input_sha256") != config["input_sha256"]:
        raise ValueError(f"Checkpoint inputs differ from configuration: {directory}")
    metrics = read_json(directory / "validation_metrics.json")
    score = float(metrics["mAP"])
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError(f"Invalid validation mAP: {score}")
    if score != checkpoint.get("best_validation_mAP") or metrics.get("selected_epoch") != checkpoint.get("epoch"):
        raise ValueError(f"Validation score/epoch differs from best checkpoint: {directory}")
    return {**expected, "output_sha256": hashes}


def completed_score(directory, expected):
    record = read_json(directory / "completed.json")
    if {key: record.get(key) for key in expected} != expected:
        raise ValueError(f"Completion/configuration mismatch: {directory}")
    hashes = record.get("output_sha256", {})
    if set(hashes) != set(TRAINING_FILES):
        raise ValueError(f"Completion record lacks verified training hashes; use a fresh sweep: {directory}")
    if any(file_hash(directory / name) != digest for name, digest in hashes.items()):
        raise ValueError(f"Completed training files changed: {directory}")
    config = read_json(directory / "config.json")
    if config.get("training_protocol") != PROTOCOL:
        raise ValueError(f"Incorrect training protocol: {directory}")
    validate_training_settings(config, expected)
    score = float(read_json(directory / "validation_metrics.json")["mAP"])
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError(f"Invalid validation mAP in {directory}: {score}")
    return score


def select(experiment, method):
    manifest = read_json(experiment / "manifest.json")
    verify_manifest(manifest)
    rankings = []
    for index, settings in enumerate(manifest["grids"][method]):
        expected = {"method": method, "seed": 42, "settings": settings}
        score = completed_score(candidate_dir(experiment, method, index), expected)
        rankings.append({"candidate": index, "validation_mAP": score, "settings": settings})
    rankings.sort(key=lambda row: (-row["validation_mAP"], row["candidate"]))
    output = experiment / "selection" / method
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "ranking.json", rankings)
    write_json(output / "best_config.json", rankings[0])
    print(f"{method}: selected candidate {rankings[0]['candidate']} "
          f"with validation mAP {rankings[0]['validation_mAP']:.6f}")


def run(experiment, method, phase, index):
    manifest = read_json(experiment / "manifest.json")
    verify_manifest(manifest)
    if phase == "sweep":
        candidates = manifest["grids"][method]
        if not 0 <= index < len(candidates):
            raise ValueError("Candidate index out of range")
        settings, seed = candidates[index], 42
        output = candidate_dir(experiment, method, index)
    else:
        if not 0 <= index < len(SEEDS):
            raise ValueError("Seed index out of range")
        settings = read_json(experiment / "selection" / method / "best_config.json")["settings"]
        seed = SEEDS[index]
        output = experiment / "final" / method / f"seed_{seed}"
    expected = {"method": method, "seed": seed, "settings": settings}
    # Never let an existing trainer overwrite an old or partially completed run.
    output.mkdir(parents=True, exist_ok=False)
    project = Path(manifest["project_root"])
    options = {**settings, "project_root": project,
               "dataset_root": manifest["dataset_root"],
               "test_list": str(Path(manifest["dataset_root"]) / "test_list.txt"),
               "seed": seed, "output_dir": output, "device": "cuda"}
    options["metadata_csv"] = str(Path(manifest["dataset_root"]) / "Data_Entry_2017.csv")
    if method == "crt":
        source = Path(manifest["bce_runs"]) / f"seed_{seed}" / "checkpoint_best.pt"
        if not source.is_file():
            raise ValueError(f"Missing seed-matched BCE checkpoint: {source}")
        options["source_checkpoint"] = source
    subprocess.run([sys.executable, str(project / manifest["scripts"][method]),
                    *cli_options(options)], check=True, cwd=project)
    # Training already writes metrics for the checkpoint selected on validation mAP.
    if phase == "final":
        for split in ("val", "test"):
            subprocess.run([sys.executable, str(project / "evaluate.py"),
                            "--project-root", str(project), "--dataset-root", manifest["dataset_root"],
                            "--metadata-csv", str(Path(manifest["dataset_root"]) / "Data_Entry_2017.csv"),
                            "--test-list", str(Path(manifest["dataset_root"]) / "test_list.txt"),
                            "--method", method, "--seed", str(seed), "--run-dir", str(output),
                            "--split", split, "--device", "cuda"], check=True, cwd=project)
    # A killed/failed process never gets a completion marker.
    verify_manifest(manifest)
    write_json(output / "completed.json", verified_completion(output, expected))


def submit(args):
    project = args.project_root.resolve()
    experiment = project / "sweeps" / args.name
    methods = list(SCRIPTS) if args.method == "all" else [args.method]
    data = project / "1. Data analysis + inspection/results/data"
    for filename in ("train_manifest.csv", "val_manifest.csv", "class_groups.csv"):
        if not (data / filename).is_file():
            raise ValueError(f"Preprocessing file missing: {data / filename}")
    bce_runs = (args.bce_runs.resolve() if args.bce_runs else
                (experiment / "final/bce_baseline" if args.method == "all" else project / "runs/bce_baseline"))
    if args.method == "crt":
        for seed in SEEDS:
            if not (bce_runs / f"seed_{seed}/checkpoint_best.pt").is_file():
                raise ValueError(f"Missing BCE seed {seed} under {bce_runs}")
    source_paths = [project / name for name in ["common/metrics.py", "common/run_state.py", "common/nih_longtail.py", "evaluate.py", *SCRIPTS.values()]]
    data_paths = [data / name for name in ("train_manifest.csv", "val_manifest.csv", "class_groups.csv")]
    if args.method == "crt":
        data_paths.extend(bce_runs / f"seed_{seed}/checkpoint_best.pt" for seed in SEEDS)
    dataset_files = [args.dataset_root / name for name in ("Data_Entry_2017.csv", "test_list.txt")]
    if not args.dry_run:
        data_paths.extend(dataset_files)
    manifest = {"training_protocol": PROTOCOL,
                "source_sha256": {str(p): file_hash(p) for p in source_paths},
                "data_sha256": {str(p.resolve()): file_hash(p) for p in data_paths},
                "project_root": str(project), "dataset_root": str(args.dataset_root.resolve()),
                "bce_runs": str(bce_runs), "scripts": SCRIPTS, "seeds": SEEDS,
                "selection_metric": "validation_mAP", "sweep_seed": 42,
                "grids": {m: [{**FIXED, **c} for c in GRIDS[m]] for m in methods}}
    if not (project / "slurm/hyperparameter_job.sh").is_file():
        raise ValueError("Missing slurm/hyperparameter_job.sh")
    experiment.mkdir(parents=True, exist_ok=False)
    (experiment / "logs").mkdir()
    write_json(experiment / "manifest.json", manifest)
    # Snapshot the small workflow so later grid edits cannot alter queued jobs.
    shutil.copy2(Path(__file__), experiment / "hyperparameter_sweep.py")
    shutil.copy2(project / "slurm/hyperparameter_job.sh", experiment / "hyperparameter_job.sh")
    jobs = {}

    def enqueue(method, phase, dependency=None):
        command = ["sbatch", "--parsable", f"--partition={args.partition}",
                   f"--job-name={method}-{phase}",
                   f"--output={experiment}/logs/{method}_{phase}_%A_%a.out",
                   f"--error={experiment}/logs/{method}_{phase}_%A_%a.err"]
        if dependency:
            command.append(f"--dependency=afterok:{dependency}")
        if phase == "select":
            command += ["--cpus-per-task=1", "--mem=1G", "--time=00:10:00"]
        else:
            count = len(GRIDS[method]) if phase == "sweep" else len(SEEDS)
            command.append(f"--array=0-{count - 1}%{args.max_parallel}")
            if args.exclude:
                command.append(f"--exclude={args.exclude}")
            if args.gres != "none":
                command.append(f"--gres={args.gres}")
        command += [str(experiment / "hyperparameter_job.sh"), phase, str(experiment), method,
                    str(project), args.container, args.env_python]
        if args.dry_run:
            import shlex
            print(shlex.join(command))
            return str(1000 + sum(len(v) for v in jobs.values()))
        result = subprocess.run(command, check=True, text=True, capture_output=True)
        job_id = result.stdout.strip().split(";")[0]
        if not job_id.isdigit():
            raise ValueError(f"Unexpected sbatch response: {result.stdout!r}")
        return job_id

    for method in methods:
        jobs[method] = {}
        dependency = jobs["bce_baseline"]["final"] if method == "crt" and args.method == "all" else None
        for phase in ("sweep", "select", "final"):
            job_id = enqueue(method, phase, dependency)
            jobs[method][phase] = job_id
            write_json(experiment / "jobs.json", {"dry_run": args.dry_run, "jobs": jobs})
            dependency = job_id
            print(f"{method} {phase}: {job_id}")
    print(f"Experiment: {experiment}")
    if args.dry_run:
        print("Preview only: no jobs submitted. Use a NEW name for the real submission.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    launch = sub.add_parser("submit", help="Submit sweep -> selection -> three final seeds")
    launch.add_argument("--method", choices=["all", *SCRIPTS], required=True)
    launch.add_argument("--name", required=True, help="New experiment name; letters, numbers, _ and - only")
    launch.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    launch.add_argument("--dataset-root", type=Path, default=Path("/datasets/pgohil"))
    launch.add_argument("--bce-runs", type=Path, help="For standalone cRT: directory containing seed_*/checkpoint_best.pt")
    launch.add_argument("--max-parallel", type=int, default=2, help="Concurrency per array, not across methods")
    launch.add_argument("--partition", default="bigbatch")
    launch.add_argument("--gres", default="none", help="Defaults to existing scripts' allocation convention; use gpu:1 if required")
    launch.add_argument("--exclude", default="mscluster51", help="Excluded compute nodes; default retains the existing MixUp job exclusion. Use an empty string to disable.")
    launch.add_argument("--container", default="/usr/local/containers/anaconda/anaconda-2025-12-2.sif")
    launch.add_argument("--env-python", default="/home-mscluster/pgohil/.conda/envs/cxr-longtail/bin/python")
    launch.add_argument("--dry-run", action="store_true", help="Write experiment plan and print submissions without calling sbatch")
    checker = sub.add_parser("check", help="Verify every final run before submitting analysis")
    checker.add_argument("experiment", type=Path)
    for name in ("run", "select"):
        worker = sub.add_parser(name)
        worker.add_argument("experiment", type=Path)
        worker.add_argument("method", choices=list(SCRIPTS))
        if name == "run":
            worker.add_argument("--phase", choices=["sweep", "final"], required=True)
            worker.add_argument("--index", type=int, required=True)
    args = parser.parse_args()
    if args.command == "submit":
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.name) or args.max_parallel < 1:
            parser.error("Use a simple experiment name and --max-parallel >= 1")
        if args.bce_runs and args.method != "crt":
            parser.error("--bce-runs applies only to --method crt; all uses its selected BCE runs")
        submit(args)
    elif args.command == "check":
        check_experiment(args.experiment.resolve())
    elif args.command == "select":
        select(args.experiment.resolve(), args.method)
    else:
        run(args.experiment.resolve(), args.method, args.phase, args.index)


if __name__ == "__main__":
    main()
