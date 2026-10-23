"""Protect run provenance and save state needed for epoch-boundary resume."""
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch

PROTOCOL = "corrected_v2"


def input_hashes(args):
    paths = [args.train_manifest, args.val_manifest, args.test_list, args.metadata_csv]
    paths += [getattr(args, key, None) for key in ("class_groups", "source_checkpoint")]
    result = {}
    for path in paths:
        if path is None:
            continue
        path = Path(path)
        if not path.is_file():
            raise ValueError(f"Required input not found: {path}")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        result[str(path.resolve())] = digest.hexdigest()
    return result


def require_mutable_run(output):
    if any((output / name).exists() for name in ("completed.json", "eval_val", "eval_test")):
        raise ValueError("Evaluated/completed runs are immutable; use a fresh output directory")


def validate_run(args):
    """Read-only checks must happen before any metadata or checkpoint writes."""
    output = args.output_dir
    if args.epochs < 1 or args.patience < 1 or args.batch_size < 1:
        raise ValueError("epochs, patience and batch size must be positive")
    if args.resume:
        require_mutable_run(output)
        checkpoint = output / "checkpoint_final.pt"
        if not checkpoint.is_file() or not (output / "config.json").is_file():
            raise ValueError(f"Resume requires a checkpoint and configuration in {output}")
        old = json.loads((output / "config.json").read_text())
        if old.get("training_protocol") != PROTOCOL:
            raise ValueError("Old runs cannot resume under the corrected protocol; use a new output directory")
        ignored = {"resume", "allow_overwrite", "epochs", "output_dir", "mode", "check_batches"}
        changed = [key for key, value in vars(args).items()
                   if key not in ignored and old.get(key) != (str(value) if isinstance(value, Path) else value)]
        if changed:
            raise ValueError(f"Resume configuration differs: {changed}")
        if old.get("input_sha256") != input_hashes(args):
            raise ValueError("Resume inputs changed; use a new output directory")
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
        state = saved.get("resume_state")
        if saved.get("training_protocol") != PROTOCOL or not state:
            raise ValueError("Checkpoint lacks corrected resume state; start a fresh run")
        best_path = output / "checkpoint_best.pt"
        if not best_path.is_file():
            raise ValueError("Resume requires the matching best checkpoint")
        best = torch.load(best_path, map_location="cpu", weights_only=False, mmap=True)
        if best.get("epoch") != state["best_epoch"] or best.get("best_validation_mAP") != saved["best_validation_mAP"]:
            raise ValueError("Interrupted checkpoint commit: best/final checkpoints disagree; start a fresh run")
        if args.epochs < saved["epoch"]:
            raise ValueError("Resume epoch cap is smaller than the completed epoch")
    elif output.exists() and any(p.name not in {".gitkeep", ".DS_Store"} for p in output.iterdir()):
        raise ValueError(f"Refusing to overwrite existing run {output}; use a new output directory or --resume")


def capture_rng(loaders):
    return {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "loaders": [None if loader is None else {
            "generator": loader.generator.get_state() if loader.generator is not None else None,
            "sampler_rng": loader.sampler.rng.getstate() if hasattr(loader.sampler, "rng") else None,
        } for loader in loaders],
    }


def restore_rng(state, loaders):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("Resume requires the same number of visible CUDA devices")
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])
    for loader, saved in zip(loaders, state["loaders"]):
        if loader is None:
            continue
        if saved["generator"] is not None:
            loader.generator.set_state(saved["generator"].cpu())
        if saved["sampler_rng"] is not None:
            loader.sampler.rng.setstate(saved["sampler_rng"])


def atomic_checkpoint(payload, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
