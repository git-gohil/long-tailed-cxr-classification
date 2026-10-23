#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.nih_longtail import PATHOLOGIES, project_default, sha256_file
from common.run_state import PROTOCOL


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the BCE baseline checkpoint for cRT Stage 1.")
    parser.add_argument("--project-root", type=Path, default=project_default())
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source-checkpoint", type=Path, default=None)
    args = parser.parse_args()
    ckpt = args.source_checkpoint or args.project_root / "runs" / "bce_baseline" / f"seed_{args.seed}" / "checkpoint_best.pt"
    if not ckpt.exists():
        raise SystemExit(f"BCE checkpoint not found: {ckpt}")
    data = torch.load(ckpt, map_location="cpu", weights_only=False)
    if (data.get("training_protocol"), data.get("method"), data.get("random_seed")) != (PROTOCOL, "bce_baseline", args.seed):
        raise SystemExit("Stage 1 must be a corrected BCE checkpoint with the requested seed.")
    if data.get("pathology_order", data.get("class_names")) != PATHOLOGIES:
        raise SystemExit("Stage 1 checkpoint pathology order differs.")
    state = data.get("model_state_dict", {})
    fc_weight = state.get("fc.weight")
    if fc_weight is None or tuple(fc_weight.shape)[0] != len(PATHOLOGIES):
        raise SystemExit("Checkpoint does not contain a 14-output ResNet final layer.")
    info = {"source_checkpoint": str(ckpt), "sha256": sha256_file(ckpt), "epoch": data.get("epoch"), "class_names": PATHOLOGIES}
    print(json.dumps(info, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
