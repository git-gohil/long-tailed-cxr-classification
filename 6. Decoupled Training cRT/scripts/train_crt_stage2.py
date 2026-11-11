#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.nih_longtail import add_common_args, main_with_args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 2 classifier re-training for ResNet-152 BCE baseline.")
    add_common_args(parser, method="crt", folder="6. Decoupled Training cRT")
    parser.add_argument("--stage", default="stage2", choices=["stage2"])
    parser.add_argument("--source-checkpoint", type=Path, default=None)
    parser.add_argument("--retain-classifier", action="store_true", help="Keep the BCE classifier instead of reinitialising fc.")
    args = parser.parse_args()
    if args.source_checkpoint is None:
        args.source_checkpoint = args.project_root / "runs" / "bce_baseline" / f"seed_{args.seed}" / "checkpoint_best.pt"
    return args


if __name__ == "__main__":
    main_with_args(parse_args())
