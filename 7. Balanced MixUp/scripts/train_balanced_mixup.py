#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.nih_longtail import add_common_args, main_with_args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ResNet-152 with Balanced-MixUp for multi-label classification.")
    add_common_args(parser, method="balanced_mixup", folder="7. Balanced MixUp")
    parser.add_argument("--mixup-alpha", type=float, default=0.4)
    parser.add_argument("--disable-mixup", action="store_true")
    args = parser.parse_args()
    if args.mixup_alpha <= 0:
        raise SystemExit("--mixup-alpha must be > 0")
    return args


if __name__ == "__main__":
    main_with_args(parse_args())

