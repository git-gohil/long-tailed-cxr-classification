#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.nih_longtail import add_common_args, main_with_args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ResNet-152 with multi-label sigmoid focal loss.")
    add_common_args(parser, method="focal_loss", folder="3. Focal Loss")
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--alpha", type=float, default=0.25, help="Focal alpha in [0, 1]. Default: 0.25.")
    parser.add_argument("--disable-alpha", action="store_const", const=None, dest="alpha", default=argparse.SUPPRESS,
                        help="Disable positive/negative alpha weighting (used by the hyperparameter sweep).")
    return parser.parse_args()


if __name__ == "__main__":
    main_with_args(parse_args())
