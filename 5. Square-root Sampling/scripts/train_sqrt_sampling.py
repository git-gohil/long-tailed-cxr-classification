#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.nih_longtail import add_common_args, main_with_args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ResNet-152 with square-root instance re-sampling and BCE.")
    add_common_args(parser, method="sqrt_sampling", folder="5. Square-root Sampling")
    parser.add_argument("--sampling-exponent", type=float, default=0.5, help="0.5 gives 1/sqrt(n_rarest); 0 gives uniform instance weights.")
    parser.add_argument("--no-positive-weight", type=float, default=None, help="Optional fixed weight for no-positive/No Finding images.")
    return parser.parse_args()


if __name__ == "__main__":
    main_with_args(parse_args())

