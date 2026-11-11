#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.nih_longtail import DB_BIAS_VERSION, add_common_args, main_with_args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ResNet-152 with Distribution-Balanced Loss and natural sampling.")
    add_common_args(parser, method="db_loss_natural", folder="8. DB Loss natural")
    parser.set_defaults(db_bias_version=DB_BIAS_VERSION)
    parser.add_argument("--db-map-alpha", type=float, default=10.0)
    parser.add_argument("--db-map-beta", type=float, default=0.2)
    parser.add_argument("--db-map-gamma", type=float, default=0.1)
    parser.add_argument("--db-neg-scale", type=float, default=2.0)
    parser.add_argument("--db-init-bias", type=float, default=0.05)
    args = parser.parse_args()
    if args.method != "db_loss_natural":
        parser.error("This entry point requires --method db_loss_natural")
    return args


if __name__ == "__main__":
    main_with_args(parse_args())
