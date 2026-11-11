#!/usr/bin/env python3
"""Regenerate BCE baseline plots from saved validation metrics."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


if "--mode" not in sys.argv:
    sys.argv[1:1] = ["--mode", "plot-only"]

runpy.run_path(str(Path(__file__).with_name("train_bce_baseline.py")), run_name="__main__")
