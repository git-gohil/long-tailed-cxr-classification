#!/usr/bin/env python3
from __future__ import annotations

import sys
import runpy
from pathlib import Path

if "--mode" not in sys.argv:
    sys.argv[1:1] = ["--mode", "evaluate-val"]

runpy.run_path(str(Path(__file__).with_name("train_balanced_mixup.py")), run_name="__main__")

