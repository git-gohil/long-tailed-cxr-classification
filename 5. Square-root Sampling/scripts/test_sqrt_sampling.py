#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.nih_longtail import PATHOLOGIES, square_root_weights


def main() -> None:
    rows = []
    for i in range(12):
        row = {"image_filename": f"{i}.png", "image_path": f"/tmp/{i}.png"}
        row.update({p: 0 for p in PATHOLOGIES})
        row["Atelectasis"] = int(i < 10)
        row["Hernia"] = int(i == 11)
        rows.append(row)
    df = pd.DataFrame(rows)
    weights = square_root_weights(df, exponent=0.5)
    assert weights[11] > weights[0]
    assert max(abs(square_root_weights(df, exponent=0.0) - 1.0)) < 1e-9
    print("Square-root sampling tests passed.")


if __name__ == "__main__":
    main()

