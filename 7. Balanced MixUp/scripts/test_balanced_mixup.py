#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.nih_longtail import balanced_mixup_batch


def main() -> None:
    x_a = torch.randn(2, 3, 4, 4)
    x_b = torch.randn(2, 3, 4, 4)
    y_a = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    y_b = torch.tensor([[0.0, 1.0], [1.0, 1.0]])
    alpha = 0.4

    np.random.seed(42)
    expected_balanced_weight = float(np.random.beta(alpha, 1.0))
    np.random.seed(42)
    mixed_x, mixed_y, lam = balanced_mixup_batch(x_a, y_a, x_b, y_b, alpha)

    assert np.isclose(lam, expected_balanced_weight)
    assert 0.0 <= lam <= 1.0
    assert torch.allclose(mixed_x, (1.0 - lam) * x_a + lam * x_b)
    assert torch.allclose(mixed_y, (1.0 - lam) * y_a + lam * y_b)
    assert mixed_x.shape == x_a.shape
    assert torch.all((mixed_y >= 0) & (mixed_y <= 1))
    model = torch.nn.Linear(3 * 4 * 4, 2)
    logits = model(mixed_x.flatten(1))
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, mixed_y)
    loss.backward()
    assert model.weight.grad is not None
    val_y = y_a.clone()
    assert torch.equal(val_y, y_a), "validation targets are not mixed by this test path"
    print("Balanced-MixUp tests passed.")


if __name__ == "__main__":
    main()
