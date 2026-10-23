#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.nih_longtail import SigmoidFocalLoss


def main() -> None:
    logits = torch.tensor([[3.0, -3.0], [0.1, -0.1]], requires_grad=True)
    targets = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    loss = SigmoidFocalLoss(gamma=2.0)(logits, targets)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits.detach(), targets)
    focal_g0 = SigmoidFocalLoss(gamma=0.0)(logits.detach(), targets)
    assert torch.allclose(focal_g0, bce, atol=1e-6)
    easy = SigmoidFocalLoss(gamma=2.0)(torch.tensor([[6.0, -6.0]]), torch.tensor([[1.0, 0.0]]))
    hard = SigmoidFocalLoss(gamma=2.0)(torch.tensor([[0.0, 0.0]]), torch.tensor([[1.0, 0.0]]))
    assert easy < hard
    print("Focal loss tests passed.")


if __name__ == "__main__":
    main()

