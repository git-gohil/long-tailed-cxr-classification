#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.nih_longtail import train_one_epoch


class TinyClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4), nn.ReLU())
        self.fc = nn.Linear(4, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.fc(self.backbone(inputs))


def main() -> None:
    torch.manual_seed(42)
    model = TinyClassifier()
    for parameter in model.backbone.parameters():
        parameter.requires_grad = False

    batch_norm = model.backbone[1]
    running_mean_before = batch_norm.running_mean.clone()
    running_var_before = batch_norm.running_var.clone()
    batches_before = batch_norm.num_batches_tracked.clone()
    classifier_before = model.fc.weight.detach().clone()

    images = torch.randn(8, 4)
    labels = torch.randint(0, 2, (8, 2)).float()
    loader = DataLoader(TensorDataset(images, labels, torch.arange(8)), batch_size=4)
    optimizer = torch.optim.SGD(model.fc.parameters(), lr=0.1)
    args = SimpleNamespace(method="crt", stage="stage2", gradient_clip=0.0)

    loss = train_one_epoch(
        model,
        loader,
        torch.device("cpu"),
        nn.BCEWithLogitsLoss(),
        optimizer,
        scaler=None,
        args=args,
    )

    assert torch.isfinite(torch.tensor(loss))
    assert not torch.equal(model.fc.weight.detach(), classifier_before)
    assert torch.equal(batch_norm.running_mean, running_mean_before)
    assert torch.equal(batch_norm.running_var, running_var_before)
    assert torch.equal(batch_norm.num_batches_tracked, batches_before)
    assert not batch_norm.training and model.fc.training
    assert all(parameter.grad is None for parameter in model.backbone.parameters())
    print("cRT freezing test passed.")


if __name__ == "__main__":
    main()
