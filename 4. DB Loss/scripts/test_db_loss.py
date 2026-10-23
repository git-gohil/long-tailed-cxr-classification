#!/usr/bin/env python3
from __future__ import annotations

import sys
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common.nih_longtail import ClassAwareSampler, DistributionBalancedLoss, PATHOLOGIES, make_train_sampler


def check_class_aware_sampling() -> None:
    # Disjoint classes expose the selected class directly in sampled targets.
    labels = np.zeros((45, len(PATHOLOGIES)), dtype=np.float32)
    labels[:37, 0] = 1
    labels[37:41, 1] = 1
    labels[41, 2] = 1
    df = pd.DataFrame(labels, columns=PATHOLOGIES)
    sampler = ClassAwareSampler(df, seed=42)
    drawn = list(sampler)
    assert len(drawn) == len(sampler) == 27
    assert np.all(labels[drawn].sum(axis=1) == 1)  # Excludes all-negative rows.
    selected_classes = labels[drawn].argmax(axis=1)
    assert np.array_equal(np.bincount(selected_classes, minlength=3), [9, 9, 9])
    assert np.all(selected_classes.reshape(-1, 3) == selected_classes[::3, None])
    assert len(set(i for i in drawn if i < 37)) == 9  # No repeat before pool exhaustion.
    assert list(ClassAwareSampler(df, seed=42)) == drawn
    sampler.set_epoch(1)
    assert list(sampler) != drawn
    resumed = ClassAwareSampler(df, seed=42)
    resumed.set_epoch(1)
    assert list(resumed) == list(sampler)
    assert list(ClassAwareSampler(df, seed=123)) != drawn

    # A co-occurring image must be reachable through BOTH positive classes.
    shared = df.copy()
    shared.loc[0, PATHOLOGIES[1]] = 1
    before = shared.copy()
    shared_sampler = ClassAwareSampler(shared, seed=42)
    assert 0 in shared_sampler.buckets[0] and 0 in shared_sampler.buckets[1]
    assert len(list(shared_sampler)) == 27
    pd.testing.assert_frame_equal(shared, before)
    assert list(ClassAwareSampler(df.iloc[[41, 42]])) == [0]  # Tiny subset budget.
    try:
        ClassAwareSampler(df.iloc[42:])
    except ValueError:
        pass
    else:
        raise AssertionError("All-negative training data must be rejected.")

    with tempfile.TemporaryDirectory() as tmp:
        args = SimpleNamespace(method="db_loss", seed=42)
        wired = make_train_sampler(args, df, Path(tmp))
        assert isinstance(wired, ClassAwareSampler) and list(wired) == drawn
        summary = json.loads((Path(tmp) / "class_aware_sampling.json").read_text())
        assert summary["samples_per_epoch"] == 27
        assert summary["excluded_all_negative_images"] == 3
        args.method = "focal_loss"
        assert make_train_sampler(args, df, Path(tmp)) is None


def main() -> None:
    check_class_aware_sampling()
    counts = np.array([1000, 10, 100, 1], dtype=np.float32)
    loss_fn = DistributionBalancedLoss(counts, total_samples=1200)
    logits = torch.tensor([
        [0.2, -0.3, 0.4, -0.5],
        [-0.1, 0.6, -0.7, 0.8],
        [0.9, -1.0, 1.1, -1.2],
        [-1.3, 1.4, -1.5, 1.6],
    ], requires_grad=True)
    targets = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ])
    weights = loss_fn.rebalance_weights(targets)
    assert weights.shape == targets.shape

    freq_inv = 1.0 / torch.from_numpy(counts)
    positive_mass = (targets * freq_inv).sum(dim=1, keepdim=True).clamp_min(loss_fn.eps)
    expected_weights = torch.sigmoid(
        loss_fn.map_alpha * (freq_inv.view(1, -1) / positive_mass - loss_fn.map_beta)
    ) + loss_fn.map_gamma
    assert torch.allclose(weights, expected_weights)
    assert torch.allclose(weights[0, 0], weights[1, 1])
    assert not torch.allclose(weights[0, 0], weights[2, 0])
    assert not torch.allclose(weights[0, 1:], torch.ones_like(weights[0, 1:]))

    prior = torch.from_numpy(counts) / 1200.0
    expected_bias = torch.log(prior / (1.0 - prior)) * 0.05
    # Independent softplus calculation of the positive/negative terms, including
    # the all-negative row. Bias stays fixed for every swept negative scale.
    biased = logits.detach().numpy() + expected_bias.numpy()
    y = targets.numpy()
    for scale in (1, 2, 5):
        loss_fn = DistributionBalancedLoss(counts, total_samples=1200, neg_scale=scale)
        assert torch.allclose(loss_fn.init_bias, expected_bias)
        expected_loss = np.mean(expected_weights.numpy() * (
            y * np.logaddexp(0, -biased)
            + (1 - y) * np.logaddexp(0, scale * biased) / scale
        ))
        loss = loss_fn(logits, targets)
        assert loss.ndim == 0 and torch.isfinite(loss)
        assert np.isclose(loss.item(), expected_loss, rtol=1e-6)
        logits.grad = None
        loss.backward()
        assert logits.grad is not None and torch.isfinite(logits.grad).all()
    print("DB loss and class-aware sampling tests passed.")


if __name__ == "__main__":
    main()
