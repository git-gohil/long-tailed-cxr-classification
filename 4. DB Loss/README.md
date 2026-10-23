# Distribution-Balanced Loss

This method combines co-occurrence-aware loss weighting with class-aware
sampling for 14-label classification using the shared ResNet-152.

## Loss

Class counts and priors come only from the training manifest. For image `i`,
the rebalancing ratio for class `c` is:

```text
r[i, c] = (1 / n[c]) / sum_j(y[i, j] / n[j])
w[i, c] = sigmoid(map_alpha * (r[i, c] - map_beta)) + map_gamma
```

Counts and denominators are clamped to avoid division by zero. The mapped
weights apply to both positive and negative class terms, so the labels that
co-occur in an image affect its loss weights.

The loss adds `init_bias * log(p / (1 - p))` to logits, where `p` is training
prevalence. It then scales negative logits by `neg_scale` and divides their
weights by the same factor before computing weighted BCE. The prior bias is
independent of negative scaling. Checkpoints record this formula as
`db_bias_version=2`.

Default parameters are mapping slope `10`, midpoint `0.2`, lift `0.1`, negative
scale `2`, and initial bias `0.05`. The sweep compares negative scales `1`, `2`,
and `5` with midpoints `0.2` and `0.3`.

## Sampling

Each training image belongs to every positive pathology's pool. The sampler
visits classes and their image pools in shuffled cycles, drawing three images
per class visit. Its epoch length is
`max(1, floor(nonempty_classes * largest_class_count / 4))`.
Images without positive labels are excluded from training sampling.
Validation and test retain their full image sets.

The sampler uses `seed + epoch` with zero-based epochs. Loss frequencies and
priors use the complete training manifest, rather than sampled frequencies.

## Implementation and outputs

- `../common/nih_longtail.py`: `DistributionBalancedLoss`, `ClassAwareSampler`, and training loop.
- `scripts/train_db_loss.py`: training entry point and DB parameters.
- `scripts/evaluate_db_loss.py`: validation evaluation of `best_model.pt`.
- `scripts/test_db_loss.py`: loss, gradient, co-occurrence weighting, and sampler checks.

Standalone outputs are under `runs/db_loss/seed_<seed>/`; final sweep outputs
are under `sweeps/<name>/final/db_loss/seed_<seed>/`. Alongside checkpoints,
configuration, history, and metrics, `class_aware_sampling.json` records class
pool sizes, epoch length, seed convention, and excluded-image count.
