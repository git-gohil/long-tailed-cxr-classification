# Focal Loss

This method trains the shared ResNet-152 with sigmoid focal loss for 14-label
classification. The model returns raw logits.

## Method

For each label, focal loss multiplies binary cross-entropy with logits by
`(1 - p_t) ** gamma`, where `p_t` is the predicted probability of the observed
binary target. This reduces the contribution of easy examples. With gamma zero
and alpha weighting disabled, the loss reduces to ordinary BCE.

Standalone defaults are `--gamma 2.0` and `--alpha 0.25`.
`--disable-alpha` removes positive/negative alpha weighting. The sweep disables
alpha weighting and compares gamma values `1`, `2`, and `5`, selecting by
validation mAP.

## Implementation

- `../common/nih_longtail.py`: `SigmoidFocalLoss` and the shared training loop.
- `scripts/train_focal_loss.py`: training entry point and focal parameters.
- `scripts/evaluate_focal_loss.py`: validation evaluation of `best_model.pt`.
- `scripts/test_focal_loss.py`: checks finite loss, gradients, BCE equivalence, and easy-example weighting.

Training uses the shared manifests and class groups from
`1. Data analysis + inspection/results/data/`. The sampler, preprocessing,
optimiser, and checkpoint selection follow the baseline.

Standalone outputs are under `runs/focal_loss/seed_<seed>/`; final sweep outputs
are under `sweeps/<name>/final/focal_loss/seed_<seed>/`. They include checkpoints,
configuration, training history, validation predictions, per-class metrics, and
learning curves. The shared `../evaluate.py` produces validation and test outputs.
