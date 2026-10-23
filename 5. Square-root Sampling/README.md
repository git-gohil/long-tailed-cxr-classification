# Square-root Sampling

This method keeps the baseline ResNet-152 and BCE loss and changes the training
sampler to increase exposure to images containing rare pathologies.

## Method

For each image with positive labels, the sampler finds the rarest positive
class using training counts and assigns weight `n_rarest ** (-exponent)`.
The default exponent `0.5` gives inverse-square-root weighting.

Images with no positive pathology labels receive
`median(class_counts) ** (-exponent)`. Counts are clamped to at least one, and
weights are normalised to mean one. `--no-positive-weight` can override the
all-negative reference. With exponent zero and the default reference, every
image has the same sampling weight.

`WeightedRandomSampler` draws with replacement for an epoch of the original
training-set size. The sweep compares exponents `0.25`, `0.5`, and `0.75` using
validation mAP. Validation and test sampling are unchanged.

## Implementation and outputs

- `../common/nih_longtail.py`: `square_root_weights`, sampler construction, and training loop.
- `scripts/train_sqrt_sampling.py`: training entry point and sampling parameters.
- `scripts/evaluate_sqrt_sampling.py`: validation evaluation of `best_model.pt`.
- `scripts/test_sqrt_sampling.py`: checks weighting of rare-class images and exponent behaviour.

Standalone outputs are under `runs/sqrt_sampling/seed_<seed>/`; final sweep
outputs are under `sweeps/<name>/final/sqrt_sampling/seed_<seed>/`.
`sampling_weights_summary.json` and `class_counts_used.json` document the
sampling distribution alongside the shared checkpoints, history, and metrics.
