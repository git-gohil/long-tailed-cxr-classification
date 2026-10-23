# Balanced MixUp

Balanced MixUp combines ordinary sampling with a class-balanced partner stream.
It uses the shared ResNet-152 and BCE with logits on soft multi-label targets.

## Method

Each training step draws an ordinary batch and a partner batch. The partner
sampler assigns images to their rarest positive pathology's bucket, with a
separate bucket for images without positive labels, then samples uniformly
across nonempty buckets.

A single mixing weight is drawn for the batch:

```text
lambda ~ Beta(alpha, 1)
mixed_image = (1 - lambda) * ordinary_image + lambda * balanced_image
mixed_target = (1 - lambda) * ordinary_target + lambda * balanced_target
```

Mixing follows augmentation and ImageNet normalisation. All 14 target entries
are mixed, clamped to `[0, 1]`, and passed to BCE without thresholding.
Validation and test use unmixed images.

The standalone default is `--mixup-alpha 0.4`. The sweep compares alpha values
`0.1`, `0.2`, and `0.3`, selecting by validation mAP. `--disable-mixup` disables
mixing for an ablation.

## Implementation and outputs

- `../common/nih_longtail.py`: `balanced_mixup_batch`, partner sampler, and training loop.
- `scripts/train_balanced_mixup.py`: training entry point and MixUp parameters.
- `scripts/evaluate_balanced_mixup.py`: validation evaluation of `best_model.pt`.
- `scripts/test_balanced_mixup.py`: checks tensor shapes, target bounds, mixing weights, gradients, and unmixed validation.

Standalone outputs are under `runs/balanced_mixup/seed_<seed>/`; final sweep
outputs are under `sweeps/<name>/final/balanced_mixup/seed_<seed>/`.
`balanced_mixup_sampling_stats.json` records mixing weights and partner-stream
class counts alongside the shared checkpoints, history, and metrics.
