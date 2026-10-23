# Decoupled Training: Classifier Re-training

Classifier re-training (cRT) separates feature learning from classifier
balancing. Stage 1 is the trained BCE baseline from `2. Degraded Performance`.
Stage 2 uses its best checkpoint from the matching training seed.

## Method

Stage 2 freezes the ResNet-152 feature extractor and reinitialises the final
14-output linear classifier. Only classifier parameters receive updates.
The backbone stays in evaluation mode, keeping BatchNorm running statistics
fixed. `--retain-classifier` preserves the BCE classifier for an ablation.

Training uses BCE and a class-balanced sampler. Each image is assigned to its
rarest positive pathology's bucket; images with no positive pathology form a
separate bucket. The sampler chooses uniformly among nonempty buckets and then
uniformly among images in the selected bucket. Targets retain all 14 labels.

The sweep compares stage-2 learning rates `1e-4`, `3e-4`, and `1e-3`, with a
10-epoch cap. It selects by validation mAP. Each final seed uses the selected
BCE checkpoint trained with that same seed.

## Implementation and outputs

- `../common/nih_longtail.py`: model freezing, balanced sampler, and training loop.
- `scripts/prepare_crt_stage1.py`: BCE source-checkpoint validation.
- `scripts/train_crt_stage2.py`: stage-2 training entry point.
- `scripts/evaluate_crt_stage2.py`: validation evaluation of `best_model.pt`.
- `scripts/test_crt_freezing.py`: verifies that backbone parameters and buffers stay fixed.

Source checks verify the BCE method, seed, pathology order, and data inputs.
`source_bce_checkpoint_info.json` records the source path and checksum;
`parameter_report.json` lists trainable and frozen parameters.

Standalone outputs are under `runs/crt/seed_<seed>/`; final sweep outputs are
under `sweeps/<name>/final/crt/seed_<seed>/`, alongside the shared checkpoints,
configuration, history, and evaluation metrics.
