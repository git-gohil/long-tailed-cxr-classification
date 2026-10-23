# Long-tailed Classification of Pathologies in Chest X-rays

Can long-tailed learning improve recognition of rare chest X-ray findings? This
project compares six training strategies on NIH ChestX-ray14 using a shared
ImageNet-pretrained ResNet-152. It examines overall performance, rare-pathology
recognition, and how co-occurring labels affect missed cases and false alarms.

See the [research report](research_report.pdf) for methods and full analysis.

## Results

Test performance (%), mean ± sample standard deviation across three seeds.
Mean average precision (mAP) is the primary metric; balanced accuracy (bACC)
uses thresholds selected on validation data. The tail classes are hernia and
pneumonia.

| Method | Overall mAP | Tail mAP | Tuned bACC |
|---|---:|---:|---:|
| BCE baseline | **28.37 ± 0.34** | 22.19 ± 1.41 | 72.56 ± 0.21 |
| Focal loss | 28.20 ± 0.30 | 22.27 ± 2.34 | 72.70 ± 0.41 |
| Distribution-balanced loss | 25.44 ± 0.31 | 16.59 ± 3.40 | 70.16 ± 0.03 |
| Power sampling (`q = 0.75`) | 26.55 ± 0.52 | 21.40 ± 3.48 | 71.43 ± 0.21 |
| Classifier re-training (cRT) | 28.18 ± 0.39 | 22.17 ± 1.22 | **73.05 ± 0.36** |
| Balanced MixUp | 28.33 ± 0.29 | **23.74 ± 1.48** | 72.68 ± 0.19 |

Balanced MixUp's tail improvement over BCE is uncertain: +1.54 percentage
points, with a 95% patient-bootstrap interval of [-0.56, 3.55]. The gain comes
from hernia. cRT reduces false alarms but misses more positive cases, showing
why aggregate scores need pathology-level error analysis.

## Repository structure

```text
1. Data analysis + inspection/  Dataset inspection and preprocessing
2. Degraded Performance/        BCE baseline
3. Focal Loss/                  Focal loss
4. DB Loss/                     Distribution-balanced loss
5. Square-root Sampling/        Power-law sampling and exponent sweep
6. Decoupled Training cRT/      Classifier re-training
7. Balanced MixUp/              Balanced image and label mixing
common/                        Shared training, metrics, and sweep utilities
analysis/                      Result aggregation and patient bootstrap
evaluate.py                    Validation and test evaluation
slurm/                         Job scripts
sweeps/                        Experiment configurations and results
```

## Experimental setup

All methods use the same patient-disjoint splits: 75,505 training images,
11,019 validation images, and 25,596 test images. Each image retains its full
14-label target vector. Images are resized to 256 × 256 and cropped to
224 × 224, with ImageNet normalisation. Training adds random crops, horizontal
flips, rotations, and brightness/contrast jitter; evaluation uses centre crops.

Training uses PyTorch and an ImageNet-pretrained ResNet-152 with a 14-output
classifier. AdamW uses batch size 16, an initial learning rate of `1e-4` for
the selected configurations, weight decay `1e-4`, mixed precision, and gradient
clipping at 1.0. A validation-mAP plateau scheduler halves the learning rate
after two non-improving epochs. Training runs for up to 30 epochs, with
early-stopping patience seven; cRT trains only the classifier for up to 10 epochs.

Hyperparameter candidates are compared using seed 42 and selected by validation
mAP. Each selected configuration is then trained with seeds 42, 123, and 2024,
giving 18 final runs. cRT starts from the matching seed's BCE backbone. The
checkpoint with the highest validation mAP is used for final evaluation.

Test results report overall and head/medium/tail mAP, with per-pathology bACC
thresholds chosen on validation data and applied unchanged to test predictions.
Results are averaged across three seeds. For mAP, 1,000 paired patient-bootstrap
resamples retain all images from each sampled patient and use identical draws
across methods. Pathology-level analysis also examines missed cases, false
alarms, and label co-occurrence.
