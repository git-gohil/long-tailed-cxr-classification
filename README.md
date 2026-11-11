# Long-tailed chest X-ray classification

Research code comparing six training strategies on NIH ChestX-ray14 using
ResNet-152, patient-disjoint splits, and seeds 42, 123, and 2024.

Reported test results (%, mean ± sample standard deviation over three seeds):

| Method | Overall mAP | Tail mAP | Validation-tuned bACC |
| --- | --- | --- | --- |
| BCE | **28.37 ± 0.34** | 22.19 ± 1.41 | 72.56 ± 0.21 |
| Focal loss | 28.20 ± 0.30 | 22.27 ± 2.34 | 72.70 ± 0.41 |
| DB loss | 28.11 ± 0.35 | 22.56 ± 1.64 | 72.03 ± 0.20 |
| Power sampling (q = 0.75) | 26.55 ± 0.52 | 21.40 ± 3.48 | 71.43 ± 0.21 |
| Classifier retraining (cRT) | 28.18 ± 0.39 | 22.17 ± 1.22 | **73.05 ± 0.36** |
| Balanced MixUp | 28.33 ± 0.29 | **23.74 ± 1.48** | 72.68 ± 0.19 |

BCE achieved the highest overall mAP. Balanced MixUp's mean tail gain was
uncertain and concentrated in hernia. cRT's higher balanced accuracy accompanied
fewer false positives and more missed positive cases.

Numbered folders contain preprocessing and method scripts; `common/` contains
shared training and sweep utilities, `analysis/` aggregates results, and `slurm/`
contains cluster jobs. The reported DB method is `db_loss_natural` in folder 8;
folder 4 retains the class-aware sampling variant.

Cluster project root: `/home-mscluster/pgohil/cxr-research`.
Dataset root: `/datasets/pgohil`.

## Launch the selected configurations on the cluster

Run the following in **Bash on the cluster login node**, after preprocessing has
created `train_manifest.csv`, `val_manifest.csv`, and `class_groups.csv` under
`1. Data analysis + inspection/results/data/`. 

```bash
cd /home-mscluster/pgohil/cxr-research
mkdir -p logs
export CXR_RUN_NAME=report_selected_001

submit_method() {
  local method="$1"
  shift
  sbatch --parsable --job-name="${method}-selected" \
    --export="ALL,CXR_METHOD=${method}" "$@" <<'SLURM'
#!/usr/bin/env bash
#SBATCH --partition=bigbatch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --exclude=mscluster51
#SBATCH --array=0-2%2
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail
PROJECT_ROOT=/home-mscluster/pgohil/cxr-research
CONTAINER=/usr/local/containers/anaconda/anaconda-2025-12-2.sif
ENV_PYTHON=/home-mscluster/pgohil/.conda/envs/cxr-longtail/bin/python
RUNS_DIR="$PROJECT_ROOT/runs/$CXR_RUN_NAME"
SEEDS=(42 123 2024)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
EPOCHS=30
METHOD_ARGS=()

case "$CXR_METHOD" in
  bce_baseline)
    SCRIPT="2. Degraded Performance/scripts/train_bce_baseline.py" ;;
  focal_loss)
    SCRIPT="3. Focal Loss/scripts/train_focal_loss.py"
    METHOD_ARGS=(--gamma 2 --disable-alpha) ;;
  db_loss_natural)
    SCRIPT="8. DB Loss natural/scripts/train_db_loss_natural.py"
    METHOD_ARGS=(--db-neg-scale 5 --db-map-beta 0.3) ;;
  sqrt_sampling)
    SCRIPT="5. Square-root Sampling/scripts/train_sqrt_sampling.py"
    METHOD_ARGS=(--sampling-exponent 0.75) ;;
  crt)
    SCRIPT="6. Decoupled Training cRT/scripts/train_crt_stage2.py"
    EPOCHS=10
    METHOD_ARGS=(--source-checkpoint "$RUNS_DIR/bce_baseline/seed_$SEED/checkpoint_best.pt") ;;
  balanced_mixup)
    SCRIPT="7. Balanced MixUp/scripts/train_balanced_mixup.py"
    METHOD_ARGS=(--mixup-alpha 0.3) ;;
  *) echo "Unknown method: $CXR_METHOD" >&2; exit 1 ;;
esac

cd "$PROJECT_ROOT"
module purge
unset LD_PRELOAD
export MKL_THREADING_LAYER=GNU
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"

exec apptainer exec --nv --bind /datasets:/datasets "$CONTAINER" "$ENV_PYTHON" -u \
  "$SCRIPT" --project-root "$PROJECT_ROOT" --dataset-root /datasets/pgohil \
  --metadata-csv /datasets/pgohil/Data_Entry_2017.csv \
  --test-list /datasets/pgohil/test_list.txt \
  --output-dir "$RUNS_DIR/$CXR_METHOD/seed_$SEED" --seed "$SEED" \
  --device cuda --lr 1e-4 --weight-decay 1e-4 --batch-size 16 \
  --epochs "$EPOCHS" --patience 7 --gradient-clip 1.0 --num-workers 4 \
  "${METHOD_ARGS[@]}"
SLURM
}

BCE_JOB=$(submit_method bce_baseline)
submit_method focal_loss
submit_method db_loss_natural
submit_method sqrt_sampling
submit_method balanced_mixup
submit_method crt --dependency="afterok:${BCE_JOB%%;*}"
```

The cRT array waits for all three BCE jobs to succeed, then loads each seed's
matching checkpoint. To launch a subset, define the function and run only the
relevant submission lines, keeping the BCE submission and dependency for cRT.
Monitor jobs with `squeue -u "$USER"`. Training checkpoints, configurations, and
validation metrics are written to
`runs/<CXR_RUN_NAME>/<method>/seed_<seed>/`; Slurm logs are written to `logs/`.

## Evaluate bACC after all methods finish

Run this final step **after all selected training jobs, including cRT, have
finished successfully**. It reports test balanced accuracy at 0.5 and test
balanced accuracy with one threshold per disease, method, and seed chosen to
maximize **validation bACC**. Test labels are used only for final scoring.
The existing `eval_val/tuned_thresholds.json` optimizes F1; this analysis saves
its own bACC thresholds separately.

The training commands above save validation `.npz` files but do not export test
predictions. The following job first uses the existing checkpoint evaluator to
create `eval_val/` and `eval_test/`, then runs the bACC analysis at the end.
Existing complete evaluations are verified and reused. This step uses the
already trained checkpoints; it does not train models again.

Submit from **Bash on the cluster login node**, using the same run name as the
training jobs:

```bash
cd /home-mscluster/pgohil/cxr-research
mkdir -p logs
export CXR_RUN_NAME=report_selected_001
sbatch --job-name=bacc-final --export=ALL,CXR_RUN_NAME <<'SLURM'
#!/usr/bin/env bash
#SBATCH --partition=bigbatch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --exclude=mscluster51
#SBATCH --output=logs/bacc_final_%j.out
#SBATCH --error=logs/bacc_final_%j.err

set -euo pipefail
PROJECT_ROOT=/home-mscluster/pgohil/cxr-research
CONTAINER=/usr/local/containers/anaconda/anaconda-2025-12-2.sif
ENV_PYTHON=/home-mscluster/pgohil/.conda/envs/cxr-longtail/bin/python
RUNS_DIR="$PROJECT_ROOT/runs/$CXR_RUN_NAME"
METHODS=(bce_baseline focal_loss db_loss_natural sqrt_sampling crt balanced_mixup)

cd "$PROJECT_ROOT"
module purge
unset LD_PRELOAD
export MKL_THREADING_LAYER=GNU
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"

apptainer exec --nv --bind /datasets:/datasets "$CONTAINER" "$ENV_PYTHON" -u \
  common/evaluate_final_test_all.py --project-root "$PROJECT_ROOT" \
  --runs-dir "$RUNS_DIR" --dataset-root /datasets/pgohil \
  --methods "${METHODS[@]}" --seeds 42 123 2024 --device cuda

apptainer exec "$CONTAINER" "$ENV_PYTHON" -u \
  analysis/evaluate_bacc_from_saved.py --runs-dir "$RUNS_DIR" \
  --output-dir "$PROJECT_ROOT/analysis/bacc_results/$CXR_RUN_NAME" \
  --methods "${METHODS[@]}" --seeds 42 123 2024 \
  --class-groups "$PROJECT_ROOT/1. Data analysis + inspection/results/data/class_groups.csv"
SLURM
```

If both evaluation folders already exist, run only the following array analysis
from the project root, in an environment with Python and NumPy. It runs on CPU
and does not need checkpoints, original images, or PyTorch:

```bash
python analysis/evaluate_bacc_from_saved.py \
  --runs-dir runs/report_selected_001 \
  --output-dir analysis/bacc_results/report_selected_001 \
  --methods bce_baseline focal_loss db_loss_natural sqrt_sampling crt balanced_mixup \
  --seeds 42 123 2024 \
  --class-groups "1. Data analysis + inspection/results/data/class_groups.csv"
```

For sweep results, set `--runs-dir` to the relevant `sweeps/<experiment>/final`
directory or an existing combined `analysis/<comparison>/runs` directory.
Set `--methods` and `--seeds` to exactly the completed runs you want to compare;
`db_loss` is also supported separately from `db_loss_natural`. Use a new output
directory for each analysis, such as `analysis/bacc_results/report_selected_001_v2`
when repeating an analysis. Existing output directories are never overwritten.

Each `<runs-dir>/<method>/seed_<seed>/` must contain `config.json` and, in **both**
`eval_val/` and `eval_test/`, `predictions.npy`, `labels.npy`, `image_ids.npy`,
`patient_ids.npy`, and `evaluation_complete.json`. These are the existing
evaluator's outputs. The script checks their recorded hashes, matching
validation/test checkpoint fingerprints, disease order, identical evaluation
rows across runs, and disjoint validation/test images and patients. It rejects
missing runs, subset/debug runs, and diseases lacking positives or negatives;
it does not silently omit them. `logits.npy`, F1 thresholds, and existing metric
JSON files are not needed. Omit `--class-groups` for overall/per-disease results
only; when supplied, its contents must match the recorded evaluation inputs.

Threshold selection searches every distinct validation probability, keeping
tied scores together, plus an all-negative endpoint just above the largest
validation probability. Equal validation bACC maxima use the highest candidate
threshold. An all-negative endpoint can be slightly above 1 when the maximum
probability is 1. bACC is `(sensitivity + specificity) / 2`; overall and group
scores average diseases equally, then summarize the separate seed scores.
This tunes decision thresholds for the saved models; their original checkpoint
and hyperparameter selection remains based on validation mAP.

The job log prints mean test bACC and sample standard deviation as percentages.
The four files below are created in `analysis/bacc_results/<CXR_RUN_NAME>/`:

| File | Contents |
| --- | --- |
| `bacc_thresholds.json` | Validation-bACC-selected thresholds, validation scores, tuning rule, and input fingerprints for every method/seed |
| `bacc_per_class.csv` | Disease-level thresholds, validation bACC, test bACC at 0.5 and tuned thresholds, sensitivity, specificity, and positive/negative counts |
| `bacc_by_seed.csv` | Overall and optional head/medium/tail test metrics for each method/seed |
| `bacc_summary.csv` | Test bACC mean and sample standard deviation across seeds for each method/group, with seed and class counts |

Metric values in CSV/JSON are fractions from 0 to 1; multiply by 100 for
percentages. With one seed, the sample-standard-deviation fields are blank.
The script also accepts `--self-test` to check the calculation and file workflow
using temporary synthetic data, without needing saved runs.
