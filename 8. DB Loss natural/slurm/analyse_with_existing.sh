#!/usr/bin/env bash
#SBATCH --job-name=analyse-db-natural-all
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=logs/analyse_db_natural_all_%j.out
#SBATCH --error=logs/analyse_db_natural_all_%j.err

set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/home-mscluster/pgohil/cxr-research}"
CONTAINER="${CONTAINER:-/usr/local/containers/anaconda/anaconda-2025-12-2.sif}"
ENV_PYTHON="${ENV_PYTHON:-/home-mscluster/pgohil/.conda/envs/cxr-longtail/bin/python}"
EXPERIMENT="${1:-sweeps/db_natural_001}"
ANALYSIS_NAME="${2:-comparison_with_db_natural_v1}"
if [[ ! "$ANALYSIS_NAME" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]]; then
  echo "Analysis name must contain only letters, numbers, underscores or hyphens." >&2
  exit 1
fi
cd "$PROJECT_ROOT"
EXPERIMENT=$(cd "$EXPERIMENT" && pwd)
OUTPUT="$PROJECT_ROOT/analysis/$ANALYSIS_NAME"
if [[ -e "$OUTPUT" ]]; then
  echo "Use a new analysis name; this directory already exists: $OUTPUT" >&2
  exit 1
fi
python3 common/hyperparameter_sweep.py check "$EXPERIMENT"
OLD_FINAL=$(cd "${OLD_EXPERIMENT:-sweeps/corrected_001}/final" && pwd)
BCE_FINAL=$(cd "${BCE_EXPERIMENT:-sweeps/corrected_002_bce}/final" && pwd)
CRT_FINAL=$(cd "${CRT_EXPERIMENT:-sweeps/corrected_002_crt}/final" && pwd)
SOURCES=("$BCE_FINAL/bce_baseline" "$OLD_FINAL/focal_loss" "$OLD_FINAL/db_loss"
         "$OLD_FINAL/sqrt_sampling" "$CRT_FINAL/crt" "$OLD_FINAL/balanced_mixup"
         "$EXPERIMENT/final/db_loss_natural")
for source in "${SOURCES[@]}"; do
  if [[ ! -d "$source" ]]; then
    echo "Missing method results: $source" >&2
    exit 1
  fi
done
mkdir "$OUTPUT"
mkdir "$OUTPUT/runs"
for source in "${SOURCES[@]}"; do
  ln -s "$source" "$OUTPUT/runs/$(basename "$source")"
done
METHODS=(bce_baseline focal_loss db_loss sqrt_sampling crt balanced_mixup db_loss_natural)
module purge
unset LD_PRELOAD
export MKL_THREADING_LAYER=GNU
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONUNBUFFERED=1

# Each aggregator verifies saved evaluation hashes and the common test population.
apptainer exec --bind /datasets:/datasets "$CONTAINER" "$ENV_PYTHON" -u \
  analysis/aggregate_results.py --project-root "$PROJECT_ROOT" \
  --runs-dir "$OUTPUT/runs" --output-dir "$OUTPUT/summary" \
  --methods "${METHODS[@]}" --seeds 42 123 2024 --bootstrap-mode none

apptainer exec --bind /datasets:/datasets "$CONTAINER" "$ENV_PYTHON" -u \
  analysis/aggregate_results_patient_bootstrap.py --project-root "$PROJECT_ROOT" \
  --runs-dir "$OUTPUT/runs" --output-dir "$OUTPUT/patient_bootstrap" \
  --methods "${METHODS[@]}" --seeds 42 123 2024 \
  --bootstrap-resamples 1000 --bootstrap-seed 42

echo "Seven-method summary and patient-bootstrap confidence intervals complete: $OUTPUT"
