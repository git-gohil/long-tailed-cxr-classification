#!/usr/bin/env bash
#SBATCH --job-name=analyse-cxr-sweep
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=logs/analyse_sweep_%j.out
#SBATCH --error=logs/analyse_sweep_%j.err

set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/home-mscluster/pgohil/cxr-research}"
CONTAINER="${CONTAINER:-/usr/local/containers/anaconda/anaconda-2025-12-2.sif}"
ENV_PYTHON="${ENV_PYTHON:-/home-mscluster/pgohil/.conda/envs/cxr-longtail/bin/python}"
EXPERIMENT="${1:?Pass an experiment directory, e.g. sweeps/corrected_001}"
ANALYSIS_NAME="${2:-analysis_v1}"
if [[ ! "$ANALYSIS_NAME" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]]; then
  echo "Analysis name must contain only letters, numbers, underscores or hyphens." >&2
  exit 1
fi
cd "$PROJECT_ROOT"
EXPERIMENT=$(cd "$EXPERIMENT" && pwd)
OUTPUT="$EXPERIMENT/$ANALYSIS_NAME"
if [[ -e "$OUTPUT" ]]; then
  echo "Analysis directory already exists. Use a new name, e.g. analysis_v2: $OUTPUT" >&2
  exit 1
fi
# Check completion before either analysis starts. No analysis is queued by training.
python3 common/hyperparameter_sweep.py check "$EXPERIMENT"
module purge
unset LD_PRELOAD
export MKL_THREADING_LAYER=GNU
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONUNBUFFERED=1

apptainer exec --bind /datasets:/datasets "$CONTAINER" "$ENV_PYTHON" -u \
  analysis/aggregate_results.py --project-root "$PROJECT_ROOT" \
  --runs-dir "$EXPERIMENT/final" --output-dir "$OUTPUT/summary" --bootstrap-mode none

apptainer exec --bind /datasets:/datasets "$CONTAINER" "$ENV_PYTHON" -u \
  analysis/aggregate_results_patient_bootstrap.py --project-root "$PROJECT_ROOT" \
  --runs-dir "$EXPERIMENT/final" --output-dir "$OUTPUT/patient_bootstrap" \
  --bootstrap-resamples 1000 --bootstrap-seed 42

echo "Summary tables and patient-bootstrap confidence intervals complete: $OUTPUT"
