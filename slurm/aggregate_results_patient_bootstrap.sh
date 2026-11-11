#!/usr/bin/env bash
#SBATCH --job-name=patient-bootstrap-cxr
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=logs/patient_bootstrap_%j.out
#SBATCH --error=logs/patient_bootstrap_%j.err

set -euo pipefail

PROJECT_ROOT="/home-mscluster/pgohil/cxr-research"
OUTPUT_DIR="${PROJECT_ROOT}/analysis/results_patient_bootstrap"
CONTAINER="/usr/local/containers/anaconda/anaconda-2025-12-2.sif"
ENV_PYTHON="/home-mscluster/pgohil/.conda/envs/cxr-longtail/bin/python"

mkdir -p "${PROJECT_ROOT}/logs" "${OUTPUT_DIR}"
cd "${PROJECT_ROOT}"

module purge
unset LD_PRELOAD
export MKL_THREADING_LAYER=GNU
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1

echo "Patient-bootstrap job started: $(date)"
echo "Output directory: ${OUTPUT_DIR}"
echo "The old analysis/results directory will not be modified."

apptainer exec --bind /datasets:/datasets "${CONTAINER}" "${ENV_PYTHON}" -u \
  analysis/aggregate_results_patient_bootstrap.py \
  --project-root "${PROJECT_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --bootstrap-resamples 1000 \
  --bootstrap-seed 42

echo "Patient-bootstrap job finished: $(date)"
echo "New results: ${OUTPUT_DIR}"
