#!/usr/bin/env bash
#SBATCH --job-name=aggregate-cxr
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/aggregate_results_%j.out
#SBATCH --error=logs/aggregate_results_%j.err

set -euo pipefail

PROJECT_ROOT="/home-mscluster/pgohil/cxr-research"
CONTAINER="/usr/local/containers/anaconda/anaconda-2025-12-2.sif"
ENV_PYTHON="/home-mscluster/pgohil/.conda/envs/cxr-longtail/bin/python"

mkdir -p "${PROJECT_ROOT}/logs"
cd "${PROJECT_ROOT}"

module purge
unset LD_PRELOAD
export MKL_THREADING_LAYER=GNU
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"

apptainer exec --bind /datasets:/datasets "${CONTAINER}" "${ENV_PYTHON}" -u \
  analysis/aggregate_results.py --project-root "${PROJECT_ROOT}" --bootstrap-mode representative --bootstrap-resamples 1000
