#!/bin/bash
#SBATCH --job-name=sqrt-smoke
#SBATCH --partition=bigbatch
##SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=logs/smoke_sqrt_sampling_%j.out
#SBATCH --error=logs/smoke_sqrt_sampling_%j.err
set -euo pipefail
PROJECT_ROOT="/home-mscluster/pgohil/cxr-research"; EXP_DIR="${PROJECT_ROOT}/5. Square-root Sampling"; DATASET_ROOT="/datasets/pgohil"
CONTAINER="/usr/local/containers/anaconda/anaconda-2025-12-2.sif"; ENV_PYTHON="/home-mscluster/pgohil/.conda/envs/cxr-longtail/bin/python"
mkdir -p "${EXP_DIR}/logs" "${EXP_DIR}/results" "${EXP_DIR}/checkpoints"; cd "${EXP_DIR}"
echo "Host: $(hostname)"; echo "Date: $(date)"; echo "Job ID: ${SLURM_JOB_ID:-none}"
module purge; unset LD_PRELOAD; export MKL_THREADING_LAYER=GNU OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" PYTHONUNBUFFERED=1
APPTAINER_GPU_ARGS=(); if command -v nvidia-smi >/dev/null 2>&1; then APPTAINER_GPU_ARGS+=(--nv); nvidia-smi || true; fi
apptainer exec "${APPTAINER_GPU_ARGS[@]}" --bind /datasets:/datasets "${CONTAINER}" "${ENV_PYTHON}" - <<'PY'
import sys, torch
print("Python:", sys.version); print("PyTorch:", torch.__version__); print("CUDA available:", torch.cuda.is_available())
PY
apptainer exec "${APPTAINER_GPU_ARGS[@]}" --bind /datasets:/datasets "${CONTAINER}" "${ENV_PYTHON}" -u scripts/train_sqrt_sampling.py --project-root "${PROJECT_ROOT}" --dataset-root "${DATASET_ROOT}" --smoke-test --epochs 1

