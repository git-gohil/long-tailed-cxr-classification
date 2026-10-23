#!/usr/bin/env bash
#SBATCH --job-name=focal-resnet152
#SBATCH --array=0-2
#SBATCH --partition=bigbatch
##SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --output=logs/train_focal_loss_%A_%a.out
#SBATCH --error=logs/train_focal_loss_%A_%a.err

set -euo pipefail

PROJECT_ROOT="/home-mscluster/pgohil/cxr-research"
DATASET_ROOT="/datasets/pgohil"
CONTAINER="/usr/local/containers/anaconda/anaconda-2025-12-2.sif"
ENV_PYTHON="/home-mscluster/pgohil/.conda/envs/cxr-longtail/bin/python"
SEEDS=(42 123 2024)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
METHOD="focal_loss"
OUTPUT_DIR="${PROJECT_ROOT}/runs/${METHOD}/seed_${SEED}"
START_TIME=$(date +%s)

mkdir -p "${PROJECT_ROOT}/logs" "${OUTPUT_DIR}"
cd "${PROJECT_ROOT}"

echo "Host: $(hostname)"
echo "Date: $(date)"
echo "Job ID: ${SLURM_JOB_ID:-none}"
echo "Seed: ${SEED}"
echo "Output dir: ${OUTPUT_DIR}"

module purge
unset LD_PRELOAD
export MKL_THREADING_LAYER=GNU
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1

APPTAINER_GPU_ARGS=()
if command -v nvidia-smi >/dev/null 2>&1; then APPTAINER_GPU_ARGS+=(--nv); nvidia-smi || true; fi

apptainer exec "${APPTAINER_GPU_ARGS[@]}" --bind /datasets:/datasets "${CONTAINER}" "${ENV_PYTHON}" -u \
  "3. Focal Loss/scripts/train_focal_loss.py" --project-root "${PROJECT_ROOT}" --dataset-root "${DATASET_ROOT}" \
  --seed "${SEED}" --output-dir "${OUTPUT_DIR}" --device cuda

apptainer exec "${APPTAINER_GPU_ARGS[@]}" --bind /datasets:/datasets "${CONTAINER}" "${ENV_PYTHON}" -u \
  evaluate.py --project-root "${PROJECT_ROOT}" --dataset-root "${DATASET_ROOT}" --method "${METHOD}" --seed "${SEED}" --split val --device cuda

apptainer exec "${APPTAINER_GPU_ARGS[@]}" --bind /datasets:/datasets "${CONTAINER}" "${ENV_PYTHON}" -u \
  evaluate.py --project-root "${PROJECT_ROOT}" --dataset-root "${DATASET_ROOT}" --method "${METHOD}" --seed "${SEED}" --split test --device cuda

python3 common/write_compute_log.py --output-dir "${OUTPUT_DIR}" --start-time "${START_TIME}" --gpus 1

echo "Finish date: $(date)"
