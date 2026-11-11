#!/usr/bin/env bash
#SBATCH --job-name=mixup-resnet152
#SBATCH --array=0-2
#SBATCH --partition=bigbatch
#SBATCH --exclude=mscluster51
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --output=logs/train_balanced_mixup_%A_%a.out
#SBATCH --error=logs/train_balanced_mixup_%A_%a.err

set -euo pipefail

PROJECT_ROOT="/home-mscluster/pgohil/cxr-research"
DATASET_ROOT="/datasets/pgohil"
CONTAINER="/usr/local/containers/anaconda/anaconda-2025-12-2.sif"
ENV_PYTHON="/home-mscluster/pgohil/.conda/envs/cxr-longtail/bin/python"
IFS=: read -r -a SEEDS <<< "${TRAINING_SEEDS:-42:123:2024}"
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
if (( TASK_ID < 0 || TASK_ID >= ${#SEEDS[@]} )); then
  echo "Array task ${TASK_ID} has no matching seed in TRAINING_SEEDS=${TRAINING_SEEDS:-42:123:2024}." >&2
  exit 1
fi
SEED=${SEEDS[$TASK_ID]}
if [[ ! "${SEED}" =~ ^[0-9]+$ ]]; then
  echo "Invalid seed '${SEED}'; seeds must be non-negative integers." >&2
  exit 1
fi
METHOD="balanced_mixup"
OUTPUT_DIR="${PROJECT_ROOT}/runs/${METHOD}/seed_${SEED}"
START_TIME=$(date +%s)

mkdir -p "${PROJECT_ROOT}/logs" "${OUTPUT_DIR}"
cd "${PROJECT_ROOT}"
echo "Host: $(hostname)"; echo "Date: $(date)"; echo "Job ID: ${SLURM_JOB_ID:-none}"; echo "Seed: ${SEED}"

module purge
unset LD_PRELOAD
export MKL_THREADING_LAYER=GNU OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" PYTHONUNBUFFERED=1
if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi; then
  echo "A working GPU is required, but nvidia-smi failed. Stopping without CPU fallback." >&2
  exit 1
fi
APPTAINER_GPU_ARGS=(--nv)
apptainer exec "${APPTAINER_GPU_ARGS[@]}" "${CONTAINER}" "${ENV_PYTHON}" -c \
  'import sys, torch; print("CUDA available inside container:", torch.cuda.is_available()); sys.exit(0 if torch.cuda.is_available() else "CUDA is required but unavailable inside the container.")'

apptainer exec "${APPTAINER_GPU_ARGS[@]}" --bind /datasets:/datasets "${CONTAINER}" "${ENV_PYTHON}" -u \
  "7. Balanced MixUp/scripts/train_balanced_mixup.py" --project-root "${PROJECT_ROOT}" --dataset-root "${DATASET_ROOT}" \
  --seed "${SEED}" --output-dir "${OUTPUT_DIR}" --device cuda

apptainer exec "${APPTAINER_GPU_ARGS[@]}" --bind /datasets:/datasets "${CONTAINER}" "${ENV_PYTHON}" -u \
  evaluate.py --project-root "${PROJECT_ROOT}" --dataset-root "${DATASET_ROOT}" --method "${METHOD}" --seed "${SEED}" --split val --device cuda

apptainer exec "${APPTAINER_GPU_ARGS[@]}" --bind /datasets:/datasets "${CONTAINER}" "${ENV_PYTHON}" -u \
  evaluate.py --project-root "${PROJECT_ROOT}" --dataset-root "${DATASET_ROOT}" --method "${METHOD}" --seed "${SEED}" --split test --device cuda

python3 common/write_compute_log.py --output-dir "${OUTPUT_DIR}" --start-time "${START_TIME}" --gpus 1

echo "Finish date: $(date)"
