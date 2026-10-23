#!/usr/bin/env bash
#SBATCH --job-name=crt-resnet152
#SBATCH --array=0-2
#SBATCH --partition=bigbatch
##SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --output=logs/train_crt_stage2_%A_%a.out
#SBATCH --error=logs/train_crt_stage2_%A_%a.err

set -euo pipefail

PROJECT_ROOT="/home-mscluster/pgohil/cxr-research"
DATASET_ROOT="/datasets/pgohil"
CONTAINER="/usr/local/containers/anaconda/anaconda-2025-12-2.sif"
ENV_PYTHON="/home-mscluster/pgohil/.conda/envs/cxr-longtail/bin/python"
SEEDS=(42 123 2024)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
METHOD="crt"
BCE_FINAL="${PROJECT_ROOT}/runs/bce_baseline/seed_${SEED}/checkpoint_best.pt"
OUTPUT_DIR="${PROJECT_ROOT}/runs/${METHOD}/seed_${SEED}"
START_TIME=$(date +%s)

mkdir -p "${PROJECT_ROOT}/logs" "${OUTPUT_DIR}"
cd "${PROJECT_ROOT}"
echo "Host: $(hostname)"; echo "Date: $(date)"; echo "Job ID: ${SLURM_JOB_ID:-none}"; echo "Seed: ${SEED}"

if [[ ! -f "${BCE_FINAL}" ]]; then
  echo "Missing seed-matched BCE checkpoint: ${BCE_FINAL}" >&2
  echo "Submit this cRT array with --dependency=afterok:<bce_array_job_id> or run slurm/submit_crt_after_bce.sh after BCE finishes." >&2
  exit 1
fi

module purge
unset LD_PRELOAD
export MKL_THREADING_LAYER=GNU OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" PYTHONUNBUFFERED=1
APPTAINER_GPU_ARGS=()
if command -v nvidia-smi >/dev/null 2>&1; then APPTAINER_GPU_ARGS+=(--nv); nvidia-smi || true; fi

apptainer exec "${APPTAINER_GPU_ARGS[@]}" --bind /datasets:/datasets "${CONTAINER}" "${ENV_PYTHON}" -u \
  "6. Decoupled Training cRT/scripts/train_crt_stage2.py" --project-root "${PROJECT_ROOT}" --dataset-root "${DATASET_ROOT}" \
  --seed "${SEED}" --source-checkpoint "${BCE_FINAL}" --output-dir "${OUTPUT_DIR}" --device cuda

apptainer exec "${APPTAINER_GPU_ARGS[@]}" --bind /datasets:/datasets "${CONTAINER}" "${ENV_PYTHON}" -u \
  evaluate.py --project-root "${PROJECT_ROOT}" --dataset-root "${DATASET_ROOT}" --method "${METHOD}" --seed "${SEED}" --split val --device cuda

apptainer exec "${APPTAINER_GPU_ARGS[@]}" --bind /datasets:/datasets "${CONTAINER}" "${ENV_PYTHON}" -u \
  evaluate.py --project-root "${PROJECT_ROOT}" --dataset-root "${DATASET_ROOT}" --method "${METHOD}" --seed "${SEED}" --split test --device cuda

python3 common/write_compute_log.py --output-dir "${OUTPUT_DIR}" --start-time "${START_TIME}" --gpus 1

echo "Finish date: $(date)"
