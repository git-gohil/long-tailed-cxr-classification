#!/usr/bin/env bash
#SBATCH --job-name=nih-preprocess-eda
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=logs/preprocess_nih_%j.out
#SBATCH --error=logs/preprocess_nih_%j.err

set -euo pipefail

PROJECT_ROOT="/home-mscluster/pgohil/cxr-research"
EXP_DIR="${PROJECT_ROOT}/1. Data analysis + inspection"
DATASET_ROOT="/datasets/pgohil"

mkdir -p "${EXP_DIR}/logs" "${EXP_DIR}/results/data" "${EXP_DIR}/results/figures"

echo "Job started on: $(hostname)"
echo "Start date: $(date)"
echo "Project root: ${PROJECT_ROOT}"
echo "Experiment dir: ${EXP_DIR}"
echo "Dataset root: ${DATASET_ROOT}"

module purge

CONTAINER="/usr/local/containers/anaconda/anaconda-2025-12-2.sif"
ENV_PYTHON="/home-mscluster/pgohil/.conda/envs/cxr-longtail/bin/python"

echo "Container: ${CONTAINER}"
echo "Python environment: ${ENV_PYTHON}"

apptainer exec "${CONTAINER}" "${ENV_PYTHON}" --version

cd "${EXP_DIR}"

unset LD_PRELOAD
export MKL_THREADING_LAYER=GNU
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

apptainer exec --bind /datasets:/datasets "${CONTAINER}" "${ENV_PYTHON}" scripts/preprocess_and_analyse.py  \
  --dataset-root "${DATASET_ROOT}" \
  --project-root "${EXP_DIR}" \
  --metadata-csv /datasets/pgohil/Data_Entry_2017.csv \
  --train-val-list /datasets/pgohil/train_val_list.txt \
  --test-list /datasets/pgohil/test_list.txt \
  --num-workers 2

echo "Finish date: $(date)"
