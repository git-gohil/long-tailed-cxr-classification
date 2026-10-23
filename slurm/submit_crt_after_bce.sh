#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home-mscluster/pgohil/cxr-research"
SEEDS=(42 123 2024)

for SEED in "${SEEDS[@]}"; do
  CHECKPOINT="${PROJECT_ROOT}/runs/bce_baseline/seed_${SEED}/checkpoint_best.pt"
  if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "Missing ${CHECKPOINT}" >&2
    exit 1
  fi
done

cd "${PROJECT_ROOT}"
mkdir -p logs
sbatch "6. Decoupled Training cRT/slurm/train_crt_stage2.sh"
