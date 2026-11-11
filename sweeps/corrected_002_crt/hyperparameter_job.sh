#!/usr/bin/env bash
#SBATCH --partition=bigbatch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=48:00:00

set -euo pipefail
PHASE=$1
EXPERIMENT=$2
METHOD=$3
PROJECT_ROOT=$4
CONTAINER=$5
ENV_PYTHON=$6
cd "$PROJECT_ROOT"

if [[ "$PHASE" == select ]]; then
  exec python3 "$EXPERIMENT/hyperparameter_sweep.py" select "$EXPERIMENT" "$METHOD"
fi

module purge
unset LD_PRELOAD
export MKL_THREADING_LAYER=GNU
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONUNBUFFERED=1
exec apptainer exec --nv --bind /datasets:/datasets "$CONTAINER" "$ENV_PYTHON" -u \
  "$EXPERIMENT/hyperparameter_sweep.py" run "$EXPERIMENT" "$METHOD" \
  --phase "$PHASE" --index "$SLURM_ARRAY_TASK_ID"
