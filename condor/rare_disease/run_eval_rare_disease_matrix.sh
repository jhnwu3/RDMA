#!/bin/bash
# Run all missing evals sequentially via the aggregation script.
# Args:
#   1 force_eval  (0|1, default 0)
set -euo pipefail

FORCE_EVAL=${1:-0}
LOG_DIR=/home/johnwu3/projects/rare_disease/workspace/logs/rare_disease_eval_matrix

# Ensure condor stdout/stderr target directory exists before execution.
mkdir -p "$LOG_DIR"

source /home/johnwu3/miniconda3/etc/profile.d/conda.sh
conda activate rd_pyhealth

export DASK_DISTRIBUTED__DIAGNOSTICS__NVML=False

cd /home/johnwu3/projects/rare_disease/workspace/repos/RDMA

ARGS=""

# Prefer running evals when supported by the current script version.
if python scripts/aggregate_rare_disease_eval_matrix.py --help | grep -q -- "--run_evals"; then
  ARGS="--run_evals"
fi

if [[ "$FORCE_EVAL" == "1" ]]; then
  if python scripts/aggregate_rare_disease_eval_matrix.py --help | grep -q -- "--force_eval"; then
    ARGS="$ARGS --force_eval"
  fi
fi

echo "[$(date)] START aggregate_rare_disease_eval_matrix.py $ARGS"
python scripts/aggregate_rare_disease_eval_matrix.py $ARGS
echo "[$(date)] DONE"
