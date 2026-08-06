#!/bin/bash
# Run all missing evals across all 5 benchmarks, then aggregate with bootstrap CIs.
# Args:
#   1 force_eval  (0|1, default 0)
set -euo pipefail

FORCE_EVAL=${1:-0}
LOG_DIR=/home/johnwu3/projects/rare_disease/workspace/logs/eval_all_benchmarks

mkdir -p "$LOG_DIR"

source /home/johnwu3/miniconda3/etc/profile.d/conda.sh
conda activate rd_pyhealth

export DASK_DISTRIBUTED__DIAGNOSTICS__NVML=False

cd /home/johnwu3/projects/rare_disease/workspace/repos/RDMA

ARGS="--manifest /home/johnwu3/projects/rare_disease/workspace/condor/rare_disease/eval_manifest_all_benchmarks.tsv"
ARGS="$ARGS --csv_out /home/johnwu3/projects/rare_disease/workspace/results/full_eval_matrix.csv"
ARGS="$ARGS --md_out /home/johnwu3/projects/rare_disease/workspace/results/full_eval_matrix.md"
ARGS="$ARGS --run_evals --bootstrap"

if [[ "$FORCE_EVAL" == "1" ]]; then
    ARGS="$ARGS --force_eval"
fi

echo "[$(date)] START aggregate_rare_disease_eval_matrix.py $ARGS"
python scripts/aggregate_rare_disease_eval_matrix.py $ARGS
echo "[$(date)] DONE"
