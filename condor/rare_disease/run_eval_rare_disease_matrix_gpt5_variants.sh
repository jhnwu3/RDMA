#!/bin/bash
# run_eval_rare_disease_matrix_gpt5_variants.sh
#
# Runs RareDis + MIMIC3 evals for GPT-5 variants using the existing
# RDMA aggregation script + a GPT-5-only manifest.
#
# Args:
#   1 force_eval  (0|1, default 0)
set -euo pipefail

FORCE_EVAL=${1:-0}

source /home/johnwu3/miniconda3/etc/profile.d/conda.sh
conda activate rd_pyhealth

export DASK_DISTRIBUTED__DIAGNOSTICS__NVML=False

cd /home/johnwu3/projects/rare_disease/workspace/repos/RDMA

MANIFEST=/home/johnwu3/projects/rare_disease/workspace/condor/rare_disease/eval_manifest_gpt5_variants.tsv
CSV_OUT=/home/johnwu3/projects/rare_disease/workspace/results/rare_disease_eval_matrix_gpt-5-john.csv
MD_OUT=/home/johnwu3/projects/rare_disease/workspace/results/rare_disease_eval_matrix_gpt-5-john.md

ARGS="--manifest $MANIFEST --csv_out $CSV_OUT --md_out $MD_OUT --run_evals"

if [[ "$FORCE_EVAL" == "1" ]]; then
  ARGS="$ARGS --force_eval"
fi

echo "[$(date)] START aggregate_rare_disease_eval_matrix.py $ARGS"
python scripts/aggregate_rare_disease_eval_matrix.py $ARGS
echo "[$(date)] DONE"
