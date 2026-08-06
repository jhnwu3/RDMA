#!/bin/bash
# run_eval_raredis_rdrag_gpt5_resume.sh
#
# Phase 1: Force re-evaluates the raredis rdrag gpt-5-john entry, which previously
#           only scored 601/1011 documents. Writes to a throwaway CSV to avoid
#           clobbering the full matrix output.
# Phase 2: Re-runs the full manifest (no force_eval) to rebuild the complete
#           rare_disease_eval_matrix_gpt-5-john.csv with the updated eval JSON.
set -euo pipefail

source /home/johnwu3/miniconda3/etc/profile.d/conda.sh
conda activate rd_pyhealth

export DASK_DISTRIBUTED__DIAGNOSTICS__NVML=False

cd /home/johnwu3/projects/rare_disease/workspace/repos/RDMA

RESUME_MANIFEST=/home/johnwu3/projects/rare_disease/workspace/condor/rare_disease/eval_manifest_gpt5_raredis_rdrag_resume.tsv
FULL_MANIFEST=/home/johnwu3/projects/rare_disease/workspace/condor/rare_disease/eval_manifest_gpt5_variants.tsv
CSV_OUT=/home/johnwu3/projects/rare_disease/workspace/results/rare_disease_eval_matrix_gpt-5-john.csv
MD_OUT=/home/johnwu3/projects/rare_disease/workspace/results/rare_disease_eval_matrix_gpt-5-john.md
TMP_CSV=/tmp/raredis_rdrag_resume_tmp.csv
TMP_MD=/tmp/raredis_rdrag_resume_tmp.md

echo "[$(date)] Phase 1: force re-eval raredis rdrag gpt-5-john"
python scripts/aggregate_rare_disease_eval_matrix.py \
    --manifest "$RESUME_MANIFEST" \
    --csv_out "$TMP_CSV" \
    --md_out "$TMP_MD" \
    --run_evals \
    --force_eval

echo "[$(date)] Phase 2: rebuild full matrix CSV from all eval JSONs"
python scripts/aggregate_rare_disease_eval_matrix.py \
    --manifest "$FULL_MANIFEST" \
    --csv_out "$CSV_OUT" \
    --md_out "$MD_OUT"

echo "[$(date)] DONE — $CSV_OUT updated"
