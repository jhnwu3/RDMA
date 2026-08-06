#!/bin/bash
# run_biolarkgsc_ablation.sh — runs one ablation job for BioLark GSC prompt-flag sweep
# Usage: run_biolarkgsc_ablation.sh <combo_tag> <extra_args...>
# e.g.:  run_biolarkgsc_ablation.sh dc1_eq1_ai1 --decompose_compound --extract_qualified --allow_inheritance
source /home/johnwu3/miniconda3/etc/profile.d/conda.sh
conda activate rd_pyhealth

export DASK_DISTRIBUTED__DIAGNOSTICS__NVML=False

MODEL_TYPE="mistral_24b"
COMBO_TAG="$1"
shift 1   # remaining args are the flag overrides

RESULTS_DIR="/home/johnwu3/projects/rare_disease/workspace/results/biolarkgsc/ablation"
OUTPUT="${RESULTS_DIR}/${MODEL_TYPE}_${COMBO_TAG}_predictions.jsonl"
LOG_DIR="/home/johnwu3/projects/rare_disease/workspace/logs/biolarkgsc/hpo/ablation/${MODEL_TYPE}"

mkdir -p "$RESULTS_DIR"
mkdir -p "$LOG_DIR"

cd /home/johnwu3/projects/rare_disease/workspace/repos/RDMA

echo "[$(date)] Starting ablation  model=${MODEL_TYPE}  combo=${COMBO_TAG}"
echo "[$(date)] Extra args: $@"

python scripts/biolarkgsc/run_hpo.py \
    --condor \
    --model_type "$MODEL_TYPE" \
    --output "$OUTPUT" \
    "$@"

echo "[$(date)] Finished ablation  model=${MODEL_TYPE}  combo=${COMBO_TAG}"
echo "[$(date)] Output: $OUTPUT"
