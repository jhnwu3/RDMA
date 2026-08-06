#!/bin/bash
# run_eval_mimic3_rd_mining_text.sh — runs one MIMIC-III text-extraction evaluation job under HTCondor
# $1 = model_type (e.g. llama3_8b, mistral_24b, llama3_70b, qwen3_32b, biobert_mrc)
# $2 = approach    (rdma | zeroshot | rdrag | dict | biobert_mrc | bioclinicalbert_ner | i2b2_rd)
source /home/johnwu3/miniconda3/etc/profile.d/conda.sh
conda activate rd_pyhealth

export DASK_DISTRIBUTED__DIAGNOSTICS__NVML=False

MODEL=$1
APPROACH=$2
LOG_BASE=/home/johnwu3/projects/rare_disease/workspace/logs/mimic3_rd_mining_text
mkdir -p "${LOG_BASE}/eval_${APPROACH}/${MODEL}" "${LOG_BASE}/bert"

cd /home/johnwu3/projects/rare_disease/workspace/repos/RDMA

RESULTS_DIR=/home/johnwu3/projects/rare_disease/workspace/results/mimic3_rd_mining

# Resolve predictions file path (mirrors eval.py logic but for the correct directory)
if   [ "$APPROACH" = "zeroshot" ]; then
    PREDS_FILE="${RESULTS_DIR}/zeroshot_${MODEL}_predictions.jsonl"
elif [ "$APPROACH" = "rdrag" ]; then
    PREDS_FILE="${RESULTS_DIR}/${MODEL}_rdrag_predictions.jsonl"
elif [ "$APPROACH" = "dict" ]; then
    PREDS_FILE="/home/johnwu3/projects/rare_disease/workspace/results/mimic3_rd_mining_text/dict_predictions.jsonl"
elif [ "$APPROACH" = "biobert_mrc" ]; then
    PREDS_FILE="/home/johnwu3/projects/rare_disease/workspace/results/mimic3/biobert_mrc/per_note_predictions.jsonl"
elif [ "$APPROACH" = "bioclinicalbert_ner" ]; then
    PREDS_FILE="/home/johnwu3/projects/rare_disease/workspace/results/mimic3/bioclinicalbert_ner/per_note_predictions.jsonl"
elif [ "$APPROACH" = "i2b2_rd" ]; then
    PREDS_FILE="/home/johnwu3/projects/rare_disease/workspace/results/mimic3/i2b2_rd/per_note_predictions.jsonl"
else  # rdma
    PREDS_FILE="${RESULTS_DIR}/${MODEL}_predictions.jsonl"
fi

echo "[$(date)] Starting eval_mimic3_rd_mining_text  model_type=${MODEL}  approach=${APPROACH}"
echo "[$(date)] Predictions file: ${PREDS_FILE}"
python scripts/mimic3_rd_mining_text/eval.py \
    --model_type "${MODEL}" \
    --approach "${APPROACH}" \
    --predictions_file "$PREDS_FILE" \
    --condor
echo "[$(date)] Finished eval_mimic3_rd_mining_text  model_type=${MODEL}  approach=${APPROACH}"
