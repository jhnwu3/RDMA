#!/bin/bash
# run_revised_hpo_biolarkgsc.sh — BioLark GSC HPO pipeline with prompt engineering fixes
# $1 = model_type (e.g. llama3_8b, mistral_24b, llama3_70b, qwen3_32b)
# $2... = optional extra run_hpo.py flags
#
# Enabled fixes vs baseline:
#   --exclude_etiology   : suppress molecular-mechanism/chromosomal terms at extraction
#   --prefer_specific    : prefer most-specific HPO node over parent/ancestor at matching
#   --require_grounding  : suppress fallback match when LLM finds no textual grounding
source /home/johnwu3/miniconda3/etc/profile.d/conda.sh
conda activate rd_pyhealth

export DASK_DISTRIBUTED__DIAGNOSTICS__NVML=False

cd /home/johnwu3/projects/rare_disease/workspace/repos/RDMA

OUTPUT_DIR=/home/johnwu3/projects/rare_disease/workspace/results/biolarkgsc/revised_hpo
LOG_DIR=/home/johnwu3/projects/rare_disease/workspace/logs/biolarkgsc/revised_hpo/"$1"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

echo "[$(date)] Starting revised_hpo_biolarkgsc  model_type=$1"
python scripts/biolarkgsc/run_hpo.py \
    --condor \
    --model_type "$1" \
    --exclude_etiology \
    --prefer_specific \
    --require_grounding \
    --output "$OUTPUT_DIR"/"$1"_predictions.jsonl \
    "${@:2}"
echo "[$(date)] Finished revised_hpo_biolarkgsc  model_type=$1"
