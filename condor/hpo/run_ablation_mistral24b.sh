#!/bin/bash
# run_ablation_mistral24b.sh — mistral_24b BioLarkGSC ablation runner
#
# Usage: run_ablation_mistral24b.sh <job_name> [extra run_hpo.py flags...]
#
# job_name is used to name the output file and log directory.
# All jobs use the retrieval extractor + v4 verifier + allow_inheritance as base
# unless overridden by extra flags.

source /home/johnwu3/miniconda3/etc/profile.d/conda.sh
conda activate rd_pyhealth

export DASK_DISTRIBUTED__DIAGNOSTICS__NVML=False

cd /home/johnwu3/projects/rare_disease/workspace/repos/RDMA

JOB_NAME="$1"
OUTPUT_DIR=/home/johnwu3/projects/rare_disease/workspace/results/biolarkgsc/ablation_mistral24b
LOG_DIR=/home/johnwu3/projects/rare_disease/workspace/logs/biolarkgsc/ablation_mistral24b/"$JOB_NAME"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

echo "[$(date)] Starting ablation_mistral24b job=$JOB_NAME"
python scripts/biolarkgsc/run_hpo.py \
    --condor \
    --model_type mistral_24b \
    --allow_inheritance \
    --output "$OUTPUT_DIR"/"$JOB_NAME"_predictions.jsonl \
    "${@:2}"
echo "[$(date)] Finished ablation_mistral24b job=$JOB_NAME"
