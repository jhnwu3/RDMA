#!/bin/bash
# run_eval_biolarkgsc_LLM.sh — LLM-based FP/FN root-cause analysis for BioLark GSC
source /home/johnwu3/miniconda3/etc/profile.d/conda.sh
conda activate rd_pyhealth

export DASK_DISTRIBUTED__DIAGNOSTICS__NVML=False

cd /home/johnwu3/projects/rare_disease/workspace/repos/RDMA

mkdir -p /home/johnwu3/projects/rare_disease/workspace/logs/biolarkgsc/llm_analysis

echo "[$(date)] Starting LLM error analysis (azure / gpt-5-john)"
python scripts/biolarkgsc/eval_biolarkgsc_LLM.py \
    --llm_type azure \
    --model_type gpt-5-john \
    --predictions /home/johnwu3/projects/rare_disease/workspace/results/biolarkgsc/mistral_24b_predictions.jsonl
echo "[$(date)] Finished LLM error analysis"
