#!/bin/bash
# run_biolarkgsc_gpt5.sh — runs BioLark GSC HPO pipeline with Azure GPT-5 (RDMA)
source /home/johnwu3/miniconda3/etc/profile.d/conda.sh
conda activate rd_pyhealth

export DASK_DISTRIBUTED__DIAGNOSTICS__NVML=False

cd /home/johnwu3/projects/rare_disease/workspace/repos/RDMA

mkdir -p /home/johnwu3/projects/rare_disease/workspace/logs/biolarkgsc/hpo/gpt-5-john

echo "[$(date)] Starting biolarkgsc_gpt5 (azure / gpt-5-john)"
python scripts/biolarkgsc/run_hpo.py \
    --condor \
    --llm_type azure \
    --model_type gpt-5-john \
    --temperature 1 \
    --extraction_temperature 1 \
    --entity_extractor simple
echo "[$(date)] Finished biolarkgsc_gpt5"
