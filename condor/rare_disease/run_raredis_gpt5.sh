#!/bin/bash
# run_raredis_gpt5.sh — runs RareDis RDMA pipeline with Azure GPT-5
source /home/johnwu3/miniconda3/etc/profile.d/conda.sh
conda activate rd_pyhealth

export DASK_DISTRIBUTED__DIAGNOSTICS__NVML=False

cd /home/johnwu3/projects/rare_disease/workspace/repos/RDMA

echo "[$(date)] Starting raredis_gpt5 (azure / gpt-5-john)"
python scripts/raredis/run_rdma.py \
    --condor \
    --llm_type azure \
    --model_type gpt-5-john \
    --temperature 1 \
    --entity_extractor simple \
    --resume
echo "[$(date)] Finished raredis_gpt5"
