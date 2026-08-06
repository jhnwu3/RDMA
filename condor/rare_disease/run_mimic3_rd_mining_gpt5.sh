#!/bin/bash
# run_mimic3_rd_mining_gpt5.sh — runs MIMIC-III RDMA pipeline with Azure GPT-5
source /home/johnwu3/miniconda3/etc/profile.d/conda.sh
conda activate rd_pyhealth

export DASK_DISTRIBUTED__DIAGNOSTICS__NVML=False

cd /home/johnwu3/projects/rare_disease/workspace/repos/RDMA

echo "[$(date)] Starting mimic3_rd_mining_gpt5 (azure / gpt-5-john)"
python scripts/mimic3_rd_mining_code/run_rdma.py \
    --condor \
    --llm_type azure \
    --model_type gpt-5-john \
    --temperature 1 \
    --entity_extractor simple \
    --mimic3_root /srv/local/data/MIMIC-III/mimic-iii-clinical-database-1.4 \
    --resume
echo "[$(date)] Finished mimic3_rd_mining_gpt5"
