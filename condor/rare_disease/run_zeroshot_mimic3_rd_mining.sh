#!/bin/bash
# run_zeroshot_mimic3_rd_mining.sh — runs one zero-shot MIMIC-III rare-disease mining baseline job under HTCondor
# $1 = model_type (e.g. llama3_8b, mistral_24b, llama3_70b, qwen3_32b)
source /home/johnwu3/miniconda3/etc/profile.d/conda.sh
conda activate rd_pyhealth

export DASK_DISTRIBUTED__DIAGNOSTICS__NVML=False

cd /home/johnwu3/projects/rare_disease/workspace/repos/RDMA

echo "[$(date)] Starting zeroshot_mimic3_rd_mining  model_type=$1"
python baselines/mimic3_rd_mining_code/zeroshot.py \
    --condor \
    --model_type "$1" \
    --mimic3_root /srv/local/data/MIMIC-III/mimic-iii-clinical-database-1.4 \
    --resume
echo "[$(date)] Finished zeroshot_mimic3_rd_mining  model_type=$1"
