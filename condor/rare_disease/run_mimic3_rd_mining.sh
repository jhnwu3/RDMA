#!/bin/bash
# run_mimic3_rd_mining.sh — runs one MIMIC-III rare-disease mining job under HTCondor
# $1 = model_type (e.g. llama3_8b, mistral_24b, llama3_70b, qwen3_32b)
source /home/johnwu3/miniconda3/etc/profile.d/conda.sh
conda activate rd_pyhealth

export DASK_DISTRIBUTED__DIAGNOSTICS__NVML=False

cd /home/johnwu3/projects/rare_disease/workspace/repos/RDMA

echo "[$(date)] Starting run_mimic3_rd_mining  model_type=$1"
python scripts/mimic3_rd_mining_code/run_rdma.py \
    --condor \
    --model_type "$1" \
    --mimic3_root /srv/local/data/MIMIC-III/mimic-iii-clinical-database-1.4 \
    --resume
echo "[$(date)] Finished run_mimic3_rd_mining  model_type=$1"
