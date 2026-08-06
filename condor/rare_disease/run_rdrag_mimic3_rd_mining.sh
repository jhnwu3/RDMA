#!/bin/bash
# run_rdrag_mimic3_rd_mining.sh — runs one RDRAG MIMIC-III rare-disease mining job under HTCondor
# $1 = model_type (e.g. llama3_8b, mistral_24b, llama3_70b, qwen3_32b)
source /home/johnwu3/miniconda3/etc/profile.d/conda.sh
conda activate rd_pyhealth

export DASK_DISTRIBUTED__DIAGNOSTICS__NVML=False

cd /home/johnwu3/projects/rare_disease/workspace/repos/RDMA

mkdir -p /home/johnwu3/projects/rare_disease/workspace/logs/mimic3_rd_mining/rdrag/"$1"

echo "[$(date)] Starting rdrag_mimic3_rd_mining  model_type=$1"
python baselines/mimic3_rd_mining_code/rdrag.py \
    --condor \
    --model_type "$1" \
    --mimic3_root /srv/local/data/jw3/physionet.org/files/mimic-iii-clinical-database-1.4/ \
    --resume
echo "[$(date)] Finished rdrag_mimic3_rd_mining  model_type=$1"
