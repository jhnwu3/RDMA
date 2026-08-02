#!/bin/bash
# run_rdrag_raredis.sh — runs one RDRAG RareDis baseline job under HTCondor
# $1 = model_type (e.g. llama3_8b, mistral_24b, llama3_70b, qwen_32b)
source /home/johnwu3/miniconda3/etc/profile.d/conda.sh
conda activate rd_pyhealth

export DASK_DISTRIBUTED__DIAGNOSTICS__NVML=False

cd /home/johnwu3/projects/rare_disease/workspace/repos/RDMA

mkdir -p /home/johnwu3/projects/rare_disease/workspace/logs/raredis/rdrag/"$1"

echo "[$(date)] Starting rdrag_raredis  model_type=$1"
python baselines/raredis/rdrag.py --condor --model_type "$1"
echo "[$(date)] Finished rdrag_raredis  model_type=$1"
