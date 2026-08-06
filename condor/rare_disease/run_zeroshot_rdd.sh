#!/bin/bash
# run_zeroshot_rdd.sh — runs one zero-shot RDD baseline job under HTCondor
# $1 = model_type (e.g. llama3_8b, mistral_24b, llama3_70b)
source /home/johnwu3/miniconda3/etc/profile.d/conda.sh
conda activate rd_pyhealth

export DASK_DISTRIBUTED__DIAGNOSTICS__NVML=False

cd /home/johnwu3/projects/rare_disease/workspace/repos/RDMA

echo "[$(date)] Starting zeroshot_rdd  model_type=$1"
python baselines/rdd/zeroshot.py --condor --model_type "$1"
echo "[$(date)] Finished zeroshot_rdd  model_type=$1"
