#!/bin/bash
# run_eval_rdd.sh — runs one RDD evaluation job under HTCondor
# $1 = model_type (e.g. llama3_8b, mistral_24b, llama3_70b, qwen_32b)
# $2 = approach    (rdma | zeroshot | rdrag)
source /home/johnwu3/miniconda3/etc/profile.d/conda.sh
conda activate rd_pyhealth

export DASK_DISTRIBUTED__DIAGNOSTICS__NVML=False

cd /home/johnwu3/projects/rare_disease/workspace/repos/RDMA

echo "[$(date)] Starting eval_rdd  model_type=$1  approach=$2"
python scripts/rdd/eval.py --model_type "$1" --approach "$2"
echo "[$(date)] Finished eval_rdd  model_type=$1  approach=$2"
