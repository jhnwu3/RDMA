#!/bin/bash
# run_raghpo_biolarkgsc.sh — runs one RAG-HPO BioLark GSC baseline job under HTCondor
# $1 = model_type (e.g. llama3_8b, mistral_24b, llama3_70b, qwen3_32b)
source /home/johnwu3/miniconda3/etc/profile.d/conda.sh
conda activate rd_pyhealth

export DASK_DISTRIBUTED__DIAGNOSTICS__NVML=False

cd /home/johnwu3/projects/rare_disease/workspace/repos/RDMA

mkdir -p /home/johnwu3/projects/rare_disease/workspace/logs/biolarkgsc/raghpo/"$1"

echo "[$(date)] Starting raghpo_biolarkgsc  model_type=$1"
python baselines/biolarkgsc/raghpo.py --condor --model_type "$1"
echo "[$(date)] Finished raghpo_biolarkgsc  model_type=$1"
