#!/bin/bash
# run_biolarkgsc.sh — runs one BioLark GSC HPO pipeline job under HTCondor
# $1 = model_type (e.g. llama3_8b, mistral_24b, llama3_70b, qwen3_32b)
# $2... = optional extra run_hpo.py flags
source /home/johnwu3/miniconda3/etc/profile.d/conda.sh
conda activate rd_pyhealth

export DASK_DISTRIBUTED__DIAGNOSTICS__NVML=False

cd /home/johnwu3/projects/rare_disease/workspace/repos/RDMA

mkdir -p /home/johnwu3/projects/rare_disease/workspace/logs/biolarkgsc/hpo/"$1"

echo "[$(date)] Starting run_biolarkgsc  model_type=$1"
python scripts/biolarkgsc/run_hpo.py --condor --model_type "$1" "${@:2}"
echo "[$(date)] Finished run_biolarkgsc  model_type=$1"
