#!/bin/bash
# run_csc.sh — runs one CSC HPO pipeline job under HTCondor
# $1 = model_type (e.g. llama3_8b, mistral_24b, llama3_70b, qwen3_32b)
# $2... = optional extra run_hpo.py flags
#        (e.g. --debug --dev --dev_n N --output /abs/path.jsonl)
source /home/johnwu3/miniconda3/etc/profile.d/conda.sh
conda activate rd_pyhealth

export DASK_DISTRIBUTED__DIAGNOSTICS__NVML=False

cd /home/johnwu3/projects/rare_disease/workspace/repos/RDMA

mkdir -p /home/johnwu3/projects/rare_disease/workspace/logs/csc/hpo/"$1"

echo "[$(date)] Starting run_csc  model_type=$1  extra_args=${*:2}"
python scripts/csc/run_hpo.py --condor --model_type "$1" "${@:2}"
echo "[$(date)] Finished run_csc  model_type=$1"
