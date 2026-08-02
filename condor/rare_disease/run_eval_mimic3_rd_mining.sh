#!/bin/bash
# run_eval_mimic3_rd_mining.sh — runs one MIMIC-III rare-disease mining evaluation job under HTCondor
# $1 = model_type (e.g. llama3_8b, mistral_24b, llama3_70b, qwen3_32b)
# $2 = approach    (rdma | zeroshot | rdrag | dict | biobert_mrc | bioclinicalbert_ner | i2b2_rd)
source /home/johnwu3/miniconda3/etc/profile.d/conda.sh
conda activate rd_pyhealth

export DASK_DISTRIBUTED__DIAGNOSTICS__NVML=False

MODEL=$1
APPROACH=$2
LOG_BASE=/home/johnwu3/projects/rare_disease/workspace/logs/mimic3_rd_mining
mkdir -p "${LOG_BASE}/eval_${APPROACH}/${MODEL}" "${LOG_BASE}/bert"

cd /home/johnwu3/projects/rare_disease/workspace/repos/RDMA

echo "[$(date)] Starting eval_mimic3_rd_mining  model_type=${MODEL}  approach=${APPROACH}"
python scripts/mimic3_rd_mining_code/eval.py --model_type "${MODEL}" --approach "${APPROACH}"
echo "[$(date)] Finished eval_mimic3_rd_mining  model_type=${MODEL}  approach=${APPROACH}"
