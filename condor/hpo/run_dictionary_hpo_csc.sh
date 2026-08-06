#!/bin/bash
# run_dictionary_hpo_csc.sh — runs Dictionary HPO CSC baseline under HTCondor
source /home/johnwu3/miniconda3/etc/profile.d/conda.sh
conda activate rd_pyhealth

export DASK_DISTRIBUTED__DIAGNOSTICS__NVML=False

cd /home/johnwu3/projects/rare_disease/workspace/repos/RDMA

mkdir -p /home/johnwu3/projects/rare_disease/workspace/logs/csc/dictionary_hpo

echo "[$(date)] Starting dictionary_hpo_csc"
python baselines/csc/dictionary_hpo.py
echo "[$(date)] Finished dictionary_hpo_csc"
