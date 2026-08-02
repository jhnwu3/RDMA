#!/bin/bash
# run_dictionary_hpo_biolarkgsc.sh — runs Dictionary HPO BioLark GSC baseline under HTCondor
source /home/johnwu3/miniconda3/etc/profile.d/conda.sh
conda activate rd_pyhealth

export DASK_DISTRIBUTED__DIAGNOSTICS__NVML=False

cd /home/johnwu3/projects/rare_disease/workspace/repos/RDMA

mkdir -p /home/johnwu3/projects/rare_disease/workspace/logs/biolarkgsc/dictionary_hpo

echo "[$(date)] Starting dictionary_hpo_biolarkgsc"
python baselines/biolarkgsc/dictionary_hpo.py
echo "[$(date)] Finished dictionary_hpo_biolarkgsc"
