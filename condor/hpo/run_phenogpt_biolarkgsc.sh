#!/bin/bash
# run_phenogpt_biolarkgsc.sh — runs PhenoGPT BioLark GSC baseline under HTCondor
source /home/johnwu3/miniconda3/etc/profile.d/conda.sh
conda activate rd_pyhealth

export DASK_DISTRIBUTED__DIAGNOSTICS__NVML=False

cd /home/johnwu3/projects/rare_disease/workspace/repos/RDMA

mkdir -p /home/johnwu3/projects/rare_disease/workspace/logs/biolarkgsc/phenogpt

echo "[$(date)] Starting phenogpt_biolarkgsc"
python baselines/biolarkgsc/phenogpt.py --condor
echo "[$(date)] Finished phenogpt_biolarkgsc"
