#!/bin/bash
# run_phenogpt_csc.sh — runs PhenoGPT CSC baseline under HTCondor
source /home/johnwu3/miniconda3/etc/profile.d/conda.sh
conda activate rd_pyhealth

export DASK_DISTRIBUTED__DIAGNOSTICS__NVML=False

cd /home/johnwu3/projects/rare_disease/workspace/repos/RDMA

mkdir -p /home/johnwu3/projects/rare_disease/workspace/logs/csc/phenogpt

echo "[$(date)] Starting phenogpt_csc"
python baselines/csc/phenogpt.py --condor
echo "[$(date)] Finished phenogpt_csc"
