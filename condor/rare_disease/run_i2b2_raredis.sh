#!/bin/bash
# run_i2b2_raredis.sh — Stanza i2b2 NER baseline on RareDis under HTCondor
source /home/johnwu3/miniconda3/etc/profile.d/conda.sh
conda activate rd_pyhealth

export DASK_DISTRIBUTED__DIAGNOSTICS__NVML=False

cd /home/johnwu3/projects/rare_disease/workspace/repos/RDMA

echo "[$(date)] Starting run_i2b2_raredis"
python baselines/raredis/i2b2.py --condor
echo "[$(date)] Finished run_i2b2_raredis"
