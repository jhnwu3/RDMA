#!/bin/bash
# run_dict_rdd.sh — runs the dictionary baseline on RDD under HTCondor
source /home/johnwu3/miniconda3/etc/profile.d/conda.sh
conda activate rd_pyhealth

export DASK_DISTRIBUTED__DIAGNOSTICS__NVML=False

cd /home/johnwu3/projects/rare_disease/workspace/repos/RDMA

echo "[$(date)] Starting run_dict_rdd"
python baselines/rdd/dict.py --condor
echo "[$(date)] Finished run_dict_rdd"
