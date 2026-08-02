#!/bin/bash
# run_dataset_stats.sh — compute corpus statistics for all RDMA benchmarks
source /home/johnwu3/miniconda3/etc/profile.d/conda.sh
conda activate rd_pyhealth

export DASK_DISTRIBUTED__DIAGNOSTICS__NVML=False

cd /home/johnwu3/projects/rare_disease/workspace/repos/RDMA

echo "[$(date)] Starting dataset_stats"
python ../../scripts/dataset_stats.py
echo "[$(date)] Finished dataset_stats"
