#!/bin/bash
# run_fasthpocr_csc.sh — runs FastHPOCR CSC baseline under HTCondor
source /home/johnwu3/miniconda3/etc/profile.d/conda.sh
conda activate rd_pyhealth

export DASK_DISTRIBUTED__DIAGNOSTICS__NVML=False

cd /home/johnwu3/projects/rare_disease/workspace/repos/RDMA

mkdir -p /home/johnwu3/projects/rare_disease/workspace/logs/csc/fasthpocr

echo "[$(date)] Starting fasthpocr_csc"
python baselines/csc/fasthpocr.py
echo "[$(date)] Finished fasthpocr_csc"
