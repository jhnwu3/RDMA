#!/bin/bash
# run_fasthpocr_biolarkgsc.sh — runs FastHPOCR BioLark GSC baseline under HTCondor
source /home/johnwu3/miniconda3/etc/profile.d/conda.sh
conda activate rd_pyhealth

export DASK_DISTRIBUTED__DIAGNOSTICS__NVML=False

cd /home/johnwu3/projects/rare_disease/workspace/repos/RDMA

mkdir -p /home/johnwu3/projects/rare_disease/workspace/logs/biolarkgsc/fasthpocr

echo "[$(date)] Starting fasthpocr_biolarkgsc"
python baselines/biolarkgsc/fasthpocr.py
echo "[$(date)] Finished fasthpocr_biolarkgsc"
