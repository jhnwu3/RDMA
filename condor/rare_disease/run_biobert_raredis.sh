#!/bin/bash
# run_biobert_raredis.sh — runs the BioBERT-MRC baseline on RareDis under HTCondor
source /home/johnwu3/miniconda3/etc/profile.d/conda.sh
conda activate rd_pyhealth

export DASK_DISTRIBUTED__DIAGNOSTICS__NVML=False

cd /home/johnwu3/projects/rare_disease/workspace/repos/RDMA

echo "[$(date)] Starting run_biobert_raredis"
python baselines/raredis/biobert_mrc_trainer.py --condor --tune
echo "[$(date)] Finished run_biobert_raredis"
