#!/bin/bash
# run_i2b2_mimic3.sh — Stanza i2b2 rare-disease baseline on MIMIC-III under HTCondor
source /home/johnwu3/miniconda3/etc/profile.d/conda.sh
conda activate rd_pyhealth

export DASK_DISTRIBUTED__DIAGNOSTICS__NVML=False

MIMIC3_ROOT=/srv/local/data/physionet.org/files/mimic-iii-clinical-database-1.4/

cd /home/johnwu3/projects/rare_disease/workspace/repos/RDMA

echo "[$(date)] Starting run_i2b2_mimic3"
python baselines/mimic3_rd_mining_text/i2b2.py \
    --condor \
    --mimic3_root "$MIMIC3_ROOT"
echo "[$(date)] Finished run_i2b2_mimic3"
