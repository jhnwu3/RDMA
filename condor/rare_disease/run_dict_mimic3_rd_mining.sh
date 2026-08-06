#!/bin/bash
# run_dict_mimic3_rd_mining.sh — runs the dictionary baseline on MIMIC-III under HTCondor (code-eval format)
source /home/johnwu3/miniconda3/etc/profile.d/conda.sh
conda activate rd_pyhealth

export DASK_DISTRIBUTED__DIAGNOSTICS__NVML=False

cd /home/johnwu3/projects/rare_disease/workspace/repos/RDMA

echo "[$(date)] Starting run_dict_mimic3_rd_mining"
python baselines/mimic3_rd_mining_code/dict.py \
    --condor \
    --mimic3_root /srv/local/data/physionet.org/files/mimic-iii-clinical-database-1.4/ \
    --resume
echo "[$(date)] Finished run_dict_mimic3_rd_mining"
