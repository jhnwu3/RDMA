#!/bin/bash
# run_biobert_mimic3.sh — BioBERT-MRC inference on MIMIC-III under HTCondor
#
# Set MIMIC3_ROOT to your MIMIC-III v1.4 path before submitting.
# CHECKPOINT is the best_hf/ directory produced by the RareDis training job.
source /home/johnwu3/miniconda3/etc/profile.d/conda.sh
conda activate rd_pyhealth

export DASK_DISTRIBUTED__DIAGNOSTICS__NVML=False

CHECKPOINT=/home/johnwu3/projects/rare_disease/workspace/results/raredis/biobert_mrc_trainer/best_hf
MIMIC3_ROOT=/srv/local/data/physionet.org/files/mimic-iii-clinical-database-1.4/

cd /home/johnwu3/projects/rare_disease/workspace/repos/RDMA

echo "[$(date)] Starting run_biobert_mimic3"
python baselines/mimic3_rd_mining_text/biobert_mrc.py \
    --condor \
    --checkpoint_dir "$CHECKPOINT" \
    --mimic3_root "$MIMIC3_ROOT"
echo "[$(date)] Finished run_biobert_mimic3"
