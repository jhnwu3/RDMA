#!/bin/bash
# run_bioclinicalbert_ner_raredis.sh — runs the BioClinicalBERT NER baseline on RareDis under HTCondor
source /home/johnwu3/miniconda3/etc/profile.d/conda.sh
conda activate rd_pyhealth

export DASK_DISTRIBUTED__DIAGNOSTICS__NVML=False

cd /home/johnwu3/projects/rare_disease/workspace/repos/RDMA

echo "[$(date)] Starting run_bioclinicalbert_ner_raredis"
python baselines/raredis/bioclinicalbert_ner_trainer.py --condor --tune
echo "[$(date)] Finished run_bioclinicalbert_ner_raredis"
