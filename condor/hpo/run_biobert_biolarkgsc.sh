#!/bin/bash
# run_biobert_biolarkgsc.sh — runs the BioBERT-MRC baseline on BioLarkGSC under HTCondor
source /home/johnwu3/miniconda3/etc/profile.d/conda.sh
conda activate rd_pyhealth

export DASK_DISTRIBUTED__DIAGNOSTICS__NVML=False

cd /home/johnwu3/projects/rare_disease/workspace/repos/RDMA

echo "[$(date)] Starting run_biobert_biolarkgsc"
python baselines/biolarkgsc/biobert_mrc.py \
	--condor \
	--predictions_path /home/johnwu3/projects/rare_disease/workspace/results/biolarkgsc/biobert_mrc_predictions.jsonl
echo "[$(date)] Finished run_biobert_biolarkgsc"
