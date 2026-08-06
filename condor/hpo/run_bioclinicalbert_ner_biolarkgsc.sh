#!/bin/bash
# run_bioclinicalbert_ner_biolarkgsc.sh — runs the BioClinicalBERT NER baseline on BioLarkGSC under HTCondor
source /home/johnwu3/miniconda3/etc/profile.d/conda.sh
conda activate rd_pyhealth

export DASK_DISTRIBUTED__DIAGNOSTICS__NVML=False

cd /home/johnwu3/projects/rare_disease/workspace/repos/RDMA

echo "[$(date)] Starting run_bioclinicalbert_ner_biolarkgsc"
python baselines/biolarkgsc/bioclinicalbert_ner.py \
	--condor \
	--predictions_path /home/johnwu3/projects/rare_disease/workspace/results/biolarkgsc/bioclinicalbert_ner_predictions.jsonl
echo "[$(date)] Finished run_bioclinicalbert_ner_biolarkgsc"
