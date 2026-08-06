#!/bin/bash
# run_bioclinicalbert_ner_csc.sh — runs BioClinicalBERT NER inference on CSC under HTCondor
#
# Run AFTER bioclinicalbert_ner_biolarkgsc has completed and produced a checkpoint at
# results/biolarkgsc/bioclinicalbert_ner/best_hf/.
source /home/johnwu3/miniconda3/etc/profile.d/conda.sh
conda activate rd_pyhealth

export DASK_DISTRIBUTED__DIAGNOSTICS__NVML=False

cd /home/johnwu3/projects/rare_disease/workspace/repos/RDMA

echo "[$(date)] Starting run_bioclinicalbert_ner_csc"
python baselines/csc/bioclinicalbert_ner.py \
	--condor
echo "[$(date)] Finished run_bioclinicalbert_ner_csc"
