#!/bin/bash
# run_biobert_csc.sh — runs BioBERT-MRC inference on CSC under HTCondor
#
# Run AFTER biobert_biolarkgsc has completed and produced a checkpoint at
# results/biolarkgsc/biobert_mrc/best_hf/.
source /home/johnwu3/miniconda3/etc/profile.d/conda.sh
conda activate rd_pyhealth

export DASK_DISTRIBUTED__DIAGNOSTICS__NVML=False

cd /home/johnwu3/projects/rare_disease/workspace/repos/RDMA

echo "[$(date)] Starting run_biobert_csc"
python baselines/csc/biobert_mrc.py \
	--condor
echo "[$(date)] Finished run_biobert_csc"
