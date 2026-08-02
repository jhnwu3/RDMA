#!/bin/bash
# run_i2b2_biolarkgsc.sh — runs i2b2 BioLark GSC baseline under HTCondor
source /home/johnwu3/miniconda3/etc/profile.d/conda.sh
conda activate rd_pyhealth

export DASK_DISTRIBUTED__DIAGNOSTICS__NVML=False

cd /home/johnwu3/projects/rare_disease/workspace/repos/RDMA

mkdir -p /home/johnwu3/projects/rare_disease/workspace/logs/biolarkgsc/i2b2

echo "[$(date)] Starting i2b2_biolarkgsc"
python baselines/biolarkgsc/i2b2.py --condor --stanza_device cuda --top_k 50 --retriever sentence_transformer --retriever_model abhinand/MedEmbed-small-v0.1
echo "[$(date)] Finished i2b2_biolarkgsc"
