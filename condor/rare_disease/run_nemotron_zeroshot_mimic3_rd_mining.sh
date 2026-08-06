#!/bin/bash
# run_nemotron_zeroshot_mimic3_rd_mining.sh — runs zero-shot MIMIC-III baseline with nemotron-120b-q4 via llama_cpp
source /home/johnwu3/miniconda3/etc/profile.d/conda.sh
conda activate rd_pyhealth

export DASK_DISTRIBUTED__DIAGNOSTICS__NVML=False

cd /home/johnwu3/projects/rare_disease/workspace/repos/RDMA

echo "[$(date)] Starting nemotron_zeroshot_mimic3_rd_mining (llama_cpp / nemotron-120b-q4)"
python baselines/mimic3_rd_mining_code/zeroshot.py \
    --condor \
    --llm_type llama_cpp \
    --model_type nemotron-120b-q4 \
    --mimic3_root /srv/local/data/jw3/physionet.org/files/mimic-iii-clinical-database-1.4 \
    --resume
echo "[$(date)] Finished nemotron_zeroshot_mimic3_rd_mining"
