#!/bin/bash
# run_nemotron_rdrag_raredis.sh — runs RDRAG RareDis baseline with nemotron-120b-q4 via llama_cpp
source /home/johnwu3/miniconda3/etc/profile.d/conda.sh
conda activate rd_pyhealth

export DASK_DISTRIBUTED__DIAGNOSTICS__NVML=False

cd /home/johnwu3/projects/rare_disease/workspace/repos/RDMA

echo "[$(date)] Starting nemotron_rdrag_raredis (llama_cpp / nemotron-120b-q4)"
python baselines/raredis/rdrag.py \
    --condor \
    --llm_type llama_cpp \
    --model_type nemotron-120b-q4 \
    --resume
echo "[$(date)] Finished nemotron_rdrag_raredis"
