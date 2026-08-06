#!/bin/bash
# run_qwen3_122b_raredis.sh — runs RDMA RareDis pipeline with qwen3-122b-q4ks via llama_cpp
source /home/johnwu3/miniconda3/etc/profile.d/conda.sh
conda activate rd_pyhealth

export DASK_DISTRIBUTED__DIAGNOSTICS__NVML=False

cd /home/johnwu3/projects/rare_disease/workspace/repos/RDMA

echo "[$(date)] Starting qwen3_122b_raredis (llama_cpp / qwen3-122b-q4ks)"
python scripts/raredis/run_rdma.py \
    --condor \
    --llm_type llama_cpp \
    --model_type qwen3-122b-q4ks \
    --resume
echo "[$(date)] Finished qwen3_122b_raredis"
