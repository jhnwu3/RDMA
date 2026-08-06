#!/bin/bash
# run_csc_llm_extractor.sh — ablation: CSC HPO pipeline with simple (LLM) entity extractor
# $1 = model_type (e.g. llama3_8b, mistral_24b, llama3_70b, qwen3_32b)
# $2... = optional extra run_hpo.py flags
source /home/johnwu3/miniconda3/etc/profile.d/conda.sh
conda activate rd_pyhealth

export DASK_DISTRIBUTED__DIAGNOSTICS__NVML=False

cd /home/johnwu3/projects/rare_disease/workspace/repos/RDMA

mkdir -p /home/johnwu3/projects/rare_disease/workspace/logs/csc/hpo_llm_extractor/"$1"

OUTPUT=/home/johnwu3/projects/rare_disease/workspace/results/csc/rdma_simple_extractor/"$1"_predictions.jsonl
mkdir -p /home/johnwu3/projects/rare_disease/workspace/results/csc/rdma_simple_extractor

echo "[$(date)] Starting run_csc_llm_extractor  model_type=$1  output=$OUTPUT  extra_args=${*:2}"
python scripts/csc/run_hpo.py --condor --model_type "$1" --entity_extractor simple --output "$OUTPUT" "${@:2}"
echo "[$(date)] Finished run_csc_llm_extractor  model_type=$1"
