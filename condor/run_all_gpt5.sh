#!/bin/bash
# run_all_gpt5.sh — runs all 5 benchmarks x 3 approaches sequentially with Azure GPT-5
source /home/johnwu3/miniconda3/etc/profile.d/conda.sh
conda activate rd_pyhealth

export DASK_DISTRIBUTED__DIAGNOSTICS__NVML=False
export AZURE_REQUEST_DELAY=5    # 5s between each API request

cd /home/johnwu3/projects/rare_disease/workspace/repos/RDMA

MIMIC3_ROOT=/srv/local/data/MIMIC-III/mimic-iii-clinical-database-1.4

mkdir -p /home/johnwu3/projects/rare_disease/workspace/logs/gpt5_all

run_step() {
    local desc=$1; shift
    echo "[$(date)] START: $desc"
    if python "$@"; then
        echo "[$(date)] DONE:  $desc"
    else
        echo "[$(date)] FAIL:  $desc (exit $?)"
    fi
}

# ── CSC ────────────────────────────────────────────────────────────────────────
# run_step "csc/zeroshot" \
#     baselines/csc/zeroshot.py \
#     --condor --llm_type azure --model_type gpt-5-john --temperature 1 --resume

# run_step "csc/raghpo" \
#     baselines/csc/raghpo.py \
#     --condor --llm_type azure --model_type gpt-5-john --temperature 1 \
#     --entity_extractor simple --resume

# run_step "csc/pipeline" \
#     scripts/csc/run_hpo.py \
#     --condor --llm_type azure --model_type gpt-5-john --temperature 1 \
#     --extraction_temperature 1 --entity_extractor simple

# ── BioLarkGSC ────────────────────────────────────────────────────────────────
# run_step "biolarkgsc/zeroshot" \
#     baselines/biolarkgsc/zeroshot.py \
#     --condor --llm_type azure --model_type gpt-5-john --temperature 1 --resume

# run_step "biolarkgsc/raghpo" \
#     baselines/biolarkgsc/raghpo.py \
#     --condor --llm_type azure --model_type gpt-5-john --temperature 1 \
#     --entity_extractor simple --resume

# run_step "biolarkgsc/pipeline" \
#     scripts/biolarkgsc/run_hpo.py \
#     --condor --llm_type azure --model_type gpt-5-john --temperature 1 \
#     --extraction_temperature 1 --entity_extractor simple

# ── RareDis ───────────────────────────────────────────────────────────────────
# run_step "raredis/zeroshot" \
#     baselines/raredis/zeroshot.py \
#     --condor --llm_type azure --model_type gpt-5-john --temperature 1 --resume

run_step "raredis/rdrag" \
    baselines/raredis/rdrag.py \
    --condor --llm_type azure --model_type gpt-5-john --temperature 1 --resume

# run_step "raredis/pipeline" \
#     scripts/raredis/run_rdma.py \
#     --condor --llm_type azure --model_type gpt-5-john --temperature 1 \
#     --entity_extractor simple

# ── MIMIC3 (shared inference; text+code eval handled post-hoc) ────────────────
run_step "mimic3/zeroshot" \
    baselines/mimic3_rd_mining_code/zeroshot.py \
    --condor --llm_type azure --model_type gpt-5-john --temperature 1 \
    --mimic3_root "$MIMIC3_ROOT" --resume

run_step "mimic3/rdrag" \
    baselines/mimic3_rd_mining_code/rdrag.py \
    --condor --llm_type azure --model_type gpt-5-john --temperature 1 \
    --mimic3_root "$MIMIC3_ROOT" --resume

# run_step "mimic3/pipeline" \
#     scripts/mimic3_rd_mining_code/run_rdma.py \
#     --condor --llm_type azure --model_type gpt-5-john --temperature 1 \
#     --entity_extractor simple --mimic3_root "$MIMIC3_ROOT"

echo "[$(date)] All runs complete."
