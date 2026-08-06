#!/bin/bash
# run_bioclinicalbert_ner_mimic3.sh — Bio_ClinicalBERT NER inference on MIMIC-III under HTCondor
#
# Run AFTER the RareDis training job (bioclinicalbert_ner_raredis.sub) has
# completed and produced a checkpoint at
# results/raredis/bioclinicalbert_ner_trainer/best_hf/.
source /home/johnwu3/miniconda3/etc/profile.d/conda.sh
conda activate rd_pyhealth

export DASK_DISTRIBUTED__DIAGNOSTICS__NVML=False

CHECKPOINT=/home/johnwu3/projects/rare_disease/workspace/results/raredis/bioclinicalbert_ner_trainer/best_hf
MIMIC3_ROOT=/srv/local/data/physionet.org/files/mimic-iii-clinical-database-1.4/

cd /home/johnwu3/projects/rare_disease/workspace/repos/RDMA

echo "[$(date)] Starting run_bioclinicalbert_ner_mimic3"
python baselines/mimic3_rd_mining_text/bioclinicalbert_ner.py \
    --condor \
    --checkpoint_dir "$CHECKPOINT" \
    --mimic3_root "$MIMIC3_ROOT"
echo "[$(date)] Finished run_bioclinicalbert_ner_mimic3"
