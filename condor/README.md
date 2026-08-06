# Condor Job Layout

These are the exact HTCondor submit files used to produce the benchmark tables
reported in the paper. They are included for provenance and as a template.

> **These files are site-specific and will not run unmodified.** Every `.sub`
> and `run_*.sh` file contains absolute paths from the original machine
> (`/home/johnwu3/projects/rare_disease/workspace/...`) and pins
> `Requirements = (Machine == "sunlab-c01/c02.cs.illinois.edu")`. Before
> submitting anywhere else you must rewrite, at minimum:
>
> | What | Where |
> |---|---|
> | `executable = /home/johnwu3/.../run_*.sh` | every `.sub` |
> | `log` / `output` / `error` paths | every `.sub` |
> | `Requirements = (Machine == ...)` | every `.sub` |
> | `cd /home/johnwu3/.../repos/RDMA` and `conda activate rd_pyhealth` | every `run_*.sh` |
> | `predictions_file` / `eval_output` columns | `rare_disease/eval_manifest_*.tsv` |
>
> A quick starting point:
>
> ```bash
> grep -rl '/home/johnwu3' condor/ \
>   | xargs sed -i "s|/home/johnwu3/projects/rare_disease/workspace|$PWD/..|g"
> ```
>
> You do **not** need Condor to reproduce anything. Every job is a thin wrapper
> around one Python entrypoint; see the "Reproducing the benchmark results"
> section of the top-level `readme.md` for the equivalent single-machine
> commands.

## Structure

- `condor/hpo/`: HPO extraction and phenotype-mining baselines (CSC and BioLarkGSC)
- `condor/rare_disease/`: Rare disease benchmark pipelines (RareDis, RDD, MIMIC3 RD mining, and related eval/model variants)
- `condor/rare_disease/eval_manifest_*.tsv`: the ledger of which
  (dataset, track, approach, model) runs exist, consumed by
  `scripts/aggregate_rare_disease_eval_matrix.py`

## Quick Start

Run all commands from the workspace directory that contains `repos/RDMA`:

```bash
cd /path/to/your/workspace
```

Submit all HPO jobs:

```bash
find condor/hpo -maxdepth 1 -type f -name "*.sub" | sort | xargs -I{} condor_submit {}
```

Submit all rare disease jobs:

```bash
find condor/rare_disease -maxdepth 1 -type f -name "*.sub" | sort | xargs -I{} condor_submit {}
```

Submit everything:

```bash
find condor/hpo condor/rare_disease -maxdepth 1 -type f -name "*.sub" | sort | xargs -I{} condor_submit {}
```

## HPO Jobs (`condor/hpo`)

Submit all HPO jobs:

```bash
find condor/hpo -maxdepth 1 -type f -name "*.sub" | sort | xargs -I{} condor_submit {}
```

Individual submit files:

- `condor/hpo/biolarkgsc.sub`
- `condor/hpo/csc.sub`
- `condor/hpo/raghpo_biolarkgsc.sub`
- `condor/hpo/raghpo_csc.sub`
- `condor/hpo/fasthpocr_biolarkgsc.sub`
- `condor/hpo/fasthpocr_csc.sub`
- `condor/hpo/phenogpt_biolarkgsc.sub`
- `condor/hpo/phenogpt_csc.sub`
- `condor/hpo/dictionary_hpo_biolarkgsc.sub`
- `condor/hpo/dictionary_hpo_csc.sub`
- `condor/hpo/i2b2_biolarkgsc.sub`
- `condor/hpo/i2b2_csc.sub`

## Rare Disease Jobs (`condor/rare_disease`)

Submit all rare disease jobs:

```bash
find condor/rare_disease -maxdepth 1 -type f -name "*.sub" | sort | xargs -I{} condor_submit {}
```

Individual submit files:

- `condor/rare_disease/raredis.sub`
- `condor/rare_disease/rdd.sub`
- `condor/rare_disease/mimic3_rd_mining.sub`
- `condor/rare_disease/rdrag_raredis.sub`
- `condor/rare_disease/rdrag_rdd.sub`
- `condor/rare_disease/rdrag_mimic3_rd_mining.sub`
- `condor/rare_disease/zeroshot_raredis.sub`
- `condor/rare_disease/zeroshot_rdd.sub`
- `condor/rare_disease/zeroshot_mimic3_rd_mining.sub`
- `condor/rare_disease/dict_raredis.sub`
- `condor/rare_disease/dict_rdd.sub`
- `condor/rare_disease/dict_mimic3_rd_mining.sub`
- `condor/rare_disease/biobert_raredis.sub`
- `condor/rare_disease/biobert_mimic3.sub`
- `condor/rare_disease/bioclinicalbert_ner_raredis.sub`
- `condor/rare_disease/bioclinicalbert_ner_mimic3.sub`
- `condor/rare_disease/nemotron.sub`
- `condor/rare_disease/qwen3_122b_raredis.sub`
- `condor/rare_disease/eval_raredis.sub`
- `condor/rare_disease/eval_rdd.sub`
- `condor/rare_disease/eval_mimic3_rd_mining.sub`
- `condor/rare_disease/eval_mimic3_rd_mining_text.sub`
- `condor/rare_disease/eval_raredis_qwen3_32b_rdma.sub`
- `condor/rare_disease/eval_rdd_qwen3_32b_rdma.sub`

## Notes

- All submit files now point to runner scripts inside their domain folder (`condor/hpo` or `condor/rare_disease`).
- Runner scripts still execute from `repos/RDMA`, so existing Python entrypoints and log paths remain unchanged.
- To inspect queue status after submission:

```bash
condor_q
```
