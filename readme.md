<div align="center">

# RDMA — Rare Disease Mining Agents

**Agent-driven extraction of rare diseases and phenotypes from clinical text.**

[![Paper](https://img.shields.io/badge/arXiv-2507.15867-b31b1b.svg)](https://arxiv.org/abs/2507.15867)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

*John Wu · Adam Cross · Jimeng Sun*

</div>

---

RDMA maps free-text clinical notes to structured **ORPHA** (Orphanet) and **HPO**
(Human Phenotype Ontology) codes. It runs four agents in sequence, using a small
quantized LLM plus ontology retrieval instead of a fine-tuned model:

```
                ┌───────────┐   ┌──────────┐   ┌─────────┐   ┌────────────┐
  clinical  ──▶ │ Extractor │──▶│ Verifier │──▶│ Matcher │──▶│ Supervisor │──▶  ORPHA / HPO
    text        └───────────┘   └──────────┘   └─────────┘   └────────────┘         codes
                 find candidate  drop negated,  ground to     audit and flag
                 mentions per    hypothetical   the ontology  disagreements
                 sentence        & non-disease  via retrieval for human review
```

**The headline result:** a 4-bit quantized **Mistral 24B** beats Llama 3.3 70B and
GPT-5 on 4 of 5 benchmarks, at roughly a tenth of the inference cost. You do not
need a frontier model, and you do not need to fine-tune anything.

---

## Table of contents

**Start here**
- [Choose your path](#choose-your-path)
- [Installation](#installation)

**Using RDMA on your own notes**
- [Quick start](#quick-start)
- [Recommended setup](#recommended-setup)
- [LLM backends and API keys](#llm-backends-and-api-keys)

**Reproducing the paper**
- [Benchmarks](#benchmarks)
- [Results](#results)
- [Reproduction guide](#reproduction-guide)
- [Baselines](#baselines)

**Reference**
- [Configuration](#configuration)
- [Step-by-step pipeline](#step-by-step-pipeline)
- [Annotation UI](#annotation-ui)
- [Restricted data and rehydration](#restricted-data-and-rehydration)
- [Repository layout](#repository-layout)
- [License and citation](#license-and-citation)

---

## Choose your path

|  | 🏥 **Applied** — run RDMA on your notes | 🎓 **Academic** — reproduce the paper |
|---|---|---|
| **Install** | `pip install -e .` | `pip install -e ".[benchmarks]"` |
| **You need** | 1× 24 GB GPU (RTX 3090 / A5000) | same, plus benchmark data (ships in repo) |
| **Model** | Mistral 24B, 4-bit — our recommendation | 5 models × 6 approaches × 6 tracks |
| **Read** | [Quick start](#quick-start) → [Recommended setup](#recommended-setup) | [Benchmarks](#benchmarks) → [Results](#results) → [Reproduction guide](#reproduction-guide) |

---

## Installation

```bash
git clone https://github.com/jhnwu3/RDMA.git
cd RDMA
pip install -e .
```

Python 3.10+. This installs the `rdma` library and the recommended quantized
backend (`transformers` + `accelerate` + `bitsandbytes`).

| Extra | Command | Adds |
|---|---|---|
| *(none)* | `pip install -e .` | the RDMA pipeline |
| `benchmarks` | `pip install -e ".[benchmarks]"` | PyHealth loaders, tasks, evaluators |
| `baselines` | `pip install -e ".[baselines]"` | Stanza, NLTK, Optuna, PEFT for baseline methods |
| `llamacpp` | `pip install -e ".[llamacpp]"` | GGUF backend (build is platform-specific) |
| `all` | `pip install -e ".[all]"` | everything |

<details>
<summary><b>Why only <code>rdma</code> gets installed</b></summary>

`pip install -e .` puts the `rdma` package on your path and nothing else. The
benchmark harness (`datasets/`, `tasks/`, `models/`, `scripts/`, `baselines/`) is
intentionally **not** packaged — `datasets/` would otherwise shadow HuggingFace
`datasets` for every project in the same environment.

Those directories are still importable when you run from the repository root,
which is how the reproduction scripts are invoked anyway. So: `import rdma`
works anywhere; `from datasets.csc import CSCDataset` requires `cd RDMA` first.

</details>

### Vector stores (required)

RDMA retrieves ontology candidates from prebuilt FAISS stores. They are too
large for git:

📦 **[Download (Orphanet + HPO + abbreviations)](https://drive.google.com/file/d/16wpcexHf2KDZ4w2qBHrTp8dn1oa59ABM/view?usp=sharing)**

Unzip to `data/vector_stores/` and `data/tools/` inside the repo and the
defaults will find them. Anywhere else, pass `--embeddings_file` /
`--abbreviations_file`.

Or build them yourself (needs the HPO and Orphanet source files):

```bash
python scripts/create_vector_store_hpo.py     # HPO terms
python scripts/create_vector_store_orpha.py   # Orphanet terms
python scripts/create_vector_store_rd.py      # rare disease names
```

---

## Quick start

The smallest end-to-end run — 116 clinical case reports, ships with the repo:

```bash
# Extract phenotypes, then score them
python scripts/csc/run_hpo.py --model_type mistral_24b --gpu_id 0 \
    --output ./results/csc/mistral_24b_predictions.jsonl

python scripts/csc/eval.py \
    --predictions ./results/csc/mistral_24b_predictions.jsonl
```

Expected: **F1 ≈ 0.657** (P 0.644 / R 0.671), matching the paper.

Add `--dev` for a 2-document smoke test, `--debug` for verbose tracing, and
`--resume` to continue from a checkpoint. `example.ipynb` walks through the same
pipeline interactively.

### On your own notes

```python
from rdma.utils.llm_client import LocalLLMClient
from rdma.rd.extractor import LLMRDExtractor

client = LocalLLMClient(model_type="mistral_24b", device="cuda:0")
extractor = LLMRDExtractor(llm_client=client)
entities = extractor.extract_entities("Patient presents with Marfan syndrome.")
```

For batch processing, `rd_steps/` runs each agent as a separate CLI stage over a
JSON file of notes — see [Step-by-step pipeline](#step-by-step-pipeline).

---

## Recommended setup

The paper's central practical finding is that **bigger models do not help**.
RDMA F1 by backbone, across every benchmark:

| Backbone | Params | BioLark | CSC | RareDis | MIMIC3 Entity | MIMIC3 Code |
|---|---:|---:|---:|---:|---:|---:|
| Llama 3 8B | 8B | 0.331 | 0.515 | 0.595 | 0.073 | 0.097 |
| **Mistral Small 24B** ⭐ | 24B (4-bit) | **0.559** | **0.657** | 0.814 | **0.592** | **0.526** |
| Qwen3 32B | 32B (4-bit) | 0.473 | 0.583 | 0.828 | 0.528 | 0.467 |
| Llama 3.3 70B | 70B (4-bit) | 0.484 | 0.589 | **0.845** | 0.442 | 0.439 |
| Nemotron 120B | 120B (4-bit) | — | — | 0.604 | 0.070 | 0.070 |
| GPT-5 (Azure) | — | 0.545 | 0.646 | 0.780 | 0.513 | 0.465 |

Mistral 24B wins **4 of 5** benchmarks — including both MIMIC-III tracks, the
only ones on real clinical notes — and loses RareDis to Llama 3.3 70B by 3
points. Below 24B, models stop following structured output reliably; above it,
gains plateau.

**So we recommend:**

```bash
--model_type mistral_24b --gpu_id 0
```

→ `mistralai/Mistral-Small-24B-Instruct-2501`, 4-bit NF4 via bitsandbytes,
quantized automatically on first load and cached.

### Hardware and throughput

| | RAG-HPO | RDMA |
|---|---:|---:|
| Extraction | 15 min | 68 min |
| Verification | — | 29 min |
| Matching | 24 min | 24 min |
| **Total** | **39 min** | **121 min** |

Measured on the 116-document CSC benchmark (32,260 words) on a single **RTX
3090** — about **278 words/minute**. RDMA's extra verification and implication
steps cost ~3× the runtime of plain RAG for a 6–18 point F1 gain.

| Tier | GPU | Approx. cost | Runs |
|---|---|---:|---|
| Low | 1× RTX 3090 (24 GB) | $2,200 | **Mistral 24B 4-bit — recommended** |
| Medium | 1× A6000 (48 GB) | $6,520 | Llama 3.3 70B 4-bit |
| High | 4× A6000 | $38,500 | Llama 3.3 70B unquantized |

Workstation prices as of April 2025. The 10× cost gap between the Low and High
tiers buys no accuracy on 4 of 5 benchmarks.

---

## LLM backends and API keys

Create a `.env` at the repo root (gitignored):

```bash
OPENROUTER_API_KEY=sk-or-...
GROQ_API_KEY=gsk_...
HF_API_KEY=hf_...
ACCESS_TOKEN=hf_...
```

Load with `export $(grep -v '^#' .env | xargs)`.

| `--llm_type` | Requires | Notes |
|---|---|---|
| `local` *(default)* | GPU; `ACCESS_TOKEN` for gated weights | HuggingFace, 4-bit by default |
| `openrouter` | `OPENROUTER_API_KEY` | no GPU needed; many free-tier models |
| `api` | `GROQ_API_KEY` | Groq |
| `azure` | `scripts/azure_openai_config.json` | copy from `.example.json` and fill in |
| `llama_cpp` | a local GGUF file | pass `--gguf_file` |

OpenRouter `--model_type` shortcuts: `nemotron-120b`, `llama3-70b`,
`qwen3-235b`, `deepseek-r1`. Raw OpenRouter IDs work too.

For gated HuggingFace models (Llama 3, Mistral, Qwen), create a Read token under
**Settings → Access Tokens** and accept the license on the model page.

---

## Benchmarks

Full provenance, licenses and citations: **[`public_data/README.md`](public_data/README.md)**.

| Benchmark | Task | Docs | Entities | Words | Avg/doc | In repo? |
|---|---|---:|---:|---:|---:|:-:|
| **BioLark GSC+** | HPO phenotypes, PubMed abstracts | 228 | 2,773 | 33,942 | 149 | ✅ |
| **CSC** | HPO phenotypes, case reports | 116 | 1,813 | 32,260 | 278 | ✅ |
| **RareDis** | Rare-disease NER, NORD descriptions | 1,011 | 5,221 | 157,679 | 156 | ✅ |
| **MIMIC3-RD** | Rare diseases, real clinical notes | 117 | 176 | 221,980 | 1,897 | ⚠️ annotations only |
| **RDD Corpus** | Rare-disease NER | 684 | — | — | — | ❌ download separately |

MIMIC3-RD is evaluated on two tracks: **Entity** (did you find the right
mention?) and **Code** (did you map it to the right ORPHA code?). RareDis splits
are train 711 / dev 97 / test 203; BioLark GSC+ uses a random 80-10-10 split for
the fine-tuned baselines, and the full corpus for everything else.

The three vendored corpora load with zero configuration:

```python
from datasets.biolarkgsc import BioLarkGSCDataset   # 228 docs
from datasets.csc import CSCDataset                 # 116 docs
from datasets.raredis import RareDisDataset         # 1011 docs

BioLarkGSCDataset().stats()
```

> **MIMIC-III/IV note text is not redistributed.** PhysioNet's Credentialed
> Health Data Use Agreement forbids it. We ship our annotations plus the join
> keys; see [Restricted data and rehydration](#restricted-data-and-rehydration).

---

## Results

Values below are **as published in the paper**. Re-running the
[reproduction guide](#reproduction-guide) reproduces them to within ±0.005 F1;
the [backbone table](#recommended-setup) above reports the raw eval-file numbers.
**Bold** = best per column, <u>underline</u> = second.

### Phenotype mining (HPO codes)

| Approach | Fine-tuned? | BioLark P | R | F1 | CSC P | R | F1 |
|---|:-:|---:|---:|---:|---:|---:|---:|
| Dictionary match | ✗ | 0.682 | 0.214 | 0.326 | 0.600 | 0.210 | 0.310 |
| FastHPOCR | ✗ | 0.721 | 0.518 | **0.603** | 0.520 | 0.450 | 0.480 |
| i2b2 Clinical BERT | ✓ | 0.599 | 0.417 | 0.491 | 0.480 | 0.600 | 0.530 |
| BioBERT | ✓ | 0.635 | 0.409 | 0.498 | 0.614 | 0.278 | 0.382 |
| Bio_ClinicalBERT | ✓ | 0.459 | 0.424 | 0.441 | 0.449 | 0.348 | 0.392 |
| PhenoGPT | ✓ | 0.676 | 0.312 | 0.427 | 0.570 | 0.390 | 0.460 |
| Zero-shot (Llama 3.3 70B) | ✗ | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| RAG-HPO (Llama 3.3 70B) | ✗ | 0.410 | 0.563 | 0.475 | 0.674 | 0.580 | <u>0.624</u> |
| **RDMA (Mistral 24B)** | ✗ | 0.565 | 0.553 | <u>0.559</u> | 0.644 | 0.671 | **0.657** |

### Rare disease mining (ORPHA codes)

| Approach | FT? | RareDis F1 | MIMIC3 Entity F1 | MIMIC3 Code F1 |
|---|:-:|---:|---:|---:|
| Dictionary match | ✗ | 0.550 | <u>0.431</u> | <u>0.360</u> |
| i2b2 Clinical BERT | ✓ | 0.260 | 0.020 | 0.019 |
| BioBERT | ✓ | <u>0.716</u> | 0.033 | 0.047 |
| Bio_ClinicalBERT | ✓ | 0.673 | 0.027 | 0.050 |
| Zero-shot (Llama 3.3 70B) | ✗ | 0.700 | 0.228 | 0.002 |
| RAG-RD (Llama 3.3 70B) | ✗ | 0.230 | 0.085 | 0.030 |
| **RDMA (Mistral 24B)** | ✗ | **0.810** | **0.592** | **0.530** |

Three things to read out of these tables:

1. **Fine-tuned encoders do not transfer.** BioBERT scores 0.716 on RareDis and
   0.033 on MIMIC3 Entity — the same task on real clinical notes. RDMA needs no
   training and holds up across both.
2. **LLMs hallucinate ontology codes.** Zero-shot Llama 3.3 70B gets 0.700 on
   RareDis (surface spans) but 0.002 on MIMIC3 Code. It finds the text and
   invents the identifier. Retrieval is what closes that gap.
3. **Verification is what separates RDMA from RAG.** Same backbone, same
   retrieval; adding the verifier moves RareDis from 0.230 → 0.810, almost
   entirely through precision.

Confidence intervals use a per-document bootstrap (1000 iterations, seed 42);
stage ④ of the [reproduction guide](#reproduction-guide) writes them to
`results/bootstrap_ci_all.md`. The method is documented in
[`scripts/analysis/bootstrap.md`](scripts/analysis/bootstrap.md).

### Agent-assisted annotation

We used RDMA to clean the noisy MIMIC-III rare-disease annotations, flagging only
contentious cases for expert review rather than asking a clinician to re-read
everything.

| | Initial | Human only | RDMA + Human |
|---|---:|---:|---:|
| Documents | 117 | 117 | 117 |
| Annotations re-reviewed | — | 333 | **122** |
| Unique rare diseases | 192 | 120 | 135 |

**63% less review burden**, and it recovered 15 valid rare diseases the
human-only pass had dropped. Agreement with the fully-human reference:

| | RDMA alone | RDMA + Human |
|---|---:|---:|
| Cohen's κ | 0.46 | **0.81** |
| F1 | 0.74 | **0.94** |
| Precision | 0.92 | 0.92 |
| Recall | 0.62 | **0.96** |

RDMA alone is not accurate enough to replace an annotator (κ = 0.46). What it
does well is *triage* — deciding which 122 of 333 annotations a human should
look at.

---

## Reproduction guide

Every benchmark follows the same four stages:

```
  ①  run_rdma.py / run_hpo.py        ──▶  results/<bench>/<model>_predictions.jsonl
      or baselines/<bench>/<method>.py

  ②  eval.py                         ──▶  results/<bench>/eval_<...>.json

  ③  aggregate_rare_disease_eval_matrix.py
                                     ──▶  results/full_eval_matrix.{csv,md}

  ④  scripts/analysis/bootstrap_ci.py
                                     ──▶  results/bootstrap_ci_all.{csv,md}
```

### The six tracks

| Track | Runner | Evaluator | Metric |
|---|---|---|---|
| `biolarkgsc` | `scripts/biolarkgsc/run_hpo.py` | `scripts/biolarkgsc/eval.py` | HPO-ID P/R/F1, lenient (ancestor-resolved) + strict |
| `csc` | `scripts/csc/run_hpo.py` | `scripts/csc/eval.py` | HPO-ID P/R/F1, lenient + strict |
| `raredis` | `scripts/raredis/run_rdma.py` | `scripts/raredis/eval.py` | micro-F1 over spans, exact then LLM judge |
| `rdd` | `scripts/rdd/run_rdma.py` | `scripts/rdd/eval.py` | micro-F1, LLM judge |
| `mimic3_rd_mining_code` | `scripts/mimic3_rd_mining_code/run_rdma.py` | `scripts/mimic3_rd_mining_code/eval.py` | ORPHA-ID exact-match micro-F1 |
| `mimic3_rd_mining_text` | *(reuses code-track predictions)* | `scripts/mimic3_rd_mining_text/eval.py` | micro-F1 over spans, LLM judge |

Metrics are micro-averaged: TP/FP/FN are summed across documents, then P/R/F1
computed once over the totals.

### The two evaluator conventions

```bash
# Rare-disease tracks — resolve prediction paths by convention
python scripts/raredis/eval.py --model_type mistral_24b --approach rdma
#   --approach ∈ {rdma, zeroshot, rdrag, dict}

# HPO tracks — point at a predictions file directly
python scripts/csc/eval.py --predictions results/csc/mistral_24b_predictions.jsonl
#   --inspect --output per_doc.csv  for a per-document breakdown
```

LLM-judge evaluators load a local model (default `mistral_24b`) to adjudicate
non-exact matches, so they need a GPU — pass `--gpu_id`.

> **Note on file naming.** HPO tracks write RDMA results as
> `eval_<model>.json` and RAG-HPO as `eval_<model>_raghpo.json`, while
> rare-disease tracks use `eval_<approach>_<model>.json`. The aggregate matrix
> expects the latter, so HPO rows show as `missing_eval` in the generated
> `full_eval_matrix.md`. Use `scripts/analysis/evaluate_hpo_runs.py` for those
> two tracks.

### The full grid

| | biolarkgsc | csc | raredis | rdd | mimic3 code | mimic3 text |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| RDMA (full) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| RDRAG (no verifier) | | | ✅ | ✅ | ✅ | ✅ |
| Zero-shot LLM | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Dictionary (no LLM) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| RAG-HPO | ✅ | ✅ | | | | |
| FastHPOCR | ✅ | ✅ | | | | |
| PhenoGPT | ✅ | ✅ | | | | |
| i2b2 (Stanza) | ✅ | ✅ | ✅ | | ✅ | ✅ |
| BioBERT-MRC | ✅ | ✅ | ✅ | | ✅ | ✅ |
| Bio_ClinicalBERT | ✅ | ✅ | ✅ | | ✅ | ✅ |

Backbones: `llama3_8b`, `mistral_24b`, `qwen3_32b`, `llama3_70b`,
`nemotron-120b-q4`, and GPT-5 via Azure. All local models ≥24B run 4-bit NF4.

### Aggregating

```bash
python scripts/aggregate_rare_disease_eval_matrix.py \
    --manifest condor/rare_disease/eval_manifest_all_benchmarks.tsv \
    --run_evals --bootstrap --n_bootstrap 1000

python scripts/analysis/bootstrap_ci.py --all
```

`--run_evals` fills in any row with predictions but no eval output;
`--bootstrap` appends 95% CIs. Other helpers in `scripts/analysis/`:
`evaluate_hpo_runs.py`, `print_approach_comparison.py`, `dataset_stats.py`.

`condor/` holds the exact HTCondor submit files used for the paper, one job per
(benchmark × approach × model). They are **site-specific** — read
[`condor/README.md`](condor/README.md) first. Each is a thin wrapper around the
Python commands above, so Condor is not required.

---

## Baselines

Every baseline lives at `baselines/<benchmark>/<method>.py` and accepts the same
core flags (`--llm_type`, `--model_type`, `--gpu_id`, `--output`).

```bash
# LLM baselines
python baselines/raredis/zeroshot.py --model_type mistral_24b --gpu_id 0
python baselines/raredis/rdrag.py    --model_type mistral_24b --gpu_id 0

# No-LLM baselines
python baselines/raredis/dict.py --embeddings_file /path/to/rd_orpha_medembed.npy
python baselines/csc/fasthpocr.py
python baselines/csc/dictionary_hpo.py

# HPO-specific
python baselines/csc/raghpo.py --model_type mistral_24b --gpu_id 0
```

The BERT baselines train first, then run inference elsewhere:

```bash
python baselines/raredis/biobert_mrc_trainer.py        # writes a checkpoint
python baselines/mimic3_rd_mining_text/biobert_mrc.py  # reuses it
```

`pip install -e ".[baselines]"` covers the Python dependencies. Four baselines
additionally wrap **external repositories** that must be cloned alongside RDMA:
PhenoGPT, PhenoGPT2, FastHPOCR and BioBERT-MRC. Sources, licenses and citations
for all of them are in **[`CITATIONS.md`](CITATIONS.md)**.

---

## Configuration

Scripts under `scripts/` and `baselines/` carry absolute paths from our lab
machine as **defaults**. Every one is overridable on the command line — you never
need to edit source.

| Default | Override with | What it is |
|---|---|---|
| `/home/johnwu3/.../workspace/results` | `--output` | where predictions are written |
| `/shared/rsaas/jw3/.../model_cache` | `--model_cache_dir` | HuggingFace model cache |
| `/shared/eng/pyhealth/<dataset>` | `--dataset_cache_dir` | PyHealth dataset cache |
| `data/vector_stores/*.npy` | `--embeddings_file` | ontology embedding store |
| `data/tools/abbreviations_*.npy` | `--abbreviations_file` or `$RDMA_ABBREVIATIONS_FILE` | abbreviation store |

A fully-specified invocation on a new machine:

```bash
python scripts/raredis/run_rdma.py \
    --model_type mistral_24b --gpu_id 0 \
    --model_cache_dir   ~/.cache/huggingface \
    --dataset_cache_dir ~/.cache/pyhealth/raredis \
    --embeddings_file   /path/to/rd_orpha_medembed.npy \
    --output            ./results/raredis/mistral_24b_predictions.jsonl
```

**Run scripts from the repository root.** They `sys.path`-insert a hardcoded
repo root, and the dataset loaders resolve `public_data/` relative to it.

---

## Step-by-step pipeline

`rd_steps/` exposes each agent as its own CLI stage, for finer control or to
swap a single component.

```bash
# 1 — extract entities with surrounding context
python rd_steps/step1_extract_rd_context.py \
    --input_file notes.json --output_file step1.json \
    --model_type mistral_24b --entity_extractor retrieval \
    --embeddings_file /path/to/rd_orpha_medembed.npy --top_k 10

# 2 — verify: negation, hypotheticals, non-diseases
python rd_steps/step2_verify_rd_context.py \
    --input_file step1.json --output_file step2.json \
    --model_type mistral_24b --verifier_type multi_stage \
    --embeddings_file /path/to/rd_orpha_medembed.npy \
    --use_abbreviations --abbreviations_file /path/to/abbreviations_medembed_sm.npy

# 3 — match to ORPHA codes
python rd_steps/step3_match_rd.py \
    --input_file step2.json --output_file step3.json \
    --model_type mistral_24b \
    --embeddings_file /path/to/rd_orpha_medembed.npy --top_k 5

# 4 — supervisor: audit and flag for review
python rd_steps/step4_supervisor.py \
    --predictions step3.json --ground-truth gold.json \
    --evaluation step3_eval.json --output supervised.json \
    --model_type mistral_24b \
    --embeddings_file /path/to/rd_orpha_medembed.npy
```

`--entity_extractor` accepts `llm`, `retrieval` (RAG), `iterative`, or `multi`
(multi-temperature ensemble). `--verifier_type multi_stage` is what the paper
uses.

---

## Annotation UI

`annotation_tool.html` is a standalone, no-backend annotation interface — open it
directly in a browser. It surfaces retrieved candidates, context and prior
ORPHA codes, and is what produced the 63% review reduction above.

<div align="center">
<img src="figs/AnnotationToolUI.png" width="80%" alt="Annotation tool">
</div>

1. Upload a predictions JSON with the upload button.
2. Step through each entity and mark whether it is a rare disease.
3. Export corrections with the green button, top right.

> ⚠️ **Do not refresh or close the page.** There is no backend and no autosave —
> refreshing loses everything in progress.

`public_data/annotation_tool_input.json` is a working example, but its context
panes stay empty until you [rehydrate](#restricted-data-and-rehydration) it.

If you see `[Entity '...' occurrence #1 (index 0) not found by string search...]`,
the entity string could not be located in the document text.

---

## Restricted data and rehydration

MIMIC-III and MIMIC-IV are distributed by PhysioNet under a Credentialed Health
Data Use Agreement that forbids republishing note text.

**Ships:** annotation labels, ORPHA/HPO codes, review decisions, and the join
keys (`document_id` → `NOTEEVENTS.ROW_ID` for MIMIC-III, `subject_id` for
MIMIC-IV). **Does not ship:** any clinical note text.

With PhysioNet credentials, rebuild the contexts from your own copy:

```bash
python scripts/data/rehydrate_mimic_text.py \
    --mimic3-root      /path/to/mimic-iii-clinical-database-1.4 \
    --mimic4-note-root /path/to/mimic-iv-note/2.2/note \
    --out rehydrated/
```

The originals stored snippets without character offsets, so this re-locates each
entity in its source note and re-extracts a window. Results are **equivalent but
not byte-identical** — enclosing sentences match, window edges may not. The
script reports anything it could not relocate.

Before publishing anything derived from `public_data/`:

```bash
python scripts/data/check_public_data_leakage.py
```

Exits non-zero on a residual `context` key, an over-long string, or a MIMIC
de-identification marker.

---

## Repository layout

```
rdma/                    core library (the pip-installable package)
  rd/                      rare-disease agents: extractor, verifier, matcher, supervisor
  hpo/                     phenotype agents
  rdrag/  hporag/          RAG layers the agents build on
  utils/                   LLM clients, embeddings, abbreviations, search
datasets/                PyHealth dataset loaders
tasks/                   PyHealth task definitions
models/                  BioBERT-MRC and Bio_ClinicalBERT NER
scripts/
  <benchmark>/             run_rdma.py or run_hpo.py, plus eval.py
  analysis/                bootstrap CIs, aggregation, corpus statistics
  data/                    MIMIC strip / rehydrate / leakage-check tooling
baselines/<benchmark>/   baseline implementations
rd_steps/                step-by-step CLI version of the pipeline
public_data/             redistributable benchmark data
condor/                  HTCondor submit files used for the paper (site-specific)
notebooks/               analysis notebooks
```

---

## License and citation

Code is **MIT** licensed — see [`LICENSE`](LICENSE). *The license covers source
code only.* Benchmark data under `public_data/` carries its original authors'
licenses; see [`public_data/README.md`](public_data/README.md).

If you use RDMA, please cite the paper — and cite the datasets and baselines you
actually used. Full entries for every one are in **[`CITATIONS.md`](CITATIONS.md)**.

```bibtex
@article{wu2025rdma,
  title   = {RDMA: Cost Effective Agent-Driven Rare Disease Mining from
             Electronic Health Records},
  author  = {Wu, John and Cross, Adam and Sun, Jimeng},
  journal = {arXiv preprint arXiv:2507.15867},
  year    = {2025},
  url     = {https://arxiv.org/abs/2507.15867}
}
```
