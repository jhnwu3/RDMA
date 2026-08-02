# RDMA — Rare Disease Mining Agents

Agent-driven extraction of rare diseases and phenotypes from clinical text.

📄 **Paper:** [RDMA: Cost Effective Agent-Driven Rare Disease Mining from Electronic Health Records](https://arxiv.org/abs/2507.15867) (arXiv:2507.15867)
John Wu, Adam Cross, Jimeng Sun

RDMA runs a four-agent pipeline over clinical text:

```
  text ──▶ Extractor ──▶ Verifier ──▶ Matcher ──▶ Supervisor ──▶ ORPHA / HPO codes
           find candidate  filter out   ground to    catch and
           mentions        negated,     the ontology correct
                           hypothetical              residual errors
                           and false
                           positives
```

Implementation: `rdma/rd/` for rare diseases (Orphanet), `rdma/hpo/` for
phenotypes (Human Phenotype Ontology). Shared infrastructure — LLM clients,
embedding managers, abbreviation expansion — lives in `rdma/utils/`.

---

## Contents

- [Install](#install)
- [⚠️ Paths you must configure](#️-paths-you-must-configure)
- [API keys](#api-keys)
- [Data](#data)
- [Quick start](#quick-start)
- [Reproducing the benchmark results](#reproducing-the-benchmark-results)
- [Baselines](#baselines)
- [Step-by-step pipeline](#step-by-step-pipeline)
- [Annotation UI](#annotation-ui)
- [Restricted data and rehydration](#restricted-data-and-rehydration)
- [Repository layout](#repository-layout)
- [License and citation](#license-and-citation)

---

## Install

```bash
git clone https://github.com/jhnwu3/RDMA.git
cd RDMA
pip install -r requirements.txt
```

Python 3.10+ is expected. `llama-cpp-python` is optional and only needed for the
`llama_cpp` backend; install it separately since the build is platform-specific.

**Prebuilt embedding stores** (Orphanet, HPO and abbreviation vector stores) are
a separate download — they are far too large for git:

📦 [Download here](https://drive.google.com/file/d/16wpcexHf2KDZ4w2qBHrTp8dn1oa59ABM/view?usp=sharing)

Unzip them anywhere and pass the paths with `--embeddings_file` /
`--abbreviations_file`. If you place them at `data/vector_stores/` and
`data/tools/` inside the repo, the built-in defaults will find them.

You can also build the stores yourself:

```bash
python scripts/create_vector_store_hpo.py    # HPO terms
python scripts/create_vector_store_orpha.py  # Orphanet terms
python scripts/create_vector_store_rd.py     # rare disease names
```

---

## ⚠️ Paths you must configure

**This is the main thing standing between a fresh clone and a working run.**

Scripts under `scripts/` and `baselines/` were written for a single lab machine
and still carry absolute paths from it as *defaults*. The good news: every one of
them is overridable on the command line — you do not need to edit any source.

| Default (hardcoded) | Override with | What it is |
|---|---|---|
| `/home/johnwu3/.../workspace/results` | `--output <path>` | where predictions are written |
| `/shared/rsaas/jw3/rare_disease/model_cache` | `--model_cache_dir <path>` | HuggingFace model cache |
| `/shared/eng/pyhealth/<dataset>` | `--dataset_cache_dir <path>` | PyHealth dataset cache |
| `data/vector_stores/*.npy` | `--embeddings_file <path>` | ontology embedding store |
| `data/tools/abbreviations_*.npy` | `--abbreviations_file <path>`, or `RDMA_ABBREVIATIONS_FILE` | abbreviation store |

So a runnable invocation on a new machine looks like:

```bash
python scripts/raredis/run_rdma.py \
  --model_type qwen_32b --gpu_id 0 \
  --model_cache_dir   ~/.cache/huggingface \
  --dataset_cache_dir ~/.cache/pyhealth/raredis \
  --embeddings_file   /path/to/rd_orpha_medembed.npy \
  --output            ./results/raredis/qwen_32b_predictions.jsonl
```

Two further notes:

- Scripts `sys.path`-insert a hardcoded `_RDMA_ROOT`. **Run them from the repo
  root** (`cd RDMA && python scripts/...`) and it resolves correctly.
- Everything under `condor/` is site-specific and will not run unmodified. See
  [`condor/README.md`](condor/README.md).

The dataset **loaders** need no configuration — they default to the vendored
copies in `public_data/`.

---

## API keys

RDMA runs LLMs locally via HuggingFace or through hosted APIs. Keys are read
from environment variables.

Create a `.env` at the repo root (it is gitignored):

```bash
OPENROUTER_API_KEY=sk-or-...
GROQ_API_KEY=gsk_...
HF_API_KEY=hf_...
ACCESS_TOKEN=hf_...
```

Load it before running:

```bash
export $(grep -v '^#' .env | xargs)
```

| Backend (`--llm_type`) | Needs | Notes |
|---|---|---|
| `local` (default) | GPU, `ACCESS_TOKEN` for gated models | HuggingFace weights |
| `openrouter` | `OPENROUTER_API_KEY` | hundreds of models, many free-tier; no GPU |
| `api` | `GROQ_API_KEY` | Groq |
| `azure` | Azure OpenAI credentials | copy `scripts/azure_openai_config.example.json` to `scripts/azure_openai_config.json` and fill it in |
| `llama_cpp` | local GGUF file | pass `--gguf_file` |

**OpenRouter shortcuts** for `--model_type`: `nemotron-120b`, `llama3-70b`,
`qwen3-235b`, `deepseek-r1`. Raw OpenRouter model IDs also work.

For HuggingFace: create a Read token at **Settings → Access Tokens**, put it in
`.env` as both `HF_API_KEY` and `ACCESS_TOKEN`, and accept the model license on
the model page for any gated model (Llama 3, Qwen, Mistral).

---

## Data

Full provenance, licenses and citations: **[`public_data/README.md`](public_data/README.md)**.

| Benchmark | Ships in repo? | Docs | Notes |
|---|---|---|---|
| BioLark GSC+ | ✅ `public_data/biolarkgsc/` | 228 | HPO phenotype NER over PubMed abstracts |
| CSC | ✅ `public_data/csc/` | 116 | HPO phenotypes over clinical case reports |
| RareDis | ✅ `public_data/raredis/` | 1011 | rare-disease NER over NORD descriptions |
| RDD Corpus | ❌ download separately | — | pass `root=` to `RDDDataset` |
| MIMIC-III RD mining | ⚠️ annotations only | 117 | needs PhysioNet credentials for the notes |
| MIMIC-IV diff. diagnosis | ⚠️ annotations only | 145 | needs PhysioNet credentials for the notes |

The three vendored corpora load with no configuration:

```python
from datasets.biolarkgsc import BioLarkGSCDataset
from datasets.csc import CSCDataset
from datasets.raredis import RareDisDataset

BioLarkGSCDataset().stats()   # 228 documents
```

MIMIC data is **not** redistributed — the files here contain our annotations and
the join keys, but no clinical note text. See
[Restricted data and rehydration](#restricted-data-and-rehydration).

---

## Quick start

`example.ipynb` walks through the pipeline interactively.

From the command line, the smallest end-to-end run is CSC (116 documents):

```bash
# 1. Extract phenotypes
python scripts/csc/run_hpo.py \
  --llm_type openrouter --model_type nemotron-120b \
  --output ./results/csc/nemotron_predictions.jsonl

# 2. Score them
python scripts/csc/eval.py \
  --predictions ./results/csc/nemotron_predictions.jsonl
```

Add `--dev` for a 2-document smoke test and `--debug` for verbose tracing.
All pipeline scripts support `--resume` to continue from a checkpoint.

---

## Reproducing the benchmark results

This is how the tables in the paper were produced. The flow is the same for
every benchmark:

```
run_rdma.py / run_hpo.py        →  results/<bench>/<model>_predictions.jsonl
  or baselines/<bench>/<method>.py

eval.py                         →  results/<bench>/eval_<...>.json

aggregate_rare_disease_eval_matrix.py
                                →  results/full_eval_matrix.{csv,md}

scripts/analysis/bootstrap_ci.py
                                →  results/bootstrap_ci_all.{csv,md}
```

### The six benchmark tracks

| Track | Runner | Evaluator | Metric |
|---|---|---|---|
| `biolarkgsc` | `scripts/biolarkgsc/run_hpo.py` | `scripts/biolarkgsc/eval.py` | HPO-ID P/R/F1, lenient (ancestor-resolved) + strict |
| `csc` | `scripts/csc/run_hpo.py` | `scripts/csc/eval.py` | HPO-ID P/R/F1, lenient + strict |
| `raredis` | `scripts/raredis/run_rdma.py` | `scripts/raredis/eval.py` | micro-F1 over surface forms, exact match then LLM judge |
| `rdd` | `scripts/rdd/run_rdma.py` | `scripts/rdd/eval.py` | micro-F1, LLM judge |
| `mimic3_rd_mining_code` | `scripts/mimic3_rd_mining_code/run_rdma.py` | `scripts/mimic3_rd_mining_code/eval.py` | ORPHA-ID exact-match micro-F1 |
| `mimic3_rd_mining_text` | *(reuses the code track's predictions)* | `scripts/mimic3_rd_mining_text/eval.py` | micro-F1 over surface forms, LLM judge |

The two evaluator families take different arguments:

```bash
# Rare-disease tracks: resolve prediction paths by convention
python scripts/raredis/eval.py --model_type mistral_24b --approach rdma
#   --approach ∈ {rdma, zeroshot, rdrag, dict}

# HPO tracks: point at a predictions file directly
python scripts/csc/eval.py --predictions results/csc/mistral_24b_predictions.jsonl
#   add --inspect --output per_doc.csv for a per-document breakdown
```

The LLM-judge evaluators load a local model (default `mistral_24b`) to adjudicate
non-exact matches, so they need a GPU — pass `--gpu_id`.

### Approaches evaluated

|  | biolarkgsc | csc | raredis | rdd | mimic3 code | mimic3 text |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| RDMA (full) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| RDRAG (no verifier/supervisor) | | | ✅ | ✅ | ✅ | ✅ |
| Zero-shot LLM | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Dictionary (no LLM) | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| RAG-HPO | ✅ | ✅ | | | | |
| FastHPOCR | ✅ | ✅ | | | | |
| PhenoGPT / PhenoGPT2 | ✅ | ✅ | | | | |
| i2b2 (Stanza) | ✅ | ✅ | ✅ | | | ✅ |
| BioBERT-MRC | ✅ | ✅ | ✅ | | | ✅ |
| Bio_ClinicalBERT NER | ✅ | ✅ | ✅ | | | ✅ |

Models reported: `llama3_8b`, `mistral_24b`, `llama3_70b`, `qwen3_32b`,
`nemotron-120b-q4`, and GPT-5 via Azure.

### Aggregating

`scripts/aggregate_rare_disease_eval_matrix.py` reads a TSV manifest listing
every (dataset, track, approach, model) run and collects the results into one
table. The manifests used for the paper are in `condor/rare_disease/`:

```bash
python scripts/aggregate_rare_disease_eval_matrix.py \
  --manifest condor/rare_disease/eval_manifest_all_benchmarks.tsv \
  --run_evals \
  --bootstrap --n_bootstrap 1000
```

`--run_evals` fills in any row that has predictions but no eval output.
`--bootstrap` appends 95% confidence intervals.

Confidence intervals can also be computed standalone — the method (per-document
resampling of TP/FP/FN, 1000 iterations, seed 42) is documented in
[`scripts/analysis/bootstrap.md`](scripts/analysis/bootstrap.md):

```bash
python scripts/analysis/bootstrap_ci.py --all
```

Other analysis helpers in `scripts/analysis/`:
`evaluate_hpo_runs.py` (sweep all HPO prediction files),
`print_approach_comparison.py` (approach comparison table),
`dataset_stats.py` (corpus statistics).

### Running the whole grid

`condor/` holds the exact HTCondor submit files used, one job per
(benchmark × approach × model). They are site-specific — read
[`condor/README.md`](condor/README.md) before submitting. Each job is a thin
wrapper around one of the Python commands above, so Condor is not required.

---

## Baselines

Every baseline is `baselines/<benchmark>/<method>.py` and takes the same core
flags as the main runners (`--llm_type`, `--model_type`, `--gpu_id`, `--output`).

```bash
# Zero-shot LLM
python baselines/raredis/zeroshot.py --llm_type openrouter --model_type nemotron-120b

# RDRAG — extraction + embedding matching, no verifier or supervisor
python baselines/raredis/rdrag.py --model_type mistral_24b --gpu_id 0

# Dictionary matching, no LLM
python baselines/raredis/dict.py --embeddings_file /path/to/rd_orpha_medembed.npy

# HPO-specific baselines
python baselines/csc/raghpo.py     --llm_type openrouter --model_type nemotron-120b
python baselines/csc/fasthpocr.py
python baselines/csc/dictionary_hpo.py
```

The BERT baselines train first, then run inference elsewhere:

```bash
python baselines/raredis/biobert_mrc_trainer.py           # writes a checkpoint
python baselines/mimic3_rd_mining_text/biobert_mrc.py     # reuses it
```

Several baselines wrap external repositories that must be cloned alongside RDMA
— PhenoGPT, PhenoGPT2, FastHPOCR and BioBERT-MRC. Sources and citations for all
of them are in **[`CITATIONS.md`](CITATIONS.md)**.

---

## Step-by-step pipeline

For finer control, `rd_steps/` runs each agent separately.

```bash
# 1 — extract entities with context
python rd_steps/step1_extract_rd_context.py \
  --input_file your_notes.json --output_file step1.json \
  --llm_type openrouter --model_type nemotron-120b \
  --entity_extractor retrieval \
  --embeddings_file /path/to/rd_orpha_medembed.npy --top_k 10

# 2 — verify (negation, hypotheticals, false positives)
python rd_steps/step2_verify_rd_context.py \
  --input_file step1.json --output_file step2.json \
  --llm_type openrouter --model_type nemotron-120b \
  --embeddings_file /path/to/rd_orpha_medembed.npy \
  --verifier_type multi_stage \
  --use_abbreviations --abbreviations_file /path/to/abbreviations_medembed_sm.npy

# 3 — match to ORPHA codes
python rd_steps/step3_match_rd.py \
  --input_file step2.json --output_file step3.json \
  --llm_type openrouter --model_type nemotron-120b \
  --embeddings_file /path/to/rd_orpha_medembed.npy --top_k 5

# 4 — supervisor: evaluate and correct
python rd_steps/step4_supervisor.py \
  --predictions step3.json --ground-truth gold.json \
  --evaluation step3_eval.json --output supervised.json \
  --llm_type openrouter --model_type nemotron-120b \
  --embeddings_file /path/to/rd_orpha_medembed.npy
```

`--entity_extractor` accepts `llm`, `retrieval` (RAG), `iterative`, or `multi`
(multi-temperature ensemble).

---

## Annotation UI

`annotation_tool.html` is a standalone, no-backend annotation interface — open it
directly in a browser.

![Annotation tool](figs/AnnotationToolUI.png)

Upload a predictions JSON with the upload button:

![Upload button](figs/UploadButton.png)

Then step through each entity and mark whether it is a rare disease:

![Annotating](figs/AnnotatingUI.png)

Export a corrections JSON with the green button, top right:

![Export button](figs/ExportButton.png)

**Two warnings:**

- **Do not refresh or close the page.** There is no backend and no autosave —
  refreshing loses all annotations in progress.
- You may see `[Entity '...' occurrence #1 (index 0) not found by string search
  or overlaps with previously used positions ...]`. That means the entity string
  could not be located in the document text.

`public_data/annotation_tool_input.json` is a working example input, but its
`context` panes will be empty until you rehydrate it (below) — the note text was
stripped for the public release.

---

## Restricted data and rehydration

MIMIC-III and MIMIC-IV are distributed by PhysioNet under a Credentialed Health
Data Use Agreement that forbids republishing note text. Our annotation files
originally embedded note excerpts in `context` fields; those have been removed.

**What ships:** all annotation labels, ORPHA/HPO codes, decisions, and the
identifiers needed to join back to the source notes
(`document_id` → `NOTEEVENTS.ROW_ID` for MIMIC-III, `subject_id` for MIMIC-IV).

**What does not:** any clinical note text.

If you hold PhysioNet credentials, rebuild the contexts from your own copy:

```bash
python scripts/data/rehydrate_mimic_text.py \
  --mimic3-root      /path/to/mimic-iii-clinical-database-1.4 \
  --mimic4-note-root /path/to/mimic-iv-note/2.2/note \
  --out rehydrated/
```

The originals stored snippets without character offsets, so this re-locates each
entity in its source note and re-extracts a window. The result is **equivalent
but not byte-identical** — enclosing sentences match, window edges may not. The
script reports any entity it could not relocate.

Before publishing anything derived from `public_data/`, run the leakage gate:

```bash
python scripts/data/check_public_data_leakage.py
```

It exits non-zero if it finds a residual `context` key, an over-long string, or a
MIMIC de-identification marker.

---

## Repository layout

```
rdma/                  core library
  rd/                    rare-disease agents: extractor, verifier, matcher, supervisor
  hpo/                   phenotype agents
  rdrag/  hporag/        earlier RAG layers the agents build on
  utils/                 LLM clients, embeddings, abbreviations, search
datasets/              PyHealth dataset loaders (biolarkgsc, csc, raredis, rdd)
tasks/                 PyHealth task definitions
models/                BioBERT-MRC and Bio_ClinicalBERT NER models
scripts/               per-benchmark runners and evaluators
  <benchmark>/           run_rdma.py or run_hpo.py, plus eval.py
  analysis/              bootstrap CIs, aggregation, corpus statistics
  data/                  MIMIC strip / rehydrate / leakage-check tooling
baselines/<benchmark>/ baseline implementations
rd_steps/              step-by-step CLI version of the pipeline
public_data/           redistributable benchmark data
condor/                HTCondor submit files used for the paper (site-specific)
notebooks/             analysis notebooks
figs/                  figures
```

---

## License and citation

Code is MIT licensed — see [`LICENSE`](LICENSE). **The license covers the source
code only.** Benchmark data under `public_data/` carries its original authors'
licenses; see [`public_data/README.md`](public_data/README.md).

If you use RDMA, please cite the paper — and cite the datasets and baselines you
actually used. Full entries for all of them are in
**[`CITATIONS.md`](CITATIONS.md)**.

```bibtex
@article{wu2025rdma,
  title  = {RDMA: Cost Effective Agent-Driven Rare Disease Mining from
            Electronic Health Records},
  author = {Wu, John and Cross, Adam and Sun, Jimeng},
  journal = {arXiv preprint arXiv:2507.15867},
  year   = {2025},
  url    = {https://arxiv.org/abs/2507.15867}
}
```
