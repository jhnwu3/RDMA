# Bootstrap Confidence Intervals — Algorithm & Parameters

## What it does

Estimates the stability of a micro-F1 score without rerunning any experiments.
Instead of new predictions, it resamples the per-document evaluation results
already on disk and recomputes F1 across 1000 random resamples to produce a
95% confidence interval.

---

## Algorithm

### 1. Collect per-document counts

For each evaluated run, load `(tp, fp, fn)` per document from one of two sources:

| Source | Benchmarks | Format |
|--------|-----------|--------|
| Audit JSON (`eval_*.json`) | mimic3, raredis, rdd | `"documents": [{"tp": int, "fp": int, "fn": int, ...}]` |
| JSONL prediction file (inline eval) | csc, biolarkgsc (all variant dirs) | HPO ID set matching evaluated on the fly |

For audit JSONs written by the mimic3/raredis/rdd eval scripts, documents may
store lists instead of integers (`matched_pairs`, `fp`, `fn` as lists).
The script handles both formats transparently via `_doc_counts()`.

### 2. Point estimate

Compute micro-F1 from the full (non-resampled) document set:

```
TP = Σ tp_i    FP = Σ fp_i    FN = Σ fn_i

P  = TP / (TP + FP)      (0 if denominator = 0)
R  = TP / (TP + FN)      (0 if denominator = 0)
F1 = 2PR / (P + R)       (0 if denominator = 0)
```

This matches exactly what the eval script reported.

### 3. Bootstrap resampling

Repeat `n_bootstrap` times:

1. Draw N documents **with replacement** from the N documents in the run
   (using `random.Random(seed).choices`).
2. Sum `tp`, `fp`, `fn` over the resampled N documents.
3. Compute micro-F1 from those totals.

This yields a distribution of `n_bootstrap` F1 values.

### 4. Confidence interval

Sort the bootstrap F1 samples. Take the percentile bounds:

```
alpha      = (100 - ci) / 2          # e.g. 2.5 for 95% CI
lo_index   = max(0, floor(alpha/100 * n_bootstrap) - 1)
hi_index   = min(n_bootstrap - 1, floor((1 - alpha/100) * n_bootstrap))
CI         = [samples[lo_index], samples[hi_index]]
```

Default: 95% CI (alpha = 2.5), so bounds are approximately the 2.5th and
97.5th percentiles of the bootstrap distribution.

---

## Why document-level resampling

The natural unit of variance is the clinical note or disease case — some are
much harder than others. Resampling at the document level captures that
real-world heterogeneity. Resampling at the entity or token level would
underestimate uncertainty by assuming all mentions are statistically independent.

---

## Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--n_bootstrap` | `1000` | Number of bootstrap iterations |
| `--seed` | `42` | RNG seed for reproducibility |
| `--ci` | `95.0` | Confidence level (%) |
| `--min_docs` | `20` | Drop runs with fewer than this many documents (filters debug/partial runs) |

---

## Input modes (combinable)

| Flag | What it reads |
|------|--------------|
| `--eval_json PATH` | Single audit JSON |
| `--glob PATTERN` | All audit JSONs matching a glob |
| `--manifest PATH` | TSV manifest: `dataset track approach model_type predictions_file eval_output` |
| `--hpo_results_root PATH` | Scan JSONL prediction files and evaluate HPO ID matching inline |
| `--all` | Default manifest + default HPO results root (shorthand for most runs) |

Pass `--write_audit_jsons` with `--hpo_results_root` / `--all` to persist
`eval_*.json` files next to each JSONL for faster future runs.

---

## Output columns (CSV / Markdown)

| Column | Description |
|--------|-------------|
| `dataset` | Benchmark name (mimic3, raredis, rdd, csc, biolarkgsc, ...) |
| `track` | Eval track (code, text, na, hpo) |
| `approach` | Method name (rdma, rdrag, zeroshot, dict, biobert_mrc, ...) |
| `model_type` | Backbone model (gpt-5-john, llama3_8b, ...; empty for HPO baselines) |
| `f1` | Point F1 from full document set |
| `f1_ci_lower` | Bootstrap CI lower bound |
| `f1_ci_upper` | Bootstrap CI upper bound |
| `precision` | Micro precision |
| `recall` | Micro recall |
| `tp / fp / fn` | Aggregate counts |
| `n_docs` | Number of documents evaluated |
| `n_bootstrap` | Bootstrap iterations used |
| `eval_json` | Path to audit JSON (or JSONL for HPO inline-eval rows) |

---

## Dataset document counts (approximate minimums for full runs)

| Dataset | Expected n_docs |
|---------|----------------|
| biolarkgsc | ≥ 228 |
| csc | ≥ 100 |
| mimic3 | ≥ 66 |
| raredis | ≥ 1000 |
| rdd | ≥ 300 |

Use `--min_docs` to exclude anything below these thresholds automatically.

---

## Typical invocation

```bash
# Everything at once (recommended)
python scripts/bootstrap_ci.py \
    --all \
    --write_audit_jsons \
    --csv_out results/bootstrap_ci_all.csv \
    --md_out  results/bootstrap_ci_all.md

# Single file spot-check
python scripts/bootstrap_ci.py \
    --eval_json results/raredis/eval_rdrag_gpt-5-john.json

# Stricter run-size filter
python scripts/bootstrap_ci.py --all --min_docs 60 \
    --csv_out results/bootstrap_ci_filtered.csv
```
