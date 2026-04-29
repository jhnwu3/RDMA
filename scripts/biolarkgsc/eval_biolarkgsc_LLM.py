#!/usr/bin/env python3
"""
LLM-based FP/FN root-cause analysis for BioLark GSC predictions.

For each document with at least one FP or FN, calls a strong frontier LLM
(Azure, OpenRouter, or Groq API) with the analysis_prompt.md system prompt
and asks it to categorize each error into one of 5 FP or 5 FN categories.
Writes per-doc results to a JSONL file and prints an aggregate category
summary at the end.

Usage (from RDMA repo root):
  python scripts/biolarkgsc/eval_biolarkgsc_LLM.py \\
      --llm_type azure --model_type gpt-5-john --resume
"""

import argparse
import json
import pickle
import re
import sys
import time
import traceback
from collections import Counter
from datetime import datetime
from pathlib import Path

_RDMA_ROOT = Path("/home/johnwu3/projects/rare_disease/workspace/repos/RDMA")
_RESULTS_DIR = Path("/home/johnwu3/projects/rare_disease/workspace/results")
sys.path.insert(0, str(_RDMA_ROOT))

from rdma.utils.llm_client import APILLMClient, OpenRouterLLMClient, AzureOpenAILLMClient  # noqa: E402
from datasets.biolarkgsc import BioLarkGSCDataset  # noqa: E402
from tasks.biolarkgsc import BioLarkGSCNER  # noqa: E402
from scripts.biolarkgsc.eval import (  # noqa: E402
    _build_hpo_lookup,
    _normalize_hpo,
    _resolve,
    _truncate_text,
)

_DEFAULT_PREDICTIONS = _RESULTS_DIR / "biolarkgsc" / "mistral_24b_predictions.jsonl"
_DEFAULT_ONTOLOGY = _RDMA_ROOT / "data" / "ontology" / "hpo_data_with_lineage.json"
_DEFAULT_DATASET_CACHE_DIR = "/shared/eng/pyhealth/biolarkgsc"
_DEFAULT_PROMPT_FILE = Path(__file__).parent / "analysis_prompt.md"

_FP_CATS = ["FP-A", "FP-B", "FP-C", "FP-D", "FP-E"]
_FN_CATS = ["FN-A", "FN-B", "FN-C", "FN-D", "FN-E"]

_FP_LABELS = {
    "FP-A": "etiology/mechanism term",
    "FP-B": "inheritance mode (annotation inconsistency)",
    "FP-C": "hallucination / retrieval contamination",
    "FP-D": "correct concept, wrong granularity",
    "FP-E": "annotation noise / legitimate ambiguity",
}
_FN_LABELS = {
    "FN-A": "named syndrome not decomposed",
    "FN-B": "extractor missed explicit span",
    "FN-C": "matcher granularity mismatch",
    "FN-D": "obsolete or unmapped gold ID",
    "FN-E": "annotation noise / legitimate omission",
}


def ts(msg: str) -> None:
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S} - {msg}", flush=True)


def load_done_ids(path: Path) -> set:
    done = set()
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        done.add(json.loads(line)["doc_id"])
                    except Exception:
                        pass
    return done


def _fmt_terms(terms: list) -> str:
    return "\n".join(f"  {hp}  {label}" for hp, label in terms) or "  (none)"


def build_user_message(doc_id: str, text: str, tp_terms, fp_terms, fn_terms,
                       predicted_terms, gold_terms, max_text_chars: int) -> str:
    text_snippet = _truncate_text(text, max_text_chars)
    lines = [
        f"DOC_ID: {doc_id}",
        "",
        "TEXT:",
        text_snippet,
        "",
        "PREDICTED (system output):",
        _fmt_terms(predicted_terms),
        "",
        "GOLD STANDARD:",
        _fmt_terms(gold_terms),
        "",
        "TPs (correct predictions):",
        _fmt_terms(tp_terms),
        "",
        "FPs (predicted but not in gold):",
        _fmt_terms(fp_terms),
        "",
        "FNs (in gold but not predicted):",
        _fmt_terms(fn_terms),
    ]
    return "\n".join(lines)


def extract_json(response: str) -> dict:
    """Try to extract a JSON object from the LLM response."""
    response = response.strip()
    # Strip markdown fences
    response = re.sub(r"^```(?:json)?\s*", "", response)
    response = re.sub(r"\s*```$", "", response)
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass
    # Fallback: find the outermost {...}
    match = re.search(r"\{.*\}", response, re.DOTALL)
    if match:
        return json.loads(match.group())
    raise ValueError("No JSON object found in response")


def per_doc_metrics(predicted: set, gold: set) -> dict:
    tp = len(predicted & gold)
    fp = len(predicted - gold)
    fn = len(gold - predicted)
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4),
            "n_tp": tp, "n_fp": fp, "n_fn": fn}


def print_summary(output_path: Path) -> str:
    fp_counts: Counter = Counter()
    fn_counts: Counter = Counter()
    parse_errors = 0
    n_docs = 0

    with open(output_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("error"):
                parse_errors += 1
                continue
            n_docs += 1
            for item in rec.get("fp_analysis", []):
                cat = item.get("category", "unknown")
                fp_counts[cat] += 1
            for item in rec.get("fn_analysis", []):
                cat = item.get("category", "unknown")
                fn_counts[cat] += 1

    total_fp = sum(fp_counts.values())
    total_fn = sum(fn_counts.values())

    lines = [
        "── FP category distribution ──────────────────────────────────",
    ]
    for cat in _FP_CATS:
        n = fp_counts.get(cat, 0)
        pct = 100 * n / total_fp if total_fp else 0.0
        lines.append(f"  {cat} ({_FP_LABELS[cat]}): {n}  ({pct:.1f}%)")
    lines.append(f"  Total FPs analyzed: {total_fp}")
    lines.append("")
    lines.append("── FN category distribution ──────────────────────────────────")
    for cat in _FN_CATS:
        n = fn_counts.get(cat, 0)
        pct = 100 * n / total_fn if total_fn else 0.0
        lines.append(f"  {cat} ({_FN_LABELS[cat]}): {n}  ({pct:.1f}%)")
    lines.append(f"  Total FNs analyzed: {total_fn}")
    lines.append("")
    lines.append(f"  Docs analyzed: {n_docs}")
    lines.append(f"  Docs with parse errors: {parse_errors}")

    summary = "\n".join(lines)
    print(summary, flush=True)

    sidecar = output_path.with_suffix("").with_suffix("") if output_path.suffixes else output_path
    sidecar = output_path.parent / (output_path.stem + "_summary.txt")
    sidecar.write_text(summary + "\n", encoding="utf-8")
    ts(f"Summary written → {sidecar}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="LLM-based FP/FN analysis for BioLark GSC")
    parser.add_argument("--predictions", type=Path, default=_DEFAULT_PREDICTIONS)
    parser.add_argument("--llm_type", type=str, default="azure",
                        choices=["api", "openrouter", "azure"])
    parser.add_argument("--model_type", type=str, default="gpt-5-john")
    parser.add_argument("--api_config", type=str, default=None)
    parser.add_argument("--dotenv_path", type=str, default=None)
    parser.add_argument("--dataset_cache_dir", type=str, default=_DEFAULT_DATASET_CACHE_DIR)
    parser.add_argument("--ontology", type=Path, default=_DEFAULT_ONTOLOGY)
    parser.add_argument("--prompt_file", type=Path, default=_DEFAULT_PROMPT_FILE)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--only_errors", action=argparse.BooleanOptionalAction, default=True,
                        help="Only analyze docs with at least one FP or FN (default: True)")
    parser.add_argument("--max_text_chars", type=int, default=2000)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--sleep_between", type=float, default=1.0,
                        help="Seconds to sleep between API calls (default: 1.0)")
    parser.add_argument("--dev", action="store_true")
    parser.add_argument("--dev_n", type=int, default=3)
    args = parser.parse_args()

    stem = args.predictions.stem
    output = args.output or (_RESULTS_DIR / "biolarkgsc" / f"{stem}_llm_analysis.jsonl")

    ts(f"Predictions  : {args.predictions}")
    ts(f"LLM type     : {args.llm_type} / {args.model_type}")
    ts(f"Output       : {output}")
    ts(f"Resume       : {args.resume}")
    ts(f"Only errors  : {args.only_errors}")
    ts(f"Dev mode     : {args.dev} (n={args.dev_n})")

    # ── Load resources ────────────────────────────────────────────────
    ts("Loading HPO ontology labels…")
    hpo_lookup = _build_hpo_lookup(args.ontology)
    ts(f"  {len(hpo_lookup)} terms loaded")

    ts("Loading system prompt…")
    system_prompt = args.prompt_file.read_text(encoding="utf-8")

    ts("Loading predictions…")
    records = []
    with open(args.predictions, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    ts(f"  {len(records)} prediction records")

    ts("Loading dataset text…")
    dataset = BioLarkGSCDataset(cache_dir=args.dataset_cache_dir)
    samples = dataset.set_task(BioLarkGSCNER())
    text_lookup = {s["patient_id"]: pickle.loads(s["text"]) for s in samples}
    ts(f"  {len(text_lookup)} texts loaded")

    # ── LLM client ────────────────────────────────────────────────────
    ts(f"Building LLM client ({args.llm_type} / {args.model_type})…")
    if args.llm_type == "api":
        llm = (APILLMClient.from_config(args.api_config) if args.api_config
               else APILLMClient(model_type=args.model_type, temperature=args.temperature))
    elif args.llm_type == "openrouter":
        llm = (OpenRouterLLMClient.from_config(args.api_config) if args.api_config
               else OpenRouterLLMClient(model_type=args.model_type, temperature=args.temperature))
    else:  # azure
        llm = (AzureOpenAILLMClient.from_config(args.api_config) if args.api_config
               else AzureOpenAILLMClient(
                   model_type=args.model_type,
                   azure_deployment=args.model_type,
                   temperature=args.temperature,
                   dotenv_path=args.dotenv_path,
               ))

    # ── Resume ────────────────────────────────────────────────────────
    done_ids = load_done_ids(output) if args.resume else set()
    if args.resume:
        ts(f"Resuming — {len(done_ids)} already done")

    output.parent.mkdir(parents=True, exist_ok=True)

    run_records = records[:args.dev_n] if args.dev else records
    analyzed = 0
    skipped = 0

    with open(output, "a" if args.resume else "w", encoding="utf-8") as out_f:
        for i, rec in enumerate(run_records):
            doc_id = rec.get("id", f"doc_{i}")

            if doc_id in done_ids:
                skipped += 1
                continue

            predicted = set(_normalize_hpo(h) for h in rec.get("predicted", []) if h)
            gold = set(_normalize_hpo(h) for h in rec.get("ground_truth", []) if h)
            tp_ids = predicted & gold
            fp_ids = predicted - gold
            fn_ids = gold - predicted

            if args.only_errors and not fp_ids and not fn_ids:
                skipped += 1
                continue

            text = text_lookup.get(doc_id, "")
            metrics = per_doc_metrics(predicted, gold)

            tp_terms = _resolve(tp_ids, hpo_lookup)
            fp_terms = _resolve(fp_ids, hpo_lookup)
            fn_terms = _resolve(fn_ids, hpo_lookup)
            predicted_terms = _resolve(predicted, hpo_lookup)
            gold_terms = _resolve(gold, hpo_lookup)

            user_msg = build_user_message(
                doc_id, text, tp_terms, fp_terms, fn_terms,
                predicted_terms, gold_terms, args.max_text_chars,
            )

            try:
                response = llm.query(user_input=user_msg, system_message=system_prompt)
                parsed = extract_json(response)
                result = {
                    "doc_id": doc_id,
                    **metrics,
                    "fp_analysis": parsed.get("fp_analysis", []),
                    "fn_analysis": parsed.get("fn_analysis", []),
                    "overall_notes": parsed.get("overall_notes", None),
                }
                ts(f"  [{doc_id}] FP={metrics['n_fp']} FN={metrics['n_fn']} F1={metrics['f1']:.4f} ✓")
            except Exception as e:
                ts(f"  [{doc_id}] ERROR: {e}")
                traceback.print_exc()
                result = {
                    "doc_id": doc_id,
                    **metrics,
                    "error": "parse_failed",
                    "raw_response": str(locals().get("response", "")),
                    "exception": str(e),
                }

            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
            out_f.flush()
            analyzed += 1

            if args.sleep_between > 0:
                time.sleep(args.sleep_between)

    ts(f"Done — analyzed {analyzed} docs, skipped {skipped}")
    ts("── Aggregate summary ─────────────────────────────────────────")
    print_summary(output)
    ts(f"Results → {output}")


if __name__ == "__main__":
    main()
