#!/usr/bin/env python3
"""
Evaluate RDMA predictions on the RareDis benchmark.

Set-based micro F1: per document, duplicates in predicted/gold are ignored.
Exact string matches (case-insensitive, stripped) are handled directly.
Non-exact pairs are evaluated by an LLM for semantic equivalence.

Writes an audit JSON to EVAL_OUTPUT with per-document matched pairs,
false positives, and false negatives for inspection.
"""

import argparse
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from tqdm import tqdm

_RDMA_ROOT = Path("/home/johnwu3/projects/rare_disease/workspace/repos/RDMA")
sys.path.insert(0, str(_RDMA_ROOT))

from rdma.utils.llm_client import LocalLLMClient  # noqa: E402
from rdma.utils.setup import setup_device  # noqa: E402
from datasets.raredis import RareDisDataset  # noqa: E402
from tasks.raredis import RareDisNER  # noqa: E402

# ── CONFIG ────────────────────────────────────────────────────────────────────

_RESULTS_DIR = Path("/home/johnwu3/projects/rare_disease/workspace/results")
MODEL_CACHE_DIR = "/shared/rsaas/jw3/rare_disease/model_cache"
TEMPERATURE = 0.01

# ─────────────────────────────────────────────────────────────────────────────


def ts(msg):
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S} - {msg}", flush=True)


def load_predictions(path):
    preds = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line)
                preds[obj["id"]] = obj["predicted"]
    return preds


def normalize(s):
    return s.strip().lower()


_MATCH_SYSTEM = (
    "You are a biomedical expert. "
    "Your response must be exactly one word: YES or NO. "
    "Do not include any other text, punctuation, or explanation."
)


def llm_match_pairs(llm_client, pairs):
    """
    Evaluate (pred, gold) string pairs with the LLM one at a time.
    Returns a dict mapping (pred, gold) -> bool.
    """
    results = {}
    for pred, gold in tqdm(pairs, desc="LLM matching"):
        prompt = (
            "Do these two terms refer to the same rare disease or condition?\n"
            f"Term A: {pred}\n"
            f"Term B: {gold}\n"
            "Answer YES if they are the same disease (including different names, "
            "abbreviations, or minor spelling variants). Answer NO otherwise."
        )
        resp = llm_client.query(prompt, _MATCH_SYSTEM).strip()
        matched = resp.upper().startswith("YES")
        if resp.upper() not in ("YES", "NO"):
            ts(f"  WARN unexpected LLM response for ({pred!r}, {gold!r}): {resp!r}")
        results[(pred, gold)] = matched
    return results


def score_document(pred_set, gold_set, llm_results):
    """
    Score one document with a two-pass approach.

    Both pred_set and gold_set should already be de-duplicated (sets of
    normalised strings); exact-string matches are resolved first and removed
    from both sides so the LLM only evaluates genuinely ambiguous pairs.

    Pass 1 — TP / FP  (predicted → gold direction)
        For each predicted entity: is it semantically present in the gold set?
          YES  → TP  (consume the matched gold so it cannot be reused)
          NO   → FP

    Pass 2 — FN  (gold → predicted direction)
        For each gold entity not consumed in Pass 1:
          is it semantically present in the predicted set?
          Because Pass 1 was exhaustive from the predicted side (every
          predicted entity was checked against every remaining gold entity),
          any gold that is still unmatched here is a confirmed FN.

    Returns
    -------
    matched_pairs : list of {predicted, gold, match_type}  (match_type: 'exact' | 'llm')
    fp_list       : predicted entities with no match
    fn_list       : gold entities with no match
    tp, fp, fn    : counts
    """
    # ── Step 1: exact matches → immediate TPs ────────────────────────────
    # Both inputs are already sets; intersection gives zero-cost dedup.
    exact = pred_set & gold_set
    matched_pairs = [
        {"predicted": p, "gold": p, "match_type": "exact"} for p in sorted(exact)
    ]

    # Remove exact matches from both sides before LLM assessment.
    unmatched_pred = pred_set - exact
    unmatched_gold = gold_set - exact

    matched_pred: set = set()
    matched_gold: set = set()

    # ── Pass 1: TP / FP — iterate over every predicted entity ────────────
    # For each predicted entity p: is p semantically among the gold set?
    #   YES (any g in unmatched_gold where llm_results[(p, g)] is True)
    #       → TP; consume g so it cannot be claimed by another prediction.
    #   NO  (no g matches p)
    #       → FP; p is a false positive.
    # Each predicted entity can match at most one gold entity (1-to-1).
    for p in sorted(unmatched_pred):
        for g in sorted(unmatched_gold):
            if g not in matched_gold and llm_results.get((p, g), False):
                matched_pairs.append({"predicted": p, "gold": g, "match_type": "llm"})
                matched_pred.add(p)
                matched_gold.add(g)
                break  # p is resolved; move to next predicted entity

    fp_list = sorted(unmatched_pred - matched_pred)

    # ── Pass 2: FN — iterate over every remaining gold entity ────────────
    # For each gold entity g not consumed in Pass 1:
    #   is g semantically among the predicted set?
    #   Pass 1 was exhaustive (every p was checked against every g), so any
    #   gold that remains unmatched is a confirmed FN — no predicted entity
    #   covers it, either exactly or semantically.
    fn_list = sorted(unmatched_gold - matched_gold)

    tp = len(matched_pairs)
    fp = len(fp_list)
    fn = len(fn_list)

    return matched_pairs, fp_list, fn_list, tp, fp, fn


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate RDMA predictions on the RareDis benchmark"
    )
    parser.add_argument(
        "--model_type",
        required=True,
        help="Model whose predictions to evaluate (e.g. llama3_8b, qwen_32b)",
    )
    parser.add_argument(
        "--approach",
        choices=["rdma", "zeroshot", "rdrag", "dict"],
        default="rdma",
        help="Prediction approach: 'rdma' (default), 'zeroshot', 'rdrag', or 'dict'",
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,
        help="GPU index to use (default: 0); pass -1 to force CPU",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    gpu_id = None if args.gpu_id == -1 else args.gpu_id

    if args.approach == "zeroshot":
        predictions_file = (
            _RESULTS_DIR / "raredis" / f"zeroshot_{args.model_type}_predictions.jsonl"
        )
        eval_output = _RESULTS_DIR / "raredis" / f"eval_zeroshot_{args.model_type}.json"
    elif args.approach == "rdrag":
        predictions_file = (
            _RESULTS_DIR / "raredis" / f"{args.model_type}_rdrag_predictions.jsonl"
        )
        eval_output = _RESULTS_DIR / "raredis" / f"eval_rdrag_{args.model_type}.json"
    elif args.approach == "dict":
        predictions_file = _RESULTS_DIR / "raredis" / "dict_predictions.jsonl"
        eval_output = _RESULTS_DIR / "raredis" / "eval_dict.json"
    else:  # rdma
        predictions_file = (
            _RESULTS_DIR / "raredis" / f"{args.model_type}_predictions.jsonl"
        )
        eval_output = _RESULTS_DIR / "raredis" / f"eval_{args.model_type}.json"

    cfg = SimpleNamespace(
        gpu_id=gpu_id,
        condor=False,
        cpu=(gpu_id is None),
        retriever_gpu_id=None,
        retriever_cpu=False,
    )
    devices = setup_device(cfg)

    # ── Ground truth ──────────────────────────────────────────────────────
    ts("Loading RareDisDataset...")
    dataset = RareDisDataset()
    samples = dataset.set_task(RareDisNER(), num_workers=4)
    gold_map = {
        sample["patient_id"]: set(
            normalize(a) for a in pickle.loads(sample["annotations"])
        )
        for sample in samples
    }
    # Keep original (non-normalized) lists for the audit file
    gold_orig = {
        sample["patient_id"]: pickle.loads(sample["annotations"]) for sample in samples
    }
    ts(f"  {len(gold_map)} documents with gold annotations")

    # ── Predictions ───────────────────────────────────────────────────────
    ts(f"Loading predictions: {predictions_file}")
    raw_pred_map = load_predictions(predictions_file)
    norm_pred_map = {
        doc_id: set(normalize(p) for p in preds if p)
        for doc_id, preds in raw_pred_map.items()
    }
    ts(f"  {len(norm_pred_map)} prediction entries")

    common_ids = sorted(set(norm_pred_map) & set(gold_map))
    ts(f"  {len(common_ids)} documents in common (will be scored)")

    # ── Collect non-exact pairs for LLM ───────────────────────────────────
    # We collect all (predicted, gold) pairs that are not exact matches.
    # These cover Pass 1 (predicted → gold) exhaustively: for every predicted
    # entity that is not an exact match, we check it against every gold entity
    # that is also not an exact match.  Because matching is symmetric (A ≡ B
    # iff B ≡ A for disease equivalence), the same results also serve Pass 2
    # (gold → predicted) without extra LLM calls.
    # Pairs are de-duplicated globally so identical (pred, gold) strings that
    # appear across multiple documents are only sent to the LLM once.
    ts("Collecting non-exact pairs for LLM evaluation...")
    all_pairs: set = set()
    for doc_id in common_ids:
        pred_set = norm_pred_map[doc_id]
        gold_set = gold_map[doc_id]
        unmatched_pred = pred_set - gold_set
        unmatched_gold = gold_set - pred_set
        for p in unmatched_pred:
            for g in unmatched_gold:
                all_pairs.add((p, g))
    ts(f"  {len(all_pairs)} unique non-exact pairs to evaluate")

    # ── LLM evaluation ────────────────────────────────────────────────────
    llm_results = {}
    if all_pairs:
        judge_model = "mistral_24b"
        ts(f"Loading LLM ({judge_model})...")
        llm_client = LocalLLMClient(
            model_type=judge_model,
            device=devices["llm"],
            cache_dir=MODEL_CACHE_DIR,
            temperature=TEMPERATURE,
        )
        ts(f"Evaluating {len(all_pairs)} pairs...")
        llm_results = llm_match_pairs(llm_client, sorted(all_pairs))
        yes_count = sum(1 for v in llm_results.values() if v)
        ts(f"  LLM judged {yes_count}/{len(all_pairs)} pairs " "as equivalent")

    # ── Micro F1 + audit ──────────────────────────────────────────────────
    total_tp = total_fp = total_fn = 0
    audit_docs = []

    for doc_id in tqdm(common_ids, desc="Scoring"):
        matched_pairs, fp_list, fn_list, tp, fp, fn = score_document(
            norm_pred_map[doc_id], gold_map[doc_id], llm_results
        )
        total_tp += tp
        total_fp += fp
        total_fn += fn

        audit_docs.append(
            {
                "id": doc_id,
                "predicted": sorted(raw_pred_map[doc_id]),
                "annotations": sorted(gold_orig[doc_id]),
                "matched_pairs": matched_pairs,
                "fp": fp_list,
                "fn": fn_list,
            }
        )

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    ts("── Results ──────────────────────────────────────────────────────")
    ts(f"  Documents scored : {len(common_ids)}")
    ts(f"  TP={total_tp}  FP={total_fp}  FN={total_fn}")
    ts(f"  Precision : {precision:.4f}")
    ts(f"  Recall    : {recall:.4f}")
    ts(f"  Micro F1  : {f1:.4f}")

    # ── Write audit JSON ──────────────────────────────────────────────────
    audit = {
        "metrics": {
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "documents_scored": len(common_ids),
        },
        "documents": audit_docs,
    }
    eval_output.parent.mkdir(parents=True, exist_ok=True)
    with open(eval_output, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, ensure_ascii=False)
    ts(f"Audit written → {eval_output}")


if __name__ == "__main__":
    main()
