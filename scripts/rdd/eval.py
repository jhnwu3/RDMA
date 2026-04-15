#!/usr/bin/env python3
"""
Evaluate RDMA predictions on the RDD benchmark.

Supports two tasks (--task):

  ner (default)
      Set-based micro F1: per document, duplicates in predicted/gold are
      ignored.  Exact string matches (case-insensitive, stripped) are handled
      directly.  Non-exact pairs are evaluated by an LLM for semantic
      equivalence.  Writes an audit JSON with per-document matched pairs,
      false positives, and false negatives.

  relation
      Binary classification metrics (accuracy, precision, recall, F1) for
      the (rare disease, disability) relation pairs from
      Relationships/positive.csv and negative.csv.  No LLM is needed: gold
      labels are 0/1 integers and predictions are compared directly.
      Writes an audit JSON with per-sample predictions and gold labels.
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
from datasets.rdd import RDDDataset  # noqa: E402
from tasks.rdd import RDDNER, RDDRelationExtraction  # noqa: E402

# ── CONFIG ───────────────────────────────────────────────────────────────────

_RESULTS_DIR = Path("/home/johnwu3/projects/rare_disease/workspace/results")
MODEL_CACHE_DIR = "/shared/rsaas/jw3/rare_disease/model_cache"
TEMPERATURE = 0.01

# ─────────────────────────────────────────────────────────────────────────────


def ts(msg):
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S} - {msg}", flush=True)


def _rel_sample_id(sample: dict) -> str:
    """Stable unique ID for a relation sample: doc_id|rd_start|dis_start."""
    return f"{sample['patient_id']}" f"|{sample['rd_start']}|{sample['dis_start']}"


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
            "Answer YES if they are the same disease (including different "
            "names, abbreviations, or minor spelling variants). "
            "Answer NO otherwise."
        )
        resp = llm_client.query(prompt, _MATCH_SYSTEM).strip()
        matched = resp.upper().startswith("YES")
        if resp.upper() not in ("YES", "NO"):
            ts(f"  WARN unexpected LLM response for " f"({pred!r}, {gold!r}): {resp!r}")
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
    matched_pairs : list of {predicted, gold, match_type}
                    (match_type: 'exact' | 'llm')
    fp_list       : predicted entities with no match
    fn_list       : gold entities with no match
    tp, fp, fn    : counts
    """
    # ── Step 1: exact matches → immediate TPs ────────────────────────────
    exact = pred_set & gold_set
    matched_pairs = [
        {"predicted": p, "gold": p, "match_type": "exact"} for p in sorted(exact)
    ]

    unmatched_pred = pred_set - exact
    unmatched_gold = gold_set - exact

    matched_pred: set = set()
    matched_gold: set = set()

    # ── Pass 1: TP / FP ──────────────────────────────────────────────────
    for p in sorted(unmatched_pred):
        for g in sorted(unmatched_gold):
            if g not in matched_gold and llm_results.get((p, g), False):
                matched_pairs.append({"predicted": p, "gold": g, "match_type": "llm"})
                matched_pred.add(p)
                matched_gold.add(g)
                break

    fp_list = sorted(unmatched_pred - matched_pred)

    # ── Pass 2: FN ───────────────────────────────────────────────────────
    fn_list = sorted(unmatched_gold - matched_gold)

    tp = len(matched_pairs)
    fp = len(fp_list)
    fn = len(fn_list)

    return matched_pairs, fp_list, fn_list, tp, fp, fn


def eval_relation(samples, pred_map):
    """Score binary relation classification.

    Args:
        samples: Iterable of sample dicts from RDDRelationExtraction.
        pred_map: Dict mapping sample_id -> predicted label (int 0 or 1).

    Returns:
        Tuple of (metrics dict, audit_rows list).
    """
    gold_map = {}
    meta_map = {}
    for s in samples:
        sid = _rel_sample_id(s)
        gold_map[sid] = pickle.loads(s["label"])
        meta_map[sid] = {
            "rare_disease": s["rare_disease"],
            "disability": s["disability"],
            "patient_id": s["patient_id"],
        }

    common = sorted(set(pred_map) & set(gold_map))
    tp = fp = tn = fn = 0
    audit_rows = []

    for sid in common:
        pred = int(pred_map[sid])
        gold = int(gold_map[sid])
        if pred == 1 and gold == 1:
            tp += 1
            outcome = "TP"
        elif pred == 1 and gold == 0:
            fp += 1
            outcome = "FP"
        elif pred == 0 and gold == 0:
            tn += 1
            outcome = "TN"
        else:
            fn += 1
            outcome = "FN"

        audit_rows.append(
            {
                "id": sid,
                **meta_map[sid],
                "predicted": pred,
                "gold": gold,
                "outcome": outcome,
            }
        )

    n = len(common)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    accuracy = (tp + tn) / n if n > 0 else 0.0

    metrics = dict(
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        precision=round(precision, 6),
        recall=round(recall, 6),
        f1=round(f1, 6),
        accuracy=round(accuracy, 6),
        samples_scored=n,
        samples_in_gold=len(gold_map),
        samples_in_pred=len(pred_map),
    )
    return metrics, audit_rows


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate RDMA predictions on the RDD benchmark"
    )
    parser.add_argument(
        "--task",
        choices=["ner", "relation"],
        default="ner",
        help="Task to evaluate: 'ner' (default) or 'relation'",
    )
    parser.add_argument(
        "--model_type",
        required=True,
        help="Model whose predictions to evaluate (e.g. qwen_32b)",
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

    task_suffix = "_rel" if args.task == "relation" else ""

    if args.approach == "zeroshot":
        predictions_file = (
            _RESULTS_DIR
            / "rdd"
            / f"zeroshot_{args.model_type}{task_suffix}_predictions.jsonl"
        )
        eval_output = (
            _RESULTS_DIR / "rdd" / f"eval_zeroshot_{args.model_type}{task_suffix}.json"
        )
    elif args.approach == "rdrag":
        predictions_file = (
            _RESULTS_DIR
            / "rdd"
            / f"{args.model_type}_rdrag{task_suffix}_predictions.jsonl"
        )
        eval_output = (
            _RESULTS_DIR / "rdd" / f"eval_rdrag_{args.model_type}{task_suffix}.json"
        )
    elif args.approach == "dict":
        predictions_file = _RESULTS_DIR / "rdd" / "dict_predictions.jsonl"
        eval_output = _RESULTS_DIR / "rdd" / f"eval_dict{task_suffix}.json"
    else:  # rdma
        predictions_file = (
            _RESULTS_DIR / "rdd" / f"{args.model_type}{task_suffix}_predictions.jsonl"
        )
        eval_output = _RESULTS_DIR / "rdd" / f"eval_{args.model_type}{task_suffix}.json"

    cfg = SimpleNamespace(
        gpu_id=gpu_id,
        condor=False,
        cpu=(gpu_id is None),
        retriever_gpu_id=None,
        retriever_cpu=False,
    )
    devices = setup_device(cfg)

    # ── Ground truth ──────────────────────────────────────────────────────
    ts("Loading RDDDataset...")
    dataset = RDDDataset()

    if args.task == "relation":
        samples = dataset.set_task(RDDRelationExtraction())
        ts(f"  {len(samples)} relation samples (gold)")

        # ── Predictions ───────────────────────────────────────────────────
        ts(f"Loading predictions: {predictions_file}")
        raw_pred_map = load_predictions(predictions_file)
        ts(f"  {len(raw_pred_map)} prediction entries")

        # ── Score ─────────────────────────────────────────────────────────
        ts("Scoring relation classification…")
        metrics, audit_rows = eval_relation(samples, raw_pred_map)

        ts("── Results ──────────────────────────────────────────────────")
        ts(f"  Samples scored : {metrics['samples_scored']}")
        ts(
            f"  TP={metrics['tp']}  FP={metrics['fp']}  "
            f"TN={metrics['tn']}  FN={metrics['fn']}"
        )
        ts(f"  Accuracy  : {metrics['accuracy']:.4f}")
        ts(f"  Precision : {metrics['precision']:.4f}")
        ts(f"  Recall    : {metrics['recall']:.4f}")
        ts(f"  F1        : {metrics['f1']:.4f}")

        audit = {"metrics": metrics, "samples": audit_rows}

    else:  # ner
        samples = dataset.set_task(RDDNER())
        gold_map = {
            sample["patient_id"]: set(
                normalize(a) for a in pickle.loads(sample["annotations"])
            )
            for sample in samples
        }
        gold_orig = {
            sample["patient_id"]: pickle.loads(sample["annotations"])
            for sample in samples
        }
        ts(f"  {len(gold_map)} documents with gold annotations")

        # ── Predictions ───────────────────────────────────────────────────
        ts(f"Loading predictions: {predictions_file}")
        raw_pred_map = load_predictions(predictions_file)
        norm_pred_map = {
            doc_id: set(normalize(p) for p in preds if p)
            for doc_id, preds in raw_pred_map.items()
        }
        ts(f"  {len(norm_pred_map)} prediction entries")

        common_ids = sorted(set(norm_pred_map) & set(gold_map))
        ts(f"  {len(common_ids)} documents in common (will be scored)")

        # ── Collect non-exact pairs for LLM ───────────────────────────────
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

        # ── LLM evaluation ────────────────────────────────────────────────
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

        # ── Micro F1 + audit ──────────────────────────────────────────────
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

        precision = (
            total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        )
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        ts("── Results ──────────────────────────────────────────────────")
        ts(f"  Documents scored : {len(common_ids)}")
        ts(f"  TP={total_tp}  FP={total_fp}  FN={total_fn}")
        ts(f"  Precision : {precision:.4f}")
        ts(f"  Recall    : {recall:.4f}")
        ts(f"  Micro F1  : {f1:.4f}")

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

    # ── Write audit JSON ──────────────────────────────────────────────────
    eval_output.parent.mkdir(parents=True, exist_ok=True)
    with open(eval_output, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, ensure_ascii=False)
    ts(f"Audit written → {eval_output}")


if __name__ == "__main__":
    main()
