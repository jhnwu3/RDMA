#!/usr/bin/env python3
"""
Evaluate RDMA predictions on the MIMIC-III rare-disease mining benchmark.

Matching is purely ORPHA-code-based: the numeric part of the ORPHA code is
extracted from both the prediction and the gold annotation and compared.
A prediction is a TP if and only if its ORPHA number matches a gold ORPHA
number for that note (1-to-1, de-duplicated).  Predictions without an ORPHA
code, or whose ORPHA number has no corresponding gold entry, are FPs.  Gold
entries not matched by any prediction are FNs.

Gold annotations are loaded from mimic3_mining_rdma_human_annotations.json.
Predictions are loaded from a JSONL file with 'predicted' and
'predicted_orpha_ids' fields (as written by run_mimic3_rd_mining.py).

Writes an audit JSON with per-note matched pairs (with ORPHA numbers),
false positives, and false negatives for inspection.
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

_RDMA_ROOT = Path("/home/johnwu3/projects/rare_disease/workspace/repos/RDMA")
sys.path.insert(0, str(_RDMA_ROOT))

# ── CONFIG ────────────────────────────────────────────────────────────────────

_RESULTS_DIR = Path("/home/johnwu3/projects/rare_disease/workspace/results")
_DEFAULT_ANNOTATION_PATH = str(
    _RDMA_ROOT
    / "public_data"
    / "rare_disease_mining"
    / "mimic3_mining_rdma_human_annotations.json"
)

# ─────────────────────────────────────────────────────────────────────────────


def ts(msg):
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S} - {msg}", flush=True)


def extract_orpha_number(code) -> str:
    """Extract the bare numeric part from an ORPHA code string.

    Handles all common forms::

        "ORPHA:324"  →  "324"
        "orpha:324"  →  "324"
        "324"        →  "324"
        "324.0"      →  "324"
        "" / None    →  ""

    Returns an empty string when no digit sequence can be extracted.
    """
    if not code:
        return ""
    s = str(code).strip()
    m = re.search(r"\d+", s)
    return m.group(0) if m else ""


def load_gold(annotation_path: str):
    """Load gold annotations from mimic3_mining_rdma_human_annotations.json.

    Returns:
        gold_map:  {note_id -> list of orpha_number strings (bare digits)}
        gold_orig: {note_id -> list of original mention strings (for audit)}
    """
    with open(annotation_path, encoding="utf-8") as fh:
        data = json.load(fh)

    gold_map: dict = {}
    gold_orig: dict = {}

    for doc_id, entry in data.items():
        numbers = []
        orig = []
        for anno in entry.get("annotations", []):
            if not anno.get("is_rare_disease", False):
                continue
            num = extract_orpha_number(anno.get("orpha_code", ""))
            if num:
                numbers.append(num)
                orig.append(anno["mention"])
        if numbers:
            gold_map[doc_id] = numbers
            gold_orig[doc_id] = orig

    return gold_map, gold_orig


def load_predictions(path: Path):
    """Load predictions from a JSONL file.

    Returns:
        raw_pred: {note_id -> list[str]}  (original entity strings, for audit)
        pred_map: {note_id -> list of (entity_str, orpha_number)}
    """
    raw_pred: dict = {}
    pred_map: dict = {}

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            note_id = obj["id"]
            entities = [e for e in obj.get("predicted", []) if e]
            orpha_ids = obj.get("predicted_orpha_ids", [])

            pairs = []
            for i, entity in enumerate(entities):
                raw_id = orpha_ids[i] if i < len(orpha_ids) else ""
                num = extract_orpha_number(raw_id)
                pairs.append((entity, num))

            raw_pred[note_id] = entities
            pred_map[note_id] = pairs

    return raw_pred, pred_map


def score_document(pred_pairs, gold_numbers):
    """Score one note using ORPHA-number-only matching.

    Args:
        pred_pairs:   list of (entity_str, orpha_number) for predictions.
        gold_numbers: list of orpha_number strings from gold annotations.

    Returns:
        matched_pairs: list of dicts {predicted, predicted_orpha, gold_orpha}
        fp_list:       list of {entity, orpha_number} for unmatched predictions
        fn_list:       list of orpha_number strings for unmatched gold entries
        tp, fp, fn:    counts
    """
    remaining_gold = list(gold_numbers)  # mutable copy; remove on match
    matched_pairs = []
    fp_list = []

    for entity, num in pred_pairs:
        if num and num in remaining_gold:
            matched_pairs.append(
                {
                    "predicted": entity,
                    "predicted_orpha": num,
                    "gold_orpha": num,
                }
            )
            remaining_gold.remove(num)  # 1-to-1: consumed gold can't match again
        else:
            fp_list.append({"entity": entity, "orpha_number": num or ""})

    fn_list = remaining_gold  # gold numbers not claimed by any prediction

    tp = len(matched_pairs)
    fp = len(fp_list)
    fn = len(fn_list)

    return matched_pairs, fp_list, fn_list, tp, fp, fn


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate RDMA predictions on the "
            "MIMIC-III rare-disease mining benchmark (ORPHA-code matching)"
        )
    )
    parser.add_argument(
        "--model_type",
        required=True,
        help="Model whose predictions to evaluate (e.g. llama3_8b, qwen_32b)",
    )
    parser.add_argument(
        "--approach",
        choices=[
            "rdma",
            "zeroshot",
            "rdrag",
            "dict",
            "biobert_mrc",
            "bioclinicalbert_ner",
            "i2b2_rd",
        ],
        default="rdma",
        help=(
            "Prediction approach: rdma (default), zeroshot, rdrag, dict, "
            "biobert_mrc, bioclinicalbert_ner, or i2b2_rd. "
            "Determines the predictions filename."
        ),
    )
    parser.add_argument(
        "--predictions_file",
        type=Path,
        default=None,
        help="Override predictions file path (ignores --approach / --model_type)",
    )
    parser.add_argument(
        "--annotation_path",
        type=str,
        default=_DEFAULT_ANNOTATION_PATH,
        help=(
            "Path to mimic3_mining_rdma_human_annotations.json "
            "(default: %(default)s)"
        ),
    )
    return parser.parse_args()


def _predictions_path(approach: str, model_type: str) -> Path:
    base = _RESULTS_DIR / "mimic3_rd_mining"
    if approach == "zeroshot":
        return base / f"zeroshot_{model_type}_predictions.jsonl"
    if approach == "rdrag":
        return base / f"{model_type}_rdrag_predictions.jsonl"
    if approach == "dict":
        return base / "dict_predictions.jsonl"
    if approach == "biobert_mrc":
        return base / "biobert_mrc.jsonl"
    if approach == "bioclinicalbert_ner":
        return base / "bioclinicalbert_ner.jsonl"
    if approach == "i2b2_rd":
        return base / "i2b2_rd.jsonl"
    # rdma (default)
    return base / f"{model_type}_predictions.jsonl"


def main():
    args = parse_args()

    predictions_file = args.predictions_file or _predictions_path(
        args.approach, args.model_type
    )
    tag = (
        f"{args.approach}_{args.model_type}"
        if args.approach != "dict"
        else f"dict_{args.model_type}"
    )
    eval_output = _RESULTS_DIR / "mimic3_rd_mining" / f"eval_{tag}.json"

    # ── Ground truth ──────────────────────────────────────────────────────
    ts(f"Loading gold annotations: {args.annotation_path}")
    gold_map, gold_orig = load_gold(args.annotation_path)
    ts(f"  {len(gold_map)} annotated notes with confirmed rare-disease mentions")

    # ── Predictions ───────────────────────────────────────────────────────
    ts(f"Loading predictions: {predictions_file}")
    raw_pred, pred_map = load_predictions(predictions_file)
    ts(f"  {len(pred_map)} prediction entries")

    common_ids = sorted(set(pred_map) & set(gold_map))
    ts(f"  {len(common_ids)} notes in common (will be scored)")

    # ── Micro F1 + audit ──────────────────────────────────────────────────
    total_tp = total_fp = total_fn = 0
    audit_docs = []

    for note_id in tqdm(common_ids, desc="Scoring"):
        matched, fp_list, fn_list, tp, fp, fn = score_document(
            pred_map[note_id], gold_map[note_id]
        )
        total_tp += tp
        total_fp += fp
        total_fn += fn

        audit_docs.append(
            {
                "id": note_id,
                "predicted": raw_pred.get(note_id, []),
                "annotations": gold_orig[note_id],
                "matched_pairs": matched,
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
    ts(f"  Notes scored     : {len(common_ids)}")
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
            "notes_scored": len(common_ids),
        },
        "documents": audit_docs,
    }
    eval_output.parent.mkdir(parents=True, exist_ok=True)
    with open(eval_output, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, ensure_ascii=False)
    ts(f"Audit written → {eval_output}")


if __name__ == "__main__":
    main()
