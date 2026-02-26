#!/usr/bin/env python3
"""
Evaluate RDMA predictions on the MIMIC-III rare-disease mining benchmark.

Set-based micro F1: per note, duplicates in predicted/gold are ignored.
Matching priority (first match wins per pair):
  1. ORPHA code exact match  (ORPHA:NNNNN == ORPHA:NNNNN)
  2. Entity string exact match  (case-insensitive)
  3. LLM semantic equivalence  (only for remaining unmatched pairs)

Gold annotations are loaded from mimic3_mining_rdma_human_annotations.json.
Predictions are loaded from a JSONL file with 'predicted' and optionally
'predicted_orpha_ids' fields.

Writes an audit JSON with per-note matched pairs, false positives, and
false negatives for inspection.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from tqdm import tqdm

_RDMA_ROOT = Path("/home/johnwu3/projects/rare_disease/workspace/repos/RDMA")
sys.path.insert(0, str(_RDMA_ROOT))

from rdma.utils.llm_client import LocalLLMClient  # noqa: E402
from rdma.utils.setup import setup_device  # noqa: E402

# ── CONFIG ────────────────────────────────────────────────────────────────────

_RESULTS_DIR = Path("/home/johnwu3/projects/rare_disease/workspace/results")
_DEFAULT_ANNOTATION_PATH = str(
    _RDMA_ROOT
    / "public_data"
    / "rare_disease_mining"
    / "mimic3_mining_rdma_human_annotations.json"
)
MODEL_CACHE_DIR = "/shared/rsaas/jw3/rare_disease/model_cache"
TEMPERATURE = 0.01

# ─────────────────────────────────────────────────────────────────────────────


def ts(msg):
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S} - {msg}", flush=True)


def normalize(s: str) -> str:
    return s.strip().lower()


def format_orpha_id(code) -> str:
    """Normalise an ORPHA code to 'ORPHA:12345' form."""
    code = str(code).strip() if code else ""
    if not code or code == "None":
        return ""
    if code.upper().startswith("ORPHA:"):
        return code.upper()
    if code.isdigit():
        return f"ORPHA:{code}"
    return code.upper()


def load_gold(annotation_path: str):
    """Load gold annotations from mimic3_mining_rdma_human_annotations.json.

    Returns:
        gold_pairs: {note_id -> list of (norm_mention, norm_orpha_code)}
        gold_orig:  {note_id -> list of original mention strings}
    """
    with open(annotation_path, encoding="utf-8") as fh:
        data = json.load(fh)

    gold_pairs: dict = {}
    gold_orig: dict = {}

    for doc_id, entry in data.items():
        pairs = []
        orig = []
        for anno in entry.get("annotations", []):
            if not anno.get("is_rare_disease", False):
                continue
            mention = anno["mention"]
            orpha = format_orpha_id(anno.get("orpha_code", ""))
            pairs.append((normalize(mention), normalize(orpha)))
            orig.append(mention)
        if pairs:
            gold_pairs[doc_id] = pairs
            gold_orig[doc_id] = orig

    return gold_pairs, gold_orig


def load_predictions(path: Path):
    """Load predictions from a JSONL file.

    Returns:
        raw_pred:  {note_id -> list[str]}  (original entity strings)
        pred_pairs: {note_id -> list of (norm_entity, norm_orpha_id)}
    """
    raw_pred: dict = {}
    pred_pairs: dict = {}

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
                orpha = (
                    format_orpha_id(orpha_ids[i])
                    if i < len(orpha_ids)
                    else ""
                )
                pairs.append((normalize(entity), normalize(orpha)))

            raw_pred[note_id] = entities
            pred_pairs[note_id] = pairs

    return raw_pred, pred_pairs


_MATCH_SYSTEM = (
    "You are a biomedical expert. "
    "Your response must be exactly one word: YES or NO. "
    "Do not include any other text, punctuation, or explanation."
)


def llm_match_pairs(llm_client, pairs):
    """Evaluate (pred_entity, gold_entity) string pairs with the LLM.

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
            ts(
                f"  WARN unexpected LLM response "
                f"for ({pred!r}, {gold!r}): {resp!r}"
            )
        results[(pred, gold)] = matched
    return results


def score_document(pred_pairs, gold_pairs, llm_results):
    """Score one document using three-tier matching.

    pred_pairs / gold_pairs: list of (norm_entity, norm_orpha_id)

    Matching order (first match wins for each predicted entity):
      1. ORPHA code exact match
      2. Entity string exact match
      3. LLM semantic match

    Returns:
        matched_pairs: list of dicts {predicted, gold, match_type, ...}
        fp_list:  unmatched predicted entity strings
        fn_list:  unmatched gold entity strings
        tp, fp, fn: counts
    """
    matched_gold_idx = set()
    matched_pred_idx = set()
    matched_pairs = []

    # ── Step 1: ORPHA code exact match ────────────────────────────────────
    for pi, (pe, po) in enumerate(pred_pairs):
        if not po:
            continue
        for gi, (ge, go) in enumerate(gold_pairs):
            if gi in matched_gold_idx or not go:
                continue
            if po == go:
                matched_pairs.append(
                    {
                        "predicted": pe,
                        "gold": ge,
                        "match_type": "orpha",
                        "orpha_id": po,
                    }
                )
                matched_gold_idx.add(gi)
                matched_pred_idx.add(pi)
                break

    # ── Step 2: Entity string exact match ─────────────────────────────────
    for pi, (pe, _) in enumerate(pred_pairs):
        if pi in matched_pred_idx:
            continue
        for gi, (ge, _) in enumerate(gold_pairs):
            if gi in matched_gold_idx:
                continue
            if pe == ge:
                matched_pairs.append(
                    {"predicted": pe, "gold": ge, "match_type": "exact"}
                )
                matched_gold_idx.add(gi)
                matched_pred_idx.add(pi)
                break

    # ── Step 3: LLM semantic match ─────────────────────────────────────────
    for pi, (pe, _) in enumerate(pred_pairs):
        if pi in matched_pred_idx:
            continue
        for gi, (ge, _) in enumerate(gold_pairs):
            if gi in matched_gold_idx:
                continue
            if llm_results.get((pe, ge), False):
                matched_pairs.append(
                    {"predicted": pe, "gold": ge, "match_type": "llm"}
                )
                matched_gold_idx.add(gi)
                matched_pred_idx.add(pi)
                break

    fp_list = [
        pred_pairs[pi][0]
        for pi in range(len(pred_pairs))
        if pi not in matched_pred_idx
    ]
    fn_list = [
        gold_pairs[gi][0]
        for gi in range(len(gold_pairs))
        if gi not in matched_gold_idx
    ]
    tp = len(matched_pairs)
    fp = len(fp_list)
    fn = len(fn_list)

    return matched_pairs, fp_list, fn_list, tp, fp, fn


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate RDMA predictions on the "
            "MIMIC-III rare-disease mining benchmark"
        )
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
        help=(
            "Prediction approach: rdma (default), zeroshot, rdrag, or dict. "
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
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,
        help="GPU index to use (default: 0); pass -1 to force CPU",
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
    # rdma (default)
    return base / f"{model_type}_predictions.jsonl"


def main():
    args = parse_args()
    gpu_id = None if args.gpu_id == -1 else args.gpu_id

    predictions_file = args.predictions_file or _predictions_path(
        args.approach, args.model_type
    )
    tag = (
        f"{args.approach}_{args.model_type}"
        if args.approach != "dict"
        else f"dict_{args.model_type}"
    )
    eval_output = _RESULTS_DIR / "mimic3_rd_mining" / f"eval_{tag}.json"

    cfg = SimpleNamespace(
        gpu_id=gpu_id,
        condor=False,
        cpu=(gpu_id is None),
        retriever_gpu_id=None,
        retriever_cpu=False,
    )
    devices = setup_device(cfg)

    # ── Ground truth ──────────────────────────────────────────────────────
    ts(f"Loading gold annotations: {args.annotation_path}")
    gold_pairs, gold_orig = load_gold(args.annotation_path)
    ts(
        f"  {len(gold_pairs)} annotated notes with confirmed "
        f"rare-disease mentions"
    )

    # ── Predictions ───────────────────────────────────────────────────────
    ts(f"Loading predictions: {predictions_file}")
    raw_pred, pred_pairs = load_predictions(predictions_file)
    ts(f"  {len(pred_pairs)} prediction entries")

    common_ids = sorted(set(pred_pairs) & set(gold_pairs))
    ts(f"  {len(common_ids)} notes in common (will be scored)")

    # ── Collect non-exact / non-ORPHA pairs for LLM ───────────────────────
    ts("Collecting pairs for LLM evaluation...")
    llm_pairs: set = set()

    for note_id in common_ids:
        pp = pred_pairs[note_id]
        gp = gold_pairs[note_id]

        # Track which gold items are already covered by ORPHA or exact match
        matched_gold = set()
        matched_pred = set()

        # ORPHA pre-pass
        for pi, (pe, po) in enumerate(pp):
            if not po:
                continue
            for gi, (ge, go) in enumerate(gp):
                if gi in matched_gold or not go:
                    continue
                if po == go:
                    matched_gold.add(gi)
                    matched_pred.add(pi)
                    break

        # Exact entity pre-pass
        for pi, (pe, _) in enumerate(pp):
            if pi in matched_pred:
                continue
            for gi, (ge, _) in enumerate(gp):
                if gi in matched_gold:
                    continue
                if pe == ge:
                    matched_gold.add(gi)
                    matched_pred.add(pi)
                    break

        # Remaining unmatched → need LLM
        for pi, (pe, _) in enumerate(pp):
            if pi in matched_pred:
                continue
            for gi, (ge, _) in enumerate(gp):
                if gi in matched_gold:
                    continue
                llm_pairs.add((pe, ge))

    ts(f"  {len(llm_pairs)} unique pairs requiring LLM evaluation")

    # ── LLM evaluation ────────────────────────────────────────────────────
    llm_results = {}
    if llm_pairs:
        ts(f"Loading LLM ({args.model_type})...")
        llm_client = LocalLLMClient(
            model_type=args.model_type,
            device=devices["llm"],
            cache_dir=MODEL_CACHE_DIR,
            temperature=TEMPERATURE,
        )
        ts(f"Evaluating {len(llm_pairs)} pairs...")
        llm_results = llm_match_pairs(llm_client, sorted(llm_pairs))
        yes_count = sum(1 for v in llm_results.values() if v)
        ts(
            f"  LLM judged {yes_count}/{len(llm_pairs)} pairs as equivalent"
        )

    # ── Micro F1 + audit ──────────────────────────────────────────────────
    total_tp = total_fp = total_fn = 0
    audit_docs = []

    for note_id in tqdm(common_ids, desc="Scoring"):
        matched, fp_list, fn_list, tp, fp, fn = score_document(
            pred_pairs[note_id], gold_pairs[note_id], llm_results
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

    precision = (
        total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    )
    recall = (
        total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    )
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
