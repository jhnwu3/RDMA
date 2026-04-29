#!/usr/bin/env python3
"""
Evaluate predictions on the MIMIC-III rare-disease text-extraction benchmark.

Set-based micro F1: per note, duplicates in predicted/gold are ignored.
Exact string matches (case-insensitive, stripped) are handled directly.
Non-exact pairs are evaluated by an LLM for semantic equivalence.

Gold annotations are loaded from mimic3_mining_rdma_human_annotations.json
(``mention`` strings where ``is_rare_disease=True``).

Predictions are loaded from a JSONL file where each row has the form:
    {"id": "<note_id>", "predicted": ["entity1", "entity2", ...]}
or
    {"note_id": "<note_id>", "predicted": [...]}

Writes an audit JSON with per-note matched pairs, false positives, and
false negatives for inspection.

Usage (from RDMA repo root):
    python scripts/mimic3_rd_mining_text/eval.py \\
        --model_type llama3_8b --approach rdma

    # Dry-run (load gold + predictions, report counts, skip scoring):
    python scripts/mimic3_rd_mining_text/eval.py \\
        --model_type llama3_8b --approach rdma --dry_run
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
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


def normalize(s):
    return s.strip().lower()


def load_gold(annotation_path: str, rare_only: bool = True):
    """Load gold entity strings from the MIMIC-III annotation JSON.

    Returns:
        gold_map:  {note_id -> set of normalised mention strings}
        gold_orig: {note_id -> list of original mention strings (for audit)}
    """
    with open(annotation_path, encoding="utf-8") as fh:
        data = json.load(fh)

    gold_map: dict = {}
    gold_orig: dict = {}
    for doc_id, entry in data.items():
        mentions = []
        for anno in entry.get("annotations", []):
            if rare_only and not anno.get("is_rare_disease", False):
                continue
            mention = anno.get("mention", "").strip()
            if mention:
                mentions.append(mention)
        gold_map[doc_id] = set(normalize(m) for m in mentions)
        gold_orig[doc_id] = mentions

    return gold_map, gold_orig


def load_predictions(path):
    """Load JSONL predictions, accepting both 'id' and 'note_id' key."""
    preds = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            note_id = obj.get("id") or obj.get("note_id")
            if note_id is None:
                continue
            preds[str(note_id)] = obj.get("predicted", [])
    return preds


_MATCH_SYSTEM = (
    "You are a biomedical expert. "
    "Your response must be exactly one word: YES or NO. "
    "Do not include any other text, punctuation, or explanation."
)


def llm_match_pairs(llm_client, pairs):
    """Evaluate (pred, gold) string pairs with the LLM.

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
    """Score one note using a two-pass exact+LLM approach.

    Returns:
        matched_pairs, fp_list, fn_list, tp, fp, fn
    """
    exact = pred_set & gold_set
    matched_pairs = [
        {"predicted": p, "gold": p, "match_type": "exact"} for p in sorted(exact)
    ]

    unmatched_pred = pred_set - exact
    unmatched_gold = gold_set - exact
    matched_gold: set = set()
    matched_pred: set = set()

    for p in sorted(unmatched_pred):
        for g in sorted(unmatched_gold):
            if g not in matched_gold and llm_results.get((p, g), False):
                matched_pairs.append({"predicted": p, "gold": g, "match_type": "llm"})
                matched_pred.add(p)
                matched_gold.add(g)
                break

    fp_list = sorted(unmatched_pred - matched_pred)
    fn_list = sorted(unmatched_gold - matched_gold)

    tp = len(matched_pairs)
    fp = len(fp_list)
    fn = len(fn_list)

    return matched_pairs, fp_list, fn_list, tp, fp, fn


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate predictions on the MIMIC-III rare-disease "
            "text-extraction benchmark"
        )
    )
    parser.add_argument(
        "--model_type",
        required=True,
        help="Model whose predictions to evaluate (e.g. llama3_8b, biobert_mrc)",
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
        help="Prediction approach",
    )
    parser.add_argument(
        "--annotation_path",
        default=_DEFAULT_ANNOTATION_PATH,
        help="Path to mimic3_mining_rdma_human_annotations.json",
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,
        help="GPU index for the judge LLM (default: 0); -1 to force CPU",
    )
    parser.add_argument(
        "--condor",
        action="store_true",
        help="Running under HTCondor: use generic 'cuda' device",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU (no GPU)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Override output directory for the audit JSON",
    )
    parser.add_argument(
        "--predictions_file",
        type=Path,
        default=None,
        help="Explicit path to predictions JSONL (overrides approach-based path resolution)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Load gold + predictions, print counts, then exit without scoring",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Resolve prediction file and output path ───────────────────────────
    results_base = _RESULTS_DIR / "mimic3_rd_mining_text"
    if args.approach == "zeroshot":
        predictions_file = (
            results_base / f"zeroshot_{args.model_type}_predictions.jsonl"
        )
        eval_output = results_base / f"eval_zeroshot_{args.model_type}.json"
    elif args.approach == "rdrag":
        predictions_file = results_base / f"{args.model_type}_rdrag_predictions.jsonl"
        eval_output = results_base / f"eval_rdrag_{args.model_type}.json"
    elif args.approach == "dict":
        predictions_file = results_base / "dict_predictions.jsonl"
        eval_output = results_base / "eval_dict.json"
    elif args.approach == "biobert_mrc":
        # biobert_mrc_mimic3.py writes to results/mimic3/biobert_mrc/
        predictions_file = (
            _RESULTS_DIR / "mimic3" / "biobert_mrc" / "per_note_predictions.jsonl"
        )
        eval_output = results_base / "eval_biobert_mrc.json"
    elif args.approach == "bioclinicalbert_ner":
        # bioclinicalbert_ner.py writes to results/mimic3/bioclinicalbert_ner/
        predictions_file = (
            _RESULTS_DIR
            / "mimic3"
            / "bioclinicalbert_ner"
            / "per_note_predictions.jsonl"
        )
        eval_output = results_base / "eval_bioclinicalbert_ner.json"
    elif args.approach == "i2b2_rd":
        predictions_file = (
            _RESULTS_DIR / "mimic3" / "i2b2_rd" / "per_note_predictions.jsonl"
        )
        eval_output = results_base / "eval_i2b2_rd.json"
    else:  # rdma
        predictions_file = results_base / f"{args.model_type}_predictions.jsonl"
        eval_output = results_base / f"eval_{args.model_type}.json"

    if args.predictions_file is not None:
        predictions_file = args.predictions_file

    if args.output_dir is not None:
        eval_output = args.output_dir / eval_output.name

    # ── Load gold ─────────────────────────────────────────────────────────
    ts(f"Loading gold annotations from {args.annotation_path} ...")
    gold_map, gold_orig = load_gold(args.annotation_path)
    ts(f"  {len(gold_map)} annotated notes loaded")

    # ── Load predictions ──────────────────────────────────────────────────
    ts(f"Loading predictions: {predictions_file}")
    if not Path(predictions_file).exists():
        ts(f"ERROR: predictions file not found: {predictions_file}")
        raise SystemExit(1)
    raw_pred_map = load_predictions(predictions_file)
    norm_pred_map = {
        note_id: set(normalize(p) for p in preds if p)
        for note_id, preds in raw_pred_map.items()
    }
    ts(f"  {len(norm_pred_map)} prediction entries loaded")

    common_ids = sorted(set(norm_pred_map) & set(gold_map))
    ts(f"  {len(common_ids)} notes in common (will be scored)")

    if args.dry_run:
        ts("Dry-run complete. Exiting.")
        return

    # ── Device setup ──────────────────────────────────────────────────────
    from types import SimpleNamespace

    gpu_id = None if (args.cpu or args.gpu_id == -1) else args.gpu_id
    cfg = SimpleNamespace(
        gpu_id=gpu_id,
        condor=args.condor,
        cpu=(gpu_id is None and not args.condor),
        retriever_gpu_id=None,
        retriever_cpu=False,
    )
    devices = setup_device(cfg)

    # ── Collect non-exact pairs for LLM ───────────────────────────────────
    ts("Collecting non-exact pairs for LLM evaluation...")
    all_pairs: set = set()
    for note_id in common_ids:
        pred_set = norm_pred_map[note_id]
        gold_set = gold_map[note_id]
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
        ts(f"  LLM judged {yes_count}/{len(all_pairs)} pairs as equivalent")

    # ── Micro F1 + audit ──────────────────────────────────────────────────
    total_tp = total_fp = total_fn = 0
    audit_docs = []

    for note_id in tqdm(common_ids, desc="Scoring"):
        matched_pairs, fp_list, fn_list, tp, fp, fn = score_document(
            norm_pred_map[note_id], gold_map[note_id], llm_results
        )
        total_tp += tp
        total_fp += fp
        total_fn += fn

        audit_docs.append(
            {
                "id": note_id,
                "predicted": sorted(raw_pred_map[note_id]),
                "gold": sorted(gold_orig[note_id]),
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
    ts(f"  Notes scored  : {len(common_ids)}")
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
