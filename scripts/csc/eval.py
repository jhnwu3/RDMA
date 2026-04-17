#!/usr/bin/env python3
"""
Evaluate HPO pipeline predictions on the CSC phenotype-mining benchmark.

Reads a JSONL file produced by run_hpo.py (each line has ``id``,
``predicted``, ``ground_truth``) and computes micro-averaged
precision, recall, and F1 at the HP-ID level.

Usage (from RDMA repo root):
  python scripts/csc/eval.py --predictions <path>.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

_RDMA_ROOT = Path("/home/johnwu3/projects/rare_disease/workspace/repos/RDMA")
sys.path.insert(0, str(_RDMA_ROOT))

_DEFAULT_ONTOLOGY = _RDMA_ROOT / "data" / "ontology" / "hpo_data_with_lineage.json"


def _build_hpo_lookup(ontology_path: Path) -> dict:
    """Return a dict mapping HP:XXXXXXX → human-readable label."""
    with open(ontology_path, encoding="utf-8") as f:
        data = json.load(f)
    return {k.replace("_", ":", 1): v.get("label", k) for k, v in data.items()}


def _resolve(ids: set, lookup: dict) -> list:
    """Return sorted list of (hp_id, label) pairs for a set of HP IDs."""
    return sorted((hid, lookup.get(hid, hid)) for hid in ids)


def compute_metrics(records: list, strict: bool = False) -> dict:
    """Compute micro-averaged P/R/F1 across all documents.

    Args:
        records: List of dicts with ``predicted`` and ``ground_truth`` lists.
        strict: If True, deduplicate ground-truth per document (lenient by
            default, counts duplicate gold labels once each).

    Returns:
        Dict with keys ``precision``, ``recall``, ``f1``,
        ``tp``, ``fp``, ``fn``, ``n_docs``.
    """
    tp = fp = fn = 0
    n_docs = 0

    for rec in records:
        predicted = set(h for h in rec.get("predicted", []) if h)
        gold_raw = [h for h in rec.get("ground_truth", []) if h]
        gold_set = set(gold_raw) if strict else set(gold_raw)

        tp += len(predicted & gold_set)
        fp += len(predicted - gold_set)
        fn += len(gold_set - predicted)
        n_docs += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "n_docs": n_docs,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate CSC HPO predictions"
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="Path to predictions JSONL (from run_hpo.py)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional CSV path to write per-document results",
    )
    parser.add_argument(
        "--hpo_ontology",
        type=Path,
        default=_DEFAULT_ONTOLOGY,
        help="Path to hpo_data_with_lineage.json for resolving term labels (default: %(default)s)",
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Print per-sample breakdown with term names to stdout (worst F1 first)",
    )
    args = parser.parse_args()

    records = []
    with open(args.predictions, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    print(f"Loaded {len(records)} prediction records from {args.predictions}")

    hpo_lookup: dict = {}
    if args.hpo_ontology and args.hpo_ontology.exists():
        hpo_lookup = _build_hpo_lookup(args.hpo_ontology)

    # Lenient and strict micro-averaged metrics (identical for CSC since
    # ground truth HPO IDs are already unique per phenotype name, but we
    # keep both for consistency with the BioLarkGSC eval script)
    lenient = compute_metrics(records, strict=False)
    strict = compute_metrics(records, strict=True)

    print()
    print("── Lenient (multi-set gold) ────────────────────────────────")
    print(f"  Precision : {lenient['precision']:.4f}")
    print(f"  Recall    : {lenient['recall']:.4f}")
    print(f"  F1        : {lenient['f1']:.4f}")
    print(f"  TP={lenient['tp']}  FP={lenient['fp']}  FN={lenient['fn']}")

    print()
    print("── Strict (deduplicated gold) ──────────────────────────────")
    print(f"  Precision : {strict['precision']:.4f}")
    print(f"  Recall    : {strict['recall']:.4f}")
    print(f"  F1        : {strict['f1']:.4f}")
    print(f"  TP={strict['tp']}  FP={strict['fp']}  FN={strict['fn']}")
    print(f"  Docs      : {strict['n_docs']}")

    if args.output:
        import csv

        rows = []
        for rec in records:
            predicted = set(h for h in rec.get("predicted", []) if h)
            gold = set(h for h in rec.get("ground_truth", []) if h)
            tp_doc = len(predicted & gold)
            fp_doc = len(predicted - gold)
            fn_doc = len(gold - predicted)
            p = tp_doc / (tp_doc + fp_doc) if (tp_doc + fp_doc) > 0 else 0.0
            r = tp_doc / (tp_doc + fn_doc) if (tp_doc + fn_doc) > 0 else 0.0
            f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            rows.append(
                {
                    "id": rec.get("id", ""),
                    "precision": round(p, 4),
                    "recall": round(r, 4),
                    "f1": round(f, 4),
                    "tp": tp_doc,
                    "fp": fp_doc,
                    "fn": fn_doc,
                    "n_predicted": len(predicted),
                    "n_gold": len(gold),
                }
            )

        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", newline="", encoding="utf-8") as csvf:
            writer = csv.DictWriter(
                csvf,
                fieldnames=[
                    "id",
                    "precision",
                    "recall",
                    "f1",
                    "tp",
                    "fp",
                    "fn",
                    "n_predicted",
                    "n_gold",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nPer-document results written to {args.output}")


if __name__ == "__main__":
    main()
