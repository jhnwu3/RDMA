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
import pickle
import sys
from pathlib import Path

_RDMA_ROOT = Path("/home/johnwu3/projects/rare_disease/workspace/repos/RDMA")
sys.path.insert(0, str(_RDMA_ROOT))

_DEFAULT_ONTOLOGY = _RDMA_ROOT / "data" / "ontology" / "hpo_data_with_lineage.json"
_DEFAULT_DATASET_CACHE_DIR = "/shared/eng/pyhealth/csc"
_SNIPPET_LEN = 300


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
    parser.add_argument(
        "--dataset_cache_dir",
        type=str,
        default=_DEFAULT_DATASET_CACHE_DIR,
        help="PyHealth dataset cache directory for loading text snippets (default: %(default)s)",
    )
    parser.add_argument(
        "--audit_json",
        type=Path,
        default=None,
        help="Write per-document audit + aggregate metrics to this JSON path (for bootstrap CI)",
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

    text_lookup: dict = {}
    try:
        from datasets.csc import CSCDataset
        from tasks.csc import CSCPhenotypeMining
        dataset = CSCDataset(cache_dir=args.dataset_cache_dir)
        samples = dataset.set_task(CSCPhenotypeMining())
        text_lookup = {
            s["patient_id"]: pickle.loads(s["text"]) for s in samples
        }
        print(f"Loaded text for {len(text_lookup)} samples from dataset")
    except Exception as e:
        print(f"Warning: could not load dataset text ({e}); snippets will be omitted")

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

    # Build per-sample rows (used by both --output and --inspect)
    per_sample = []
    for rec in records:
        doc_id = rec.get("id", "")
        predicted = set(h for h in rec.get("predicted", []) if h)
        gold = set(h for h in rec.get("ground_truth", []) if h)
        tp_ids = predicted & gold
        fp_ids = predicted - gold
        fn_ids = gold - predicted
        tp_doc = len(tp_ids)
        fp_doc = len(fp_ids)
        fn_doc = len(fn_ids)
        p = tp_doc / (tp_doc + fp_doc) if (tp_doc + fp_doc) > 0 else 0.0
        r = tp_doc / (tp_doc + fn_doc) if (tp_doc + fn_doc) > 0 else 0.0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        full_text = text_lookup.get(doc_id, "")
        snippet = full_text[:_SNIPPET_LEN] + ("…" if len(full_text) > _SNIPPET_LEN else "")
        per_sample.append(
            {
                "id": doc_id,
                "precision": round(p, 4),
                "recall": round(r, 4),
                "f1": round(f, 4),
                "tp": tp_doc,
                "fp": fp_doc,
                "fn": fn_doc,
                "n_predicted": len(predicted),
                "n_gold": len(gold),
                "tp_ids": tp_ids,
                "fp_ids": fp_ids,
                "fn_ids": fn_ids,
                "snippet": snippet,
            }
        )

    if args.inspect:
        sorted_samples = sorted(per_sample, key=lambda r: r["f1"])
        print()
        print("── Per-sample inspection (worst F1 first) ──────────────────")
        for row in sorted_samples:
            print(
                f"\n  [{row['id']}]  "
                f"F1={row['f1']:.4f}  P={row['precision']:.4f}  R={row['recall']:.4f}  "
                f"TP={row['tp']}  FP={row['fp']}  FN={row['fn']}"
            )
            if row["snippet"]:
                print(f"    TEXT: {row['snippet']}")
            for label, ids in [("TP", row["tp_ids"]), ("FP", row["fp_ids"]), ("FN", row["fn_ids"])]:
                if ids:
                    terms = _resolve(ids, hpo_lookup)
                    for hid, name in terms:
                        print(f"    {label}: {hid}  {name}")

    if args.output:
        import csv

        def _join_ids(ids):
            return ";".join(sorted(ids))

        def _join_names(ids):
            return ";".join(name for _, name in _resolve(ids, hpo_lookup))

        rows = [
            {
                "id": row["id"],
                "precision": row["precision"],
                "recall": row["recall"],
                "f1": row["f1"],
                "tp": row["tp"],
                "fp": row["fp"],
                "fn": row["fn"],
                "n_predicted": row["n_predicted"],
                "n_gold": row["n_gold"],
                "tp_ids": _join_ids(row["tp_ids"]),
                "tp_names": _join_names(row["tp_ids"]),
                "fp_ids": _join_ids(row["fp_ids"]),
                "fp_names": _join_names(row["fp_ids"]),
                "fn_ids": _join_ids(row["fn_ids"]),
                "fn_names": _join_names(row["fn_ids"]),
                "text_snippet": row["snippet"],
            }
            for row in per_sample
        ]

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
                    "tp_ids",
                    "tp_names",
                    "fp_ids",
                    "fp_names",
                    "fn_ids",
                    "fn_names",
                    "text_snippet",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nPer-document results written to {args.output}")

    if args.audit_json:
        payload = {
            "metrics": {
                "precision": strict["precision"],
                "recall": strict["recall"],
                "f1": strict["f1"],
                "tp": strict["tp"],
                "fp": strict["fp"],
                "fn": strict["fn"],
                "documents_scored": strict["n_docs"],
            },
            "documents": [
                {
                    "id": s["id"],
                    "tp": s["tp"],
                    "fp": s["fp"],
                    "fn": s["fn"],
                    "matched": sorted(s["tp_ids"]),
                    "fp_ids": sorted(s["fp_ids"]),
                    "fn_ids": sorted(s["fn_ids"]),
                }
                for s in per_sample
            ],
        }
        args.audit_json.parent.mkdir(parents=True, exist_ok=True)
        args.audit_json.write_text(json.dumps(payload, indent=2))
        print(f"\nAudit JSON written to {args.audit_json}")


if __name__ == "__main__":
    main()
