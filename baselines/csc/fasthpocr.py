#!/usr/bin/env python3
"""
FastHPOCR baseline for the CSC phenotype-mining benchmark.

Pipeline:
  HPOAnnotator (rule-based, no LLM) directly annotates text with HPO codes.

Output JSONL format (matches scripts/csc/run_hpo.py):
    {"id": "<doc_id>", "predicted": [...], "ground_truth": [...], "timing": {...}}

Usage (from RDMA repo root):
  python baselines/csc/fasthpocr.py
"""

import argparse
import json
import sys
import os
import pickle
import time
import traceback
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

_RDMA_ROOT = Path("/home/johnwu3/projects/rare_disease/workspace/repos/RDMA")
_FASTHPOCR_ROOT = Path("/home/johnwu3/projects/rare_disease/workspace/repos/fast_hpo_cr")
sys.path.insert(0, str(_RDMA_ROOT))
sys.path.insert(0, str(_FASTHPOCR_ROOT))

from HPOAnnotator import HPOAnnotator  # noqa: E402

from datasets.csc import CSCDataset  # noqa: E402
from tasks.csc import CSCPhenotypeMining  # noqa: E402

_RESULTS_DIR = Path("/home/johnwu3/projects/rare_disease/workspace/results")
_DEFAULT_HPO_INDEX = str(_FASTHPOCR_ROOT / "hp.index")
_DEFAULT_DATASET_CACHE_DIR = "/shared/eng/pyhealth/csc"


def ts(msg):
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S} - {msg}", flush=True)


def load_done_ids(path):
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        done.add(json.loads(line)["id"])
                    except Exception:
                        pass
    return done


def _uri_to_hpo_id(uri: str) -> str:
    """Convert OBO URI (http://.../HP_XXXXXXX) or HP_XXXXXXX to HP:XXXXXXX."""
    if "HP_" in uri:
        return "HP:" + uri.split("HP_")[-1]
    return uri


def compute_metrics(records: list) -> dict:
    tp = fp = fn = 0
    for rec in records:
        predicted = set(h for h in rec["predicted"] if h)
        gold = set(h for h in rec["ground_truth"] if h)
        tp += len(predicted & gold)
        fp += len(predicted - gold)
        fn += len(gold - predicted)
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
        "n_docs": len(records),
    }


def main():
    parser = argparse.ArgumentParser(
        description="FastHPOCR baseline on CSC phenotype-mining benchmark"
    )
    parser.add_argument(
        "--hpo_index",
        type=str,
        default=_DEFAULT_HPO_INDEX,
        help="Path to FastHPOCR hp.index file (default: %(default)s)",
    )
    parser.add_argument(
        "--dataset_cache_dir",
        type=str,
        default=_DEFAULT_DATASET_CACHE_DIR,
        help="PyHealth dataset cache directory (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output JSONL path "
            "(default: <results_dir>/csc/fasthpocr_predictions.jsonl)"
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output file",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable verbose debug logging"
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Dev mode: process only the first 2 samples",
    )
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=20,
        help="Log a checkpoint every N samples (default: %(default)s)",
    )
    args = parser.parse_args()

    output = args.output or (
        _RESULTS_DIR / "csc" / "fasthpocr_predictions.jsonl"
    )

    ts(f"HPO index         : {args.hpo_index}")
    ts(f"Dataset cache dir : {args.dataset_cache_dir}")
    ts(f"Output            : {output}")
    ts(f"Resume            : {args.resume}")
    ts(f"Debug             : {args.debug}")
    ts(f"Dev mode          : {args.dev}")
    ts(f"Checkpoint every  : {args.checkpoint_interval}")

    # ── Dataset ───────────────────────────────────────────────────────────
    ts("Loading CSCDataset...")
    dataset = CSCDataset(cache_dir=args.dataset_cache_dir)
    samples = dataset.set_task(CSCPhenotypeMining())
    ts(f"  {len(samples)} samples loaded")

    first = next(iter(samples))
    ts(f"  Sample preview — id: {first['patient_id']!r}")
    ts(f"    text[:120]: {pickle.loads(first['text'])[:120]!r}")
    ph_preview = pickle.loads(first["phenotypes"])[:3]
    n_ph = len(pickle.loads(first["phenotypes"]))
    ts(f"    phenotypes ({n_ph}): {ph_preview}")

    # ── Annotator ─────────────────────────────────────────────────────────
    ts(f"Loading HPOAnnotator from {args.hpo_index}...")
    annotator = HPOAnnotator(args.hpo_index)
    ts("  HPOAnnotator ready")

    # ── Run ───────────────────────────────────────────────────────────────
    done_ids = load_done_ids(output) if args.resume else set()
    if args.resume:
        ts(f"Resuming – {len(done_ids)} already done")

    output.parent.mkdir(parents=True, exist_ok=True)
    out_f = open(output, "a" if args.resume else "w", encoding="utf-8")

    try:
        timings: list = []
        records: list = []
        run_samples = samples.subset(slice(0, 2)) if args.dev else samples
        for i, sample in enumerate(
            tqdm(run_samples, total=len(run_samples), desc="CSC-FastHPOCR")
        ):
            try:
                doc_id = sample["patient_id"]
                text = pickle.loads(sample["text"])
                ground_truth = [
                    p["hpo_id"]
                    for p in pickle.loads(sample["phenotypes"])
                    if p.get("hpo_id")
                ]
            except Exception as e:
                ts(f"  SKIP sample {i} (data error): {e}")
                if args.debug:
                    traceback.print_exc()
                continue

            if doc_id in done_ids:
                continue

            annot_s = 0.0
            try:
                t0 = time.perf_counter()
                annotations = annotator.annotate(text)
                annot_s = time.perf_counter() - t0

                predicted = list(
                    {_uri_to_hpo_id(a.getHPOUri()) for a in annotations if a.getHPOUri()}
                )

                if args.debug:
                    ts(f"  [{doc_id}] annotated {len(annotations)} → {len(predicted)} unique HP codes")

                ts(f"  [{doc_id}] annotate={annot_s:.2f}s  n_pred={len(predicted)}")
            except Exception as e:
                ts(f"  ERROR [{doc_id}]: {e}")
                if args.debug:
                    traceback.print_exc()
                predicted = []

            timings.append(annot_s)
            records.append({"predicted": predicted, "ground_truth": ground_truth})
            out_f.write(
                json.dumps(
                    {
                        "id": doc_id,
                        "predicted": predicted,
                        "ground_truth": ground_truth,
                        "timing": {"annotation_s": round(annot_s, 3)},
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            out_f.flush()

            if (i + 1) % args.checkpoint_interval == 0:
                ts(f"Checkpoint {i + 1}/{len(samples)}")
    finally:
        out_f.close()

    if timings:
        avg_a = sum(timings) / len(timings)
        ts("── Timing summary ──────────────────────────────────────────")
        ts(f"  Samples           : {len(timings)}")
        ts(f"  Avg annotation    : {avg_a:.2f}s")

    if records:
        metrics = compute_metrics(records)
        ts("── Code evaluation ─────────────────────────────────────────")
        ts(f"  Precision : {metrics['precision']:.4f}")
        ts(f"  Recall    : {metrics['recall']:.4f}")
        ts(f"  F1        : {metrics['f1']:.4f}")
        ts(f"  TP={metrics['tp']}  FP={metrics['fp']}  FN={metrics['fn']}  Docs={metrics['n_docs']}")

    ts(f"Done → {output}")


if __name__ == "__main__":
    main()
