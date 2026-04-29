#!/usr/bin/env python3
"""
Stanza i2b2 NER baseline for the RareDis rare-disease NER benchmark.

Pipeline:
  1. RareDisDataset + RareDisNER → one sample per annotated document
  2. Stanza i2b2 NER extracts PROBLEM entities (already filtered internally)
  3. Extracted entity strings are compared against gold annotations

Output JSONL format:
    {"id": "<doc_id>", "predicted": [...], "ground_truth": [...]}

Usage (from RDMA repo root):
    python baselines/raredis/i2b2.py

    # Dry-run (no full inference):
    python baselines/raredis/i2b2.py --dry_run
"""

import argparse
import json
import pickle
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import List

from tqdm import tqdm

# ── Path setup ───────────────────────────────────────────────────────────────
_RDMA_ROOT = Path("/home/johnwu3/projects/rare_disease/workspace/repos/RDMA")
_RESULTS_DIR = Path("/home/johnwu3/projects/rare_disease/workspace/results")

sys.path.insert(0, str(_RDMA_ROOT))

from rdma.hporag.entity import StanzaEntityExtractor  # noqa: E402

from datasets.raredis import RareDisDataset  # noqa: E402
from tasks.raredis import RareDisNER  # noqa: E402


def ts(msg: str) -> None:
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S} - {msg}", flush=True)


def load_done_ids(path: Path) -> set:
    done = set()
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    done.add(json.loads(line).get("id"))
                except Exception:
                    continue
    return done


def compute_metrics(records: List[dict]) -> dict:
    tp = fp = fn = 0
    for rec in records:
        predicted = set(h for h in rec["predicted"] if h)
        gold = set(h for h in rec["ground_truth"] if h)
        tp += len(predicted & gold)
        fp += len(predicted - gold)
        fn += len(gold - predicted)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "n_docs": len(records),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stanza i2b2 NER baseline on RareDis rare-disease NER benchmark"
    )
    parser.add_argument(
        "--stanza_device",
        type=str,
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device for Stanza pipeline (default: %(default)s)",
    )
    parser.add_argument(
        "--condor",
        action="store_true",
        help="Running under HTCondor: use generic 'cuda' device if available",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSONL path (default: <results_dir>/raredis/i2b2_predictions.jsonl)",
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume from existing output file"
    )
    parser.add_argument("--debug", action="store_true", help="Enable verbose debug logging")
    parser.add_argument("--dev", action="store_true", help="Process first 2 samples only")
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=20,
        help="Log checkpoint every N samples (default: %(default)s)",
    )
    args = parser.parse_args()

    import torch

    if args.condor:
        stanza_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        stanza_device = args.stanza_device

    output = args.output or (_RESULTS_DIR / "raredis" / "i2b2_predictions.jsonl")

    ts(f"Stanza device     : {stanza_device}")
    ts(f"Output            : {output}")
    ts(f"Resume            : {args.resume}")
    ts(f"Debug             : {args.debug}")
    ts(f"Dev mode          : {args.dev}")

    # ── Dataset ───────────────────────────────────────────────────────────────
    ts("Loading RareDisDataset ...")
    dataset = RareDisDataset(num_workers=4)
    samples = dataset.set_task(RareDisNER(), num_workers=4)
    ts(f"  {len(samples)} samples loaded")

    first = next(iter(samples))
    ts(f"  Sample preview — id: {first['patient_id']!r}")
    ts(f"    text[:120]: {pickle.loads(first['text'])[:120]!r}")
    ann_preview = pickle.loads(first["annotations"])[:5]
    ts(f"    annotations (first 5): {ann_preview}")

    # ── Stanza pipeline ───────────────────────────────────────────────────────
    ts("Loading Stanza i2b2 pipeline ...")
    import stanza

    stanza.download("en", package="mimic", processors={"ner": "i2b2"})
    stanza_pipeline = stanza.Pipeline(
        "en",
        package="mimic",
        processors={"ner": "i2b2"},
        device=stanza_device,
    )
    extractor = StanzaEntityExtractor(stanza_pipeline)
    ts("Stanza i2b2 pipeline ready.")

    # ── Dry-run ───────────────────────────────────────────────────────────────
    if args.dev and not args.resume:
        ts("Dev mode: processing first 2 samples only.")

    done_ids = load_done_ids(output) if args.resume else set()
    if args.resume:
        ts(f"Resuming — {len(done_ids)} already done")

    output.parent.mkdir(parents=True, exist_ok=True)
    out_f = open(output, "a" if args.resume else "w", encoding="utf-8")

    try:
        records = []
        run_samples = samples.subset(slice(0, 2)) if args.dev else samples
        for i, sample in enumerate(
            tqdm(run_samples, total=len(run_samples), desc="RareDis-i2b2")
        ):
            try:
                doc_id = sample["patient_id"]
                text = pickle.loads(sample["text"])
                ground_truth = [e.lower() for e in pickle.loads(sample["annotations"]) if e]
            except Exception as e:
                ts(f"  SKIP sample {i} (data error): {e}")
                if args.debug:
                    traceback.print_exc()
                continue

            if doc_id in done_ids:
                continue

            predicted = []
            try:
                predicted = extractor.extract_entities(text)
                if args.debug:
                    ts(f"  [{doc_id}] n_pred={len(predicted)}  n_gold={len(ground_truth)}")
            except Exception as e:
                ts(f"  ERROR [{doc_id}]: {e}")
                if args.debug:
                    traceback.print_exc()

            records.append({"predicted": predicted, "ground_truth": ground_truth})
            out_f.write(
                json.dumps(
                    {"id": doc_id, "predicted": predicted, "ground_truth": ground_truth},
                    ensure_ascii=False,
                )
                + "\n"
            )
            out_f.flush()

            if (i + 1) % args.checkpoint_interval == 0:
                ts(f"Checkpoint {i + 1}/{len(run_samples)}")
    finally:
        out_f.close()

    if records:
        metrics = compute_metrics(records)
        ts("── Evaluation ───────────────────────────────────────────────")
        ts(f"  Precision : {metrics['precision']:.4f}")
        ts(f"  Recall    : {metrics['recall']:.4f}")
        ts(f"  F1        : {metrics['f1']:.4f}")
        ts(
            f"  TP={metrics['tp']}  FP={metrics['fp']}  FN={metrics['fn']}  Docs={metrics['n_docs']}"
        )

    ts(f"Done → {output}")


if __name__ == "__main__":
    main()
