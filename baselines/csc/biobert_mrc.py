#!/usr/bin/env python3
"""
BioBERT-MRC inference on the CSC phenotype-mining benchmark.

Loads a BioBERT-MRC checkpoint trained on BioLarkGSC
(via baselines/biolarkgsc/biobert_mrc.py), runs it over the 116 CSC clinical
case reports, extracts predicted phenotype strings via MRC span extraction,
maps them to HPO codes via EmbeddingFuzzyMatcher, and evaluates against gold
HP IDs using micro P/R/F1.

Pipeline:
  1. CSCDataset + BioBERTMRCCSCTask → one sample per (document, chunk)
  2. BertSpanNERModel.load_from_checkpoint(checkpoint_dir)
  3. Per sample: model.forward() → start_logits / end_logits
  4. BertSpanNERModel.predict() decodes (label, sub_start, sub_end) spans
  5. Spans mapped to word strings via chunk_sub_offset + subtoken_word_starts
  6. Predictions aggregated per document (union across chunks)
  7. EmbeddingFuzzyMatcher maps entity strings → HP IDs
  8. Micro P/R/F1 vs gold HP IDs, per-doc JSONL + summary JSON saved

Usage (from RDMA repo root):
    python baselines/csc/biobert_mrc.py \\
        --checkpoint_dir results/biolarkgsc/biobert_mrc/best_hf

    # Dry-run (no inference):
    python baselines/csc/biobert_mrc.py --dry_run
"""

import argparse
import bisect
import json
import logging
import pickle
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set

import torch

# ── Path setup ───────────────────────────────────────────────────────────────
_RDMA_ROOT = Path("/home/johnwu3/projects/rare_disease/workspace/repos/RDMA")
_BIOBERT_ROOT = Path("/home/johnwu3/projects/rare_disease/workspace/repos/BioBERT-MRC")
_PYHEALTH_ROOT = Path("/home/johnwu3/projects/rare_disease/workspace/repos/PyHealth")
_RESULTS_DIR = Path("/home/johnwu3/projects/rare_disease/workspace/results")
_DEFAULT_EMBEDDINGS_FILE = str(
    _RDMA_ROOT / "data" / "vector_stores" / "G2GHPO_metadata_medembed.npy"
)
_DEFAULT_CACHE_DIR = "/shared/eng/pyhealth/csc"

sys.path.insert(0, str(_BIOBERT_ROOT))
sys.path.insert(0, str(_PYHEALTH_ROOT))
sys.path.insert(0, str(_RDMA_ROOT))

from rdma.hpo.embedding_fuzzy_matcher import EmbeddingFuzzyMatcher  # noqa: E402
from datasets.csc import CSCDataset  # noqa: E402
from tasks.biobert_mrc_csc import BioBERTMRCCSCTask  # noqa: E402
from models.biobert_span_ner import BertSpanNERModel  # noqa: E402

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def ts(msg: str) -> None:
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S} - {msg}", flush=True)


# ── Evaluation helpers ────────────────────────────────────────────────────────


def compute_metrics(records: List[dict]) -> dict:
    tp = fp = fn = 0
    for rec in records:
        pred = set(h for h in rec["predicted"] if h)
        gold = set(h for h in rec["ground_truth"] if h)
        tp += len(pred & gold)
        fp += len(pred - gold)
        fn += len(gold - pred)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="BioBERT-MRC inference on CSC (BioLarkGSC checkpoint)"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=str(_RESULTS_DIR / "biolarkgsc" / "biobert_mrc" / "best_hf"),
        help="HuggingFace checkpoint directory (output of biolarkgsc/biobert_mrc.py)",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=_DEFAULT_CACHE_DIR,
        help="PyHealth CSC dataset cache directory",
    )
    parser.add_argument("--max_seq_length", type=int, default=256)
    parser.add_argument(
        "--stride_tokens",
        type=int,
        default=64,
        help="Overlap between consecutive chunks in subword tokens (0 = no overlap)",
    )
    parser.add_argument(
        "--gpu_id",
        type=lambda x: None if x.lower() == "none" else int(x),
        default=0,
        metavar="N|none",
    )
    parser.add_argument(
        "--condor",
        action="store_true",
        help="Running under HTCondor: use generic 'cuda' device",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=_RESULTS_DIR / "csc" / "biobert_mrc",
    )
    parser.add_argument(
        "--embeddings_file",
        type=str,
        default=_DEFAULT_EMBEDDINGS_FILE,
        help="Path to HPO .npy embeddings file",
    )
    parser.add_argument(
        "--retriever",
        type=str,
        choices=["fastembed", "sentence_transformer", "medcpt"],
        default="sentence_transformer",
    )
    parser.add_argument(
        "--retriever_model",
        type=str,
        default="BAAI/bge-small-en-v1.5",
    )
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument(
        "--similarity_threshold",
        type=float,
        default=0.9,
        help="Fuzzy match threshold",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Load data and model, inspect one sample, then exit",
    )
    args = parser.parse_args()

    if args.condor:
        device_str = "cuda"
    elif args.gpu_id is not None and torch.cuda.is_available():
        device_str = f"cuda:{args.gpu_id}"
    else:
        device_str = "cpu"
    device = torch.device(device_str)
    ts(f"Device: {device}")

    # ── Dataset ───────────────────────────────────────────────────────
    ts("Loading CSCDataset …")
    dataset = CSCDataset(cache_dir=args.cache_dir)
    ts("Applying BioBERTMRCCSCTask (tokenising + chunking) …")
    task = BioBERTMRCCSCTask(
        max_seq_length=args.max_seq_length,
        stride_tokens=args.stride_tokens,
    )
    samples = dataset.set_task(task)
    ts(f"  Total samples (document × chunk): {len(samples)}")

    # ── Model ─────────────────────────────────────────────────────────
    ts(f"Loading BertSpanNERModel from {args.checkpoint_dir} …")
    model = BertSpanNERModel.load_from_checkpoint(args.checkpoint_dir, dataset=None)
    model.to(device)
    model.eval()

    # ── Dry-run ───────────────────────────────────────────────────────
    if args.dry_run:
        ts("Dry-run: inspecting first sample …")
        sample = samples[0]
        phenotypes = pickle.loads(sample["gold_phenotypes"])
        ts(f"  doc_id         : {sample['patient_id']}")
        ts(f"  input_ids shape: {tuple(sample['input_ids'].shape)}")
        ts(f"  input_len      : {int(sample['input_len'])}")
        ts(f"  gold_hpo_ids   : {[p['hpo_id'] for p in phenotypes]}")
        ts("Dry-run complete.")
        return

    # ── Inference ─────────────────────────────────────────────────────
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pred_strings_per_doc: Dict[str, Set[str]] = defaultdict(set)
    gold_hpo_per_doc: Dict[str, List[str]] = {}

    ts(f"Running inference on {len(samples)} samples …")
    for idx, sample in enumerate(samples):
        doc_id = sample["patient_id"]

        if doc_id not in gold_hpo_per_doc:
            phenotypes = pickle.loads(sample["gold_phenotypes"])
            gold_hpo_per_doc[doc_id] = [
                p["hpo_id"] for p in phenotypes if p.get("hpo_id")
            ]

        input_len = int(sample["input_len"])
        with torch.no_grad():
            outputs = model(
                input_ids=sample["input_ids"].unsqueeze(0).to(device),
                attention_mask=sample["attention_mask"].unsqueeze(0).to(device),
                segment_ids=sample["segment_ids"].unsqueeze(0).to(device),
            )
        spans = BertSpanNERModel.predict(
            outputs["start_logits"], outputs["end_logits"], input_len
        )

        if spans:
            words = pickle.loads(sample["words"])
            sub_starts = pickle.loads(sample["subtoken_word_starts"])
            chunk_offset = int(sample["chunk_sub_offset"])
            for _, local_s, local_e in spans:
                ws = bisect.bisect_right(sub_starts, chunk_offset + local_s) - 1
                we = bisect.bisect_right(sub_starts, chunk_offset + local_e) - 1
                if 0 <= ws <= we < len(words):
                    entity = " ".join(words[ws : we + 1]).lower()
                    if entity:
                        pred_strings_per_doc[doc_id].add(entity)

        if (idx + 1) % 20 == 0:
            ts(f"  Processed {idx + 1}/{len(samples)} samples …")

    # ── HPO code matching ─────────────────────────────────────────────
    ts(f"Initialising EmbeddingFuzzyMatcher ({args.embeddings_file}) …")
    code_matcher = EmbeddingFuzzyMatcher(
        embeddings_file=args.embeddings_file,
        retriever=args.retriever,
        retriever_model=args.retriever_model,
        top_k=args.top_k,
        fuzzy_threshold=args.similarity_threshold,
        device=device_str,
    )

    doc_ids = sorted(gold_hpo_per_doc)
    records = []
    ts(f"Matching entity strings to HPO codes for {len(doc_ids)} documents …")
    for doc_id in doc_ids:
        pred_entities = sorted(pred_strings_per_doc.get(doc_id, set()))
        gold_hpo_ids  = gold_hpo_per_doc.get(doc_id, [])

        predicted_hpo_ids: List[str] = []
        if pred_entities:
            matched = code_matcher.match([{"entity": e} for e in pred_entities])
            predicted_hpo_ids = list(dict.fromkeys(
                m["hp_id"] for m in matched if m.get("hp_id")
            ))

        records.append(
            {
                "id":           doc_id,
                "predicted":    predicted_hpo_ids,
                "ground_truth": gold_hpo_ids,
            }
        )

    # ── Evaluation ────────────────────────────────────────────────────
    metrics = compute_metrics(records)
    ts(
        f"Results — P={metrics['precision']:.4f}  R={metrics['recall']:.4f}"
        f"  F1={metrics['f1']:.4f}  (over {len(doc_ids)} documents)"
    )

    # ── Save ──────────────────────────────────────────────────────────
    jsonl_path   = args.output_dir / "predictions.jsonl"
    summary_path = args.output_dir / "results.json"

    with open(jsonl_path, "w") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    summary = {
        "num_docs":         len(doc_ids),
        "micro_precision":  round(metrics["precision"], 4),
        "micro_recall":     round(metrics["recall"], 4),
        "micro_f1":         round(metrics["f1"], 4),
        "tp":               metrics["tp"],
        "fp":               metrics["fp"],
        "fn":               metrics["fn"],
        "checkpoint_dir":   str(args.checkpoint_dir),
        "stride_tokens":    args.stride_tokens,
    }
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)

    ts(f"Predictions saved to {jsonl_path}")
    ts(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
