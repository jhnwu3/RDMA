#!/usr/bin/env python3
"""
BioBERT-MRC inference on MIMIC-III rare-disease notes.

Loads a BioBERT-MRC checkpoint trained on RareDis
(via biobert_mrc_trainer.py), runs it over the MIMIC-III NOTEEVENTS rows
that have human-reviewed rare-disease annotations, extracts predicted entity
strings, and evaluates against the gold standard using set-based P/R/F1.

Pipeline:
  1. MIMIC3Dataset + BioBERTMRCMIMIC3Task → one sample per (note, entity_type)
  2. BertSpanNERModel.load_from_checkpoint(checkpoint_dir)
  3. BioBERTMRCNERProcessor tokenises the full note and chunks it into
     overlapping windows (at word boundaries) → [num_chunks, max_seq_length]
  4. All chunks of one note are forwarded in a single batched GPU call
  5. bert_extract_item decodes predicted (label, sub_start, sub_end) spans
     per chunk
  6. _spans_to_strings maps subword indices → original word strings
  7. Predictions aggregated per note (union across entity types & chunks)
  8. String-level P/R/F1 vs gold, per-note JSONL + summary JSON saved

Usage (from RDMA repo root):
    python baselines/biobert_mrc_mimic3.py \\
        --checkpoint_dir results/raredis/biobert_mrc_trainer/best_hf

    # Dry-run (no inference):
    python baselines/biobert_mrc_mimic3.py --dry_run
"""

import argparse
import json
import logging
import pickle
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple

import torch

# ── Path setup ───────────────────────────────────────────────────────────────
_RDMA_ROOT = Path("/home/johnwu3/projects/rare_disease/workspace/repos/RDMA")
_BIOBERT_ROOT = Path("/home/johnwu3/projects/rare_disease/workspace/repos/BioBERT-MRC")
_PYHEALTH_ROOT = Path("/home/johnwu3/projects/rare_disease/workspace/repos/PyHealth")
_RESULTS_DIR = Path("/home/johnwu3/projects/rare_disease/workspace/results")
_DEFAULT_MIMIC3_ROOT = (
    "/srv/local/data/physionet.org/files/mimic-iii-clinical-database-1.4/"
)
_DEFAULT_MIMIC3_CACHE_DIR = "/shared/eng/pyhealth/mimic3"

sys.path.insert(0, str(_BIOBERT_ROOT))
sys.path.insert(0, str(_PYHEALTH_ROOT))
sys.path.insert(0, str(_RDMA_ROOT))  # inserted last → highest priority

from pyhealth.datasets import MIMIC3Dataset  # noqa: E402
from tasks.biobert_mrc_mimic3 import BioBERTMRCMIMIC3Task  # noqa: E402
from processors.biobert_mrc_ner_processor import (  # noqa: E402
    BioBERTMRCNERProcessor,
)
from models.biobert_span_ner import BertSpanNERModel  # noqa: E402

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def ts(msg: str) -> None:
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S} - {msg}", flush=True)


# ── Inlined from BioBERT-MRC/processors/utils_ner.py ─────────────────────────
def bert_extract_item(start_logits, end_logits, input_len):
    """Extract non-overlapping (label, start, end) spans from logits."""
    start_pred = torch.argmax(start_logits, -1).cpu().numpy()[0][1 : input_len - 1]
    end_pred = torch.argmax(end_logits, -1).cpu().numpy()[0][1 : input_len - 1]
    spans = []
    for i, s_l in enumerate(start_pred):
        if s_l == 0:
            continue
        for j, e_l in enumerate(end_pred[i:]):
            if s_l == e_l:
                spans.append((s_l, i, i + j))
                break
    result = []
    if len(spans) < 2:
        result = spans
    else:
        for i in range(len(spans) - 1):
            if spans[i][2] < spans[i + 1][1]:
                result.append(spans[i])
        result.append(spans[-1])
    return result


# ── Span → text string ───────────────────────────────────────────────────────


def _word_offsets(text: str) -> Tuple[List[str], List[int]]:
    """Whitespace-split text, returning (words, char_starts)."""
    words, starts = [], []
    pos, n = 0, len(text)
    while pos < n:
        while pos < n and text[pos].isspace():
            pos += 1
        if pos >= n:
            break
        word_start = pos
        while pos < n and not text[pos].isspace():
            pos += 1
        words.append(text[word_start:pos])
        starts.append(word_start)
    return words, starts


def _spans_to_strings(
    tokenizer,
    text: str,
    spans: List[Tuple[int, int, int]],
    max_text_tokens: int,
) -> List[str]:
    """Convert predicted subword span tuples to original word strings.

    Args:
        tokenizer: BertTokenizer used by the processor.
        text: Chunk text (already sliced to the chunk's word range).
        spans: List of (label_id, sub_start, sub_end) in 0-based
               text-subword space (as returned by ``bert_extract_item``).
        max_text_tokens: Maximum text subword tokens that fit in the
               sequence (= max_seq_length - len(query_tokens) - 3).

    Returns:
        List of extracted entity strings (joined words, lowercased).
    """
    if not spans:
        return []

    words, _ = _word_offsets(text)

    # Rebuild subtoken structure (same logic as BioBERTMRCNERProcessor)
    word_subtokens: List[List[str]] = []
    for word in words:
        subs = tokenizer.tokenize(word)
        word_subtokens.append(subs if subs else [tokenizer.unk_token])

    subtoken_word_starts: List[int] = []
    cum = 0
    for subs in word_subtokens:
        subtoken_word_starts.append(cum)
        cum += len(subs)
    total_text_subtokens = cum

    # Build reverse map: subtoken_index → word_index
    sub_to_word: List[int] = []
    for wi, subs in enumerate(word_subtokens):
        sub_to_word.extend([wi] * len(subs))

    cap = min(total_text_subtokens, max_text_tokens)
    results: List[str] = []
    for _label_id, sub_start, sub_end in spans:
        if sub_start >= cap:
            continue
        if sub_end >= cap:
            sub_end = cap - 1
        word_start = sub_to_word[sub_start]
        word_end = sub_to_word[sub_end]
        entity_str = " ".join(words[word_start : word_end + 1]).lower()
        if entity_str:
            results.append(entity_str)

    return results


# ── String-level evaluation ──────────────────────────────────────────────────


def _note_scores(pred_set: Set[str], gold_set: Set[str]) -> Tuple[float, float, float]:
    """Precision, recall, F1 for a single note (exact string match)."""
    if not gold_set and not pred_set:
        return 1.0, 1.0, 1.0
    if not gold_set:
        return 0.0, 1.0, 0.0  # false positives only
    if not pred_set:
        return 1.0, 0.0, 0.0  # false negatives only
    tp = len(pred_set & gold_set)
    p = tp / len(pred_set)
    r = tp / len(gold_set)
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


# ── Entity types to run inference for ───────────────────────────────────────
_EVAL_ENTITY_TYPES = {"RAREDISEASE", "SKINRAREDISEASE"}

# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="BioBERT-MRC inference on MIMIC-III rare-disease notes"
    )
    parser.add_argument(
        "--mimic3_root",
        type=str,
        default=_DEFAULT_MIMIC3_ROOT,
        help="Path to the MIMIC-III v1.4 dataset root",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=_DEFAULT_MIMIC3_CACHE_DIR,
        help="Directory for PyHealth MIMIC-III dataset cache (default: %(default)s)",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=str(_RESULTS_DIR / "raredis" / "biobert_mrc_trainer" / "best_hf"),
        help=("HuggingFace checkpoint directory " "(output of biobert_mrc_trainer.py)"),
    )
    parser.add_argument("--max_seq_length", type=int, default=256)
    parser.add_argument(
        "--stride_tokens",
        type=int,
        default=64,
        help=(
            "Overlap between consecutive text chunks, in subword tokens. "
            "Set to 0 to disable overlap."
        ),
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
        default=_RESULTS_DIR / "mimic3" / "biobert_mrc",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Load data and model, process one sample, then exit",
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
    ts(f"Loading MIMIC3Dataset from {args.mimic3_root} …")
    mimic3_dataset = MIMIC3Dataset(
        root=args.mimic3_root,
        tables=["noteevents"],
        cache_dir=args.cache_dir,
        dev=False,
        num_workers=4,
    )
    ts("Applying BioBERTMRCMIMIC3Task …")
    task = BioBERTMRCMIMIC3Task()
    samples = mimic3_dataset.set_task(task)
    ts(f"  Total samples (note x entity_type): {len(samples)}")

    # ── Processor + model ─────────────────────────────────────────────
    ts(f"Loading processor from {args.checkpoint_dir} …")
    processor = BioBERTMRCNERProcessor(
        model_name_or_path=args.checkpoint_dir,
        max_seq_length=args.max_seq_length,
        stride_tokens=args.stride_tokens,
    )

    ts(f"Loading BertSpanNERModel from {args.checkpoint_dir} …")
    model = BertSpanNERModel.load_from_checkpoint(args.checkpoint_dir, dataset=None)
    model.to(device)
    model.eval()

    # ── Dry-run ───────────────────────────────────────────────────────
    if args.dry_run:
        ts("Dry-run: processing first sample …")
        sample = next(
            s for s in samples
            if pickle.loads(s["entity_type"]) in _EVAL_ENTITY_TYPES
        )
        feat = processor.process(sample)
        num_chunks = feat["input_ids"].shape[0]
        ts(f"  note_id        : {sample['note_id']}")
        ts(f"  entity_type    : {pickle.loads(sample['entity_type'])}")
        ts(f"  num_chunks     : {num_chunks}")
        ts(f"  input_ids shape: {tuple(feat['input_ids'].shape)}")
        ts(f"  input_len      : {feat['input_len']}")
        gold = pickle.loads(sample["gold_entities"])
        ts(f"  gold_entities  : {gold}")
        ts("Dry-run complete.")
        return

    # ── Inference ─────────────────────────────────────────────────────
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pred_per_note: Dict[str, Set[str]] = defaultdict(set)
    gold_per_note: Dict[str, Set[str]] = defaultdict(set)

    ts(f"Running inference on {len(samples)} samples …")
    for idx, sample in enumerate(samples):
        note_id = sample["note_id"]
        entity_type = pickle.loads(sample["entity_type"])
        gold = pickle.loads(sample["gold_entities"])
        gold_per_note[note_id].update(gold)

        if entity_type not in _EVAL_ENTITY_TYPES:
            continue

        feat = processor.process(sample)
        # feat["input_ids"]: [num_chunks, max_seq_length]

        with torch.no_grad():
            outputs = model(
                input_ids=feat["input_ids"].to(device),
                attention_mask=feat["attention_mask"].to(device),
                segment_ids=feat["segment_ids"].to(device),
            )
        # outputs["start_logits"]: [num_chunks, max_seq_length, num_labels]

        query_tokens = (
            processor.tokenizer.tokenize(pickle.loads(sample["mrc_query"])) or []
        )
        max_text_tokens = args.max_seq_length - len(query_tokens) - 3

        num_chunks = feat["input_ids"].shape[0]
        for i in range(num_chunks):
            spans = bert_extract_item(
                outputs["start_logits"][i : i + 1],
                outputs["end_logits"][i : i + 1],
                feat["input_len"][i],
            )
            start_w, end_w = feat["chunk_word_ranges"][i]
            chunk_text = " ".join(feat["words"][start_w:end_w])
            pred_strings = _spans_to_strings(
                processor.tokenizer, chunk_text, spans, max_text_tokens
            )
            pred_per_note[note_id].update(pred_strings)

        if (idx + 1) % 500 == 0:
            ts(f"  Processed {idx + 1}/{len(samples)} samples …")

    # ── Evaluation ────────────────────────────────────────────────────
    ts("Computing evaluation metrics …")
    note_ids = sorted(set(gold_per_note) | set(pred_per_note))
    per_note_p, per_note_r, per_note_f1 = [], [], []
    per_note_rows = []

    for note_id in note_ids:
        gold_set = gold_per_note.get(note_id, set())
        pred_set = pred_per_note.get(note_id, set())
        p, r, f1 = _note_scores(pred_set, gold_set)
        per_note_p.append(p)
        per_note_r.append(r)
        per_note_f1.append(f1)
        per_note_rows.append(
            {
                "note_id": note_id,
                "gold": sorted(gold_set),
                "predicted": sorted(pred_set),
                "precision": round(p, 4),
                "recall": round(r, 4),
                "f1": round(f1, 4),
            }
        )

    macro_p = sum(per_note_p) / len(per_note_p) if per_note_p else 0.0
    macro_r = sum(per_note_r) / len(per_note_r) if per_note_r else 0.0
    macro_f1 = sum(per_note_f1) / len(per_note_f1) if per_note_f1 else 0.0

    summary = {
        "num_notes": len(note_ids),
        "macro_precision": round(macro_p, 4),
        "macro_recall": round(macro_r, 4),
        "macro_f1": round(macro_f1, 4),
        "checkpoint_dir": str(args.checkpoint_dir),
        "stride_tokens": args.stride_tokens,
    }

    ts(
        f"Results — P={macro_p:.4f}  R={macro_r:.4f}  F1={macro_f1:.4f}"
        f"  (over {len(note_ids)} notes)"
    )

    # ── Save ──────────────────────────────────────────────────────────
    jsonl_path = args.output_dir / "per_note_predictions.jsonl"
    summary_path = args.output_dir / "results.json"

    with open(jsonl_path, "w") as fh:
        for row in per_note_rows:
            fh.write(json.dumps(row) + "\n")

    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)

    ts(f"Per-note predictions saved to {jsonl_path}")
    ts(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
