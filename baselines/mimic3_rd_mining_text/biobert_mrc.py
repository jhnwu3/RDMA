#!/usr/bin/env python3
"""
BioBERT-MRC inference on MIMIC-III rare-disease notes.

Loads a BioBERT-MRC checkpoint trained on RareDis
(via biobert_mrc_trainer.py), runs it over the MIMIC-III NOTEEVENTS rows
that have human-reviewed rare-disease annotations, extracts predicted entity
strings, and evaluates against the gold standard using set-based P/R/F1.

Pipeline:
  1. MIMIC3Dataset + BioBERTMRCMIMIC3Task → one sample per (note, entity_type, chunk)
  2. BertSpanNERModel.load_from_checkpoint(checkpoint_dir)
  3. Per sample: model.forward() → start_logits / end_logits
  4. BertSpanNERModel.predict() decodes (label, sub_start, sub_end) spans
  5. Spans mapped to word strings via chunk_sub_offset + subtoken_word_starts
  6. Predictions aggregated per note (union across entity types and chunks)
  7. String-level P/R/F1 vs gold, per-note JSONL + summary JSON saved

Usage (from RDMA repo root):
    python baselines/mimic3_rd_mining_text/biobert_mrc.py \\
        --checkpoint_dir results/raredis/biobert_mrc_trainer/best_hf

    # Dry-run (no inference):
    python baselines/mimic3_rd_mining_text/biobert_mrc.py --dry_run
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
_DEFAULT_MIMIC3_ROOT = (
    "/srv/local/data/physionet.org/files/mimic-iii-clinical-database-1.4/"
)
_DEFAULT_MIMIC3_CACHE_DIR = "/shared/eng/pyhealth/mimic3"
_DEFAULT_ORPHA_EMBEDDINGS_FILE = str(
    _RDMA_ROOT / "data" / "vector_stores" / "rd_orpha_medembed.npy"
)

sys.path.insert(0, str(_BIOBERT_ROOT))
sys.path.insert(0, str(_PYHEALTH_ROOT))
sys.path.insert(0, str(_RDMA_ROOT))

from rdma.hpo.embedding_fuzzy_matcher import EmbeddingFuzzyMatcher  # noqa: E402
from pyhealth.datasets import MIMIC3Dataset  # noqa: E402
from tasks.biobert_mrc_mimic3 import BioBERTMRCMIMIC3Task  # noqa: E402
from models.biobert_span_ner import BertSpanNERModel  # noqa: E402

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def ts(msg: str) -> None:
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S} - {msg}", flush=True)


# ── String-level evaluation ──────────────────────────────────────────────────


def _note_scores(pred_set: Set[str], gold_set: Set[str]):
    if not gold_set and not pred_set:
        return 1.0, 1.0, 1.0
    if not gold_set:
        return 0.0, 1.0, 0.0
    if not pred_set:
        return 1.0, 0.0, 0.0
    tp = len(pred_set & gold_set)
    p = tp / len(pred_set)
    r = tp / len(gold_set)
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


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
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=_DEFAULT_MIMIC3_CACHE_DIR,
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=str(_RESULTS_DIR / "raredis" / "biobert_mrc_trainer" / "best_hf"),
    )
    parser.add_argument("--max_seq_length", type=int, default=256)
    parser.add_argument(
        "--stride_tokens",
        type=int,
        default=64,
        help="Overlap between consecutive chunks in subword tokens",
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
        help="Load data and model, inspect one sample, then exit",
    )
    parser.add_argument(
        "--skip_code_matching",
        action="store_true",
        help="Skip ORPHA code matching; produce text-eval output only",
    )
    parser.add_argument(
        "--orpha_embeddings_file",
        type=str,
        default=_DEFAULT_ORPHA_EMBEDDINGS_FILE,
    )
    parser.add_argument("--retriever", type=str, default="sentence_transformer")
    parser.add_argument(
        "--retriever_model",
        type=str,
        default="abhinand/MedEmbed-small-v0.1",
    )
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--fuzzy_threshold", type=float, default=0.85)
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
    ts("Applying BioBERTMRCMIMIC3Task (tokenising + chunking) …")
    task = BioBERTMRCMIMIC3Task(
        max_seq_length=args.max_seq_length,
        stride_tokens=args.stride_tokens,
    )
    samples = mimic3_dataset.set_task(task)
    ts(f"  Total samples (note × entity_type × chunk): {len(samples)}")

    # ── Model ─────────────────────────────────────────────────────────
    ts(f"Loading BertSpanNERModel from {args.checkpoint_dir} …")
    model = BertSpanNERModel.load_from_checkpoint(args.checkpoint_dir, dataset=None)
    model.to(device)
    model.eval()

    # ── Dry-run ───────────────────────────────────────────────────────
    if args.dry_run:
        ts("Dry-run: inspecting first RAREDISEASE sample …")
        sample = next(s for s in samples if s["entity_type"] in _EVAL_ENTITY_TYPES)
        ts(f"  note_id        : {sample['note_id']}")
        ts(f"  entity_type    : {sample['entity_type']}")
        ts(f"  input_ids shape: {tuple(sample['input_ids'].shape)}")
        ts(f"  input_len      : {int(sample['input_len'])}")
        ts(f"  gold_entities  : {pickle.loads(sample['gold_entities'])}")
        ts("Dry-run complete.")
        return

    # ── Inference ─────────────────────────────────────────────────────
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pred_per_note: Dict[str, Set[str]] = defaultdict(set)
    gold_per_note: Dict[str, Set[str]] = defaultdict(set)

    ts(f"Running inference on {len(samples)} samples …")
    for idx, sample in enumerate(samples):
        note_id = sample["note_id"]
        entity_type = sample["entity_type"]
        gold = pickle.loads(sample["gold_entities"])
        gold_per_note[note_id].update(gold)

        if entity_type not in _EVAL_ENTITY_TYPES:
            continue

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
                        pred_per_note[note_id].add(entity)

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
                "note_id":    note_id,
                "gold":       sorted(gold_set),
                "predicted":  sorted(pred_set),
                "precision":  round(p, 4),
                "recall":     round(r, 4),
                "f1":         round(f1, 4),
            }
        )

    macro_p  = sum(per_note_p)  / len(per_note_p)  if per_note_p  else 0.0
    macro_r  = sum(per_note_r)  / len(per_note_r)  if per_note_r  else 0.0
    macro_f1 = sum(per_note_f1) / len(per_note_f1) if per_note_f1 else 0.0

    summary = {
        "num_notes":       len(note_ids),
        "macro_precision": round(macro_p, 4),
        "macro_recall":    round(macro_r, 4),
        "macro_f1":        round(macro_f1, 4),
        "checkpoint_dir":  str(args.checkpoint_dir),
        "stride_tokens":   args.stride_tokens,
    }

    ts(
        f"Results — P={macro_p:.4f}  R={macro_r:.4f}  F1={macro_f1:.4f}"
        f"  (over {len(note_ids)} notes)"
    )

    # ── Save (text eval) ──────────────────────────────────────────────
    jsonl_path   = args.output_dir / "per_note_predictions.jsonl"
    summary_path = args.output_dir / "results.json"

    with open(jsonl_path, "w") as fh:
        for row in per_note_rows:
            fh.write(json.dumps(row) + "\n")

    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)

    ts(f"Per-note predictions saved to {jsonl_path}")
    ts(f"Summary saved to {summary_path}")

    # ── Code matching (embedding + fuzzy → ORPHA IDs) ─────────────────
    if not args.skip_code_matching:
        ts(f"Initialising EmbeddingFuzzyMatcher ({args.orpha_embeddings_file}) …")
        code_matcher = EmbeddingFuzzyMatcher(
            embeddings_file=args.orpha_embeddings_file,
            retriever=args.retriever,
            retriever_model=args.retriever_model,
            top_k=args.top_k,
            fuzzy_threshold=args.fuzzy_threshold,
            device=device_str,
        )

        code_jsonl_path = args.output_dir / "per_note_code_predictions.jsonl"
        ts(f"Writing code-eval JSONL to {code_jsonl_path} …")
        with open(code_jsonl_path, "w") as fh:
            for note_id in note_ids:
                pred_entities = sorted(pred_per_note.get(note_id, set()))
                matched = code_matcher.match([{"entity": e} for e in pred_entities])
                orpha_ids = list(dict.fromkeys(
                    m["hp_id"] for m in matched if m.get("hp_id")
                ))
                fh.write(
                    json.dumps(
                        {
                            "id":                  note_id,
                            "predicted":           pred_entities,
                            "predicted_orpha_ids": orpha_ids,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        ts(f"Code-eval predictions saved to {code_jsonl_path}")


if __name__ == "__main__":
    main()
