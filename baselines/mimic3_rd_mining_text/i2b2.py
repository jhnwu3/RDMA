#!/usr/bin/env python3
"""
Stanza i2b2 NER baseline for MIMIC-III rare-disease mining.

Pipeline:
  1. MIMIC3Dataset + MIMIC3RDMiningText → one sample per annotated note
  2. Stanza i2b2 NER extracts PROBLEM entities from each note
  3. EmbeddingFuzzyMatcher maps extracted entities to ORPHA codes
     using embedding retrieval + fuzzy matching (no LLM required)

Output:
    per_note_predictions.jsonl  — string-level predictions + gold
    results.json                — macro P/R/F1 summary
    per_note_code_predictions.jsonl — ORPHA code predictions (unless --skip_code_matching)

Usage (from RDMA repo root):
    python baselines/mimic3_rd_mining_text/i2b2.py

    # Dry-run (no full inference):
    python baselines/mimic3_rd_mining_text/i2b2.py --dry_run
"""

import argparse
import json
import pickle
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple

# ── Path setup ───────────────────────────────────────────────────────────────
_RDMA_ROOT = Path("/home/johnwu3/projects/rare_disease/workspace/repos/RDMA")
_PYHEALTH_ROOT = Path("/home/johnwu3/projects/rare_disease/workspace/repos/PyHealth")
_RESULTS_DIR = Path("/home/johnwu3/projects/rare_disease/workspace/results")
_DEFAULT_MIMIC3_ROOT = (
    "/srv/local/data/physionet.org/files/mimic-iii-clinical-database-1.4/"
)
_DEFAULT_MIMIC3_CACHE_DIR = "/shared/eng/pyhealth/mimic3"
_DEFAULT_ORPHA_EMBEDDINGS_FILE = str(
    _RDMA_ROOT / "data" / "vector_stores" / "rd_orpha_medembed.npy"
)

sys.path.insert(0, str(_PYHEALTH_ROOT))
sys.path.insert(0, str(_RDMA_ROOT))

from rdma.hpo.embedding_fuzzy_matcher import EmbeddingFuzzyMatcher  # noqa: E402
from rdma.hporag.entity import StanzaEntityExtractor  # noqa: E402

from pyhealth.datasets import MIMIC3Dataset  # noqa: E402
from tasks.mimic3_rd_mining_text import MIMIC3RDMiningText  # noqa: E402


def ts(msg: str) -> None:
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S} - {msg}", flush=True)


def _note_scores(pred_set: Set[str], gold_set: Set[str]) -> Tuple[float, float, float]:
    """Precision, recall, F1 for a single note (exact string match)."""
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stanza i2b2 NER baseline on MIMIC-III rare-disease notes"
    )
    parser.add_argument(
        "--mimic3_root",
        type=str,
        default=_DEFAULT_MIMIC3_ROOT,
        help="Path to the MIMIC-III v1.4 dataset root (default: %(default)s)",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=_DEFAULT_MIMIC3_CACHE_DIR,
        help="Directory for PyHealth MIMIC-III dataset cache (default: %(default)s)",
    )
    parser.add_argument(
        "--stanza_device",
        type=str,
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device for Stanza pipeline (default: %(default)s)",
    )
    parser.add_argument(
        "--gpu_id",
        type=lambda x: None if x.lower() == "none" else int(x),
        default=0,
        metavar="N|none",
        help="GPU device id for embedding model; pass 'none' for CPU (default: %(default)s)",
    )
    parser.add_argument(
        "--condor",
        action="store_true",
        help="Running under HTCondor: use generic 'cuda' device",
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
        help="Path to ORPHA .npy embeddings file (default: %(default)s)",
    )
    parser.add_argument(
        "--retriever",
        type=str,
        default="sentence_transformer",
        help="Retriever type for EmbeddingFuzzyMatcher (default: %(default)s)",
    )
    parser.add_argument(
        "--retriever_model",
        type=str,
        default="abhinand/MedEmbed-small-v0.1",
        help="Retriever model name (default: %(default)s)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=20,
        help="Top-k FAISS candidates per entity (default: %(default)s)",
    )
    parser.add_argument(
        "--fuzzy_threshold",
        type=float,
        default=0.85,
        help="Minimum SequenceMatcher ratio for code matching (default: %(default)s)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=_RESULTS_DIR / "mimic3" / "i2b2_rd",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Load data and model, process one sample, then exit",
    )
    parser.add_argument("--debug", action="store_true", help="Enable verbose debug logging")
    args = parser.parse_args()

    import torch

    if args.condor:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        stanza_device = "cuda" if torch.cuda.is_available() else "cpu"
    elif args.gpu_id is not None and torch.cuda.is_available():
        device_str = f"cuda:{args.gpu_id}"
        stanza_device = args.stanza_device
    else:
        device_str = "cpu"
        stanza_device = "cpu"

    ts(f"MIMIC-III root    : {args.mimic3_root}")
    ts(f"Cache dir         : {args.cache_dir}")
    ts(f"Stanza device     : {stanza_device}")
    ts(f"Embed device      : {device_str}")
    ts(f"Output dir        : {args.output_dir}")
    ts(f"Skip code match   : {args.skip_code_matching}")

    # ── Dataset ───────────────────────────────────────────────────────────────
    ts(f"Loading MIMIC3Dataset from {args.mimic3_root} ...")
    dataset = MIMIC3Dataset(
        root=args.mimic3_root,
        tables=["noteevents"],
        cache_dir=args.cache_dir,
        dev=False,
        num_workers=4,
    )
    ts("Applying MIMIC3RDMiningText ...")
    samples = dataset.set_task(MIMIC3RDMiningText())
    ts(f"  Total samples (annotated notes): {len(samples)}")

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
    if args.dry_run:
        ts("Dry-run: processing first sample ...")
        sample = next(iter(samples))
        note_id = sample["note_id"]
        text = pickle.loads(sample["text"])
        gold = pickle.loads(sample["entities"])
        pred = extractor.extract_entities(text)
        ts(f"  note_id       : {note_id}")
        ts(f"  text[:120]    : {text[:120]!r}")
        ts(f"  gold_entities : {gold}")
        ts(f"  pred_entities : {pred}")
        ts("Dry-run complete.")
        return

    # ── Inference ─────────────────────────────────────────────────────────────
    pred_per_note: Dict[str, Set[str]] = defaultdict(set)
    gold_per_note: Dict[str, Set[str]] = defaultdict(set)

    ts(f"Running inference on {len(samples)} samples ...")
    for idx, sample in enumerate(samples):
        note_id = sample["note_id"]
        text = pickle.loads(sample["text"])
        gold = pickle.loads(sample["entities"])
        gold_per_note[note_id].update(gold)

        pred = extractor.extract_entities(text)
        pred_per_note[note_id].update(pred)

        if args.debug:
            ts(f"  [{note_id}] n_pred={len(pred)}  n_gold={len(gold)}")

        if (idx + 1) % 500 == 0:
            ts(f"  Processed {idx + 1}/{len(samples)} samples ...")

    # ── String-level evaluation ───────────────────────────────────────────────
    ts("Computing evaluation metrics ...")
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
    }

    ts(
        f"Results — P={macro_p:.4f}  R={macro_r:.4f}  F1={macro_f1:.4f}"
        f"  (over {len(note_ids)} notes)"
    )

    # ── Save (text eval) ──────────────────────────────────────────────────────
    args.output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.output_dir / "per_note_predictions.jsonl"
    summary_path = args.output_dir / "results.json"

    with open(jsonl_path, "w") as fh:
        for row in per_note_rows:
            fh.write(json.dumps(row) + "\n")

    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)

    ts(f"Per-note predictions saved to {jsonl_path}")
    ts(f"Summary saved to {summary_path}")

    # ── Code matching (embedding + fuzzy → ORPHA IDs) ─────────────────────────
    if not args.skip_code_matching:
        ts(f"Initialising EmbeddingFuzzyMatcher ({args.orpha_embeddings_file}) ...")
        code_matcher = EmbeddingFuzzyMatcher(
            embeddings_file=args.orpha_embeddings_file,
            retriever=args.retriever,
            retriever_model=args.retriever_model,
            top_k=args.top_k,
            fuzzy_threshold=args.fuzzy_threshold,
            device=device_str,
        )

        code_jsonl_path = args.output_dir / "per_note_code_predictions.jsonl"
        ts(f"Writing code-eval JSONL to {code_jsonl_path} ...")
        with open(code_jsonl_path, "w") as fh:
            for note_id in note_ids:
                pred_entities = sorted(pred_per_note.get(note_id, set()))
                entity_payload = [{"entity": e} for e in pred_entities]
                matched = code_matcher.match(entity_payload)
                orpha_ids = list(dict.fromkeys(
                    m["hp_id"] for m in matched if m.get("hp_id")
                ))
                fh.write(
                    json.dumps(
                        {
                            "id": note_id,
                            "predicted": pred_entities,
                            "predicted_orpha_ids": orpha_ids,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        ts(f"Code-eval predictions saved to {code_jsonl_path}")


if __name__ == "__main__":
    main()
