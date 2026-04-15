#!/usr/bin/env python3
"""
Dictionary-matching baseline for the MIMIC-III rare-disease mining benchmark.

Loads annotated notes via the PyHealth MIMIC3RareDiseaseMining task, then for
each note runs an embedding-retrieval + fuzzy n-gram match against the ORPHA
rare-disease dictionary to extract entity strings and ORPHA codes.

No LLM is required — the only model is the sentence embedding retriever used
to find candidate ORPHA terms, followed by exact / fuzzy string matching.

Output:
  • <output_dir>/dict_predictions.jsonl      — code-eval format
        {"id": "<note_id>", "patient_id": "...",
         "predicted": [...], "predicted_orpha_ids": [...], "timing": {...}}
  • <output_dir_text>/dict_predictions.jsonl — text-eval format
        {"note_id": "<note_id>", "predicted": [...]}

Usage (from RDMA repo root):
    python baselines/mimic3_rd_mining_code/dict.py \\
        --mimic3_root /path/to/mimic-iii/1.4

    # Dry-run (load data, skip matching):
    python baselines/mimic3_rd_mining_code/dict.py \\
        --mimic3_root /path/to/mimic-iii/1.4 --dry_run
"""

import argparse
import json
import os
import pickle
import re
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from tqdm import tqdm

# ── Path setup ────────────────────────────────────────────────────────────────
_RDMA_ROOT = Path("/home/johnwu3/projects/rare_disease/workspace/repos/RDMA")
_PYHEALTH_ROOT = Path("/home/johnwu3/projects/rare_disease/workspace/repos/PyHealth")
_RESULTS_DIR = Path("/home/johnwu3/projects/rare_disease/workspace/results")

sys.path.insert(0, str(_PYHEALTH_ROOT))
sys.path.insert(0, str(_RDMA_ROOT))

from pyhealth.datasets import MIMIC3Dataset  # noqa: E402
from tasks.mimic3_rd_mining import MIMIC3RareDiseaseMining  # noqa: E402
from rdma.utils.embedding import EmbeddingsManager  # noqa: E402

# ── Defaults ──────────────────────────────────────────────────────────────────
_DEFAULT_MIMIC3_CACHE_DIR = "/shared/eng/pyhealth/mimic3"
_DEFAULT_ANNOTATION_PATH = str(
    _RDMA_ROOT
    / "public_data"
    / "rare_disease_mining"
    / "mimic3_mining_rdma_human_annotations.json"
)
_DEFAULT_EMBEDDINGS_FILE = str(
    _RDMA_ROOT / "data" / "vector_stores" / "rd_orpha_medembed.npy"
)
_DEFAULT_RETRIEVER_MODEL = "abhinand/MedEmbed-small-v0.1"


def ts(msg: str) -> None:
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S} - {msg}", flush=True)


def format_orpha_id(code: str) -> str:
    """Normalise an ORPHA code to 'ORPHA:12345' form."""
    if not code:
        return ""
    s = str(code).strip()
    if s.upper().startswith("ORPHA:"):
        return s
    if s.isdigit():
        return f"ORPHA:{s}"
    return s


def load_done_ids(path: Path) -> Set[str]:
    done: Set[str] = set()
    if path.exists():
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        obj = json.loads(line)
                        nid = obj.get("id") or obj.get("note_id")
                        if nid:
                            done.add(str(nid))
                    except Exception:
                        pass
    return done


# ── Matching logic (ported from old dict.py) ──────────────────────────────────

def _sentence_split(text: str) -> List[str]:
    """Split clinical text into sentences."""
    sentences = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s", text)
    return [s.strip() for s in sentences if s.strip()]


def _resolve_term(doc: dict) -> Tuple[str, str, str]:
    """Extract (term, orpha_id, definition) from an embedded document dict."""
    if "name" in doc and "id" in doc:
        return doc.get("name", ""), doc.get("id", ""), doc.get("definition", "")
    meta = doc.get("unique_metadata", {})
    return meta.get("name", ""), meta.get("id", ""), meta.get("definition", "")


def match_note(
    text: str,
    embedding_manager: EmbeddingsManager,
    embedded_documents: np.ndarray,
    faiss_index,
    top_k: int,
    similarity_threshold: float,
) -> List[Dict[str, Any]]:
    """Run dictionary matching on one clinical note.

    Returns a list of matched entity dicts, each with keys:
        entity, rd_term, orpha_id, match_method, confidence_score
    """
    sentences = _sentence_split(text)
    extracted: List[Dict[str, Any]] = []
    seen_keys: Set[str] = set()

    for sentence in sentences:
        if len(sentence) < 10:
            continue

        clean_sentence = sentence.lower()
        query_vec = embedding_manager.query_text(sentence).reshape(1, -1)
        distances, indices = embedding_manager.search(query_vec, faiss_index, k=top_k)

        for idx, _dist in zip(indices[0], distances[0]):
            term, orpha_id, _ = _resolve_term(embedded_documents[idx])
            if not orpha_id:
                continue

            entity_key = f"{term}::{orpha_id}"
            if entity_key in seen_keys:
                continue

            clean_term = term.lower()

            # Method 1: exact substring match
            if clean_term in clean_sentence:
                extracted.append(
                    {
                        "entity": term,
                        "rd_term": term,
                        "orpha_id": orpha_id,
                        "match_method": "exact",
                        "confidence_score": 1.0,
                    }
                )
                seen_keys.add(entity_key)
                continue

            # Method 2: n-gram fuzzy match
            term_tokens = re.findall(r"\b\w+\b", clean_term)
            if len(term_tokens) <= 1:
                continue

            sentence_tokens = re.findall(r"\b\w+\b", clean_sentence)
            max_n = min(len(term_tokens) + 1, 5)
            best_score = 0.0
            best_ngram = ""

            for n in range(2, max_n):
                for i in range(len(sentence_tokens) - n + 1):
                    ngram = " ".join(sentence_tokens[i : i + n])
                    score = SequenceMatcher(None, ngram, clean_term).ratio()
                    if score > best_score:
                        best_score = score
                        best_ngram = ngram

            if best_score >= similarity_threshold:
                extracted.append(
                    {
                        "entity": best_ngram,
                        "rd_term": term,
                        "orpha_id": orpha_id,
                        "match_method": "fuzzy",
                        "confidence_score": best_score,
                    }
                )
                seen_keys.add(entity_key)

    extracted.sort(key=lambda x: x["confidence_score"], reverse=True)
    return extracted


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dictionary baseline for MIMIC-III rare-disease mining"
    )
    parser.add_argument(
        "--mimic3_root",
        type=str,
        required=True,
        help="Root directory of the MIMIC-III 1.4 dataset",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=_DEFAULT_MIMIC3_CACHE_DIR,
        help="PyHealth MIMIC-III dataset cache directory (default: %(default)s)",
    )
    parser.add_argument(
        "--annotation_path",
        type=str,
        default=_DEFAULT_ANNOTATION_PATH,
        help="Path to mimic3_mining_rdma_human_annotations.json (default: %(default)s)",
    )
    parser.add_argument(
        "--embeddings_file",
        type=str,
        default=_DEFAULT_EMBEDDINGS_FILE,
        help="Path to .npy embeddings file (default: %(default)s)",
    )
    parser.add_argument(
        "--retriever",
        type=str,
        default="sentence_transformer",
        choices=["fastembed", "sentence_transformer", "medcpt"],
        help="Retriever type (default: %(default)s)",
    )
    parser.add_argument(
        "--retriever_model",
        type=str,
        default=_DEFAULT_RETRIEVER_MODEL,
        help="Retriever model name (default: %(default)s)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Top-k ORPHA candidates to retrieve per sentence (default: %(default)s)",
    )
    parser.add_argument(
        "--similarity_threshold",
        type=float,
        default=0.9,
        help="Fuzzy match threshold (default: %(default)s)",
    )
    # GPU / device
    gpu_group = parser.add_mutually_exclusive_group()
    gpu_group.add_argument(
        "--gpu_id",
        type=lambda x: None if x.lower() == "none" else int(x),
        default=0,
        metavar="N|none",
        help="GPU device id; pass 'none' for CPU (default: 0)",
    )
    gpu_group.add_argument(
        "--condor",
        action="store_true",
        help="Running under HTCondor: use generic 'cuda' device",
    )
    gpu_group.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU usage",
    )
    # Output
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output JSONL path for code-eval "
            "(default: results/mimic3_rd_mining/dict_predictions.jsonl)"
        ),
    )
    parser.add_argument(
        "--output_text",
        type=Path,
        default=None,
        help=(
            "Output JSONL path for text-eval "
            "(default: results/mimic3_rd_mining_text/dict_predictions.jsonl)"
        ),
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume from existing output file"
    )
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=20,
        help="Log a checkpoint every N samples (default: %(default)s)",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable verbose debug logging"
    )
    parser.add_argument(
        "--dev", action="store_true", help="Dev mode: process only first 2 samples"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Load data and embeddings, print counts, then exit without running matching",
    )
    return parser.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    import torch

    args = parse_args()

    output_code = args.output or (
        _RESULTS_DIR / "mimic3_rd_mining" / "dict_predictions.jsonl"
    )
    output_text = args.output_text or (
        _RESULTS_DIR / "mimic3_rd_mining_text" / "dict_predictions.jsonl"
    )

    # ── Device ────────────────────────────────────────────────────────────
    if args.cpu or args.gpu_id is None:
        retriever_device = "cpu"
    elif args.condor:
        retriever_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        if torch.cuda.is_available() and args.gpu_id < torch.cuda.device_count():
            retriever_device = f"cuda:{args.gpu_id}"
        else:
            retriever_device = "cpu"

    ts(f"Retriever device  : {retriever_device}")
    ts(f"Retriever         : {args.retriever} / {args.retriever_model}")
    ts(f"Embeddings file   : {args.embeddings_file}")
    ts(f"Top-k             : {args.top_k}")
    ts(f"Similarity thresh : {args.similarity_threshold}")
    ts(f"MIMIC-III root    : {args.mimic3_root}")
    ts(f"Annotation path   : {args.annotation_path}")
    ts(f"Output (code)     : {output_code}")
    ts(f"Output (text)     : {output_text}")
    ts(f"Resume            : {args.resume}")
    ts(f"Dev mode          : {args.dev}")

    # ── Dataset ───────────────────────────────────────────────────────────
    ts("Loading MIMIC3Dataset...")
    dataset = MIMIC3Dataset(
        root=args.mimic3_root,
        tables=["noteevents"],
        cache_dir=args.cache_dir,
        num_workers=4,
    )
    task = MIMIC3RareDiseaseMining(annotation_path=args.annotation_path)
    samples = dataset.set_task(task, num_workers=4)
    ts(f"  {len(samples)} annotated notes loaded")

    if len(samples) == 0:
        ts("ERROR: no samples found — check mimic3_root and annotation_path")
        sys.exit(1)

    first = next(iter(samples))
    ts(
        f"  Sample preview — note_id: {first['note_id']!r}"
        f"  patient_id: {first['patient_id']!r}"
    )
    ts(f"    text[:120]: {pickle.loads(first['text'])[:120]!r}")

    if args.dry_run:
        ts("Dry-run complete. Exiting.")
        return

    # ── Embeddings ────────────────────────────────────────────────────────
    ts(f"Loading embeddings: {args.embeddings_file}")
    embedded_documents = np.load(args.embeddings_file, allow_pickle=True)
    ts(f"  {len(embedded_documents)} documents loaded")

    ts(f"Initialising embedding manager ({args.retriever})...")
    embedding_manager = EmbeddingsManager(
        model_type=args.retriever,
        model_name=args.retriever_model,
        device=retriever_device,
    )

    # Pre-build FAISS index once (shared across all notes)
    ts("Building FAISS index...")
    embeddings_array = embedding_manager.prepare_embeddings(embedded_documents)
    faiss_index = embedding_manager.create_index(embeddings_array)
    ts("  FAISS index ready")

    # ── Resume ────────────────────────────────────────────────────────────
    done_ids = load_done_ids(output_code) if args.resume else set()
    if args.resume:
        ts(f"Resuming – {len(done_ids)} already done")

    # ── Output files ──────────────────────────────────────────────────────
    output_code.parent.mkdir(parents=True, exist_ok=True)
    output_text.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume else "w"
    f_code = open(output_code, mode, encoding="utf-8")
    f_text = open(output_text, mode, encoding="utf-8")

    # ── Inference loop ────────────────────────────────────────────────────
    try:
        run_samples = samples.subset(slice(0, 2)) if args.dev else samples
        for i, sample in enumerate(
            tqdm(run_samples, total=len(run_samples), desc="MIMIC3-RD-Dict")
        ):
            note_id = sample["note_id"]
            patient_id = sample.get("patient_id", "")

            if note_id in done_ids:
                continue

            try:
                text = pickle.loads(sample["text"])
            except Exception as e:
                ts(f"  SKIP [{note_id}] (data error): {e}")
                continue

            t0 = time.perf_counter()
            try:
                matches = match_note(
                    text,
                    embedding_manager,
                    embedded_documents,
                    faiss_index,
                    top_k=args.top_k,
                    similarity_threshold=args.similarity_threshold,
                )
                elapsed = time.perf_counter() - t0
            except Exception as e:
                ts(f"  ERROR [{note_id}]: {e}")
                if args.debug:
                    traceback.print_exc()
                matches = []
                elapsed = time.perf_counter() - t0

            predicted_entities = [m["entity"] for m in matches]
            predicted_orpha_ids = [format_orpha_id(m["orpha_id"]) for m in matches]

            if args.debug:
                ts(f"  [{note_id}] {len(matches)} matches  ({elapsed:.2f}s)")

            # Code-eval output
            f_code.write(
                json.dumps(
                    {
                        "id": note_id,
                        "patient_id": patient_id,
                        "predicted": predicted_entities,
                        "predicted_orpha_ids": predicted_orpha_ids,
                        "timing": {"matching_s": round(elapsed, 3)},
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            f_code.flush()

            # Text-eval output
            f_text.write(
                json.dumps(
                    {
                        "note_id": note_id,
                        "predicted": predicted_entities,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            f_text.flush()

            if (i + 1) % args.checkpoint_interval == 0:
                ts(f"Checkpoint {i + 1}/{len(run_samples)}")

    finally:
        f_code.close()
        f_text.close()

    ts(f"Done → code eval: {output_code}")
    ts(f"Done → text eval: {output_text}")


if __name__ == "__main__":
    main()
