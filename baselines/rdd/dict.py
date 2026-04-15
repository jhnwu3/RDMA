#!/usr/bin/env python3
"""
Run the dictionary-matching baseline on the RDD benchmark.

Usage (from RDMA repo root):
  python baselines/run_dict_rdd.py
"""

import argparse
import json
import sys
import os
import pickle
import re
import time
import traceback
import numpy as np
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from types import SimpleNamespace
from tqdm import tqdm

_RDMA_ROOT = Path("/home/johnwu3/projects/rare_disease/workspace/repos/RDMA")
sys.path.insert(0, str(_RDMA_ROOT))

from rdma.utils.embedding import EmbeddingsManager  # noqa: E402
from rdma.utils.setup import setup_device  # noqa: E402

from datasets.rdd import RDDDataset  # noqa: E402
from tasks.rdd import RDDNER  # noqa: E402

_RESULTS_DIR = Path("/home/johnwu3/projects/rare_disease/workspace/results")
_DEFAULT_EMBEDDINGS_FILE = (
    "/home/johnwu3/projects/rare_disease/workspace/repos/RDMA"
    "/data/vector_stores/rd_orpha_medembed.npy"
)
_DEFAULT_MODEL_CACHE_DIR = "/shared/rsaas/jw3/rare_disease/model_cache"


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


def process_clinical_text(
    clinical_text, embedding_manager, embedded_documents, top_k, similarity_threshold
):
    """Extract rare disease entities via dictionary matching."""
    sentences = re.split(
        r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s", clinical_text
    )
    sentences = [s.strip() for s in sentences if s.strip()]

    embeddings_array = embedding_manager.prepare_embeddings(embedded_documents)
    faiss_index = embedding_manager.create_index(embeddings_array)

    extracted_entities = []
    unique_entities = set()

    for sentence in sentences:
        if len(sentence) < 10:
            continue

        clean_sentence = sentence.lower()
        query_vector = embedding_manager.query_text(sentence).reshape(1, -1)
        distances, indices = embedding_manager.search(query_vector, faiss_index, k=top_k)

        for idx, distance in zip(indices[0], distances[0]):
            doc = embedded_documents[idx]
            if "name" in doc and "id" in doc:
                term = doc.get("name", "")
                orpha_id = doc.get("id", "")
            else:
                metadata = doc.get("unique_metadata", {})
                term = metadata.get("name", "")
                orpha_id = metadata.get("id", "")

            entity_key = f"{term}::{orpha_id}"
            if not orpha_id or entity_key in unique_entities:
                continue

            clean_term = term.lower()

            if clean_term in clean_sentence:
                extracted_entities.append({"entity": term, "orpha_id": orpha_id})
                unique_entities.add(entity_key)
                continue

            term_tokens = re.findall(r"\b\w+\b", clean_term)
            sentence_tokens = re.findall(r"\b\w+\b", clean_sentence)
            if len(term_tokens) <= 1:
                continue

            best_score = 0
            best_text = ""
            max_ngram_size = min(len(term_tokens) + 1, 5)
            for n in range(2, max_ngram_size):
                for i in range(len(sentence_tokens) - n + 1):
                    ngram = " ".join(sentence_tokens[i : i + n])
                    score = SequenceMatcher(None, ngram, clean_term).ratio()
                    if score > best_score:
                        best_score = score
                        best_text = ngram

            if best_score >= similarity_threshold:
                extracted_entities.append({"entity": best_text, "orpha_id": orpha_id})
                unique_entities.add(entity_key)

    return extracted_entities


def main():
    parser = argparse.ArgumentParser(description="Dictionary baseline on RDD")
    parser.add_argument(
        "--embeddings_file",
        type=Path,
        default=Path(_DEFAULT_EMBEDDINGS_FILE),
        help="Path to .npy embeddings file (default: %(default)s)",
    )
    parser.add_argument(
        "--retriever",
        type=str,
        default="sentence_transformer",
        help="Retriever type (default: %(default)s)",
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
        default=5,
        help="Top-k retrieved candidates per sentence (default: %(default)s)",
    )
    parser.add_argument(
        "--similarity_threshold",
        type=float,
        default=0.9,
        help="Fuzzy match similarity threshold (default: %(default)s)",
    )
    parser.add_argument(
        "--gpu_id",
        type=lambda x: None if x.lower() == "none" else int(x),
        default=1,
        metavar="N|none",
        help="GPU device id; pass 'none' for CPU (default: %(default)s)",
    )
    parser.add_argument(
        "--condor",
        action="store_true",
        help="Running under HTCondor: use generic 'cuda' device",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSONL path (default: <results_dir>/rdd/dict_predictions.jsonl)",
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume from existing output file"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable verbose debug logging"
    )
    parser.add_argument(
        "--dev", action="store_true", help="Dev mode: process only first 2 samples"
    )
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=20,
        help="Log a checkpoint every N samples (default: %(default)s)",
    )
    args = parser.parse_args()

    output = args.output or (_RESULTS_DIR / "rdd" / "dict_predictions.jsonl")

    cfg = SimpleNamespace(
        gpu_id=args.gpu_id,
        condor=args.condor,
        cpu=(args.gpu_id is None and not args.condor),
        retriever_gpu_id=None,
        retriever_cpu=False,
    )
    devices = setup_device(cfg)
    ts(f"Device           : {devices['llm']}")
    ts(f"Embeddings       : {args.embeddings_file}")
    ts(f"Retriever        : {args.retriever} / {args.retriever_model}")
    ts(f"Top-k            : {args.top_k}")
    ts(f"Sim threshold    : {args.similarity_threshold}")
    ts(f"Output           : {output}")
    ts(f"Resume           : {args.resume}")

    ts("Loading RDDDataset...")
    dataset = RDDDataset(num_workers=4)
    samples = dataset.set_task(RDDNER(), num_workers=4)
    ts(f"  {len(samples)} samples loaded")

    ts(f"Loading embeddings: {args.embeddings_file}")
    embedded_documents = np.load(args.embeddings_file, allow_pickle=True)
    ts(f"  {len(embedded_documents)} documents")

    embedding_manager = EmbeddingsManager(
        model_type=args.retriever,
        model_name=args.retriever_model,
        device=devices.get("retriever", devices["llm"]),
    )

    done_ids = load_done_ids(output) if args.resume else set()
    if args.resume:
        ts(f"Resuming – {len(done_ids)} already done")

    output.parent.mkdir(parents=True, exist_ok=True)
    out_f = open(output, "a" if args.resume else "w", encoding="utf-8")

    try:
        timings = []
        run_samples = samples.subset(slice(0, 2)) if args.dev else samples
        for i, sample in enumerate(tqdm(run_samples, total=len(run_samples), desc="RDD-Dict")):
            try:
                doc_id = sample["patient_id"]
                text = pickle.loads(sample["text"])
            except Exception as e:
                ts(f"  SKIP sample {i} (data error): {e}")
                if args.debug:
                    traceback.print_exc()
                continue

            if doc_id in done_ids:
                continue

            match_s = 0.0
            try:
                t0 = time.perf_counter()
                matched = process_clinical_text(
                    text,
                    embedding_manager,
                    embedded_documents,
                    args.top_k,
                    args.similarity_threshold,
                )
                match_s = time.perf_counter() - t0

                predicted = [m["entity"] for m in matched]
                if args.debug:
                    ts(f"  [{doc_id}] matched {len(predicted)}")
            except Exception as e:
                ts(f"  ERROR [{doc_id}]: {e}")
                if args.debug:
                    traceback.print_exc()
                predicted = []

            timings.append(match_s)
            out_f.write(
                json.dumps(
                    {
                        "id": doc_id,
                        "predicted": predicted,
                        "timing": {"matching_s": round(match_s, 3)},
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
        avg = sum(timings) / len(timings)
        ts("── Timing summary ──────────────────────────────────────────")
        ts(f"  Samples        : {len(timings)}")
        ts(f"  Avg match/sample: {avg:.2f}s")

    ts(f"Done → {output}")


if __name__ == "__main__":
    main()
