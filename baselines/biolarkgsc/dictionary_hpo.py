#!/usr/bin/env python3
"""
Retrieval + string-match Dictionary HPO baseline for the BioLark GSC benchmark.

Pipeline:
  1. Embed each sentence and retrieve top-k HPO candidates.
  2. Match candidate terms in text via exact substring and fuzzy n-gram matching.

Output JSONL format:
    {"id": "<doc_id>", "predicted": [...], "ground_truth": [...], "timing": {...}}
"""

import argparse
import json
import pickle
import re
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
from difflib import SequenceMatcher
from tqdm import tqdm

_RDMA_ROOT = Path("/home/johnwu3/projects/rare_disease/workspace/repos/RDMA")
sys.path.insert(0, str(_RDMA_ROOT))

from rdma.utils.embedding import EmbeddingsManager  # noqa: E402
from datasets.biolarkgsc import BioLarkGSCDataset  # noqa: E402
from tasks.biolarkgsc import BioLarkGSCNER  # noqa: E402

_RESULTS_DIR = Path("/home/johnwu3/projects/rare_disease/workspace/results")
_DEFAULT_EMBEDDINGS_FILE = str(
    _RDMA_ROOT / "data" / "vector_stores" / "G2GHPO_metadata_medembed.npy"
)
_DEFAULT_DATASET_CACHE_DIR = "/shared/eng/pyhealth/biolarkgsc"


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


def _normalize_hpo(hpo_id: str) -> str:
    return hpo_id.replace("_", ":", 1) if hpo_id.startswith("HP_") else hpo_id


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


def _split_sentences(text: str) -> List[str]:
    sentences = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s", text)
    return [s.strip() for s in sentences if s and s.strip()]


def _match_candidate_in_sentence(
    sentence: str,
    term: str,
    similarity_threshold: float,
) -> float:
    clean_sentence = sentence.lower()
    clean_term = term.lower()

    if clean_term and clean_term in clean_sentence:
        return 1.0

    sentence_tokens = re.findall(r"\b\w+\b", clean_sentence)
    term_tokens = re.findall(r"\b\w+\b", clean_term)
    if len(term_tokens) <= 1:
        return 0.0

    best_match_score = 0.0
    max_ngram_size = min(len(term_tokens) + 1, 5)
    for n in range(2, max_ngram_size):
        for i in range(len(sentence_tokens) - n + 1):
            ngram = " ".join(sentence_tokens[i : i + n])
            similarity = SequenceMatcher(None, ngram, clean_term).ratio()
            if similarity > best_match_score:
                best_match_score = similarity

    return best_match_score if best_match_score >= similarity_threshold else 0.0


def run_dictionary_matching(
    text: str,
    embedding_manager: EmbeddingsManager,
    embedded_documents: np.ndarray,
    faiss_index,
    top_k: int,
    similarity_threshold: float,
) -> List[str]:
    predicted = []
    unique_hpo = set()

    for sentence in _split_sentences(text):
        query_vector = embedding_manager.query_text(sentence).reshape(1, -1)
        distances, indices = embedding_manager.search(
            query_vector, faiss_index, k=top_k
        )

        for idx, _distance in zip(indices[0], distances[0]):
            metadata = embedded_documents[idx].get("unique_metadata", {})
            term = metadata.get("info", "")
            hp_id = metadata.get("hp_id", "")

            if not hp_id or hp_id in unique_hpo:
                continue

            score = _match_candidate_in_sentence(
                sentence=sentence,
                term=term,
                similarity_threshold=similarity_threshold,
            )
            if score > 0:
                unique_hpo.add(hp_id)
                predicted.append(hp_id)

    return predicted


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dictionary HPO baseline on BioLark GSC phenotype-mining benchmark"
    )
    parser.add_argument(
        "--embeddings_file",
        type=Path,
        default=Path(_DEFAULT_EMBEDDINGS_FILE),
        help="Path to HPO .npy embeddings file (default: %(default)s)",
    )
    parser.add_argument(
        "--retriever",
        type=str,
        choices=["fastembed", "sentence_transformer", "medcpt"],
        default="sentence_transformer",
        help="Retriever type (default: %(default)s)",
    )
    parser.add_argument(
        "--retriever_model",
        type=str,
        default="BAAI/bge-small-en-v1.5",
        help="Retriever model name (default: %(default)s)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=20,
        help="Top-k retrieved candidates per sentence (default: %(default)s)",
    )
    parser.add_argument(
        "--similarity_threshold",
        type=float,
        default=0.9,
        help="Fuzzy match threshold (default: %(default)s)",
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
            "(default: <results_dir>/biolarkgsc/dictionary_hpo_predictions.jsonl)"
        ),
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume from existing output"
    )
    parser.add_argument("--debug", action="store_true", help="Enable verbose logging")
    parser.add_argument(
        "--dev", action="store_true", help="Process only first 2 samples"
    )
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=20,
        help="Log checkpoint every N samples (default: %(default)s)",
    )
    args = parser.parse_args()

    output = args.output or (
        _RESULTS_DIR / "biolarkgsc" / "dictionary_hpo_predictions.jsonl"
    )

    ts(f"Embeddings file   : {args.embeddings_file}")
    ts(f"Retriever         : {args.retriever} / {args.retriever_model}")
    ts(f"Top-k             : {args.top_k}")
    ts(f"Similarity thres. : {args.similarity_threshold}")
    ts(f"Dataset cache dir : {args.dataset_cache_dir}")
    ts(f"Output            : {output}")
    ts(f"Resume            : {args.resume}")
    ts(f"Debug             : {args.debug}")
    ts(f"Dev mode          : {args.dev}")
    ts(f"Checkpoint every  : {args.checkpoint_interval}")

    ts("Loading BioLarkGSCDataset...")
    dataset = BioLarkGSCDataset(cache_dir=args.dataset_cache_dir)
    samples = dataset.set_task(BioLarkGSCNER())
    ts(f"  {len(samples)} samples loaded")

    first = next(iter(samples))
    ts(f"  Sample preview - id: {first['patient_id']!r}")
    ts(f"    text[:120]: {pickle.loads(first['text'])[:120]!r}")

    ts("Loading embeddings and building retrieval index...")
    embedded_documents = np.load(args.embeddings_file, allow_pickle=True)
    embedding_manager = EmbeddingsManager(
        model_type=args.retriever,
        model_name=(
            args.retriever_model
            if args.retriever in ["fastembed", "sentence_transformer"]
            else None
        ),
    )
    embeddings_array = embedding_manager.prepare_embeddings(embedded_documents)
    faiss_index = embedding_manager.create_index(embeddings_array)
    ts(f"  loaded {len(embedded_documents)} embedded docs")

    done_ids = load_done_ids(output) if args.resume else set()
    if args.resume:
        ts(f"Resuming - {len(done_ids)} already done")

    output.parent.mkdir(parents=True, exist_ok=True)
    out_f = open(output, "a" if args.resume else "w", encoding="utf-8")

    try:
        timings = []
        records = []
        run_samples = samples.subset(slice(0, 2)) if args.dev else samples
        for i, sample in enumerate(
            tqdm(run_samples, total=len(run_samples), desc="BioLarkGSC-DictionaryHPO")
        ):
            try:
                doc_id = sample["patient_id"]
                text = pickle.loads(sample["text"])
                ground_truth = [
                    _normalize_hpo(a["hpo_id"])
                    for a in pickle.loads(sample["annotations"])
                    if a.get("hpo_id")
                ]
            except Exception as e:
                ts(f"  SKIP sample {i} (data error): {e}")
                if args.debug:
                    traceback.print_exc()
                continue

            if doc_id in done_ids:
                continue

            extract_s = match_s = 0.0
            predicted = []
            try:
                t0 = time.perf_counter()
                sentences = _split_sentences(text)
                extract_s = time.perf_counter() - t0

                t0 = time.perf_counter()
                predicted = run_dictionary_matching(
                    text=" ".join(sentences),
                    embedding_manager=embedding_manager,
                    embedded_documents=embedded_documents,
                    faiss_index=faiss_index,
                    top_k=args.top_k,
                    similarity_threshold=args.similarity_threshold,
                )
                match_s = time.perf_counter() - t0

                if args.debug:
                    ts(f"  [{doc_id}] n_sent={len(sentences)} n_pred={len(predicted)}")
                ts(f"  [{doc_id}] extract={extract_s:.2f}s  match={match_s:.2f}s")
            except Exception as e:
                ts(f"  ERROR [{doc_id}]: {e}")
                if args.debug:
                    traceback.print_exc()

            timings.append((extract_s, match_s))
            records.append({"predicted": predicted, "ground_truth": ground_truth})
            out_f.write(
                json.dumps(
                    {
                        "id": doc_id,
                        "predicted": predicted,
                        "ground_truth": ground_truth,
                        "timing": {
                            "extraction_s": round(extract_s, 3),
                            "matching_s": round(match_s, 3),
                        },
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            out_f.flush()

            if (i + 1) % args.checkpoint_interval == 0:
                ts(f"Checkpoint {i + 1}/{len(run_samples)}")
    finally:
        out_f.close()

    if timings:
        avg_e = sum(t[0] for t in timings) / len(timings)
        avg_m = sum(t[1] for t in timings) / len(timings)
        ts("-- Timing summary -------------------------------")
        ts(f"  Samples           : {len(timings)}")
        ts(f"  Avg extraction    : {avg_e:.2f}s")
        ts(f"  Avg matching      : {avg_m:.2f}s")
        ts(f"  Avg total/sample  : {avg_e + avg_m:.2f}s")

    if records:
        metrics = compute_metrics(records)
        ts("-- Code evaluation ------------------------------")
        ts(f"  Precision : {metrics['precision']:.4f}")
        ts(f"  Recall    : {metrics['recall']:.4f}")
        ts(f"  F1        : {metrics['f1']:.4f}")
        ts(
            f"  TP={metrics['tp']}  FP={metrics['fp']}  FN={metrics['fn']}  Docs={metrics['n_docs']}"
        )

    ts(f"Done -> {output}")


if __name__ == "__main__":
    main()
