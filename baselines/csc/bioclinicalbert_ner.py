#!/usr/bin/env python3
"""
Bio_ClinicalBERT NER inference on the CSC phenotype-mining benchmark.

Loads a Bio_ClinicalBERT NER checkpoint trained on BioLarkGSC
(via baselines/biolarkgsc/bioclinicalbert_ner.py), runs it over the 116
CSC clinical case reports, extracts predicted phenotype strings, maps them
to HPO codes via EmbeddingFuzzyMatcher, and evaluates against gold HP IDs
using micro P/R/F1.

Pipeline:
  1. CSCDataset + CSCPhenotypeMining → one sample per document
  2. BioClinicalBERTNERModel.load_from_checkpoint(checkpoint_dir)
  3. Long documents are split into overlapping word-boundary chunks
  4. Per chunk: tokenise → argmax → collapse subwords → BIO spans → strings
  5. Predictions aggregated per document (union across chunks)
  6. EmbeddingFuzzyMatcher maps entity strings → HP IDs
  7. Micro P/R/F1 vs gold HP IDs, per-doc JSONL + summary JSON saved

Usage (from RDMA repo root):
    python baselines/csc/bioclinicalbert_ner.py \\
        --checkpoint_dir results/biolarkgsc/bioclinicalbert_ner/best_hf

    # Dry-run (no inference):
    python baselines/csc/bioclinicalbert_ner.py --dry_run
"""

import argparse
import json
import logging
import pickle
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterator, List, Set, Tuple

import torch
from transformers import AutoTokenizer

# ── Path setup ───────────────────────────────────────────────────────────────
_RDMA_ROOT = Path("/home/johnwu3/projects/rare_disease/workspace/repos/RDMA")
_PYHEALTH_ROOT = Path("/home/johnwu3/projects/rare_disease/workspace/repos/PyHealth")
_RESULTS_DIR = Path("/home/johnwu3/projects/rare_disease/workspace/results")
_DEFAULT_EMBEDDINGS_FILE = str(
    _RDMA_ROOT / "data" / "vector_stores" / "G2GHPO_metadata_medembed.npy"
)
_DEFAULT_CACHE_DIR = "/shared/eng/pyhealth/csc"

sys.path.insert(0, str(_PYHEALTH_ROOT))
sys.path.insert(0, str(_RDMA_ROOT))

from rdma.hpo.embedding_fuzzy_matcher import EmbeddingFuzzyMatcher  # noqa: E402
from datasets.csc import CSCDataset  # noqa: E402
from tasks.csc import CSCPhenotypeMining  # noqa: E402
from tasks.bioclinicalbert_ner_biolarkgsc import BioClinicalBERTNERBioLarkGSCTask  # noqa: E402
from models.bioclinicalbert_ner import BioClinicalBERTNERModel  # noqa: E402

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def ts(msg: str) -> None:
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S} - {msg}", flush=True)


# ── Chunking + tokenisation helpers ──────────────────────────────────────────

_TOKENS_PER_WORD_EST = 1.5
_TASK = BioClinicalBERTNERBioLarkGSCTask


def _word_offsets(text: str) -> Tuple[List[str], List[int]]:
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


def _chunk_words(
    words: List[str],
    max_text_tokens: int,
    stride_words: int,
) -> Iterator[Tuple[int, int]]:
    max_words = max(1, int(max_text_tokens / _TOKENS_PER_WORD_EST))
    stride = max(1, stride_words) if stride_words > 0 else max_words
    start = 0
    n = len(words)
    while start < n:
        end = min(start + max_words, n)
        yield start, end
        if end == n:
            break
        start += stride


def _tokenize_chunk(
    tokenizer,
    words: List[str],
    max_seq_length: int,
) -> Dict[str, torch.Tensor]:
    encoding = tokenizer(
        words,
        is_split_into_words=True,
        truncation=True,
        max_length=max_seq_length,
        padding="max_length",
    )
    word_ids_out = [(-1 if wi is None else wi) for wi in encoding.word_ids()]
    return {
        "input_ids":      torch.tensor(encoding["input_ids"],      dtype=torch.long),
        "attention_mask": torch.tensor(encoding["attention_mask"], dtype=torch.long),
        "word_ids":       torch.tensor(word_ids_out,               dtype=torch.long),
    }


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
        description="Bio_ClinicalBERT NER inference on CSC (BioLarkGSC checkpoint)"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=str(_RESULTS_DIR / "biolarkgsc" / "bioclinicalbert_ner" / "best_hf"),
        help="HuggingFace checkpoint directory (output of biolarkgsc/bioclinicalbert_ner.py)",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=_DEFAULT_CACHE_DIR,
        help="PyHealth CSC dataset cache directory",
    )
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument(
        "--stride_words",
        type=int,
        default=64,
        help="Overlap between consecutive text chunks, in words (0 = no overlap)",
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
        default=_RESULTS_DIR / "csc" / "bioclinicalbert_ner",
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
    ts("Loading CSCDataset …")
    dataset = CSCDataset(cache_dir=args.cache_dir)
    ts("Applying CSCPhenotypeMining …")
    samples = dataset.set_task(CSCPhenotypeMining())
    ts(f"  Total documents: {len(samples)}")

    # ── Tokenizer + model ─────────────────────────────────────────────
    _TOKENIZER_BASE = "emilyalsentzer/Bio_ClinicalBERT"
    ts(f"Loading tokenizer from {_TOKENIZER_BASE} …")
    tokenizer = AutoTokenizer.from_pretrained(_TOKENIZER_BASE, do_lower_case=False)

    ts(f"Loading BioClinicalBERTNERModel from {args.checkpoint_dir} …")
    model = BioClinicalBERTNERModel.load_from_checkpoint(
        args.checkpoint_dir, dataset=None
    )
    model.to(device)
    model.eval()

    max_text_tokens = args.max_seq_length - 2  # reserve [CLS] and [SEP]

    # ── Dry-run ───────────────────────────────────────────────────────
    if args.dry_run:
        ts("Dry-run: processing first sample …")
        sample = samples[0]
        text = pickle.loads(sample["text"])
        gold_phenotypes = pickle.loads(sample["phenotypes"])
        words, _ = _word_offsets(text)
        chunks = list(_chunk_words(words, max_text_tokens, args.stride_words))
        ts(f"  doc_id      : {sample['patient_id']}")
        ts(f"  num_words   : {len(words)}")
        ts(f"  num_chunks  : {len(chunks)}")
        ts(f"  gold_hpo_ids: {[p['hpo_id'] for p in gold_phenotypes]}")
        feat = _tokenize_chunk(tokenizer, words[chunks[0][0]:chunks[0][1]], args.max_seq_length)
        ts(f"  input_ids shape (chunk 0): {tuple(feat['input_ids'].shape)}")
        ts("Dry-run complete.")
        return

    # ── Inference ─────────────────────────────────────────────────────
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pred_strings_per_doc: Dict[str, Set[str]] = defaultdict(set)
    gold_hpo_per_doc: Dict[str, List[str]] = {}

    ts(f"Running inference on {len(samples)} documents …")
    for idx, sample in enumerate(samples):
        doc_id = sample["patient_id"]
        text = pickle.loads(sample["text"])
        gold_phenotypes = pickle.loads(sample["phenotypes"])
        gold_hpo_per_doc[doc_id] = [p["hpo_id"] for p in gold_phenotypes if p.get("hpo_id")]

        words, _ = _word_offsets(text)
        if not words:
            continue

        for start_w, end_w in _chunk_words(words, max_text_tokens, args.stride_words):
            chunk_words = words[start_w:end_w]
            if not chunk_words:
                continue

            feat = _tokenize_chunk(tokenizer, chunk_words, args.max_seq_length)

            with torch.no_grad():
                out = model(
                    input_ids=feat["input_ids"].unsqueeze(0).to(device),
                    attention_mask=feat["attention_mask"].unsqueeze(0).to(device),
                )
            pred_ids = torch.argmax(out["logits"][0].cpu(), dim=-1)

            word_preds = BioClinicalBERTNERModel.collapse_to_word_preds(
                feat["word_ids"], pred_ids
            )
            spans = BioClinicalBERTNERModel.bio_to_spans(
                word_preds, _TASK.ID2LABEL, _TASK.LABEL2ID
            )
            for _, ws, we in spans:
                entity_str = " ".join(chunk_words[ws : we + 1]).lower()
                if entity_str:
                    pred_strings_per_doc[doc_id].add(entity_str)

        if (idx + 1) % 20 == 0:
            ts(f"  Processed {idx + 1}/{len(samples)} documents …")

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
        gold_hpo_ids = gold_hpo_per_doc.get(doc_id, [])

        predicted_hpo_ids: List[str] = []
        if pred_entities:
            matched = code_matcher.match([{"entity": e} for e in pred_entities])
            predicted_hpo_ids = list(dict.fromkeys(
                m["hp_id"] for m in matched if m.get("hp_id")
            ))

        records.append(
            {
                "id": doc_id,
                "predicted": predicted_hpo_ids,
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
    jsonl_path = args.output_dir / "predictions.jsonl"
    summary_path = args.output_dir / "results.json"

    with open(jsonl_path, "w") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    summary = {
        "num_docs": len(doc_ids),
        "micro_precision": round(metrics["precision"], 4),
        "micro_recall": round(metrics["recall"], 4),
        "micro_f1": round(metrics["f1"], 4),
        "tp": metrics["tp"],
        "fp": metrics["fp"],
        "fn": metrics["fn"],
        "checkpoint_dir": str(args.checkpoint_dir),
        "stride_words": args.stride_words,
    }
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)

    ts(f"Predictions saved to {jsonl_path}")
    ts(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
