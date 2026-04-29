#!/usr/bin/env python3
"""
Bio_ClinicalBERT NER inference on MIMIC-III rare-disease notes.

Loads a Bio_ClinicalBERT NER checkpoint trained on RareDis
(via bioclinicalbert_ner_trainer.py), runs it over the MIMIC-III NOTEEVENTS
rows that have human-reviewed rare-disease annotations, extracts predicted
entity strings, and evaluates against the gold standard using set-based
P/R/F1.

Pipeline:
  1. MIMIC3Dataset + MIMIC3RDMiningText → one sample per annotated note
  2. BioClinicalBERTNERModel.load_from_checkpoint(checkpoint_dir)
  3. Long notes are split into overlapping word-boundary chunks; each chunk
     is tokenised independently with the BioClinicalBERT tokenizer
  4. Per chunk: argmax → word-level BIO preds (via word_ids) → entity strings
  5. Predictions aggregated per note (union across chunks)
  6. String-level P/R/F1 vs gold, per-note JSONL + summary JSON saved

Usage (from RDMA repo root):
    python baselines/mimic3_rd_mining_text/bioclinicalbert_ner.py \\
        --checkpoint_dir results/raredis/bioclinicalbert_ner_trainer/best_hf

    # Dry-run (no inference):
    python baselines/mimic3_rd_mining_text/bioclinicalbert_ner.py --dry_run
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
_DEFAULT_MIMIC3_ROOT = (
    "/srv/local/data/physionet.org/files/mimic-iii-clinical-database-1.4/"
)
_DEFAULT_MIMIC3_CACHE_DIR = "/shared/eng/pyhealth/mimic3"
_DEFAULT_ORPHA_EMBEDDINGS_FILE = str(
    _RDMA_ROOT / "data" / "vector_stores" / "rd_orpha_medembed.npy"
)

sys.path.insert(0, str(_PYHEALTH_ROOT))
sys.path.insert(0, str(_RDMA_ROOT))  # inserted last → highest priority

from rdma.hpo.embedding_fuzzy_matcher import EmbeddingFuzzyMatcher  # noqa: E402

from pyhealth.datasets import MIMIC3Dataset  # noqa: E402
from tasks.mimic3_rd_mining_text import MIMIC3RDMiningText  # noqa: E402
from tasks.bioclinicalbert_ner_raredis import BioClinicalBERTNERTask  # noqa: E402
from models.bioclinicalbert_ner import BioClinicalBERTNERModel  # noqa: E402

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def ts(msg: str) -> None:
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S} - {msg}", flush=True)


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


# ── Chunking + tokenisation helpers ──────────────────────────────────────────

# Conservative estimate of subword tokens per whitespace word.
_TOKENS_PER_WORD_EST = 1.5


def _chunk_words(
    words: List[str],
    max_text_tokens: int,
    stride_words: int,
) -> Iterator[Tuple[int, int]]:
    """Yield (start_w, end_w) word-index ranges for overlapping chunks.

    Args:
        words: Full whitespace-split word list.
        max_text_tokens: Maximum subword tokens allowed in a chunk
            (= max_seq_length - 2 for [CLS]/[SEP]).
        stride_words: Number of words to advance between chunks.
            Set to 0 to disable overlap (non-overlapping chunks).
    """
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
    """Tokenise a word list into a single padded sequence.

    Returns:
        Dict with:
          input_ids      [max_seq_length]  – token ids
          attention_mask [max_seq_length]  – 1 for real tokens, 0 for pad
          word_ids       [max_seq_length]  – local word index (-1 = special)
    """
    encoding = tokenizer(
        words,
        is_split_into_words=True,
        truncation=True,
        max_length=max_seq_length,
        padding="max_length",
    )

    raw_word_ids = encoding.word_ids()  # None for special tokens
    word_ids_out = [(-1 if wi is None else wi) for wi in raw_word_ids]

    return {
        "input_ids":      torch.tensor(encoding["input_ids"],      dtype=torch.long),
        "attention_mask": torch.tensor(encoding["attention_mask"], dtype=torch.long),
        "word_ids":       torch.tensor(word_ids_out,               dtype=torch.long),
    }


# ── Span → string ─────────────────────────────────────────────────────────────


# Label ids for the entity types we extract during MIMIC-III inference.
# MIMIC-III gold annotations use is_rare_disease=True; we restrict predictions
# to the two rare-disease entity types so that DISEASE/SIGN/ANAPHOR/SYMPTOM
# spans don't inflate false positives against the gold rare-disease mentions.
_KEEP_LABEL_IDS: Set[int] = {
    BioClinicalBERTNERTask.LABEL2ID["B-RAREDISEASE"],
    BioClinicalBERTNERTask.LABEL2ID["B-SKINRAREDISEASE"],
}


def _bio_to_strings(word_preds: List[int], words: List[str]) -> List[str]:
    """Convert word-level BIO predictions to lowercased entity strings.

    Only spans whose B- label is RAREDISEASE or SKINRAREDISEASE are included,
    matching the scope of the MIMIC-III gold annotations.

    Args:
        word_preds: List of predicted label ids, one per word.
        words: The corresponding word list (chunk_words).

    Returns:
        List of entity strings (joined words, lowercased).
    """
    spans = BioClinicalBERTNERModel.bio_to_spans(word_preds, BioClinicalBERTNERTask.ID2LABEL, BioClinicalBERTNERTask.LABEL2ID)
    results: List[str] = []
    for label_id, span_start, span_end in spans:
        if label_id not in _KEEP_LABEL_IDS:
            continue
        entity_str = " ".join(words[span_start : span_end + 1]).lower()
        if entity_str:
            results.append(entity_str)
    return results


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bio_ClinicalBERT NER inference on MIMIC-III rare-disease notes"
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
        default=str(
            _RESULTS_DIR / "raredis" / "bioclinicalbert_ner_trainer" / "best_hf"
        ),
        help="HuggingFace checkpoint directory (output of bioclinicalbert_ner_trainer.py)",
    )
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument(
        "--stride_words",
        type=int,
        default=64,
        help=(
            "Overlap between consecutive text chunks, in words. "
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
        default=_RESULTS_DIR / "mimic3" / "bioclinicalbert_ner",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Load data and model, process one sample, then exit",
    )
    # Code-matching args (embedding + fuzzy, no LLM)
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
    ts("Applying MIMIC3RDMiningText …")
    task = MIMIC3RDMiningText()
    samples = mimic3_dataset.set_task(task)
    ts(f"  Total annotated notes: {len(samples)}")

    # ── Tokenizer + model ─────────────────────────────────────────────
    _TOKENIZER_BASE = "emilyalsentzer/Bio_ClinicalBERT"
    ts(f"Loading tokenizer from {_TOKENIZER_BASE} …")
    tokenizer = AutoTokenizer.from_pretrained(
        _TOKENIZER_BASE, do_lower_case=False
    )

    ts(f"Loading BioClinicalBERTNERModel from {args.checkpoint_dir} …")
    model = BioClinicalBERTNERModel.load_from_checkpoint(
        args.checkpoint_dir, dataset=None
    )
    model.to(device)
    model.eval()

    # max subword tokens available for text (excluding [CLS] and [SEP])
    max_text_tokens = args.max_seq_length - 2

    # ── Dry-run ───────────────────────────────────────────────────────
    if args.dry_run:
        ts("Dry-run: processing first sample …")
        sample = samples[0]
        text = pickle.loads(sample["text"])
        gold = pickle.loads(sample["entities"])
        words, _ = _word_offsets(text)
        chunks = list(_chunk_words(words, max_text_tokens, args.stride_words))
        ts(f"  note_id      : {sample['note_id']}")
        ts(f"  num_words    : {len(words)}")
        ts(f"  num_chunks   : {len(chunks)}")
        ts(f"  gold_entities: {gold}")
        feat = _tokenize_chunk(tokenizer, words[chunks[0][0] : chunks[0][1]], args.max_seq_length)
        ts(f"  input_ids shape (chunk 0): {tuple(feat['input_ids'].shape)}")
        ts("Dry-run complete.")
        return

    # ── Inference ─────────────────────────────────────────────────────
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pred_per_note: Dict[str, Set[str]] = defaultdict(set)
    gold_per_note: Dict[str, Set[str]] = {}

    ts(f"Running inference on {len(samples)} samples …")
    for idx, sample in enumerate(samples):
        note_id = sample["note_id"]
        text = pickle.loads(sample["text"])
        gold = set(pickle.loads(sample["entities"]))
        gold_per_note[note_id] = gold

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
            # logits: [1, seq, num_labels] → [seq, num_labels]
            logits = out["logits"][0].cpu()
            pred_ids = torch.argmax(logits, dim=-1)  # [seq]

            # Collapse subwords → word-level predictions (first subword per word)
            word_ids = feat["word_ids"]  # [seq]; -1 = special token
            word_preds = BioClinicalBERTNERModel.collapse_to_word_preds(word_ids, pred_ids)

            entity_strings = _bio_to_strings(word_preds, chunk_words)
            pred_per_note[note_id].update(entity_strings)

        if (idx + 1) % 100 == 0:
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
        "stride_words": args.stride_words,
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
