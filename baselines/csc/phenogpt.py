#!/usr/bin/env python3
"""
PhenoGPT baseline for the CSC phenotype-mining benchmark.

Pipeline:
  1. PhenoGPT (LLaMA-2-7B + LoRA) extracts phenotype text strings.
  2a. Default: EmbeddingFuzzyMatcher maps extracted strings to HPO codes
      using embedding retrieval + fuzzy matching (no LLM required).
  2b. Optional (--use_biosent2vec): PhenoGPTWithHPO maps via BioSent2Vec similarity
      (requires --biosent2vec_path and the sent2vec library).

Output JSONL format (matches scripts/csc/run_hpo.py):
    {"id": "<doc_id>", "predicted": [...], "ground_truth": [...], "timing": {...}}

Usage (from RDMA repo root):
  python baselines/csc/phenogpt.py --gpu_id 0
  python baselines/csc/phenogpt.py --use_biosent2vec --biosent2vec_path /path/to/BioSentVec.bin
"""

import argparse
import json
import sys
import os
import pickle
import time
import traceback
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

_RDMA_ROOT = Path("/home/johnwu3/projects/rare_disease/workspace/repos/RDMA")
_PHENOGPT_ROOT = Path("/home/johnwu3/projects/rare_disease/workspace/repos/PhenoGPT")
sys.path.insert(0, str(_RDMA_ROOT))
sys.path.insert(0, str(_PHENOGPT_ROOT))

from rdma.hpo.embedding_fuzzy_matcher import EmbeddingFuzzyMatcher  # noqa: E402

from run_phenogpt import PhenoGPT, PhenoGPTWithHPO  # noqa: E402

from datasets.csc import CSCDataset  # noqa: E402
from tasks.csc import CSCPhenotypeMining  # noqa: E402

_RESULTS_DIR = Path("/home/johnwu3/projects/rare_disease/workspace/results")
_DEFAULT_LORA_WEIGHTS = str(_PHENOGPT_ROOT / "model/llama2/llama2_lora_weights")
_DEFAULT_HPO_DATABASE = str(_PHENOGPT_ROOT / "hpo_database.json")
_DEFAULT_BASE_MODEL = "meta-llama/Llama-2-7b-chat-hf"
_DEFAULT_EMBEDDINGS_FILE = str(
    _RDMA_ROOT / "data" / "vector_stores" / "G2GHPO_metadata_medembed.npy"
)
_DEFAULT_MODEL_CACHE_DIR = "/shared/rsaas/jw3/rare_disease/model_cache"
_DEFAULT_DATASET_CACHE_DIR = "/shared/eng/pyhealth/csc"


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


def compute_metrics(records: list) -> dict:
    tp = fp = fn = 0
    for rec in records:
        predicted = set(h for h in rec["predicted"] if h)
        gold = set(h for h in rec["ground_truth"] if h)
        tp += len(predicted & gold)
        fp += len(predicted - gold)
        fn += len(gold - predicted)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "n_docs": len(records),
    }


def main():
    parser = argparse.ArgumentParser(
        description="PhenoGPT baseline on CSC phenotype-mining benchmark"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default=_DEFAULT_BASE_MODEL,
        help="Base LLM model path or HuggingFace ID (default: %(default)s)",
    )
    parser.add_argument(
        "--lora_weights",
        type=str,
        default=_DEFAULT_LORA_WEIGHTS,
        help="Path to PhenoGPT LoRA weights (default: %(default)s)",
    )
    parser.add_argument(
        "--model_cache_dir",
        type=str,
        default=_DEFAULT_MODEL_CACHE_DIR,
        help="HuggingFace model cache directory (default: %(default)s)",
    )
    parser.add_argument(
        "--dataset_cache_dir",
        type=str,
        default=_DEFAULT_DATASET_CACHE_DIR,
        help="PyHealth dataset cache directory (default: %(default)s)",
    )
    parser.add_argument(
        "--use_biosent2vec",
        action="store_true",
        help=(
            "Use PhenoGPTWithHPO + BioSent2Vec for HPO mapping "
            "(requires --biosent2vec_path and sent2vec library)"
        ),
    )
    parser.add_argument(
        "--biosent2vec_path",
        type=str,
        default=None,
        help="Path to BioSentVec .bin model file (required if --use_biosent2vec)",
    )
    parser.add_argument(
        "--hpo_database",
        type=str,
        default=_DEFAULT_HPO_DATABASE,
        help="Path to HPO database JSON for BioSent2Vec mapping (default: %(default)s)",
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
    # EmbeddingFuzzyMatcher args (used when --use_biosent2vec is NOT set)
    parser.add_argument(
        "--embeddings_file",
        type=Path,
        default=Path(_DEFAULT_EMBEDDINGS_FILE),
        help="Path to HPO .npy embeddings file (default: %(default)s)",
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
        help="Minimum SequenceMatcher ratio (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output JSONL path "
            "(default: <results_dir>/csc/phenogpt_predictions.jsonl)"
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output file",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable verbose debug logging"
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Dev mode: process only the first 2 samples",
    )
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=20,
        help="Log a checkpoint every N samples (default: %(default)s)",
    )
    args = parser.parse_args()

    if args.use_biosent2vec and not args.biosent2vec_path:
        parser.error("--biosent2vec_path is required when --use_biosent2vec is set")

    output = args.output or (
        _RESULTS_DIR / "csc" / "phenogpt_predictions.jsonl"
    )

    # Set HuggingFace cache
    os.environ["TRANSFORMERS_CACHE"] = args.model_cache_dir
    os.environ["HF_HOME"] = args.model_cache_dir

    import torch
    if args.condor:
        embed_device = "cuda" if torch.cuda.is_available() else "cpu"
        llm_device = "cuda" if torch.cuda.is_available() else "cpu"
    elif args.gpu_id is not None and torch.cuda.is_available():
        embed_device = f"cuda:{args.gpu_id}"
        llm_device = f"cuda:{args.gpu_id}"
    else:
        embed_device = "cpu"
        llm_device = "cpu"

    ts(f"LLM device        : {llm_device}")
    ts(f"Base model        : {args.base_model}")
    ts(f"LoRA weights      : {args.lora_weights}")
    ts(f"HPO mapping       : {'BioSent2Vec' if args.use_biosent2vec else 'EmbeddingFuzzyMatcher'}")
    ts(f"Model cache dir   : {args.model_cache_dir}")
    ts(f"Dataset cache dir : {args.dataset_cache_dir}")
    ts(f"GPU id            : {args.gpu_id}")
    ts(f"Condor mode       : {args.condor}")
    ts(f"Output            : {output}")
    ts(f"Resume            : {args.resume}")
    ts(f"Debug             : {args.debug}")
    ts(f"Dev mode          : {args.dev}")
    ts(f"Checkpoint every  : {args.checkpoint_interval}")

    # ── Dataset ───────────────────────────────────────────────────────────
    ts("Loading CSCDataset...")
    dataset = CSCDataset(cache_dir=args.dataset_cache_dir)
    samples = dataset.set_task(CSCPhenotypeMining())
    ts(f"  {len(samples)} samples loaded")

    first = next(iter(samples))
    ts(f"  Sample preview — id: {first['patient_id']!r}")
    ts(f"    text[:120]: {pickle.loads(first['text'])[:120]!r}")

    # ── Pipeline ──────────────────────────────────────────────────────────
    matcher = None
    if args.use_biosent2vec:
        ts("Loading PhenoGPTWithHPO (BioSent2Vec HPO mapping)...")
        phenogpt_model = PhenoGPTWithHPO(
            base_model_path=args.base_model,
            lora_weights_path=args.lora_weights,
            biosent2vec_path=args.biosent2vec_path,
            hpo_database_path=args.hpo_database,
        )
    else:
        ts("Loading PhenoGPT (EmbeddingFuzzyMatcher for HPO mapping)...")
        phenogpt_model = PhenoGPT(
            base_model_path=args.base_model,
            lora_weights_path=args.lora_weights,
        )
        ts(f"Initialising EmbeddingFuzzyMatcher (embeddings: {args.embeddings_file})...")
        matcher = EmbeddingFuzzyMatcher(
            embeddings_file=str(args.embeddings_file),
            retriever=args.retriever,
            retriever_model=args.retriever_model,
            top_k=args.top_k,
            fuzzy_threshold=args.fuzzy_threshold,
            device=embed_device,
            debug=args.debug,
        )

    # ── Run ───────────────────────────────────────────────────────────────
    done_ids = load_done_ids(output) if args.resume else set()
    if args.resume:
        ts(f"Resuming – {len(done_ids)} already done")

    output.parent.mkdir(parents=True, exist_ok=True)
    out_f = open(output, "a" if args.resume else "w", encoding="utf-8")

    try:
        timings: list = []
        records: list = []
        run_samples = samples.subset(slice(0, 2)) if args.dev else samples
        for i, sample in enumerate(
            tqdm(run_samples, total=len(run_samples), desc="CSC-PhenoGPT")
        ):
            try:
                doc_id = sample["patient_id"]
                text = pickle.loads(sample["text"])
                ground_truth = [
                    p["hpo_id"]
                    for p in pickle.loads(sample["phenotypes"])
                    if p.get("hpo_id")
                ]
            except Exception as e:
                ts(f"  SKIP sample {i} (data error): {e}")
                if args.debug:
                    traceback.print_exc()
                continue

            if doc_id in done_ids:
                continue

            extract_s = match_s = 0.0
            try:
                if args.use_biosent2vec:
                    t0 = time.perf_counter()
                    result_dict = phenogpt_model.generate_with_hpo(text)
                    extract_s = time.perf_counter() - t0
                    predicted = [
                        v for v in result_dict.values()
                        if v and v != "N/A"
                    ]
                    if args.debug:
                        ts(f"  [{doc_id}] phenogpt+biosent2vec → {len(predicted)} HP codes")
                    ts(f"  [{doc_id}] generate_with_hpo={extract_s:.2f}s  n_pred={len(predicted)}")
                else:
                    t0 = time.perf_counter()
                    phenotypes = phenogpt_model.generate(text)
                    extract_s = time.perf_counter() - t0
                    if args.debug:
                        ts(f"  [{doc_id}] extracted {len(phenotypes)} phenotype strings")

                    entities = [{"entity": p, "context": text} for p in phenotypes if p]
                    t0 = time.perf_counter()
                    matched = matcher.match(entities)
                    match_s = time.perf_counter() - t0
                    if args.debug:
                        ts(f"  [{doc_id}] matched {len(matched)} HP codes")

                    predicted = [
                        m.get("hp_id", "") for m in matched if m.get("hp_id")
                    ]
                    ts(
                        f"  [{doc_id}] "
                        f"extract={extract_s:.2f}s  "
                        f"match={match_s:.2f}s"
                    )
            except Exception as e:
                ts(f"  ERROR [{doc_id}]: {e}")
                if args.debug:
                    traceback.print_exc()
                predicted = []

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
                ts(f"Checkpoint {i + 1}/{len(samples)}")
    finally:
        out_f.close()

    if timings:
        avg_e = sum(t[0] for t in timings) / len(timings)
        avg_m = sum(t[1] for t in timings) / len(timings)
        ts("── Timing summary ──────────────────────────────────────────")
        ts(f"  Samples           : {len(timings)}")
        ts(f"  Avg extraction    : {avg_e:.2f}s")
        ts(f"  Avg matching      : {avg_m:.2f}s")
        ts(f"  Avg total/sample  : {avg_e + avg_m:.2f}s")

    if records:
        metrics = compute_metrics(records)
        ts("── Code evaluation ─────────────────────────────────────────")
        ts(f"  Precision : {metrics['precision']:.4f}")
        ts(f"  Recall    : {metrics['recall']:.4f}")
        ts(f"  F1        : {metrics['f1']:.4f}")
        ts(f"  TP={metrics['tp']}  FP={metrics['fp']}  FN={metrics['fn']}  Docs={metrics['n_docs']}")

    ts(f"Done → {output}")


if __name__ == "__main__":
    main()
