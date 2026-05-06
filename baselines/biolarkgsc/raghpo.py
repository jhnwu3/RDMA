#!/usr/bin/env python3
"""
RAG-HPO baseline for the BioLark GSC phenotype-mining benchmark.

Pipeline:
  1. PhenotypeExtractor extracts phenotype mentions with retrieval context.
  2. HPOMatcher maps extracted entities directly to HPO codes.
     No HPOVerifier LLM calls.

Output JSONL format (matches scripts/biolarkgsc/run_hpo.py):
    {"id": "<doc_id>", "predicted": [...], "ground_truth": [...], "timing": {...}}

Usage (from RDMA repo root):
  python baselines/biolarkgsc/raghpo.py
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
from types import SimpleNamespace
from tqdm import tqdm

_RDMA_ROOT = Path("/home/johnwu3/projects/rare_disease/workspace/repos/RDMA")
sys.path.insert(0, str(_RDMA_ROOT))

from rdma.hpo.extractor import PhenotypeExtractor  # noqa: E402
from rdma.hpo.matcher import HPOMatcher  # noqa: E402
from rdma.utils.llm_client import (  # noqa: E402
    LocalLLMClient,
    APILLMClient,
    AzureOpenAILLMClient,
    OpenRouterLLMClient,
    LlamaCppLLMClient,
)
from rdma.utils.setup import setup_device  # noqa: E402

from datasets.biolarkgsc import BioLarkGSCDataset  # noqa: E402
from tasks.biolarkgsc import BioLarkGSCNER  # noqa: E402

_RESULTS_DIR = Path("/home/johnwu3/projects/rare_disease/workspace/results")
_DEFAULT_EMBEDDINGS_FILE = str(
    _RDMA_ROOT / "data" / "vector_stores" / "G2GHPO_metadata_medembed.npy"
)
_DEFAULT_MODEL_CACHE_DIR = "/shared/rsaas/jw3/rare_disease/model_cache"
_DEFAULT_DATASET_CACHE_DIR = "/shared/eng/pyhealth/biolarkgsc"


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


def _normalize_hpo(hpo_id: str) -> str:
    """Normalise BioLark GSC's ``HP_XXXXXXX`` to ``HP:XXXXXXX``."""
    return hpo_id.replace("_", ":", 1) if hpo_id.startswith("HP_") else hpo_id


def compute_metrics(records: list, strict: bool = False) -> dict:
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
        description="RAG-HPO baseline on BioLark GSC phenotype-mining benchmark"
    )
    parser.add_argument(
        "--llm_type",
        type=str,
        default="local",
        choices=["local", "api", "openrouter", "azure", "llama_cpp"],
        help="LLM backend (default: %(default)s)",
    )
    parser.add_argument(
        "--api_config",
        type=str,
        default=None,
        help="Path to saved API config JSON (api/openrouter only)",
    )
    parser.add_argument(
        "--gguf_file",
        type=str,
        default=None,
        help="GGUF filename override for llama_cpp backend",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="qwen_32b",
        help="LLM model type (default: %(default)s)",
    )
    parser.add_argument(
        "--model_cache_dir",
        type=str,
        default=_DEFAULT_MODEL_CACHE_DIR,
        help="Model cache directory (default: %(default)s)",
    )
    parser.add_argument(
        "--dataset_cache_dir",
        type=str,
        default=_DEFAULT_DATASET_CACHE_DIR,
        help="PyHealth dataset cache directory (default: %(default)s)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.01,
        help="LLM sampling temperature (default: %(default)s)",
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
        "--embeddings_file",
        type=Path,
        default=Path(_DEFAULT_EMBEDDINGS_FILE),
        help="Path to HPO .npy embeddings file (default: %(default)s)",
    )
    parser.add_argument(
        "--retriever",
        type=str,
        default="fastembed",
        help="Retriever type (default: %(default)s)",
    )
    parser.add_argument(
        "--retriever_model",
        type=str,
        default="BAAI/bge-small-en-v1.5",
        help="Retriever model name (default: %(default)s)",
    )
    parser.add_argument(
        "--entity_extractor",
        type=str,
        default="simple",
        help="Entity extraction method (default: %(default)s)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Top-k retrieved candidates (default: %(default)s)",
    )
    parser.add_argument(
        "--matcher_version",
        type=str,
        default="standard",
        choices=["standard", "optimized"],
        help="HPOMatcher optimizer version (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output JSONL path "
            "(default: <results_dir>/biolarkgsc/<model_type>_raghpo_predictions.jsonl)"
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

    output = args.output or (
        _RESULTS_DIR / "biolarkgsc" / f"{args.model_type}_raghpo_predictions.jsonl"
    )

    cfg = SimpleNamespace(
        gpu_id=args.gpu_id,
        condor=args.condor,
        cpu=(args.gpu_id is None and not args.condor),
        retriever_gpu_id=None,
        retriever_cpu=False,
    )
    devices = setup_device(cfg)
    ts(f"LLM device        : {devices['llm']}")
    ts(f"LLM type          : {args.llm_type}")
    ts(f"Model type        : {args.model_type}")
    ts(f"Temperature       : {args.temperature}")
    ts(f"GPU id            : {args.gpu_id}")
    ts(f"Condor mode       : {args.condor}")
    ts(f"Model cache dir   : {args.model_cache_dir}")
    ts(f"Dataset cache dir : {args.dataset_cache_dir}")
    ts(f"HPO embeddings    : {args.embeddings_file}")
    ts(f"Retriever         : {args.retriever} / {args.retriever_model}")
    ts(f"Extractor         : {args.entity_extractor}")
    ts(f"Matcher version   : {args.matcher_version}")
    ts(f"Top-k             : {args.top_k}")
    ts(f"Output            : {output}")
    ts(f"Resume            : {args.resume}")
    ts(f"Debug             : {args.debug}")
    ts(f"Dev mode          : {args.dev}")
    ts(f"Checkpoint every  : {args.checkpoint_interval}")

    # ── Dataset ───────────────────────────────────────────────────────────
    ts("Loading BioLarkGSCDataset...")
    dataset = BioLarkGSCDataset(cache_dir=args.dataset_cache_dir)
    samples = dataset.set_task(BioLarkGSCNER())
    ts(f"  {len(samples)} samples loaded")

    first = next(iter(samples))
    ts(f"  Sample preview — id: {first['patient_id']!r}")
    ts(f"    text[:120]: {pickle.loads(first['text'])[:120]!r}")
    ann_preview = pickle.loads(first["annotations"])[:3]
    n_ann = len(pickle.loads(first["annotations"]))
    ts(f"    annotations ({n_ann}): {ann_preview}")

    # ── Pipeline ──────────────────────────────────────────────────────────
    ts(f"Loading LLM ({args.llm_type} / {args.model_type})")
    if args.llm_type == "api":
        llm_client = (
            APILLMClient.from_config(args.api_config)
            if args.api_config
            else APILLMClient(
                model_type=args.model_type, temperature=args.temperature
            )
        )
    elif args.llm_type == "openrouter":
        llm_client = (
            OpenRouterLLMClient.from_config(args.api_config)
            if args.api_config
            else OpenRouterLLMClient(
                model_type=args.model_type, temperature=args.temperature
            )
        )
    elif args.llm_type == "azure":
        llm_client = (
            AzureOpenAILLMClient.from_config(args.api_config)
            if args.api_config
            else AzureOpenAILLMClient(
                model_type=args.model_type,
                azure_deployment=args.model_type,
                temperature=args.temperature,
            )
        )
    elif args.llm_type == "llama_cpp":
        llm_client = LlamaCppLLMClient(
            model_type=args.model_type,
            gguf_file=args.gguf_file,
            main_gpu=args.gpu_id if args.gpu_id is not None else 0,
            temperature=args.temperature,
            cache_dir=args.model_cache_dir,
        )
    else:
        llm_client = LocalLLMClient(
            model_type=args.model_type,
            device=devices["llm"],
            cache_dir=args.model_cache_dir,
            temperature=args.temperature,
        )

    ts("Initialising RAG-HPO pipeline (no verifier)…")
    extractor = PhenotypeExtractor(
        llm_client=llm_client,
        extractor_type=args.entity_extractor,
        embeddings_file=str(args.embeddings_file),
        retriever=args.retriever,
        retriever_model=args.retriever_model,
        top_k=args.top_k,
        debug=args.debug,
    )
    matcher = HPOMatcher(
        llm_client=llm_client,
        embeddings_file=str(args.embeddings_file),
        optimizer_version=args.matcher_version,
        retriever=args.retriever,
        retriever_model=args.retriever_model,
        top_k=args.top_k,
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
            tqdm(run_samples, total=len(run_samples), desc="BioLarkGSC-RAGHPO")
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
            try:
                t0 = time.perf_counter()
                entities_with_contexts = extractor.extract([text])
                extract_s = time.perf_counter() - t0
                if args.debug:
                    ts(f"  [{doc_id}] extracted {len(entities_with_contexts)}")

                t0 = time.perf_counter()
                matched_phenotypes = matcher.match(entities_with_contexts)
                match_s = time.perf_counter() - t0
                if args.debug:
                    ts(f"  [{doc_id}] matched   {len(matched_phenotypes)}")

                ts(
                    f"  [{doc_id}] "
                    f"extract={extract_s:.2f}s  "
                    f"match={match_s:.2f}s"
                )
                predicted = [
                    m.get("hp_id", "")
                    for m in matched_phenotypes
                    if m.get("hp_id")
                ]
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
        lenient = compute_metrics(records, strict=False)
        strict = compute_metrics(records, strict=True)
        ts("── Code evaluation (lenient) ───────────────────────────────")
        ts(f"  Precision : {lenient['precision']:.4f}")
        ts(f"  Recall    : {lenient['recall']:.4f}")
        ts(f"  F1        : {lenient['f1']:.4f}")
        ts(f"  TP={lenient['tp']}  FP={lenient['fp']}  FN={lenient['fn']}")
        ts("── Code evaluation (strict) ────────────────────────────────")
        ts(f"  Precision : {strict['precision']:.4f}")
        ts(f"  Recall    : {strict['recall']:.4f}")
        ts(f"  F1        : {strict['f1']:.4f}")
        ts(
            f"  TP={strict['tp']}  FP={strict['fp']}  FN={strict['fn']}"
            f"  Docs={strict['n_docs']}"
        )

    ts(f"Done → {output}")


if __name__ == "__main__":
    main()
