#!/usr/bin/env python3
"""
RDRAG baseline for the RareDis benchmark.

Pipeline:
  1. LLM extracts rare disease mentions from the full text (no retrieval context).
  2. Each extracted entity is matched against the ORPHA dictionary via
     RDMAMatcher (embedding search + LLM-assisted matching).
     No verifier LLM calls.

Output is a JSONL file in the same format as run_raredis.py:
    {"id": "<doc_id>", "predicted": ["entity1", ...], "timing": {...}}

Usage (from RDMA repo root):
  python baselines/run_rdrag_raredis.py
"""

import argparse
import json
import sys
import os
import pickle
import time
import traceback
import numpy as np
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from tqdm import tqdm

_RDMA_ROOT = Path("/home/johnwu3/projects/rare_disease/workspace/repos/RDMA")
sys.path.insert(0, str(_RDMA_ROOT))

from rdma.rd.extractor import RDMAExtractor  # noqa: E402
from rdma.rd.matcher import RDMAMatcher  # noqa: E402
from rdma.utils.embedding import EmbeddingsManager  # noqa: E402
from rdma.utils.llm_client import (  # noqa: E402
    LocalLLMClient,
    APILLMClient,
    AzureOpenAILLMClient,
    OpenRouterLLMClient,
    LlamaCppLLMClient,
)
from rdma.utils.setup import setup_device  # noqa: E402

from datasets.raredis import RareDisDataset  # noqa: E402
from tasks.raredis import RareDisNER  # noqa: E402

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


def main():
    parser = argparse.ArgumentParser(description="RDRAG baseline on RareDis")
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
        "--temperature",
        type=float,
        default=0.01,
        help="LLM sampling temperature (default: %(default)s)",
    )
    parser.add_argument(
        "--llm_type",
        type=str,
        default="local",
        choices=["local", "api", "openrouter", "azure", "llama_cpp"],
        help="LLM backend (default: %(default)s)",
    )
    parser.add_argument(
        "--gguf_file",
        type=str,
        default=None,
        help="GGUF filename override for llama_cpp backend (default: per model_type)",
    )
    parser.add_argument(
        "--api_config",
        type=str,
        default=None,
        help="Path to saved API config JSON (api/openrouter only)",
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
        help="Top-k ORPHA candidates to retrieve per entity (default: %(default)s)",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=5,
        help="Context window size for the extractor (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSONL path (default: <results_dir>/raredis/<model_type>_rdrag_predictions.jsonl)",
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

    output = args.output or (
        _RESULTS_DIR / "raredis" / f"{args.model_type}_rdrag_predictions.jsonl"
    )

    cfg = SimpleNamespace(
        gpu_id=args.gpu_id,
        condor=args.condor,
        cpu=(args.gpu_id is None and not args.condor),
        retriever_gpu_id=None,
        retriever_cpu=False,
    )
    devices = setup_device(cfg)
    ts(f"LLM device       : {devices['llm']}")
    ts(f"LLM type         : {args.llm_type}")
    ts(f"Model type       : {args.model_type}")
    ts(f"Temperature      : {args.temperature}")
    ts(f"Embeddings       : {args.embeddings_file}")
    ts(f"Retriever        : {args.retriever} / {args.retriever_model}")
    ts(f"Top-k            : {args.top_k}")
    ts(f"Window size      : {args.window_size}")
    ts(f"Output           : {output}")
    ts(f"Resume           : {args.resume}")

    # ── Dataset ──────────────────────────────────────────────────────────
    ts("Loading RareDisDataset...")
    dataset = RareDisDataset(num_workers=4)
    samples = dataset.set_task(RareDisNER(), num_workers=4)
    ts(f"  {len(samples)} samples loaded")

    # ── Pipeline ─────────────────────────────────────────────────────────
    ts(f"Loading LLM ({args.llm_type} / {args.model_type})")
    if args.llm_type == "api":
        llm_client = (
            APILLMClient.from_config(args.api_config)
            if args.api_config
            else APILLMClient(model_type=args.model_type, temperature=args.temperature)
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

    ts(f"Loading embeddings: {args.embeddings_file}")
    embedded_documents = np.load(args.embeddings_file, allow_pickle=True)
    ts(f"  {len(embedded_documents)} documents")

    embedding_manager = EmbeddingsManager(
        model_type=args.retriever,
        model_name=args.retriever_model,
        device=devices.get("retriever", devices["llm"]),
    )

    ts("Initialising LLM extractor (extraction_method=llm)...")
    extractor = RDMAExtractor(
        llm_client=llm_client,
        extraction_method="llm",
        window_size=args.window_size,
        debug=args.debug,
    )

    ts("Initialising RDMAMatcher...")
    matcher = RDMAMatcher(
        llm_client=llm_client,
        embedding_manager=embedding_manager,
        embedded_documents=embedded_documents,
        top_k=args.top_k,
        debug=args.debug,
    )

    # ── Run ──────────────────────────────────────────────────────────────
    done_ids = load_done_ids(output) if args.resume else set()
    if args.resume:
        ts(f"Resuming – {len(done_ids)} already done")

    output.parent.mkdir(parents=True, exist_ok=True)
    out_f = open(output, "a" if args.resume else "w", encoding="utf-8")

    try:
        timings: list = []
        run_samples = samples.subset(slice(0, 2)) if args.dev else samples
        for i, sample in enumerate(tqdm(run_samples, total=len(run_samples), desc="RareDis-RDRAG")):
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

            extract_s = match_s = 0.0
            try:
                t0 = time.perf_counter()
                entities_with_contexts = extractor.extract_from_text(text)
                extract_s = time.perf_counter() - t0
                if args.debug:
                    ts(f"  [{doc_id}] extracted {len(entities_with_contexts)}")

                t0 = time.perf_counter()
                matched = matcher.match_entities(entities_with_contexts)
                match_s = time.perf_counter() - t0

                predicted = [m["entity"] for m in matched]
                if args.debug:
                    ts(f"  [{doc_id}] matched   {len(predicted)}")

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
            out_f.write(
                json.dumps(
                    {
                        "id": doc_id,
                        "predicted": predicted,
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
        ts(f"  Samples          : {len(timings)}")
        ts(f"  Avg extraction   : {avg_e:.2f}s")
        ts(f"  Avg matching     : {avg_m:.2f}s")
        ts(f"  Avg total/sample : {avg_e + avg_m:.2f}s")

    ts(f"Done → {output}")


if __name__ == "__main__":
    main()
