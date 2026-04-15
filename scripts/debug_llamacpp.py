#!/usr/bin/env python3
"""
Quick debug script — loads a GGUF model via llama-cpp-python and sends a single
query. Use this to verify GPU offloading and model output before running the
full pipeline.

Requires: pip install llama-cpp-python

Usage (from RDMA repo root):
  python scripts/debug_llamacpp.py
  python scripts/debug_llamacpp.py --model_type nemotron-120b-q4
  python scripts/debug_llamacpp.py --model_type nemotron-120b-q3 --n_gpu_layers 40
"""

import argparse
import sys
from pathlib import Path

_RDMA_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_RDMA_ROOT))

from rdma.utils.llm_client import LlamaCppLLMClient  # noqa: E402

_SYSTEM = "You are a helpful assistant. Be concise."
_USER = "In one sentence, what is a rare disease?"


def main():
    parser = argparse.ArgumentParser(description="Debug llama-cpp-python query")
    parser.add_argument(
        "--model_type",
        type=str,
        default=LlamaCppLLMClient.DEFAULT_MODEL,
        choices=list(LlamaCppLLMClient.GGUF_MAPPING.keys()),
        help="GGUF model shortcut (default: %(default)s)",
    )
    parser.add_argument(
        "--gguf_file",
        type=str,
        default=None,
        help="Override the GGUF filename (e.g. a specific quantization variant)",
    )
    parser.add_argument(
        "--n_gpu_layers",
        type=int,
        default=-1,
        help="Number of layers to offload to GPU (-1 = all, 0 = CPU only, default: %(default)s)",
    )
    parser.add_argument(
        "--n_ctx",
        type=int,
        default=8192,
        help="Context window size (default: %(default)s)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.01,
        help="Sampling temperature (default: %(default)s)",
    )
    parser.add_argument(
        "--model_cache_dir",
        type=str,
        default="/shared/rsaas/jw3/rare_disease/model_cache",
        help="Directory to cache downloaded model files (default: %(default)s)",
    )
    args = parser.parse_args()

    print(f"Model type  : {args.model_type}")
    print(f"GGUF file   : {args.gguf_file or '(default for model_type)'}")
    print(f"GPU layers  : {args.n_gpu_layers}")
    print(f"Context     : {args.n_ctx}")
    print(f"Temperature : {args.temperature}")
    print(f"Cache dir   : {args.model_cache_dir}")
    print(f"System      : {_SYSTEM}")
    print(f"User        : {_USER}")
    print("-" * 60)

    client = LlamaCppLLMClient(
        model_type=args.model_type,
        gguf_file=args.gguf_file,
        n_gpu_layers=args.n_gpu_layers,
        n_ctx=args.n_ctx,
        temperature=args.temperature,
        cache_dir=args.model_cache_dir,
    )
    response = client.query(_USER, _SYSTEM)

    print(f"Response    : {response}")


if __name__ == "__main__":
    main()
