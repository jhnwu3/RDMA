#!/usr/bin/env python3
"""
Quick debug script — sends a single query to the OpenRouter client and prints
the response. Use this to verify the API key and model are working before
running the full pipeline.

Usage (from RDMA repo root):
  python scripts/debug_openrouter.py
  python scripts/debug_openrouter.py --model_type llama3-70b
"""

import argparse
import sys
from pathlib import Path

_RDMA_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_RDMA_ROOT))

from rdma.utils.llm_client import OpenRouterLLMClient  # noqa: E402

_SYSTEM = "You are a helpful assistant. Be concise."
_USER = "In one sentence, what is a rare disease?"


def main():
    parser = argparse.ArgumentParser(description="Debug OpenRouter query")
    parser.add_argument(
        "--model_type",
        type=str,
        default="nemotron-120b",
        help="Model shortcut or raw OpenRouter model ID (default: %(default)s)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.01,
        help="Sampling temperature (default: %(default)s)",
    )
    args = parser.parse_args()

    print(f"Model     : {args.model_type}")
    print(f"System    : {_SYSTEM}")
    print(f"User      : {_USER}")
    print("-" * 60)

    client = OpenRouterLLMClient(
        model_type=args.model_type,
        temperature=args.temperature,
    )
    response = client.query(_USER, _SYSTEM)

    print(f"Response  : {response}")
    print(f"Tokens used: {client.total_tokens_used}")


if __name__ == "__main__":
    main()
