#!/usr/bin/env python3
"""
Zero-shot HPO phenotype extraction baseline for the CSC benchmark.

Each document is passed directly to the LLM with a single extraction prompt.
No retrieval, verification, or matching steps are used — the model is asked
to return HPO identifiers directly.

Output JSONL format (matches scripts/csc/run_hpo.py):
    {"id": "<doc_id>", "predicted": ["HP:XXXXXXX", ...]}

Usage (from RDMA repo root):
    python baselines/csc/zeroshot.py --model_type llama3_8b
"""

import argparse
import json
import os
import pickle
import re
import sys
import traceback
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from tqdm import tqdm

_RDMA_ROOT = Path("/home/johnwu3/projects/rare_disease/workspace/repos/RDMA")
sys.path.insert(0, str(_RDMA_ROOT))

from rdma.utils.llm_client import (  # noqa: E402
    LocalLLMClient,
    APILLMClient,
    AzureOpenAILLMClient,
    OpenRouterLLMClient,
    LlamaCppLLMClient,
    LLMClient,
)
from rdma.utils.setup import setup_device  # noqa: E402
from datasets.csc import CSCDataset  # noqa: E402
from tasks.csc import CSCPhenotypeMining  # noqa: E402

_RESULTS_DIR = Path("/home/johnwu3/projects/rare_disease/workspace/results")
_DEFAULT_MODEL_CACHE_DIR = "/shared/rsaas/jw3/rare_disease/model_cache"
_DEFAULT_DATASET_CACHE_DIR = "/shared/eng/pyhealth/csc"

_SYSTEM = (
    "You are a biomedical phenotype extraction expert specializing in Human Phenotype Ontology. "
    "Your output must be valid JSON only — no prose, no explanation."
)

_PROMPT_TMPL = """\
Extract all Human Phenotype Ontology (HPO) phenotype terms from the text below \
and return their HPO identifiers.

Rules:
- Return only HPO identifiers in the format HP:XXXXXXX (7-digit code).
- Include every phenotype or clinical abnormality mentioned.
- If no HPO phenotypes are present, return an empty array.

Return ONLY a JSON array of HPO identifier strings, for example:
["HP:0000118", "HP:0001250"]

Text:
{text}"""


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


def parse_hpo_ids(response: str) -> list:
    """Extract HP:XXXXXXX identifiers from the LLM response."""
    return re.findall(r"HP:\d{7}", response)


def zero_shot_hpo(text: str, llm_client: LLMClient) -> list:
    """Run zero-shot HPO extraction on a single document. Returns a list of HP IDs."""
    prompt = _PROMPT_TMPL.format(text=text)
    response = llm_client.query(prompt, _SYSTEM)
    return parse_hpo_ids(response)


def main():
    parser = argparse.ArgumentParser(description="Zero-shot HPO extraction baseline for CSC")
    parser.add_argument("--model_type", type=str, default="qwen_32b")
    parser.add_argument("--model_cache_dir", type=str, default=_DEFAULT_MODEL_CACHE_DIR)
    parser.add_argument("--dataset_cache_dir", type=str, default=_DEFAULT_DATASET_CACHE_DIR)
    parser.add_argument("--temperature", type=float, default=0.01)
    parser.add_argument(
        "--llm_type", type=str, default="local",
        choices=["local", "api", "openrouter", "azure", "llama_cpp"],
    )
    parser.add_argument("--gguf_file", type=str, default=None)
    parser.add_argument("--api_config", type=str, default=None)
    parser.add_argument(
        "--gpu_id",
        type=lambda x: None if x.lower() == "none" else int(x),
        default=3, metavar="N|none",
    )
    parser.add_argument(
        "--condor", action="store_true",
        help="Running under HTCondor: use generic 'cuda' device instead of cuda:N",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--dev", action="store_true", help="Process only the first 2 samples")
    parser.add_argument("--checkpoint_interval", type=int, default=20)
    args = parser.parse_args()

    output = args.output or (
        _RESULTS_DIR / "csc" / f"zeroshot_{args.model_type}_predictions.jsonl"
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
    ts(f"Output           : {output}")
    ts(f"Resume           : {args.resume}")
    ts(f"Dev mode         : {args.dev}")

    ts("Loading CSCDataset...")
    dataset = CSCDataset(cache_dir=args.dataset_cache_dir)
    samples = dataset.set_task(CSCPhenotypeMining())
    ts(f"  {len(samples)} samples loaded")

    ts(f"Loading LLM ({args.llm_type} / {args.model_type})...")
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
            else OpenRouterLLMClient(model_type=args.model_type, temperature=args.temperature)
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

    done_ids = load_done_ids(output) if args.resume else set()
    if args.resume:
        ts(f"Resuming – {len(done_ids)} already done")

    output.parent.mkdir(parents=True, exist_ok=True)
    out_f = open(output, "a" if args.resume else "w", encoding="utf-8")

    try:
        run_samples = samples.subset(slice(0, 2)) if args.dev else samples
        for i, sample in enumerate(tqdm(run_samples, total=len(run_samples), desc="CSC-ZS")):
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

            try:
                predicted = zero_shot_hpo(text, llm_client)
                if args.debug:
                    ts(f"  [{doc_id}] {len(predicted)} HPO IDs")
            except Exception as e:
                ts(f"  ERROR [{doc_id}]: {e}")
                if args.debug:
                    traceback.print_exc()
                predicted = []

            out_f.write(
                json.dumps({"id": doc_id, "predicted": predicted}, ensure_ascii=False) + "\n"
            )
            out_f.flush()

            if (i + 1) % args.checkpoint_interval == 0:
                ts(f"Checkpoint {i + 1}/{len(samples)}")
    finally:
        out_f.close()

    ts(f"Done → {output}")


if __name__ == "__main__":
    main()
