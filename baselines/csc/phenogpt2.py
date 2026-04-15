#!/usr/bin/env python3
"""
PhenoGPT2 baseline for the CSC phenotype-mining benchmark.

Pipeline:
  PhenoGPT2 (fine-tuned LLaMA-3.1-8B-Instruct) generates structured JSON
  with HPO terms and IDs directly from clinical text.

NOTE: This baseline requires the PhenoGPT2 fine-tuned model weights.
  The weights are NOT present in repos/PhenoGPT2/ by default.
  Place the fine-tuned model directory at --model_dir (default shown below).
  The base LLaMA-3.1-8B-Instruct is cached at:
    /shared/rsaas/jw3/rare_disease/model_cache/models--meta-llama--Meta-Llama-3.1-8B-Instruct

Output JSONL format (matches scripts/csc/run_hpo.py):
    {"id": "<doc_id>", "predicted": [...], "ground_truth": [...], "timing": {...}}

Usage (from RDMA repo root):
  python baselines/csc/phenogpt2.py --model_dir /path/to/phenogpt2_weights
"""

import argparse
import ast
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
_PHENOGPT2_ROOT = Path("/home/johnwu3/projects/rare_disease/workspace/repos/PhenoGPT2")
sys.path.insert(0, str(_RDMA_ROOT))
sys.path.insert(0, str(_PHENOGPT2_ROOT))

from datasets.csc import CSCDataset  # noqa: E402
from tasks.csc import CSCPhenotypeMining  # noqa: E402

_RESULTS_DIR = Path("/home/johnwu3/projects/rare_disease/workspace/results")
_DEFAULT_MODEL_DIR = str(_PHENOGPT2_ROOT / "model")
_DEFAULT_DATASET_CACHE_DIR = "/shared/eng/pyhealth/csc"
_HPO_ADDED_TOKENS_FILE = str(_PHENOGPT2_ROOT / "hpo_added_tokens.json")


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


def _extract_hpo_ids_from_output(raw_output: str) -> list:
    """
    Parse PhenoGPT2 JSON output and extract HPO IDs.

    Expected format: {"phenotypes": {"term_name": "HP:XXXXXXX", ...}, ...}
    Returns list of HP: IDs (skips 'unknown' values).
    """
    try:
        text = "{" + raw_output if not raw_output.strip().startswith("{") else raw_output
        try:
            parsed = ast.literal_eval(text)
        except Exception:
            parsed = json.loads(text)
        phenotypes = parsed.get("phenotypes", {})
        hpo_ids = []
        for term, hpo_id in phenotypes.items():
            if hpo_id and hpo_id.lower() != "unknown" and str(hpo_id).startswith("HP:"):
                hpo_ids.append(str(hpo_id))
        return hpo_ids
    except Exception:
        return []


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
        description="PhenoGPT2 baseline on CSC phenotype-mining benchmark"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=_DEFAULT_MODEL_DIR,
        help="Path to PhenoGPT2 fine-tuned model weights directory (default: %(default)s)",
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
            "(default: <results_dir>/csc/phenogpt2_predictions.jsonl)"
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
        _RESULTS_DIR / "csc" / "phenogpt2_predictions.jsonl"
    )

    # ── Check model weights ───────────────────────────────────────────────
    if not os.path.isdir(args.model_dir):
        ts(f"ERROR: PhenoGPT2 model directory not found: {args.model_dir}")
        ts("  PhenoGPT2 fine-tuned weights are required to run this baseline.")
        ts("  Place the fine-tuned model at --model_dir and re-run.")
        sys.exit(1)

    ts(f"Model dir         : {args.model_dir}")
    ts(f"Dataset cache dir : {args.dataset_cache_dir}")
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

    # ── Model ─────────────────────────────────────────────────────────────
    ts("Loading PhenoGPT2 model and tokenizer...")
    import torch
    import numpy as np
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from tokenizers import AddedToken

    tokenizer_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_id,
        cache_dir="/shared/rsaas/jw3/rare_disease/model_cache",
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    tokenizer.padding_side = "left"

    with open(_HPO_ADDED_TOKENS_FILE, "r") as f:
        name2hpo = json.load(f)
    all_hpo_ids = list(np.unique(list(name2hpo.values())))
    for hpo_id in all_hpo_ids:
        tokenizer.add_tokens(
            [AddedToken(hpo_id, single_word=True, normalized=False, special=False)],
            special_tokens=False,
        )
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()
    ts("  PhenoGPT2 ready")

    device = next(model.parameters()).device

    def _generate(text: str) -> str:
        instruction = (
            "You are a genetic counselor specializing in extracting demographic details "
            "and Human Phenotype Ontology (HPO) terms from text and generating a JSON object. "
            "Your task is to provide accurate and concise information without generating random "
            "answers. When demographic details or phenotype information is not explicitly "
            "mentioned in the input, use 'unknown' as the value."
        )
        question = (
            "Read the following input text and generate a JSON-formatted output with the "
            "following keys: demographics and phenotypes. For the demographics key, create a "
            "sub-dictionary with age, sex, ethnicity, and race as keys. For the phenotype key, "
            "create a sub-dictionary where each HPO term is a key and its HPO identifier is the "
            "value. If any information is unavailable, return 'unknown' for that field.\nInput: "
        )
        base_prompt = (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            "    {system_prompt}<|eot_id|>\n    \n"
            "    <|start_header_id|>user<|end_header_id|>\n\n"
            "    {user_prompt}<|eot_id|>\n    \n"
            "    <|start_header_id|>assistant<|end_header_id|>\n"
            "    |==|Response|==|\n"
            "    {{"
        )
        prompt = base_prompt.format(
            system_prompt=instruction,
            user_prompt=question + text.replace("'", "").replace('"', ""),
        )
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]
        with torch.no_grad():
            generation_output = model.generate(
                input_ids,
                max_new_tokens=1024,
                eos_token_id=terminators,
                do_sample=True,
                temperature=0.5,
                top_p=0.8,
            )
        response = generation_output[0][input_ids.shape[-1]:]
        return tokenizer.decode(response, skip_special_tokens=True).strip()

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
            tqdm(run_samples, total=len(run_samples), desc="CSC-PhenoGPT2")
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

            gen_s = 0.0
            try:
                t0 = time.perf_counter()
                raw_output = _generate(text)
                gen_s = time.perf_counter() - t0

                predicted = _extract_hpo_ids_from_output(raw_output)

                if args.debug:
                    ts(f"  [{doc_id}] raw: {raw_output[:200]!r}")
                    ts(f"  [{doc_id}] predicted: {predicted}")

                ts(f"  [{doc_id}] generate={gen_s:.2f}s  n_pred={len(predicted)}")
            except Exception as e:
                ts(f"  ERROR [{doc_id}]: {e}")
                if args.debug:
                    traceback.print_exc()
                predicted = []

            timings.append(gen_s)
            records.append({"predicted": predicted, "ground_truth": ground_truth})
            out_f.write(
                json.dumps(
                    {
                        "id": doc_id,
                        "predicted": predicted,
                        "ground_truth": ground_truth,
                        "timing": {"generation_s": round(gen_s, 3)},
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
        avg_g = sum(timings) / len(timings)
        ts("── Timing summary ──────────────────────────────────────────")
        ts(f"  Samples           : {len(timings)}")
        ts(f"  Avg generation    : {avg_g:.2f}s")

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
