#!/usr/bin/env python3
"""
Zero-shot rare-disease NER baseline for the MIMIC-III rare-disease mining
benchmark.

Each annotated discharge note is passed directly to the LLM with a single
extraction prompt.  No retrieval, verification, or matching steps are used.

Output is a JSONL file in the same format as the run scripts:
    {"id": "<note_id>", "predicted": ["entity1", "entity2", ...]}

This output is directly consumable by eval_mimic3_rd_mining.py
(pass --approach zeroshot).

Usage (from RDMA repo root):
    python baselines/zeroshot_mimic3_rd_mining.py --mimic3_root /path/to/mimic-iii/1.4
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

from rdma.utils.llm_client import LocalLLMClient  # noqa: E402
from rdma.utils.setup import setup_device  # noqa: E402
from pyhealth.datasets import MIMIC3Dataset  # noqa: E402
from tasks.mimic3_rd_mining import MIMIC3RareDiseaseMining  # noqa: E402

_RESULTS_DIR = Path("/home/johnwu3/projects/rare_disease/workspace/results")
_DEFAULT_ANNOTATION_PATH = str(
    _RDMA_ROOT / "public_data" / "rare_disease_mining" / "mimic3_mining_rdma_human_annotations.json"
)
_DEFAULT_MODEL_CACHE_DIR = "/shared/rsaas/jw3/rare_disease/model_cache"

_SYSTEM = (
    "You are a biomedical named-entity recognition expert. "
    "Your output must be valid JSON only — no prose, no explanation."
)

_PROMPT_TMPL = """\
Extract all rare disease mentions from the clinical discharge note below.

Rules:
- Include specific named rare diseases and syndromes exactly as they appear \
in the text (e.g. "sarcoidosis", "tracheobronchomalacia", "retinitis pigmentosa").
- Include conditions that are recognised as rare diseases in medical literature, \
even if referred to by abbreviation (e.g. "ALS", "HIT").
- Do NOT include common diseases, generic symptoms, body-part terms, or lab \
findings (e.g. "hypertension", "fever", "elevated creatinine").
- Do NOT include anaphoric references (e.g. "the condition", "this disorder").
- Use the exact surface form as it appears in the note.
- If no rare disease entity is present, return an empty array.

Return ONLY a JSON array of strings, for example:
["Disease Name A", "Disease Name B"]

Discharge note:
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


def parse_entities(response: str) -> list:
    """Extract a list of entity strings from the LLM response.

    Tries JSON parsing first, then falls back to extracting quoted strings.
    """
    text = response.strip()

    # 1. Try to find a JSON array anywhere in the response
    match = re.search(r"\[.*?\]", text, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group(0))
            if isinstance(result, list):
                return [str(e).strip() for e in result if e]
        except json.JSONDecodeError:
            pass

    # 2. Fallback: collect all double-quoted tokens
    quoted = re.findall(r'"([^"]+)"', text)
    if quoted:
        return [q.strip() for q in quoted if q.strip()]

    return []


def zero_shot_ner(text: str, llm_client: LocalLLMClient) -> list:
    """Run zero-shot NER on a single note. Returns a list of entity strings."""
    prompt = _PROMPT_TMPL.format(text=text)
    response = llm_client.query(prompt, _SYSTEM)
    return parse_entities(response)


def main():
    parser = argparse.ArgumentParser(
        description="Zero-shot NER baseline for MIMIC-III rare-disease mining"
    )
    parser.add_argument(
        "--mimic3_root",
        type=str,
        required=True,
        help="Root directory of the MIMIC-III 1.4 dataset (contains NOTEEVENTS.csv.gz etc.)",
    )
    parser.add_argument(
        "--annotation_path",
        type=str,
        default=_DEFAULT_ANNOTATION_PATH,
        help="Path to mimic3_mining_rdma_human_annotations.json (default: %(default)s)",
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
        "--temperature",
        type=float,
        default=0.01,
        help="LLM sampling temperature (default: %(default)s)",
    )
    parser.add_argument(
        "--gpu_id",
        type=lambda x: None if x.lower() == "none" else int(x),
        default=0,
        metavar="N|none",
        help="GPU device id; pass 'none' for CPU (default: %(default)s)",
    )
    parser.add_argument(
        "--condor",
        action="store_true",
        help="Running under HTCondor: use generic 'cuda' device instead of cuda:N",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output JSONL path "
            "(default: <results_dir>/mimic3_rd_mining/zeroshot_<model_type>_predictions.jsonl)"
        ),
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume from existing output file"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable verbose debug logging"
    )
    parser.add_argument(
        "--dev", action="store_true", help="Dev mode: process only the first 2 samples"
    )
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=20,
        help="Log a checkpoint every N samples (default: %(default)s)",
    )
    args = parser.parse_args()

    output = args.output or (
        _RESULTS_DIR
        / "mimic3_rd_mining"
        / f"zeroshot_{args.model_type}_predictions.jsonl"
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
    ts(f"Model type       : {args.model_type}")
    ts(f"Temperature      : {args.temperature}")
    ts(f"GPU id           : {args.gpu_id}")
    ts(f"Condor mode      : {args.condor}")
    ts(f"MIMIC-III root   : {args.mimic3_root}")
    ts(f"Annotation path  : {args.annotation_path}")
    ts(f"Output           : {output}")
    ts(f"Resume           : {args.resume}")
    ts(f"Debug            : {args.debug}")
    ts(f"Dev mode         : {args.dev}")
    ts(f"Checkpoint every : {args.checkpoint_interval}")

    # ── Dataset ───────────────────────────────────────────────────────────
    ts("Loading MIMIC3Dataset...")
    dataset = MIMIC3Dataset(
        root=args.mimic3_root,
        tables=["noteevents"],
    )
    task = MIMIC3RareDiseaseMining(annotation_path=args.annotation_path)
    samples = dataset.set_task(task)
    ts(f"  {len(samples)} annotated notes loaded")

    if len(samples) == 0:
        ts("ERROR: no samples found — check mimic3_root and annotation_path")
        sys.exit(1)

    first = next(iter(samples))
    ts(f"  Sample preview — note_id: {first['note_id']!r}  patient_id: {first['patient_id']!r}")
    ts(f"    text[:120]: {pickle.loads(first['text'])[:120]!r}")
    first_annos = pickle.loads(first["annotations"])
    ts(f"    annotations ({len(first_annos)}): {first_annos[:3]}")

    # ── LLM ───────────────────────────────────────────────────────────────
    ts(f"Loading LLM ({args.model_type})...")
    llm_client = LocalLLMClient(
        model_type=args.model_type,
        device=devices["llm"],
        cache_dir=args.model_cache_dir,
        temperature=args.temperature,
    )

    # ── Run ───────────────────────────────────────────────────────────────
    done_ids = load_done_ids(output) if args.resume else set()
    if args.resume:
        ts(f"Resuming – {len(done_ids)} already done")

    output.parent.mkdir(parents=True, exist_ok=True)
    out_f = open(output, "a" if args.resume else "w", encoding="utf-8")

    try:
        run_samples = samples.subset(slice(0, 2)) if args.dev else samples
        for i, sample in enumerate(tqdm(run_samples, total=len(run_samples), desc="MIMIC3-RD-ZS")):
            try:
                note_id = sample["note_id"]
                text = pickle.loads(sample["text"])
            except Exception as e:
                ts(f"  SKIP sample {i} (data error): {e}")
                if args.debug:
                    traceback.print_exc()
                continue

            if note_id in done_ids:
                continue

            try:
                predicted = zero_shot_ner(text, llm_client)
                if args.debug:
                    ts(f"  [{note_id}] {len(predicted)} entities")
            except Exception as e:
                ts(f"  ERROR [{note_id}]: {e}")
                if args.debug:
                    traceback.print_exc()
                predicted = []

            out_f.write(
                json.dumps(
                    {
                        "id": note_id,
                        "patient_id": sample["patient_id"],
                        "predicted": predicted,
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

    ts(f"Done → {output}")


if __name__ == "__main__":
    main()
