#!/usr/bin/env python3
"""
Zero-shot LLM baseline for the RDD benchmark.

Supports two tasks:
  ner      — extract disability/disease mentions from a full abstract.
  relation — classify whether a (rare disease, disability) pair in a
             sentence represents a confirmed association (1) or not (0).

Output is a JSONL file consumable by eval_rdd.py:
  NER:      {"id": "<doc_id>",    "predicted": ["entity1", ...]}
  Relation: {"id": "<sample_id>", "predicted": 0|1}

Usage (from RDMA repo root):
    python baselines/zeroshot_rdd.py
    python baselines/zeroshot_rdd.py --task relation
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
from datasets.rdd import RDDDataset  # noqa: E402
from tasks.rdd import RDDNER, RDDRelationExtraction  # noqa: E402

_RESULTS_DIR = Path("/home/johnwu3/projects/rare_disease/workspace/results")
_DEFAULT_MODEL_CACHE_DIR = "/shared/rsaas/jw3/rare_disease/model_cache"

# ── NER prompt ───────────────────────────────────────────────────────────────

_NER_SYSTEM = (
    "You are a biomedical named-entity recognition expert. "
    "Your output must be valid JSON only — no prose, no explanation."
)

_NER_PROMPT_TMPL = """\
Extract all disability and disease condition mentions from the biomedical \
abstract below.

Rules:
- Include full expressions for conditions and diseases as they appear in \
the text (e.g. "developmental delay", "intellectual disability", \
"hearing loss").
- Do NOT include isolated body-function terms without a disorder sense \
(e.g. do not include "hearing" or "intellectual" alone).
- Do NOT include negation markers, speculation scopes, or abbreviation \
expansions as separate entities.
- Use the exact surface form from the abstract.
- If no disability or disease entity is present, return an empty array.

Return ONLY a JSON array of strings, for example:
["disability expression A", "disability expression B"]

Abstract:
{text}"""

# ── Relation prompt ──────────────────────────────────────────────────────────

_REL_SYSTEM = (
    "You are a biomedical expert. "
    "Your response must be exactly one word: YES or NO. "
    "Do not include any other text, punctuation, or explanation."
)

_REL_PROMPT_TMPL = """\
Sentence: "{sentence}"

In this sentence, is the rare disease "{rare_disease}" directly \
characterized by or associated with the disability/condition "{disability}"?

Answer YES if the sentence states or implies a confirmed association. \
Answer NO otherwise."""

# ─────────────────────────────────────────────────────────────────────────────


def _rel_sample_id(sample: dict) -> str:
    """Stable unique ID for a relation sample: doc_id|rd_start|dis_start."""
    return (
        f"{sample['patient_id']}"
        f"|{sample['rd_start']}|{sample['dis_start']}"
    )


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
    """Run zero-shot NER on a single document. Returns a list of strings."""
    prompt = _NER_PROMPT_TMPL.format(text=text)
    response = llm_client.query(prompt, _NER_SYSTEM)
    return parse_entities(response)


def classify_relation(sample: dict, llm_client: LocalLLMClient) -> int:
    """Ask the LLM to classify one (rare disease, disability) pair.

    Returns 1 for a positive (confirmed) relationship, 0 for negative.
    """
    sentence = pickle.loads(sample["text"])
    prompt = _REL_PROMPT_TMPL.format(
        sentence=sentence,
        rare_disease=sample["rare_disease"],
        disability=sample["disability"],
    )
    resp = llm_client.query(prompt, _REL_SYSTEM).strip().upper()
    return 1 if resp.startswith("YES") else 0


def main():
    parser = argparse.ArgumentParser(
        description="Zero-shot LLM baseline for RDD"
    )
    parser.add_argument(
        "--task",
        choices=["ner", "relation"],
        default="ner",
        help="Task to run: 'ner' (default) or 'relation' classification",
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
        default=2,
        metavar="N|none",
        help="GPU device id; pass 'none' for CPU (default: %(default)s)",
    )
    parser.add_argument(
        "--condor",
        action="store_true",
        help=(
            "Running under HTCondor: use generic 'cuda' device "
            "instead of cuda:N"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output JSONL path "
            "(default: <results_dir>/rdd/"
            "zeroshot_<model_type>[_rel]_predictions.jsonl)"
        ),
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from existing output file",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable verbose debug logging",
    )
    parser.add_argument(
        "--dev", action="store_true",
        help="Dev mode: process only the first 2 samples",
    )
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=20,
        help="Log a checkpoint every N samples (default: %(default)s)",
    )
    args = parser.parse_args()

    task_suffix = "_rel" if args.task == "relation" else ""
    output = args.output or (
        _RESULTS_DIR / "rdd"
        / f"zeroshot_{args.model_type}{task_suffix}_predictions.jsonl"
    )

    cfg = SimpleNamespace(
        gpu_id=args.gpu_id,
        condor=args.condor,
        cpu=(args.gpu_id is None and not args.condor),
        retriever_gpu_id=None,
        retriever_cpu=False,
    )
    devices = setup_device(cfg)
    ts(f"Task             : {args.task}")
    ts(f"LLM device       : {devices['llm']}")
    ts(f"Model type       : {args.model_type}")
    ts(f"Temperature      : {args.temperature}")
    ts(f"GPU id           : {args.gpu_id}")
    ts(f"Condor mode      : {args.condor}")
    ts(f"Output           : {output}")
    ts(f"Resume           : {args.resume}")
    ts(f"Debug            : {args.debug}")
    ts(f"Dev mode         : {args.dev}")
    ts(f"Checkpoint every : {args.checkpoint_interval}")

    # ── Dataset ───────────────────────────────────────────────────────────
    ts("Loading RDDDataset...")
    dataset = RDDDataset()
    if args.task == "relation":
        samples = dataset.set_task(RDDRelationExtraction())
    else:
        samples = dataset.set_task(RDDNER())
    ts(f"  {len(samples)} samples loaded")

    first = next(iter(samples))
    ts(f"  Sample preview — id: {first['patient_id']!r}")
    ts(f"    text[:120]: {pickle.loads(first['text'])[:120]!r}")
    if args.task == "relation":
        ts(
            f"    rare_disease: {first['rare_disease']!r}  "
            f"disability: {first['disability']!r}  "
            f"label: {pickle.loads(first['label'])}"
        )
    else:
        ann_preview = pickle.loads(first["annotations"])[:5]
        ts(
            f"    annotations "
            f"({len(pickle.loads(first['annotations']))}): {ann_preview}"
        )

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
        run_samples = (
            samples.subset(slice(0, 2)) if args.dev else samples
        )
        for i, sample in enumerate(
            tqdm(run_samples, total=len(run_samples), desc="RDD-ZS")
        ):
            if args.task == "relation":
                sample_id = _rel_sample_id(sample)
                if sample_id in done_ids:
                    continue

                try:
                    predicted = classify_relation(sample, llm_client)
                    if args.debug:
                        ts(f"  [{sample_id}] predicted={predicted}")
                except Exception as e:
                    ts(f"  ERROR [{sample_id}]: {e}")
                    if args.debug:
                        traceback.print_exc()
                    predicted = 0

                out_f.write(
                    json.dumps(
                        {
                            "id": sample_id,
                            "predicted": predicted,
                            "rare_disease": sample["rare_disease"],
                            "disability": sample["disability"],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            else:  # ner
                try:
                    doc_id = sample["patient_id"]
                    text = sample["text"]
                except Exception as e:
                    ts(f"  SKIP sample {i} (data error): {e}")
                    if args.debug:
                        traceback.print_exc()
                    continue

                if doc_id in done_ids:
                    continue

                try:
                    predicted = zero_shot_ner(text, llm_client)
                    if args.debug:
                        ts(f"  [{doc_id}] {len(predicted)} entities")
                except Exception as e:
                    ts(f"  ERROR [{doc_id}]: {e}")
                    if args.debug:
                        traceback.print_exc()
                    predicted = []

                out_f.write(
                    json.dumps(
                        {"id": doc_id, "predicted": predicted},
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
