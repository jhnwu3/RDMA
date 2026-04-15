#!/usr/bin/env python3
"""
Run the RDMA rare-disease extraction pipeline on the RDD benchmark.

Usage (from RDMA repo root):
  python scripts/run_rdd.py
  python scripts/run_rdd.py --task relation
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
from rdma.rd.verifier import RDMAVerifier  # noqa: E402
from rdma.utils.embedding import EmbeddingsManager  # noqa: E402
from rdma.utils.llm_client import LocalLLMClient  # noqa: E402
from rdma.utils.setup import setup_device  # noqa: E402

from datasets.rdd import RDDDataset  # noqa: E402
from tasks.rdd import RDDNER, RDDRelationExtraction  # noqa: E402

_RESULTS_DIR = Path("/home/johnwu3/projects/rare_disease/workspace/results")
_DEFAULT_EMBEDDINGS_FILE = (
    "/home/johnwu3/projects/rare_disease/workspace/repos/RDMA"
    "/data/vector_stores/rd_orpha_medembed.npy"
)
_DEFAULT_ABBREVIATIONS_FILE = (
    "/home/johnwu3/projects/rare_disease/workspace/repos/RDMA"
    "/data/tools/abbreviations_medembed_sm.npy"
)
_DEFAULT_MODEL_CACHE_DIR = "/shared/rsaas/jw3/rare_disease/model_cache"
_DEFAULT_DATASET_CACHE_DIR = "/shared/eng/pyhealth/rdd"

# ── Relation classification prompt ───────────────────────────────────────────

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


def _rel_sample_id(sample: dict) -> str:
    """Stable unique ID for a relation sample: doc_id|rd_start|dis_start."""
    return (
        f"{sample['patient_id']}"
        f"|{sample['rd_start']}|{sample['dis_start']}"
    )


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

# ─────────────────────────────────────────────────────────────────────────────


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
    parser = argparse.ArgumentParser(description="RDMA pipeline on RDD")
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
        default=0,
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
        "--entity_extractor",
        type=str,
        default="retrieval",
        help="Entity extraction method (default: %(default)s)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Top-k retrieved documents (default: %(default)s)",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=5,
        help="Sentence window size (default: %(default)s)",
    )
    parser.add_argument(
        "--min_sentence_size",
        type=lambda x: None if x.lower() == "none" else int(x),
        default=50,
        metavar="N|none",
        help=(
            "Minimum sentence size; 'none' to disable "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output JSONL path "
            "(default: <results_dir>/rdd/<model_type>[_rel]_predictions.jsonl)"
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
    parser.add_argument(
        "--verifier_type",
        type=str,
        default="multi_stage",
        help="Verifier type (default: %(default)s)",
    )
    parser.add_argument(
        "--abbreviations_file",
        type=str,
        default=_DEFAULT_ABBREVIATIONS_FILE,
        help=(
            "Path to abbreviations embeddings .npy file "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--use_abbreviations",
        action="store_true",
        help="Enable abbreviation resolution in the multi-stage verifier",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Enable strict ORPHA-matching in the verifier "
            "(default: lax/best-judgement mode)"
        ),
    )
    parser.add_argument(
        "--exact_match",
        action="store_true",
        help=(
            "Accept entities with an exact/high-similarity ORPHA string match "
            "immediately, skipping all LLM verification calls (faster, no LLM)"
        ),
    )
    parser.add_argument(
        "--disease_check",
        action="store_true",
        help=(
            "Run a final LLM disease-check gate (Stage 3) when Stages 1 and 2 "
            "did not return early (default: disabled)"
        ),
    )
    args = parser.parse_args()

    task_suffix = "_rel" if args.task == "relation" else ""
    output = args.output or (
        _RESULTS_DIR / "rdd"
        / f"{args.model_type}{task_suffix}_predictions.jsonl"
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
    if args.task == "ner":
        ts(f"Embeddings       : {args.embeddings_file}")
        ts(f"Retriever        : {args.retriever} / {args.retriever_model}")
        ts(f"Extractor        : {args.entity_extractor}")
        ts(f"Top-k            : {args.top_k}")
        ts(f"Window size      : {args.window_size}")
        ts(f"Min sentence     : {args.min_sentence_size}")
        ts(f"Verifier type    : {args.verifier_type}")
        ts(f"Abbreviations    : {args.abbreviations_file or 'disabled'}")
        ts(f"Use abbrev.      : {args.use_abbreviations}")
        ts(f"Strict verifier  : {args.strict}")
        ts(f"Exact match      : {args.exact_match}")
        ts(f"Disease check    : {args.disease_check}")

    # ── Dataset ──────────────────────────────────────────────────────────
    ts(f"Dataset cache dir : {args.dataset_cache_dir}")
    ts("Loading RDDDataset...")
    dataset = RDDDataset(cache_dir=args.dataset_cache_dir)
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
        anns = pickle.loads(first["annotations"])
        ts(f"    annotations ({len(anns)}): {anns[:5]}")

    # ── Pipeline ─────────────────────────────────────────────────────────
    ts(f"Loading LLM ({args.model_type})")
    llm_client = LocalLLMClient(
        model_type=args.model_type,
        device=devices["llm"],
        cache_dir=args.model_cache_dir,
        temperature=args.temperature,
    )

    if args.task == "ner":
        ts(f"Loading embeddings: {args.embeddings_file}")
        embedded_documents = np.load(
            args.embeddings_file, allow_pickle=True
        )
        ts(f"  {len(embedded_documents)} documents")

        embedding_manager = EmbeddingsManager(
            model_type=args.retriever,
            model_name=args.retriever_model,
            device=devices.get("retriever", devices["llm"]),
        )

        ts("Initialising pipeline…")
        extractor = RDMAExtractor(
            llm_client=llm_client,
            extraction_method=args.entity_extractor,
            embedding_manager=embedding_manager,
            embedded_documents=embedded_documents,
            window_size=args.window_size,
            top_k=args.top_k,
            min_sentence_size=args.min_sentence_size,
            strict=args.strict,
            debug=args.debug,
        )
        verifier = RDMAVerifier(
            llm_client=llm_client,
            embedding_manager=embedding_manager,
            embedded_documents=embedded_documents,
            verifier_type=args.verifier_type,
            abbreviations_file=args.abbreviations_file,
            use_abbreviations=args.use_abbreviations,
            strict=args.strict,
            exact_match=args.exact_match,
            disease_check=args.disease_check,
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
        run_samples = (
            samples.subset(slice(0, 2)) if args.dev else samples
        )
        for i, sample in enumerate(
            tqdm(run_samples, total=len(run_samples), desc="RDD")
        ):
            if args.task == "relation":
                sample_id = _rel_sample_id(sample)
                if sample_id in done_ids:
                    continue

                classify_s = 0.0
                try:
                    t0 = time.perf_counter()
                    predicted = classify_relation(sample, llm_client)
                    classify_s = time.perf_counter() - t0
                    if args.debug:
                        ts(f"  [{sample_id}] predicted={predicted}")
                    ts(
                        f"  [{sample_id}] "
                        f"classify={classify_s:.2f}s"
                    )
                except Exception as e:
                    ts(f"  ERROR [{sample_id}]: {e}")
                    if args.debug:
                        traceback.print_exc()
                    predicted = 0

                timings.append((classify_s,))
                out_f.write(
                    json.dumps(
                        {
                            "id": sample_id,
                            "predicted": predicted,
                            "rare_disease": sample["rare_disease"],
                            "disability": sample["disability"],
                            "timing": {
                                "classification_s": round(classify_s, 3),
                            },
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            else:  # ner
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

                extract_s = verify_s = 0.0
                try:
                    t0 = time.perf_counter()
                    entities_with_contexts = extractor.extract_from_text(
                        text
                    )
                    extract_s = time.perf_counter() - t0
                    if args.debug:
                        ts(
                            f"  [{doc_id}] extracted "
                            f"{len(entities_with_contexts)}"
                        )

                    t0 = time.perf_counter()
                    verified = verifier.verify_entities(
                        entities_with_contexts
                    )
                    verify_s = time.perf_counter() - t0
                    if args.debug:
                        ts(f"  [{doc_id}] verified  {len(verified)}")

                    ts(
                        f"  [{doc_id}] "
                        f"extract={extract_s:.2f}s  "
                        f"verify={verify_s:.2f}s"
                    )
                    predicted = [v.get("entity", "") for v in verified]
                except Exception as e:
                    ts(f"  ERROR [{doc_id}]: {e}")
                    if args.debug:
                        traceback.print_exc()
                    predicted = []

                timings.append((extract_s, verify_s))
                out_f.write(
                    json.dumps(
                        {
                            "id": doc_id,
                            "predicted": predicted,
                            "timing": {
                                "extraction_s": round(extract_s, 3),
                                "verification_s": round(verify_s, 3),
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
        ts("── Timing summary ──────────────────────────────────────────")
        ts(f"  Samples          : {len(timings)}")
        if args.task == "relation":
            avg_c = sum(t[0] for t in timings) / len(timings)
            ts(f"  Avg classify     : {avg_c:.2f}s")
        else:
            avg_e = sum(t[0] for t in timings) / len(timings)
            avg_v = sum(t[1] for t in timings) / len(timings)
            ts(f"  Avg extraction   : {avg_e:.2f}s")
            ts(f"  Avg verification : {avg_v:.2f}s")
            ts(f"  Avg total/sample : {avg_e + avg_v:.2f}s")

    ts(f"Done → {output}")


if __name__ == "__main__":
    main()
