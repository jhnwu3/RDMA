#!/usr/bin/env python3
"""
Run the HPO phenotype-mining pipeline on the BioLark GSC benchmark.

Usage (from RDMA repo root):
  python scripts/biolarkgsc/run_hpo.py
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
from rdma.hpo.verifier import HPOVerifier  # noqa: E402
from rdma.hpo.matcher import HPOMatcher  # noqa: E402
from rdma.utils.llm_client import (  # noqa: E402
    LocalLLMClient,
    APILLMClient,
    OpenRouterLLMClient,
    AzureOpenAILLMClient,
    LlamaCppLLMClient,
)
from rdma.utils.setup import setup_device  # noqa: E402

from datasets.biolarkgsc import BioLarkGSCDataset  # noqa: E402
from tasks.biolarkgsc import BioLarkGSCNER  # noqa: E402
from scripts.biolarkgsc.eval import _build_hpo_lookup  # noqa: E402

_RESULTS_DIR = Path("/home/johnwu3/projects/rare_disease/workspace/results")
_DEFAULT_ONTOLOGY = _RDMA_ROOT / "data" / "ontology" / "hpo_data_with_lineage.json"
_DEFAULT_EMBEDDINGS_FILE = str(
    _RDMA_ROOT / "data" / "vector_stores" / "G2GHPO_metadata_medembed.npy"
)
_DEFAULT_LAB_EMBEDDINGS_FILE = str(
    _RDMA_ROOT / "data" / "tools" / "lab_tables_medembed_sm.npy"
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
    parser = argparse.ArgumentParser(description="HPO pipeline baseline on BioLark GSC")
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
        help="Path to saved API config JSON (api/openrouter/azure)",
    )
    parser.add_argument(
        "--dotenv_path",
        type=str,
        default=None,
        help="Optional dotenv file for API backends (e.g. PyHealthAgent/.env)",
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
        "--extraction_temperature",
        type=float,
        default=0.001,
        help="LLM sampling temperature for entity extraction (default: %(default)s)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.01,
        help="LLM sampling temperature for verification and matching (default: %(default)s)",
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
        "--lab_embeddings_file",
        type=Path,
        default=Path(_DEFAULT_LAB_EMBEDDINGS_FILE),
        help="Path to lab-tables .npy embeddings file (default: %(default)s)",
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
        default=20,
        help="Top-k retrieved candidates (default: %(default)s)",
    )
    parser.add_argument(
        "--multi_match",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "If true, allow matcher to return multiple valid HPO IDs per "
            "entity. Use --no-multi_match to force a single best match "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--verifier_version",
        type=str,
        default="v4",
        help="HPOVerifier version (default: %(default)s)",
    )
    parser.add_argument(
        "--matcher_version",
        type=str,
        default="standard",
        choices=["standard", "optimized"],
        help="HPOMatcher optimizer version (default: %(default)s)",
    )
    parser.add_argument(
        "--use_demographics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable demographic extraction for v4 implied"
            " lab-test reasoning (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--negation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "If true, exclude negated findings during extraction. "
            "Use --negation to enable this filter (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--family_history",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "If true, exclude family-history findings during extraction. "
            "Use --family_history to enable this filter (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--decompose_compound",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "If true, prompt the extractor to decompose named syndromes into "
            "their explicitly described component phenotypes (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--extract_qualified",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "If true, prompt the extractor to extract fully qualified phrases "
            "and inheritance pattern terms (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--allow_inheritance",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "If true, allow the verifier to accept inheritance pattern, "
            "expressivity, and onset modifier terms as valid HPO phenotypes "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--parent_match",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "If true, expand each matched HPO term with its immediate parent "
            "terms from the HPO ontology (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--exhaustive",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "If true, instruct the extractor to prefer recall over precision — "
            "include borderline terms rather than omit them (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--exclude_etiology",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "If true, instruct the extractor to exclude molecular mechanisms and "
            "chromosomal etiology terms (e.g., trisomy, UPD) from extraction. "
            "Use --no-exclude_etiology to allow them (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--prefer_specific",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "If true, instruct the matcher to prefer the most specific HPO term "
            "supported by the text over parent/ancestor terms. "
            "Use --no-prefer_specific to disable (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--require_grounding",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "If true, the matcher will output NO_MATCH when no candidate is clearly "
            "grounded in the text, suppressing the fallback to the top candidate. "
            "Use --require_grounding to enable (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--context_window_size",
        type=int,
        default=0,
        help=(
            "Number of sentences of context to include around each extracted entity "
            "(default: %(default)s — entity sentence only)"
        ),
    )
    parser.add_argument(
        "--require_text_match",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "If true, drop extracted entities that do not appear verbatim as a "
            "substring of the source text (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--fuzzy_threshold",
        type=float,
        default=0.93,
        help=(
            "Fuzzy similarity threshold (0–1) used in the verifier's fast-match "
            "stage; higher = stricter (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output JSONL path "
            "(default: <results_dir>/biolarkgsc/<model_type>_predictions.jsonl)"
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
        help="Dev mode: process only the first --dev_n samples",
    )
    parser.add_argument(
        "--dev_n",
        type=int,
        default=2,
        help="Number of samples to use when --dev is active (default: %(default)s)",
    )
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=20,
        help="Log a checkpoint every N samples (default: %(default)s)",
    )
    args = parser.parse_args()

    output = args.output or (
        _RESULTS_DIR / "biolarkgsc" / f"{args.model_type}_predictions.jsonl"
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
    ts(f"Extraction temp   : {args.extraction_temperature}")
    ts(f"Verify/match temp : {args.temperature}")
    ts(f"GPU id            : {args.gpu_id}")
    ts(f"Condor mode       : {args.condor}")
    ts(f"Model cache dir   : {args.model_cache_dir}")
    ts(f"Dataset cache dir : {args.dataset_cache_dir}")
    ts(f"HPO embeddings    : {args.embeddings_file}")
    ts(f"Lab embeddings    : {args.lab_embeddings_file}")
    ts(f"Retriever         : {args.retriever} / {args.retriever_model}")
    ts(f"Extractor         : {args.entity_extractor}")
    ts(f"Verifier version  : {args.verifier_version}")
    ts(f"Matcher version   : {args.matcher_version}")
    ts(f"Use demographics  : {args.use_demographics}")
    ts(f"Exclude negation  : {args.negation}")
    ts(f"Exclude fam hist  : {args.family_history}")
    ts(f"Decompose compound: {args.decompose_compound}")
    ts(f"Extract qualified : {args.extract_qualified}")
    ts(f"Allow inheritance : {args.allow_inheritance}")
    ts(f"Parent match      : {args.parent_match}")
    ts(f"Exhaustive        : {args.exhaustive}")
    ts(f"Exclude etiology  : {args.exclude_etiology}")
    ts(f"Prefer specific   : {args.prefer_specific}")
    ts(f"Require grounding : {args.require_grounding}")
    ts(f"Context window    : {args.context_window_size}")
    ts(f"Require text match: {args.require_text_match}")
    ts(f"Fuzzy threshold   : {args.fuzzy_threshold}")
    ts(f"Top-k             : {args.top_k}")
    ts(f"Multi match       : {args.multi_match}")
    ts(f"Output            : {output}")
    ts(f"Resume            : {args.resume}")
    ts(f"Debug             : {args.debug}")
    ts(f"Dev mode          : {args.dev}")
    ts(f"Checkpoint every  : {args.checkpoint_interval}")

    # ── HPO label lookup (for debug output) ───────────────────────────────
    hpo_lookup: dict = {}
    if args.debug and _DEFAULT_ONTOLOGY.exists():
        hpo_lookup = _build_hpo_lookup(_DEFAULT_ONTOLOGY)
        ts(f"Loaded HPO ontology labels ({len(hpo_lookup)} terms)")

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
    def _make_llm_client(temperature):
        if args.llm_type == "api":
            return (
                APILLMClient.from_config(args.api_config)
                if args.api_config
                else APILLMClient(model_type=args.model_type, temperature=temperature)
            )
        elif args.llm_type == "openrouter":
            return (
                OpenRouterLLMClient.from_config(args.api_config)
                if args.api_config
                else OpenRouterLLMClient(
                    model_type=args.model_type, temperature=temperature
                )
            )
        elif args.llm_type == "azure":
            return (
                AzureOpenAILLMClient.from_config(args.api_config)
                if args.api_config
                else AzureOpenAILLMClient(
                    model_type=args.model_type,
                    azure_deployment=args.model_type,
                    temperature=temperature,
                    dotenv_path=args.dotenv_path,
                )
            )
        elif args.llm_type == "llama_cpp":
            return LlamaCppLLMClient(
                model_type=args.model_type,
                gguf_file=args.gguf_file,
                main_gpu=args.gpu_id if args.gpu_id is not None else 0,
                temperature=temperature,
                cache_dir=args.model_cache_dir,
            )
        else:
            return LocalLLMClient(
                model_type=args.model_type,
                device=devices["llm"],
                cache_dir=args.model_cache_dir,
                temperature=temperature,
            )

    ts(f"Loading LLM ({args.llm_type} / {args.model_type})")
    extraction_llm_client = _make_llm_client(args.extraction_temperature)
    llm_client = _make_llm_client(args.temperature)

    ts("Initialising HPO pipeline…")
    extractor = PhenotypeExtractor(
        llm_client=extraction_llm_client,
        extractor_type=args.entity_extractor,
        embeddings_file=str(args.embeddings_file),
        retriever=args.retriever,
        retriever_model=args.retriever_model,
        top_k=args.top_k,
        negation=args.negation,
        family_history=args.family_history,
        debug=args.debug,
        decompose_compound=args.decompose_compound,
        extract_qualified=args.extract_qualified,
        exhaustive=args.exhaustive,
        exclude_etiology=args.exclude_etiology,
        window_size=args.context_window_size,
    )
    verifier = HPOVerifier(
        llm_client=llm_client,
        embeddings_file=str(args.embeddings_file),
        verifier_version=args.verifier_version,
        lab_embeddings_file=str(args.lab_embeddings_file),
        retriever=args.retriever,
        retriever_model=args.retriever_model,
        debug=args.debug,
        use_demographics=args.use_demographics,
        allow_inheritance=args.allow_inheritance,
        require_text_match=args.require_text_match,
        fuzzy_threshold=args.fuzzy_threshold,
    )
    matcher = HPOMatcher(
        llm_client=llm_client,
        embeddings_file=str(args.embeddings_file),
        optimizer_version=args.matcher_version,
        retriever=args.retriever,
        retriever_model=args.retriever_model,
        top_k=args.top_k,
        multi_match=args.multi_match,
        parent_match=args.parent_match,
        debug=args.debug,
        prefer_specific=args.prefer_specific,
        require_grounding=args.require_grounding,
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
        run_samples = samples.subset(slice(0, args.dev_n)) if args.dev else samples
        for i, sample in enumerate(
            tqdm(run_samples, total=len(run_samples), desc="BioLarkGSC")
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

            extract_s = verify_s = match_s = 0.0
            entities_with_contexts: list = []
            verified_phenotypes: list = []
            matched_phenotypes: list = []
            try:
                t0 = time.perf_counter()
                entities_with_contexts = extractor.extract([text])
                extract_s = time.perf_counter() - t0
                if args.debug:
                    ts(f"  [{doc_id}] extracted {len(entities_with_contexts)}")

                t0 = time.perf_counter()
                verified_phenotypes = verifier.verify(entities_with_contexts, text)
                verify_s = time.perf_counter() - t0
                if args.debug:
                    ts(f"  [{doc_id}] verified  {len(verified_phenotypes)}")

                t0 = time.perf_counter()
                matched_phenotypes = matcher.match(verified_phenotypes)
                match_s = time.perf_counter() - t0
                if args.debug:
                    ts(f"  [{doc_id}] matched   {len(matched_phenotypes)}")

                ts(
                    f"  [{doc_id}] "
                    f"extract={extract_s:.2f}s  "
                    f"verify={verify_s:.2f}s  "
                    f"match={match_s:.2f}s"
                )
                if args.debug:
                    pred_set = set()
                    for m in matched_phenotypes:
                        hps = m.get("hp_ids") or (
                            [m.get("hp_id")] if m.get("hp_id") else []
                        )
                        for hp in hps:
                            if hp:
                                pred_set.add(hp)
                    gold_set = set(h for h in ground_truth if h)
                    fp_codes = pred_set - gold_set
                    fn_codes = gold_set - pred_set

                    def _label(hp):
                        return hpo_lookup.get(hp, hp)

                    ts(
                        f"  [{doc_id}] GT  : {[(h, _label(h)) for h in sorted(gold_set)] or '(none)'}"
                    )
                    ts(
                        f"  [{doc_id}] FP  : {[(h, _label(h)) for h in sorted(fp_codes)] or '(none)'}"
                    )
                    ts(
                        f"  [{doc_id}] FN  : {[(h, _label(h)) for h in sorted(fn_codes)] or '(none)'}"
                    )
                    if fp_codes:
                        ts(f"  [{doc_id}] FP spans:")
                        for m in matched_phenotypes:
                            hps = m.get("hp_ids") or (
                                [m.get("hp_id")] if m.get("hp_id") else []
                            )
                            span = m.get("phenotype") or m.get("entity", "")
                            for hp in hps:
                                if hp in fp_codes:
                                    ts(f'    {hp}  ({_label(hp)})  "{span}"')
                if args.parent_match or args.multi_match:
                    predicted = list(
                        {
                            hp
                            for m in matched_phenotypes
                            for hp in (
                                m.get("hp_ids")
                                or ([m.get("hp_id")] if m.get("hp_id") else [])
                            )
                            if hp
                        }
                    )
                else:
                    predicted = [
                        m.get("hp_id", "") for m in matched_phenotypes if m.get("hp_id")
                    ]
            except Exception as e:
                ts(f"  ERROR [{doc_id}]: {e}")
                if args.debug:
                    traceback.print_exc()
                predicted = []

            timings.append((extract_s, verify_s, match_s))
            records.append(
                {"id": doc_id, "predicted": predicted, "ground_truth": ground_truth}
            )
            record = {
                "id": doc_id,
                "predicted": predicted,
                "ground_truth": ground_truth,
                "timing": {
                    "extraction_s": round(extract_s, 3),
                    "verification_s": round(verify_s, 3),
                    "matching_s": round(match_s, 3),
                },
            }
            if args.debug:
                record["stages"] = {
                    "extraction": [
                        {"entity": e.get("entity"), "context": e.get("context", "")}
                        for e in entities_with_contexts
                    ],
                    "verification": [
                        {
                            "phenotype": v.get("phenotype"),
                            "original_entity": v.get("original_entity"),
                            "status": v.get("status"),
                            "confidence": v.get("confidence"),
                            "method": v.get("method"),
                            "lab_info": v.get("lab_info"),
                        }
                        for v in verified_phenotypes
                    ],
                    "matching": [
                        {
                            "phenotype": m.get("phenotype"),
                            "original_entity": m.get("original_entity"),
                            "hp_id": m.get("hp_id"),
                            "hp_ids": m.get("hp_ids"),
                            "match_method": m.get("match_method"),
                            "confidence_score": m.get("confidence_score"),
                            "top_candidates": [
                                {
                                    "hp_id": c.get("metadata", {}).get("hp_id"),
                                    "label": c.get("metadata", {}).get("info"),
                                    "score": c.get("similarity_score"),
                                }
                                for c in (m.get("top_candidates") or [])[:3]
                            ],
                        }
                        for m in matched_phenotypes
                    ],
                }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()

            if (i + 1) % args.checkpoint_interval == 0:
                ts(f"Checkpoint {i + 1}/{len(samples)}")
    finally:
        out_f.close()

    if timings:
        avg_e = sum(t[0] for t in timings) / len(timings)
        avg_v = sum(t[1] for t in timings) / len(timings)
        avg_m = sum(t[2] for t in timings) / len(timings)
        ts("── Timing summary ──────────────────────────────────────────")
        ts(f"  Samples           : {len(timings)}")
        ts(f"  Avg extraction    : {avg_e:.2f}s")
        ts(f"  Avg verification  : {avg_v:.2f}s")
        ts(f"  Avg matching      : {avg_m:.2f}s")
        ts(f"  Avg total/sample  : {avg_e + avg_v + avg_m:.2f}s")

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

    if args.debug and records:

        def _label(hp):
            return hpo_lookup.get(hp, hp)

        ts("── Per-doc ground truth / FP / FN summary ──────────────────")
        for rec in records:
            doc = rec.get("id", "?")
            pred_set = set(h for h in rec["predicted"] if h)
            gold_set = set(h for h in rec["ground_truth"] if h)
            fp_codes = pred_set - gold_set
            fn_codes = gold_set - pred_set
            ts(f"  [{doc}]")
            ts(f"    GT  : {[(h, _label(h)) for h in sorted(gold_set)] or '(none)'}")
            ts(f"    FP  : {[(h, _label(h)) for h in sorted(fp_codes)] or '(none)'}")
            ts(f"    FN  : {[(h, _label(h)) for h in sorted(fn_codes)] or '(none)'}")

    ts(f"Done → {output}")


if __name__ == "__main__":
    main()
