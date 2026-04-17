#!/usr/bin/env python3
"""Run a 4-agent RDMA demonstration workflow on MIMIC-III notes.

The script supports an unlabeled full-notes task mode for inference/demo runs,
and optional labeled task modes for compatibility with existing pipelines.
"""

import argparse
import json
import pickle
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import numpy as np
from tqdm import tqdm

_RDMA_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_RDMA_ROOT))

from pyhealth.datasets import MIMIC3Dataset  # noqa: E402
from rdma.rd.extractor import RDMAExtractor  # noqa: E402
from rdma.rd.matcher import RDMAMatcher  # noqa: E402
from rdma.rd.supervisor import RDMASupervisor  # noqa: E402
from rdma.rd.verifier import RDMAVerifier  # noqa: E402
from rdma.utils.embedding import EmbeddingsManager  # noqa: E402
from rdma.utils.llm_client import (  # noqa: E402
    APILLMClient,
    LlamaCppLLMClient,
    LocalLLMClient,
    OpenRouterLLMClient,
)
from rdma.utils.setup import setup_device  # noqa: E402
from tasks.mimic3_full_notes_unlabeled import (  # noqa: E402
    MIMIC3FullNotesUnlabeled,
)
from tasks.mimic3_rd_mining import MIMIC3RareDiseaseMining  # noqa: E402
from tasks.mimic3_rd_mining_text import MIMIC3RDMiningText  # noqa: E402


_DEFAULT_RESULTS_DIR = _RDMA_ROOT.parent / "results" / "mimic3_demo"
_DEFAULT_CACHE_DIR = "/shared/eng/pyhealth/mimic3"
_DEFAULT_EMBEDDINGS_FILE = (
    _RDMA_ROOT / "data" / "vector_stores" / "rd_orpha_medembed.npy"
)
_DEFAULT_ABBREVIATIONS_FILE = (
    _RDMA_ROOT / "data" / "tools" / "abbreviations_medembed_sm.npy"
)
_DEFAULT_ANNOTATION_PATH = (
    _RDMA_ROOT
    / "public_data"
    / "rare_disease_mining"
    / "mimic3_mining_rdma_human_annotations.json"
)
_DEFAULT_MODEL_CACHE_DIR = "/shared/rsaas/jw3/rare_disease/model_cache"


def ts(message: str) -> None:
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S} - {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RDMA 4-agent demo workflow on MIMIC-III via PyHealth"
    )
    parser.add_argument("--mimic3_root", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, default=_DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--task_mode",
        choices=["unlabeled", "mimic3_rd_mining", "mimic3_rd_mining_text"],
        default="unlabeled",
        help="PyHealth task mode (default: unlabeled inference demo)",
    )
    parser.add_argument(
        "--annotation_path",
        type=str,
        default=str(_DEFAULT_ANNOTATION_PATH),
        help="Only used for labeled MIMIC3-RD task modes",
    )
    parser.add_argument(
        "--discharge_only",
        action="store_true",
        help="For unlabeled mode, include only 'Discharge summary' notes",
    )
    parser.add_argument(
        "--note_categories",
        nargs="*",
        default=None,
        help="Optional category allow-list for unlabeled mode",
    )
    parser.add_argument(
        "--max_notes_per_patient",
        type=int,
        default=None,
        help="Optional per-patient cap for unlabeled mode",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=100,
        help="Max samples to process for demo safety (default: 100)",
    )
    parser.add_argument("--model_type", type=str, default="qwen_32b")
    parser.add_argument("--temperature", type=float, default=0.01)
    parser.add_argument(
        "--llm_type",
        type=str,
        choices=["local", "api", "openrouter", "llama_cpp"],
        default="local",
    )
    parser.add_argument("--api_config", type=str, default=None)
    parser.add_argument("--gguf_file", type=str, default=None)
    parser.add_argument(
        "--model_cache_dir",
        type=str,
        default=_DEFAULT_MODEL_CACHE_DIR,
    )
    parser.add_argument(
        "--gpu_id",
        type=lambda x: None if x.lower() == "none" else int(x),
        default=0,
        metavar="N|none",
    )
    parser.add_argument("--condor", action="store_true")
    parser.add_argument(
        "--embeddings_file",
        type=Path,
        default=_DEFAULT_EMBEDDINGS_FILE,
    )
    parser.add_argument(
        "--retriever",
        type=str,
        default="sentence_transformer",
    )
    parser.add_argument(
        "--retriever_model",
        type=str,
        default="abhinand/MedEmbed-small-v0.1",
    )
    parser.add_argument("--entity_extractor", type=str, default="retrieval")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--window_size", type=int, default=5)
    parser.add_argument(
        "--min_sentence_size",
        type=lambda x: None if x.lower() == "none" else int(x),
        default=200,
        metavar="N|none",
    )
    parser.add_argument("--verifier_type", type=str, default="multi_stage")
    parser.add_argument(
        "--abbreviations_file",
        type=str,
        default=str(_DEFAULT_ABBREVIATIONS_FILE),
    )
    parser.add_argument(
        "--use_abbreviations",
        action="store_true",
        default=True,
    )
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--exact_match", action="store_true")
    parser.add_argument("--disease_check", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=_DEFAULT_RESULTS_DIR,
        help="Output directory for JSONL/summary artifacts",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Optional run subdirectory name",
    )
    return parser.parse_args()


def build_task(args: argparse.Namespace):
    if args.task_mode == "mimic3_rd_mining":
        return MIMIC3RareDiseaseMining(annotation_path=args.annotation_path)
    if args.task_mode == "mimic3_rd_mining_text":
        return MIMIC3RDMiningText(annotation_path=args.annotation_path)
    return MIMIC3FullNotesUnlabeled(
        discharge_only=args.discharge_only,
        note_categories=args.note_categories,
        max_notes_per_patient=args.max_notes_per_patient,
    )


def init_llm(args: argparse.Namespace, llm_device: str):
    if args.llm_type == "api":
        return (
            APILLMClient.from_config(args.api_config)
            if args.api_config
            else APILLMClient(
                model_type=args.model_type,
                temperature=args.temperature,
            )
        )
    if args.llm_type == "openrouter":
        return (
            OpenRouterLLMClient.from_config(args.api_config)
            if args.api_config
            else OpenRouterLLMClient(
                model_type=args.model_type, temperature=args.temperature
            )
        )
    if args.llm_type == "llama_cpp":
        return LlamaCppLLMClient(
            model_type=args.model_type,
            gguf_file=args.gguf_file,
            main_gpu=args.gpu_id if args.gpu_id is not None else 0,
            temperature=args.temperature,
            cache_dir=args.model_cache_dir,
        )
    return LocalLLMClient(
        model_type=args.model_type,
        device=llm_device,
        cache_dir=args.model_cache_dir,
        temperature=args.temperature,
    )


def build_supervision_inputs(
    note_id: str, text: str, matched_entities: List[Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    predictions = {
        note_id: {
            "clinical_text": text,
            "matched_diseases": matched_entities,
        }
    }
    true_positives = []
    for entity in matched_entities:
        orpha_code = str(entity.get("orpha_id", "")).strip()
        if not orpha_code:
            continue
        true_positives.append({"sample_id": note_id, "code": orpha_code})
    evaluation = {"corpus_true_positives": true_positives}
    return {
        "predictions": predictions,
        "ground_truth": {},
        "evaluation": evaluation,
    }


def decode_optional_labels(sample: Dict[str, Any]) -> Optional[Any]:
    # Keep this optional so unlabeled mode stays label-free.
    if "annotations" in sample:
        return pickle.loads(sample["annotations"])
    if "entities" in sample:
        return pickle.loads(sample["entities"])
    return None


def decode_text(sample: Dict[str, Any]) -> str:
    """Decode text from either plain string or pickled payload."""
    value = sample.get("text", "")
    if isinstance(value, str):
        return value
    try:
        return pickle.loads(value)
    except Exception:
        return str(value)


def main() -> None:
    args = parse_args()
    run_name = args.run_name or f"{args.task_mode}_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    output_jsonl = run_dir / "predictions_with_stages.jsonl"
    summary_json = run_dir / "summary.json"

    cfg = SimpleNamespace(
        gpu_id=args.gpu_id,
        condor=args.condor,
        cpu=(args.gpu_id is None and not args.condor),
        retriever_gpu_id=None,
        retriever_cpu=False,
    )
    devices = setup_device(cfg)

    ts(f"Task mode         : {args.task_mode}")
    ts(f"LLM type          : {args.llm_type}")
    ts(f"Model             : {args.model_type}")
    ts(f"LLM device        : {devices['llm']}")
    ts(f"Retriever device  : {devices['retriever']}")
    ts(f"MIMIC-III root    : {args.mimic3_root}")
    ts(f"Output JSONL      : {output_jsonl}")

    ts("Loading MIMIC3Dataset...")
    dataset = MIMIC3Dataset(
        root=args.mimic3_root,
        tables=["noteevents"],
        cache_dir=args.cache_dir,
        num_workers=4,
    )
    task = build_task(args)
    samples = dataset.set_task(task, num_workers=4)
    total_samples = len(samples)
    ts(f"Loaded {total_samples} task samples")
    if total_samples == 0:
        raise RuntimeError("No samples produced by task; check root/cache/task filters")

    ts("Loading LLM and embedding resources...")
    llm_client = init_llm(args, devices["llm"])
    embedded_documents = np.load(args.embeddings_file, allow_pickle=True)
    embedding_manager = EmbeddingsManager(
        model_type=args.retriever,
        model_name=args.retriever_model,
        device=devices.get("retriever", devices["llm"]),
    )

    ts("Initializing RDMA 4-agent wrappers...")
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
    matcher = RDMAMatcher(
        llm_client=llm_client,
        embedding_manager=embedding_manager,
        embedded_documents=embedded_documents,
        top_k=args.top_k,
        debug=args.debug,
    )
    supervisor = RDMASupervisor(
        llm_client=llm_client,
        embedding_manager=embedding_manager,
        embedded_documents=embedded_documents,
        abbreviations_file=args.abbreviations_file,
        use_abbreviations=args.use_abbreviations,
        top_k=args.top_k,
        debug=args.debug,
    )

    stage_timings: List[Dict[str, float]] = []
    n_errors = 0
    n_empty_predictions = 0
    n_flagged = 0

    with output_jsonl.open("w", encoding="utf-8") as out_f:
        for idx, sample in enumerate(
            tqdm(samples, total=total_samples, desc="RDMA-demo")
        ):
            if args.max_samples is not None and idx >= args.max_samples:
                break

            note_id = str(sample.get("note_id", f"sample_{idx}"))
            patient_id = str(sample.get("patient_id", ""))
            text = decode_text(sample)
            expected = decode_optional_labels(sample)

            extract_s = verify_s = match_s = supervise_s = 0.0
            extracted: List[Dict[str, Any]] = []
            verified: List[Dict[str, Any]] = []
            matched: List[Dict[str, Any]] = []
            supervision: Dict[str, Any] = {}
            error = None

            try:
                t0 = time.perf_counter()
                extracted = extractor.extract_from_text(text)
                extract_s = time.perf_counter() - t0

                t0 = time.perf_counter()
                verified = verifier.verify_entities(extracted)
                verify_s = time.perf_counter() - t0

                t0 = time.perf_counter()
                matched = matcher.match_entities(verified)
                match_s = time.perf_counter() - t0

                t0 = time.perf_counter()
                sup_inputs = build_supervision_inputs(note_id, text, matched)
                supervision = supervisor.supervise(
                    predictions_data=sup_inputs["predictions"],
                    ground_truth_data=sup_inputs["ground_truth"],
                    evaluation_data=sup_inputs["evaluation"],
                )
                supervise_s = time.perf_counter() - t0
            except Exception as exc:  # pragma: no cover - runtime guard
                n_errors += 1
                error = str(exc)
                if args.debug:
                    traceback.print_exc()

            predicted = [item.get("entity", "") for item in matched]
            predicted_orpha_ids = [item.get("orpha_id", "") for item in matched]

            if not predicted:
                n_empty_predictions += 1

            flagged_here = len(
                supervision.get("summary", {}).get("flagged_entities", [])
            )
            n_flagged += flagged_here

            stage_timings.append(
                {
                    "extract_s": extract_s,
                    "verify_s": verify_s,
                    "match_s": match_s,
                    "supervise_s": supervise_s,
                }
            )

            payload = {
                "id": note_id,
                "patient_id": patient_id,
                "task_mode": args.task_mode,
                "predicted": predicted,
                "predicted_orpha_ids": predicted_orpha_ids,
                "stage_outputs": {
                    "extracted": extracted,
                    "verified": verified,
                    "matched": matched,
                    "supervision": supervision,
                },
                "timing": {
                    "extraction_s": round(extract_s, 3),
                    "verification_s": round(verify_s, 3),
                    "matching_s": round(match_s, 3),
                    "supervision_s": round(supervise_s, 3),
                },
                "error": error,
            }
            if expected is not None:
                payload["expected"] = expected

            out_f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    processed = len(stage_timings)
    avg_extract = (
        sum(x["extract_s"] for x in stage_timings) / processed if processed else 0.0
    )
    avg_verify = (
        sum(x["verify_s"] for x in stage_timings) / processed if processed else 0.0
    )
    avg_match = (
        sum(x["match_s"] for x in stage_timings) / processed if processed else 0.0
    )
    avg_supervise = (
        sum(x["supervise_s"] for x in stage_timings) / processed if processed else 0.0
    )

    summary = {
        "task_mode": args.task_mode,
        "processed_samples": processed,
        "max_samples": args.max_samples,
        "errors": n_errors,
        "empty_prediction_count": n_empty_predictions,
        "empty_prediction_rate": (
            (n_empty_predictions / processed) if processed else 0.0
        ),
        "total_flagged_for_review": n_flagged,
        "avg_timing_s": {
            "extract": round(avg_extract, 3),
            "verify": round(avg_verify, 3),
            "match": round(avg_match, 3),
            "supervise": round(avg_supervise, 3),
            "total": round(
                avg_extract + avg_verify + avg_match + avg_supervise,
                3,
            ),
        },
        "artifacts": {
            "predictions_with_stages_jsonl": str(output_jsonl),
            "summary_json": str(summary_json),
        },
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    ts(f"Done. Processed {processed} samples")
    ts(f"Summary: {summary_json}")


if __name__ == "__main__":
    main()
