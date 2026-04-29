#!/usr/bin/env python3
"""
i2b2 NER baseline for the BioLark GSC phenotype-mining benchmark.

Pipeline:
  1. Stanza i2b2 NER extracts PROBLEM entities.
  2a. Default: EmbeddingFuzzyMatcher maps extracted entities to HPO codes
      using embedding retrieval + fuzzy matching (no LLM required).
  2b. Optional: BioSent2Vec nearest-neighbor mapping.

Output JSONL format:
    {"id": "<doc_id>", "predicted": [...], "ground_truth": [...], "timing": {...}}
"""

import argparse
import json
import os
import pickle
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
from tqdm import tqdm

_RDMA_ROOT = Path("/home/johnwu3/projects/rare_disease/workspace/repos/RDMA")
_PHENOGPT_ROOT = Path("/home/johnwu3/projects/rare_disease/workspace/repos/PhenoGPT")
sys.path.insert(0, str(_RDMA_ROOT))

from rdma.hpo.embedding_fuzzy_matcher import EmbeddingFuzzyMatcher  # noqa: E402
from rdma.hporag.entity import StanzaEntityExtractor  # noqa: E402

from datasets.biolarkgsc import BioLarkGSCDataset  # noqa: E402
from tasks.biolarkgsc import BioLarkGSCNER  # noqa: E402

_RESULTS_DIR = Path("/home/johnwu3/projects/rare_disease/workspace/results")
_DEFAULT_EMBEDDINGS_FILE = str(
    _RDMA_ROOT / "data" / "vector_stores" / "G2GHPO_metadata_medembed.npy"
)
_DEFAULT_DATASET_CACHE_DIR = "/shared/eng/pyhealth/biolarkgsc"
_DEFAULT_HPO_DATABASE = str(_PHENOGPT_ROOT / "hpo_database.json")


def ts(msg: str) -> None:
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S} - {msg}", flush=True)


def load_done_ids(path: Path) -> set:
    done = set()
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    done.add(json.loads(line).get("id"))
                except Exception:
                    continue
    return done


def _normalize_hpo(hpo_id: str) -> str:
    return hpo_id.replace("_", ":", 1) if hpo_id.startswith("HP_") else hpo_id


def compute_metrics(records: List[dict]) -> dict:
    tp = fp = fn = 0
    for rec in records:
        predicted = set(h for h in rec["predicted"] if h)
        gold = set(h for h in rec["ground_truth"] if h)
        tp += len(predicted & gold)
        fp += len(predicted - gold)
        fn += len(gold - predicted)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "n_docs": len(records),
    }


class BioSent2VecMapper:
    def __init__(self, biosent2vec_path: str, hpo_database_path: str):
        import joblib
        import nltk
        import sent2vec
        from nltk.corpus import stopwords

        for pkg in ["stopwords", "punkt", "punkt_tab"]:
            try:
                nltk.data.find(
                    f"corpora/{pkg}" if pkg == "stopwords" else f"tokenizers/{pkg}"
                )
            except LookupError:
                nltk.download(pkg)

        if not os.path.exists(biosent2vec_path):
            raise FileNotFoundError(
                f"BioSent2Vec model file not found: {biosent2vec_path}"
            )
        if not os.path.exists(hpo_database_path):
            raise FileNotFoundError(f"HPO database file not found: {hpo_database_path}")

        self.stop_words = set(stopwords.words("english"))
        self.hpo_database = joblib.load(hpo_database_path)
        self.model = sent2vec.Sent2vecModel()
        self.model.load_model(biosent2vec_path)

        all_terms = list(self.hpo_database.keys())
        all_terms_preprocessed = [self._preprocess_sentence(t) for t in all_terms]
        all_term_vecs = self.model.embed_sentences(all_terms_preprocessed)
        self.term_to_vec = {
            k: v for k, v in zip(all_terms, all_term_vecs) if k != "All"
        }

    def _preprocess_sentence(self, text: str) -> str:
        from nltk import word_tokenize
        from string import punctuation

        text = text.replace("/", " / ")
        text = text.replace(".-", " .- ")
        text = text.replace(".", " . ")
        text = text.replace("'", " ' ")
        text = text.lower()
        tokens = [
            token
            for token in word_tokenize(text)
            if token not in punctuation and token not in self.stop_words
        ]
        return " ".join(tokens)

    def map_terms(self, entities: List[str]) -> List[str]:
        from scipy.spatial import distance

        if not entities:
            return []

        all_terms = list(self.term_to_vec.keys())
        all_vectors = list(self.term_to_vec.values())
        preprocessed = [self._preprocess_sentence(e) for e in entities]
        entity_vectors = self.model.embed_sentences(preprocessed)

        predicted = []
        for i, e_vec in enumerate(entity_vectors):
            entity = entities[i]
            if entity.capitalize() in self.hpo_database:
                predicted.append(self.hpo_database[entity.capitalize()])
                continue

            best_term = None
            best_score = -1.0
            for j, ref_vec in enumerate(all_vectors):
                sim = 1 - distance.cosine(e_vec, ref_vec)
                if np.isnan(sim):
                    continue
                if sim > best_score:
                    best_score = sim
                    best_term = all_terms[j]

            if best_term is not None:
                hp = self.hpo_database.get(best_term)
                if hp:
                    predicted.append(hp)

        # de-duplicate while preserving order
        seen = set()
        out = []
        for hp in predicted:
            if hp not in seen:
                seen.add(hp)
                out.append(hp)
        return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="i2b2 NER baseline on BioLark GSC phenotype-mining benchmark"
    )
    parser.add_argument(
        "--dataset_cache_dir",
        type=str,
        default=_DEFAULT_DATASET_CACHE_DIR,
        help="PyHealth dataset cache directory (default: %(default)s)",
    )
    parser.add_argument(
        "--stanza_device",
        type=str,
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device for Stanza pipeline (default: %(default)s)",
    )
    parser.add_argument(
        "--use_biosent2vec",
        action="store_true",
        help="Use BioSent2Vec for HPO mapping instead of EmbeddingFuzzyMatcher",
    )
    parser.add_argument(
        "--biosent2vec_path",
        type=str,
        default=None,
        help="Path to BioSent2Vec .bin file (required if --use_biosent2vec)",
    )
    parser.add_argument(
        "--hpo_database",
        type=str,
        default=_DEFAULT_HPO_DATABASE,
        help="Path to HPO database file for BioSent2Vec mapping",
    )

    parser.add_argument(
        "--gpu_id",
        type=lambda x: None if x.lower() == "none" else int(x),
        default=0,
        metavar="N|none",
        help="GPU device id for embedding model; pass 'none' for CPU (default: %(default)s)",
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
        "--retriever",
        type=str,
        default="sentence_transformer",
        help="Retriever type for EmbeddingFuzzyMatcher (default: %(default)s)",
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
        default=20,
        help="Top-k FAISS candidates to retrieve per entity (default: %(default)s)",
    )
    parser.add_argument(
        "--fuzzy_threshold",
        type=float,
        default=0.85,
        help="Minimum SequenceMatcher ratio to accept a fuzzy match (default: %(default)s)",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output JSONL path "
            "(default: <results_dir>/biolarkgsc/i2b2_predictions.jsonl)"
        ),
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume from existing output"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable verbose debug logging"
    )
    parser.add_argument("--dev", action="store_true", help="Process first 2 samples")
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=20,
        help="Log checkpoint every N samples",
    )

    args = parser.parse_args()

    if args.use_biosent2vec and not args.biosent2vec_path:
        parser.error("--biosent2vec_path is required when --use_biosent2vec is set")

    import torch
    if args.condor:
        embed_device = "cuda" if torch.cuda.is_available() else "cpu"
    elif args.gpu_id is not None and torch.cuda.is_available():
        embed_device = f"cuda:{args.gpu_id}"
    else:
        embed_device = "cpu"

    output = args.output or (_RESULTS_DIR / "biolarkgsc" / "i2b2_predictions.jsonl")

    ts(f"Dataset cache dir : {args.dataset_cache_dir}")
    ts(f"Stanza device     : {args.stanza_device}")
    ts(
        f"Mapping           : {'BioSent2Vec' if args.use_biosent2vec else 'EmbeddingFuzzyMatcher'}"
    )
    ts(f"Embeddings file   : {args.embeddings_file}")
    ts(f"Fuzzy threshold   : {args.fuzzy_threshold}")
    ts(f"Output            : {output}")
    ts(f"Resume            : {args.resume}")
    ts(f"Debug             : {args.debug}")
    ts(f"Dev mode          : {args.dev}")

    ts("Loading BioLarkGSCDataset...")
    dataset = BioLarkGSCDataset(cache_dir=args.dataset_cache_dir)
    samples = dataset.set_task(BioLarkGSCNER())
    ts(f"  {len(samples)} samples loaded")

    first = next(iter(samples))
    ts(f"  Sample preview - id: {first['patient_id']!r}")
    ts(f"    text[:120]: {pickle.loads(first['text'])[:120]!r}")

    ts("Loading Stanza i2b2 pipeline...")
    try:
        import stanza

        stanza.download("en", package="mimic", processors={"ner": "i2b2"})
        stanza_pipeline = stanza.Pipeline(
            "en",
            package="mimic",
            processors={"ner": "i2b2"},
            device=args.stanza_device,
        )
    except Exception as e:
        raise RuntimeError(
            "Failed to initialize Stanza i2b2 pipeline. "
            "Verify stanza resources are available and device is valid."
        ) from e

    extractor = StanzaEntityExtractor(stanza_pipeline)
    matcher = None
    biosent_mapper = None

    if args.use_biosent2vec:
        ts("Initializing BioSent2Vec mapper...")
        biosent_mapper = BioSent2VecMapper(
            biosent2vec_path=args.biosent2vec_path,
            hpo_database_path=args.hpo_database,
        )
    else:
        ts(f"Initializing EmbeddingFuzzyMatcher (embeddings: {args.embeddings_file})...")
        matcher = EmbeddingFuzzyMatcher(
            embeddings_file=str(args.embeddings_file),
            retriever=args.retriever,
            retriever_model=args.retriever_model,
            top_k=args.top_k,
            fuzzy_threshold=args.fuzzy_threshold,
            device=embed_device,
            debug=args.debug,
        )

    done_ids = load_done_ids(output) if args.resume else set()
    if args.resume:
        ts(f"Resuming - {len(done_ids)} already done")

    output.parent.mkdir(parents=True, exist_ok=True)
    out_f = open(output, "a" if args.resume else "w", encoding="utf-8")

    try:
        timings = []
        records = []
        run_samples = samples.subset(slice(0, 2)) if args.dev else samples
        for i, sample in enumerate(
            tqdm(run_samples, total=len(run_samples), desc="BioLarkGSC-i2b2")
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

            extract_s = match_s = 0.0
            predicted = []
            try:
                t0 = time.perf_counter()
                entities = extractor.extract_entities(text)
                extract_s = time.perf_counter() - t0

                t0 = time.perf_counter()
                if args.use_biosent2vec:
                    predicted = biosent_mapper.map_terms(entities)
                else:
                    entity_payload = [
                        {"entity": e, "context": text} for e in entities if e
                    ]
                    matched = matcher.match(entity_payload)
                    predicted = [m.get("hp_id", "") for m in matched if m.get("hp_id")]
                match_s = time.perf_counter() - t0

                if args.debug:
                    ts(
                        f"  [{doc_id}] n_entities={len(entities)} n_pred={len(predicted)}"
                    )
                ts(f"  [{doc_id}] extract={extract_s:.2f}s  match={match_s:.2f}s")
            except Exception as e:
                ts(f"  ERROR [{doc_id}]: {e}")
                if args.debug:
                    traceback.print_exc()

            timings.append((extract_s, match_s))
            records.append({"predicted": predicted, "ground_truth": ground_truth})
            out_f.write(
                json.dumps(
                    {
                        "id": doc_id,
                        "predicted": predicted,
                        "ground_truth": ground_truth,
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
                ts(f"Checkpoint {i + 1}/{len(run_samples)}")
    finally:
        out_f.close()

    if timings:
        avg_e = sum(t[0] for t in timings) / len(timings)
        avg_m = sum(t[1] for t in timings) / len(timings)
        ts("-- Timing summary -------------------------------")
        ts(f"  Samples           : {len(timings)}")
        ts(f"  Avg extraction    : {avg_e:.2f}s")
        ts(f"  Avg matching      : {avg_m:.2f}s")
        ts(f"  Avg total/sample  : {avg_e + avg_m:.2f}s")

    if records:
        metrics = compute_metrics(records)
        ts("-- Code evaluation ------------------------------")
        ts(f"  Precision : {metrics['precision']:.4f}")
        ts(f"  Recall    : {metrics['recall']:.4f}")
        ts(f"  F1        : {metrics['f1']:.4f}")
        ts(
            f"  TP={metrics['tp']}  FP={metrics['fp']}  FN={metrics['fn']}  Docs={metrics['n_docs']}"
        )

    ts(f"Done -> {output}")


if __name__ == "__main__":
    main()
