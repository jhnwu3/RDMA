#!/usr/bin/env python3
"""
Embedding + fuzzy matching for entity-to-code linking.

No LLM required. Uses FAISS retrieval over pre-computed embeddings followed
by exact substring and SequenceMatcher fuzzy matching to select the best
candidate above a configurable threshold.

Works with both HPO (unique_metadata["info"] / "hp_id") and ORPHA
("name" / "id") embedding file formats.
"""

import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _resolve_doc(doc: dict) -> Tuple[str, str]:
    """Return (term_name, code_id) from an embedded document dict."""
    # ORPHA format: top-level "name" / "id"
    if "name" in doc and "id" in doc:
        return doc.get("name", ""), str(doc.get("id", ""))
    # HPO format: nested under "unique_metadata"
    meta = doc.get("unique_metadata", {})
    return meta.get("info", ""), meta.get("hp_id", "")


class EmbeddingFuzzyMatcher:
    """Entity-to-code matcher using embedding retrieval + fuzzy matching.

    Interface is intentionally compatible with HPOMatcher.match():
      - Input:  list of {"entity": str, "context": str, ...}
      - Output: list of {"entity": str, "hp_id": str,
                         "match_method": str, "confidence_score": float}

    "hp_id" holds the matched code regardless of vocabulary (HPO or ORPHA).
    """

    def __init__(
        self,
        embeddings_file: str,
        retriever: str = "sentence_transformer",
        retriever_model: str = "abhinand/MedEmbed-small-v0.1",
        top_k: int = 20,
        fuzzy_threshold: float = 0.85,
        device: Optional[str] = None,
        debug: bool = False,
    ):
        import torch
        from rdma.utils.embedding import EmbeddingsManager

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.top_k = top_k
        self.fuzzy_threshold = fuzzy_threshold
        self.debug = debug

        if debug:
            print(f"EmbeddingFuzzyMatcher: loading embeddings from {embeddings_file}")

        self.embedded_documents: np.ndarray = np.load(
            embeddings_file, allow_pickle=True
        )
        if debug:
            print(f"  {len(self.embedded_documents)} documents loaded")

        model_name = retriever_model if retriever in ("fastembed", "sentence_transformer") else None
        self.embedding_manager = EmbeddingsManager(
            model_type=retriever, model_name=model_name, device=device
        )

        embeddings_array = self.embedding_manager.prepare_embeddings(
            self.embedded_documents
        )
        self.faiss_index = self.embedding_manager.create_index(embeddings_array)
        if debug:
            print("  FAISS index built")

    # ------------------------------------------------------------------
    # Core matching helpers
    # ------------------------------------------------------------------

    def _retrieve_candidates(self, entity: str) -> List[Tuple[str, str]]:
        """FAISS search → list of (term_name, code_id) pairs."""
        query_vec = self.embedding_manager.query_text(entity).reshape(1, -1)
        distances, indices = self.embedding_manager.search(
            query_vec, self.faiss_index, k=self.top_k
        )
        seen: set = set()
        candidates: List[Tuple[str, str]] = []
        for idx in indices[0]:
            term, code_id = _resolve_doc(self.embedded_documents[idx])
            if not code_id or code_id in seen:
                continue
            seen.add(code_id)
            candidates.append((term, code_id))
        return candidates

    @staticmethod
    def _clean(text: str) -> str:
        return text.lower().strip()

    def _best_match(
        self, entity: str, candidates: List[Tuple[str, str]]
    ) -> Tuple[Optional[str], str, float]:
        """Return (code_id, match_method, confidence) or (None, "no_match", 0)."""
        clean_entity = self._clean(entity)
        entity_tokens = re.findall(r"\b\w+\b", clean_entity)

        best_code: Optional[str] = None
        best_score = 0.0
        best_method = "no_match"

        for term, code_id in candidates:
            clean_term = self._clean(term)

            # 1. Exact substring
            if clean_term in clean_entity or clean_entity in clean_term:
                return code_id, "exact", 1.0

            # 2. Direct fuzzy ratio
            direct_score = SequenceMatcher(None, clean_entity, clean_term).ratio()
            if direct_score >= self.fuzzy_threshold and direct_score > best_score:
                best_score = direct_score
                best_code = code_id
                best_method = "fuzzy"
                continue

            # 3. N-gram fuzzy match (entity tokens against term)
            term_tokens = re.findall(r"\b\w+\b", clean_term)
            if len(term_tokens) <= 1:
                continue
            max_n = min(len(term_tokens) + 1, 5)
            for n in range(2, max_n):
                for i in range(len(entity_tokens) - n + 1):
                    ngram = " ".join(entity_tokens[i : i + n])
                    score = SequenceMatcher(None, ngram, clean_term).ratio()
                    if score >= self.fuzzy_threshold and score > best_score:
                        best_score = score
                        best_code = code_id
                        best_method = "fuzzy_ngram"

        return best_code, best_method, best_score

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def match(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Match a list of entity dicts to codes.

        Args:
            entities: List of dicts with at least an "entity" key.

        Returns:
            List of dicts with "entity", "hp_id", "match_method",
            "confidence_score" keys.
        """
        results: List[Dict[str, Any]] = []
        for item in entities:
            entity_str = item.get("entity", "")
            if not entity_str:
                results.append({**item, "hp_id": "", "match_method": "no_match", "confidence_score": 0.0})
                continue

            candidates = self._retrieve_candidates(entity_str)
            code_id, method, score = self._best_match(entity_str, candidates)

            if self.debug:
                print(f"  match({entity_str!r}) → {code_id!r} via {method} (score={score:.3f})")

            results.append(
                {
                    **item,
                    "hp_id": code_id or "",
                    "match_method": method,
                    "confidence_score": score,
                }
            )
        return results
