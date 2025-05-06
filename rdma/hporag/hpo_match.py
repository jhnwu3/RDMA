from abc import ABC, abstractmethod
import json
import pandas as pd
import numpy as np
import re
import string  # Added for string.punctuation
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from fuzzywuzzy import fuzz


class BaseHPOMatcher(ABC):
    """Abstract base class for HPO term matching."""

    @abstractmethod
    def match_hpo_terms(
        self, entities: List[str], metadata: List[Dict]
    ) -> List[Tuple[str, str]]:
        """Match entities to HPO terms."""
        pass

    @abstractmethod
    def process_batch(
        self, entities_batch: List[List[str]], metadata_batch: List[List[Dict]]
    ) -> List[List[Tuple[str, str]]]:
        """Process a batch of entities for HPO term matching."""
        pass


class RAGHPOMatcher(BaseHPOMatcher):
    """HPO term matcher using RAG approach with enhanced match tracking."""

    def __init__(self, embeddings_manager, llm_client=None, system_message: str = None):
        self.embeddings_manager = embeddings_manager
        self.llm_client = llm_client
        self.system_message = system_message
        self.index = None
        self.embedded_documents = None

    def prepare_index(self, metadata: List[Dict]):
        """Prepare FAISS index from metadata."""
        embeddings_array = self.embeddings_manager.prepare_embeddings(metadata)
        self.index = self.embeddings_manager.create_index(embeddings_array)
        self.embedded_documents = metadata

    def clean_text(self, text: str) -> str:
        """Cleans input text by converting to lowercase and removing punctuation."""
        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation)).strip()
        return text

    def _retrieve_candidates(self, entity: str) -> List[Dict]:
        """Retrieve relevant candidates with metadata and similarity scores."""
        query_vector = self.embeddings_manager.query_text(entity).reshape(1, -1)
        distances, indices = self.embeddings_manager.search(
            query_vector, self.index, k=100
        )

        seen_metadata = set()
        candidate_metadata = []

        for idx, distance in zip(indices[0], distances[0]):
            metadata = self.embedded_documents[idx]["unique_metadata"]
            metadata_str = json.dumps(metadata)

            if metadata_str not in seen_metadata:
                seen_metadata.add(metadata_str)
                candidate_metadata.append(
                    {
                        "metadata": metadata,
                        "similarity_score": 1
                        / (1 + distance),  # Convert distance to similarity
                    }
                )
                if len(candidate_metadata) == 20:
                    break

        return candidate_metadata

    def _enrich_metadata(self, entity: str, candidates: List[Dict]) -> List[Dict]:
        """Enrich metadata through multiple matching strategies."""
        cleaned_phrase = self.clean_text(entity)
        enriched_candidates = candidates.copy()
        seen_metadata = {json.dumps(c["metadata"]) for c in candidates}

        # Helper function to add new candidates
        def add_candidate(new_metadata, score, match_type):
            metadata_str = json.dumps(new_metadata)
            if metadata_str not in seen_metadata:
                seen_metadata.add(metadata_str)
                enriched_candidates.append(
                    {
                        "metadata": new_metadata,
                        "similarity_score": score,
                        "match_type": match_type,
                    }
                )

        # 1. Fuzzy Matching
        for candidate in candidates:
            term = candidate["metadata"]["info"]
            hp_id = candidate["metadata"]["hp_id"]
            cleaned_term = self.clean_text(term)
            score = fuzz.ratio(cleaned_phrase, cleaned_term) / 100.0

            if score > 0.8:  # 80% similarity threshold
                add_candidate({"info": term, "hp_id": hp_id}, score, "fuzzy")

        # 2. Substring Matching
        for candidate in candidates:
            term = candidate["metadata"]["info"]
            hp_id = candidate["metadata"]["hp_id"]
            cleaned_term = self.clean_text(term)

            if cleaned_term in cleaned_phrase or cleaned_phrase in cleaned_term:
                # Score based on length ratio
                score = min(len(cleaned_term), len(cleaned_phrase)) / max(
                    len(cleaned_term), len(cleaned_phrase)
                )
                add_candidate({"info": term, "hp_id": hp_id}, score, "substring")

        return enriched_candidates

    def match_hpo_terms(
        self,
        entities: List[str],
        metadata: List[Dict],
        original_sentences: Optional[List[str]] = None,
    ) -> List[Dict]:
        """Match entities to HPO terms using enriched matching followed by LLM."""
        if self.index is None:
            self.prepare_index(metadata)

        matches = []

        if original_sentences is None:
            original_sentences = [None] * len(entities)

        for entity, original_sentence in zip(entities, original_sentences):
            # Get initial candidates
            candidates = self._retrieve_candidates(entity)

            # Enrich candidates with additional matching strategies
            enriched_candidates = self._enrich_metadata(entity, candidates)

            match_info = {
                "entity": entity,
                "top_candidates": enriched_candidates[
                    :5
                ],  # Store top 5 enriched candidates
            }

            # Try exact matching
            hpo_term = self._try_exact_match(entity, enriched_candidates)
            if hpo_term:
                match_info.update(
                    {
                        "hpo_term": hpo_term,
                        "match_method": "exact",
                        "confidence_score": 1.0,
                    }
                )
                matches.append(match_info)
                continue

            # If no exact match, try LLM matching with enriched candidates
            if self.llm_client:
                hpo_term = self._try_llm_match(
                    entity, enriched_candidates, original_sentence
                )
                if hpo_term:
                    match_info.update(
                        {
                            "hpo_term": hpo_term,
                            "match_method": "llm",
                            "confidence_score": 0.7,
                        }
                    )
                else:
                    # If no match found, use the first candidate's HPO term
                    if enriched_candidates:
                        first_candidate = enriched_candidates[0]
                        match_info.update(
                            {
                                "hpo_term": first_candidate["metadata"]["hp_id"],
                                "match_method": "fallback",
                                "confidence_score": first_candidate.get(
                                    "similarity_score", 0.1
                                ),  # Low confidence for fallback
                            }
                        )
                    else:
                        # If no candidates at all, still include the entity with null HPO term
                        match_info.update(
                            {
                                "hpo_term": None,
                                "match_method": "no_match",
                                "confidence_score": 0.0,
                            }
                        )
            matches.append(match_info)

        return matches

    def _try_exact_match(self, entity: str, candidates: List[Dict]) -> Optional[str]:
        """Try exact matching on candidate terms."""
        cleaned_entity = self.clean_text(entity)

        for candidate in candidates:
            term = candidate["metadata"]["info"]
            hp_id = candidate["metadata"]["hp_id"]
            if self.clean_text(term) == cleaned_entity:
                return hp_id

        return None

    def _try_llm_match(
        self,
        entity: str,
        candidates: List[Dict],
        original_sentence: Optional[str] = None,
    ) -> Optional[str]:
        """Try LLM matching using candidates as context."""
        if not self.llm_client or not self.system_message:
            return None

        # Build context items with just descriptions and HPO terms
        context_items = []
        seen_terms = set()

        for candidate in candidates:
            info = candidate["metadata"]["info"]
            hp_id = candidate["metadata"]["hp_id"]
            term_key = f"{info.lower()}_{hp_id}"

            if term_key not in seen_terms:
                seen_terms.add(term_key)
                context_items.append(f"- {info} ({hp_id})")

        # Build the prompt, matching their simpler format
        prompt_parts = [
            f"Query: {entity}",
            f"Original Sentence: {original_sentence}" if original_sentence else None,
            "Context: The following related information is available to assist in determining the appropriate HPO terms:",
            "\n".join(context_items),
        ]

        # Remove None entries and join
        prompt = "\n".join(filter(None, prompt_parts))
        response = self.llm_client.query(prompt, self.system_message)

        hpo_match = re.search(r"HP:\d+", response)
        return hpo_match.group(0) if hpo_match else None

    def process_batch(
        self,
        entities_batch: List[List[str]],
        metadata_batch: List[List[Dict]],
        original_sentences_batch: Optional[List[List[str]]] = None,
    ) -> List[List[Dict]]:
        """Process a batch of entities for HPO term matching."""
        results = []

        if original_sentences_batch is None:
            original_sentences_batch = [None] * len(entities_batch)

        for entities, metadata, original_sentences in zip(
            entities_batch, metadata_batch, original_sentences_batch
        ):
            matches = self.match_hpo_terms(entities, metadata, original_sentences)
            results.append(matches)

        return results


import json
import pandas as pd
import numpy as np
import re
import string
import time
from typing import List, Dict, Any, Optional, Tuple, Set, Union
from functools import lru_cache
from datetime import datetime
import faiss

# from hporag.hpo_match import BaseHPOMatcher

# Replace fuzzywuzzy with rapidfuzz - much faster implementation
try:
    from rapidfuzz import fuzz as rfuzz
    from rapidfuzz import process as rprocess
except ImportError:
    print(
        "Warning: rapidfuzz not found. Install with 'pip install rapidfuzz' for better performance."
    )
    # Fallback to fuzzywuzzy if rapidfuzz isn't available
    from fuzzywuzzy import fuzz as rfuzz
    from fuzzywuzzy import process as rprocess


class OptimizedRAGHPOMatcher(BaseHPOMatcher):
    """HPO term matcher using RAG approach with optimized performance and high accuracy."""

    def __init__(
        self,
        embeddings_manager,
        llm_client=None,
        system_message: str = None,
        debug: bool = False,
        fuzzy_threshold: int = 90,
        max_candidates: int = 20,
        faiss_k: int = 800,
    ):
        """Initialize the improved HPO matcher.

        Args:
            embeddings_manager: Manager for embedding operations
            llm_client: LLM client for text generation (optional)
            system_message: System message for LLM context (optional)
            debug: Enable debug mode with timing information
            fuzzy_threshold: Threshold for fuzzy matching (0-100)
            max_candidates: Maximum number of candidates to process
            faiss_k: K value for FAISS search (restored to 800)
        """
        self.embeddings_manager = embeddings_manager
        self.llm_client = llm_client
        self.system_message = system_message
        self.debug = debug
        self.fuzzy_threshold = fuzzy_threshold
        self.max_candidates = max_candidates
        self.faiss_k = faiss_k
        self.index = None
        self.embedded_documents = None

        # Token cache for prefiltering
        self.token_cache = {}

    def _debug_time(self, operation):
        """Context manager for timing operations in debug mode."""

        class TimerContext:
            def __init__(self, matcher, operation):
                self.matcher = matcher
                self.operation = operation

            def __enter__(self):
                self.start_time = time.time()
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                elapsed = time.time() - self.start_time
                if self.matcher.debug:
                    print(f"[DEBUG] {self.operation} took {elapsed:.4f} seconds")

        return TimerContext(self, operation)

    def prepare_index(self, metadata: List[Dict]):
        """Prepare FAISS index from metadata."""
        with self._debug_time("Preparing embeddings and FAISS index"):
            embeddings_array = self.embeddings_manager.prepare_embeddings(metadata)
            self.index = self.embeddings_manager.create_index(embeddings_array)
            self.embedded_documents = metadata

            # Pre-compute token sets for common metadata
            self._precompute_token_sets()

    def _precompute_token_sets(self):
        """Precompute token sets for all metadata to speed up matching."""
        if self.embedded_documents is None:
            return

        with self._debug_time("Precomputing token sets"):
            for idx, doc in enumerate(self.embedded_documents):
                if "unique_metadata" in doc and "info" in doc["unique_metadata"]:
                    info = doc["unique_metadata"]["info"]
                    # Clean and tokenize
                    cleaned = self._clean_text(info)
                    tokens = set(cleaned.split())
                    # Store in cache
                    cache_key = f"tokens_{idx}"
                    self.token_cache[cache_key] = tokens

    @staticmethod
    @lru_cache(maxsize=10000)
    def _clean_text(text: str) -> str:
        """Clean text with caching for repeated terms."""
        # Process None or empty strings
        if not text:
            return ""
        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation)).strip()
        return text

    @staticmethod
    @lru_cache(maxsize=10000)
    def _cached_fuzzy_ratio(s1: str, s2: str) -> float:
        """Cached fuzzy comparison for repeated comparisons."""
        return rfuzz.ratio(s1, s2)

    def _retrieve_candidates(self, entity: str) -> List[Dict]:
        """Optimized retrieval of candidate metadata."""
        with self._debug_time(f"Retrieving candidates for '{entity}'"):
            # Embed query
            query_vector = self.embeddings_manager.query_text(entity).reshape(1, -1)

            # Search with restored higher k value
            distances, indices = self.index.search(query_vector, self.faiss_k)

            # Create candidate metadata list
            candidates = []
            seen_hp_ids = set()

            for i, idx in enumerate(indices[0][: self.faiss_k]):
                metadata = self.embedded_documents[idx]["unique_metadata"]
                hp_id = metadata.get("hp_id", "")

                # Skip duplicates
                if hp_id in seen_hp_ids:
                    continue

                seen_hp_ids.add(hp_id)
                candidates.append(
                    {
                        "metadata": metadata,
                        "similarity_score": 1 / (1 + distances[0][i]),
                    }
                )

                # Limit to max_candidates
                if len(candidates) >= self.max_candidates:
                    break

            return candidates

    def _enrich_candidates(self, entity: str, candidates: List[Dict]) -> List[Dict]:
        """Enrich candidates with fuzzy and substring matches for better LLM context."""
        cleaned_entity = self._clean_text(entity)
        enriched_candidates = candidates.copy()
        seen_metadata = {json.dumps(c["metadata"]) for c in candidates}

        # Helper function to add new candidates
        def add_candidate(new_metadata, score, match_type):
            metadata_str = json.dumps(new_metadata)
            if metadata_str not in seen_metadata:
                seen_metadata.add(metadata_str)
                enriched_candidates.append(
                    {
                        "metadata": new_metadata,
                        "similarity_score": score,
                        "match_type": match_type,
                    }
                )

        # Fuzzy matching to enrich the candidate pool (not for direct matching)
        for candidate in candidates:
            term = candidate["metadata"].get("info", "")
            hp_id = candidate["metadata"].get("hp_id", "")
            cleaned_term = self._clean_text(term)

            score = self._cached_fuzzy_ratio(cleaned_entity, cleaned_term)
            if score >= self.fuzzy_threshold:
                add_candidate({"info": term, "hp_id": hp_id}, score / 100.0, "fuzzy")

        # Substring matching to enrich the candidate pool (not for direct matching)
        for candidate in candidates:
            term = candidate["metadata"].get("info", "")
            hp_id = candidate["metadata"].get("hp_id", "")
            cleaned_term = self._clean_text(term)

            if cleaned_term in cleaned_entity or cleaned_entity in cleaned_term:
                # Better confidence scoring than the original
                score = 0.8  # Use fixed higher score for substring matches
                add_candidate({"info": term, "hp_id": hp_id}, score, "substring")

        return enriched_candidates

    def _try_exact_match(self, entity: str, candidates: List[Dict]) -> Optional[str]:
        """Try exact matching on candidate terms."""
        cleaned_entity = self._clean_text(entity)

        # Try exact string match first
        for candidate in candidates:
            term = candidate["metadata"].get("info", "")
            hp_id = candidate["metadata"].get("hp_id", "")

            if self._clean_text(term) == cleaned_entity:
                return hp_id

        # Also check for extremely high fuzzy scores (95+) as a fallback
        for candidate in candidates:
            term = candidate["metadata"].get("info", "")
            hp_id = candidate["metadata"].get("hp_id", "")

            score = self._cached_fuzzy_ratio(cleaned_entity, self._clean_text(term))
            if score >= 95:  # Only extremely high confidence fuzzy matches
                return hp_id

        return None

    def _try_llm_match(
        self,
        entity: str,
        candidates: List[Dict],
        original_sentence: Optional[str] = None,
    ) -> Optional[str]:
        """Optimized LLM matching using candidates as context."""
        if not self.llm_client or not self.system_message:
            return None

        with self._debug_time("LLM matching"):
            # Build context items with descriptions and HPO terms
            context_items = []
            seen_terms = set()

            # Sort enriched candidates by similarity score
            sorted_candidates = sorted(
                candidates, key=lambda x: x.get("similarity_score", 0), reverse=True
            )

            for candidate in sorted_candidates[
                :15
            ]:  # Increased from 10 to 15 for better context
                info = candidate["metadata"].get("info", "")
                hp_id = candidate["metadata"].get("hp_id", "")
                term_key = f"{info.lower()}_{hp_id}"

                if term_key not in seen_terms:
                    seen_terms.add(term_key)
                    context_items.append(f"- {info} ({hp_id})")

            # Build the prompt
            prompt_parts = [
                f"Query: {entity}",
                (
                    f"Original Sentence: {original_sentence}"
                    if original_sentence
                    else None
                ),
                "Context: The following related information is available to assist in determining the appropriate HPO terms:",
                "\n".join(context_items),
            ]

            # Remove None entries and join
            prompt = "\n".join(filter(None, prompt_parts))
            response = self.llm_client.query(prompt, self.system_message)

            hpo_match = re.search(r"HP:\d+", response)
            return hpo_match.group(0) if hpo_match else None

    def match_hpo_terms(
        self,
        entities: List[str],
        metadata: List[Dict],
        original_sentences: Optional[List[str]] = None,
    ) -> List[Dict]:
        """Match entities to HPO terms with optimizations that prioritize accuracy."""
        with self._debug_time(f"Matching {len(entities)} entities"):
            if self.index is None:
                self.prepare_index(metadata)

            matches = []

            if original_sentences is None:
                original_sentences = [None] * len(entities)

            for entity, original_sentence in zip(entities, original_sentences):
                # Skip empty entities
                if not entity or not entity.strip():
                    continue

                # Get initial candidates
                candidates = self._retrieve_candidates(entity)

                # Enrich candidates for better LLM context (not used for direct matching)
                enriched_candidates = self._enrich_candidates(entity, candidates)

                match_info = {"entity": entity, "top_candidates": candidates[:5]}

                # Try exact matching first (like the original)
                hpo_term = self._try_exact_match(entity, candidates)
                if hpo_term:
                    match_info.update(
                        {
                            "hpo_term": hpo_term,
                            "match_method": "exact",
                            "confidence_score": 1.0,
                        }
                    )
                    matches.append(match_info)
                    continue

                # If no exact match, go straight to LLM (like the original)
                if self.llm_client:
                    hpo_term = self._try_llm_match(
                        entity, enriched_candidates, original_sentence
                    )
                    if hpo_term:
                        match_info.update(
                            {
                                "hpo_term": hpo_term,
                                "match_method": "llm",
                                "confidence_score": 0.8,  # Higher confidence for LLM matches
                            }
                        )
                        matches.append(match_info)
                        continue

                # If all else fails, use the first candidate as fallback (like the original)
                if candidates:
                    first_candidate = candidates[0]
                    match_info.update(
                        {
                            "hpo_term": first_candidate["metadata"]["hp_id"],
                            "match_method": "fallback",
                            "confidence_score": min(
                                0.5, first_candidate.get("similarity_score", 0.1)
                            ),  # Cap at 0.5
                        }
                    )
                else:
                    # If no candidates at all, include the entity with null HPO term
                    match_info.update(
                        {
                            "hpo_term": None,
                            "match_method": "no_match",
                            "confidence_score": 0.0,
                        }
                    )

                matches.append(match_info)

            return matches

    def process_batch(
        self,
        entities_batch: List[List[str]],
        metadata_batch: List[List[Dict]],
        original_sentences_batch: Optional[List[List[str]]] = None,
    ) -> List[List[Dict]]:
        """Process a batch of entities for HPO term matching."""
        with self._debug_time(
            f"Processing batch of {len(entities_batch)} sets of entities"
        ):
            results = []

            if original_sentences_batch is None:
                original_sentences_batch = [None] * len(entities_batch)

            for entities, metadata, original_sentences in zip(
                entities_batch, metadata_batch, original_sentences_batch
            ):
                matches = self.match_hpo_terms(entities, metadata, original_sentences)
                results.append(matches)

            return results
