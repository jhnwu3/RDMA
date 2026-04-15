import re
import json
import numpy as np
import torch
from datetime import datetime
from fuzzywuzzy import fuzz
from typing import List, Dict, Any, Optional, Tuple, Union, Set
import time
from rdma.utils.search_tools import ToolSearcher


class MultiStageRDVerifier:
    """
    Multistage rare disease verifier with optional abbreviation resolution.

    This verifier implements a streamlined workflow:
    1. Abbreviation resolution (if enabled)
    2. Direct rare disease verification with retrieved candidates

    Key features:
    - Resolves clinical abbreviations to full terms before verification (optional)
    - Focuses on rare disease verification rather than phenotype identification
    - Uses ORPHA code lookup for rare disease identification
    - Caches results for improved performance
    """

    def __init__(
        self,
        embedding_manager,
        llm_client,
        config=None,
        debug=False,
        abbreviations_file=None,
        use_abbreviations=True,
        strict=True,
        exact_match=False,
        disease_check=False,
    ):
        """
        Initialize the multistage rare disease verifier.

        The pipeline has three optional stages, each controlled by a flag:

        Stage 1 — String match (always runs):
            exact_match=True  → accept immediately on string match, no LLM calls.
            exact_match=False → string match result is noted but pipeline continues.

        Stage 2 — LLM ORPHA match (runs when exact_match=False OR no string match):
            strict=True  → LLM must confirm same specific disease as an ORPHA entry.
            strict=False → LLM uses ORPHA candidates as reference, decides on best
                           clinical judgement (higher recall, lower precision).
            On YES → accept immediately. On NO → continue to Stage 3.

        Stage 3 — LLM disease check (optional final gate):
            disease_check=True  → runs direct_verification_system_message gate;
                                   returns the final YES/NO answer.
            disease_check=False → no final gate; reject (Stage 2 LLM has already said NO).

        Args:
            embedding_manager: Manager for embedding operations
            llm_client: LLM client for verification
            config: Configuration for the verifier (currently unused, kept for API compatibility)
            debug: Enable debug output
            abbreviations_file: Path to abbreviations embeddings file
            use_abbreviations: Whether to use abbreviation resolution
            strict: Controls Stage 2 LLM prompt variant (default True).
            exact_match: If True, Stage 1 string match triggers immediate acceptance
                with no LLM calls (default False).
            disease_check: If True, Stage 3 LLM disease check runs as a final gate
                when Stages 1 and 2 did not early-return (default False).
        """
        self.embedding_manager = embedding_manager
        self.llm_client = llm_client
        self.debug = debug
        self.config = config  # Kept for API compatibility
        self.strict = strict
        self.exact_match = exact_match
        self.disease_check = disease_check
        self.index = None
        self.embedded_documents = None
        self.use_abbreviations = use_abbreviations

        # Initialize abbreviation searcher if enabled
        self.abbreviations_file = abbreviations_file
        self.abbreviation_searcher = None
        if use_abbreviations and abbreviations_file:
            self.initialize_abbreviation_searcher()

        # Direct verification with binary matching
        self.direct_verification_system_message = (
            "You are a clinical expert specializing in rare disease identification. "
            "Your task is to determine if the given entity represents a specific named rare disease.\n\n"
            "A rare disease is typically defined as a condition that affects fewer than 1 in 2,000 people. "
            "Examples include Fabry disease, Gaucher disease, Pompe disease, etc.\n\n"
            "When candidate definitions are provided, read them carefully — if a candidate's definition "
            "describes it as a grouping, classification node, meta-category, or umbrella term rather than "
            "a specific condition, that candidate does NOT confirm the entity as a rare disease.\n\n"
            "IMPORTANT EXCLUSIONS — respond 'NO' for any of the following:\n"
            "- Generic or anaphoric references that do not name a specific disease "
            "(e.g. 'the disorder', 'the disease', 'the condition', 'the syndrome', 'this condition')\n"
            "- Category or umbrella terms that describe a class of diseases rather than one specific disease "
            "(e.g. 'rare disorders', 'rare disease', 'disorder', 'rare genetic disease', 'disease group')\n"
            "- Ontological grouping entries in ORPHA that serve as classification nodes, not specific conditions\n"
            "- Symptoms or signs without a specific disease name (e.g. ataxia, seizures, fatigue)\n"
            "- Lab tests, measurements, or procedures\n\n"
            "YOUR RESPONSE MUST BE EXACTLY 'YES' OR 'NO' WITH NO ADDITIONAL TEXT."
        )

        # Lax matching: use candidates as contextual reference only; the LLM
        # decides based on best judgement without requiring an exact ORPHA match.
        self.lax_orpha_matching_system_message = (
            "You are a clinical expert with comprehensive knowledge of rare diseases. "
            "You will be given a medical entity and a list of rare disease candidates "
            "from the ORPHANET database as contextual reference. "
            "Your task is to determine, based on the clinical context and your medical "
            "knowledge, whether the entity is a rare disease — even if it does not "
            "exactly match any of the provided candidates. "
            "A rare disease typically affects fewer than 1 in 2,000 people.\n\n"
            "IMPORTANT EXCLUSIONS — respond 'NO' for any of the following:\n"
            "- Proteins, genes, enzymes, or molecular markers (e.g. ATM, BRCA1, albumin)\n"
            "- Symptoms or signs without a specific disease name (e.g. ataxia, seizures, fatigue)\n"
            "- Lab tests, measurements, or procedures\n"
            "- General categories or classes of conditions (e.g. 'peripheral nerve diseases', 'brain tumors')\n"
            "- Generic or anaphoric references that do not name a specific disease "
            "(e.g. 'the disorder', 'the disease', 'the condition', 'the syndrome', 'this condition', 'the illness')\n"
            "- Anything that is not itself a named disorder, syndrome, or disease"
        )

        # Caches for performance
        self.verification_cache = {}
        self.abbreviation_cache = {}

    def initialize_abbreviation_searcher(self):
        """Initialize the abbreviation searcher with embeddings file."""
        try:
            if not self.abbreviations_file:
                self._debug_print(
                    "No abbreviations file provided, abbreviation resolution disabled"
                )
                return

            self._debug_print(
                f"Initializing abbreviation searcher with {self.abbreviations_file}"
            )
            self.abbreviation_searcher = ToolSearcher(
                model_type=self.embedding_manager.model_type,
                model_name=self.embedding_manager.model_name,
                device="cpu",  # Use CPU for abbreviation searching to avoid GPU conflicts
                top_k=3,  # Get top 3 matches for abbreviations
            )
            self.abbreviation_searcher.load_embeddings(self.abbreviations_file)
            self._debug_print("Abbreviation searcher initialized successfully")
        except Exception as e:
            self._debug_print(f"Error initializing abbreviation searcher: {e}")
            self.abbreviation_searcher = None
            self.use_abbreviations = False

    def _debug_print(self, message: str, level: int = 0):
        """Print debug messages if debug mode is enabled."""
        if self.debug:
            indent = "  " * level
            print(f"{datetime.now().strftime('%H:%M:%S')} | {indent}{message}")

    def prepare_index(self, metadata: List[Dict]):
        """Prepare FAISS index from metadata for similarity search."""
        if self.index is None:
            self._debug_print("Preparing FAISS index for rare disease verification...")
            embeddings_array = self.embedding_manager.prepare_embeddings(metadata)
            self.index = self.embedding_manager.create_index(embeddings_array)
            self.embedded_documents = metadata
            self._debug_print(f"Index prepared with {len(metadata)} embedded documents")

    def clear_caches(self):
        """Clear all caches to prepare for a fresh evaluation run."""
        self.verification_cache = {}
        self.abbreviation_cache = {}
        self._debug_print("All caches cleared")

    def preprocess_entity(self, entity: str) -> str:
        """
        Minimal preprocessing of entity text.

        Args:
            entity: Raw entity text

        Returns:
            Preprocessed entity text
        """
        if not entity:
            return ""

        # Remove unnecessary metadata patterns like "(resolved)"
        cleaned = re.sub(r"\s*\([^)]*\)", "", entity)

        # Remove extra whitespace
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        return cleaned

    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        if not text:
            return ""

        # Convert to lowercase
        normalized = text.lower()

        # Remove punctuation except hyphens
        normalized = re.sub(r"[^\w\s-]", "", normalized)

        # Replace multiple spaces with a single space
        normalized = re.sub(r"\s+", " ", normalized)

        return normalized.strip()

    def _retrieve_similar_diseases(self, entity: str, k: int = 20) -> List[Dict]:
        """Retrieve similar rare diseases from the embeddings."""
        if self.index is None:
            raise ValueError("Index not prepared. Call prepare_index() first.")

        # Embed the query
        query_vector = self.embedding_manager.query_text(entity).reshape(1, -1)

        # Search for similar items
        distances, indices = self.embedding_manager.search(
            query_vector, self.index, k=min(800, len(self.embedded_documents))
        )

        # Extract unique metadata
        similar_diseases = []
        seen_metadata = set()

        for idx, distance in zip(indices[0], distances[0]):
            try:
                document = self.embedded_documents[idx]

                # Normalise to a flat dict regardless of storage layout
                if "unique_metadata" in document:
                    src = document["unique_metadata"]
                else:
                    src = document

                metadata_id = f"{src.get('name', '')}-{src.get('id', '')}"
                if metadata_id not in seen_metadata:
                    seen_metadata.add(metadata_id)
                    similar_diseases.append(
                        {
                            "name": src.get("name", ""),
                            "id": src.get("id", ""),
                            "definition": src.get("definition", ""),
                            "similarity_score": 1.0 / (1.0 + distance),
                        }
                    )
                    if len(similar_diseases) >= k:
                        break
            except Exception as e:
                print(f"Error processing metadata at index {idx}: {e}")
                continue

        return similar_diseases

    def check_abbreviation(self, entity: str, context: Optional[str] = None) -> Dict:
        """
        Check if an entity is a clinical abbreviation with case-sensitive exact matching.
        """
        # Skip if abbreviation checking is disabled
        if not self.use_abbreviations or not self.abbreviation_searcher:
            return {
                "is_abbreviation": False,
                "expanded_term": None,
                "method": "abbreviations_disabled",
            }

        # Handle empty entities
        if not entity:
            return {
                "is_abbreviation": False,
                "expanded_term": None,
                "method": "empty_entity",
            }

        # Create a cache key - preserve original case
        cache_key = f"abbr::{entity}"

        # Check cache
        if cache_key in self.abbreviation_cache:
            result = self.abbreviation_cache[cache_key]
            self._debug_print(
                f"Cache hit for abbreviation check '{entity}': {result['is_abbreviation']}",
                level=1,
            )
            return result

        self._debug_print(f"Checking if '{entity}' is an abbreviation", level=1)

        # Quick check if entity looks like an abbreviation
        looks_like_abbreviation = entity.isupper() or "." in entity or len(entity) <= 5

        if not looks_like_abbreviation:
            result = {
                "is_abbreviation": False,
                "expanded_term": None,
                "method": "quick_check",
            }
            self.abbreviation_cache[cache_key] = result
            return result

        # Search for abbreviation
        try:
            search_results = self.abbreviation_searcher.search(entity)

            if not search_results:
                result = {
                    "is_abbreviation": False,
                    "expanded_term": None,
                    "method": "no_match_found",
                }
                self.abbreviation_cache[cache_key] = result
                return result

            # Check for exact match (case-sensitive)
            exact_match = None
            for result in search_results:
                # The matched_term is the abbreviation from our database
                if (
                    result.get("matched_term", "") == entity
                ):  # Exact case-sensitive match
                    exact_match = result
                    self._debug_print(
                        f"Found exact case-sensitive match for '{entity}'", level=2
                    )
                    break

            # Use exactly matched result if found, otherwise use top result
            if exact_match:
                top_result = exact_match
                is_exact_match = True
                self._debug_print(f"Using exact match for '{entity}'", level=2)
            else:
                top_result = search_results[0]
                is_exact_match = False
                self._debug_print(
                    f"No exact match found, using top result for '{entity}'", level=2
                )

            # Extract information from the result
            similarity = top_result.get("similarity", 0.0)
            matched_term = top_result.get("matched_term", "")
            expanded_term = top_result.get("result", "")

            # Determine if this is a good match
            is_abbreviation = (
                is_exact_match or similarity > 0.96
            )  # needs to be high match.

            result = {
                "is_abbreviation": is_abbreviation,
                "expanded_term": expanded_term if is_abbreviation else None,
                "method": "exact_match" if is_exact_match else "similarity_match",
            }

            # Only include similarity if it's a similarity match and not exact
            if not is_exact_match and is_abbreviation:
                result["similarity_score"] = similarity

            # Include top matches for debugging
            if self.debug:
                result["all_matches"] = search_results[:3]

            self.abbreviation_cache[cache_key] = result

            if is_abbreviation:
                self._debug_print(
                    f"'{entity}' is an abbreviation for '{expanded_term}' (method: {result['method']})",
                    level=2,
                )
            else:
                self._debug_print(
                    f"'{entity}' is not a recognized abbreviation", level=2
                )

            return result

        except Exception as e:
            self._debug_print(f"Error checking abbreviation: {e}", level=2)
            result = {
                "is_abbreviation": False,
                "expanded_term": None,
                "method": "error",
            }
            self.abbreviation_cache[cache_key] = result
            return result

    def _check_if_disease(self, entity: str, context: Optional[str] = None) -> bool:
        """
        Simple check to determine if an entity represents a disease (not necessarily a rare one).

        Args:
            entity: Entity text to check
            context: Original sentence containing the entity

        Returns:
            Boolean indicating if the entity is likely a disease
        """
        # Create a cache key
        cache_key = f"disease_check::{entity}::{context or ''}"

        # Check cache first
        if cache_key in self.verification_cache:
            result = self.verification_cache[cache_key]
            self._debug_print(
                f"Cache hit for disease check '{entity}': {result}", level=1
            )
            return result

        self._debug_print(f"Checking if '{entity}' is a disease", level=1)

        # Define the system message for this specific check
        system_message = (
            "You are a medical expert specializing in disease classification. "
            "Your task is to determine if the given entity represents a disease or medical condition. "
            "Answer with ONLY 'YES' or 'NO' with no additional text."
        )

        # Format the context part
        context_part = ""
        if context:
            context_part = f"\nOriginal sentence: '{context}'"

        # Create the prompt
        prompt = (
            f"I need to determine if the term '{entity}' represents a disease or medical condition."
            f"{context_part}\n\n"
            f"A disease or medical condition is a pathological state or disorder with characteristic symptoms and causes, "
            f"such as cancer, diabetes, asthma, or genetic syndromes."
            f"\n\nIs '{entity}' a disease or medical condition? Answer with ONLY 'YES' or 'NO'."
        )

        # Query the LLM
        response = self.llm_client.query(prompt, system_message)

        # Parse the response - strictly look for "YES"
        response_text = response.strip().upper()
        is_disease = "YES" in response_text and "NO" not in response_text

        # Cache the result
        self.verification_cache[cache_key] = is_disease

        self._debug_print(f"Disease check result for '{entity}': {is_disease}", level=2)
        return is_disease

    def _build_candidates_text(self, candidates: List[Dict], n: int = 5) -> str:
        """Build a numbered candidate list string, including definitions when present."""
        items = []
        for i, c in enumerate(candidates[:n], 1):
            defn = c.get("definition", "").strip()
            if defn:
                items.append(f"{i}. {c['name']} (ORPHA:{c['id']}): {defn}")
            else:
                items.append(f"{i}. {c['name']} (ORPHA:{c['id']})")
        return "\n".join(items)

    def _string_match(
        self, entity: str, candidates: List[Dict]
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """Check if entity has an exact or high-similarity (>96%) string match in top-5 candidates.

        Returns:
            (matched, matched_name, matched_id)
        """
        normalized_entity = self._normalize_text(entity)
        for candidate in candidates[:5]:
            normalized_name = self._normalize_text(candidate["name"])
            if normalized_name == normalized_entity:
                self._debug_print(
                    f"Exact ORPHA string match: '{entity}' -> "
                    f"'{candidate['name']}' ({candidate['id']})",
                    level=2,
                )
                return True, candidate["name"], candidate["id"]
            similarity = fuzz.ratio(normalized_name, normalized_entity)
            if similarity > 96:
                self._debug_print(
                    f"Fuzzy ORPHA string match ({similarity}%): '{entity}' -> "
                    f"'{candidate['name']}' ({candidate['id']})",
                    level=2,
                )
                return True, candidate["name"], candidate["id"]
        return False, None, None

    def _llm_verify(
        self,
        entity: str,
        context: Optional[str],
        candidates_text: str,
        *,
        disease_check: bool = False,
    ) -> bool:
        """Single parameterized LLM call for both ORPHA matching and disease verification.

        ``disease_check`` and ``self.strict`` are independent dimensions:

        * ``disease_check=False`` → ORPHA-matching step; prompt varies by ``self.strict``.
        * ``disease_check=True``  → final disease-verification gate
          (``direct_verification_system_message``); ``self.strict`` does not alter
          this prompt — the gate is the same regardless of mode.

        Args:
            entity: Entity text to verify.
            context: Original sentence for grounding.
            candidates_text: Pre-built numbered ORPHA candidates string.
            disease_check: If True, runs the final disease-verification gate.
                If False, runs the ORPHA-matching step (strict or lax).

        Returns:
            True if LLM answered YES.
        """
        context_part = f"\nContext: '{context}'" if context else ""

        if disease_check:
            system_msg = self.direct_verification_system_message
            prompt = (
                f"I need to determine if the entity '{entity}' is a specific named rare disease or syndrome.\n\n"
                f"Here are ORPHANET database candidates retrieved for this entity. "
                f"Pay close attention to each candidate's definition — if a definition indicates the entry is a "
                f"meta-category, ontological grouping, or umbrella term rather than a specific condition, "
                f"that candidate does NOT confirm '{entity}' as a rare disease:\n\n"
                f"{candidates_text}\n"
                f"Context around entity:{context_part}\n\n"
                f"Is '{entity}' a specific named rare disease or syndrome "
                f"(not a category term, ontological grouping, or generic reference)?\n"
                f"Respond with ONLY 'YES' or 'NO'."
            )
        else:
            # ORPHA-matching step: strict vs. lax are independent of disease_check
            if self.strict:
                system_msg = (
                    "You are a medical expert specializing in rare diseases with comprehensive knowledge of the ORPHANET database. "
                    "Your task is to determine if a given medical term is semantically equivalent to any of the ORPHA entries provided. "
                    "For a match to be valid, the entities must refer to the same specific rare disease or syndrome, not just similar conditions."
                )
                prompt = (
                    f"I need to determine if the term '{entity}' is among any of these rare diseases from ORPHANET:\n\n"
                    f"{candidates_text}\n\n"
                    f"Context around entity:\n{context}\n\n"
                    f"Decide if '{entity}' is the same disease as any of these entries. Consider synonyms, abbreviations, and variant names. "
                    f"Account for spelling variations and different naming conventions for the same disease entity.\n\n"
                    f"For variants of common diseases, it must be explicitly marked as a rare variant. "
                    f"If there is a partial match, i.e cholangitis vs. sclerosing cholangitis. There must be a mention of its descriptor (sclerosing) in the term or context itself, otherwise it's an invalid match. "
                    f"Respond with ONLY 'YES' if there is a match, and 'NO' if there is no match."
                )
            else:  # lax ORPHA match
                system_msg = self.lax_orpha_matching_system_message
                prompt = (
                    f"I need to determine if the term '{entity}' is a rare disease.\n\n"
                    f"Here are related rare disease entries from ORPHANET for contextual reference:\n\n"
                    f"{candidates_text}\n\n"
                    f"Clinical context:\n{context}\n\n"
                    f"Using the candidates above as reference and your medical knowledge, decide whether '{entity}' "
                    f"is a specific named rare disease or syndrome. You do not need an exact match — use your best clinical judgement.\n\n"
                    f"Proteins, lab tests, and generic/category terms are not specific diseases — respond 'NO' for these.\n\n"
                    f"Also respond 'NO' if '{entity}' is a broad category or umbrella term (e.g. 'rare disorders', 'rare disease', "
                    f"'disorder', 'rare genetic disease') rather than a specific named condition.\n\n"
                    f"If any candidate's definition indicates it is a referring to a large broad category of terms, "
                    f"that should inform your judgement that '{entity}' may not be a specific rare disease. However, terms that describe a group or class of rare diseases should still be considered.\n\n"
                    f"Respond with ONLY 'YES' if it is a specific named rare disease or syndrome or class of conditions, and 'NO' if it is not."
                )

        response = self.llm_client.query(prompt, system_msg)
        return "YES" in response.strip().upper()

    def verify_rare_disease(self, entity: str, context: Optional[str] = None) -> Dict:
        """Verify if an entity is a rare disease through an optional three-stage pipeline.

        Stage 1 — String match (always runs):
            If exact_match=True and match found → immediate YES, no LLM calls.
            Otherwise match result is carried forward; pipeline continues to Stage 2.

        Stage 2 — LLM ORPHA match (runs when exact_match=False OR no string match):
            Strict or lax depending on self.strict.
            On YES → immediate YES, no further stages.
            On NO  → continue (no early stop).

        Stage 3 — LLM disease check (runs only when disease_check=True):
            Final YES/NO gate using direct_verification_system_message.
            Only reached when Stages 1 and 2 did not return early.
            If disease_check=False: reject (Stage 2 LLM has already said NO).

        Args:
            entity: Entity text to verify.
            context: Original sentence containing the entity.

        Returns:
            Dict with at minimum {"is_rare_disease": bool, "method": str}.
        """
        if not entity:
            return {"is_rare_disease": False, "method": "empty_entity"}

        cache_key = f"verify::{entity}::{context or ''}"
        if cache_key in self.verification_cache:
            cached = self.verification_cache[cache_key]
            self._debug_print(
                f"Cache hit for '{entity}': {cached['is_rare_disease']}", level=1
            )
            return cached

        self._debug_print(f"Verifying '{entity}' as rare disease", level=1)

        # ── Retrieve ORPHA candidates ────────────────────────────────────────
        candidates = self._retrieve_similar_diseases(entity)
        if not candidates:
            self._debug_print(f"No ORPHA candidates for '{entity}'", level=2)
            result = {"is_rare_disease": False, "method": "no_orpha_candidates"}
            self.verification_cache[cache_key] = result
            return result

        candidates_text = self._build_candidates_text(candidates)

        # ── Stage 1: String match ────────────────────────────────────────────
        string_matched, matched_name, matched_id = self._string_match(entity, candidates)

        if string_matched:
            if self.exact_match:
                result = {
                    "is_rare_disease": True,
                    "method": "exact_match_shortcircuit",
                    "matched_term": matched_name,
                    "orpha_id": matched_id,
                    "match_method": "string_match",
                }
                self.verification_cache[cache_key] = result
                self._debug_print(
                    f"exact_match mode: '{entity}' -> '{matched_name}' ({matched_id})",
                    level=2,
                )
                return result
            self._debug_print(
                f"String match found ('{matched_name}'), exact_match disabled — continuing",
                level=2,
            )

        # ── Stage 2: LLM ORPHA match ─────────────────────────────────────────
        # Skipped only when exact_match=True short-circuited in Stage 1.
        # When exact_match=False, LLM always runs regardless of string match.
        orpha_matched = False
        if not string_matched or not self.exact_match:
            orpha_matched = self._llm_verify(
                entity, context, candidates_text, disease_check=False
            )
            if orpha_matched:
                matched_name = matched_name or candidates[0]["name"]
                matched_id = matched_id or candidates[0]["id"]
                self._debug_print(
                    f"LLM {'strict' if self.strict else 'lax'} ORPHA match: "
                    f"'{entity}' -> '{matched_name}' ({matched_id})",
                    level=2,
                )
                result = {
                    "is_rare_disease": True,
                    "method": "multi_step_verification",
                    "matched_term": matched_name,
                    "orpha_id": matched_id,
                    "match_method": "llm_semantic_match",
                }
                self.verification_cache[cache_key] = result
                return result
            self._debug_print(f"No LLM ORPHA match for '{entity}'", level=2)

        # ── Stage 3: LLM disease check (optional final gate) ─────────────────
        if not self.disease_check:
            # Stage 2 LLM has already run and returned NO; reject.
            result = {"is_rare_disease": False, "method": "not_in_orpha"}
            self.verification_cache[cache_key] = result
            self._debug_print(f"No disease check: '{entity}' is not a rare disease", level=2)
            return result

        is_disease = self._llm_verify(
            entity, context, candidates_text, disease_check=True
        )

        if is_disease:
            result = {"is_rare_disease": True, "method": "multi_step_verification"}
            if matched_name and matched_id:
                result["matched_term"] = matched_name
                result["orpha_id"] = matched_id
                result["match_method"] = "string_match" if string_matched else "no_orpha_match"
        else:
            result = {
                "is_rare_disease": False,
                "method": (
                    "failed_disease_check" if string_matched else "not_in_orpha"
                ),
            }

        self.verification_cache[cache_key] = result
        self._debug_print(
            f"'{entity}' is{'' if is_disease else ' not'} a rare disease", level=2
        )
        return result

    def process_entity(self, entity: str, context: Optional[str] = None) -> Dict:
        """
        Process an entity through the multistage pipeline with two-step verification.

        Args:
            entity: Entity text to process
            context: Original sentence containing the entity

        Returns:
            Dictionary with processing results
        """
        # Handle empty entities
        if not entity:
            return {
                "status": "not_rare_disease",
                "entity": None,
                "original_entity": entity,
                "method": "empty_entity",
            }

        self._debug_print(f"Processing entity: '{entity}'", level=0)

        # Clean and preprocess the entity
        cleaned_entity = self.preprocess_entity(entity)
        if not cleaned_entity:
            return {
                "status": "not_rare_disease",
                "entity": None,
                "original_entity": entity,
                "method": "empty_after_preprocessing",
            }

        # STAGE 1: Check if it's an abbreviation
        entity_to_verify = cleaned_entity
        is_abbreviation = False
        expanded_term = None

        if self.use_abbreviations and self.abbreviation_searcher:
            abbreviation_result = self.check_abbreviation(cleaned_entity, context)

            if abbreviation_result["is_abbreviation"]:
                expanded_term = abbreviation_result["expanded_term"]
                is_abbreviation = True
                entity_to_verify = expanded_term
                self._debug_print(
                    f"'{entity}' is an abbreviation for '{expanded_term}'", level=1
                )

        # STAGE 2: Verify if it's a rare disease using the two-step process
        verification_result = self.verify_rare_disease(entity_to_verify, context)

        if verification_result["is_rare_disease"]:
            self._debug_print(f"'{entity_to_verify}' is a rare disease", level=1)

            result = {
                "status": "verified_rare_disease",
                "entity": entity_to_verify,
                "original_entity": entity,
                "method": (
                    "abbreviation_expansion"
                    if is_abbreviation
                    else verification_result["method"]
                ),
            }

            # Include expansion information if this was an abbreviation
            if is_abbreviation:
                result["expanded_term"] = entity_to_verify

            # Include ORPHA canonical name and ID if available
            if "matched_term" in verification_result:
                result["matched_term"] = verification_result["matched_term"]
            if "orpha_id" in verification_result:
                result["orpha_id"] = verification_result["orpha_id"]

            return result

        # Not a rare disease
        self._debug_print(f"'{entity_to_verify}' is not a rare disease", level=1)
        return {
            "status": "not_rare_disease",
            "entity": None,
            "original_entity": entity,
            "expanded_term": entity_to_verify if is_abbreviation else None,
            "method": verification_result["method"],
        }

    def batch_process(self, entities_with_context: List[Dict]) -> List[Dict]:
        """
        Process a batch of entities with their contexts.

        Args:
            entities_with_context: List of dicts with 'entity' and 'context' keys

        Returns:
            List of dicts with processing results (verified rare diseases only)
        """
        self._debug_print(f"Processing batch of {len(entities_with_context)} entities")

        # Ensure input data is in correct format
        formatted_entries = []
        for item in entities_with_context:
            if isinstance(item, dict) and "entity" in item:
                formatted_entries.append(item)
            elif isinstance(item, str):
                formatted_entries.append({"entity": item, "context": ""})
            else:
                self._debug_print(f"Skipping invalid entry: {item}")
                continue

        # Remove duplicates while preserving order
        unique_entries = []
        seen = set()
        for item in formatted_entries:
            entity = str(item.get("entity", "")).lower().strip()
            context = str(item.get("context", ""))

            # Create a unique key for deduplication
            entry_key = f"{entity}::{context}"

            if entry_key not in seen and entity:
                seen.add(entry_key)
                unique_entries.append(item)

        self._debug_print(f"Found {len(unique_entries)} unique entity-context pairs")

        # Process each entity through the pipeline
        results = []
        for item in unique_entries:
            entity = item.get("entity", "")
            context = item.get("context", "")

            result = self.process_entity(entity, context)

            # Only include entities that are verified rare diseases
            if result["status"] == "verified_rare_disease":
                # Add original context
                result["context"] = context
                results.append(result)

        self._debug_print(f"Identified {len(results)} verified rare diseases")
        return results


# Factory function to create a verifier
def create_rd_verifier(
    embedding_manager,
    llm_client,
    config=None,
    debug=False,
    abbreviations_file=None,
    use_abbreviations=True,
    strict=True,
    exact_match=False,
    disease_check=False,
):
    """
    Factory function to create a multistage rare disease verifier.

    Args:
        embedding_manager: Manager for embedding operations
        llm_client: LLM client for verification
        config: Configuration for the verifier (currently unused)
        debug: Enable debug output
        abbreviations_file: Path to abbreviations embeddings file
        use_abbreviations: Whether to use abbreviation resolution
        strict: Controls Stage 2 LLM prompt — True requires same specific ORPHA
            disease; False uses ORPHA candidates as reference only (higher recall).
        exact_match: If True, a Stage 1 string match immediately accepts the entity
            with no LLM calls.
        disease_check: If True, Stage 3 LLM disease check runs as a final gate
            when Stages 1 and 2 did not return early (default False).

    Returns:
        MultiStageRDVerifier instance
    """
    return MultiStageRDVerifier(
        embedding_manager=embedding_manager,
        llm_client=llm_client,
        config=config,
        debug=debug,
        abbreviations_file=abbreviations_file,
        use_abbreviations=use_abbreviations,
        strict=strict,
        exact_match=exact_match,
        disease_check=disease_check,
    )
