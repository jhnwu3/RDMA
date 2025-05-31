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
    ):
        """
        Initialize the multistage rare disease verifier.

        Args:
            embedding_manager: Manager for embedding operations
            llm_client: LLM client for verification
            config: Configuration for the verifier (currently unused, kept for API compatibility)
            debug: Enable debug output
            abbreviations_file: Path to abbreviations embeddings file
            use_abbreviations: Whether to use abbreviation resolution
        """
        self.embedding_manager = embedding_manager
        self.llm_client = llm_client
        self.debug = debug
        self.config = config  # Kept for API compatibility
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
            "Your task is to determine if the given entity represents a rare disease based on the provided candidates."
            "\n\nA rare disease is typically defined as a condition that affects fewer than 1 in 2,000 people. "
            "Examples include Fabry disease, Gaucher disease, Pompe disease, etc."
            "\n\nAFTER REVIEWING THE CANDIDATES, respond with ONLY 'YES' if:"
            "\n- The entity EXACTLY or CLOSELY matches any rare disease candidate, OR"
            "\n- The entity clearly refers to a rare disease even if not in the candidates"
            "\n\nRespond with ONLY 'NO' if:"
            "\n- The entity is NOT a rare disease (e.g., common diseases, symptoms without specificity, lab tests)"
            "\n- The entity does not represent a specific rare condition"
            "\n\nYOUR RESPONSE MUST BE EXACTLY 'YES' OR 'NO' WITH NO ADDITIONAL TEXT."
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

                # Check if document has 'unique_metadata' or direct fields
                if "unique_metadata" in document:
                    metadata = document["unique_metadata"]
                else:
                    # Assume direct structure
                    metadata_id = f"{document.get('name', '')}-{document.get('id', '')}"

                    if metadata_id not in seen_metadata:
                        seen_metadata.add(metadata_id)
                        similar_diseases.append(
                            {
                                "name": document.get("name", ""),
                                "id": document.get("id", ""),
                                "definition": document.get("definition", ""),
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

    # Technically this one has much higher precision.
    def verify_rare_disease(self, entity: str, context: Optional[str] = None) -> Dict:
        """
        Verify if an entity is a rare disease through a multi-step process:
        1. Check if the term exists in the ORPHA ontology via exact/fuzzy matching
        2. If no match, use LLM to check for semantic matches in ORPHA candidates
        3. If still no match, check if similarity is high enough to proceed
        4. Verify that it's an actual disease with an LLM sanity check

        Args:
            entity: Entity text to verify
            context: Original sentence containing the entity

        Returns:
            Dictionary with verification results
        """
        # Handle empty entities
        if not entity:
            return {"is_rare_disease": False, "method": "empty_entity"}

        # Create a cache key
        cache_key = f"verify::{entity}::{context or ''}"

        # Check cache first
        if cache_key in self.verification_cache:
            result = self.verification_cache[cache_key]
            self._debug_print(
                f"Cache hit for rare disease verification '{entity}': {result['is_rare_disease']}",
                level=1,
            )
            return result

        self._debug_print(
            f"Verifying if '{entity}' is a rare disease via multi-step process", level=1
        )

        # STEP 1: Retrieve similar entities from ORPHA ontology
        similar_entities = self._retrieve_similar_diseases(entity)

        if not similar_entities:
            # No candidates found in ORPHA ontology
            self._debug_print(f"No ORPHA candidates found for '{entity}'", level=2)
            result = {"is_rare_disease": False, "method": "no_orpha_candidates"}
            self.verification_cache[cache_key] = result
            return result

        # STEP 2: Check if any of the candidates have high similarity to our entity
        has_matching_entity = False
        matched_entity = None
        matched_id = None

        for entity_data in similar_entities[:5]:  # Only check top 5 candidates
            normalized_name = self._normalize_text(entity_data["name"])
            normalized_entity = self._normalize_text(entity)

            # Check for exact match
            if normalized_name == normalized_entity:
                self._debug_print(
                    f"Exact ORPHA match found: '{entity}' matches '{entity_data['name']}' ({entity_data['id']})",
                    level=2,
                )
                has_matching_entity = True
                matched_entity = entity_data["name"]
                matched_id = entity_data["id"]
                break

            # Check for high similarity match (over 90%)
            similarity = fuzz.ratio(normalized_name, normalized_entity)
            if similarity > 96:
                self._debug_print(
                    f"High similarity ORPHA match ({similarity}%): '{entity}' matches '{entity_data['name']}' ({entity_data['id']})",
                    level=2,
                )
                has_matching_entity = True
                matched_entity = entity_data["name"]
                matched_id = entity_data["id"]
                break

        # STEP 3: NEW - LLM-based matching if no exact/fuzzy match found
        llm_match_result = False
        if not has_matching_entity:
            # Format entities for the LLM prompt
            entity_items = []
            for i, entity_data in enumerate(similar_entities[:5], 1):
                entity_items.append(
                    f"{i}. {entity_data['name']} (ORPHA:{entity_data['id']})"
                )

            entities_text = "\n".join(entity_items)

            # Create the LLM matching system message
            orpha_matching_system_message = (
                "You are a medical expert specializing in rare diseases with comprehensive knowledge of the ORPHANET database. "
                "Your task is to determine if a given medical term is semantically equivalent to any of the ORPHA entries provided. "
                "For a match to be valid, the entities must refer to the same specific rare disease or syndrome, not just similar conditions."
            )

            # Create the binary YES/NO matching prompt
            matching_prompt = (
                f"I need to determine if the term '{entity}' is among any of these rare diseases from ORPHANET:\n\n"
                f"{entities_text}\n\n"
                f"Context around entity:\n"
                f"{context}\n\n"
                f"Decide if '{entity}' is the same disease as any of these entries. Consider synonyms, abbreviations, and variant names. "
                f"Account for spelling variations and different naming conventions for the same disease entity.\n\n"
                f"For variants of common diseases, it must be explicitly marked as a rare variant."
                f"If there is a partial match, i.e cholangitis vs. sclerosing cholangitis. There must be a mention of its descriptor (sclerosing) in the term or context itself, otherwise it's an invalid match."
                f"Respond with ONLY 'YES' if there is a match, and 'NO' if there is no match."
            )

            # Query the LLM
            llm_response = self.llm_client.query(
                matching_prompt, orpha_matching_system_message
            )

            # Parse the response - strictly look for "YES"
            llm_match_result = "YES" in llm_response.strip().upper()

            # If LLM found a match, use the top similarity candidate
            if llm_match_result and similar_entities:
                has_matching_entity = True
                top_candidate = similar_entities[0]
                matched_entity = top_candidate["name"]
                matched_id = top_candidate["id"]
                self._debug_print(
                    f"LLM identified semantic match: '{entity}' matches '{matched_entity}' ({matched_id})",
                    level=2,
                )

        # STEP 4: Handle cases where neither string nor LLM matching found a match
        if not has_matching_entity:
            # No close match in ORPHA ontology
            self._debug_print(
                f"No close ORPHA match found via string or LLM matching for '{entity}'",
                level=2,
            )

            # For borderline cases, proceed to LLM check as we might have a variant name not in the ontology
            if (
                similar_entities[0]["similarity_score"] > 0.95
            ):  # Threshold for proceeding. High Match!
                self._debug_print(
                    f"But similarity is high enough to proceed to disease check",
                    level=2,
                )
            else:
                # No matches and similarity too low - return immediately without disease verification
                result = {"is_rare_disease": False, "method": "low_orpha_similarity"}
                self.verification_cache[cache_key] = result
                return result

        # STEP 5: Verify that it's a disease using the LLM
        # Format entities for the LLM prompt
        entity_items = []
        for i, entity_data in enumerate(similar_entities[:5], 1):
            entity_items.append(f"{i}. '{entity_data['name']}' ({entity_data['id']})")

        entities_text = "\n".join(entity_items)

        # Create context part
        context_part = ""
        if context:
            context_part = f"\nOriginal sentence context: '{context}'"

        # Create the binary YES/NO matching prompt
        prompt = (
            f"I need to determine if the entity '{entity}' represents a disease."
            f"\n\nHere are some similar medical entities from a rare disease database for context that may be of rare diseases:"
            f"\n\n{entities_text}\n"
            f"Context around entity:"
            f"{context_part}\n\n"
            f"First, determine if '{entity}' is actually a disease or medical condition (not a lab measurement, protein, enzyme, etc.)."
            # f"\nThen, determine if it's a disease (better if a rare disease typically affecting fewer than 1 in 2,000 people)."
            f"\n\nIs '{entity}' a disease?"
            f"\nRespond with ONLY 'YES' or 'NO'."
        )

        # Query the LLM
        response = self.llm_client.query(
            prompt, self.direct_verification_system_message
        )

        # Parse the response - strictly look for "YES" or "NO"
        response_text = response.strip().upper()
        is_rare_disease = "YES" in response_text and "NO" not in response_text

        # STEP 6: Create result based on all verification steps
        if is_rare_disease:
            result = {"is_rare_disease": True, "method": "multi_step_verification"}

            # Include the matched entity information if available
            if matched_entity and matched_id:
                result["matched_term"] = matched_entity
                result["orpha_id"] = matched_id
                # Add the match method
                if llm_match_result:
                    result["match_method"] = "llm_semantic_match"
                else:
                    result["match_method"] = "string_similarity_match"
        else:
            result = {
                "is_rare_disease": False,
                "method": (
                    "failed_disease_check" if has_matching_entity else "not_in_orpha"
                ),
            }

        # Cache the result
        self.verification_cache[cache_key] = result

        self._debug_print(
            f"Multi-step verification: '{entity}' is{'' if is_rare_disease else ' not'} a rare disease",
            level=2,
        )
        return result

    # this one gives higher recall, but much lower precision. Namely, because we do need to check which ones are a disease.
    # def verify_rare_disease(self, entity: str, context: Optional[str] = None) -> Dict:
    #     """
    #     Verify if an entity is a rare disease through a two-step process:
    #     1. Check if the term exists in the ORPHA ontology via exact/fuzzy matching
    #     2. If no match, use LLM to check for semantic matches in ORPHA candidates

    #     Args:
    #         entity: Entity text to verify
    #         context: Original sentence containing the entity

    #     Returns:
    #         Dictionary with verification results
    #     """
    #     # Handle empty entities
    #     if not entity:
    #         return {"is_rare_disease": False, "method": "empty_entity"}

    #     # Create a cache key
    #     cache_key = f"verify::{entity}::{context or ''}"

    #     # Check cache first
    #     if cache_key in self.verification_cache:
    #         result = self.verification_cache[cache_key]
    #         self._debug_print(
    #             f"Cache hit for rare disease verification '{entity}': {result['is_rare_disease']}",
    #             level=1,
    #         )
    #         return result

    #     self._debug_print(
    #         f"Verifying if '{entity}' is a rare disease via simplified process", level=1
    #     )

    #     # STEP 1: Retrieve similar entities from ORPHA ontology
    #     similar_entities = self._retrieve_similar_diseases(entity)

    #     if not similar_entities:
    #         # No candidates found in ORPHA ontology
    #         self._debug_print(f"No ORPHA candidates found for '{entity}'", level=2)
    #         result = {"is_rare_disease": False, "method": "no_orpha_candidates"}
    #         self.verification_cache[cache_key] = result
    #         return result

    #     # STEP 2: Check if any of the candidates have high similarity to our entity
    #     has_matching_entity = False
    #     matched_entity = None
    #     matched_id = None

    #     for entity_data in similar_entities[:5]:  # Only check top 5 candidates
    #         normalized_name = self._normalize_text(entity_data["name"])
    #         normalized_entity = self._normalize_text(entity)

    #         # Check for exact match
    #         if normalized_name == normalized_entity:
    #             self._debug_print(
    #                 f"Exact ORPHA match found: '{entity}' matches '{entity_data['name']}' ({entity_data['id']})",
    #                 level=2,
    #             )
    #             has_matching_entity = True
    #             matched_entity = entity_data["name"]
    #             matched_id = entity_data["id"]
    #             break

    #         # Check for high similarity match (over 90%)
    #         similarity = fuzz.ratio(normalized_name, normalized_entity)
    #         if similarity > 96:
    #             self._debug_print(
    #                 f"High similarity ORPHA match ({similarity}%): '{entity}' matches '{entity_data['name']}' ({entity_data['id']})",
    #                 level=2,
    #             )
    #             has_matching_entity = True
    #             matched_entity = entity_data["name"]
    #             matched_id = entity_data["id"]
    #             break

    #     # If we found an exact match, return it immediately
    #     if has_matching_entity:
    #         result = {
    #             "is_rare_disease": True,
    #             "method": "string_similarity_match",
    #             "matched_term": matched_entity,
    #             "orpha_id": matched_id,
    #         }
    #         self.verification_cache[cache_key] = result
    #         return result

    #     # STEP 3: Use LLM for semantic matching when exact match fails
    #     # Format entities for the LLM prompt
    #     entity_items = []
    #     for i, entity_data in enumerate(similar_entities[:5], 1):
    #         entity_items.append(
    #             f"{i}. {entity_data['name']} (ORPHA:{entity_data['id']})"
    #         )

    #     entities_text = "\n".join(entity_items)

    #     # Create the LLM matching system message
    #     orpha_matching_system_message = (
    #         "You are a medical expert specializing in rare diseases with comprehensive knowledge of the ORPHANET database. "
    #         "Your task is to determine if a given medical term is semantically equivalent to any of the ORPHA entries provided. "
    #         "For a match to be valid, the entities must refer to the same specific rare disease or syndrome, not just similar conditions."
    #     )

    #     # Create the binary YES/NO matching prompt
    #     matching_prompt = (
    #         f"I need to determine if the term '{entity}' is among any of these rare diseases from ORPHANET:\n\n"
    #         f"{entities_text}\n\n"
    #         f"Context around entity:"
    #         f"{context if context else 'No additional context available.'}\n\n"
    #         f"Decide if '{entity}' is the same disease as any of these entries. Consider synonyms, abbreviations, and variant names. "
    #         f"Account for spelling variations and different naming conventions for the same disease entity.\n\n"
    #         f"For variants of common diseases, it must be explicitly marked as a rare variant."
    #         f"If there is a partial match, i.e cholangitis vs. sclerosing cholangitis. There must be a mention of its descriptor (sclerosing) in the term or context itself, otherwise it's an invalid match."
    #         f"Respond with ONLY 'YES' if there is a match, and 'NO' if there is no match."
    #     )

    #     # Query the LLM
    #     llm_response = self.llm_client.query(
    #         matching_prompt, orpha_matching_system_message
    #     )

    #     # Parse the response - strictly look for "YES"
    #     is_rare_disease = "YES" in llm_response.strip().upper()

    #     # Create result based on LLM matching
    #     if is_rare_disease:
    #         # Use the top candidate as the match
    #         top_candidate = similar_entities[0]
    #         result = {
    #             "is_rare_disease": True,
    #             "method": "llm_semantic_match",
    #             "matched_term": top_candidate["name"],
    #             "orpha_id": top_candidate["id"],
    #         }
    #     else:
    #         result = {"is_rare_disease": False, "method": "llm_rejection"}

    #     # Cache the result
    #     self.verification_cache[cache_key] = result

    #     self._debug_print(
    #         f"LLM semantic matching: '{entity}' is{'' if is_rare_disease else ' not'} a rare disease",
    #         level=2,
    #     )
    #     return result

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
                "entity": verification_result.get("matched_term", entity_to_verify),
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

            # Include ORPHA ID if available
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
    )
