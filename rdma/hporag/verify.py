import json
import re
import numpy as np
import torch
from datetime import datetime
from fuzzywuzzy import fuzz
from typing import List, Dict, Any, Optional, Tuple, Union, Set
import itertools
import time
from rdma.utils.search_tools import ToolSearcher


# For compatibility with the original implementation
class HPOVerifierConfig:
    """Configuration for when to use retrieval and context in the HPO verification pipeline."""

    def __init__(
        self,
        use_retrieval_for_direct=True,
        use_retrieval_for_implies=True,
        use_retrieval_for_extract=True,
        use_retrieval_for_validation=True,
        use_retrieval_for_implication=True,
        use_context_for_direct=True,
        use_context_for_implies=True,
        use_context_for_extract=True,
        use_context_for_validation=False,
        use_context_for_implication=True,
    ):
        # Retrieval settings
        self.use_retrieval_for_direct = use_retrieval_for_direct
        self.use_retrieval_for_implies = use_retrieval_for_implies
        self.use_retrieval_for_extract = use_retrieval_for_extract
        self.use_retrieval_for_validation = use_retrieval_for_validation
        self.use_retrieval_for_implication = use_retrieval_for_implication

        # Context usage settings
        self.use_context_for_direct = use_context_for_direct
        self.use_context_for_implies = use_context_for_implies
        self.use_context_for_extract = use_context_for_extract
        self.use_context_for_validation = use_context_for_validation
        self.use_context_for_implication = use_context_for_implication

    def to_dict(self):
        """Convert configuration to a dictionary."""
        return {
            "retrieval": {
                "direct": self.use_retrieval_for_direct,
                "implies": self.use_retrieval_for_implies,
                "extract": self.use_retrieval_for_extract,
                "validation": self.use_retrieval_for_validation,
                "implication": self.use_retrieval_for_implication,
            },
            "context": {
                "direct": self.use_context_for_direct,
                "implies": self.use_context_for_implies,
                "extract": self.use_context_for_extract,
                "validation": self.use_context_for_validation,
                "implication": self.use_context_for_implication,
            },
        }

    @classmethod
    def from_dict(cls, config_dict):
        """Create configuration from a dictionary."""
        return cls(
            use_retrieval_for_direct=config_dict["retrieval"]["direct"],
            use_retrieval_for_implies=config_dict["retrieval"]["implies"],
            use_retrieval_for_extract=config_dict["retrieval"]["extract"],
            use_retrieval_for_validation=config_dict["retrieval"]["validation"],
            use_retrieval_for_implication=config_dict["retrieval"]["implication"],
            use_context_for_direct=config_dict["context"]["direct"],
            use_context_for_implies=config_dict["context"]["implies"],
            use_context_for_extract=config_dict["context"]["extract"],
            use_context_for_validation=config_dict["context"]["validation"],
            use_context_for_implication=config_dict["context"]["implication"],
        )

    def __str__(self):
        """String representation of the configuration."""
        return str(self.to_dict())


class MultiStageHPOVerifierV4:
    """
    Enhanced multi-stage HPO verifier with simplified lab test analysis pipeline.

    This version implements a streamlined workflow:
    1. Direct phenotype verification with retrieved candidates
    2. Lab test detection (only if numbers are present in entity/context)
    3. Lab test analysis with reference ranges
    4. Implied phenotype determination

    Lab test abnormalities are directly translated to phenotypes.
    """

    def __init__(
        self,
        embedding_manager,
        llm_client,
        config=None,
        debug=False,
        lab_embeddings_file=None,
    ):
        """Initialize with specific configuration and optional lab embeddings file."""
        self.embedding_manager = embedding_manager
        self.llm_client = llm_client
        self.debug = debug
        self.index = None
        self.embedded_documents = None
        self.config = config or HPOVerifierConfig()
        self.candidate_count = 20

        # Lab test tools
        self.lab_embeddings_file = lab_embeddings_file
        self.lab_searcher = None
        if lab_embeddings_file:
            self.initialize_lab_searcher()

        # Direct verification with binary matching
        self.direct_verification_system_message = (
            "You are a clinical expert specializing in phenotype identification for Human Phenotype Ontology (HPO) mapping. "
            "Your task is to determine if the given entity represents a valid human phenotype based on the provided HPO candidates."
            "\n\nA valid phenotype must describe an abnormal characteristic or trait, not just a normal "
            "anatomical structure, physiological process, laboratory test, or medication."
            "\n\nAFTER REVIEWING THE CANDIDATES, respond with ONLY 'YES' if:"
            "\n- The entity EXACTLY or CLOSELY matches any HPO candidate, OR"
            "\n- The entity clearly describes a phenotype even if not in the candidates"
            "\n\nRespond with ONLY 'NO' if:"
            "\n- The entity is NOT a phenotype (e.g., normal anatomy, medication, lab test without value)"
            "\n- The entity does not represent any abnormal human trait or characteristic"
            "\n\nYOUR RESPONSE MUST BE EXACTLY 'YES' OR 'NO' WITH NO ADDITIONAL TEXT."
        )

        # Lab test identification (simplified)
        self.lab_identification_system_message = (
            "You are a clinical expert analyzing laboratory test information in clinical notes. "
            "Your task is to determine if the given entity contains information about a laboratory test with a measured value."
            "\n\nRespond with ONLY 'YES' if:"
            "\n- The entity contains a lab test name AND a numerical value/result"
            "\n- The entity clearly refers to a laboratory measurement with a value"
            "\n\nRespond with ONLY 'NO' if:"
            "\n- The entity only mentions a lab test without a specific value"
            "\n- The entity is not related to laboratory measurements"
            "\n\nYOUR RESPONSE MUST BE EXACTLY 'YES' OR 'NO' WITH NO ADDITIONAL TEXT."
        )

        # Lab test extraction and analysis (combined)
        self.lab_analysis_system_message = (
            "You are a clinical laboratory expert analyzing laboratory test results. "
            "Extract and analyze the lab test from the provided entity and determine if the value is abnormal."
            "\n\nFor ABNORMAL results, provide a medically precise description of the abnormality "
            "(e.g., 'elevated glucose', 'decreased hemoglobin', 'leukocytosis')."
            "\n\nFor NORMAL results, simply state 'normal'."
            "\n\nProvide your response in this EXACT JSON format:"
            "\n{"
            '\n  "lab_name": "[extracted lab test name]",'
            '\n  "value": "[extracted value with units if available]",'
            '\n  "is_abnormal": true/false,'
            '\n  "abnormality": "[descriptive term for the abnormality, or \'normal\' if not abnormal]",'
            '\n  "direction": "[high/low/normal]",'
            '\n  "confidence": [0.0-1.0 value indicating your confidence]'
            "\n}"
            "\n\nReturn ONLY the JSON with no additional text."
        )

        # Implied phenotype check - binary YES/NO
        self.implied_phenotype_system_message = (
            "You are a clinical expert specializing in phenotype identification for Human Phenotype Ontology (HPO) mapping. "
            "Determine if the given term DIRECTLY AND UNAMBIGUOUSLY implies a specific phenotype, even though it's not a direct phenotype itself. "
            "Be extremely conservative - only say YES if the implication is clear and specific."
            "\nLaboratory values, medications, or procedures DO NOT imply phenotypes unless there is explicit abnormality mentioned."
            "\nIf you're uncertain or the implication requires multiple assumptions, say NO."
            "\n\nEXAMPLES OF VALID IMPLICATIONS:"
            "\n- 'Elevated white blood cell count of 15,000' implies 'leukocytosis' (YES - explicit abnormality)"
            "\n- 'E. coli growing in urine culture' implies 'bacteriuria' (YES - specific finding)"
            "\n- 'Hemoglobin of 6.5 g/dL' implies 'anemia' (YES - clearly below normal range)"
            "\n\nEXAMPLES OF INVALID IMPLICATIONS:"
            "\n- 'White blood cell count' does NOT imply 'leukocytosis' (NO - no value specified)"
            "\n- 'Taking insulin' does NOT imply 'diabetes mellitus' (NO - medications alone don't imply diagnoses)"
            "\n- 'Kidney biopsy' does NOT imply 'nephropathy' (NO - diagnostic procedure without findings)"
            "\n- 'Heart murmur' does NOT imply 'congenital heart defect' (NO - too specific without evidence)"
            "\n\nRespond with ONLY 'YES' if the term directly implies a phenotype, or 'NO' if it doesn't."
            "\nYOUR RESPONSE MUST BE EXACTLY 'YES' OR 'NO' WITH NO ADDITIONAL TEXT."
        )

        # Extract phenotype with option for no clear phenotype
        self.extract_phenotype_system_message = (
            "You are a clinical expert specializing in phenotype identification for Human Phenotype Ontology (HPO) mapping. "
            "A previous analysis determined that a given term might imply a phenotype but is not a direct phenotype itself. "
            "Your task is to precisely identify what specific phenotype is implied by this term. "
            "\nProvide ONLY the name of the implied phenotype as it would appear in medical terminology. "
            "Be specific and concise. Do not include explanations or multiple options."
            "\n\nEXAMPLES:"
            "\n- 'Elevated white blood cell count of 15,000' implies 'leukocytosis'"
            "\n- 'E. coli growing in urine culture' implies 'bacteriuria'"
            "\n- 'Hemoglobin of 6.5 g/dL' implies 'anemia'"
            "\n\nIf you cannot identify a specific phenotype that is DIRECTLY implied with high confidence, "
            "respond with EXACTLY 'NO_CLEAR_PHENOTYPE_IMPLIED'."
        )

        # Phenotype validation - binary YES/NO
        self.phenotype_validation_system_message = (
            "You are a clinical expert specializing in phenotype identification for Human Phenotype Ontology (HPO) mapping. "
            "Your task is to determine if the proposed phenotype is a valid medical concept based on the provided HPO phenotype candidates."
            "\n\nAFTER REVIEWING THE CANDIDATES, respond with ONLY 'YES' if:"
            "\n- The phenotype EXACTLY or CLOSELY matches any HPO candidate, OR"
            "\n- The phenotype is a valid, recognized abnormal characteristic or condition in clinical medicine even if not in the candidates"
            "\n\nRespond with ONLY 'NO' if:"
            "\n- The proposed term is NOT a valid phenotype in clinical medicine"
            "\n- The term is too vague, general, or not recognized as a specific phenotype"
            "\n\nYOUR RESPONSE MUST BE EXACTLY 'YES' OR 'NO' WITH NO ADDITIONAL TEXT."
        )

        # Implication validation - strictly binary YES/NO
        self.implication_validation_system_message = (
            "You are a clinical expert specializing in phenotype identification for Human Phenotype Ontology (HPO) mapping. "
            "Your task is to validate whether an implied phenotype is reasonable given the original entity and its context. "
            "Be extremely critical and conservative in your assessment. Say YES only if there is an unambiguous, "
            "direct clinical connection between the entity and the proposed phenotype."
            "\nThe connection must be evident from the entity itself, not inferred from general knowledge."
            "\n\nEXAMPLES of VALID implications:"
            "\n- Entity: 'E. coli in urine culture' → Implied phenotype: 'bacteriuria' (VALID: specific finding)"
            "\n- Entity: 'Hemoglobin of 6.5 g/dL' → Implied phenotype: 'anemia' (VALID: specific abnormal value)"
            "\n\nEXAMPLES of INVALID implications:"
            "\n- Entity: 'white blood cell count' → Implied phenotype: 'leukocytosis' (INVALID: no value specified)"
            "\n- Entity: 'taking insulin daily' → Implied phenotype: 'diabetes mellitus' (INVALID: medication alone)"
            "\n- Entity: 'retina' → Implied phenotype: 'retinopathy' (INVALID: normal anatomy without abnormality)"
            "\n- Entity: 'renal tissue' → Implied phenotype: 'glomerulonephritis' (INVALID: too specific without evidence)"
            "\n\nRespond with ONLY 'YES' if the implication is valid, or 'NO' if it's not valid."
            "\nYOUR RESPONSE MUST BE EXACTLY 'YES' OR 'NO' WITH NO ADDITIONAL TEXT."
        )

        # Caches to avoid redundant API calls
        self.verification_cache = {}
        self.lab_test_detection_cache = {}
        self.lab_analysis_cache = {}
        self.implied_phenotype_cache = {}
        self.extracted_phenotype_cache = {}
        self.implication_validation_cache = {}
        self.phenotype_validation_cache = {}

    def initialize_lab_searcher(self):
        """Initialize the lab test searcher with embeddings file."""
        try:
            if not self.lab_embeddings_file:
                self._debug_print(
                    "No lab embeddings file provided, lab test tools disabled"
                )
                return

            self._debug_print(
                f"Initializing lab test searcher with {self.lab_embeddings_file}"
            )
            self.lab_searcher = ToolSearcher(
                model_type=self.embedding_manager.model_type,
                model_name=self.embedding_manager.model_name,
                device="cpu",  # Use CPU for tool searching to avoid GPU conflicts
                top_k=5,  # Get top 5 matches for lab tests
            )
            self.lab_searcher.load_embeddings(self.lab_embeddings_file)
            self._debug_print("Lab test searcher initialized successfully")
        except Exception as e:
            self._debug_print(f"Error initializing lab test searcher: {e}")
            self.lab_searcher = None

    def _debug_print(self, message: str, level: int = 0):
        """Print debug messages if debug mode is enabled."""
        if self.debug:
            indent = "  " * level
            print(f"{datetime.now().strftime('%H:%M:%S')} | {indent}{message}")

    def set_config(self, config):
        """Update the configuration."""
        self.config = config
        self.clear_caches()  # Clear caches when configuration changes

    def prepare_index(self, metadata: List[Dict]):
        """Prepare FAISS index from metadata for similarity search."""
        if self.index is None:
            self._debug_print("Preparing FAISS index for phenotype verification...")
            embeddings_array = self.embedding_manager.prepare_embeddings(metadata)
            self.index = self.embedding_manager.create_index(embeddings_array)
            self.embedded_documents = metadata
            self._debug_print(f"Index prepared with {len(metadata)} embedded documents")

    def clear_caches(self):
        """Clear all caches to prepare for a fresh evaluation run."""
        self.verification_cache = {}
        self.lab_test_detection_cache = {}
        self.lab_analysis_cache = {}
        self.implied_phenotype_cache = {}
        self.extracted_phenotype_cache = {}
        self.implication_validation_cache = {}
        self.phenotype_validation_cache = {}
        self._debug_print("All caches cleared")

    def _retrieve_similar_phenotypes(self, entity: str, k: int = 20) -> List[Dict]:
        """Retrieve similar phenotypes from the HPO ontology for context."""
        if self.index is None:
            raise ValueError("Index not prepared. Call prepare_index() first.")

        # Embed the query
        query_vector = self.embedding_manager.query_text(entity).reshape(1, -1)

        # Search for similar items
        distances, indices = self.embedding_manager.search(
            query_vector, self.index, k=min(800, len(self.embedded_documents))
        )

        # Extract unique metadata
        similar_phenotypes = []
        seen_metadata = set()

        for idx, distance in zip(indices[0], distances[0]):
            metadata = self.embedded_documents[idx]["unique_metadata"]
            metadata_str = json.dumps(metadata)

            if metadata_str not in seen_metadata:
                seen_metadata.add(metadata_str)
                similar_phenotypes.append(
                    {
                        "term": metadata.get("info", ""),
                        "hp_id": metadata.get("hp_id", ""),
                        "similarity_score": 1.0
                        / (1.0 + distance),  # Convert distance to similarity
                    }
                )

                if len(similar_phenotypes) >= k:
                    break

        return similar_phenotypes

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

    def verify_direct_phenotype(
        self, entity: str, context: Optional[str] = None
    ) -> Dict:
        """
        Verify if an entity is a direct phenotype through binary YES/NO matching against HPO candidates.

        Args:
            entity: Entity text to verify
            context: Original sentence containing the entity

        Returns:
            Dictionary with verification results
        """
        # Handle empty entities
        if not entity:
            return {"is_phenotype": False, "confidence": 1.0, "method": "empty_entity"}

        # Create a cache key - only include context if configured to use it
        cache_key = (
            f"direct::{entity}::{context if self.config.use_context_for_direct else ''}"
        )

        # Check cache first
        if cache_key in self.verification_cache:
            result = self.verification_cache[cache_key]
            self._debug_print(
                f"Cache hit for direct phenotype '{entity}': {result['is_phenotype']}",
                level=1,
            )
            return result

        self._debug_print(
            f"Verifying if '{entity}' is a direct phenotype via binary matching",
            level=1,
        )

        # Check for exact matches using fuzzy matching first (optimization)
        similar_phenotypes = self._retrieve_similar_phenotypes(
            entity, k=self.candidate_count
        )

        for phenotype in similar_phenotypes:
            normalized_term = self._normalize_text(phenotype["term"])
            normalized_entity = self._normalize_text(entity)

            # Check for exact match
            if normalized_term == normalized_entity:
                self._debug_print(
                    f"Exact match found: '{entity}' matches '{phenotype['term']}' ({phenotype['hp_id']})",
                    level=2,
                )
                result = {
                    "is_phenotype": True,
                    "confidence": 1.0,
                    "method": "exact_match",
                    "hp_id": phenotype["hp_id"],
                    "matched_term": phenotype["term"],
                }
                self.verification_cache[cache_key] = result
                return result

            # Check for high similarity match (over 90%)
            similarity = fuzz.ratio(normalized_term, normalized_entity)
            if similarity > 93:
                self._debug_print(
                    f"High similarity match ({similarity}%): '{entity}' matches '{phenotype['term']}' ({phenotype['hp_id']})",
                    level=2,
                )
                result = {
                    "is_phenotype": True,
                    "confidence": similarity / 100.0,
                    "method": "high_similarity_match",
                    "hp_id": phenotype["hp_id"],
                    "matched_term": phenotype["term"],
                }
                self.verification_cache[cache_key] = result
                return result

        # Format candidates for the LLM prompt
        candidate_items = []
        for i, phenotype in enumerate(similar_phenotypes, 1):
            candidate_items.append(f"{i}. '{phenotype['term']}' ({phenotype['hp_id']})")

        candidates_text = "\n".join(candidate_items)

        # Create context part if configured
        context_part = ""
        if context and self.config.use_context_for_direct:
            context_part = f"\nOriginal sentence context: '{context}'"

        # Create the binary YES/NO matching prompt
        prompt = (
            f"I need to determine if the entity '{entity}' is a valid human phenotype."
            f"\n\nHere are some HPO phenotype candidates for reference:"
            f"\n\n{candidates_text}\n"
            f"{context_part}\n\n"
            f"A valid phenotype must describe an abnormal characteristic or trait, not just a normal "
            f"anatomical structure, physiological process, laboratory test, or medication."
            f"\n\nBased on these candidates and criteria, is '{entity}' a valid human phenotype?"
            f"\nRespond with ONLY 'YES' or 'NO'."
        )

        # Query the LLM
        response = self.llm_client.query(
            prompt, self.direct_verification_system_message
        )

        # Parse the response - strictly look for "YES" or "NO"
        response_text = response.strip().upper()
        is_phenotype = "YES" in response_text and "NO" not in response_text

        # Create result based on binary response
        if is_phenotype:
            result = {
                "is_phenotype": True,
                "confidence": 0.9,
                "method": "llm_binary_verification",
            }
        else:
            result = {
                "is_phenotype": False,
                "confidence": 0.9,
                "method": "llm_binary_verification",
            }

        # Cache the result
        self.verification_cache[cache_key] = result

        self._debug_print(
            f"LLM binary verification: '{entity}' is{'' if is_phenotype else ' not'} a phenotype",
            level=2,
        )
        return result

    def contains_number(self, text: str) -> bool:
        """
        Check if text contains any numerical values.

        Args:
            text: Text to check for numbers

        Returns:
            Boolean indicating if numbers are present
        """
        # Match any digit sequence, including decimal points, etc.
        return bool(re.search(r"\d+(?:\.\d+)?", text))

    def detect_lab_test(self, entity: str, context: Optional[str] = None) -> Dict:
        """
        Detect if an entity is a lab test with a numerical value.
        First checks if there's a number present before querying LLM.

        Args:
            entity: Entity text to check
            context: Original sentence containing the entity

        Returns:
            Dictionary with detection results
        """
        # Handle empty entities
        if not entity:
            return {"is_lab_test": False, "confidence": 1.0, "method": "empty_entity"}

        # Create a cache key
        cache_key = f"lab_detect::{entity}::{context or ''}"

        # Check cache
        if cache_key in self.lab_test_detection_cache:
            result = self.lab_test_detection_cache[cache_key]
            self._debug_print(
                f"Cache hit for lab test detection '{entity}': {result['is_lab_test']}",
                level=1,
            )
            return result

        # Quick check - if no numbers in entity or context, it's not a lab test with value
        has_number_in_entity = self.contains_number(entity)
        has_number_in_context = context and self.contains_number(context)

        if not has_number_in_entity and not has_number_in_context:
            self._debug_print(
                f"Quick check: '{entity}' is not a lab test (no numbers present)",
                level=1,
            )
            result = {
                "is_lab_test": False,
                "confidence": 0.95,
                "method": "quick_check_no_numbers",
            }
            self.lab_test_detection_cache[cache_key] = result
            return result

        self._debug_print(
            f"Detecting if '{entity}' is a lab test with measurement (numbers present)",
            level=1,
        )

        # Create context part
        context_part = f"\nOriginal sentence context: '{context}'" if context else ""

        # Create the detection prompt
        prompt = (
            f"I need to determine if the entity '{entity}' contains information about a laboratory test with a measured value."
            f"{context_part}\n\n"
            f"Laboratory tests with measured values include examples like:"
            f"\n- 'Hemoglobin 8.5 g/dL'"
            f"\n- 'Elevated white blood cell count of 15,000/μL'"
            f"\n- 'Sodium 140 mEq/L'"
            f"\n\nDoes '{entity}' represent a lab test with a numerical value/result?"
            f"\nRespond with ONLY 'YES' or 'NO'."
        )

        # Query the LLM
        response = self.llm_client.query(prompt, self.lab_identification_system_message)

        # Parse the response - strictly look for "YES" or "NO"
        response_text = response.strip().upper()
        is_lab_test = "YES" in response_text and "NO" not in response_text

        # Create result
        result = {
            "is_lab_test": is_lab_test,
            "confidence": 0.9,
            "method": "llm_binary_detection",
        }

        # Cache the result
        self.lab_test_detection_cache[cache_key] = result

        self._debug_print(
            f"LLM detection: '{entity}' is{'' if is_lab_test else ' not'} a lab test with measurement",
            level=2,
        )
        return result

    def retrieve_lab_reference_ranges(self, lab_name: str) -> List[Dict]:
        """
        Retrieve reference ranges for a lab test using the lab searcher.

        Args:
            lab_name: Name of the lab test

        Returns:
            List of reference range information
        """
        if not self.lab_searcher:
            self._debug_print(
                "Lab searcher not initialized, skipping reference range retrieval",
                level=1,
            )
            return []

        try:
            # Search for lab test
            self._debug_print(
                f"Searching for lab test reference ranges: '{lab_name}'", level=2
            )
            search_results = self.lab_searcher.search(lab_name)

            # Process results
            if not search_results:
                self._debug_print("No reference ranges found", level=2)
                return []

            # Format results
            reference_ranges = []
            for result in search_results:
                try:
                    result_data = result.get("result", {})

                    # Skip if no result data
                    if not result_data:
                        continue

                    # Extract reference ranges
                    lab_id = result_data.get("lab_id", "N/A")
                    name = result_data.get("name", lab_name)
                    ranges = result_data.get("reference_ranges", [])
                    units = result_data.get("units", "")

                    formatted_result = {
                        "lab_id": lab_id,
                        "name": name,
                        "ranges": ranges,
                        "units": units,
                        "similarity": result.get("similarity", 0.0),
                    }

                    reference_ranges.append(formatted_result)

                except Exception as e:
                    self._debug_print(f"Error processing search result: {e}", level=2)
                    continue

            self._debug_print(
                f"Retrieved {len(reference_ranges)} reference range entries", level=2
            )
            return reference_ranges

        except Exception as e:
            self._debug_print(f"Error retrieving lab reference ranges: {e}", level=1)
            return []

    def analyze_lab_test(
        self,
        entity: str,
        context: Optional[str] = None,
        sample_data: Optional[Dict] = None,
    ) -> Dict:
        """
        Comprehensive analysis of a lab test entity, extracting and determining abnormality.

        Args:
            entity: Entity text containing lab test information
            context: Original sentence containing the entity
            sample_data: Optional additional data about the sample (e.g., demographics)

        Returns:
            Dictionary with comprehensive lab analysis results
        """
        # Handle empty entities
        if not entity:
            return {
                "lab_name": None,
                "value": None,
                "units": None,
                "is_abnormal": False,
                "abnormality": None,
                "direction": "unknown",
                "confidence": 0.0,
                "method": "empty_entity",
            }

        # Create a cache key
        sample_data_str = json.dumps(sample_data) if sample_data else ""
        cache_key = f"lab_analysis::{entity}::{context or ''}::{sample_data_str}"

        # Check cache
        if cache_key in self.lab_analysis_cache:
            result = self.lab_analysis_cache[cache_key]
            self._debug_print(f"Cache hit for lab analysis '{entity}'", level=1)
            return result

        self._debug_print(
            f"Analyzing lab test '{entity}' for extraction and abnormality", level=1
        )

        # Retrieve lab candidates
        lab_name_guess = (
            entity.split()[0] if entity else ""
        )  # Simple extraction of first word as lab name guess
        reference_ranges = self.retrieve_lab_reference_ranges(lab_name_guess)

        # Format reference range information
        reference_info = []
        if reference_ranges:
            for ref_range in reference_ranges[:3]:  # Limit to top 3
                name = ref_range.get("name", lab_name_guess)
                units = ref_range.get("units", "")
                ranges = ref_range.get("ranges", [])

                range_strings = []
                for range_item in ranges[:3]:  # Limit to top 3 ranges
                    age_group = range_item.get("age_group", "Adult")
                    male = range_item.get("male", "N/A")
                    female = range_item.get("female", "N/A")
                    range_strings.append(
                        f"  {age_group}: Male: {male}, Female: {female}"
                    )

                ref_str = f"Lab: {name} (Units: {units})\n"
                ref_str += "\n".join(range_strings)
                reference_info.append(ref_str)

        # Create reference part text
        reference_part = ""
        if reference_info:
            reference_part = "Reference Range Information:\n"
            reference_part += "\n\n".join(reference_info)
            reference_part += "\n\n"

        # Create context part
        context_part = f"Original Context: {context}\n\n" if context else ""

        # Create sample data part
        sample_part = ""
        if sample_data:
            sample_part = "Sample Data:\n"
            for key, value in sample_data.items():
                if value is not None:
                    sample_part += f"  {key}: {value}\n"
            sample_part += "\n"

        # Create the analysis prompt
        prompt = (
            f"Analyze this potential laboratory test entity: '{entity}'\n\n"
            f"{context_part}"
            f"{reference_part}"
            f"{sample_part}"
            f"Extract the lab test name, value (with units if available), and determine if the result is abnormal. "
            f"If abnormal, provide a clear medical description of the abnormality (e.g., 'elevated glucose', 'leukopenia')."
            f"\n\nProvide your response in this EXACT JSON format:"
            f"\n{{"
            f'\n  "lab_name": "[extracted lab test name]",'
            f'\n  "value": "[extracted value with units if available]",'
            f'\n  "units": "[extracted units if separable from value]",'
            f'\n  "is_abnormal": true/false,'
            f'\n  "abnormality": "[descriptive term for the abnormality, or \'normal\' if not abnormal]",'
            f'\n  "direction": "[high/low/normal]",'
            f'\n  "confidence": [0.0-1.0 value indicating your confidence]'
            f"\n}}"
            f"\n\nReturn ONLY the JSON with no additional text."
        )

        # Query the LLM
        response = self.llm_client.query(prompt, self.lab_analysis_system_message)

        # Parse the JSON response
        try:
            # Extract the JSON part from the response
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                extracted_json = json_match.group(0)
                analysis_info = json.loads(extracted_json)

                # Create result
                result = {
                    "lab_name": (
                        analysis_info.get("lab_name", "").lower().strip()
                        if analysis_info.get("lab_name")
                        else None
                    ),
                    "value": (
                        analysis_info.get("value", "").strip()
                        if analysis_info.get("value")
                        else None
                    ),
                    "units": (
                        analysis_info.get("units", "").strip()
                        if analysis_info.get("units")
                        else None
                    ),
                    "is_abnormal": analysis_info.get("is_abnormal", False),
                    "abnormality": analysis_info.get("abnormality", "normal").strip(),
                    "direction": analysis_info.get("direction", "normal")
                    .lower()
                    .strip(),
                    "confidence": analysis_info.get("confidence", 0.5),
                    "method": "llm_analysis",
                }

                # Cache the result
                self.lab_analysis_cache[cache_key] = result

                if result["is_abnormal"]:
                    self._debug_print(
                        f"Lab test analysis: '{entity}' is abnormal: {result['abnormality']} ({result['direction']})",
                        level=2,
                    )
                else:
                    self._debug_print(
                        f"Lab test analysis: '{entity}' is normal", level=2
                    )

                return result
            else:
                # Failed to find JSON in response
                self._debug_print(
                    f"Failed to extract JSON from response: {response}", level=2
                )
                result = {
                    "lab_name": None,
                    "value": None,
                    "units": None,
                    "is_abnormal": False,
                    "abnormality": None,
                    "direction": "unknown",
                    "confidence": 0.0,
                    "method": "extraction_failed",
                }
                self.lab_analysis_cache[cache_key] = result
                return result

        except Exception as e:
            # Failed to parse JSON
            self._debug_print(
                f"Failed to parse JSON from response ({e}): {response}", level=2
            )
            result = {
                "lab_name": None,
                "value": None,
                "units": None,
                "is_abnormal": False,
                "abnormality": None,
                "direction": "unknown",
                "confidence": 0.0,
                "method": "json_parse_failed",
            }
            self.lab_analysis_cache[cache_key] = result
            return result

    def check_implies_phenotype(
        self, entity: str, context: Optional[str] = None
    ) -> Dict:
        """
        Check if an entity implies a phenotype with configurable retrieval and context usage.

        Args:
            entity: Entity text to check
            context: Original sentence containing the entity

        Returns:
            Dictionary with results
        """
        # Handle empty entities
        if not entity:
            return {
                "implies_phenotype": False,
                "confidence": 1.0,
                "method": "empty_entity",
            }

        # Create a cache key - only include context if configured to use it
        cache_key = f"implies::{entity}::{context if self.config.use_context_for_implies else ''}"

        # Check cache
        if cache_key in self.implied_phenotype_cache:
            result = self.implied_phenotype_cache[cache_key]
            self._debug_print(
                f"Cache hit for implied phenotype check '{entity}': {result['implies_phenotype']}",
                level=1,
            )
            return result

        self._debug_print(f"Checking if '{entity}' implies a phenotype", level=1)

        # Retrieve similar phenotypes for context if configured
        similar_phenotypes = []
        if self.config.use_retrieval_for_implies:
            similar_phenotypes = self._retrieve_similar_phenotypes(entity)

        # Format context items if using retrieval
        context_items = []
        if self.config.use_retrieval_for_implies:
            for phenotype in similar_phenotypes[:10]:  # Use top 10 for context
                context_items.append(f"- {phenotype['term']} ({phenotype['hp_id']})")

        context_text = "\n".join(context_items)

        # Create prompt for the LLM
        context_part = ""
        if context and self.config.use_context_for_implies:
            context_part = f"Original sentence context: '{context}'\n\n"

        retrieval_part = ""
        if self.config.use_retrieval_for_implies and context_items:
            retrieval_part = (
                f"Here are some phenotype terms from the Human Phenotype Ontology for context:\n\n"
                f"{context_text}\n\n"
            )

        prompt = (
            f"I need to determine if '{entity}' DIRECTLY AND UNAMBIGUOUSLY implies a specific phenotype. "
            f"Be extremely conservative - only say YES if the implication is clear and specific."
            f"\n{context_part}"
            f"{retrieval_part}"
            f"Laboratory values, medications, or procedures DO NOT imply phenotypes unless there is explicit abnormality mentioned."
            f"\nIf you're uncertain or the implication requires multiple assumptions, say NO."
            f"\n\nDoes '{entity}' directly imply a specific phenotype? "
            f"Respond with ONLY 'YES' if it directly implies a phenotype or 'NO' if it doesn't."
        )

        # Query the LLM
        response = self.llm_client.query(prompt, self.implied_phenotype_system_message)

        # Parse the response - strictly look for "YES" or "NO"
        response_text = response.strip().upper()
        implies_phenotype = "YES" in response_text and "NO" not in response_text

        # Create result
        result = {
            "implies_phenotype": implies_phenotype,
            "confidence": (
                0.8 if implies_phenotype else 0.9
            ),  # Higher confidence for "no" to be conservative
            "method": "llm_binary_verification",
        }

        # Cache the result
        self.implied_phenotype_cache[cache_key] = result

        self._debug_print(
            f"LLM binary verification: '{entity}' does{'' if implies_phenotype else ' not'} imply a phenotype",
            level=2,
        )
        return result

    def extract_implied_phenotype(
        self, entity: str, context: Optional[str] = None
    ) -> Dict:
        """
        Extract the specific phenotype implied by an entity with configurable retrieval and context usage.

        Args:
            entity: Entity text that implies a phenotype
            context: Original sentence containing the entity

        Returns:
            Dictionary with results
        """
        # Handle empty entities
        if not entity:
            return {
                "implied_phenotype": None,
                "confidence": 0.0,
                "method": "empty_entity",
            }

        # Create a cache key - only include context if configured to use it
        cache_key = f"extract::{entity}::{context if self.config.use_context_for_extract else ''}"

        # Check cache
        if cache_key in self.extracted_phenotype_cache:
            result = self.extracted_phenotype_cache[cache_key]
            self._debug_print(
                f"Cache hit for extracting implied phenotype from '{entity}': {result.get('implied_phenotype')}",
                level=1,
            )
            return result

        self._debug_print(f"Extracting implied phenotype from '{entity}'", level=1)

        # Retrieve similar phenotypes for context if configured
        similar_phenotypes = []
        if self.config.use_retrieval_for_extract:
            similar_phenotypes = self._retrieve_similar_phenotypes(entity)

        # Format context items if using retrieval
        context_items = []
        if self.config.use_retrieval_for_extract:
            for phenotype in similar_phenotypes[:10]:  # Use top 10 for context
                context_items.append(f"- {phenotype['term']} ({phenotype['hp_id']})")

        context_text = "\n".join(context_items)

        # Create prompt for the LLM
        context_part = ""
        if context and self.config.use_context_for_extract:
            context_part = f"Original sentence context: '{context}'\n\n"

        retrieval_part = ""
        if self.config.use_retrieval_for_extract and context_items:
            retrieval_part = (
                f"Here are some phenotype terms for context:\n\n" f"{context_text}\n\n"
            )

        prompt = (
            f"The term '{entity}' might imply a phenotype. "
            f"{context_part}"
            f"{retrieval_part}"
            f"What specific phenotype is directly implied by '{entity}'? "
            f"For example, 'hemoglobin of 8 g/dL' implies 'anemia'."
            f"\n\nIf you cannot identify a specific phenotype that is DIRECTLY implied with high confidence, "
            f"respond with EXACTLY 'NO_CLEAR_PHENOTYPE_IMPLIED'."
            f"\n\nProvide ONLY the name of the implied phenotype, without any explanation, "
            f"or 'NO_CLEAR_PHENOTYPE_IMPLIED' if none is clear."
        )

        # Query the LLM
        response = self.llm_client.query(prompt, self.extract_phenotype_system_message)

        # Clean the response
        implied_phenotype = response.strip()
        implied_phenotype = re.sub(r"[.,;:]$", "", implied_phenotype)

        # Check for the special "no clear phenotype" response
        if "NO_CLEAR_PHENOTYPE_IMPLIED" in implied_phenotype.upper():
            result = {
                "implied_phenotype": None,
                "confidence": 0.9,
                "method": "llm_extraction_no_clear_phenotype",
            }
        else:
            result = {
                "implied_phenotype": implied_phenotype,
                "confidence": 0.7,  # Lower confidence compared to V1 to be more conservative
                "method": "llm_extraction",
            }

        # Cache the result
        self.extracted_phenotype_cache[cache_key] = result

        if result["implied_phenotype"] is None:
            self._debug_print(
                f"LLM could not extract a clear implied phenotype from '{entity}'",
                level=2,
            )
        else:
            self._debug_print(
                f"LLM extracted implied phenotype from '{entity}': '{implied_phenotype}'",
                level=2,
            )

        return result

    def validate_phenotype_exists(self, phenotype: str) -> Dict:
        """
        Validate if a phenotype exists by binary YES/NO matching against HPO candidates.

        Args:
            phenotype: The phenotype to validate

        Returns:
            Dictionary with validation results
        """
        # Skip empty inputs
        if not phenotype:
            return {"is_valid": False, "confidence": 1.0, "method": "empty_input"}

        # Create a cache key
        cache_key = f"validate_phenotype::{phenotype}"

        # Check cache
        if cache_key in self.phenotype_validation_cache:
            result = self.phenotype_validation_cache[cache_key]
            self._debug_print(
                f"Cache hit for phenotype validation '{phenotype}': {result['is_valid']}",
                level=1,
            )
            return result

        self._debug_print(
            f"Validating phenotype '{phenotype}' via binary matching", level=1
        )

        # Check for exact matches using fuzzy matching first (optimization)
        similar_phenotypes = self._retrieve_similar_phenotypes(
            phenotype, k=self.candidate_count
        )

        for pheno in similar_phenotypes:
            normalized_term = self._normalize_text(pheno["term"])
            normalized_phenotype = self._normalize_text(phenotype)

            # Check for exact match
            if normalized_term == normalized_phenotype:
                self._debug_print(
                    f"Exact match found: '{phenotype}' matches '{pheno['term']}' ({pheno['hp_id']})",
                    level=2,
                )
                result = {
                    "is_valid": True,
                    "confidence": 1.0,
                    "method": "exact_match",
                    "hp_id": pheno["hp_id"],
                    "matched_term": pheno["term"],
                }
                self.phenotype_validation_cache[cache_key] = result
                return result

            # Check for high similarity match (over 90%)
            similarity = fuzz.ratio(normalized_term, normalized_phenotype)
            if similarity > 93:
                self._debug_print(
                    f"High similarity match ({similarity}%): '{phenotype}' matches '{pheno['term']}' ({pheno['hp_id']})",
                    level=2,
                )
                result = {
                    "is_valid": True,
                    "confidence": similarity / 100.0,
                    "method": "high_similarity_match",
                    "hp_id": pheno["hp_id"],
                    "matched_term": pheno["term"],
                }
                self.phenotype_validation_cache[cache_key] = result
                return result

        # Format candidates for the LLM prompt
        candidate_items = []
        for i, pheno in enumerate(similar_phenotypes, 1):
            candidate_items.append(f"{i}. '{pheno['term']}' ({pheno['hp_id']})")

        candidates_text = "\n".join(candidate_items)

        # Create the binary YES/NO matching prompt
        prompt = (
            f"I need to determine if the phenotype '{phenotype}' is a valid medical concept."
            f"\n\nHere are some HPO phenotype candidates for reference:"
            f"\n\n{candidates_text}\n\n"
            f"Is '{phenotype}' a valid phenotype in clinical medicine? Consider both potential matches "
            f"in the candidates and your general knowledge of medical phenotypes."
            f"\nRespond with ONLY 'YES' or 'NO'."
        )

        # Query the LLM
        response = self.llm_client.query(
            prompt, self.phenotype_validation_system_message
        )

        # Parse the response - strictly look for "YES" or "NO"
        response_text = response.strip().upper()
        is_valid = "YES" in response_text and "NO" not in response_text

        # Create result based on binary response
        result = {
            "is_valid": is_valid,
            "confidence": 0.9,
            "method": "llm_binary_validation",
        }

        # Cache the result
        self.phenotype_validation_cache[cache_key] = result

        self._debug_print(
            f"LLM binary validation: '{phenotype}' is{'' if is_valid else ' not'} a valid phenotype",
            level=2,
        )
        return result

    def validate_implication(
        self, entity: str, implied_phenotype: str, context: Optional[str] = None
    ) -> Dict:
        """
        Validate if the implication from entity to phenotype is reasonable with strictly binary YES/NO response.

        Args:
            entity: Original entity text
            implied_phenotype: Extracted implied phenotype
            context: Original sentence containing the entity

        Returns:
            Dictionary with validation results
        """
        # Skip empty inputs
        if not entity or not implied_phenotype:
            return {"is_valid": False, "confidence": 1.0, "method": "empty_input"}

        # Create a cache key - only include context if configured to use it
        cache_key = f"validate_implication::{entity}::{implied_phenotype}::{context if self.config.use_context_for_implication else ''}"

        # Check cache
        if cache_key in self.implication_validation_cache:
            result = self.implication_validation_cache[cache_key]
            self._debug_print(
                f"Cache hit for implication validation '{entity}' → '{implied_phenotype}': {result['is_valid']}",
                level=1,
            )
            return result

        self._debug_print(
            f"Validating implication from '{entity}' to '{implied_phenotype}' with binary response",
            level=1,
        )

        # Create prompt for the LLM
        context_part = ""
        if context and self.config.use_context_for_implication:
            context_part = f"Original sentence context: '{context}'\n\n"

        prompt = (
            f"I need to validate whether the following implication is reasonable:\n\n"
            f"Original entity: '{entity}'\n"
            f"Implied phenotype: '{implied_phenotype}'\n\n"
            f"{context_part}"
            f"Be extremely critical and conservative. Say YES only if there is an unambiguous, "
            f"direct clinical connection between the entity and the proposed phenotype."
            f"\nThe connection must be evident from the entity itself, not inferred from general knowledge."
            f"\n\nIs this a valid and reasonable implication? "
            f"Respond with ONLY 'YES' or 'NO'."
        )

        # Query the LLM
        response = self.llm_client.query(
            prompt, self.implication_validation_system_message
        )

        # Parse the response - strictly look for "YES" or "NO"
        response_text = response.strip().upper()
        is_valid = "YES" in response_text and "NO" not in response_text

        # Create result
        result = {
            "is_valid": is_valid,
            "confidence": 0.9,
            "method": "llm_binary_validation",
        }

        # Cache the result
        self.implication_validation_cache[cache_key] = result

        self._debug_print(
            f"LLM binary validation: Implication from '{entity}' to '{implied_phenotype}' is{'' if is_valid else ' not'} valid",
            level=2,
        )
        return result

    def process_entity(
        self,
        entity: str,
        context: Optional[str] = None,
        sample_data: Optional[Dict] = None,
    ) -> Dict:
        """
        Process an entity through the enhanced multi-stage pipeline with simplified lab test analysis.

        Args:
            entity: Entity text to process
            context: Original sentence containing the entity
            sample_data: Optional dictionary with sample-specific data (demographics, etc.)

        Returns:
            Dictionary with processing results
        """
        # Handle empty entities
        if not entity:
            return {
                "status": "not_phenotype",
                "phenotype": None,
                "original_entity": entity,
                "confidence": 1.0,
                "method": "empty_entity",
            }

        self._debug_print(f"Processing entity: '{entity}'", level=0)

        # Clean and preprocess the entity
        cleaned_entity = self.preprocess_entity(entity)
        if not cleaned_entity:
            return {
                "status": "not_phenotype",
                "phenotype": None,
                "original_entity": entity,
                "confidence": 1.0,
                "method": "empty_after_preprocessing",
            }

        # STAGE 1: Check if it's a direct phenotype using binary matching
        direct_result = self.verify_direct_phenotype(cleaned_entity, context)

        # If it's a direct phenotype, return it with details
        if direct_result.get("is_phenotype", False):
            self._debug_print(f"'{entity}' is a direct phenotype", level=1)
            result = {
                "status": "direct_phenotype",
                "phenotype": direct_result.get("matched_term", cleaned_entity),
                "original_entity": entity,
                "confidence": direct_result["confidence"],
                "method": direct_result["method"],
            }

            if "hp_id" in direct_result:
                result["hp_id"] = direct_result["hp_id"]

            return result

        # STAGE 2: Check if entity or context contains numbers (quick check for lab tests)
        has_numbers = self.contains_number(cleaned_entity) or (
            context and self.contains_number(context)
        )

        # If numbers present, check if it's a lab test
        if has_numbers:
            lab_detection_result = self.detect_lab_test(cleaned_entity, context)

            if lab_detection_result.get("is_lab_test", False):
                self._debug_print(
                    f"'{entity}' is a lab test, performing analysis", level=1
                )

                # Analyze lab test for abnormality
                lab_analysis_result = self.analyze_lab_test(
                    cleaned_entity, context, sample_data
                )

                # If it's an abnormal lab test, use the abnormality as a phenotype
                if lab_analysis_result.get(
                    "is_abnormal", False
                ) and lab_analysis_result.get("abnormality"):
                    abnormality = lab_analysis_result["abnormality"]

                    # Skip if abnormality is just "normal"
                    if abnormality.lower() == "normal":
                        self._debug_print(
                            f"Lab test '{entity}' is normal, not a phenotype", level=1
                        )
                        return {
                            "status": "not_phenotype",
                            "phenotype": None,
                            "original_entity": entity,
                            "confidence": lab_analysis_result["confidence"],
                            "method": "normal_lab_value",
                            "lab_info": {
                                "lab_name": lab_analysis_result.get("lab_name"),
                                "value": lab_analysis_result.get("value"),
                                "units": lab_analysis_result.get("units"),
                            },
                        }

                    # Validate the abnormality as a phenotype
                    phenotype_validation_result = self.validate_phenotype_exists(
                        abnormality
                    )

                    if phenotype_validation_result.get("is_valid", False):
                        self._debug_print(
                            f"Lab abnormality '{abnormality}' is a valid phenotype",
                            level=1,
                        )
                        result = {
                            "status": "implied_phenotype",
                            "phenotype": phenotype_validation_result.get(
                                "matched_term", abnormality
                            ),
                            "original_entity": entity,
                            "confidence": min(
                                lab_analysis_result["confidence"],
                                phenotype_validation_result.get("confidence", 0.7),
                            ),
                            "method": "lab_abnormality",
                            "lab_info": {
                                "lab_name": lab_analysis_result.get("lab_name"),
                                "value": lab_analysis_result.get("value"),
                                "units": lab_analysis_result.get("units"),
                                "direction": lab_analysis_result.get("direction"),
                            },
                        }

                        # Include HP ID if available
                        if "hp_id" in phenotype_validation_result:
                            result["hp_id"] = phenotype_validation_result["hp_id"]

                        return result

                    # Fall through if abnormality is not a valid phenotype
                    self._debug_print(
                        f"Lab abnormality '{abnormality}' is not a valid phenotype",
                        level=1,
                    )

                # If lab test is normal or abnormality is not a valid phenotype, continue to implied phenotype check
                self._debug_print(
                    f"Lab test '{entity}' analysis did not yield a valid phenotype",
                    level=1,
                )

        # STAGE 3: Check if it implies a phenotype (for non-lab tests or lab tests that didn't yield valid phenotypes)
        implies_result = self.check_implies_phenotype(cleaned_entity, context)

        if not implies_result.get("implies_phenotype", False):
            self._debug_print(
                f"'{entity}' is not a phenotype and doesn't imply one", level=1
            )
            return {
                "status": "not_phenotype",
                "phenotype": None,
                "original_entity": entity,
                "confidence": implies_result["confidence"],
                "method": implies_result.get("method", "llm_verification"),
            }

        # STAGE 4: Extract the implied phenotype
        extract_result = self.extract_implied_phenotype(cleaned_entity, context)
        implied_phenotype = extract_result.get("implied_phenotype")

        # If we couldn't extract an implied phenotype, not a phenotype
        if not implied_phenotype:
            self._debug_print(
                f"No clear phenotype could be extracted from '{entity}'", level=1
            )
            return {
                "status": "not_phenotype",
                "phenotype": None,
                "original_entity": entity,
                "confidence": extract_result.get("confidence", 0.7),
                "method": extract_result.get("method", "no_implied_phenotype_found"),
            }

        # STAGE 5: Validate if the phenotype exists via binary matching
        phenotype_validation_result = self.validate_phenotype_exists(implied_phenotype)

        if not phenotype_validation_result.get("is_valid", False):
            self._debug_print(
                f"Implied phenotype '{implied_phenotype}' from '{entity}' is not valid",
                level=1,
            )
            return {
                "status": "not_phenotype",
                "phenotype": None,
                "original_entity": entity,
                "confidence": phenotype_validation_result["confidence"],
                "method": "invalid_phenotype",
            }

        # If there's a matching HPO term, use it
        if "hp_id" in phenotype_validation_result:
            implied_phenotype = phenotype_validation_result.get(
                "matched_term", implied_phenotype
            )
            hp_id = phenotype_validation_result["hp_id"]
        else:
            hp_id = None

        # STAGE 6: Validate if the implication is reasonable with binary response
        implication_validation_result = self.validate_implication(
            cleaned_entity, implied_phenotype, context
        )

        if not implication_validation_result.get("is_valid", False):
            self._debug_print(
                f"Implication from '{entity}' to '{implied_phenotype}' is not valid",
                level=1,
            )
            return {
                "status": "not_phenotype",
                "phenotype": None,
                "original_entity": entity,
                "confidence": implication_validation_result["confidence"],
                "method": "invalid_implication",
            }

        # If implication and phenotype are both valid, return the implied phenotype
        self._debug_print(
            f"'{entity}' implies valid phenotype '{implied_phenotype}'", level=1
        )
        result = {
            "status": "implied_phenotype",
            "phenotype": implied_phenotype,
            "original_entity": entity,
            "confidence": min(
                extract_result["confidence"], phenotype_validation_result["confidence"]
            ),
            "method": "multi_stage_pipeline",
        }

        # Include HP ID if available
        if hp_id:
            result["hp_id"] = hp_id

        return result

    def batch_process(
        self, entities_with_context: List[Dict], sample_data: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Process a batch of entities with their contexts through the enhanced multi-stage pipeline.

        Args:
            entities_with_context: List of dicts with 'entity' and 'context' keys
            sample_data: Optional dictionary with sample-specific data (demographics, etc.)

        Returns:
            List of dicts with processing results (phenotypes only)
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

            result = self.process_entity(entity, context, sample_data)

            # Only include entities that are phenotypes (direct or implied)
            if result["status"] in ["direct_phenotype", "implied_phenotype"]:
                # Add original context
                result["context"] = context
                results.append(result)

        self._debug_print(f"Identified {len(results)} phenotypes (direct or implied)")
        return results


class MultiStageHPOVerifierV3(MultiStageHPOVerifierV4):
    """
    Enhanced multi-stage HPO verifier without lab test analysis.

    This version implements a streamlined workflow:
    1. Direct phenotype verification with retrieved candidates
    2. Implied phenotype determination

    Lab test analysis is disabled in this version.
    """

    def __init__(self, embedding_manager, llm_client, config=None, debug=False):
        """Initialize without lab embeddings capability."""
        # Call parent initializer without lab_embeddings_file
        super().__init__(
            embedding_manager, llm_client, config, debug, lab_embeddings_file=None
        )
        # Ensure lab searcher is None
        self.lab_searcher = None

    def initialize_lab_searcher(self):
        """Override to disable lab searcher functionality."""
        self._debug_print("Lab test functionality disabled in V3")
        self.lab_searcher = None

    def detect_lab_test(self, entity: str, context: Optional[str] = None) -> Dict:
        """Override to always return not a lab test."""
        return {"is_lab_test": False, "confidence": 1.0, "method": "v3_disabled"}

    def retrieve_lab_reference_ranges(self, lab_name: str) -> List[Dict]:
        """Override to return empty list."""
        return []

    def analyze_lab_test(
        self,
        entity: str,
        context: Optional[str] = None,
        sample_data: Optional[Dict] = None,
    ) -> Dict:
        """Override to return default values."""
        return {
            "lab_name": None,
            "value": None,
            "units": None,
            "is_abnormal": False,
            "abnormality": None,
            "direction": "unknown",
            "confidence": 0.0,
            "method": "v3_disabled",
        }

    def process_entity(
        self,
        entity: str,
        context: Optional[str] = None,
        sample_data: Optional[Dict] = None,
    ) -> Dict:
        """
        Process an entity with V3 pipeline (no lab tests).

        Args:
            entity: Entity text to process
            context: Original sentence containing the entity
            sample_data: Optional dictionary with sample-specific data

        Returns:
            Dictionary with processing results
        """
        # Handle empty entities
        if not entity:
            return {
                "status": "not_phenotype",
                "phenotype": None,
                "original_entity": entity,
                "confidence": 1.0,
                "method": "empty_entity",
            }

        self._debug_print(f"Processing entity: '{entity}'", level=0)

        # Clean and preprocess the entity
        cleaned_entity = self.preprocess_entity(entity)
        if not cleaned_entity:
            return {
                "status": "not_phenotype",
                "phenotype": None,
                "original_entity": entity,
                "confidence": 1.0,
                "method": "empty_after_preprocessing",
            }

        # STAGE 1: Check if it's a direct phenotype using binary matching
        direct_result = self.verify_direct_phenotype(cleaned_entity, context)

        # If it's a direct phenotype, return it with details
        if direct_result.get("is_phenotype", False):
            self._debug_print(f"'{entity}' is a direct phenotype", level=1)
            result = {
                "status": "direct_phenotype",
                "phenotype": direct_result.get("matched_term", cleaned_entity),
                "original_entity": entity,
                "confidence": direct_result["confidence"],
                "method": direct_result["method"],
            }

            if "hp_id" in direct_result:
                result["hp_id"] = direct_result["hp_id"]

            return result

        # STAGE 2: Check if it implies a phenotype (Skip lab test detection entirely)
        implies_result = self.check_implies_phenotype(cleaned_entity, context)

        if not implies_result.get("implies_phenotype", False):
            self._debug_print(
                f"'{entity}' is not a phenotype and doesn't imply one", level=1
            )
            return {
                "status": "not_phenotype",
                "phenotype": None,
                "original_entity": entity,
                "confidence": implies_result["confidence"],
                "method": implies_result.get("method", "llm_verification"),
            }

        # STAGE 3: Extract the implied phenotype
        extract_result = self.extract_implied_phenotype(cleaned_entity, context)
        implied_phenotype = extract_result.get("implied_phenotype")

        # If we couldn't extract an implied phenotype, not a phenotype
        if not implied_phenotype:
            self._debug_print(
                f"No clear phenotype could be extracted from '{entity}'", level=1
            )
            return {
                "status": "not_phenotype",
                "phenotype": None,
                "original_entity": entity,
                "confidence": extract_result.get("confidence", 0.7),
                "method": extract_result.get("method", "no_implied_phenotype_found"),
            }

        # STAGE 4: Validate if the phenotype exists via binary matching
        phenotype_validation_result = self.validate_phenotype_exists(implied_phenotype)

        if not phenotype_validation_result.get("is_valid", False):
            self._debug_print(
                f"Implied phenotype '{implied_phenotype}' from '{entity}' is not valid",
                level=1,
            )
            return {
                "status": "not_phenotype",
                "phenotype": None,
                "original_entity": entity,
                "confidence": phenotype_validation_result["confidence"],
                "method": "invalid_phenotype",
            }

        # If there's a matching HPO term, use it
        if "hp_id" in phenotype_validation_result:
            implied_phenotype = phenotype_validation_result.get(
                "matched_term", implied_phenotype
            )
            hp_id = phenotype_validation_result["hp_id"]
        else:
            hp_id = None

        # STAGE 5: Validate if the implication is reasonable with binary response
        implication_validation_result = self.validate_implication(
            cleaned_entity, implied_phenotype, context
        )

        if not implication_validation_result.get("is_valid", False):
            self._debug_print(
                f"Implication from '{entity}' to '{implied_phenotype}' is not valid",
                level=1,
            )
            return {
                "status": "not_phenotype",
                "phenotype": None,
                "original_entity": entity,
                "confidence": implication_validation_result["confidence"],
                "method": "invalid_implication",
            }

        # If implication and phenotype are both valid, return the implied phenotype
        self._debug_print(
            f"'{entity}' implies valid phenotype '{implied_phenotype}'", level=1
        )
        result = {
            "status": "implied_phenotype",
            "phenotype": implied_phenotype,
            "original_entity": entity,
            "confidence": min(
                extract_result["confidence"], phenotype_validation_result["confidence"]
            ),
            "method": "multi_stage_pipeline",
        }

        # Include HP ID if available
        if hp_id:
            result["hp_id"] = hp_id

        return result


class MultiStageHPOVerifierV2(MultiStageHPOVerifierV4):
    """
    Enhanced multi-stage HPO verifier with more verbose prompts.

    This version uses more detailed examples in its prompts compared to V3 and V4,
    but otherwise has the same functionality as V4.
    """

    def __init__(
        self,
        embedding_manager,
        llm_client,
        config=None,
        debug=False,
        lab_embeddings_file=None,
    ):
        """Initialize with more verbose system prompts."""
        # Call parent initializer first
        super().__init__(
            embedding_manager, llm_client, config, debug, lab_embeddings_file
        )

        # Override with more verbose prompts
        # Direct verification with binary matching - more examples
        self.direct_verification_system_message = (
            "You are a clinical expert specializing in phenotype identification for Human Phenotype Ontology (HPO) mapping. "
            "Your task is to determine if the given entity represents a valid human phenotype based on the provided HPO candidates."
            "\n\nA valid phenotype must describe an abnormal characteristic or trait, not just a normal "
            "anatomical structure, physiological process, laboratory test, or medication."
            "\n\nEXAMPLES OF VALID PHENOTYPES:"
            "\n- 'Seizure' - an abnormal electrical activity in the brain"
            "\n- 'Polydactyly' - having extra fingers or toes"
            "\n- 'Microcephaly' - abnormally small head"
            "\n- 'Hypercholesterolemia' - high cholesterol level"
            "\n\nEXAMPLES OF INVALID PHENOTYPES:"
            "\n- 'Brain' - normal anatomical structure"
            "\n- 'Blood test' - diagnostic procedure"
            "\n- 'Acetaminophen' - medication"
            "\n- 'Sleep' - normal physiological process"
            "\n\nAFTER REVIEWING THE CANDIDATES, respond with ONLY 'YES' if:"
            "\n- The entity EXACTLY or CLOSELY matches any HPO candidate, OR"
            "\n- The entity clearly describes a phenotype even if not in the candidates"
            "\n\nRespond with ONLY 'NO' if:"
            "\n- The entity is NOT a phenotype (e.g., normal anatomy, medication, lab test without value)"
            "\n- The entity does not represent any abnormal human trait or characteristic"
            "\n\nYOUR RESPONSE MUST BE EXACTLY 'YES' OR 'NO' WITH NO ADDITIONAL TEXT."
        )

        # Lab test identification - more examples
        self.lab_identification_system_message = (
            "You are a clinical expert analyzing laboratory test information in clinical notes. "
            "Your task is to determine if the given entity contains information about a laboratory test with a measured value."
            "\n\nEXAMPLES OF LAB TESTS WITH VALUES:"
            "\n- 'Hemoglobin 8.5 g/dL' - includes test name and numeric value"
            "\n- 'Elevated white blood cell count of 15,000/μL' - includes test, descriptor, and value"
            "\n- 'Sodium 140 mEq/L' - includes test name, value, and units"
            "\n- 'Glucose level: 180 mg/dL' - includes test name, value, and units"
            "\n\nEXAMPLES THAT ARE NOT LAB TESTS WITH VALUES:"
            "\n- 'Complete blood count' - test name only, no value"
            "\n- 'Ordered metabolic panel' - procedure ordered but no results"
            "\n- 'Normal electrolytes' - qualitative description without specific values"
            "\n- 'Kidney function' - physiological process, not a specific test with value"
            "\n\nRespond with ONLY 'YES' if:"
            "\n- The entity contains a lab test name AND a numerical value/result"
            "\n- The entity clearly refers to a laboratory measurement with a value"
            "\n\nRespond with ONLY 'NO' if:"
            "\n- The entity only mentions a lab test without a specific value"
            "\n- The entity is not related to laboratory measurements"
            "\n\nYOUR RESPONSE MUST BE EXACTLY 'YES' OR 'NO' WITH NO ADDITIONAL TEXT."
        )

        # Implied phenotype check - more examples
        self.implied_phenotype_system_message = (
            "You are a clinical expert specializing in phenotype identification for Human Phenotype Ontology (HPO) mapping. "
            "Determine if the given term DIRECTLY AND UNAMBIGUOUSLY implies a specific phenotype, even though it's not a direct phenotype itself. "
            "Be extremely conservative - only say YES if the implication is clear and specific."
            "\nLaboratory values, medications, or procedures DO NOT imply phenotypes unless there is explicit abnormality mentioned."
            "\nIf you're uncertain or the implication requires multiple assumptions, say NO."
            "\n\nEXAMPLES OF VALID IMPLICATIONS:"
            "\n- 'Elevated white blood cell count of 15,000' implies 'leukocytosis' (YES - explicit abnormality)"
            "\n- 'E. coli growing in urine culture' implies 'bacteriuria' (YES - specific finding)"
            "\n- 'Hemoglobin of 6.5 g/dL' implies 'anemia' (YES - clearly below normal range)"
            "\n- 'Staphylococcus aureus in blood culture' implies 'bacteremia' (YES - specific pathogen in blood)"
            "\n- 'Oxygen saturation 82%' implies 'hypoxemia' (YES - clearly abnormal oxygenation)"
            "\n\nEXAMPLES OF INVALID IMPLICATIONS:"
            "\n- 'White blood cell count' does NOT imply 'leukocytosis' (NO - no value specified)"
            "\n- 'Taking insulin' does NOT imply 'diabetes mellitus' (NO - medications alone don't imply diagnoses)"
            "\n- 'Kidney biopsy' does NOT imply 'nephropathy' (NO - diagnostic procedure without findings)"
            "\n- 'Heart murmur' does NOT imply 'congenital heart defect' (NO - too specific without evidence)"
            "\n- 'Respiratory rate' does NOT imply 'tachypnea' (NO - no value specified)"
            "\n- 'Ordered MRI' does NOT imply 'brain abnormality' (NO - procedure without results)"
            "\n\nRespond with ONLY 'YES' if the term directly implies a phenotype, or 'NO' if it doesn't."
            "\nYOUR RESPONSE MUST BE EXACTLY 'YES' OR 'NO' WITH NO ADDITIONAL TEXT."
        )

        # Extract phenotype with more examples
        self.extract_phenotype_system_message = (
            "You are a clinical expert specializing in phenotype identification for Human Phenotype Ontology (HPO) mapping. "
            "A previous analysis determined that a given term might imply a phenotype but is not a direct phenotype itself. "
            "Your task is to precisely identify what specific phenotype is implied by this term. "
            "\nProvide ONLY the name of the implied phenotype as it would appear in medical terminology. "
            "Be specific and concise. Do not include explanations or multiple options."
            "\n\nEXAMPLES:"
            "\n- 'Elevated white blood cell count of 15,000' implies 'leukocytosis'"
            "\n- 'E. coli growing in urine culture' implies 'bacteriuria'"
            "\n- 'Hemoglobin of 6.5 g/dL' implies 'anemia'"
            "\n- 'Blood pressure 180/110 mmHg' implies 'hypertension'"
            "\n- 'Glucose 250 mg/dL' implies 'hyperglycemia'"
            "\n- 'Positive ANA titer 1:640' implies 'autoantibody positivity'"
            "\n- 'QT interval 520 ms' implies 'prolonged QT interval'"
            "\n\nIf you cannot identify a specific phenotype that is DIRECTLY implied with high confidence, "
            "respond with EXACTLY 'NO_CLEAR_PHENOTYPE_IMPLIED'."
        )

        # Phenotype validation - more examples
        self.phenotype_validation_system_message = (
            "You are a clinical expert specializing in phenotype identification for Human Phenotype Ontology (HPO) mapping. "
            "Your task is to determine if the proposed phenotype is a valid medical concept based on the provided HPO phenotype candidates."
            "\n\nEXAMPLES OF VALID PHENOTYPES:"
            "\n- 'Seizure' - abnormal electrical activity in the brain"
            "\n- 'Microcephaly' - abnormally small head circumference"
            "\n- 'Hypercholesterolemia' - elevated cholesterol levels"
            "\n- 'Polydactyly' - presence of extra digits"
            "\n- 'Scoliosis' - abnormal lateral curvature of the spine"
            "\n\nEXAMPLES OF INVALID PHENOTYPES:"
            "\n- 'Blood draw' - procedure, not a phenotype"
            "\n- 'Brain' - normal anatomical structure"
            "\n- 'Sleeping' - normal physiological process"
            "\n- 'Not feeling good' - too vague, not a specific medical concept"
            "\n- 'Treatment with antibiotics' - therapeutic intervention, not a phenotype"
            "\n\nAFTER REVIEWING THE CANDIDATES, respond with ONLY 'YES' if:"
            "\n- The phenotype EXACTLY or CLOSELY matches any HPO candidate, OR"
            "\n- The phenotype is a valid, recognized abnormal characteristic or condition in clinical medicine even if not in the candidates"
            "\n\nRespond with ONLY 'NO' if:"
            "\n- The proposed term is NOT a valid phenotype in clinical medicine"
            "\n- The term is too vague, general, or not recognized as a specific phenotype"
            "\n\nYOUR RESPONSE MUST BE EXACTLY 'YES' OR 'NO' WITH NO ADDITIONAL TEXT."
        )

        # Implication validation - more examples
        self.implication_validation_system_message = (
            "You are a clinical expert specializing in phenotype identification for Human Phenotype Ontology (HPO) mapping. "
            "Your task is to validate whether an implied phenotype is reasonable given the original entity and its context. "
            "Be extremely critical and conservative in your assessment. Say YES only if there is an unambiguous, "
            "direct clinical connection between the entity and the proposed phenotype."
            "\nThe connection must be evident from the entity itself, not inferred from general knowledge."
            "\n\nEXAMPLES of VALID implications:"
            "\n- Entity: 'E. coli in urine culture' → Implied phenotype: 'bacteriuria' (VALID: specific finding)"
            "\n- Entity: 'Hemoglobin of 6.5 g/dL' → Implied phenotype: 'anemia' (VALID: specific abnormal value)"
            "\n- Entity: 'Blood pressure 190/110 mmHg' → Implied phenotype: 'hypertension' (VALID: clearly elevated)"
            "\n- Entity: 'Respiratory rate 38/min in adult' → Implied phenotype: 'tachypnea' (VALID: clearly elevated)"
            "\n- Entity: 'Positive rheumatoid factor' → Implied phenotype: 'rheumatoid factor positivity' (VALID: direct finding)"
            "\n\nEXAMPLES of INVALID implications:"
            "\n- Entity: 'white blood cell count' → Implied phenotype: 'leukocytosis' (INVALID: no value specified)"
            "\n- Entity: 'taking insulin daily' → Implied phenotype: 'diabetes mellitus' (INVALID: medication alone)"
            "\n- Entity: 'retina' → Implied phenotype: 'retinopathy' (INVALID: normal anatomy without abnormality)"
            "\n- Entity: 'renal tissue' → Implied phenotype: 'glomerulonephritis' (INVALID: too specific without evidence)"
            "\n- Entity: 'abdominal pain' → Implied phenotype: 'appendicitis' (INVALID: symptom with many possible causes)"
            "\n- Entity: 'headache' → Implied phenotype: 'migraine' (INVALID: symptom too general for specific diagnosis)"
            "\n\nRespond with ONLY 'YES' if the implication is valid, or 'NO' if it's not valid."
            "\nYOUR RESPONSE MUST BE EXACTLY 'YES' OR 'NO' WITH NO ADDITIONAL TEXT."
        )

        # Lab test analysis - more examples
        self.lab_analysis_system_message = (
            "You are a clinical laboratory expert analyzing laboratory test results. "
            "Extract and analyze the lab test from the provided entity and determine if the value is abnormal."
            "\n\nEXAMPLES OF ABNORMAL RESULTS:"
            "\n- 'Hemoglobin 8.5 g/dL' - abnormality: 'decreased hemoglobin', direction: 'low'"
            "\n- 'White blood cell count of 15,000/μL' - abnormality: 'leukocytosis', direction: 'high'"
            "\n- 'Potassium 2.8 mEq/L' - abnormality: 'hypokalemia', direction: 'low'"
            "\n- 'ALT 120 U/L' - abnormality: 'elevated liver enzymes', direction: 'high'"
            "\n\nEXAMPLES OF NORMAL RESULTS:"
            "\n- 'Sodium 140 mEq/L' - abnormality: 'normal', direction: 'normal'"
            "\n- 'Glucose 90 mg/dL' - abnormality: 'normal', direction: 'normal'"
            "\n\nFor ABNORMAL results, provide a medically precise description of the abnormality "
            "(e.g., 'elevated glucose', 'decreased hemoglobin', 'leukocytosis')."
            "\n\nFor NORMAL results, simply state 'normal'."
            "\n\nProvide your response in this EXACT JSON format:"
            "\n{"
            '\n  "lab_name": "[extracted lab test name]",'
            '\n  "value": "[extracted value with units if available]",'
            '\n  "units": "[extracted units if separable from value]",'
            '\n  "is_abnormal": true/false,'
            '\n  "abnormality": "[descriptive term for the abnormality, or \'normal\' if not abnormal]",'
            '\n  "direction": "[high/low/normal]",'
            '\n  "confidence": [0.0-1.0 value indicating your confidence]'
            "\n}"
            "\n\nReturn ONLY the JSON with no additional text."
        )
