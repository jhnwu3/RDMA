"""
RDMA Verifier module for verifying extracted rare disease entities.

This module provides the RDMAVerifier class which verifies whether the
extracted entities are actual rare diseases using embedding-based retrieval
and LLM-powered verification.

Author: Claude
"""

import os
import json
import numpy as np
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from rdma.utils.embedding import EmbeddingsManager
from rdma.utils.llm_client import LocalLLMClient
from rdma.rdrag.verify import MultiStageRDVerifier


class RDMAVerifier:
    """
    Rare Disease verification component that determines if extracted entities are rare diseases.

    Features:
    - Uses embeddings to retrieve similar disease terms from Orphanet
    - Resolves abbreviations to their expanded forms (optional)
    - Performs multi-stage verification of entities
    - Returns verified rare diseases with confidence scores
    """

    def __init__(
        self,
        llm_client: LocalLLMClient,
        embedding_manager: EmbeddingsManager,
        embedded_documents: List[Dict],
        verifier_type: str = "multi_stage",
        system_prompt: str = "You are a medical expert specializing in rare diseases.",
        abbreviations_file: Optional[str] = None,
        use_abbreviations: bool = False,
        min_context_length: int = 1,
        top_k: int = 5,
        debug: bool = False,
    ):
        """
        Initialize the RDMA Verifier.

        Args:
            llm_client: Client for LLM API calls
            embedding_manager: Manager for embeddings
            embedded_documents: Embedded documents for rare disease lookup
            verifier_type: Type of verifier to use ("simple" or "multi_stage")
            system_prompt: System prompt for LLM
            abbreviations_file: Path to abbreviations embeddings file (optional)
            use_abbreviations: Whether to use abbreviation resolution
            min_context_length: Minimum context length to consider valid
            top_k: Number of top candidates to include in verification
            debug: Whether to print debug information
        """
        self.llm_client = llm_client
        self.embedding_manager = embedding_manager
        self.embedded_documents = embedded_documents
        self.verifier_type = verifier_type
        self.system_prompt = system_prompt
        self.abbreviations_file = abbreviations_file
        self.use_abbreviations = use_abbreviations
        self.min_context_length = min_context_length
        self.top_k = top_k
        self.debug = debug

        # Initialize the appropriate verifier based on type
        self._initialize_verifier()

    def _debug_print(self, message: str, level: int = 0):
        """Print debug messages if debug mode is enabled."""
        if self.debug:
            indent = "  " * level
            print(f"{datetime.now().strftime('%H:%M:%S')} | {indent}{message}")

    def _initialize_verifier(self):
        """Initialize the appropriate verifier based on the chosen type."""
        if self.verifier_type == "multi_stage":
            self._debug_print("Initializing multi-stage rare disease verifier")
            self.verifier = MultiStageRDVerifier(
                embedding_manager=self.embedding_manager,
                llm_client=self.llm_client,
                config=None,  # No specific config needed
                debug=self.debug,
                abbreviations_file=self.abbreviations_file,
                use_abbreviations=self.use_abbreviations,
            )
            # Prepare verifier index
            self.verifier.prepare_index(self.embedded_documents)
        else:
            # Fallback to simple matcher/verifier
            from rdrag.rd_match import RAGRDMatcher

            self._debug_print("Initializing simple rare disease verifier")
            self.verifier = RAGRDMatcher(
                embeddings_manager=self.embedding_manager,
                llm_client=self.llm_client,
                system_message=self.system_prompt,
            )
            # Prepare verifier index
            self.verifier.prepare_index(self.embedded_documents)

    def _filter_hallucinated_entities(
        self, entities_with_contexts: List[Dict]
    ) -> List[Dict]:
        """
        Filter out potentially hallucinated entities (those without valid contexts).

        Args:
            entities_with_contexts: List of entities with their contexts

        Returns:
            Filtered list of entities with contexts
        """
        initial_count = len(entities_with_contexts)

        # Keep only entities with non-empty context
        filtered_entities = [
            entity
            for entity in entities_with_contexts
            if entity.get("context")
            and len(entity.get("context", "").strip()) >= self.min_context_length
            and entity.get("entity", "") in entity.get("context", "")
        ]

        removed_count = initial_count - len(filtered_entities)

        if removed_count > 0 and self.debug:
            self._debug_print(
                f"Filtered out {removed_count} potentially hallucinated entities with invalid context"
            )

        return filtered_entities

    def _format_entities_for_verification(
        self, entities_with_contexts: List[Dict]
    ) -> List[Dict]:
        """
        Format entities for verification and filter potential hallucinations.

        Args:
            entities_with_contexts: Raw entities with contexts

        Returns:
            Formatted entities ready for verification
        """
        # First, filter out potentially hallucinated entities
        filtered_entities = self._filter_hallucinated_entities(entities_with_contexts)

        # Then format the remaining entities for verification
        formatted_entities = []

        for entity_data in filtered_entities:
            # Ensure we have the expected fields
            entity = entity_data.get("entity", "")
            context = entity_data.get("context", "")

            if entity:  # Skip empty entities
                formatted_entities.append({"entity": entity, "context": context})

        return formatted_entities

    def verify_entities(self, entities_with_contexts: List[Dict]) -> List[Dict]:
        """
        Verify if entities are rare diseases.

        Args:
            entities_with_contexts: List of entities with contexts

        Returns:
            List of verified rare diseases with their contexts
        """
        if not entities_with_contexts:
            return []

        # Format entities for verification
        formatted_entities = self._format_entities_for_verification(
            entities_with_contexts
        )

        if self.debug:
            self._debug_print(
                f"Verifying {len(formatted_entities)} entities after filtering"
            )

        # Skip processing if all entities were filtered out
        if not formatted_entities:
            self._debug_print(
                "All entities were filtered out as potential hallucinations"
            )
            return []

        # Use the appropriate verification method based on verifier type
        if self.verifier_type == "multi_stage":
            # Use batch_process method for MultiStageRDVerifier
            verified_rare_diseases = self.verifier.batch_process(formatted_entities)
        else:
            # For the simple verifier, process each entity individually
            verified_rare_diseases = []

            for entity_data in formatted_entities:
                entity = entity_data["entity"]
                context = entity_data["context"]

                # Get candidates for verification
                candidates = self.verifier._retrieve_candidates(entity)

                # Verify if it's a rare disease
                is_rare_disease = self.verifier._verify_rare_disease(
                    entity, candidates[: self.top_k]
                )

                if is_rare_disease:
                    # Add to verified list
                    verified_entity = {
                        "entity": entity,
                        "context": context,
                        "is_verified": True,
                        "status": "verified_rare_disease",
                    }
                    verified_rare_diseases.append(verified_entity)

                    if self.debug:
                        self._debug_print(f"✓ Verified '{entity}' as a rare disease")

        return verified_rare_diseases

    def verify_from_json(self, extraction_results: Dict[str, Dict]) -> Dict[str, Dict]:
        """
        Verify rare diseases from extraction results.

        Args:
            extraction_results: Dictionary of extraction results from RDMAExtractor

        Returns:
            Dictionary mapping case IDs to verification results
        """
        verification_results = {}
        total_cases = len(extraction_results)

        for i, (case_id, case_data) in enumerate(extraction_results.items()):
            try:
                if self.debug:
                    self._debug_print(
                        f"Processing case {i+1}/{total_cases} (ID: {case_id})"
                    )

                # Get entities with contexts
                entities_with_contexts = case_data.get("entities_with_contexts", [])

                if self.debug:
                    self._debug_print(
                        f"  Processing {len(entities_with_contexts)} raw entities"
                    )

                # Verify rare diseases
                verified_rare_diseases = self.verify_entities(entities_with_contexts)

                # Calculate statistics
                original_count = len(entities_with_contexts)
                filtered_count = original_count - len(
                    self._format_entities_for_verification(entities_with_contexts)
                )
                verified_count = len(verified_rare_diseases)

                # Store results
                verification_results[case_id] = {
                    "clinical_text": case_data.get("clinical_text", ""),
                    "metadata": case_data.get("metadata", {}),
                    "verified_rare_diseases": verified_rare_diseases,
                    "stats": {
                        "original_entity_count": original_count,
                        "filtered_hallucinations": filtered_count,
                        "entities_verified": original_count - filtered_count,
                        "rare_diseases_found": verified_count,
                    },
                    "verifier_type": self.verifier_type,
                }

            except Exception as e:
                self._debug_print(f"Error processing case {case_id}: {e}")
                # Still add the case to results but mark as failed
                verification_results[case_id] = {
                    "clinical_text": case_data.get("clinical_text", ""),
                    "verified_rare_diseases": [],
                    "stats": {
                        "original_entity_count": len(
                            case_data.get("entities_with_contexts", [])
                        ),
                        "filtered_hallucinations": 0,
                        "entities_verified": 0,
                        "rare_diseases_found": 0,
                    },
                    "error": str(e),
                    "verifier_type": self.verifier_type,
                }

        return verification_results

    def save_results(self, results: Dict[str, Any], output_file: str):
        """
        Save verification results to a JSON file.

        Args:
            results: Results dictionary to save
            output_file: File path to save to
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

        # Add metadata about the verification
        output_data = {
            "metadata": {
                "verification_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "verifier_type": self.verifier_type,
                "system_prompt": self.system_prompt,
                "use_abbreviations": self.use_abbreviations,
            },
            "results": results,
        }

        # Save to file
        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)

        if self.debug:
            self._debug_print(f"Results saved to {output_file}")

    def print_verification_summary(self, results: Dict[str, Dict]):
        """
        Print a summary of the verification results.

        Args:
            results: Dictionary of verification results
        """
        # Calculate overall statistics
        total_input_entities = sum(
            len(case_data.get("entities_with_contexts", []))
            for case_id, case_data in results.items()
        )
        total_filtered_entities = sum(
            results[case_id].get("stats", {}).get("filtered_hallucinations", 0)
            for case_id in results
        )
        total_verified_entities = sum(
            results[case_id].get("stats", {}).get("entities_verified", 0)
            for case_id in results
        )
        total_rare_diseases = sum(
            len(results[case_id].get("verified_rare_diseases", []))
            for case_id in results
        )

        # Calculate rates
        filtered_rate = (
            (total_filtered_entities / total_input_entities * 100)
            if total_input_entities > 0
            else 0
        )
        verification_rate = (
            (total_rare_diseases / total_verified_entities * 100)
            if total_verified_entities > 0
            else 0
        )
        overall_rate = (
            (total_rare_diseases / total_input_entities * 100)
            if total_input_entities > 0
            else 0
        )

        print("\n===== Verification Summary =====")
        print(f"Total input entities: {total_input_entities}")
        print(
            f"Filtered out as potential hallucinations: {total_filtered_entities} ({filtered_rate:.1f}%)"
        )
        print(f"Entities after filtering: {total_verified_entities}")
        print(
            f"Verified as rare diseases: {total_rare_diseases} ({verification_rate:.1f}% of filtered entities)"
        )
        print(
            f"Overall yield: {overall_rate:.1f}% (rare diseases from original entities)"
        )
