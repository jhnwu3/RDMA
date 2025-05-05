"""
RDMA Matcher module for matching verified rare disease entities to ORPHA codes.

This module provides the RDMAMatcher class which matches verified rare disease
entities to their corresponding ORPHA codes using embedding-based retrieval
and LLM-powered matching.

Author: Claude
"""

import os
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime
import traceback

from rdma.utils.embedding import EmbeddingsManager
from rdma.utils.llm_client import LocalLLMClient
from rdma.rdrag.rd_match import SimpleRDMatcher


class RDMAMatcher:
    """
    Rare Disease matching component that matches verified disease entities to ORPHA codes.

    Features:
    - Matches entities to standard ORPHA codes
    - Uses embeddings to find the most similar disease terms
    - Supports exact matching and LLM-assisted semantic matching
    - Preserves original entity metadata throughout the matching process
    """

    def __init__(
        self,
        llm_client: LocalLLMClient,
        embedding_manager: EmbeddingsManager,
        embedded_documents: List[Dict],
        system_prompt: str = "You are a medical expert specializing in rare diseases with extensive knowledge of Orphanet classifications.",
        top_k: int = 5,
        debug: bool = False,
    ):
        """
        Initialize the RDMA Matcher.

        Args:
            llm_client: Client for LLM API calls
            embedding_manager: Manager for embeddings
            embedded_documents: Embedded documents containing ORPHA code metadata
            system_prompt: System prompt for LLM
            top_k: Number of top candidates to include in results
            debug: Whether to print debug information
        """
        self.llm_client = llm_client
        self.embedding_manager = embedding_manager
        self.embedded_documents = embedded_documents
        self.system_prompt = system_prompt
        self.top_k = top_k
        self.debug = debug

        # Initialize the matcher
        self._initialize_matcher()

    def _debug_print(self, message: str, level: int = 0):
        """Print debug messages if debug mode is enabled."""
        if self.debug:
            indent = "  " * level
            print(f"{datetime.now().strftime('%H:%M:%S')} | {indent}{message}")

    def _initialize_matcher(self):
        """Initialize the matcher."""
        self._debug_print("Initializing rare disease matcher")
        self.matcher = SimpleRDMatcher(
            embeddings_manager=self.embedding_manager,
            llm_client=self.llm_client,
            system_message=self.system_prompt,
        )

        # Prepare matcher index
        self._debug_print("Preparing matcher index")
        self.matcher.prepare_index(self.embedded_documents)

    def format_entities_for_matching(self, verified_entities: List[Dict]) -> List[Dict]:
        """
        Format verified entities for matching, preserving metadata.

        Args:
            verified_entities: List of verified rare disease entities

        Returns:
            List of formatted entities ready for matching
        """
        formatted_entities = []

        for entity_item in verified_entities:
            # Check for verified rare disease status
            if entity_item.get("status") == "verified_rare_disease" or entity_item.get(
                "is_verified", False
            ):
                formatted_entities.append(entity_item)
            elif isinstance(entity_item, dict) and "entity" in entity_item:
                # Also include entities that have entity field but might not have status
                formatted_entities.append(entity_item)

        return formatted_entities

    def match_entities(self, verified_entities: List[Dict]) -> List[Dict]:
        """
        Match verified entities to ORPHA codes.

        Args:
            verified_entities: List of verified rare disease entities

        Returns:
            List of matched entities with ORPHA codes
        """
        if not verified_entities:
            return []

        # Format entities for matching
        formatted_entities = self.format_entities_for_matching(verified_entities)

        if self.debug:
            self._debug_print(
                f"Matching {len(formatted_entities)} verified entities to ORPHA codes"
            )

        # Skip processing if no entities to match
        if not formatted_entities:
            return []

        # Match entities to ORPHA codes
        matched_results = self.matcher.match_rd_terms(
            formatted_entities, self.embedded_documents
        )

        return matched_results

    def match_from_json(self, verification_results: Dict[str, Dict]) -> Dict[str, Dict]:
        """
        Match verified rare diseases from verification results to ORPHA codes.

        Args:
            verification_results: Dictionary of verification results from RDMAVerifier

        Returns:
            Dictionary mapping case IDs to matching results
        """
        matching_results = {}
        total_cases = len(verification_results)

        for i, (case_id, case_data) in enumerate(verification_results.items()):
            try:
                if self.debug:
                    self._debug_print(
                        f"Processing case {i+1}/{total_cases} (ID: {case_id})"
                    )

                # Get verified rare diseases
                if (
                    "verified_rare_diseases" in case_data
                    and case_data["verified_rare_diseases"]
                ):
                    verified_entities = case_data["verified_rare_diseases"]
                    if self.debug:
                        self._debug_print(
                            f"  Found {len(verified_entities)} verified rare diseases"
                        )
                elif (
                    "entities_with_contexts" in case_data
                    and case_data["entities_with_contexts"]
                ):
                    # Using entities directly from extraction step
                    verified_entities = case_data["entities_with_contexts"]
                    if self.debug:
                        self._debug_print(
                            f"  Using {len(verified_entities)} extracted entities"
                        )
                else:
                    verified_entities = []
                    if self.debug:
                        self._debug_print("  No entities found")

                # Skip processing if no verified entities
                if not verified_entities:
                    matching_results[case_id] = {
                        "clinical_text": case_data.get("clinical_text", ""),
                        "metadata": case_data.get("metadata", {}),
                        "matched_diseases": [],
                        "note": "No verified rare diseases to match",
                    }
                    continue

                # Match entities to ORPHA codes
                matched_diseases = self.match_entities(verified_entities)

                # Store results
                matching_results[case_id] = {
                    "clinical_text": case_data.get("clinical_text", ""),
                    "metadata": case_data.get("metadata", {}),
                    "matched_diseases": matched_diseases,
                    "stats": {
                        "verified_diseases_count": len(verified_entities),
                        "matched_diseases_count": len(matched_diseases),
                    },
                }

            except Exception as e:
                self._debug_print(f"Error processing case {case_id}: {e}")
                if self.debug:
                    traceback.print_exc()
                # Still add the case to results but mark as failed
                matching_results[case_id] = {
                    "clinical_text": case_data.get("clinical_text", ""),
                    "metadata": case_data.get("metadata", {}),
                    "matched_diseases": [],
                    "stats": {
                        "verified_diseases_count": len(
                            case_data.get("verified_rare_diseases", [])
                        ),
                        "matched_diseases_count": 0,
                    },
                    "error": str(e),
                }

        return matching_results

    def save_results(self, results: Dict[str, Any], output_file: str):
        """
        Save matching results to a JSON file.

        Args:
            results: Results dictionary to save
            output_file: File path to save to
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

        # Convert any non-serializable objects (like numpy arrays) to serializable forms
        def convert_to_serializable(obj):
            if isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list) or isinstance(obj, tuple):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif obj is None or isinstance(obj, (bool, int, float, str)):
                return obj
            else:
                # For any other types, convert to string
                return str(obj)

        # Add metadata about the matching process
        output_data = {
            "metadata": {
                "matching_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "system_prompt": self.system_prompt,
                "top_k": self.top_k,
            },
            "results": convert_to_serializable(results),
        }

        # Save to file
        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)

        if self.debug:
            self._debug_print(f"Results saved to {output_file}")

    def save_csv(self, results: Dict[str, Dict], csv_file: str):
        """
        Save matching results as a CSV file.

        Args:
            results: Dictionary of matching results
            csv_file: File path to save CSV
        """
        csv_data = []

        for case_id, case_data in results.items():
            matched_diseases = case_data.get("matched_diseases", [])
            metadata = case_data.get("metadata", {})

            for disease in matched_diseases:
                # Extract relevant fields for CSV
                entry = {
                    "document_id": case_id,
                    "patient_id": metadata.get("patient_id", ""),
                    "admission_id": metadata.get("admission_id", ""),
                    "category": metadata.get("category", ""),
                    "chart_date": metadata.get("chart_date", ""),
                    "entity": disease.get("entity", ""),
                    "rd_term": disease.get("rd_term", ""),
                    "orpha_id": disease.get("orpha_id", ""),
                    "match_method": disease.get("match_method", ""),
                    "confidence_score": disease.get("confidence_score", 0.0),
                }
                csv_data.append(entry)

        # Convert to DataFrame and save
        df = pd.DataFrame(csv_data)
        df.to_csv(csv_file, index=False)

        if self.debug:
            self._debug_print(f"CSV results saved to {csv_file}")

    def print_matching_summary(self, results: Dict[str, Dict]):
        """
        Print a summary of the matching results.

        Args:
            results: Dictionary of matching results
        """
        # Calculate overall statistics
        total_verified_entities = sum(
            case_data.get("stats", {}).get("verified_diseases_count", 0)
            for case_id, case_data in results.items()
        )
        total_matched_entities = sum(
            case_data.get("stats", {}).get("matched_diseases_count", 0)
            for case_id, case_data in results
        )

        # Calculate match rate
        match_rate = (
            (total_matched_entities / total_verified_entities * 100)
            if total_verified_entities > 0
            else 0
        )

        print("\n===== Matching Summary =====")
        print(f"Total verified rare diseases: {total_verified_entities}")
        print(
            f"Successfully matched to ORPHA codes: {total_matched_entities} ({match_rate:.1f}%)"
        )
