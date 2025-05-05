"""
RDMA Extractor module for extracting rare disease entities and contexts from clinical texts.

This module provides the RDMAExtractor class which can use different extraction methods:
- LLM-based extraction
- Retrieval-enhanced extraction
- Iterative extraction
- Multi-temperature extraction

Author: Claude
"""

import os
import json
import numpy as np
from typing import List, Dict, Any, Optional, Union
from datetime import datetime

from rdma.utils.embedding import EmbeddingsManager
from rdma.utils.llm_client import LocalLLMClient
from rdma.hporag.context import ContextExtractor
from rdma.rdrag.entity import (
    LLMRDExtractor,
    RetrievalEnhancedRDExtractor,
    IterativeLLMRDExtractor,
    MultiIterativeRDExtractor,
)


class RDMAExtractor:
    """
    Rare Disease Extraction component that extracts disease mentions and their contexts from clinical text.

    This extractor supports multiple extraction methods:
    - "llm": Basic LLM-based extraction
    - "retrieval": Retrieval-enhanced extraction that uses embeddings
    - "iterative": Iterative extraction that makes multiple passes
    - "multi": Multi-temperature extraction that uses different temperatures
    """

    def __init__(
        self,
        llm_client: LocalLLMClient,
        extraction_method: str = "llm",
        system_prompt: str = "You are a medical expert specializing in rare diseases.",
        embedding_manager: Optional[EmbeddingsManager] = None,
        embedded_documents: Optional[List[Dict]] = None,
        window_size: int = 0,
        top_k: int = 10,
        min_sentence_size: Optional[int] = None,
        max_iterations: int = 3,
        temperatures: Optional[List[float]] = None,
        aggregation_type: str = "hybrid",
        hybrid_threshold: int = 2,
        debug: bool = False,
    ):
        """
        Initialize the RDMA Extractor.

        Args:
            llm_client: Client for LLM API calls
            extraction_method: Method for extraction ("llm", "retrieval", "iterative", or "multi")
            system_prompt: System prompt for LLM
            embedding_manager: Manager for embeddings (required for "retrieval" method)
            embedded_documents: Embedded documents (required for "retrieval" method)
            window_size: Context window size for extracted entities
            top_k: Number of top candidates to retrieve (for "retrieval" method)
            min_sentence_size: Minimum sentence size for retrieval-enhanced extraction
            max_iterations: Maximum iterations for iterative extraction
            temperatures: List of temperatures for multi-temperature extraction
            aggregation_type: Method to aggregate multi-temperature results ("union", "intersection", or "hybrid")
            hybrid_threshold: Threshold for hybrid aggregation
            debug: Whether to print debug information
        """
        self.llm_client = llm_client
        self.extraction_method = extraction_method
        self.system_prompt = system_prompt
        self.embedding_manager = embedding_manager
        self.embedded_documents = embedded_documents
        self.window_size = window_size
        self.top_k = top_k
        self.min_sentence_size = min_sentence_size
        self.max_iterations = max_iterations
        self.temperatures = temperatures or [0.01, 0.1, 0.3, 0.7, 0.9]
        self.aggregation_type = aggregation_type
        self.hybrid_threshold = hybrid_threshold
        self.debug = debug

        # Initialize context extractor
        self.context_extractor = ContextExtractor(debug=debug)

        # Initialize entity extractor based on method
        self._initialize_entity_extractor()

    def _debug_print(self, message: str, level: int = 0):
        """Print debug messages if debug mode is enabled."""
        if self.debug:
            indent = "  " * level
            print(f"{datetime.now().strftime('%H:%M:%S')} | {indent}{message}")

    def _initialize_entity_extractor(self):
        """Initialize the appropriate entity extractor based on the chosen method."""
        if self.extraction_method == "llm":
            self._debug_print("Initializing LLM-based entity extractor")
            self.entity_extractor = LLMRDExtractor(self.llm_client, self.system_prompt)

        elif self.extraction_method == "retrieval":
            # Validate required parameters for retrieval method
            if self.embedding_manager is None or self.embedded_documents is None:
                raise ValueError(
                    "embedding_manager and embedded_documents are required for retrieval method"
                )

            self._debug_print("Initializing retrieval-enhanced entity extractor")
            self.entity_extractor = RetrievalEnhancedRDExtractor(
                llm_client=self.llm_client,
                embedding_manager=self.embedding_manager,
                embedded_documents=self.embedded_documents,
                system_message=self.system_prompt,
                top_k=self.top_k,
                min_sentence_size=self.min_sentence_size,
            )

        elif self.extraction_method == "iterative":
            self._debug_print("Initializing iterative entity extractor")
            self.entity_extractor = IterativeLLMRDExtractor(
                self.llm_client, self.system_prompt, max_iterations=self.max_iterations
            )

        elif self.extraction_method == "multi":
            self._debug_print("Initializing multi-temperature entity extractor")
            self.entity_extractor = MultiIterativeRDExtractor(
                self.llm_client,
                self.system_prompt,
                temperatures=self.temperatures,
                max_iterations=self.max_iterations,
                aggregation_type=self.aggregation_type,
                hybrid_threshold=self.hybrid_threshold,
            )
        else:
            raise ValueError(f"Unsupported extraction method: {self.extraction_method}")

    def extract_from_text(self, text: str) -> List[Dict]:
        """
        Extract rare disease entities and their contexts from a single clinical text.

        Args:
            text: Clinical text to extract from

        Returns:
            List of dictionaries with entity and context information
        """
        if not text:
            return []

        # Extract entities using the selected extractor
        entities = self.entity_extractor.extract_entities(text)

        if self.debug:
            self._debug_print(
                f"Extracted {len(entities)} potential rare disease entities"
            )

        # Find contexts for entities
        entity_contexts = self.context_extractor.extract_contexts(
            entities, text, window_size=self.window_size
        )

        # Return structured output
        return entity_contexts

    def extract_from_texts(self, texts: List[str]) -> List[List[Dict]]:
        """
        Extract rare disease entities and their contexts from multiple clinical texts.

        Args:
            texts: List of clinical texts to extract from

        Returns:
            List of lists of dictionaries with entity and context information
        """
        results = []
        for i, text in enumerate(texts):
            try:
                if self.debug:
                    self._debug_print(f"Processing text {i+1}/{len(texts)}")

                # Extract entities and contexts
                entity_contexts = self.extract_from_text(text)
                results.append(entity_contexts)

            except Exception as e:
                self._debug_print(f"Error processing text {i+1}: {str(e)}")
                # Add empty result for this text
                results.append([])

        return results

    def extract_from_json(
        self, data: Dict[str, Dict[str, Any]]
    ) -> Dict[str, List[Dict]]:
        """
        Extract entities from a JSON structure with clinical notes.

        Args:
            data: Dictionary mapping case IDs to case data with clinical_text

        Returns:
            Dictionary mapping case IDs to lists of entities with contexts
        """
        results = {}
        total_cases = len(data)

        for i, (case_id, case_data) in enumerate(data.items()):
            try:
                if self.debug:
                    self._debug_print(
                        f"Processing case {i+1}/{total_cases} (ID: {case_id})"
                    )

                clinical_text = case_data.get("clinical_text", "")
                if not clinical_text:
                    self._debug_print(f"  No clinical text found for case {case_id}")
                    results[case_id] = {
                        "entities_with_contexts": [],
                        "metadata": case_data.get("metadata", {}),
                        "error": "No clinical text found",
                    }
                    continue

                # Extract entities and contexts
                entity_contexts = self.extract_from_text(clinical_text)

                # Store results
                results[case_id] = {
                    "clinical_text": clinical_text,
                    "entities_with_contexts": entity_contexts,
                    "metadata": {
                        "patient_id": case_data.get("patient_id", ""),
                        "admission_id": case_data.get("admission_id", ""),
                        "category": case_data.get("category", ""),
                        "chart_date": case_data.get("chart_date", ""),
                    },
                }

            except Exception as e:
                self._debug_print(f"Error processing case {case_id}: {str(e)}")
                # Still add the case to results but mark as failed
                results[case_id] = {
                    "clinical_text": case_data.get("clinical_text", ""),
                    "entities_with_contexts": [],
                    "metadata": {
                        "patient_id": case_data.get("patient_id", ""),
                        "admission_id": case_data.get("admission_id", ""),
                        "category": case_data.get("category", ""),
                        "chart_date": case_data.get("chart_date", ""),
                    },
                    "error": str(e),
                }

        return results

    def save_results(self, results: Dict[str, Any], output_file: str):
        """
        Save extraction results to a JSON file.

        Args:
            results: Results dictionary to save
            output_file: File path to save to
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

        # Add metadata about the extraction
        output_data = {
            "metadata": {
                "extraction_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "extraction_method": self.extraction_method,
                "system_prompt": self.system_prompt,
            },
            "results": results,
        }

        # Save to file
        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)

        if self.debug:
            self._debug_print(f"Results saved to {output_file}")
