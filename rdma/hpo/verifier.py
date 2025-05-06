#!/usr/bin/env python3
import os
import sys
import torch
import numpy as np
from typing import List, Dict, Any, Optional

# Append parent directory to path for module imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, parent_dir)

from rdma.hporag.verify import (
    HPOVerifierConfig,
    MultiStageHPOVerifierV2,
    MultiStageHPOVerifierV3,
    MultiStageHPOVerifierV4,
)
from rdma.utils.llm_client import LocalLLMClient, APILLMClient
from rdma.utils.embedding import EmbeddingsManager


class HPOVerifier:
    """
    Wrapper for phenotype verification process.

    This class simplifies the use of different HPO verifiers by providing a
    consistent interface regardless of which verifier is used.
    """

    def __init__(
        self,
        llm_client,  # Required pre-initialized LLM client
        embeddings_file: str,
        verifier_version: str = "v4",
        device: str = None,
        lab_embeddings_file: str = None,
        retriever: str = "fastembed",
        retriever_model: str = "BAAI/bge-small-en-v1.5",
        min_context_length: int = 1,
        verifier_config: Optional[Dict] = None,
        debug: bool = False,
        use_demographics: bool = False,  # Flag to enable demographic extraction and use
    ):
        """
        Initialize the phenotype verifier wrapper.

        Args:
            llm_client: Pre-initialized LLM client.
            embeddings_file: Path to HPO embeddings file (required)
            verifier_version: Verifier version to use (v2, v3, v4)
            device: Device to use for inference (if None, will auto-detect)
            lab_embeddings_file: Path to lab test embeddings file for V4 verifier
            retriever: Type of retriever/embedding model to use
            retriever_model: Model name for retriever
            min_context_length: Minimum context length to consider valid
            verifier_config: Optional configuration dict for verifier
            debug: Enable debug output
            use_demographics: Whether to enable demographic extraction and use for lab test analysis
        """
        self.verifier_version = verifier_version
        self.min_context_length = min_context_length
        self.debug = debug
        self.use_demographics = use_demographics
        self.llm_client = llm_client

        # Auto-detect device if not specified
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        if self.debug:
            print(f"Using device: {self.device}")

        # Check for required embeddings file
        if embeddings_file is None:
            raise ValueError("embeddings_file is required for phenotype verification")

        # Initialize demographics extractor if needed
        self.demographics_extractor = None
        if self.use_demographics and verifier_version == "v4":
            from rdma.utils.demographic import DemographicsExtractor

            self.demographics_extractor = DemographicsExtractor(
                self.llm_client, debug=debug
            )
            if self.debug:
                print("Initialized demographics extractor for lab test analysis")

        # Initialize embedding manager
        self.embedding_manager = self._initialize_embedding_manager(
            retriever, retriever_model
        )

        # Create verifier configuration
        self.verifier_config = self._create_verifier_config(verifier_config)

        # Load embeddings
        self.embedded_documents = self._load_embeddings(embeddings_file)

        # Initialize verifier
        self.verifier = self._initialize_verifier(verifier_version, lab_embeddings_file)

        # Prepare verifier index
        self.verifier.prepare_index(self.embedded_documents)

    def _initialize_embedding_manager(self, retriever: str, retriever_model: str):
        """Initialize embedding manager for HPO matching."""
        if self.debug:
            print(f"Initializing {retriever} embedding manager")

        # Use model name only if needed by retriever type
        model_name = None
        if retriever in ["fastembed", "sentence_transformer"]:
            model_name = retriever_model

        return EmbeddingsManager(
            model_type=retriever, model_name=model_name, device=self.device
        )

    def _create_verifier_config(
        self, config_dict: Optional[Dict] = None
    ) -> HPOVerifierConfig:
        """Create verifier configuration."""
        # Default optimized configuration
        optimized_config = {
            "retrieval": {
                "direct": True,
                "implies": True,
                "extract": True,
                "validation": True,
                "implication": True,
            },
            "context": {
                "direct": True,
                "implies": True,
                "extract": True,
                "validation": True,
                "implication": True,
            },
        }

        # Use provided config if available, otherwise use optimized defaults
        if config_dict:
            if self.debug:
                print("Using provided verifier configuration")
            return HPOVerifierConfig.from_dict(config_dict)
        else:
            if self.debug:
                print("Using default optimized configuration")
            return HPOVerifierConfig.from_dict(optimized_config)

    def _load_embeddings(self, embeddings_file: str) -> Any:
        """Load embeddings from file."""
        if self.debug:
            print(f"Loading embeddings from {embeddings_file}")

        try:
            embedded_documents = np.load(embeddings_file, allow_pickle=True)
            if self.debug:
                print(f"Loaded {len(embedded_documents)} embedded documents")
            return embedded_documents
        except Exception as e:
            raise ValueError(f"Error loading embeddings file: {e}")

    def _initialize_verifier(
        self, verifier_version: str, lab_embeddings_file: Optional[str] = None
    ):
        """Initialize appropriate verifier based on version."""
        if self.debug:
            print(f"Initializing {verifier_version.upper()} verifier")

        if verifier_version == "v2":
            return MultiStageHPOVerifierV2(
                embedding_manager=self.embedding_manager,
                llm_client=self.llm_client,
                config=self.verifier_config,
                debug=self.debug,
            )
        elif verifier_version == "v4":
            return MultiStageHPOVerifierV4(
                embedding_manager=self.embedding_manager,
                llm_client=self.llm_client,
                config=self.verifier_config,
                debug=self.debug,
                lab_embeddings_file=lab_embeddings_file,
            )
        else:  # Default to v3
            return MultiStageHPOVerifierV3(
                embedding_manager=self.embedding_manager,
                llm_client=self.llm_client,
                config=self.verifier_config,
                debug=self.debug,
            )

    def filter_hallucinated_entities(
        self, entities_with_contexts: List[Dict]
    ) -> List[Dict]:
        """Filter out potentially hallucinated entities (those without valid contexts)."""
        initial_count = len(entities_with_contexts)

        # Keep only entities with non-empty context of minimum length
        filtered_entities = [
            entity
            for entity in entities_with_contexts
            if entity.get("context")
            and len(entity.get("context", "").strip()) >= self.min_context_length
        ]

        removed_count = initial_count - len(filtered_entities)

        if self.debug and removed_count > 0:
            print(
                f"Filtered out {removed_count} potentially hallucinated entities with no context"
            )

        return filtered_entities

    def verify(
        self,
        entities_with_contexts: List[Dict[str, str]],
        clinical_text: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Verify entities as phenotypes and classify them as direct or implied.

        Args:
            entities_with_contexts: List of dictionaries with 'entity' and 'context' fields
            clinical_text: Original clinical text for demographic extraction (if enabled)

        Returns:
            List of dictionaries with entity, context, and phenotype type (direct/implied)
        """
        if self.debug:
            print(f"Verifying {len(entities_with_contexts)} entities")

        # Filter out potentially hallucinated entities
        filtered_entities = self.filter_hallucinated_entities(entities_with_contexts)

        if self.debug:
            print(f"Processing {len(filtered_entities)} entities after filtering")

        # Extract demographics if enabled and text is provided
        sample_data = None
        if self.use_demographics and self.demographics_extractor and clinical_text:
            sample_data = self.demographics_extractor.extract(clinical_text)
            if self.debug:
                print(f"Extracted demographics: {sample_data}")

        # Process entities individually or as a batch based on verifier version
        verified_phenotypes = []

        if self.verifier_version == "v4" and sample_data:
            # Process entities individually for v4 to utilize sample_data
            for entity_data in filtered_entities:
                result = self.verifier.process_entity(
                    entity=entity_data.get("entity", ""),
                    context=entity_data.get("context", ""),
                    sample_data=sample_data,
                )
                # Only add valid phenotypes to the results
                if result.get("status") in ["direct_phenotype", "implied_phenotype"]:
                    # Make sure context is included in the result
                    result["context"] = entity_data.get("context", "")
                    verified_phenotypes.append(result)
        else:
            # Use batch processing for other verifiers or when demographics not needed
            verified_phenotypes = self.verifier.batch_process(filtered_entities)

        if self.debug:
            print(
                f"Identified {len(verified_phenotypes)} phenotypes (direct or implied)"
            )

        return verified_phenotypes
