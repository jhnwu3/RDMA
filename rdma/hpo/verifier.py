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

from hporag.verify import (
    HPOVerifierConfig,
    MultiStageHPOVerifierV2,
    MultiStageHPOVerifierV3,
    MultiStageHPOVerifierV4,
)
from utils.llm_client import LocalLLMClient, APILLMClient
from utils.embedding import EmbeddingsManager


class HPOVerifier:
    """
    Wrapper for phenotype verification process.

    This class simplifies the use of different HPO verifiers by providing a
    consistent interface regardless of which verifier is used.
    """

    def __init__(
        self,
        verifier_version: str = "v3",
        llm_type: str = "local",
        model_type: str = "llama3_70b",
        device: str = None,
        cache_dir: str = None,
        temperature: float = 0.2,
        api_config: str = None,
        embeddings_file: str = None,
        lab_embeddings_file: str = None,
        retriever: str = "fastembed",
        retriever_model: str = "BAAI/bge-small-en-v1.5",
        min_context_length: int = 1,
        verifier_config: Optional[Dict] = None,
        debug: bool = False,
    ):
        """
        Initialize the phenotype verifier wrapper.

        Args:
            verifier_version: Verifier version to use (v2, v3, v4)
            llm_type: Type of LLM client ('local' or 'api')
            model_type: Model type for local LLM
            device: Device to use for inference (if None, will auto-detect)
            cache_dir: Directory for caching models
            temperature: Temperature for LLM inference
            api_config: Path to API configuration file for API LLM
            embeddings_file: Path to HPO embeddings file (required)
            lab_embeddings_file: Path to lab test embeddings file for V4 verifier
            retriever: Type of retriever/embedding model to use
            retriever_model: Model name for retriever
            min_context_length: Minimum context length to consider valid
            verifier_config: Optional configuration dict for verifier
            debug: Enable debug output
        """
        self.verifier_version = verifier_version
        self.min_context_length = min_context_length
        self.debug = debug

        # Auto-detect device if not specified
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        if self.debug:
            print(f"Using device: {self.device}")

        # Check for required embeddings file
        if embeddings_file is None:
            raise ValueError("embeddings_file is required for phenotype verification")

        # Initialize LLM client
        self.llm_client = self._initialize_llm_client(
            llm_type, model_type, device, cache_dir, temperature, api_config
        )

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

    def _initialize_llm_client(
        self,
        llm_type: str,
        model_type: str,
        device: str,
        cache_dir: Optional[str],
        temperature: float,
        api_config: Optional[str],
    ):
        """Initialize LLM client based on type."""
        if self.debug:
            print(f"Initializing {llm_type} LLM client")

        if llm_type == "api":
            if api_config:
                return APILLMClient.from_config(api_config)
            else:
                return APILLMClient.initialize_from_input()
        else:  # local
            return LocalLLMClient(
                model_type=model_type,
                device=device,
                cache_dir=cache_dir,
                temperature=temperature,
            )

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
        self, entities_with_contexts: List[Dict[str, str]]
    ) -> List[Dict[str, Any]]:
        """
        Verify entities as phenotypes and classify them as direct or implied.

        Args:
            entities_with_contexts: List of dictionaries with 'entity' and 'context' fields

        Returns:
            List of dictionaries with entity, context, and phenotype type (direct/implied)
        """
        if self.debug:
            print(f"Verifying {len(entities_with_contexts)} entities")

        # Filter out potentially hallucinated entities
        filtered_entities = self.filter_hallucinated_entities(entities_with_contexts)

        if self.debug:
            print(f"Processing {len(filtered_entities)} entities after filtering")

        # Verify entities using the appropriate verifier
        verified_phenotypes = self.verifier.batch_process(filtered_entities)

        if self.debug:
            print(
                f"Identified {len(verified_phenotypes)} phenotypes (direct or implied)"
            )

        return verified_phenotypes
