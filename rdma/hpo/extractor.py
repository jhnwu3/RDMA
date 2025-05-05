#!/usr/bin/env python3
import os
import sys
import torch
from typing import List, Dict, Any, Optional

# Append parent directory to path for module imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, parent_dir)

from hporag.entity import (
    LLMEntityExtractor,
    IterativeLLMEntityExtractor,
    MultiIterativeExtractor,
    RetrievalEnhancedEntityExtractor,
)
from hporag.context import ContextExtractor
from utils.llm_client import LocalLLMClient, APILLMClient
from utils.embedding import EmbeddingsManager


class PhenotypeExtractor:
    """
    Wrapper for entity extraction process from clinical texts.

    This class simplifies the use of different entity extractors by providing a
    consistent interface regardless of which extractor is used.
    """

    def __init__(
        self,
        extractor_type: str = "iterative",
        llm_type: str = "local",
        model_type: str = "llama3_70b",
        device: str = None,
        system_prompt_file: str = None,
        max_iterations: int = 3,
        cache_dir: str = None,
        temperature: float = 0.2,
        api_config: str = None,
        embeddings_file: str = None,
        retriever: str = "fastembed",
        retriever_model: str = "BAAI/bge-small-en-v1.5",
        top_k: int = 10,
        debug: bool = False,
    ):
        """
        Initialize the entity extractor wrapper.

        Args:
            extractor_type: Type of entity extractor to use ('simple', 'iterative', 'multi', 'retrieval')
            llm_type: Type of LLM client ('local' or 'api')
            model_type: Model type for local LLM
            device: Device to use for inference (if None, will auto-detect)
            system_prompt_file: File containing system prompts
            max_iterations: Maximum iterations for iterative extractor
            cache_dir: Directory for caching models
            temperature: Temperature for LLM inference
            api_config: Path to API configuration file for API LLM
            embeddings_file: Path to embeddings file for retrieval-enhanced extraction
            retriever: Type of retriever/embedding model to use
            retriever_model: Model name for retriever
            top_k: Number of top candidates to retrieve per sentence
            debug: Enable debug output
        """
        self.extractor_type = extractor_type
        self.debug = debug

        # Auto-detect device if not specified
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        if self.debug:
            print(f"Using device: {self.device}")

        # Load system prompt
        self.system_prompt = self._load_system_prompt(system_prompt_file)

        # Initialize LLM client
        self.llm_client = self._initialize_llm_client(
            llm_type, model_type, device, cache_dir, temperature, api_config
        )

        # Initialize entity extractor based on type
        self.entity_extractor = self._initialize_entity_extractor(
            extractor_type,
            max_iterations,
            embeddings_file,
            retriever,
            retriever_model,
            top_k,
        )

        # Initialize context extractor
        self.context_extractor = ContextExtractor(debug=debug)

    def _load_system_prompt(self, system_prompt_file: Optional[str]) -> str:
        """Load system prompt from file or use default."""
        import json

        default_prompt = """You are a medical entity extraction assistant specializing in identifying medical terms in clinical notes. 
        Extract ALL possible medical terms, symptoms, conditions, diagnoses, anatomical entities, genetic factors, and phenotypes.
        Respond with a JSON array containing just the extracted terms. Format: {"findings": [term1, term2, ...]}"""

        if system_prompt_file is None:
            if self.debug:
                print("Using default system prompt")
            return default_prompt

        try:
            with open(system_prompt_file, "r") as f:
                prompts = json.load(f)
                system_message = prompts.get("system_message_I", default_prompt)
                if self.debug:
                    print(f"Loaded system prompt from {system_prompt_file}")
                return system_message
        except Exception as e:
            if self.debug:
                print(f"Error loading system prompt: {e}")
                print("Using default system prompt")
            return default_prompt

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

    def _initialize_entity_extractor(
        self,
        extractor_type: str,
        max_iterations: int,
        embeddings_file: Optional[str],
        retriever: str,
        retriever_model: str,
        top_k: int,
    ):
        """Initialize entity extractor based on type."""
        if self.debug:
            print(f"Initializing {extractor_type} entity extractor")

        if extractor_type == "simple":
            return LLMEntityExtractor(self.llm_client, self.system_prompt)
        elif extractor_type == "multi":
            # Multi uses different temperatures for multiple passes
            temperatures = [0.01, 0.1, 0.3, 0.7, 0.9]
            return MultiIterativeExtractor(
                self.llm_client,
                self.system_prompt,
                temperatures=temperatures,
                max_iterations=max_iterations,
            )
        elif extractor_type == "retrieval":
            if not embeddings_file:
                raise ValueError(
                    "embeddings_file is required for retrieval-enhanced entity extraction"
                )

            # Initialize embedding manager
            embedding_manager = EmbeddingsManager(
                model_type=retriever,
                model_name=(
                    retriever_model
                    if retriever in ["fastembed", "sentence_transformer"]
                    else None
                ),
                device=self.device,
            )

            # Load embeddings
            import numpy as np

            try:
                embedded_documents = np.load(embeddings_file, allow_pickle=True)
                if self.debug:
                    print(f"Loaded {len(embedded_documents)} embedded documents")
            except Exception as e:
                raise ValueError(f"Error loading embeddings file: {e}")

            return RetrievalEnhancedEntityExtractor(
                self.llm_client,
                embedding_manager,
                embedded_documents,
                self.system_prompt,
                top_k=top_k,
            )
        else:  # Default to iterative
            return IterativeLLMEntityExtractor(
                self.llm_client, self.system_prompt, max_iterations=max_iterations
            )

    def extract(self, clinical_texts: List[str]) -> List[Dict[str, str]]:
        """
        Extract entities from clinical texts.

        Args:
            clinical_texts: List of clinical texts to process

        Returns:
            List of dictionaries with 'entity' and 'context' fields
        """
        if self.debug:
            print(f"Extracting entities from {len(clinical_texts)} clinical texts")

        all_entities_with_contexts = []

        for text in clinical_texts:
            # Extract entities using entity extractor
            entities = self.entity_extractor.extract_entities(text)

            if self.debug:
                print(f"Extracted {len(entities)} entities")

            # Find contexts for entities
            entity_contexts = self.context_extractor.extract_contexts(entities, text)

            # Add to results
            all_entities_with_contexts.extend(entity_contexts)

        if self.debug:
            print(f"Total entities with contexts: {len(all_entities_with_contexts)}")

        return all_entities_with_contexts
