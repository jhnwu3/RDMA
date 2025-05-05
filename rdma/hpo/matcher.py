#!/usr/bin/env python3
import os
import sys
import torch
import numpy as np
import json
from typing import List, Dict, Any, Optional, Union


from rdma.hporag.hpo_match import RAGHPOMatcher, OptimizedRAGHPOMatcher
from rdma.utils.embedding import EmbeddingsManager


class HPOMatcher:
    """
    Wrapper for HPO matching process.

    This class simplifies the use of HPO matchers by providing a consistent
    interface for matching phenotypes to HPO codes.
    """

    def __init__(
        self,
        llm_client,  # Required pre-initialized LLM client
        optimizer_version: str = "standard",
        device: str = None,
        embeddings_file: str = None,
        retriever: str = "fastembed",
        retriever_model: str = "BAAI/bge-small-en-v1.5",
        top_k: int = 5,
        system_prompt_file: str = None,
        debug: bool = False,
    ):
        """
        Initialize the HPO matcher wrapper.

        Args:
            llm_client: Pre-initialized LLM client
            optimizer_version: Version of matcher ('standard' or 'optimized')
            device: Device to use for inference (if None, will auto-detect)
            embeddings_file: Path to HPO embeddings file (required)
            retriever: Type of retriever/embedding model to use
            retriever_model: Model name for retriever
            top_k: Number of top candidates to include in results
            system_prompt_file: File containing system prompts
            debug: Enable debug output
        """
        self.optimizer_version = optimizer_version
        self.top_k = top_k
        self.debug = debug
        self.llm_client = llm_client

        # Auto-detect device if not specified
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        if self.debug:
            print(f"Using device: {self.device}")

        # Check for required embeddings file
        if embeddings_file is None:
            raise ValueError("embeddings_file is required for HPO matching")

        # Load system prompt
        self.system_prompt = self._load_system_prompt(system_prompt_file)

        # Initialize embedding manager
        self.embedding_manager = self._initialize_embedding_manager(
            retriever, retriever_model
        )

        # Load embeddings
        self.embedded_documents = self._load_embeddings(embeddings_file)

        # Initialize matcher
        self.matcher = self._initialize_matcher(optimizer_version)

        # Prepare matcher index
        self.matcher.prepare_index(self.embedded_documents)

    def _load_system_prompt(self, system_prompt_file: Optional[str]) -> str:
        """Load system prompt from file or use default."""
        default_prompt = """You are a clinical expert specialized in Human Phenotype Ontology (HPO) coding. 
        Your task is to analyze a phenotype term and its context, then determine the most appropriate HPO code.
        Consider both the entity and its context. Respond with only the HPO ID (e.g., HP:0001250)."""

        if system_prompt_file is None:
            if self.debug:
                print("Using default system prompt")
            return default_prompt

        try:
            with open(system_prompt_file, "r") as f:
                prompts = json.load(f)
                system_message = prompts.get("system_message_II", default_prompt)
                if self.debug:
                    print(f"Loaded system prompt from {system_prompt_file}")
                return system_message
        except Exception as e:
            if self.debug:
                print(f"Error loading system prompt: {e}")
                print("Using default system prompt")
            return default_prompt

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

    def _initialize_matcher(self, optimizer_version: str):
        """Initialize matcher based on version."""
        if self.debug:
            print(f"Initializing {optimizer_version} matcher")

        if optimizer_version == "optimized":
            return OptimizedRAGHPOMatcher(
                embeddings_manager=self.embedding_manager,
                llm_client=self.llm_client,
                system_message=self.system_prompt,
                debug=self.debug,
            )
        else:  # standard
            return RAGHPOMatcher(
                embeddings_manager=self.embedding_manager,
                llm_client=self.llm_client,
                system_message=self.system_prompt,
            )

    def format_phenotypes_for_matching(
        self, verified_phenotypes: List[Dict]
    ) -> List[Dict]:
        """
        Format verified phenotypes for HPO matching.

        Args:
            verified_phenotypes: List of verified phenotype dictionaries

        Returns:
            List of formatted entities with entity and context fields
        """
        formatted_entities = []
        entities = []
        contexts = []

        for phenotype in verified_phenotypes:
            # Extract entity and context
            entity_text = phenotype.get("phenotype", "")
            context = phenotype.get("context", "")

            if entity_text:  # Skip empty entities
                formatted_entities.append(
                    {
                        "entity": entity_text,
                        "context": context,
                        # Store original phenotype data for later
                        "original_data": phenotype,
                    }
                )
                entities.append(entity_text)
                contexts.append(context)

        return formatted_entities, entities, contexts

    def match(self, verified_phenotypes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Match verified phenotypes to HPO terms.

        Args:
            verified_phenotypes: List of dictionaries with phenotype information
                                (output from PhenotypeVerifierWrapper)

        Returns:
            List of dictionaries with phenotypes and their matched HPO codes
        """
        if self.debug:
            print(f"Matching {len(verified_phenotypes)} phenotypes to HPO terms")

        # Format phenotypes for matching
        formatted_entities, entities, contexts = self.format_phenotypes_for_matching(
            verified_phenotypes
        )

        if not entities:
            if self.debug:
                print("No phenotypes to match")
            return []

        # Match entities to HPO terms
        matches = self.matcher.match_hpo_terms(
            entities, self.embedded_documents, contexts
        )

        # Combine match results with original data
        matched_phenotypes = []
        for i, match in enumerate(matches):
            if i < len(formatted_entities):
                # Get original data
                original_data = formatted_entities[i].get("original_data", {}).copy()

                # Add match data
                phenotype_match = {
                    # Original phenotype data
                    **original_data,
                    # Add HPO match information
                    "hpo_term": match.get("hpo_term", ""),
                    "hp_id": match.get(
                        "hpo_term", ""
                    ),  # HPO ID is the same as hpo_term
                    "match_method": match.get("match_method", ""),
                    "confidence_score": match.get("confidence_score", 0.0),
                    "top_candidates": match.get("top_candidates", [])[
                        : self.top_k
                    ],  # Limit to top_k
                }

                matched_phenotypes.append(phenotype_match)

        if self.debug:
            print(f"Successfully matched {len(matched_phenotypes)} phenotypes")

        return matched_phenotypes
