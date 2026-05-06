#!/usr/bin/env python3
import torch
from typing import List, Dict, Any, Optional

# Append parent directory to path for module imports
# current_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.dirname(os.path.dirname(current_dir))
# sys.path.insert(0, parent_dir)

from rdma.hporag.entity import (
    LLMEntityExtractor,
    IterativeLLMEntityExtractor,
    MultiIterativeExtractor,
    RetrievalEnhancedEntityExtractor,
)
from rdma.hporag.context import ContextExtractor
from rdma.utils.demographic import DemographicsExtractor
from rdma.utils.embedding import EmbeddingsManager


class PhenotypeExtractor:
    """
    Wrapper for entity extraction process from clinical texts.

    This class simplifies the use of different entity extractors by providing a
    consistent interface regardless of which extractor is used.
    """

    def __init__(
        self,
        llm_client,  # Required pre-initialized LLM client
        extractor_type: str = "retrieval",
        system_prompt_file: str = None,
        max_iterations: int = 1,
        embeddings_file: str = None,
        retriever: str = "fastembed",
        retriever_model: str = "BAAI/bge-small-en-v1.5",
        retriever_device: str = None,
        top_k: int = 5,
        negation: bool = False,
        family_history: bool = False,
        extract_demographics: bool = False,
        debug: bool = False,
        decompose_compound: bool = False,
        extract_qualified: bool = False,
        exhaustive: bool = False,
        exclude_etiology: bool = True,
        window_size: int = 0,
    ):
        """
        Initialize the entity extractor wrapper.

        Args:
            llm_client: Pre-initialized LLM client
            extractor_type: Type of entity extractor to use ('simple', 'iterative', 'multi', 'retrieval')
            system_prompt_file: Deprecated and ignored (kept for compatibility)
            max_iterations: Maximum iterations for iterative extractor
            embeddings_file: Path to embeddings file for retrieval-enhanced extraction
            retriever: Type of retriever/embedding model to use
            retriever_model: Model name for retriever
            retriever_device: Device to use for retriever (if None, will auto-detect)
            top_k: Number of top candidates to retrieve per sentence
            negation: If True, exclude negated findings during extraction
            family_history: If True, exclude family-history findings during extraction
            extract_demographics: Whether to extract demographic information
            debug: Enable debug output
        """
        self.extractor_type = extractor_type
        self.debug = debug
        self.llm_client = llm_client
        self.extract_demographics = extract_demographics
        self.negation = negation
        self.family_history = family_history
        self.decompose_compound = decompose_compound
        self.extract_qualified = extract_qualified
        self.exhaustive = exhaustive
        self.exclude_etiology = exclude_etiology
        self.window_size = window_size

        # Auto-detect retriever device if not specified
        if retriever_device is None:
            retriever_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.retriever_device = retriever_device

        if self.debug:
            print(f"Using device for retriever: {self.retriever_device}")

        if system_prompt_file and self.debug:
            print(
                "system_prompt_file is deprecated for PhenotypeExtractor "
                "and is ignored."
            )

        # Initialize entity extractor based on type
        self.entity_extractor = self._initialize_entity_extractor(
            extractor_type,
            max_iterations,
            embeddings_file,
            retriever,
            retriever_model,
            top_k,
            self.negation,
            self.family_history,
            self.decompose_compound,
            self.extract_qualified,
            self.exhaustive,
            self.exclude_etiology,
        )

        # Initialize context extractor
        self.context_extractor = ContextExtractor(debug=debug)

        # Initialize demographics extractor if needed
        self.demographics_extractor = None
        if extract_demographics:
            if self.debug:
                print("Initializing demographics extractor")
            self.demographics_extractor = DemographicsExtractor(
                llm_client=self.llm_client, debug=self.debug
            )

    def _initialize_entity_extractor(
        self,
        extractor_type: str,
        max_iterations: int,
        embeddings_file: Optional[str],
        retriever: str,
        retriever_model: str,
        top_k: int,
        negation: bool,
        family_history: bool,
        decompose_compound: bool = False,
        extract_qualified: bool = False,
        exhaustive: bool = False,
        exclude_etiology: bool = True,
    ):
        """Initialize entity extractor based on type."""
        if self.debug:
            print(f"Initializing {extractor_type} entity extractor")

        if extractor_type == "simple":
            return LLMEntityExtractor(
                self.llm_client,
                negation=negation,
                family_history=family_history,
                decompose_compound=decompose_compound,
                extract_qualified=extract_qualified,
                exhaustive=exhaustive,
                exclude_etiology=exclude_etiology,
            )
        elif extractor_type == "multi":
            # Multi uses different temperatures for multiple passes
            temperatures = [0.01, 0.1, 0.3, 0.7, 0.9]
            return MultiIterativeExtractor(
                self.llm_client,
                temperatures=temperatures,
                max_iterations=max_iterations,
                negation=negation,
                family_history=family_history,
                decompose_compound=decompose_compound,
                extract_qualified=extract_qualified,
                exhaustive=exhaustive,
                exclude_etiology=exclude_etiology,
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
                device=self.retriever_device,
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
                top_k=top_k,
                negation=negation,
                family_history=family_history,
                decompose_compound=decompose_compound,
                extract_qualified=extract_qualified,
                exhaustive=exhaustive,
                exclude_etiology=exclude_etiology,
            )
        else:  # Default to iterative
            return IterativeLLMEntityExtractor(
                self.llm_client,
                max_iterations=max_iterations,
                negation=negation,
                family_history=family_history,
                decompose_compound=decompose_compound,
                extract_qualified=extract_qualified,
                exhaustive=exhaustive,
                exclude_etiology=exclude_etiology,
            )

    def extract(self, clinical_texts: List[str]) -> List[Dict[str, Any]]:
        """
        Extract entities from clinical texts with optional demographic information.

        Args:
            clinical_texts: List of clinical texts to process

        Returns:
            List of dictionaries with 'entity', 'context', and optional 'demographics' fields
        """
        if self.debug:
            print(f"Extracting entities from {len(clinical_texts)} clinical texts")

        all_entities_with_contexts = []

        # Extract demographics first if needed
        demographics_data = {}
        if self.extract_demographics and self.demographics_extractor:
            if self.debug:
                print("Extracting demographic information")
            for i, text in enumerate(clinical_texts):
                demographics = self.demographics_extractor.extract(text)
                demographics_data[i] = demographics
                if self.debug:
                    print(f"Text {i}: Demographics - {demographics}")

        # Process each text
        for i, text in enumerate(clinical_texts):
            # Extract entities using entity extractor
            entities = self.entity_extractor.extract_entities(text)

            if self.debug:
                print(f"Extracted {len(entities)} entities from text {i}")

            # Find contexts for entities
            entity_contexts = self.context_extractor.extract_contexts(entities, text, window_size=self.window_size)

            # Add demographics if available
            if self.extract_demographics and i in demographics_data:
                for entity_context in entity_contexts:
                    entity_context["demographics"] = demographics_data[i]

            # Add to results
            all_entities_with_contexts.extend(entity_contexts)

        if self.debug:
            print(f"Total entities with contexts: {len(all_entities_with_contexts)}")

        return all_entities_with_contexts
