#!/usr/bin/env python3
import torch
import numpy as np
from typing import List, Dict, Any


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
        multi_match: bool = False,
        parent_match: bool = False,
        debug: bool = False,
        prefer_specific: bool = True,
        require_grounding: bool = False,
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
            multi_match: If True, return multiple valid HPO IDs per entity
            debug: Enable debug output
        """
        self.optimizer_version = optimizer_version
        self.top_k = top_k
        self.multi_match = multi_match
        self.parent_match = parent_match
        self.debug = debug
        self.llm_client = llm_client
        self.prefer_specific = prefer_specific
        self.require_grounding = require_grounding

        if self.parent_match:
            from pyhpo import Ontology
            self.ontology = Ontology()

        # Auto-detect device if not specified
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        if self.debug:
            print(f"Using device: {self.device}")

        # Check for required embeddings file
        if embeddings_file is None:
            raise ValueError("embeddings_file is required for HPO matching")

        # Build a mode-specific prompt directly in code.
        self.system_prompt = self._build_matching_prompt()

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

    def _get_parent_ids(self, hp_id: str) -> List[str]:
        try:
            term = self.ontology.get_hpo_object(hp_id)
            return [p.id for p in term.parents]
        except Exception:
            return []

    def _build_matching_prompt(self) -> str:
        """Compose mode-specific matching instructions in code."""
        base_prompt = (
            "You are a rare disease expert with extensive medical knowledge. "
            "Identify the most clinically appropriate and context-supported "
            "Human Phenotype Ontology (HPO) term(s) for the given patient data "
            "and provided candidate context. Prioritize specificity and clinical "
            "relevance while avoiding tangential or weakly related terms. Do not "
            "invent findings that are not supported by the text and context."
        )

        parts = [base_prompt]

        if self.prefer_specific:
            parts.append(
                "GRANULARITY RULE: When candidates exist at different levels of "
                "specificity (parent vs child terms), always prefer the MOST SPECIFIC "
                "term that is directly and explicitly supported by the text. Do NOT "
                "choose a parent or ancestor HPO term unless the text clearly does not "
                "support a more specific classification. For example, if the text says "
                "'short metacarpals', prefer HP:0010049 over its parent HP:0001167."
            )

        if self.require_grounding:
            parts.append(
                "GROUNDING RULE: The HPO term you select MUST correspond to a phenotype "
                "that is explicitly stated or directly and unambiguously implied by the "
                "original patient text. If none of the candidate HPO terms are clearly "
                "supported by the text, output exactly: NO_MATCH"
            )

        if self.multi_match:
            mode_clause = (
                "MODE: multi_match=true. Return all applicable HPO IDs that are "
                "well-supported by the entity and context. Include only clinically "
                "relevant matches and exclude weak, tangential, or speculative "
                "matches. Output only HPO IDs (e.g., HP:0001250, HP:...), "
                "with no extra commentary."
            )
        else:
            mode_clause = (
                "MODE: multi_match=false. Return exactly one HPO ID: the single "
                "best-supported and most clinically relevant match. Output only the "
                "HPO ID (e.g., HP:0001250), with no extra commentary."
            )

        parts.append(mode_clause)
        return "\n\n".join(parts)

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
                require_grounding=self.require_grounding,
            )
        else:  # standard
            return RAGHPOMatcher(
                embeddings_manager=self.embedding_manager,
                llm_client=self.llm_client,
                system_message=self.system_prompt,
                require_grounding=self.require_grounding,
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
            entity_text = phenotype.get("phenotype") or phenotype.get("entity", "")
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
            entities,
            self.embedded_documents,
            contexts,
            multi_match=self.multi_match,
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

                if self.multi_match:
                    hp_ids = [h for h in match.get("hp_ids", []) if h]
                    if not hp_ids and phenotype_match["hp_id"]:
                        hp_ids = [phenotype_match["hp_id"]]
                    phenotype_match["hpo_terms"] = hp_ids
                    phenotype_match["hp_ids"] = hp_ids

                matched_phenotypes.append(phenotype_match)

        if self.parent_match:
            for pm in matched_phenotypes:
                base_ids = pm.get("hp_ids") or (
                    [pm["hp_id"]] if pm.get("hp_id") else []
                )
                expanded = list(base_ids)
                for hp_id in base_ids:
                    for parent_id in self._get_parent_ids(hp_id):
                        if parent_id not in expanded:
                            expanded.append(parent_id)
                pm["hpo_terms"] = expanded
                pm["hp_ids"] = expanded

        if self.debug:
            print(f"Successfully matched {len(matched_phenotypes)} phenotypes")

        return matched_phenotypes
