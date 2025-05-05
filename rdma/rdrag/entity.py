from abc import ABC, abstractmethod
import json
import pandas as pd
import numpy as np
import re
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from fuzzywuzzy import fuzz
from rdma.hporag.context import ContextExtractor


class BaseRDExtractor(ABC):
    """Abstract base class for rare disease entity extraction."""

    @abstractmethod
    def extract_entities(self, text: str) -> List[str]:
        """Extract rare disease mentions from text."""
        pass

    @abstractmethod
    def process_batch(self, texts: List[str]) -> List[List[str]]:
        """Process a batch of texts for rare disease extraction."""
        pass


class LLMRDExtractor(BaseRDExtractor):
    """LLM-based rare disease entity extractor."""

    def __init__(self, llm_client, system_message: str):
        self.llm_client = llm_client
        self.system_message = system_message

    def extract_entities(self, text: str) -> List[str]:
        """Extract rare disease mentions using LLM."""
        prompt = f"""Extract all rare diseases and conditions that are NOT 
        negated (i.e., don't include terms that are preceded by 'no', 'not', 
        'without', etc.) from the text below.

        Text: {text}

        Return only a Python list of strings, with each term exactly as it 
        appears in the text. Ensure the output is concise without any additional notes, commentary, or meta explanations."""

        findings_text = self.llm_client.query(prompt, self.system_message)
        return self._extract_findings_from_response(findings_text)

    def _extract_findings_from_response(self, response: str) -> List[str]:
        """Parse LLM response to extract findings."""
        try:
            # Extract content between square brackets if present
            if "[" in response and "]" in response:
                response = response[response.find("[") + 1 : response.rfind("]")]

            # Split on commas and clean up each term
            findings = []
            for term in response.split(","):
                cleaned_term = term.strip().strip("\"'")
                if cleaned_term:
                    findings.append(cleaned_term)

            return findings
        except Exception as e:
            print(f"Error parsing LLM response: {str(e)}")
            return []

    def process_batch(self, texts: List[str]) -> List[List[str]]:
        """Process multiple texts for rare disease extraction."""
        results = []
        for text in texts:
            try:
                findings = self.extract_entities(text)
                results.append(findings)
            except Exception as e:
                print(f"Error processing text: {str(e)}")
                results.append([])
        return results


class RetrievalEnhancedRDExtractor(BaseRDExtractor):
    """
    Entity extractor that uses embedding retrieval to enhance LLM-based extraction.

    For each sentence:
    1. Retrieves relevant rare disease terms to provide context
    2. Uses this context to prompt the LLM for more accurate disease extraction

    This approach helps the LLM with domain knowledge before extraction begins.

    Features:
    - Optionally merges small sentences to ensure minimum chunk size for processing
    """

    def __init__(
        self,
        llm_client,
        embedding_manager,
        embedded_documents,
        system_message: str,
        top_k: int = 10,
        min_sentence_size: Optional[int] = None,
    ):
        """
        Initialize the retrieval-enhanced rare disease extractor.

        Args:
            llm_client: Client for querying the language model
            embedding_manager: Manager for vector embeddings and search
            embedded_documents: Rare disease terms with embeddings
            system_message: System message for LLM extraction
            top_k: Number of top candidates to retrieve per sentence
            min_sentence_size: Minimum character length for sentences (smaller ones will be merged)
        """
        self.llm_client = llm_client
        self.embedding_manager = embedding_manager
        self.embedded_documents = embedded_documents
        self.index = None
        self.system_message = system_message
        self.top_k = top_k
        self.min_sentence_size = min_sentence_size
        self.context_extractor = ContextExtractor()

    def prepare_index(self):
        """Prepare FAISS index from embedded documents if not already prepared."""
        if self.index is None:
            embeddings_array = self.embedding_manager.prepare_embeddings(
                self.embedded_documents
            )
            self.index = self.embedding_manager.create_index(embeddings_array)

    def _retrieve_candidates(self, sentence: str) -> List[Dict]:
        """
        Retrieve relevant rare disease candidates for a sentence.

        Args:
            sentence: Clinical sentence to find relevant rare disease terms for

        Returns:
            List of dictionaries with rare disease information and similarity scores
        """
        self.prepare_index()

        # Embed the query
        query_vector = self.embedding_manager.query_text(sentence).reshape(1, -1)

        # Search for similar items
        distances, indices = self.embedding_manager.search(
            query_vector, self.index, k=min(800, len(self.embedded_documents))
        )

        # Extract unique metadata
        candidates = []
        seen_metadata = set()

        for idx, distance in zip(indices[0], distances[0]):
            try:
                # Access document directly since it already has name, id, definition
                document = self.embedded_documents[idx]

                # Create an identifier for deduplication
                metadata_id = f"{document.get('name', '')}-{document.get('id', '')}"

                if metadata_id not in seen_metadata:
                    seen_metadata.add(metadata_id)
                    candidates.append(
                        {
                            "name": document.get("name", ""),
                            "id": document.get("id", ""),
                            "definition": document.get("definition", ""),
                            "similarity_score": 1.0
                            / (1.0 + distance),  # Convert distance to similarity
                        }
                    )

                    if len(candidates) >= self.top_k:
                        break
            except Exception as e:
                print(f"Error processing metadata at index {idx}: {e}")
                continue

        return candidates

    def _create_enhanced_prompt(self, sentence: str, candidates: List[Dict]) -> str:
        """
        Create a prompt enhanced with retrieved candidates.

        Args:
            sentence: Clinical sentence to extract rare diseases from
            candidates: Retrieved rare disease candidates to use as context

        Returns:
            Formatted prompt for LLM
        """
        # Format candidates as context
        context_items = []
        for candidate in candidates:
            context_items.append(f"- {candidate['name']} (ID: {candidate['id']})")

        context_text = "\n".join(context_items)

        # Create the enhanced prompt, runs way longer like 4x longer
        # prompt = (
        #     f'I have CLINICAL TEXT: "{sentence}"\n\n'
        #     f"Here are some relevant ORPHA rare disease terms for reference that may help you identify rare disease mentions in the sentence:\n\n"
        #     f"{context_text}\n\n"
        #     f"Based on this sentence, extract ALL medically relevant entities including but not limited to:\n"
        #     f"1. Any disease or condition (common or rare)\n"
        #     f"2. Signs and symptoms\n"
        #     f"3. Syndromes\n"
        #     f"4. Disorders\n"
        #     f"5. Medical findings\n"
        #     f"6. Abnormalities\n"
        #     f"7. Medical events\n"
        #     f"8. Any abbreviations or acronyms that might refer to medical conditions\n"
        #     f"9. Phenotypic descriptions\n"
        #     f"10. Congenital anomalies\n\n"
        #     f"Only include entities that are NOT negated (i.e., NOT preceded by 'no', 'not', 'without', 'ruled out', 'denies', etc.).\n\n"
        #     f"Return only a Python list of strings, with each entity extracted exactly as it appears in the CLINICAL TEXT. "
        #     f"Ensure the output is concise without any additional notes, commentary, or meta explanations."
        # )
        prompt = (
            f'I have CLINICAL TEXT: "{sentence}"\n\n'
            f"Here are some relevant ORPHA rare disease terms for reference that may help you find rare disease mentions in the sentence:\n\n"
            f"{context_text}\n\n"
            f"Based on this sentence and the provided rare disease terms as reference, extract all medically relevant conditions "
            f"that are NOT negated (i.e., NOT preceded by 'no', 'not', 'without', 'ruled out', etc.). "
            f"Please also include any potential abbreviations that might be referring to rare diseases in the CLINICAL TEXT."
            f"\n\nReturn only a Python list of strings, with each disease exactly as it appears in the CLINICAL TEXT. "
            f"Ensure the output is concise without any additional notes, commentary, or meta explanations."
        )
        return prompt

    def _merge_small_sentences(self, sentences: List[str], min_size: int) -> List[str]:
        """
        Merge sentences smaller than the minimum size with subsequent sentences.

        Args:
            sentences: List of extracted sentences
            min_size: Minimum character length for a sentence

        Returns:
            List of merged sentences meeting the minimum size requirement
        """
        if not sentences:
            return []

        if min_size is None or min_size <= 0:
            return sentences

        merged_sentences = []
        current_idx = 0

        while current_idx < len(sentences):
            current_sentence = sentences[current_idx]

            # If the current sentence is already large enough, add it directly
            if len(current_sentence) >= min_size:
                merged_sentences.append(current_sentence)
                current_idx += 1
                continue

            # Start merging with next sentences until we reach min_size
            merged_chunk = current_sentence
            next_idx = current_idx + 1

            while next_idx < len(sentences) and len(merged_chunk) < min_size:
                # Add the next sentence to our chunk with a space
                if merged_chunk and sentences[next_idx]:
                    merged_chunk += " " + sentences[next_idx]
                else:
                    merged_chunk += sentences[next_idx]
                next_idx += 1

            # Add the merged chunk to our results
            merged_sentences.append(merged_chunk)

            # Update the index to continue after the merged sentences
            current_idx = next_idx

        return merged_sentences

    def extract_entities(self, text: str) -> List[str]:
        """
        Extract rare disease mentions from text using retrieval-enhanced prompting.
        With sentence merging for efficiency when min_sentence_size is set.

        Args:
            text: Clinical text to extract rare disease mentions from

        Returns:
            List of extracted rare disease mentions
        """
        # Split text into sentences
        original_sentences = self.context_extractor.extract_sentences(text)

        # Merge small sentences if min_sentence_size is set
        if self.min_sentence_size:
            sentences = self._merge_small_sentences(
                original_sentences, self.min_sentence_size
            )
            print(
                f"After merging: Processing {len(sentences)} chunks instead of {len(original_sentences)} raw sentences"
            )
        else:
            sentences = original_sentences

        all_entities = []

        for sentence in sentences:
            # Skip empty or very short sentences
            if not sentence or len(sentence) < 5:
                continue

            # Retrieve candidates for this sentence/chunk
            candidates = self._retrieve_candidates(sentence)

            # Create enhanced prompt
            prompt = self._create_enhanced_prompt(sentence, candidates)

            # Query LLM
            findings_text = self.llm_client.query(prompt, self.system_message)
            # Extract entities from response
            entities = self._extract_findings_from_response(findings_text)
            # Add to results
            all_entities.extend(entities)

        # Remove duplicates while preserving order
        unique_entities = []
        seen = set()
        for entity in all_entities:
            entity_lower = entity.lower()
            if entity_lower not in seen and entity:
                seen.add(entity_lower)
                unique_entities.append(entity)

        return unique_entities

    def _extract_findings_from_response(self, response: str) -> List[str]:
        """
        Parse LLM response to extract findings.

        Args:
            response: Raw LLM response text

        Returns:
            List of extracted entities
        """
        try:
            # Extract content between square brackets if present
            if "[" in response and "]" in response:
                response = response[response.find("[") + 1 : response.rfind("]")]

            # Split on commas and clean up each term
            findings = []
            for term in response.split(","):
                cleaned_term = term.strip().strip("\"'")
                if cleaned_term:
                    findings.append(cleaned_term)

            return findings
        except Exception as e:
            print(f"Error parsing LLM response: {str(e)}")
            return []

    def process_batch(self, texts: List[str]) -> List[List[str]]:
        """
        Process a batch of texts for rare disease extraction.

        Args:
            texts: List of clinical texts to process

        Returns:
            List of lists containing extracted rare disease mentions for each text
        """
        results = []
        for text in texts:
            entities = self.extract_entities(text)
            results.append(entities)
        return results


class IterativeLLMRDExtractor(BaseRDExtractor):
    """Rare disease extraction pipeline using iterative LLM passes.

    Performs multiple passes with the LLM, accumulating unique entities across passes.
    Stops early if no new entities are found or after reaching the maximum number of iterations.
    """

    def __init__(self, llm_client, system_message: str, max_iterations: int = 3):
        """Initialize the iterative LLM rare disease extractor.

        Args:
            llm_client: LLM client for querying the language model
            system_message: System message to use for all LLM queries
            max_iterations: Maximum number of iterations to perform
        """
        self.llm_client = llm_client
        self.system_message = system_message
        self.max_iterations = max_iterations

    def extract_entities(self, text: str) -> List[str]:
        """Extract rare disease mentions using iterative LLM passes.

        Args:
            text: Input clinical text

        Returns:
            List of extracted rare disease mentions after iterative refinement
        """
        # Initialize with empty set of entities
        current_entities = set()

        for iteration in range(self.max_iterations):
            # Create prompt with already extracted entities
            if iteration == 0:
                # First pass - no previous entities
                prompt = f"""Extract all diseases and conditions that are NOT negated (i.e., don't include terms that are preceded by 'no', 'not', 'without', etc.) from the text below.

                Text: {text}

                Return only a Python list of strings, with each term exactly as it appears in the text. 
                Focus especially on rare diseases and genetic disorders.
                Ensure the output is concise without any additional notes, commentary, or meta explanations."""
            else:
                # Subsequent passes - include previously extracted entities
                already_extracted = ", ".join(sorted(current_entities))
                prompt = self._create_iterative_prompt(
                    text, already_extracted, iteration
                )

            # Query LLM
            findings_text = self.llm_client.query(prompt, self.system_message)

            # Extract entities from this pass
            iteration_entities = set(
                self._extract_findings_from_response(findings_text)
            )

            # Combine with previous entities (union)
            combined_entities = current_entities.union(iteration_entities)

            # Check if we've converged (no new entities)
            if combined_entities == current_entities:
                print(
                    f"Early stopping at iteration {iteration+1}: No new rare disease entities found."
                )
                break

            # Calculate new entities found in this iteration
            new_entities = combined_entities - current_entities
            if new_entities:
                print(
                    f"Iteration {iteration+1}: Found {len(new_entities)} new rare disease entities"
                )
            else:
                print(f"Iteration {iteration+1}: No new rare disease entities found.")

            # Update current entities
            current_entities = combined_entities

        # Convert set back to list for return
        return list(current_entities)

    def _create_iterative_prompt(
        self, original_text: str, already_extracted: str, iteration: int
    ) -> str:
        """Create a prompt for iterative rare disease extraction.

        Args:
            original_text: The original clinical text
            already_extracted: Comma-separated list of already extracted entities
            iteration: Current iteration number (0-based)

        Returns:
            Formatted prompt for the current iteration
        """
        return (
            f"Text: {original_text}\n\n"
            f"I have already extracted the following rare diseases and conditions: {already_extracted}.\n"
            f"Please examine the clinical text again carefully (iteration {iteration+1} of up to {self.max_iterations}) "
            f"and identify any additional diseases and conditions that are NOT negated and were missed in the previous extractions. "
            f"Find terms that aren't in the already extracted list, focusing especially on rare diseases and genetic disorders.\n\n"
            f"Return only a Python list of strings, with each term exactly as it appears in the text. "
            f"Ensure the output is concise without any additional notes, commentary, or meta explanations."
        )

    def process_batch(self, texts: List[str]) -> List[List[str]]:
        """Process a batch of texts for rare disease extraction.

        Args:
            texts: List of clinical texts to process

        Returns:
            List of lists containing extracted rare disease mentions for each text
        """
        results = []
        for text in texts:
            entities = self.extract_entities(text)
            results.append(entities)
        return results

    def _extract_findings_from_response(self, response: str) -> List[str]:
        """
        Parse LLM response to extract findings.

        Args:
            response: Raw LLM response text

        Returns:
            List of extracted entities
        """
        try:
            # Extract content between square brackets if present
            if "[" in response and "]" in response:
                response = response[response.find("[") + 1 : response.rfind("]")]

            # Split on commas and clean up each term
            findings = []
            for term in response.split(","):
                cleaned_term = term.strip().strip("\"'")
                if cleaned_term:
                    findings.append(cleaned_term)

            return findings
        except Exception as e:
            print(f"Error parsing LLM response: {str(e)}")
            return []


class MultiIterativeRDExtractor(BaseRDExtractor):
    """Rare disease extraction using multiple iterations of the LLM with different temperatures.

    This extractor runs the IterativeLLMRDExtractor multiple times with different
    temperature settings and aggregates the results using one of three methods:
    - union: Combines all unique entities found across all runs
    - intersection: Only keeps entities that appear in all runs
    - hybrid: Keeps entities that appear in at least N runs (threshold-based)

    This approach provides more stable and consistent entity extraction through
    ensemble methods, helping to reduce variance in extraction results.
    """

    def __init__(
        self,
        llm_client,
        system_message: str,
        temperatures: List[float] = [0.1, 0.3, 0.7],
        max_iterations: int = 2,
        aggregation_type: str = "hybrid",
        hybrid_threshold: int = 2,
    ):
        """Initialize the multi-iterative rare disease extractor.

        Args:
            llm_client: LLM client for querying the language model
            system_message: System message to use for all LLM queries
            temperatures: List of temperature values to use for different runs
            max_iterations: Maximum number of iterations for each IterativeLLMRDExtractor
            aggregation_type: Method to aggregate results ("union", "intersection", or "hybrid")
            hybrid_threshold: Minimum number of runs an entity must appear in for hybrid aggregation
        """
        self.llm_client = llm_client
        self.system_message = system_message
        self.temperatures = temperatures
        self.max_iterations = max_iterations
        self.aggregation_type = aggregation_type
        self.hybrid_threshold = hybrid_threshold

        # Validate aggregation type
        if aggregation_type not in ["union", "intersection", "hybrid"]:
            raise ValueError(
                f"Invalid aggregation_type: {aggregation_type}. "
                f"Must be one of: 'union', 'intersection', 'hybrid'"
            )

        # Initialize empty cache for optimization
        self._cache = {}
        # Store original temperature to restore after extraction
        self._original_temperature = getattr(llm_client, "temperature", 0.7)

    def extract_entities(self, text: str) -> List[str]:
        """Extract rare disease mentions using multiple iterations with different temperatures.

        Args:
            text: Input clinical text

        Returns:
            List of extracted rare disease mentions after aggregation
        """
        # Check cache first for performance optimization
        cache_key = f"{hash(text)}_{self.aggregation_type}_{self.hybrid_threshold}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Store results from each temperature run
        run_results = []
        original_temperature = getattr(
            self.llm_client, "temperature", self._original_temperature
        )

        try:
            # Run extraction with each temperature
            for temp in self.temperatures:
                # Set temperature for this run
                self.llm_client.temperature = temp

                # Create iterative extractor for this run
                iterative_extractor = IterativeLLMRDExtractor(
                    self.llm_client,
                    self.system_message,
                    max_iterations=self.max_iterations,
                )

                # Extract entities
                entities = iterative_extractor.extract_entities(text)
                run_results.append(entities)

            # Aggregate results based on specified method
            aggregated = self._aggregate_results(run_results)

            # Cache and return results
            self._cache[cache_key] = aggregated
            return aggregated

        finally:
            # Restore original temperature
            self.llm_client.temperature = original_temperature

    def process_batch(self, texts: List[str]) -> List[List[str]]:
        """Process a batch of texts for rare disease extraction.

        Args:
            texts: List of clinical texts to process

        Returns:
            List of lists containing extracted rare disease mentions for each text
        """
        results = []
        for text in texts:
            entities = self.extract_entities(text)
            results.append(entities)
        return results

    def _aggregate_results(self, run_results: List[List[str]]) -> List[str]:
        """Aggregate results from multiple runs based on the specified method.

        Args:
            run_results: List of entity lists from each run

        Returns:
            Aggregated list of entities
        """
        if not run_results:
            return []

        if self.aggregation_type == "union":
            # Union: Combine all unique entities
            union_results = []
            for entities in run_results:
                for entity in entities:
                    if entity not in union_results:
                        union_results.append(entity)
            return union_results

        elif self.aggregation_type == "intersection":
            # Intersection: Keep only entities that appear in all runs
            # Convert all runs to sets and compute intersection
            if not all(run_results):  # If any run has no results
                return []

            runs_as_sets = [set(entities) for entities in run_results]
            if not runs_as_sets:
                return []

            intersection = set.intersection(*runs_as_sets)
            return list(intersection)

        elif self.aggregation_type == "hybrid":
            # Hybrid: Keep entities that appear in at least N runs
            entity_counts = {}
            for entities in run_results:
                for entity in entities:
                    entity_counts[entity] = entity_counts.get(entity, 0) + 1

            # Filter by threshold
            hybrid_results = [
                entity
                for entity, count in entity_counts.items()
                if count >= self.hybrid_threshold
            ]
            return hybrid_results

        # Should never reach here due to validation in __init__
        return []
