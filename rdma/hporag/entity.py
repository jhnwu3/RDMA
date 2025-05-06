from abc import ABC, abstractmethod
import json
import pandas as pd
import numpy as np
import re
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from fuzzywuzzy import fuzz
# classes for extracting entities 
class BaseEntityExtractor(ABC):
    """Abstract base class for entity extraction pipelines."""
    
    @abstractmethod
    def extract_entities(self, text: str) -> List[str]:
        """Extract entities from given text."""
        pass

    @abstractmethod
    def process_batch(self, texts: List[str]) -> List[List[str]]:
        """Process a batch of texts for entity extraction."""
        pass


class LLMEntityExtractor(BaseEntityExtractor):
    """Entity extraction pipeline using LLM."""
    
    def __init__(self, llm_client, system_message: str):
        self.llm_client = llm_client
        self.system_message = system_message

    def extract_entities(self, text: str) -> List[str]:
        findings_text = self.llm_client.query(text, self.system_message)
        if not findings_text:
            return []
        return self._extract_findings_from_response(findings_text)

    def process_batch(self, texts: List[str]) -> List[List[str]]:
        results = []
        for text in texts:
            entities = self.extract_entities(text)
            results.append(entities)
        return results

    def _extract_findings_from_response(self, response_content: str) -> List[str]:
        sanitized = response_content.replace("```", "").strip()
        start = sanitized.find("{")
        end = sanitized.rfind("}")
        if start == -1 or end == -1 or start > end:
            return []

        json_str = sanitized[start:end+1]
        try:
            data = json.loads(json_str)
            findings = data.get("findings", [])
            return findings if isinstance(findings, list) else []
        except json.JSONDecodeError:
            return []

class IterativeLLMEntityExtractor(BaseEntityExtractor):
    """Entity extraction pipeline using iterative LLM passes.
    
    Performs multiple passes with the LLM, accumulating unique entities across passes.
    Stops early if no new entities are found or after reaching the maximum number of iterations.
    """
    
    def __init__(self, llm_client, system_message: str, max_iterations: int = 3):
        """Initialize the iterative LLM entity extractor.
        
        Args:
            llm_client: LLM client for querying the language model
            system_message: System message to use for all LLM queries
            max_iterations: Maximum number of iterations to perform
        """
        self.llm_client = llm_client
        self.system_message = system_message
        self.max_iterations = max_iterations

    def extract_entities(self, text: str) -> List[str]:
        """Extract entities from text using iterative LLM passes.
        
        Args:
            text: Input clinical text
            
        Returns:
            List of extracted entities after iterative refinement
        """
        # Initialize with empty set of entities
        current_entities = set()
        
        for iteration in range(self.max_iterations):
            # Create prompt with already extracted entities
            if iteration == 0:
                # First pass - no previous entities
                prompt = text
            else:
                # Subsequent passes - include previously extracted entities
                already_extracted = ", ".join(sorted(current_entities))
                prompt = self._create_iterative_prompt(text, already_extracted, iteration)
            
            # Query LLM
            findings_text = self.llm_client.query(prompt, self.system_message)
            # print("Findings text:")
            # print(findings_text)
            # print("----------------")
            if not findings_text:
                break
                
            # Extract entities from this pass
            iteration_entities = set(self._extract_findings_from_response(findings_text))
            
            # Combine with previous entities (union)
            combined_entities = current_entities.union(iteration_entities)
            print("Combined entities:", combined_entities)
            # Check if we've converged (no new entities)
            if combined_entities == current_entities:
                print(f"Early stopping at iteration {iteration+1}: No new entities found.")
                break
                
            # Calculate new entities found in this iteration
            new_entities = combined_entities - current_entities
            if new_entities:
                print(f"Iteration {iteration+1}: Found {len(new_entities)} new entities")
            else:
                print(f"Iteration {iteration+1}: No new entities found.")
                
            # Update current entities
            current_entities = combined_entities
        
        # Convert set back to list for return
        return list(current_entities)

    def _create_iterative_prompt(self, original_text: str, already_extracted: str, iteration: int) -> str:
        """Create a prompt for iterative entity extraction.
        
        Args:
            original_text: The original clinical text
            already_extracted: Comma-separated list of already extracted entities
            iteration: Current iteration number (0-based)
            
        Returns:
            Formatted prompt for the current iteration
        """
        return (
            f"{original_text}\n\n"
            f"I have already extracted the following terms: {already_extracted}.\n"
            f"Please examine the clinical text again carefully (iteration {iteration+1} of up to {self.max_iterations}) "
            f"and identify any additional genetic inheritance patterns, anatomical anomalies, clinical symptoms, diagnostic findings, lab test results, and specific conditions or syndromess that were missed in the previous extractions. "
            f"Find terms that aren't in the already extracted list. Include appropriate context based only on the passage."
            f"Return the extracted terms in a JSON object with a single key 'findings', which contains the list of extracted terms spelled correctly."
        )

    def process_batch(self, texts: List[str]) -> List[List[str]]:
        """Process a batch of texts for entity extraction.
        
        Args:
            texts: List of clinical texts to process
            
        Returns:
            List of lists containing extracted entities for each text
        """
        results = []
        for text in texts:
            entities = self.extract_entities(text)
            results.append(entities)
        return results

    def _extract_findings_from_response(self, response_content: str) -> List[str]:
        """Extract findings from LLM response.
        
        Args:
            response_content: Raw LLM response text
            
        Returns:
            List of extracted entities
        """
        sanitized = response_content.replace("```", "").strip()
        start = sanitized.find("{")
        end = sanitized.rfind("}")
        if start == -1 or end == -1 or start > end:
            return []

        json_str = sanitized[start:end+1]
        try:
            data = json.loads(json_str)
            findings = data.get("findings", [])
            return findings if isinstance(findings, list) else []
        except json.JSONDecodeError:
            return []


class MultiIterativeExtractor(BaseEntityExtractor):
    """Entity extraction using multiple iterations of the LLM with different temperatures.
    
    This extractor runs the IterativeLLMEntityExtractor multiple times with different
    temperature settings and aggregates the results using one of three methods:
    - union: Combines all unique entities found across all runs
    - intersection: Only keeps entities that appear in all runs
    - hybrid: Keeps entities that appear in at least N runs (threshold-based)
    
    This approach provides more stable and consistent entity extraction through
    ensemble methods, helping to reduce variance in extraction results.
    """
    
    def __init__(self, llm_client, system_message: str, 
                 temperatures: List[float] = [0.1, 0.3, 0.7], 
                 max_iterations: int = 2,
                 aggregation_type: str = "hybrid",
                 hybrid_threshold: int = 2):
        """Initialize the multi-iterative entity extractor.
        
        Args:
            llm_client: LLM client for querying the language model
            system_message: System message to use for all LLM queries
            temperatures: List of temperature values to use for different runs
            max_iterations: Maximum number of iterations for each IterativeLLMEntityExtractor
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
            raise ValueError(f"Invalid aggregation_type: {aggregation_type}. "
                            f"Must be one of: 'union', 'intersection', 'hybrid'")
        
        # Initialize empty cache for optimization
        self._cache = {}
        # Store original temperature to restore after extraction
        self._original_temperature = getattr(llm_client, 'temperature', 0.7)
    
    def extract_entities(self, text: str) -> List[str]:
        """Extract entities from text using multiple iterations with different temperatures.
        
        Args:
            text: Input clinical text
            
        Returns:
            List of extracted entities after aggregation
        """
        # Check cache first for performance optimization
        cache_key = f"{hash(text)}_{self.aggregation_type}_{self.hybrid_threshold}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Store results from each temperature run
        run_results = []
        original_temperature = getattr(self.llm_client, 'temperature', self._original_temperature)
        
        try:
            # Run extraction with each temperature
            for temp in self.temperatures:
                # Set temperature for this run
                self.llm_client.temperature = temp
                
                # Create iterative extractor for this run
                iterative_extractor = IterativeLLMEntityExtractor(
                    self.llm_client,
                    self.system_message,
                    max_iterations=self.max_iterations
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
        """Process a batch of texts for entity extraction.
        
        Args:
            texts: List of clinical texts to process
            
        Returns:
            List of lists containing extracted entities for each text
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
            hybrid_results = [entity for entity, count in entity_counts.items() 
                             if count >= self.hybrid_threshold]
            return hybrid_results
            
        # Should never reach here due to validation in __init__
        return []

# we keep this for now.
from typing import List, Dict, Any
import json
import numpy as np
# from hporag.entity import BaseEntityExtractor
# from hporag.context import ContextExtractor

# class RetrievalEnhancedEntityExtractorOld(BaseEntityExtractor):
#     """
#     Entity extractor that uses embedding retrieval to enhance LLM-based extraction.
    
#     For each sentence:
#     1. Retrieves relevant HPO terms to provide context
#     2. Uses this context to prompt the LLM for more accurate entity extraction
    
#     This approach helps the LLM with domain knowledge before extraction begins.
#     """
    
#     def __init__(self, llm_client, embedding_manager, embedded_documents, system_message: str, top_k: int = 10):
#         """
#         Initialize the retrieval-enhanced entity extractor.
        
#         Args:
#             llm_client: Client for querying the language model
#             embedding_manager: Manager for vector embeddings and search
#             embedded_documents: HPO terms with embeddings
#             system_message: System message for LLM extraction
#             top_k: Number of top candidates to retrieve per sentence
#         """
#         self.llm_client = llm_client
#         self.embedding_manager = embedding_manager
#         self.embedded_documents = embedded_documents
#         self.index = None
#         self.system_message = system_message
#         self.top_k = top_k
#         self.context_extractor = ContextExtractor()
        
#     def prepare_index(self):
#         """Prepare FAISS index from embedded documents if not already prepared."""
#         if self.index is None:
#             embeddings_array = self.embedding_manager.prepare_embeddings(self.embedded_documents)
#             self.index = self.embedding_manager.create_index(embeddings_array)
    
#     def _retrieve_candidates(self, sentence: str) -> List[Dict]:
#         """
#         Retrieve relevant HPO candidates for a sentence.
        
#         Args:
#             sentence: Clinical sentence to find relevant HPO terms for
            
#         Returns:
#             List of dictionaries with HPO term information and similarity scores
#         """
#         self.prepare_index()
        
#         # Embed the query
#         query_vector = self.embedding_manager.query_text(sentence).reshape(1, -1)
        
#         # Search for similar items
#         distances, indices = self.embedding_manager.search(query_vector, self.index, k=min(800, len(self.embedded_documents)))
        
#         # Extract unique metadata
#         candidates = []
#         seen_metadata = set()
        
#         for idx, distance in zip(indices[0], distances[0]):
#             metadata = self.embedded_documents[idx]['unique_metadata']
#             metadata_str = json.dumps(metadata)
            
#             if metadata_str not in seen_metadata:
#                 seen_metadata.add(metadata_str)
#                 candidates.append({
#                     'term': metadata.get('info', ''),
#                     'hp_id': metadata.get('hp_id', ''),
#                     'similarity_score': 1.0 / (1.0 + distance)  # Convert distance to similarity
#                 })
                
#                 if len(candidates) >= self.top_k:
#                     break
                    
#         return candidates
    
#     def _create_enhanced_prompt(self, sentence: str, candidates: List[Dict]) -> str:
#         """
#         Create a prompt enhanced with retrieved candidates.
        
#         Args:
#             sentence: Clinical sentence to extract entities from
#             candidates: Retrieved HPO candidates to use as context
            
#         Returns:
#             Formatted prompt for LLM
#         """
#         # Format candidates as context
#         context_items = []
#         for candidate in candidates:
#             context_items.append(f"- {candidate['term']} ({candidate['hp_id']})")
        
#         context_text = "\n".join(context_items)
        
#         # Create the enhanced prompt
#         prompt = (
#             f"I have a clinical sentence: \"{sentence}\"\n\n"
#             f"Here are some relevant HPO terms for context that are potentially within the sentence:\n\n"
#             f"{context_text}\n\n"
#             f"Based on this sentence and the provided HPO terms for context, extract all phenotype terms "
#             f"(genetic inheritance patterns, anatomical anomalies, clinical symptoms, diagnostic findings, "
#             f"test results, conditions or syndromes) from the sentence. "
#             f"IGNORE NEGATIVE FINDINGS, NORMAL FINDINGS, AND ANY TERMS MENTIONED IN FAMILY HISTORY.\n\n"
#             f"Please include the full term and any additional context that is part of the term. "
#             f"MAKE SURE IT MATCHES EXACTLY AS IT APPEARS IN THE SENTENCE.\n"
#             f"Return the extracted terms as a JSON object with a single key 'findings', which contains "
#             f"the list of extracted terms spelled correctly."
#         )
        
#         return prompt
    
#     def extract_entities(self, text: str) -> List[str]:
#         """
#         Extract entities from text using retrieval-enhanced prompting.
        
#         Args:
#             text: Clinical text to extract entities from
            
#         Returns:
#             List of extracted phenotype entities
#         """
#         # Split text into sentences
#         sentences = self.context_extractor.extract_sentences(text)
        
#         all_entities = []
        
#         for sentence in sentences:
#             # Skip empty or very short sentences
#             if not sentence or len(sentence) < 5:
#                 continue
                
#             # Retrieve candidates for this sentence
#             candidates = self._retrieve_candidates(sentence)
            
#             # Create enhanced prompt
#             prompt = self._create_enhanced_prompt(sentence, candidates)
            
#             # Query LLM
#             findings_text = self.llm_client.query(prompt, self.system_message)
#             # print("Prompt")
#             # print(prompt)
#             # print("Findings Text")
#             # print(findings_text)
#             # print()

#             # Extract entities from response
#             entities = self._extract_findings_from_response(findings_text)
            
#             # Add to results
#             all_entities.extend(entities)
        
#         # Remove duplicates while preserving order
#         unique_entities = []
#         seen = set()
#         for entity in all_entities:
#             entity_lower = entity.lower()
#             if entity_lower not in seen and entity:
#                 seen.add(entity_lower)
#                 unique_entities.append(entity)
        
#         return unique_entities
    
#     def _extract_findings_from_response(self, response_content: str) -> List[str]:
#         """
#         Extract findings from LLM response.
        
#         Args:
#             response_content: Raw LLM response text
            
#         Returns:
#             List of extracted entities
#         """
#         sanitized = response_content.replace("```", "").strip()
#         start = sanitized.find("{")
#         end = sanitized.rfind("}")
#         if start == -1 or end == -1 or start > end:
#             return []

#         json_str = sanitized[start:end+1]
#         try:
#             data = json.loads(json_str)
#             findings = data.get("findings", [])
#             return findings if isinstance(findings, list) else []
#         except json.JSONDecodeError:
#             return []
    
#     def process_batch(self, texts: List[str]) -> List[List[str]]:
#         """
#         Process a batch of texts for entity extraction.
        
#         Args:
#             texts: List of clinical texts to process
            
#         Returns:
#             List of lists containing extracted entities for each text
#         """
#         results = []
#         for text in texts:
#             entities = self.extract_entities(text)
#             results.append(entities)
#         return results
    

import gc
import torch
import json
from typing import List, Dict, Tuple, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from rdma.hporag.context import ContextExtractor
class RetrievalEnhancedEntityExtractor(BaseEntityExtractor):
    """
    Entity extractor that uses embedding retrieval to enhance LLM-based extraction.
    
    For each sentence:
    1. Retrieves relevant HPO terms to provide context
    2. Uses this context to prompt the LLM for more accurate entity extraction
    
    This approach helps the LLM with domain knowledge before extraction begins.
    Uses batched processing when there are more than min_batch_size pairs to process,
    but limits each batch to max_batch_size pairs to avoid memory issues.
    """
    
    def __init__(self, llm_client, embedding_manager, embedded_documents, system_message: str, 
                 top_k: int = 10, min_batch_size: int = 32, max_batch_size: int = 256,
                 manage_gpu_memory: bool = True):
        """
        Initialize the retrieval-enhanced entity extractor.
        
        Args:
            llm_client: Client for querying the language model
            embedding_manager: Manager for vector embeddings and search
            embedded_documents: HPO terms with embeddings
            system_message: System message for LLM extraction
            top_k: Number of top candidates to retrieve per sentence
            min_batch_size: Minimum number of pairs to trigger batched processing
            max_batch_size: Maximum number of pairs to process in a single batch
            manage_gpu_memory: Whether to actively manage GPU memory with cleanup operations
        """
        self.llm_client = llm_client
        self.embedding_manager = embedding_manager
        self.embedded_documents = embedded_documents
        self.index = None
        self.system_message = system_message
        self.top_k = top_k
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.manage_gpu_memory = manage_gpu_memory
        self.context_extractor = ContextExtractor()
        
        # For memory management
        self._is_model_on_cpu = False
        self._orig_device = None
        
    def prepare_index(self):
        """Prepare FAISS index from embedded documents if not already prepared."""
        if self.index is None:
            with torch.no_grad():  # Use no_grad to save memory
                embeddings_array = self.embedding_manager.prepare_embeddings(self.embedded_documents)
                self.index = self.embedding_manager.create_index(embeddings_array)
                
                # Clean up after index creation, as this can be memory-intensive
                if self.manage_gpu_memory and torch.cuda.is_available():
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()
                    
    def _move_embedding_model_to_cpu(self):
        """Move the embedding model to CPU to free GPU memory for LLM inference."""
        if not self._is_model_on_cpu and hasattr(self.embedding_manager, 'model'):
            model = self.embedding_manager.model
            
            if hasattr(model, 'device'):
                self._orig_device = model.device
                
            if hasattr(model, 'to'):
                model.to('cpu')
                self._is_model_on_cpu = True
                
                # Force memory cleanup
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
    def _restore_embedding_model_device(self):
        """Move the embedding model back to its original device."""
        if self._is_model_on_cpu and hasattr(self.embedding_manager, 'model'):
            model = self.embedding_manager.model
            
            if hasattr(model, 'to') and self._orig_device is not None:
                model.to(self._orig_device)
                
            self._is_model_on_cpu = False
    
    def _retrieve_candidates(self, sentence: str) -> List[Dict]:
        """
        Retrieve relevant HPO candidates for a sentence.
        With memory management to release GPU memory after retrieval.
        
        Args:
            sentence: Clinical sentence to find relevant HPO terms for
            
        Returns:
            List of dictionaries with HPO term information and similarity scores
        """
        self.prepare_index()
        
        # Use no_grad for embedding and search operations
        with torch.no_grad():
            try:
                # Embed the query
                query_vector = self.embedding_manager.query_text(sentence).reshape(1, -1)
                
                # Make a CPU copy of the query vector for search
                query_vector_cpu = query_vector.cpu().numpy() if isinstance(query_vector, torch.Tensor) else query_vector
                
                # Search for similar items
                distances, indices = self.embedding_manager.search(query_vector_cpu, self.index, k=min(800, len(self.embedded_documents)))
                
                # Extract unique metadata
                candidates = []
                seen_metadata = set()
                
                for idx, distance in zip(indices[0], distances[0]):
                    metadata = self.embedded_documents[idx]['unique_metadata']
                    metadata_str = json.dumps(metadata)
                    
                    if metadata_str not in seen_metadata:
                        seen_metadata.add(metadata_str)
                        candidates.append({
                            'term': metadata.get('info', ''),
                            'hp_id': metadata.get('hp_id', ''),
                            'similarity_score': 1.0 / (1.0 + distance)  # Convert distance to similarity
                        })
                        
                        if len(candidates) >= self.top_k:
                            break
                
                return candidates
            finally:
                # Explicitly delete tensors
                if 'query_vector' in locals() and isinstance(query_vector, torch.Tensor):
                    del query_vector
                
                # Force garbage collection
                import gc
                gc.collect()
                
                # Clear CUDA cache to free up GPU memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    
    def _create_enhanced_prompt(self, sentence: str, candidates: List[Dict]) -> str:
        """
        Create a prompt enhanced with retrieved candidates.
        
        Args:
            sentence: Clinical sentence to extract entities from
            candidates: Retrieved HPO candidates to use as context
            
        Returns:
            Formatted prompt for LLM
        """
        # Format candidates as context
        context_items = []
        for candidate in candidates:
            context_items.append(f"- {candidate['term']} ({candidate['hp_id']})")
        
        context_text = "\n".join(context_items)
        
        # Create the enhanced prompt
        prompt = (
            f"I have a clinical sentence: \"{sentence}\"\n\n"
            f"Here are some relevant HPO terms for context that are potentially within the sentence:\n\n"
            f"{context_text}\n\n"
            f"Based on this sentence and the provided HPO terms for context, extract all phenotype terms "
            f"(genetic inheritance patterns, anatomical anomalies, clinical symptoms, diagnostic findings, "
            f"test results, conditions or syndromes) from the sentence. "
            f"IGNORE NEGATIVE FINDINGS, NORMAL FINDINGS, AND ANY TERMS MENTIONED IN FAMILY HISTORY.\n\n"
            f"Please include the full term and any additional context that is part of the term. "
            f"MAKE SURE IT MATCHES EXACTLY AS IT APPEARS IN THE SENTENCE.\n"
            f"Return the extracted terms as a JSON object with a single key 'findings', which contains "
            f"the list of extracted terms spelled correctly."
        )
        
        return prompt
    
    def extract_entities(self, text: str) -> List[str]:
        """
        Extract entities from text using retrieval-enhanced prompting.
        Uses batched processing only when there are more than the threshold number of pairs.
        Implements memory management to prevent GPU memory leaks.
        
        Args:
            text: Clinical text to extract entities from
            
        Returns:
            List of extracted phenotype entities
        """
        # Split text into sentences
        sentences = self.context_extractor.extract_sentences(text)
        
        # Filter out empty or very short sentences
        valid_sentences = [s for s in sentences if s and len(s) >= 5]
        
        if not valid_sentences:
            return []
        
        # Determine batch size for sentence processing to prevent OOM
        sentence_retrieve_batch_size = min(50, len(valid_sentences))  # Process at most 50 sentences at a time for retrieval
        sentence_candidate_pairs = []
        
        # Process sentences in batches to conserve memory
        for i in range(0, len(valid_sentences), sentence_retrieve_batch_size):
            batch_sentences = valid_sentences[i:i + sentence_retrieve_batch_size]
            
            for sentence in batch_sentences:
                candidates = self._retrieve_candidates(sentence)
                sentence_candidate_pairs.append((sentence, candidates))
                
            # Force memory cleanup after each batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Check if we should use batched processing or not
        if len(sentence_candidate_pairs) > self.min_batch_size:
            # Use batched processing for many pairs
            return self._process_pairs_batched(sentence_candidate_pairs)
        else:
            # Use sequential processing for fewer pairs
            return self._process_pairs_sequential(sentence_candidate_pairs)
    
    def _process_pairs_sequential(self, sentence_candidate_pairs: List[Tuple[str, List[Dict]]]) -> List[str]:
        """
        Process (sentence, candidates) pairs using sequential LLM calls.
        
        Args:
            sentence_candidate_pairs: List of (sentence, candidates) pairs
            
        Returns:
            List of unique extracted entities
        """
        all_entities = []
        
        for sentence, candidates in sentence_candidate_pairs:
            # Create enhanced prompt
            prompt = self._create_enhanced_prompt(sentence, candidates)
            
            # Query LLM individually
            findings_text = self.llm_client.query(prompt, self.system_message)
            
            # Extract entities from response
            entities = self._extract_findings_from_response(findings_text)
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
    
    def _process_pairs_batched(self, sentence_candidate_pairs: List[Tuple[str, List[Dict]]]) -> List[str]:
        """
        Process (sentence, candidates) pairs using batched LLM calls.
        Limits each batch to max_batch_size to avoid memory issues.
        Implements active GPU memory management.
        
        Args:
            sentence_candidate_pairs: List of (sentence, candidates) pairs
            
        Returns:
            List of unique extracted entities
        """
        all_entities = []
        
        try:
            # If we're using SentenceTransformers, move to CPU to free GPU memory for LLM
            if self.manage_gpu_memory and hasattr(self.embedding_manager, 'model') and \
               hasattr(self.embedding_manager.model, 'to') and \
               isinstance(self.embedding_manager.model, SentenceTransformer):
                self._move_embedding_model_to_cpu()
            
            # Process in chunks of max_batch_size
            for i in range(0, len(sentence_candidate_pairs), self.max_batch_size):
                chunk = sentence_candidate_pairs[i:i + self.max_batch_size]
                
                # Create prompts for this chunk
                prompts = []
                for sentence, candidates in chunk:
                    prompt = self._create_enhanced_prompt(sentence, candidates)
                    prompts.append(prompt)
                
                # Create system messages for each prompt
                system_messages = [self.system_message] * len(prompts)
                
                # Process this chunk in batch
                with torch.no_grad():  # Use no_grad for LLM generation if it uses tensors
                    batch_responses = self.llm_client.batched_query(prompts, system_messages)
                
                # Extract entities from all responses in this chunk
                for response in batch_responses:
                    entities = self._extract_findings_from_response(response)
                    all_entities.extend(entities)
                
                # Force memory cleanup after each batch
                if self.manage_gpu_memory and torch.cuda.is_available():
                    # Free memory
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()
            
            # Remove duplicates while preserving order
            unique_entities = []
            seen = set()
            for entity in all_entities:
                entity_lower = entity.lower()
                if entity_lower not in seen and entity:
                    seen.add(entity_lower)
                    unique_entities.append(entity)
            
            return unique_entities
            
        finally:
            # Move embedding model back to GPU if needed
            if self._is_model_on_cpu:
                self._restore_embedding_model_device()
                
            # Final memory cleanup
            if self.manage_gpu_memory and torch.cuda.is_available():
                import gc
                gc.collect()
                torch.cuda.empty_cache()
    
    def _extract_findings_from_response(self, response_content: str) -> List[str]:
        """
        Extract findings from LLM response.
        
        Args:
            response_content: Raw LLM response text
            
        Returns:
            List of extracted entities
        """
        sanitized = response_content.replace("```", "").strip()
        start = sanitized.find("{")
        end = sanitized.rfind("}")
        if start == -1 or end == -1 or start > end:
            return []

        json_str = sanitized[start:end+1]
        try:
            data = json.loads(json_str)
            findings = data.get("findings", [])
            return findings if isinstance(findings, list) else []
        except json.JSONDecodeError:
            return []
    
    def process_batch(self, texts: List[str]) -> List[List[str]]:
        """
        Process a batch of texts for entity extraction.
        
        Args:
            texts: List of clinical texts to process
            
        Returns:
            List of lists containing extracted entities for each text
        """
        results = []
        for text in texts:
            entities = self.extract_entities(text)
            results.append(entities)
        return results
    

class StanzaEntityExtractor(BaseEntityExtractor):
    """Entity extraction using Stanza's i2b2 NER model.
    
    The i2b2 model identifies three main entity types:
    - PROBLEM: Medical conditions, symptoms, diagnoses (relevant for phenotypes)
    - TEST: Diagnostic procedures and lab tests
    - TREATMENT: Medications, therapies, procedures
    
    For phenotype extraction, we focus on PROBLEM entities as they represent
    observable characteristics and medical conditions.
    """
    def __init__(self, stanza_pipeline):
        self.pipeline = stanza_pipeline
        
    def extract_entities(self, text: str) -> List[str]:
        """Extract phenotype-relevant entities from text using Stanza i2b2 NER.
        
        Args:
            text: Input clinical text
            
        Returns:
            List of extracted entities that could represent phenotypes
        """
        doc = self.pipeline(text)
        
        # Extract only PROBLEM entities as they align with phenotypes
        entities = [ent.text for ent in doc.ents if ent.type == 'PROBLEM']
        
        # Optional: Add debug printing
        if entities:
            print(f"Found {len(entities)} PROBLEM entities: {entities}")
        
        return entities
        
    def process_batch(self, texts: List[str]) -> List[List[str]]:
        """Process a batch of texts for entity extraction.
        
        Args:
            texts: List of clinical texts to process
            
        Returns:
            List of lists containing extracted entities for each text
        """
        return [self.extract_entities(text) for text in texts]
    


# Add to hporag.entity.py
from typing import List, Dict, Optional, Union
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class NuExtractor(BaseEntityExtractor):
    """Entity extraction using the NuExtract model with structured schema output.
    
    NuExtract is a specialized extraction model that can extract structured data 
    based on a predefined schema. This implementation focuses on extracting
    phenotype-related entities from clinical text.
    """
    
    def __init__(self, 
                 model_name: str = "numind/NuExtract", 
                 device: str = "cuda:0",
                 max_length: int = 4000,
                 schema: Optional[str] = None,
                 examples: List[str] = None,
                 cache_dir: Optional[str] = None):
        """
        Initialize the NuExtract extractor.
        
        Args:
            model_name: Name of the NuExtract model to use
            device: Device to use for inference (e.g., "cuda:0", "cpu")
            max_length: Maximum token length for input
            schema: Custom schema for extraction (if None, uses default phenotype schema)
            examples: List of example extractions in JSON string format
            cache_dir: Directory to cache the model
        """
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.examples = examples or ["", "", ""]
        self.cache_dir = cache_dir
        
        # Default schema for phenotype extraction
        self.schema = schema or self._get_default_schema()
        
        # Initialize model
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the NuExtract model and tokenizer."""
        print(f"Initializing NuExtract model from {self.model_name}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, 
                trust_remote_code=False,
                cache_dir=self.cache_dir
            )
            
            # Load model with appropriate precision
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, 
                torch_dtype=torch.bfloat16, 
                trust_remote_code=False,
                cache_dir=self.cache_dir
            )
            
            # Move model to specified device
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            print(f"NuExtract model loaded successfully on {self.device}")
            
        except Exception as e:
            print(f"Error initializing NuExtract model: {e}")
            raise
    
    def _get_default_schema(self) -> str:
        """Get the default schema for comprehensive phenotype extraction."""
        schema = {
            "findings": [
                {
                    "mention": "",               # The exact text of the finding
                    "category": "",              # One of: "phenotype", "lab_test", "condition", "symptom", "anatomical_site", "genetic_variant"
                    "negated": False,            # Whether the finding is negated
                    "family_history": False,     # Whether the finding is mentioned in family history
                    "section": "",               # Section of the clinical note (if identifiable)
                    "severity": "",              # Optional severity (mild, moderate, severe)
                    "temporal_context": "",      # Optional temporal information (acute, chronic, etc.)
                    "anatomical_location": "",   # Optional anatomical context
                    "measurement": {             # Optional measurement information
                        "value": "",
                        "unit": "",
                        "reference_range": ""
                    }
                }
            ]
        }
        return json.dumps(schema)
        
    def extract_entities(self, text: str) -> List[str]:
        """
        Extract entities from text using the NuExtract model.
        
        Args:
            text: Clinical text to extract entities from
            
        Returns:
            List of extracted entities as strings
        """
        if not text:
            return []
            
        # Prepare the input
        schema_formatted = json.dumps(json.loads(self.schema), indent=4)
        input_text = "<|input|>\n### Template:\n" + schema_formatted + "\n"
        
        # Add examples if provided
        for example in self.examples:
            if example:
                input_text += "### Example:\n" + json.dumps(json.loads(example), indent=4) + "\n"
        
        # Add the actual text to process
        input_text += "### Text:\n" + text + "\n<|output|>\n"
        
        # Tokenize and generate
        try:
            input_ids = self.tokenizer(
                input_text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=self.max_length
            ).to(self.device)
            
            # Generate output
            with torch.no_grad():
                output_ids = self.model.generate(**input_ids)
                
            # Decode output
            output = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            
            # Extract the structured output
            extraction_text = output.split("<|output|>")[1].split("<|end-output|>")[0].strip()
            
            # Parse the extraction as JSON
            try:
                extraction = json.loads(extraction_text)
                # Extract only the mentions from findings
                return self._process_extraction(extraction)
            except json.JSONDecodeError as e:
                print(f"Error parsing NuExtract output as JSON: {e}")
                print(f"Raw output: {extraction_text}")
                return []
                
        except Exception as e:
            print(f"Error during NuExtract inference: {e}")
            return []
    
    def _process_extraction(self, extraction: Dict) -> List[str]:
        """
        Process the extracted JSON to get a list of entity strings.
        
        Args:
            extraction: Dictionary containing the extraction results
            
        Returns:
            List of extracted entity mentions
        """
        if not extraction or 'findings' not in extraction:
            return []
            
        # Filter out negated mentions and family history
        filtered_findings = [
            finding for finding in extraction['findings']
            if (finding.get('mention') and 
                not finding.get('negated', False) and 
                not finding.get('family_history', False))
        ]
        
        # Extract just the mentions
        return [finding['mention'] for finding in filtered_findings]
    
    def process_batch(self, texts: List[str]) -> List[List[str]]:
        """
        Process a batch of texts for entity extraction.
        
        Args:
            texts: List of clinical texts to process
            
        Returns:
            List of lists containing extracted entities for each text
        """
        results = []
        for text in texts:
            entities = self.extract_entities(text)
            results.append(entities)
        return results

from rdma.hporag.phenogpt import PhenoGPT

class PhenoGPTEntityExtractor(BaseEntityExtractor):
    """Entity extraction using PhenoGPT model."""
    
    def __init__(self, phenogpt_instance : PhenoGPT, custom_prompt=None):
        """
        Initialize with PhenoGPT model.
        
        Args:
            phenogpt_instance: Initialized PhenoGPT instance
            custom_prompt: Optional custom prompt for PhenoGPT
        """
        self.phenogpt = phenogpt_instance
        self.custom_prompt = custom_prompt
        
    def extract_entities(self, text: str) -> List[str]:
        """
        Extract phenotype entities using PhenoGPT.
        
        Args:
            text: Clinical text to process
            
        Returns:
            List of phenotype entities
        """
        # Use PhenoGPT to generate phenotypes
        phenotypes = self.phenogpt.generate(text, system_prompt=self.custom_prompt)
        
        # Filter out empty entities
        return [entity for entity in phenotypes if entity and entity.strip()]
        
    def process_batch(self, texts: List[str]) -> List[List[str]]:
        """
        Process multiple texts for entity extraction.
        
        Args:
            texts: List of clinical texts to process
            
        Returns:
            List of lists containing extracted entities for each text
        """
        results = []
        for text in texts:
            try:
                entities = self.extract_entities(text)
                results.append(entities)
            except Exception as e:
                print(f"Error processing text: {str(e)}")
                results.append([])
        return results