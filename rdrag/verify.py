import re
import json
import numpy as np
import torch
from datetime import datetime
from fuzzywuzzy import fuzz
from typing import List, Dict, Any, Optional, Tuple, Union, Set
import time
from utils.search_tools import ToolSearcher

class MultiStageRDVerifier:
    """
    Multistage rare disease verifier with optional abbreviation resolution.
    
    This verifier implements a streamlined workflow:
    1. Abbreviation resolution (if enabled)
    2. Direct rare disease verification with retrieved candidates
    
    Key features:
    - Resolves clinical abbreviations to full terms before verification (optional)
    - Focuses on rare disease verification rather than phenotype identification
    - Uses ORPHA code lookup for rare disease identification
    - Caches results for improved performance
    """
    
    def __init__(self, embedding_manager, llm_client, config=None, debug=False, 
                 abbreviations_file=None, use_abbreviations=True):
        """
        Initialize the multistage rare disease verifier.
        
        Args:
            embedding_manager: Manager for embedding operations
            llm_client: LLM client for verification
            config: Configuration for the verifier (currently unused, kept for API compatibility)
            debug: Enable debug output
            abbreviations_file: Path to abbreviations embeddings file
            use_abbreviations: Whether to use abbreviation resolution
        """
        self.embedding_manager = embedding_manager
        self.llm_client = llm_client
        self.debug = debug
        self.config = config  # Kept for API compatibility
        self.index = None
        self.embedded_documents = None
        self.use_abbreviations = use_abbreviations
        
        # Initialize abbreviation searcher if enabled
        self.abbreviations_file = abbreviations_file
        self.abbreviation_searcher = None
        if use_abbreviations and abbreviations_file:
            self.initialize_abbreviation_searcher()
        
        # Direct verification with binary matching
        self.direct_verification_system_message = (
            "You are a clinical expert specializing in rare disease identification. "
            "Your task is to determine if the given entity represents a rare disease based on the provided candidates."
            "\n\nA rare disease is typically defined as a condition that affects fewer than 1 in 2,000 people. "
            "Examples include Fabry disease, Gaucher disease, Pompe disease, etc."
            "\n\nAFTER REVIEWING THE CANDIDATES, respond with ONLY 'YES' if:"
            "\n- The entity EXACTLY or CLOSELY matches any rare disease candidate, OR"
            "\n- The entity clearly refers to a rare disease even if not in the candidates"
            "\n\nRespond with ONLY 'NO' if:"
            "\n- The entity is NOT a rare disease (e.g., common diseases, symptoms without specificity, lab tests)"
            "\n- The entity does not represent a specific rare condition"
            "\n\nYOUR RESPONSE MUST BE EXACTLY 'YES' OR 'NO' WITH NO ADDITIONAL TEXT."
        )
        
        # Caches for performance
        self.verification_cache = {}
        self.abbreviation_cache = {}
    
    def initialize_abbreviation_searcher(self):
        """Initialize the abbreviation searcher with embeddings file."""
        try:
            if not self.abbreviations_file:
                self._debug_print("No abbreviations file provided, abbreviation resolution disabled")
                return
                
            self._debug_print(f"Initializing abbreviation searcher with {self.abbreviations_file}")
            self.abbreviation_searcher = ToolSearcher(
                model_type=self.embedding_manager.model_type,
                model_name=self.embedding_manager.model_name,
                device="cpu",  # Use CPU for abbreviation searching to avoid GPU conflicts
                top_k=3  # Get top 3 matches for abbreviations
            )
            self.abbreviation_searcher.load_embeddings(self.abbreviations_file)
            self._debug_print("Abbreviation searcher initialized successfully")
        except Exception as e:
            self._debug_print(f"Error initializing abbreviation searcher: {e}")
            self.abbreviation_searcher = None
            self.use_abbreviations = False
    
    def _debug_print(self, message: str, level: int = 0):
        """Print debug messages if debug mode is enabled."""
        if self.debug:
            indent = "  " * level
            print(f"{datetime.now().strftime('%H:%M:%S')} | {indent}{message}")
    
    def prepare_index(self, metadata: List[Dict]):
        """Prepare FAISS index from metadata for similarity search."""
        if self.index is None:
            self._debug_print("Preparing FAISS index for rare disease verification...")
            embeddings_array = self.embedding_manager.prepare_embeddings(metadata)
            self.index = self.embedding_manager.create_index(embeddings_array)
            self.embedded_documents = metadata
            self._debug_print(f"Index prepared with {len(metadata)} embedded documents")

    def clear_caches(self):
        """Clear all caches to prepare for a fresh evaluation run."""
        self.verification_cache = {}
        self.abbreviation_cache = {}
        self._debug_print("All caches cleared")
    
    def preprocess_entity(self, entity: str) -> str:
        """
        Minimal preprocessing of entity text.
        
        Args:
            entity: Raw entity text
            
        Returns:
            Preprocessed entity text
        """
        if not entity:
            return ""
            
        # Remove unnecessary metadata patterns like "(resolved)"
        cleaned = re.sub(r'\s*\([^)]*\)', '', entity)
        
        # Remove extra whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        return cleaned
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        if not text:
            return ""
        
        # Convert to lowercase
        normalized = text.lower()
        
        # Remove punctuation except hyphens
        normalized = re.sub(r'[^\w\s-]', '', normalized)
        
        # Replace multiple spaces with a single space
        normalized = re.sub(r'\s+', ' ', normalized)
        
        return normalized.strip()
    
    def _retrieve_similar_diseases(self, entity: str, k: int = 20) -> List[Dict]:
        """Retrieve similar rare diseases from the embeddings."""
        if self.index is None:
            raise ValueError("Index not prepared. Call prepare_index() first.")
            
        # Embed the query
        query_vector = self.embedding_manager.query_text(entity).reshape(1, -1)
        
        # Search for similar items
        distances, indices = self.embedding_manager.search(query_vector, self.index, k=min(800, len(self.embedded_documents)))
        
        # Extract unique metadata
        similar_diseases = []
        seen_metadata = set()
        
        for idx, distance in zip(indices[0], distances[0]):
            metadata = self.embedded_documents[idx]['unique_metadata']
            metadata_str = json.dumps(metadata)
            
            if metadata_str not in seen_metadata:
                seen_metadata.add(metadata_str)
                similar_diseases.append({
                    'name': metadata.get('name', ''),
                    'id': metadata.get('id', ''),
                    'definition': metadata.get('definition', ''),
                    'similarity_score': 1.0 / (1.0 + distance)  # Convert distance to similarity
                })
                
                if len(similar_diseases) >= k:
                    break
                    
        return similar_diseases
    
    def check_abbreviation(self, entity: str, context: Optional[str] = None) -> Dict:
        """
        Check if an entity is a clinical abbreviation and resolve it.
        
        Args:
            entity: Entity text to check
            context: Original sentence containing the entity
            
        Returns:
            Dictionary with abbreviation check results
        """
        # Skip if abbreviation checking is disabled
        if not self.use_abbreviations or not self.abbreviation_searcher:
            return {
                'is_abbreviation': False,
                'expanded_term': None,
                'confidence': 1.0,
                'method': 'abbreviations_disabled'
            }
        
        # Handle empty entities
        if not entity:
            return {
                'is_abbreviation': False,
                'expanded_term': None,
                'confidence': 1.0,
                'method': 'empty_entity'
            }
            
        # Create a cache key
        cache_key = f"abbr::{entity}"
        
        # Check cache
        if cache_key in self.abbreviation_cache:
            result = self.abbreviation_cache[cache_key]
            self._debug_print(f"Cache hit for abbreviation check '{entity}': {result['is_abbreviation']}", level=1)
            return result
            
        self._debug_print(f"Checking if '{entity}' is an abbreviation", level=1)
        
        # Check if entity looks like an abbreviation (all uppercase, or contains periods)
        looks_like_abbreviation = entity.isupper() or '.' in entity or len(entity) <= 5
        
        if not looks_like_abbreviation:
            result = {
                'is_abbreviation': False,
                'expanded_term': None,
                'confidence': 0.9,
                'method': 'quick_check'
            }
            self.abbreviation_cache[cache_key] = result
            return result
        
        # Search for abbreviation
        try:
            search_results = self.abbreviation_searcher.search(entity)
            
            if not search_results:
                result = {
                    'is_abbreviation': False,
                    'expanded_term': None,
                    'confidence': 0.9,
                    'method': 'no_match_found'
                }
                self.abbreviation_cache[cache_key] = result
                return result
            
            # Get top result
            top_result = search_results[0]
            similarity = top_result.get('similarity', 0.0)
            query_term = top_result.get('query_term', '')
            expanded_term = top_result.get('result', '')
            
            # Check if this is a good match
            is_abbreviation = similarity > 0.7 and query_term.lower() == entity.lower()
            
            result = {
                'is_abbreviation': is_abbreviation,
                'expanded_term': expanded_term if is_abbreviation else None,
                'confidence': similarity if is_abbreviation else 0.5,
                'method': 'abbreviation_lookup',
                'all_matches': search_results[:3]  # Include top 3 matches
            }
            
            self.abbreviation_cache[cache_key] = result
            
            if is_abbreviation:
                self._debug_print(f"'{entity}' is an abbreviation for '{expanded_term}'", level=2)
            else:
                self._debug_print(f"'{entity}' is not a recognized abbreviation", level=2)
            
            return result
        
        except Exception as e:
            self._debug_print(f"Error checking abbreviation: {e}", level=2)
            result = {
                'is_abbreviation': False,
                'expanded_term': None,
                'confidence': 1.0,
                'method': 'error'
            }
            self.abbreviation_cache[cache_key] = result
            return result
    
    def verify_rare_disease(self, entity: str, context: Optional[str] = None) -> Dict:
        """
        Verify if an entity is a rare disease through binary YES/NO matching against candidates.
        
        Args:
            entity: Entity text to verify
            context: Original sentence containing the entity
            
        Returns:
            Dictionary with verification results
        """
        # Handle empty entities
        if not entity:
            return {
                'is_rare_disease': False,
                'confidence': 1.0,
                'method': 'empty_entity'
            }
            
        # Create a cache key
        cache_key = f"verify::{entity}::{context or ''}"
        
        # Check cache first
        if cache_key in self.verification_cache:
            result = self.verification_cache[cache_key]
            self._debug_print(f"Cache hit for rare disease verification '{entity}': {result['is_rare_disease']}", level=1)
            return result
            
        self._debug_print(f"Verifying if '{entity}' is a rare disease via binary matching", level=1)
        
        # Check for exact matches using fuzzy matching first (optimization)
        similar_diseases = self._retrieve_similar_diseases(entity)
        
        for disease in similar_diseases:
            normalized_name = self._normalize_text(disease['name'])
            normalized_entity = self._normalize_text(entity)
            
            # Check for exact match
            if normalized_name == normalized_entity:
                self._debug_print(f"Exact match found: '{entity}' matches '{disease['name']}' ({disease['id']})", level=2)
                result = {
                    'is_rare_disease': True,
                    'confidence': 1.0,
                    'method': 'exact_match',
                    'orpha_id': disease['id'],
                    'matched_term': disease['name']
                }
                self.verification_cache[cache_key] = result
                return result
                
            # Check for high similarity match (over 90%)
            similarity = fuzz.ratio(normalized_name, normalized_entity)
            if similarity > 93:
                self._debug_print(f"High similarity match ({similarity}%): '{entity}' matches '{disease['name']}' ({disease['id']})", level=2)
                result = {
                    'is_rare_disease': True,
                    'confidence': similarity / 100.0,
                    'method': 'high_similarity_match',
                    'orpha_id': disease['id'],
                    'matched_term': disease['name']
                }
                self.verification_cache[cache_key] = result
                return result
        
        # Format candidates for the LLM prompt
        candidate_items = []
        for i, disease in enumerate(similar_diseases, 1):
            candidate_items.append(f"{i}. '{disease['name']}' ({disease['id']})")
        
        candidates_text = "\n".join(candidate_items)
        
        # Create context part
        context_part = ""
        if context:
            context_part = f"\nOriginal sentence context: '{context}'"
        
        # Create the binary YES/NO matching prompt
        prompt = (
            f"I need to determine if the entity '{entity}' represents a rare disease."
            f"\n\nHere are some rare disease candidates for reference:"
            f"\n\n{candidates_text}\n"
            f"{context_part}\n\n"
            f"A rare disease is typically defined as a condition that affects fewer than 1 in 2,000 people."
            f"\n\nBased on these candidates and criteria, is '{entity}' a rare disease?"
            f"\nRespond with ONLY 'YES' or 'NO'."
        )
        
        # Query the LLM
        response = self.llm_client.query(prompt, self.direct_verification_system_message)
        
        # Parse the response - strictly look for "YES" or "NO"
        response_text = response.strip().upper()
        is_rare_disease = "YES" in response_text and "NO" not in response_text
        
        # Create result based on binary response
        if is_rare_disease:
            result = {
                'is_rare_disease': True,
                'confidence': 0.9,
                'method': 'llm_binary_verification'
            }
        else:
            result = {
                'is_rare_disease': False,
                'confidence': 0.9,
                'method': 'llm_binary_verification'
            }
        
        # Cache the result
        self.verification_cache[cache_key] = result
        
        self._debug_print(f"LLM binary verification: '{entity}' is{'' if is_rare_disease else ' not'} a rare disease", level=2)
        return result

    def process_entity(self, entity: str, context: Optional[str] = None) -> Dict:
        """
        Process an entity through the multistage pipeline.
        
        Args:
            entity: Entity text to process
            context: Original sentence containing the entity
            
        Returns:
            Dictionary with processing results
        """
        # Handle empty entities
        if not entity:
            return {
                'status': 'not_rare_disease',
                'entity': None,
                'original_entity': entity,
                'confidence': 1.0,
                'method': 'empty_entity'
            }
            
        self._debug_print(f"Processing entity: '{entity}'", level=0)
        
        # Clean and preprocess the entity
        cleaned_entity = self.preprocess_entity(entity)
        if not cleaned_entity:
            return {
                'status': 'not_rare_disease',
                'entity': None,
                'original_entity': entity,
                'confidence': 1.0,
                'method': 'empty_after_preprocessing'
            }
        
        # STAGE 1: Check if it's an abbreviation
        if self.use_abbreviations and self.abbreviation_searcher:
            abbreviation_result = self.check_abbreviation(cleaned_entity, context)
            
            if abbreviation_result['is_abbreviation']:
                expanded_term = abbreviation_result['expanded_term']
                self._debug_print(f"'{entity}' is an abbreviation for '{expanded_term}'", level=1)
                
                # Use the expanded term for verification
                entity_to_verify = expanded_term
                
                # Verify if the expanded term is a rare disease
                verification_result = self.verify_rare_disease(entity_to_verify, context)
                
                if verification_result['is_rare_disease']:
                    self._debug_print(f"Expanded term '{expanded_term}' is a rare disease", level=1)
                    
                    result = {
                        'status': 'verified_rare_disease',
                        'entity': verification_result.get('matched_term', expanded_term),
                        'original_entity': entity,
                        'expanded_term': expanded_term,
                        'confidence': min(abbreviation_result['confidence'], verification_result['confidence']),
                        'method': 'abbreviation_expansion'
                    }
                    
                    if 'orpha_id' in verification_result:
                        result['orpha_id'] = verification_result['orpha_id']
                        
                    return result
                else:
                    self._debug_print(f"Expanded term '{expanded_term}' is not a rare disease", level=1)
                    
                    # Continue to STAGE 2 with the original term, as the expanded term is not a rare disease
            
        # STAGE 2: Check if the original term is a rare disease
        verification_result = self.verify_rare_disease(cleaned_entity, context)
        
        if verification_result['is_rare_disease']:
            self._debug_print(f"'{entity}' is a direct rare disease", level=1)
            
            result = {
                'status': 'verified_rare_disease',
                'entity': verification_result.get('matched_term', cleaned_entity),
                'original_entity': entity,
                'confidence': verification_result['confidence'],
                'method': verification_result['method']
            }
            
            if 'orpha_id' in verification_result:
                result['orpha_id'] = verification_result['orpha_id']
                
            return result
        
        # Not a rare disease
        self._debug_print(f"'{entity}' is not a rare disease", level=1)
        return {
            'status': 'not_rare_disease',
            'entity': None,
            'original_entity': entity,
            'confidence': verification_result['confidence'],
            'method': verification_result['method']
        }
    
    def batch_process(self, entities_with_context: List[Dict]) -> List[Dict]:
        """
        Process a batch of entities with their contexts.
        
        Args:
            entities_with_context: List of dicts with 'entity' and 'context' keys
            
        Returns:
            List of dicts with processing results (verified rare diseases only)
        """
        self._debug_print(f"Processing batch of {len(entities_with_context)} entities")
        
        # Ensure input data is in correct format
        formatted_entries = []
        for item in entities_with_context:
            if isinstance(item, dict) and 'entity' in item:
                formatted_entries.append(item)
            elif isinstance(item, str):
                formatted_entries.append({'entity': item, 'context': ''})
            else:
                self._debug_print(f"Skipping invalid entry: {item}")
                continue
        
        # Remove duplicates while preserving order
        unique_entries = []
        seen = set()
        for item in formatted_entries:
            entity = str(item.get('entity', '')).lower().strip()
            context = str(item.get('context', ''))
            
            # Create a unique key for deduplication
            entry_key = f"{entity}::{context}"
            
            if entry_key not in seen and entity:
                seen.add(entry_key)
                unique_entries.append(item)
        
        self._debug_print(f"Found {len(unique_entries)} unique entity-context pairs")
        
        # Process each entity through the pipeline
        results = []
        for item in unique_entries:
            entity = item.get('entity', '')
            context = item.get('context', '')
            
            result = self.process_entity(entity, context)
            
            # Only include entities that are verified rare diseases
            if result['status'] == 'verified_rare_disease':
                # Add original context
                result['context'] = context
                results.append(result)
        
        self._debug_print(f"Identified {len(results)} verified rare diseases")
        return results

# Factory function to create a verifier
def create_rd_verifier(embedding_manager, llm_client, config=None, debug=False, 
                     abbreviations_file=None, use_abbreviations=True):
    """
    Factory function to create a multistage rare disease verifier.
    
    Args:
        embedding_manager: Manager for embedding operations
        llm_client: LLM client for verification
        config: Configuration for the verifier (currently unused)
        debug: Enable debug output
        abbreviations_file: Path to abbreviations embeddings file
        use_abbreviations: Whether to use abbreviation resolution
        
    Returns:
        MultiStageRDVerifier instance
    """
    return MultiStageRDVerifier(
        embedding_manager=embedding_manager,
        llm_client=llm_client,
        config=config,
        debug=debug,
        abbreviations_file=abbreviations_file,
        use_abbreviations=use_abbreviations
    )