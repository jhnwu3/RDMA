
from abc import ABC, abstractmethod
import json
import pandas as pd
import numpy as np
import re
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime
from fuzzywuzzy import fuzz
from rdma.utils.embedding import EmbeddingsManager
from rdma.utils.llm_client import LocalLLMClient
class BaseRDMatcher(ABC):
    """Abstract base class for rare disease term matching."""
    
    @abstractmethod
    def match_rd_terms(self, entities: List[str], metadata: List[Dict]) -> List[Dict]:
        """Match entities to rare disease terms."""
        pass

    @abstractmethod
    def process_batch(self, entities_batch: List[List[str]], metadata_batch: List[List[Dict]]) -> List[List[Dict]]:
        """Process a batch of entities for rare disease term matching."""
        pass

class SimpleRDMatcher(BaseRDMatcher):
    """
    Simple rare disease matcher that focuses solely on matching entities to rare disease terms
    without performing verification.
    
    This class uses embedding-based retrieval to find the most similar rare disease terms
    for a given entity string. It provides both exact/fuzzy matching and LLM-assisted matching.
    """
    
    def __init__(self, embeddings_manager, llm_client=None, system_message: str = None):
        """
        Initialize the simple rare disease matcher.
        
        Args:
            embeddings_manager: Manager for embedding operations
            llm_client: Optional LLM client for enhanced matching (can be None)
            system_message: System message for LLM prompting (required if llm_client is provided)
        """
        self.embeddings_manager = embeddings_manager
        self.llm_client = llm_client
        self.system_message = system_message
        self.index = None
        self.embedded_documents = None
        
    def prepare_index(self, metadata: List[Dict]):
        """
        Prepare FAISS index from metadata.
        
        Args:
            metadata: List of dictionaries containing rare disease metadata with embeddings
        """
        embeddings_array = self.embeddings_manager.prepare_embeddings(metadata)
        self.index = self.embeddings_manager.create_index(embeddings_array)
        self.embedded_documents = metadata
        
    def _retrieve_candidates(self, entity: str, max_candidates: int = 20) -> List[Dict]:
        """
        Retrieve relevant candidates with metadata and similarity scores.
        
        Args:
            entity: Entity text to match
            max_candidates: Maximum number of candidates to retrieve
            
        Returns:
            List of candidate dictionaries with metadata and similarity scores
        """
        if self.index is None:
            raise ValueError("Index not prepared. Call prepare_index() first.")
            
        query_vector = self.embeddings_manager.query_text(entity).reshape(1, -1)
        distances, indices = self.embeddings_manager.search(query_vector, self.index, k=500)
        
        seen_metadata = set()
        candidate_metadata = []
        
        for idx, distance in zip(indices[0], distances[0]):
            try:
                # Access document directly since it already has name, id, definition
                document = self.embedded_documents[idx]
                
                # Create an identifier for deduplication
                metadata_id = f"{document.get('name', '')}-{document.get('id', '')}"
                
                if metadata_id not in seen_metadata:
                    seen_metadata.add(metadata_id)
                    candidate_metadata.append({
                        'metadata': {
                            'name': document.get('name', ''),
                            'id': document.get('id', ''),
                            'definition': document.get('definition', '')
                        },
                        'similarity_score': 1.0 / (1.0 + distance)  # Convert distance to similarity
                    })
                    
                    if len(candidate_metadata) >= max_candidates:
                        break
            except Exception as e:
                print(f"Error processing metadata at index {idx}: {e}")
                continue
                    
        return candidate_metadata
    
    def match_entity(self, entity_data: Union[str, Dict], top_k: int = 5) -> Dict:
        """
        Match an entity to the most appropriate rare disease term.
        
        Args:
            entity_data: Entity text to match or dict with entity and metadata
            top_k: Number of top candidates to include in results
            
        Returns:
            Dictionary with matching results
        """
        if self.index is None:
            raise ValueError("Index not prepared. Call prepare_index() first.")
        
        # Extract entity and metadata
        if isinstance(entity_data, dict):
            entity = entity_data.get("entity", "")
            metadata = entity_data
        else:
            entity = entity_data
            metadata = {"entity": entity}
            
        # Get candidates
        candidates = self._retrieve_candidates(entity)
        
        # Prepare result structure
        result = {
            'entity': entity,
            'top_candidates': candidates[:top_k]
        }
        
        # Preserve original entity and metadata
        if isinstance(entity_data, dict):
            # Preserve original entity if available
            if "original_entity" in metadata:
                result["original_entity"] = metadata["original_entity"]
            else:
                result["original_entity"] = entity
                
            # Preserve context if available
            if "context" in metadata:
                result["context"] = metadata["context"]
                
            # Preserve abbreviation info if available
            if "expanded_term" in metadata:
                result["expanded_term"] = metadata["expanded_term"]
                result["is_abbreviation"] = True
                
            # Preserve verification method if available
            if "status" in metadata and metadata["status"] == "verified_rare_disease":
                result["verification_status"] = "verified_rare_disease"
                
            if "method" in metadata:
                result["verification_method"] = metadata["method"]
        else:
            # For string entities, set original_entity to the entity itself
            result["original_entity"] = entity
        
        # Try to find exact/fuzzy match first
        rd_term = self._try_similarity_matching(entity, candidates)
        if rd_term:
            result.update({
                'rd_term': rd_term['name'],
                'orpha_id': rd_term['id'],
                'match_method': 'similarity',
                'confidence_score': rd_term.get('confidence', 0.9)
            })
            return result
        
        # If no similarity match but LLM available, try LLM matching
        if self.llm_client and self.system_message:
            rd_term = self._try_llm_match(entity, candidates[:5])
            if rd_term:
                result.update({
                    'rd_term': rd_term['name'],
                    'orpha_id': rd_term['id'],
                    'match_method': 'llm',
                    'confidence_score': 0.7
                })
                return result
        
        # If no match found but we have candidates, use top candidate as fallback
        if candidates:
            top_candidate = candidates[0]
            result.update({
                'rd_term': top_candidate['metadata']['name'],
                'orpha_id': top_candidate['metadata']['id'],
                'match_method': 'fallback',
                'confidence_score': min(0.5, top_candidate['similarity_score'])  # Cap confidence at 0.5
            })
        else:
            # No candidates at all
            result.update({
                'rd_term': None,
                'orpha_id': None,
                'match_method': 'no_match',
                'confidence_score': 0.0
            })
            
        return result
    
    def _try_similarity_matching(self, entity: str, candidates: List[Dict]) -> Optional[Dict]:
        """
        Try to match entity to a rare disease term using string similarity.
        
        Args:
            entity: Entity text to match
            candidates: List of candidate rare disease terms
            
        Returns:
            Dictionary with matched term info or None if no match found
        """
        from fuzzywuzzy import fuzz
        
        cleaned_entity = entity.lower().strip()
        
        # Check for exact match
        for candidate in candidates:
            name = candidate['metadata']['name']
            cleaned_name = name.lower().strip()
            
            # Exact match
            if cleaned_name == cleaned_entity:
                return {
                    'name': name,
                    'id': candidate['metadata']['id'],
                    'confidence': 1.0
                }
        
        # Check for high fuzzy match
        best_score = 0
        best_match = None
        
        for candidate in candidates:
            name = candidate['metadata']['name']
            score = fuzz.ratio(cleaned_entity, name.lower().strip())
            
            if score > best_score and score >= 90:  # Only consider very strong matches (90%+)
                best_score = score
                best_match = {
                    'name': name,
                    'id': candidate['metadata']['id'],
                    'confidence': score / 100.0
                }
        
        return best_match
    
    def _try_llm_match(self, entity: str, candidates: List[Dict]) -> Optional[Dict]:
        """
        Try to match entity to a rare disease term using LLM assistance.
        
        Args:
            entity: Entity text to match
            candidates: List of candidate rare disease terms
            
        Returns:
            Dictionary with matched term info or None if no match found
        """
        if not self.llm_client or not self.system_message:
            return None
            
        # Format candidates for LLM prompt
        context = "\n".join([
            f"{i+1}. {candidate['metadata']['name']} (ID: {candidate['metadata']['id']})"
            for i, candidate in enumerate(candidates[:5])
        ])
        
        prompt = (
            f"I need to match the entity '{entity}' to the most appropriate rare disease term.\n\n"
            f"Here are some candidate matches:\n{context}\n\n"
            f"Select the best matching rare disease from these candidates. "
            f"Return ONLY the ID of the matched rare disease (e.g., 'ORPHA:12345') or 'NONE' if none match."
        )
        
        # Query LLM
        response = self.llm_client.query(prompt, self.system_message)
        
        # Extract ORPHA ID from response
        orpha_match = re.search(r'ORPHA:\d+', response)
        if orpha_match:
            orpha_id = orpha_match.group(0)
            # Find corresponding metadata
            for candidate in candidates:
                if candidate['metadata']['id'] == orpha_id:
                    return {
                        'name': candidate['metadata']['name'],
                        'id': orpha_id
                    }
        
        return None
    
    def batch_match_entities(self, entities: List[Union[str, Dict]], top_k: int = 5) -> List[Dict]:
        """
        Match multiple entities to rare disease terms.
        
        Args:
            entities: List of entity strings or dictionaries to match
            top_k: Number of top candidates to include in results
            
        Returns:
            List of dictionaries with matching results
        """
        if self.index is None:
            raise ValueError("Index not prepared. Call prepare_index() first.")
            
        results = []
        for entity in entities:
            match_result = self.match_entity(entity, top_k)
            results.append(match_result)
            
        return results
    
    def match_rd_terms(self, entities: List[Union[str, Dict]], metadata: List[Dict]) -> List[Dict]:
        """
        Match entities to rare disease terms with better metadata preservation.
        
        Args:
            entities: List of entity strings or dictionaries
            metadata: Embedded rare disease documents
            
        Returns:
            List of matched disease terms with preserved metadata
        """
        if self.index is None:
            self.prepare_index(metadata)
            
        matches = []
        
        for entity_item in entities:
            # Default values
            original_input = entity_item
            metadata_dict = {}
            
            # Extract entity text and metadata
            if isinstance(entity_item, str):
                entity = entity_item
                original_entity = entity
            elif isinstance(entity_item, dict):
                # Detailed handling of different verification formats
                if entity_item.get("status") == "verified_rare_disease":
                    entity = entity_item.get("entity", "")
                    original_entity = entity_item.get("original_entity", entity)
                    metadata_dict = {
                        "original_entity": original_entity,
                        "expanded_term": entity_item.get("expanded_term"),
                        "method": entity_item.get("method"),
                        "context": entity_item.get("context"),
                        "orpha_id": entity_item.get("orpha_id")
                    }
                elif "entity" in entity_item:
                    entity = entity_item["entity"]
                    original_entity = entity_item.get("original_entity", entity)
                    metadata_dict = entity_item.copy()
                else:
                    continue  # Skip if no usable entity
            else:
                continue  # Skip unsupported types
            
            # Retrieve candidates
            candidates = self._retrieve_candidates(entity)
            
            # Create base match result
            match_result = {
                'entity': entity,
                'original_entity': original_entity,
                'top_candidates': [
                    {
                        'name': c['metadata']['name'],
                        'id': c['metadata']['id'],
                        'similarity': float(c['similarity_score'])
                    }
                    for c in candidates[:5]
                ]
            }
            print(match_result)
            # Try to find an exact or close match
            rd_term = self._try_similarity_matching(entity, candidates)
            if rd_term:
                match_result.update({
                    'rd_term': rd_term['name'],
                    'orpha_id': rd_term['id'],
                    'match_method': 'exact',
                    'confidence_score': 1.0
                })
            elif self.llm_client:
                # Try LLM matching if no similarity match
                rd_term = self._try_llm_match(entity, candidates[:5])
                if rd_term:
                    match_result.update({
                        'rd_term': rd_term['name'],
                        'orpha_id': rd_term['id'],
                        'match_method': 'llm',
                        'confidence_score': 0.7
                    })
            
            # Preserve additional metadata
            for key, value in metadata_dict.items():
                if value is not None:
                    match_result[key] = value

            matches.append(match_result)
        
        return matches

    def process_batch(self, entities_batch: List[List[Union[str, Dict]]], metadata_batch: List[List[Dict]]) -> List[List[Dict]]:
        """
        Process a batch of entities for rare disease term matching.
        
        Args:
            entities_batch: Batch of entity lists to process
            metadata_batch: Corresponding batch of metadata lists
            
        Returns:
            List of lists containing matching results for each batch
        """
        results = []
        
        for entities, metadata in zip(entities_batch, metadata_batch):
            # Match entities to rare disease terms
            batch_results = self.match_rd_terms(entities, metadata)
            results.append(batch_results)
            
        return results

class RAGRDMatcher(BaseRDMatcher):
    """Rare disease term matcher using RAG approach with enhanced match tracking."""
    
    def __init__(self, embeddings_manager : EmbeddingsManager, llm_client=None, system_message: str = None):
        self.embeddings_manager = embeddings_manager
        self.llm_client = llm_client
        self.system_message = system_message
        self.index = None
        self.embedded_documents = None
        
    def prepare_index(self, metadata: List[Dict]):
        """Prepare FAISS index from metadata."""
        embeddings_array = self.embeddings_manager.prepare_embeddings(metadata)
        self.index = self.embeddings_manager.create_index(embeddings_array)
        self.embedded_documents = metadata
        
    def clean_text(self, text: str) -> str:
        """Clean input text for matching."""
        return text.lower().strip()
        
    def _retrieve_candidates(self, entity: str) -> List[Dict]:
        """Retrieve relevant candidates with metadata and similarity scores."""
        query_vector = self.embeddings_manager.query_text(entity).reshape(1, -1)
        distances, indices = self.embeddings_manager.search(query_vector, self.index, k=800)
        
        seen_metadata = set()
        candidate_metadata = []
        
        for idx, distance in zip(indices[0], distances[0]):
            metadata = self.embedded_documents[idx]
            metadata_str = json.dumps(metadata.get('name', ''))
            
            if metadata_str not in seen_metadata:
                seen_metadata.add(metadata_str)
                candidate_metadata.append({
                    'metadata': metadata,
                    'similarity_score': 1 / (1 + distance)
                })
                if len(candidate_metadata) == 20:
                    break
                    
        return candidate_metadata
    
    def match_rd_terms(self, entities: List[str], metadata: List[Dict]) -> List[Dict]:
        """Match entities to rare disease terms with sequential verification."""
        if self.index is None:
            self.prepare_index(metadata)
            
        matches = []
        
        for entity in entities:
            # Step 1: Get initial candidates for context
            candidates = self._retrieve_candidates(entity)
            
            # Create clean candidates with just the fields we need
            clean_candidates = []
            for c in candidates[:10]:  # Get top 10 for verification
                metadata = c['metadata']
                clean_candidates.append({
                    'metadata': {
                        'name': metadata['name'],
                        'id': metadata['id'],
                        'definition': metadata.get('definition', '')
                    },
                    'similarity_score': c['similarity_score']
                })
            
            # Step 2: Verify if it's a rare disease using clean candidates
            is_rare_disease = self._verify_rare_disease(entity, clean_candidates[:5]) # only 5 here to save more time hopefully.
            print("Verified rare disease:", is_rare_disease)
            if not is_rare_disease:
                continue
            else:
                # Step 3: Try exact/fuzzy matching first
                match_info = {
                    'entity': entity,
                    'top_candidates': clean_candidates[:5]  # Store top 5 clean candidates
                }
                
                rd_term = self._try_enriched_matching(entity, clean_candidates)  # Use clean candidates for matching
                if rd_term:
                    match_info.update({
                        'rd_term': rd_term['name'],
                        'orpha_id': rd_term['id'],
                        'match_method': 'exact',
                        'confidence_score': 1.0
                    })
                    matches.append(match_info)
                    continue
                
                # Step 4: If no exact match but verified as rare disease, try LLM matching
                if self.llm_client:
                    rd_term = self._try_llm_match(entity, clean_candidates[:5])  # Use clean candidates for LLM matching
                    if rd_term:
                        match_info.update({
                            'rd_term': rd_term['name'],
                            'orpha_id': rd_term['id'],
                            'match_method': 'llm',
                            'confidence_score': 0.7
                        })
                        matches.append(match_info)
                    
        return matches
        
    def _try_enriched_matching(self, entity: str, candidates: List[Dict]) -> Optional[Dict]:
        """Try matching using enrichment process."""
        cleaned_phrase = self.clean_text(entity)
        
        # Convert candidates to list of disease names and IDs
        disease_entries = []
        for candidate in candidates:
            metadata = candidate['metadata']
            disease_entries.append({
                'name': metadata['name'],
                'id': metadata['id']
            })
            
        # Step 1: Exact matching
        for entry in disease_entries:
            if self.clean_text(entry['name']) == cleaned_phrase:
                return entry
                
        # Step 2: Fuzzy matching
        fuzzy_matches = []
        for entry in disease_entries:
            cleaned_term = self.clean_text(entry['name'])
            if fuzz.ratio(cleaned_phrase, cleaned_term) > 90:
                fuzzy_matches.append(entry)
                
        if fuzzy_matches:
            return fuzzy_matches[0]
            
        return None
        
    def _verify_rare_disease(self, term: str, candidates: List[Dict]) -> bool:
        """Verify if the term represents a rare disease using context from candidates."""
        if not self.llm_client:
            return True  # If no LLM client, assume all terms are valid
            
        # Format candidate context
        context = "\nPotential matches from database:\n" + "\n".join([
            f"{candidate['metadata']['name']} ({candidate['metadata']['id']})"
            for i, candidate in enumerate(candidates)
        ])
        
        prompt = f"""Analyze this medical term and determine if it represents a rare disease.

        Term: {term}
        {context}

        A term should ONLY be considered a rare disease if ALL these criteria are met:
        1. It is a disease or syndrome (not just a symptom, finding, or condition)
        2. It is rare (affecting less than 1 in 2000 people)
        3. There is clear evidence in the context or term itself indicating rarity
        4. For variants of common diseases, it must be explicitly marked as a rare variant
        5. The term should align with the type of entries in our rare disease database.
        6. If there is a partial match, i.e cholangitis vs. sclerosing cholangitis. There must be a mention of its descriptor (sclerosing) in the term itself, otherwise it's invalid match.

        Response format:
        First line: "DECISION: true" or "DECISION: false"
        Next lines: Brief explanation of decision"""

        # print("System message:", self.system_message)
        # print("Prompt:", prompt)

        response = self.llm_client.query(prompt, self.system_message).strip().lower()
        print("Prompt:")
        print(prompt)
        print("Response:")
        print(response)

        return "decision: true" in response.lower()

    def _try_llm_match(self, entity: str, candidates: List[Dict]) -> Optional[Dict]:
        """Match verified rare disease term to specific ORPHA entry."""
        if not self.llm_client:
            return None
            
        context = "\n".join([
            f"{i+1}. {candidate['metadata']['name']} (ORPHA:{candidate['metadata']['id']})"
            for i, candidate in enumerate(candidates[:5])
        ])
        
        prompt = f"""Given this verified rare disease term, find the best matching ORPHA entry.

                    Term: {entity}

                    Potential matches:
                    {context}

                    Return ONLY the ORPHA ID of the best matching entry (e.g., "ORPHA:12345") or "none" if no clear match.
                    The match should be semantically equivalent, not just similar words."""

        response = self.llm_client.query(prompt, self.system_message).strip()
        
        # Extract ORPHA ID from response
        orpha_match = re.search(r'ORPHA:\d+', response)
        if orpha_match:
            orpha_id = orpha_match.group(0)
            # Find corresponding metadata
            for candidate in candidates:
                if candidate['metadata']['id'] == orpha_id:
                    return {
                        'name': candidate['metadata']['name'],
                        'id': orpha_id
                    }
                    
        return None
        
    def process_batch(self, entities_batch: List[List[str]], metadata_batch: List[List[Dict]]) -> List[List[Dict]]:
        """Process a batch of entities for rare disease term matching."""
        results = []
        for entities, metadata in zip(entities_batch, metadata_batch):
            matches = self.match_rd_terms(entities, metadata)
            results.append(matches)
        return results