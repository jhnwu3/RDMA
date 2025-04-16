
from abc import ABC, abstractmethod
import json
import pandas as pd
import numpy as np
import re
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from fuzzywuzzy import fuzz
from utils.embedding import EmbeddingsManager
from utils.llm_client import LocalLLMClient
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