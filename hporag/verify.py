from typing import List, Dict, Any, Optional, Tuple, Union, Set
import json
import re
import numpy as np
import torch
from datetime import datetime
from fuzzywuzzy import fuzz
import itertools
import time

class HPOVerifierConfig:
    """Configuration for when to use retrieval and context in the HPO verification pipeline."""
    
    def __init__(self, 
                 use_retrieval_for_direct=True,
                 use_retrieval_for_implies=True,
                 use_retrieval_for_extract=True,
                 use_retrieval_for_validation=True,
                 use_retrieval_for_implication=True,
                 use_context_for_direct=True,
                 use_context_for_implies=True,
                 use_context_for_extract=True,
                 use_context_for_validation=False,
                 use_context_for_implication=True):
        # Retrieval settings
        self.use_retrieval_for_direct = use_retrieval_for_direct
        self.use_retrieval_for_implies = use_retrieval_for_implies
        self.use_retrieval_for_extract = use_retrieval_for_extract
        self.use_retrieval_for_validation = use_retrieval_for_validation
        self.use_retrieval_for_implication = use_retrieval_for_implication
        
        # Context usage settings
        self.use_context_for_direct = use_context_for_direct
        self.use_context_for_implies = use_context_for_implies
        self.use_context_for_extract = use_context_for_extract
        self.use_context_for_validation = use_context_for_validation
        self.use_context_for_implication = use_context_for_implication
    
    def to_dict(self):
        """Convert configuration to a dictionary."""
        return {
            "retrieval": {
                "direct": self.use_retrieval_for_direct,
                "implies": self.use_retrieval_for_implies,
                "extract": self.use_retrieval_for_extract,
                "validation": self.use_retrieval_for_validation,
                "implication": self.use_retrieval_for_implication
            },
            "context": {
                "direct": self.use_context_for_direct,
                "implies": self.use_context_for_implies,
                "extract": self.use_context_for_extract,
                "validation": self.use_context_for_validation,
                "implication": self.use_context_for_implication
            }
        }
    
    @classmethod
    def from_dict(cls, config_dict):
        """Create configuration from a dictionary."""
        return cls(
            use_retrieval_for_direct=config_dict["retrieval"]["direct"],
            use_retrieval_for_implies=config_dict["retrieval"]["implies"],
            use_retrieval_for_extract=config_dict["retrieval"]["extract"],
            use_retrieval_for_validation=config_dict["retrieval"]["validation"],
            use_retrieval_for_implication=config_dict["retrieval"]["implication"],
            use_context_for_direct=config_dict["context"]["direct"],
            use_context_for_implies=config_dict["context"]["implies"],
            use_context_for_extract=config_dict["context"]["extract"],
            use_context_for_validation=config_dict["context"]["validation"],
            use_context_for_implication=config_dict["context"]["implication"]
        )
    
    def __str__(self):
        """String representation of the configuration."""
        return str(self.to_dict())

class ConfigurableHPOVerifier:
    """A configurable version of the MultiStageHPOVerifier that allows fine-tuning of retrieval and context usage."""
    
    def __init__(self, embedding_manager, llm_client, config=None, debug=False):
        """Initialize with a specific configuration."""
        self.embedding_manager = embedding_manager
        self.llm_client = llm_client
        self.debug = debug
        self.index = None
        self.embedded_documents = None
        self.config = config or HPOVerifierConfig()
        
        # System messages
        self.direct_verification_system_message = (
            "You are a clinical expert specializing in phenotype identification for Human Phenotype Ontology (HPO) mapping. "
            "Determine if a given term from a clinical note describes a valid human phenotype "
            "(an observable characteristic, trait, or abnormality). "
            "Please use the retrieved candidates in the clinical note to determine if it represents a valid phenotype. "
            "If the entity is just a piece of anatomy without any mention of an abnormality in the entity itself, it is not a phenotype, regardless of what is in the context. "
            "If the entity is just a lab measurement, it is not a phenotype."
            "\nRespond with ONLY 'YES' if the term is a valid phenotype, or 'NO' if it's not a phenotype "
            "Consider both the term itself AND its context in the clinical note."
        )
        
        self.implied_phenotype_system_message = (
            "You are a clinical expert specializing in phenotype identification for Human Phenotype Ontology (HPO) mapping. "
            "Determine if the given term IMPLIES a phenotype, even though it's not a direct phenotype itself. "
            "\nEXAMPLES:\n"
            "1. Laboratory test names (e.g., 'white blood cell count', 'hemoglobin level') imply phenotypes if the value is abnormal.\n"
            "2. Diagnostic procedures (e.g., 'kidney biopsy', 'chest X-ray') typically do NOT imply phenotypes unless findings are mentioned.\n"
            "3. Medications (e.g., 'insulin', 'lisinopril') can imply phenotypes related to the condition being treated.\n"
            "4. Microorganisms or pathogens (e.g., 'E. coli', 'Staphylococcus aureus') imply infection phenotypes.\n"
            "\nRespond with ONLY 'YES' if the term implies a phenotype, or 'NO' if it doesn't imply any phenotype. "
            "Consider both the term itself AND its context in the clinical note."
        )
        
        self.extract_phenotype_system_message = (
            "You are a clinical expert specializing in phenotype identification for Human Phenotype Ontology (HPO) mapping. "
            "A previous analysis determined that a given term implies a phenotype but is not a direct phenotype itself. "
            "Your task is to precisely identify what specific phenotype is implied by this term. "
            "\nEXAMPLES:\n"
            "1. 'Elevated white blood cell count' implies 'leukocytosis'\n"
            "2. 'Low hemoglobin' implies 'anemia'\n"
            "3. 'E. coli in urine' implies 'urinary tract infection' or 'bacteriuria'\n"
            "4. 'Taking insulin' implies 'diabetes mellitus'\n"
            "\nProvide ONLY the name of the implied phenotype as it would appear in medical terminology. "
            "Be specific and concise. Do not include explanations or multiple options separated by commas or slashes. "
            "Consider the term's context in the clinical note to determine the most accurate phenotype."
        )
        
        self.implication_validation_system_message = (
            "You are a clinical expert specializing in phenotype identification for Human Phenotype Ontology (HPO) mapping. "
            "Your task is to validate whether an implied phenotype is reasonable given the original entity and its context. "
            "Be critical and conservative in your assessment. Only validate implications that have strong clinical justification. "
            "\nEXAMPLES of VALID implications:\n"
            "1. Entity: 'E. coli in urine culture' → Implied phenotype: 'bacteriuria' (VALID: specific finding)\n"
            "2. Entity: 'taking insulin daily' → Implied phenotype: 'diabetes mellitus' (VALID: specific medication)\n"
            "\nEXAMPLES of INVALID implications:\n"
            "1. Entity: 'white blood cell count' → Implied phenotype: 'leukocytosis' (INVALID: no value specified)\n"
            "2. Entity: 'retina' → Implied phenotype: 'retinopathy' (INVALID: normal anatomy without abnormality)\n"
            "3. Entity: 'renal tissue' → Implied phenotype: 'glomerulonephritis' (INVALID: too specific without evidence)\n"
            "\nRespond with ONLY 'YES' if the implication is valid, or 'NO' if it's not valid based on the original term and context."
        )
        
        self.phenotype_validation_system_message = (
            "You are a clinical expert specializing in phenotype identification. "
            "Your task is to validate whether a proposed phenotype is a valid medical concept. "
            "Focus only on whether the term represents a real, recognized phenotype in clinical medicine. "
            "Do not worry about whether it matches any formal ontology or coding system. "
            "\nEXAMPLES of VALID phenotypes:\n"
            "1. 'bacteriuria' (VALID: recognized condition of bacteria in urine)\n"
            "2. 'diabetes mellitus' (VALID: well-established medical condition)\n"
            "3. 'macrocephaly' (VALID: recognized condition of abnormally large head)\n"
            "\nEXAMPLES of INVALID phenotypes:\n"
            "1. 'blood abnormality' (INVALID: too vague, not a specific phenotype)\n"
            "2. 'kidney status' (INVALID: not a phenotype, just an anatomical reference)\n"
            "3. 'medication response' (INVALID: too generic, not a specific phenotype)\n"
            "4. 'lab test issue' (INVALID: not a specific phenotype)\n"
            "\nRespond with ONLY 'YES' if the term is a valid, recognized phenotype, or 'NO' if it's not."
        )
        
        # Caches to avoid redundant API calls
        self.verification_cache = {}
        self.implied_phenotype_cache = {}
        self.extracted_phenotype_cache = {}
        self.implication_validation_cache = {}
        self.phenotype_validation_cache = {}
    
    def _debug_print(self, message: str, level: int = 0):
        """Print debug messages if debug mode is enabled."""
        if self.debug:
            indent = "  " * level
            print(f"{datetime.now().strftime('%H:%M:%S')} | {indent}{message}")
    
    def set_config(self, config):
        """Update the configuration."""
        self.config = config
        self.clear_caches()  # Clear caches when configuration changes
        
    def prepare_index(self, metadata: List[Dict]):
        """Prepare FAISS index from metadata for similarity search."""
        if self.index is None:
            self._debug_print("Preparing FAISS index for phenotype verification...")
            embeddings_array = self.embedding_manager.prepare_embeddings(metadata)
            self.index = self.embedding_manager.create_index(embeddings_array)
            self.embedded_documents = metadata
            self._debug_print(f"Index prepared with {len(metadata)} embedded documents")

    def clear_caches(self):
        """Clear all caches to prepare for a fresh evaluation run."""
        self.verification_cache = {}
        self.implied_phenotype_cache = {}
        self.extracted_phenotype_cache = {}
        self.implication_validation_cache = {}
        self.phenotype_validation_cache = {}
        self._debug_print("All caches cleared")

    def _retrieve_similar_phenotypes(self, entity: str, k: int = 10) -> List[Dict]:
        """Retrieve similar phenotypes from the HPO ontology for context."""
        if self.index is None:
            raise ValueError("Index not prepared. Call prepare_index() first.")
            
        # Embed the query
        query_vector = self.embedding_manager.query_text(entity).reshape(1, -1)
        
        # Search for similar items
        distances, indices = self.embedding_manager.search(query_vector, self.index, k=min(800, len(self.embedded_documents)))
        
        # Extract unique metadata
        similar_phenotypes = []
        seen_metadata = set()
        
        for idx, distance in zip(indices[0], distances[0]):
            metadata = self.embedded_documents[idx]['unique_metadata']
            metadata_str = json.dumps(metadata)
            
            if metadata_str not in seen_metadata:
                seen_metadata.add(metadata_str)
                similar_phenotypes.append({
                    'term': metadata.get('info', ''),
                    'hp_id': metadata.get('hp_id', ''),
                    'similarity_score': 1.0 / (1.0 + distance)  # Convert distance to similarity
                })
                
                if len(similar_phenotypes) >= k:
                    break
                    
        return similar_phenotypes
    
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

    def verify_direct_phenotype(self, entity: str, context: Optional[str] = None) -> Dict:
        """
        Verify if an entity is a direct phenotype with configurable retrieval and context usage.
        
        Args:
            entity: Entity text to verify
            context: Original sentence containing the entity
            
        Returns:
            Dictionary with verification results
        """
        # Handle empty entities
        if not entity:
            return {
                'is_phenotype': False,
                'confidence': 1.0,
                'method': 'empty_entity'
            }
            
        # Create a cache key - only include context if configured to use it
        cache_key = f"direct::{entity}::{context if self.config.use_context_for_direct else ''}"
        
        # Check cache first
        if cache_key in self.verification_cache:
            result = self.verification_cache[cache_key]
            self._debug_print(f"Cache hit for direct phenotype '{entity}': {result['is_phenotype']}", level=1)
            return result
            
        self._debug_print(f"Verifying if '{entity}' is a direct phenotype", level=1)
        
        # Retrieve similar phenotypes for context if configured
        similar_phenotypes = []
        if self.config.use_retrieval_for_direct:
            similar_phenotypes = self._retrieve_similar_phenotypes(entity)
            
            # Check for exact matches first (optimization)
            for phenotype in similar_phenotypes:
                if self._normalize_text(phenotype['term']) == self._normalize_text(entity):
                    self._debug_print(f"Exact match found: '{entity}' matches '{phenotype['term']}' ({phenotype['hp_id']})", level=2)
                    result = {
                        'is_phenotype': True,
                        'confidence': 1.0,
                        'method': 'exact_match',
                        'hp_id': phenotype['hp_id'],
                        'matched_term': phenotype['term']
                    }
                    self.verification_cache[cache_key] = result
                    return result
        
        # Format context items if using retrieval
        context_items = []
        if self.config.use_retrieval_for_direct:
            for phenotype in similar_phenotypes[:10]:  # Use top 10 for context
                context_items.append(f"- {phenotype['term']} ({phenotype['hp_id']})")
            
        context_text = "\n".join(context_items)
        
        # Create prompt for the LLM, including the sentence context if configured
        context_part = ""
        if context and self.config.use_context_for_direct:
            context_part = f"Original sentence context: '{context}'\n\n"
        
        retrieval_part = ""
        if self.config.use_retrieval_for_direct and context_items:
            retrieval_part = (
                f"Here are some retrieved candidates from the Human Phenotype Ontology to help you make your decision:\n\n"
                f"{context_text}\n\n"
            )
        
        prompt = (
            f"I need to determine if ENTITY:'{entity}' is a valid human phenotype. "
            f"{context_part}"
            f"{retrieval_part}"
            f"Based on {'these examples and ' if retrieval_part else ''}{'the original context' if context_part else 'your knowledge'}, "
            f"is just the ENTITY: '{entity}' a valid human phenotype? "
            f"Respond with ONLY 'YES' if the term is a valid phenotype, or 'NO' if it's not a phenotype."
        )
        
        # Query the LLM
        response = self.llm_client.query(prompt, self.direct_verification_system_message)
        
        # Parse the response
        is_phenotype = "yes" in response.lower() and "no" not in response.lower()
        
        # Create result
        if is_phenotype:
            result = {
                'is_phenotype': True, 
                'confidence': 0.8,
                'method': 'llm_verification'
            }
        else:
            result = {
                'is_phenotype': False,
                'confidence': 0.8,
                'method': 'llm_verification'
            }
        
        # Cache the result
        self.verification_cache[cache_key] = result
        
        self._debug_print(f"LLM says '{entity}' is{'' if is_phenotype else ' not'} a phenotype", level=2)
        return result

    def check_implies_phenotype(self, entity: str, context: Optional[str] = None) -> Dict:
        """
        Check if an entity implies a phenotype with configurable retrieval and context usage.
        
        Args:
            entity: Entity text to check
            context: Original sentence containing the entity
            
        Returns:
            Dictionary with results
        """
        # Handle empty entities
        if not entity:
            return {
                'implies_phenotype': False,
                'confidence': 1.0,
                'method': 'empty_entity'
            }
            
        # Create a cache key - only include context if configured to use it
        cache_key = f"implies::{entity}::{context if self.config.use_context_for_implies else ''}"
        
        # Check cache
        if cache_key in self.implied_phenotype_cache:
            result = self.implied_phenotype_cache[cache_key]
            self._debug_print(f"Cache hit for implied phenotype check '{entity}': {result['implies_phenotype']}", level=1)
            return result
            
        self._debug_print(f"Checking if '{entity}' implies a phenotype", level=1)
        
        # Retrieve similar phenotypes for context if configured
        similar_phenotypes = []
        if self.config.use_retrieval_for_implies:
            similar_phenotypes = self._retrieve_similar_phenotypes(entity)
        
        # Format context items if using retrieval
        context_items = []
        if self.config.use_retrieval_for_implies:
            for phenotype in similar_phenotypes[:10]:  # Use top 10 for context
                context_items.append(f"- {phenotype['term']} ({phenotype['hp_id']})")
        
        context_text = "\n".join(context_items)
        
        # Create prompt for the LLM
        context_part = ""
        if context and self.config.use_context_for_implies:
            context_part = f"Original sentence context: '{context}'\n\n"
        
        retrieval_part = ""
        if self.config.use_retrieval_for_implies and context_items:
            retrieval_part = (
                f"Here are some phenotype terms from the Human Phenotype Ontology for context:\n\n"
                f"{context_text}\n\n"
            )
        
        prompt = (
            f"I need to determine if '{entity}' implies a phenotype, even though it's not a direct phenotype itself. "
            f"{context_part}"
            f"{retrieval_part}"
            f"Based on {'this information and ' if retrieval_part or context_part else ''}clinical knowledge, does '{entity}' imply a phenotype? "
            f"For example, 'E. coli in urine' implies 'urinary tract infection'.\n\n"
            f"Respond with ONLY 'YES' if it implies a phenotype or 'NO' if it doesn't."
        )
        
        # Query the LLM
        response = self.llm_client.query(prompt, self.implied_phenotype_system_message)
        
        # Parse the response
        implies_phenotype = "yes" in response.lower() and "no" not in response.lower()
        
        # Create result
        result = {
            'implies_phenotype': implies_phenotype,
            'confidence': 0.8 if implies_phenotype else 0.7,
            'method': 'llm_verification'
        }
        
        # Cache the result
        self.implied_phenotype_cache[cache_key] = result
        
        self._debug_print(f"LLM says '{entity}' does{'' if implies_phenotype else ' not'} imply a phenotype", level=2)
        return result

    def extract_implied_phenotype(self, entity: str, context: Optional[str] = None) -> Dict:
        """
        Extract the specific phenotype implied by an entity with configurable retrieval and context usage.
        
        Args:
            entity: Entity text that implies a phenotype
            context: Original sentence containing the entity
            
        Returns:
            Dictionary with results
        """
        # Handle empty entities
        if not entity:
            return {
                'implied_phenotype': None,
                'confidence': 0.0,
                'method': 'empty_entity'
            }
            
        # Create a cache key - only include context if configured to use it
        cache_key = f"extract::{entity}::{context if self.config.use_context_for_extract else ''}"
        
        # Check cache
        if cache_key in self.extracted_phenotype_cache:
            result = self.extracted_phenotype_cache[cache_key]
            self._debug_print(f"Cache hit for extracting implied phenotype from '{entity}': {result.get('implied_phenotype')}", level=1)
            return result
            
        self._debug_print(f"Extracting implied phenotype from '{entity}'", level=1)
        
        # Retrieve similar phenotypes for context if configured
        similar_phenotypes = []
        if self.config.use_retrieval_for_extract:
            similar_phenotypes = self._retrieve_similar_phenotypes(entity)
        
        # Format context items if using retrieval
        context_items = []
        if self.config.use_retrieval_for_extract:
            for phenotype in similar_phenotypes[:10]:  # Use top 10 for context
                context_items.append(f"- {phenotype['term']} ({phenotype['hp_id']})")
        
        context_text = "\n".join(context_items)
        
        # Create prompt for the LLM
        context_part = ""
        if context and self.config.use_context_for_extract:
            context_part = f"Original sentence context: '{context}'\n\n"
        
        retrieval_part = ""
        if self.config.use_retrieval_for_extract and context_items:
            retrieval_part = (
                f"Here are some phenotype terms for context:\n\n"
                f"{context_text}\n\n"
            )
        
        prompt = (
            f"The term '{entity}' implies a phenotype but is not a direct phenotype itself. "
            f"{context_part}"
            f"{retrieval_part}"
            f"What specific phenotype is implied by '{entity}'? "
            f"For example, 'hemoglobin of 8 g/dL' implies 'anemia'.\n\n"
            f"Provide ONLY the name of the implied phenotype, without any explanation. "
            f"Use standard medical terminology."
        )
        
        # Query the LLM
        response = self.llm_client.query(prompt, self.extract_phenotype_system_message)
        
        # Clean the response
        implied_phenotype = response.strip()
        implied_phenotype = re.sub(r'[.,;:]$', '', implied_phenotype)
        
        # Create result
        result = {
            'implied_phenotype': implied_phenotype,
            'confidence': 0.8 if implied_phenotype else 0.0,
            'method': 'llm_extraction'
        }
        
        # Cache the result
        self.extracted_phenotype_cache[cache_key] = result
        
        self._debug_print(f"LLM extracted implied phenotype from '{entity}': '{implied_phenotype}'", level=2)
        return result

    def validate_phenotype_exists(self, phenotype: str) -> Dict:
        """
        Validate if a phenotype exists as a recognized medical concept with configurable retrieval.
        
        Args:
            phenotype: The phenotype to validate
            
        Returns:
            Dictionary with validation results
        """
        # Skip empty inputs
        if not phenotype:
            return {
                'is_valid': False,
                'confidence': 1.0,
                'method': 'empty_input'
            }
            
        # Create a cache key
        cache_key = f"validate_phenotype::{phenotype}"
        
        # Check cache
        if cache_key in self.phenotype_validation_cache:
            result = self.phenotype_validation_cache[cache_key]
            self._debug_print(f"Cache hit for phenotype validation '{phenotype}': {result['is_valid']}", level=1)
            return result
            
        self._debug_print(f"Validating if phenotype '{phenotype}' exists", level=1)
        
        # Retrieve similar phenotypes for context if configured
        similar_phenotypes = []
        if self.config.use_retrieval_for_validation:
            similar_phenotypes = self._retrieve_similar_phenotypes(phenotype)
        
        # Format context items if using retrieval
        context_items = []
        if self.config.use_retrieval_for_validation:
            for similar_pheno in similar_phenotypes[:10]:  # Use top 10 for context
                context_items.append(f"- {similar_pheno['term']} ({similar_pheno['hp_id']})")
        
        context_text = "\n".join(context_items)
        
        # Create prompt for the LLM
        retrieval_part = ""
        if self.config.use_retrieval_for_validation and context_items:
            retrieval_part = (
                f"Here are some similar phenotype terms for context:\n\n"
                f"{context_text}\n\n"
            )
        
        prompt = (
            f"I need to validate whether '{phenotype}' is a valid, recognized phenotype in clinical medicine.\n\n"
            f"{retrieval_part}"
            f"Based on {'this context and ' if retrieval_part else ''}your clinical knowledge, is '{phenotype}' a valid medical phenotype concept?\n\n"
            f"Respond with ONLY 'YES' if it's a valid phenotype or 'NO' if it's not."
        )
        
        # Query the LLM
        response = self.llm_client.query(prompt, self.phenotype_validation_system_message)
        
        # Parse the response
        is_valid = "yes" in response.lower() and "no" not in response.lower()
        
        # Create result
        result = {
            'is_valid': is_valid,
            'confidence': 0.9 if is_valid else 0.8,
            'method': 'llm_validation'
        }
        
        # Cache the result
        self.phenotype_validation_cache[cache_key] = result
        
        self._debug_print(f"Phenotype '{phenotype}' is{'' if is_valid else ' not'} valid", level=2)
        return result

    def validate_implication(self, entity: str, implied_phenotype: str, context: Optional[str] = None) -> Dict:
        """
        Validate if the implication from entity to phenotype is reasonable with configurable context usage.
        
        Args:
            entity: Original entity text
            implied_phenotype: Extracted implied phenotype 
            context: Original sentence containing the entity
            
        Returns:
            Dictionary with validation results
        """
        # Skip empty inputs
        if not entity or not implied_phenotype:
            return {
                'is_valid': False,
                'confidence': 1.0,
                'method': 'empty_input'
            }
            
        # Create a cache key - only include context if configured to use it
        cache_key = f"validate_implication::{entity}::{implied_phenotype}::{context if self.config.use_context_for_implication else ''}"
        
        # Check cache
        if cache_key in self.implication_validation_cache:
            result = self.implication_validation_cache[cache_key]
            self._debug_print(f"Cache hit for implication validation '{entity}' → '{implied_phenotype}': {result['is_valid']}", level=1)
            return result
            
        self._debug_print(f"Validating implication from '{entity}' to '{implied_phenotype}'", level=1)
        
        # Create prompt for the LLM
        context_part = ""
        if context and self.config.use_context_for_implication:
            context_part = f"Original sentence context: '{context}'\n\n"
        
        prompt = (
            f"I need to validate whether the following implication is reasonable:\n\n"
            f"Original entity: '{entity}'\n"
            f"Implied phenotype: '{implied_phenotype}'\n\n"
            f"{context_part}"
            f"Is this a valid and reasonable implication based on clinical knowledge? "
            f"Remember to be conservative - only approve implications with strong clinical justification.\n\n"
            f"Respond with ONLY 'YES' if the implication is valid or 'NO' if it's not valid."
        )
        
        # Query the LLM
        response = self.llm_client.query(prompt, self.implication_validation_system_message)
        
        # Parse the response
        is_valid = "yes" in response.lower() and "no" not in response.lower()
        
        # Create result
        result = {
            'is_valid': is_valid,
            'confidence': 0.9 if is_valid else 0.8,
            'method': 'llm_validation'
        }
        
        # Cache the result
        self.implication_validation_cache[cache_key] = result
        
        self._debug_print(f"Implication from '{entity}' to '{implied_phenotype}' is{'' if is_valid else ' not'} valid", level=2)
        return result

    def process_entity(self, entity: str, context: Optional[str] = None) -> Dict:
        """
        Process an entity through the multi-stage pipeline with configurable components.
        
        Args:
            entity: Entity text to process
            context: Original sentence containing the entity
            
        Returns:
            Dictionary with processing results
        """
        # Handle empty entities
        if not entity:
            return {
                'status': 'not_phenotype',
                'phenotype': None,
                'original_entity': entity,
                'confidence': 1.0,
                'method': 'empty_entity'
            }
            
        self._debug_print(f"Processing entity: '{entity}'", level=0)
        
        # Clean and preprocess the entity
        cleaned_entity = self.preprocess_entity(entity)
        if not cleaned_entity:
            return {
                'status': 'not_phenotype',
                'phenotype': None,
                'original_entity': entity,
                'confidence': 1.0,
                'method': 'empty_after_preprocessing'
            }
        
        # STAGE 1: Check if it's a direct phenotype
        direct_result = self.verify_direct_phenotype(cleaned_entity, context)
        
        # If it's a direct phenotype, return it with details
        if direct_result.get('is_phenotype', False):
            self._debug_print(f"'{entity}' is a direct phenotype", level=1)
            result = {
                'status': 'direct_phenotype',
                'phenotype': cleaned_entity,
                'original_entity': entity,
                'confidence': direct_result['confidence'],
                'method': direct_result['method']
            }
            
            if 'hp_id' in direct_result:
                result['hp_id'] = direct_result['hp_id']
                result['matched_term'] = direct_result['matched_term']
                
            return result
        
        # STAGE 2: Check if it implies a phenotype
        implies_result = self.check_implies_phenotype(cleaned_entity, context)
        
        if not implies_result.get('implies_phenotype', False):
            self._debug_print(f"'{entity}' is not a phenotype and doesn't imply one", level=1)
            return {
                'status': 'not_phenotype',
                'phenotype': None,
                'original_entity': entity,
                'confidence': implies_result['confidence'],
                'method': implies_result.get('method', 'llm_verification')
            }
            
        # STAGE 3: Extract the implied phenotype
        extract_result = self.extract_implied_phenotype(cleaned_entity, context)
        implied_phenotype = extract_result.get('implied_phenotype')
        
        # If we couldn't extract an implied phenotype, not a phenotype
        if not implied_phenotype:
            return {
                'status': 'not_phenotype',
                'phenotype': None,
                'original_entity': entity,
                'confidence': 0.7,
                'method': 'no_implied_phenotype_found'
            }
        
        # STAGE 4: Validate if the implication is reasonable
        implication_validation_result = self.validate_implication(cleaned_entity, implied_phenotype, context)
        
        if not implication_validation_result.get('is_valid', False):
            self._debug_print(f"Implication from '{entity}' to '{implied_phenotype}' is not valid", level=1)
            return {
                'status': 'not_phenotype',
                'phenotype': None,
                'original_entity': entity,
                'confidence': implication_validation_result['confidence'],
                'method': 'invalid_implication'
            }
        
        # STAGE 5: Validate if the implied phenotype exists as a recognized medical concept
        phenotype_validation_result = self.validate_phenotype_exists(implied_phenotype)
        
        if not phenotype_validation_result.get('is_valid', False):
            self._debug_print(f"Implied phenotype '{implied_phenotype}' from '{entity}' is not valid", level=1)
            return {
                'status': 'not_phenotype',
                'phenotype': None,
                'original_entity': entity,
                'confidence': phenotype_validation_result['confidence'],
                'method': 'invalid_phenotype'
            }
        
        # If implication and phenotype are both valid, return the implied phenotype
        self._debug_print(f"'{entity}' implies valid phenotype '{implied_phenotype}'", level=1)
        result = {
            'status': 'implied_phenotype',
            'phenotype': implied_phenotype,
            'original_entity': entity,
            'confidence': extract_result['confidence'],
            'method': 'multi_stage_pipeline'
        }
            
        return result
    
    def batch_process(self, entities_with_context: List[Dict]) -> List[Dict]:
        """
        Process a batch of entities with their contexts through the multi-stage pipeline.
        
        Args:
            entities_with_context: List of dicts with 'entity' and 'context' keys
            
        Returns:
            List of dicts with processing results (phenotypes only)
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
            
            # Only include entities that are phenotypes (direct or implied)
            if result['status'] in ['direct_phenotype', 'implied_phenotype']:
                # Add original context
                result['context'] = context
                results.append(result)
        
        self._debug_print(f"Identified {len(results)} phenotypes (direct or implied)")
        return results

class HPOConfigEvaluator:
    """Evaluates different configurations of the HPO verifier."""
    
    def __init__(self, embedding_manager, llm_client, ground_truth, extracted_entities, debug=False):
        """
        Initialize the evaluator.
        
        Args:
            embedding_manager: Embedding manager for vectorization
            llm_client: LLM client for queries
            ground_truth: List of ground truth phenotypes
            extracted_entities: List of dictionaries with 'entity' and 'context'
            debug: Whether to print debug information
        """
        self.embedding_manager = embedding_manager
        self.llm_client = llm_client
        self.ground_truth = ground_truth
        self.extracted_entities = extracted_entities
        self.debug = debug
        self.embedded_documents = None
        self.verifier = None
        
    def prepare(self, embedded_documents):
        """Prepare the evaluator with embedded documents."""
        self.embedded_documents = embedded_documents
        self.verifier = ConfigurableHPOVerifier(
            self.embedding_manager, 
            self.llm_client,
            debug=self.debug
        )
        self.verifier.prepare_index(embedded_documents)
        
    def evaluate_config(self, config, n_runs=3):
        """
        Evaluate a configuration by running the pipeline and measuring performance.
        
        Args:
            config: HPOVerifierConfig to evaluate
            n_runs: Number of runs to average over
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.verifier:
            raise ValueError("Evaluator not prepared. Call prepare() first.")
            
        self.verifier.set_config(config)
        metrics_list = []
        
        for i in range(n_runs):
            self.verifier.clear_caches()
            results = self.verifier.batch_process(self.extracted_entities)
            
            # Extract phenotypes for evaluation
            phenotypes = [entity["phenotype"] for entity in results]
            
            # Calculate metrics
            metrics = set_based_evaluation(phenotypes, self.ground_truth, similarity_threshold=50.0)
            metrics_list.append(metrics)
            
            if self.debug:
                print(f"Run {i+1}: {metrics}")
        
        # Average the metrics
        avg_metrics = {}
        for key in metrics_list[0].keys():
            if key != "matches":  # Skip the matches list when averaging
                avg_metrics[key] = sum(m[key] for m in metrics_list) / len(metrics_list)
            
        # Add the configuration to the metrics
        avg_metrics["config"] = config.to_dict()
        
        return avg_metrics

def set_based_evaluation(predicted, ground_truth, similarity_threshold=50.0):
    """Evaluate predictions using fuzzy set-based metrics."""
    # Helper function to find matching pairs
    def find_matches(preds, gt, threshold):
        matches = []
        used_gt = set()
        for pred in preds:
            best_score = -1
            best_match = None
            for gt_item in gt:
                if gt_item in used_gt:
                    continue
                score = fuzz.ratio(pred.lower(), gt_item.lower())
                if score > threshold and score > best_score:
                    best_score = score
                    best_match = gt_item
            if best_match:
                matches.append((pred, best_match))
                used_gt.add(best_match)
        return matches

    # Find matched pairs
    matches = find_matches(predicted, ground_truth, similarity_threshold)
    
    # Calculate metrics
    tp = len(matches)
    fp = len(predicted) - tp
    fn = len(ground_truth) - tp
    
    # avoid division by zero
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "matches": matches
    }

class HPOConfigSearch:
    """Search for the best HPO verifier configuration."""
    
    def __init__(self, evaluator):
        """
        Initialize the search.
        
        Args:
            evaluator: HPOConfigEvaluator instance
        """
        self.evaluator = evaluator
        self.results = []
        
    def grid_search(self, params_to_tune=None, n_runs=3):
        """
        Perform grid search over configuration space.
        
        Args:
            params_to_tune: List of parameter names to tune, or None for all
            n_runs: Number of runs per configuration
            
        Returns:
            Best configuration based on F1 score
        """
        # Default: tune all parameters
        if params_to_tune is None:
            params_to_tune = [
                "use_retrieval_for_direct",
                "use_retrieval_for_implies",
                "use_retrieval_for_extract",
                "use_retrieval_for_validation",
                "use_retrieval_for_implication",
                "use_context_for_direct",
                "use_context_for_implies",
                "use_context_for_extract",
                "use_context_for_validation",
                "use_context_for_implication"
            ]
        
        # Generate all combinations of parameters
        param_values = {param: [True, False] for param in params_to_tune}
        configs = self._generate_configs(param_values)
        
        print(f"Evaluating {len(configs)} configurations...")
        
        # Evaluate each configuration
        for i, config in enumerate(configs):
            print(f"Evaluating configuration {i+1}/{len(configs)}: {config}")
            metrics = self.evaluator.evaluate_config(config, n_runs=n_runs)
            
            # Save results
            self.results.append({
                "config": config.to_dict(),
                "metrics": {k: v for k, v in metrics.items() if k != "config"}
            })
            
            print(f"F1: {metrics.get('f1', 0):.4f}, Precision: {metrics.get('precision', 0):.4f}, Recall: {metrics.get('recall', 0):.4f}")
        
        # Sort by F1 score
        self.results.sort(key=lambda x: x["metrics"].get("f1", 0), reverse=True)
        
        return HPOVerifierConfig.from_dict(self.results[0]["config"])
    
    def _generate_configs(self, param_values):
        """Generate all combinations of parameter values."""
        # Get default config
        default_config = HPOVerifierConfig()
        default_dict = default_config.to_dict()
        
        # Flatten the nested dict
        flat_params = {}
        for category in ["retrieval", "context"]:
            for param, value in default_dict[category].items():
                flat_param = f"use_{category}_for_{param}"
                flat_params[flat_param] = value
        
        # Generate parameter combinations for parameters being tuned
        param_names = list(param_values.keys())
        value_combos = list(itertools.product(*(param_values[name] for name in param_names)))
        
        configs = []
        for values in value_combos:
            # Start with default values
            config_params = flat_params.copy()
            
            # Override with tuned values
            for name, value in zip(param_names, values):
                config_params[name] = value
            
            # Reconstruct nested dict
            config_dict = {"retrieval": {}, "context": {}}
            for param, value in config_params.items():
                if param.startswith("use_retrieval_for_"):
                    key = param.replace("use_retrieval_for_", "")
                    config_dict["retrieval"][key] = value
                elif param.startswith("use_context_for_"):
                    key = param.replace("use_context_for_", "")
                    config_dict["context"][key] = value
            
            configs.append(HPOVerifierConfig.from_dict(config_dict))
        
        return configs
    
    def save_results(self, filename):
        """Save search results to a file."""
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=4)

class HPOPipelineOptimizer:
    """A combined class for HPO pipeline optimization."""
    
    def __init__(self, embedding_manager, llm_client, ground_truth, extracted_entities, 
                 embedded_documents, debug=False):
        """
        Initialize the optimizer.
        
        Args:
            embedding_manager: Embedding manager for vectorization
            llm_client: LLM client for queries
            ground_truth: List of ground truth phenotypes 
            extracted_entities: List of dicts with 'entity' and 'context'
            embedded_documents: Embedded HPO documents
            debug: Whether to print debug information
        """
        self.embedding_manager = embedding_manager
        self.llm_client = llm_client
        self.ground_truth = ground_truth
        self.extracted_entities = extracted_entities
        self.embedded_documents = embedded_documents
        self.debug = debug
        
        # Create evaluator
        self.evaluator = HPOConfigEvaluator(
            embedding_manager, 
            llm_client, 
            ground_truth,
            extracted_entities,
            debug
        )
        self.evaluator.prepare(embedded_documents)
        
        # Create searcher
        self.searcher = HPOConfigSearch(self.evaluator)
        
    def optimize(self, params_to_tune=None, n_runs=3):
        """
        Run optimization to find the best configuration.
        
        Args:
            params_to_tune: List of parameters to tune, or None for all
            n_runs: Number of runs per configuration
            
        Returns:
            Tuple of (best_config, best_metrics)
        """
        best_config = self.searcher.grid_search(params_to_tune, n_runs)
        
        # Get metrics for the best config
        best_result = self.searcher.results[0]
        best_metrics = best_result["metrics"]
        
        print("\nBest Configuration:")
        print(json.dumps(best_config.to_dict(), indent=4))
        print("\nBest Metrics:")
        print(json.dumps(best_metrics, indent=4))
        
        return best_config, best_metrics
    
    def save_results(self, filename):
        """Save optimization results to a file."""
        self.searcher.save_results(filename)
        
    def get_verifier_with_best_config(self):
        """Get a verifier instance with the best configuration."""
        if not self.searcher.results:
            raise ValueError("No optimization results available. Run optimize() first.")
            
        best_config = HPOVerifierConfig.from_dict(self.searcher.results[0]["config"])
        
        verifier = ConfigurableHPOVerifier(
            self.embedding_manager,
            self.llm_client,
            config=best_config,
            debug=self.debug
        )
        verifier.prepare_index(self.embedded_documents)
        
        return verifier

# Example usage
def run_optimization(embedding_manager, llm_client, ground_truth, extracted_entities, embedded_documents, debug=True):
    """
    Run the optimization process to find the best HPO verifier configuration.
    
    Args:
        embedding_manager: Embedding manager for vectorization
        llm_client: LLM client for queries
        ground_truth: List of ground truth phenotypes
        extracted_entities: List of dictionaries with 'entity' and 'context'
        embedded_documents: Embedded HPO documents
        debug: Whether to print debug information
        
    Returns:
        ConfigurableHPOVerifier instance with the best configuration
    """
    # Create optimizer
    optimizer = HPOPipelineOptimizer(
        embedding_manager=embedding_manager,
        llm_client=llm_client,
        ground_truth=ground_truth,
        extracted_entities=extracted_entities,
        embedded_documents=embedded_documents,
        debug=debug
    )
    
    # For full grid search (all parameters - can be very slow)
    # best_config, best_metrics = optimizer.optimize(n_runs=3)
    
    # For reduced parameter search (faster)
    params_to_tune = [
        "use_retrieval_for_direct",
        "use_retrieval_for_implies",
        "use_retrieval_for_extract",
        "use_retrieval_for_validation",
        "use_retrieval_for_implication",
        "use_context_for_direct",
        "use_context_for_implies",
        "use_context_for_extract",
        "use_context_for_validation",
        "use_context_for_implication"
    ]
    best_config, best_metrics = optimizer.optimize(params_to_tune=params_to_tune, n_runs=3)
    
    # Save results
    optimizer.save_results("hpo_optimization_results.json")
    
    # Get verifier with best config
    return optimizer.get_verifier_with_best_config()


from typing import List, Dict, Any, Optional, Tuple, Union, Set
import json
import re
import numpy as np
from datetime import datetime
from fuzzywuzzy import fuzz


class MultiStageHPOVerifierV2:
    """
    An improved version of the ConfigurableHPOVerifier with enhanced prompts for better precision.
    This version is designed to be more conservative in identifying phenotypes, especially implied ones.
    """
    
    def __init__(self, embedding_manager, llm_client, config=None, debug=False):
        """Initialize with a specific configuration."""
        self.embedding_manager = embedding_manager
        self.llm_client = llm_client
        self.debug = debug
        self.index = None
        self.embedded_documents = None
        self.config = config or HPOVerifierConfig()
        
        # Improved system messages with clearer examples and more conservative thresholds
        
        # Direct verification - clearer examples and criteria
        self.direct_verification_system_message = (
            "You are a clinical expert specializing in phenotype identification for Human Phenotype Ontology (HPO) mapping. "
            "Determine if a given term from a clinical note describes a valid human phenotype "
            "(an observable characteristic, trait, or abnormality). "
            "\nA valid phenotype must describe an abnormal characteristic, not just a normal anatomical structure or physiological process."
            "\nEXAMPLES OF VALID PHENOTYPES:"
            "\n- 'Macrocephaly' (YES - describes abnormal head size)"
            "\n- 'Seizure' (YES - describes abnormal neurological event)"
            "\n- 'Hepatomegaly' (YES - describes abnormal liver condition)"
            "\n- 'Congenital heart defect' (YES - describes an abnormality)"
            "\n\nEXAMPLES OF INVALID PHENOTYPES:"
            "\n- 'Brain' (NO - just an anatomical structure)"
            "\n- 'Blood pressure' (NO - just a physiological measurement)"
            "\n- 'Medication' (NO - just a treatment)"
            "\n- 'Kidney' (NO - just an organ, no abnormality described)"
            "\n- 'Liver function tests' (NO - diagnostic procedure, not a phenotype)"
            "\n\nRespond with ONLY 'YES' if the term is a valid phenotype, or 'NO' if it's not a phenotype."
        )
        
        # Implied phenotype - much more conservative, clear examples of invalid implications
        self.implied_phenotype_system_message = (
            "You are a clinical expert specializing in phenotype identification for Human Phenotype Ontology (HPO) mapping. "
            "Determine if the given term DIRECTLY AND UNAMBIGUOUSLY implies a specific phenotype, even though it's not a direct phenotype itself. "
            "Be extremely conservative - only say YES if the implication is clear and specific."
            "\nLaboratory values, medications, or procedures DO NOT imply phenotypes unless there is explicit abnormality mentioned."
            "\nIf you're uncertain or the implication requires multiple assumptions, say NO."
            "\n\nEXAMPLES OF VALID IMPLICATIONS:"
            "\n- 'Elevated white blood cell count of 15,000' implies 'leukocytosis' (YES - explicit abnormality)"
            "\n- 'E. coli growing in urine culture' implies 'bacteriuria' (YES - specific finding)"
            "\n- 'Hemoglobin of 6.5 g/dL' implies 'anemia' (YES - clearly below normal range)"
            "\n\nEXAMPLES OF INVALID IMPLICATIONS:"
            "\n- 'White blood cell count' does NOT imply 'leukocytosis' (NO - no value specified)"
            "\n- 'Taking insulin' does NOT imply 'diabetes mellitus' (NO - medications alone don't imply diagnoses)"
            "\n- 'Kidney biopsy' does NOT imply 'nephropathy' (NO - diagnostic procedure without findings)"
            "\n- 'Heart murmur' does NOT imply 'congenital heart defect' (NO - too specific without evidence)"
            "\n\nRespond with ONLY 'YES' if the term directly implies a phenotype, or 'NO' if it doesn't."
        )
        
        # Extract phenotype - added option to indicate no clear phenotype implied
        self.extract_phenotype_system_message = (
            "You are a clinical expert specializing in phenotype identification for Human Phenotype Ontology (HPO) mapping. "
            "A previous analysis determined that a given term might imply a phenotype but is not a direct phenotype itself. "
            "Your task is to precisely identify what specific phenotype is implied by this term. "
            "\nProvide ONLY the name of the implied phenotype as it would appear in medical terminology. "
            "Be specific and concise. Do not include explanations or multiple options."
            "\n\nEXAMPLES:"
            "\n- 'Elevated white blood cell count of 15,000' implies 'leukocytosis'"
            "\n- 'E. coli growing in urine culture' implies 'bacteriuria'"
            "\n- 'Hemoglobin of 6.5 g/dL' implies 'anemia'"
            "\n\nIf you cannot identify a specific phenotype that is DIRECTLY implied with high confidence, "
            "respond with EXACTLY 'NO_CLEAR_PHENOTYPE_IMPLIED'."
        )
        
        # Implication validation - stronger criteria for validation
        self.implication_validation_system_message = (
            "You are a clinical expert specializing in phenotype identification for Human Phenotype Ontology (HPO) mapping. "
            "Your task is to validate whether an implied phenotype is reasonable given the original entity and its context. "
            "Be extremely critical and conservative in your assessment. Say NO unless there is an unambiguous, "
            "direct clinical connection between the entity and the proposed phenotype."
            "\nThe connection must be evident from the entity itself, not inferred from general knowledge."
            "\n\nEXAMPLES of VALID implications:"
            "\n- Entity: 'E. coli in urine culture' → Implied phenotype: 'bacteriuria' (VALID: specific finding)"
            "\n- Entity: 'Hemoglobin of 6.5 g/dL' → Implied phenotype: 'anemia' (VALID: specific abnormal value)"
            "\n\nEXAMPLES of INVALID implications:"
            "\n- Entity: 'white blood cell count' → Implied phenotype: 'leukocytosis' (INVALID: no value specified)"
            "\n- Entity: 'taking insulin daily' → Implied phenotype: 'diabetes mellitus' (INVALID: medication alone)"
            "\n- Entity: 'retina' → Implied phenotype: 'retinopathy' (INVALID: normal anatomy without abnormality)"
            "\n- Entity: 'renal tissue' → Implied phenotype: 'glomerulonephritis' (INVALID: too specific without evidence)"
            "\n\nRespond with ONLY 'YES' if the implication is valid, or 'NO' if it's not valid based on the original term and context."
        )
        
        # Phenotype validation - clearer criteria for valid phenotypes
        self.phenotype_validation_system_message = (
            "You are a clinical expert specializing in phenotype identification. "
            "Your task is to validate whether a proposed phenotype is a valid, specific medical concept. "
            "Focus only on whether the term represents a real, recognized phenotype in clinical medicine. "
            "A valid phenotype must be specific enough to be clinically meaningful."
            "\n\nEXAMPLES of VALID phenotypes:"
            "\n- 'bacteriuria' (VALID: recognized specific condition of bacteria in urine)"
            "\n- 'leukocytosis' (VALID: specific condition of elevated white blood cells)"
            "\n- 'macrocephaly' (VALID: recognized condition of abnormally large head)"
            "\n\nEXAMPLES of INVALID phenotypes:"
            "\n- 'blood abnormality' (INVALID: too vague, not a specific phenotype)"
            "\n- 'kidney disease' (INVALID: too general, not a specific phenotype)"
            "\n- 'medication response' (INVALID: too generic, not a specific phenotype)"
            "\n- 'lab test issue' (INVALID: not a specific phenotype)"
            "\n- 'immune system problem' (INVALID: too vague, not a specific condition)"
            "\n\nRespond with ONLY 'YES' if the term is a valid, specific, recognized phenotype, or 'NO' if it's not."
        )
        
        # Additional term filtering system message - new stage for checking common false positives
        self.term_filtering_system_message = (
            "You are a clinical expert specializing in phenotype identification. "
            "Your task is to identify terms that are commonly mistaken for phenotypes but are not. "
            "These include normal anatomical structures, lab tests without values, medications, or general clinical terms."
            "\n\nIdentify if the given term falls into one of these categories that are NOT phenotypes:"
            "\n1. Normal anatomical structure (e.g., 'liver', 'kidney', 'brain')"
            "\n2. Lab test or measurement without values (e.g., 'blood pressure', 'white blood cell count')"
            "\n3. Medication or treatment (e.g., 'insulin', 'furosemide', 'antibiotic')"
            "\n4. Medical procedure (e.g., 'biopsy', 'MRI', 'ultrasound')"
            "\n5. General medical concept (e.g., 'follow-up', 'assessment', 'plan')"
            "\n\nRespond with 'FILTER_OUT' if the term should be filtered out as a non-phenotype, "
            "or 'KEEP' if it does not clearly fall into one of these categories."
        )
        
        # Caches to avoid redundant API calls
        self.verification_cache = {}
        self.implied_phenotype_cache = {}
        self.extracted_phenotype_cache = {}
        self.implication_validation_cache = {}
        self.phenotype_validation_cache = {}
        self.term_filtering_cache = {}  # New cache for term filtering stage
    
    def _debug_print(self, message: str, level: int = 0):
        """Print debug messages if debug mode is enabled."""
        if self.debug:
            indent = "  " * level
            print(f"{datetime.now().strftime('%H:%M:%S')} | {indent}{message}")
    
    def set_config(self, config):
        """Update the configuration."""
        self.config = config
        self.clear_caches()  # Clear caches when configuration changes
        
    def prepare_index(self, metadata: List[Dict]):
        """Prepare FAISS index from metadata for similarity search."""
        if self.index is None:
            self._debug_print("Preparing FAISS index for phenotype verification...")
            embeddings_array = self.embedding_manager.prepare_embeddings(metadata)
            self.index = self.embedding_manager.create_index(embeddings_array)
            self.embedded_documents = metadata
            self._debug_print(f"Index prepared with {len(metadata)} embedded documents")

    def clear_caches(self):
        """Clear all caches to prepare for a fresh evaluation run."""
        self.verification_cache = {}
        self.implied_phenotype_cache = {}
        self.extracted_phenotype_cache = {}
        self.implication_validation_cache = {}
        self.phenotype_validation_cache = {}
        self.term_filtering_cache = {}
        self._debug_print("All caches cleared")

    def _retrieve_similar_phenotypes(self, entity: str, k: int = 10) -> List[Dict]:
        """Retrieve similar phenotypes from the HPO ontology for context."""
        if self.index is None:
            raise ValueError("Index not prepared. Call prepare_index() first.")
            
        # Embed the query
        query_vector = self.embedding_manager.query_text(entity).reshape(1, -1)
        
        # Search for similar items
        distances, indices = self.embedding_manager.search(query_vector, self.index, k=min(800, len(self.embedded_documents)))
        
        # Extract unique metadata
        similar_phenotypes = []
        seen_metadata = set()
        
        for idx, distance in zip(indices[0], distances[0]):
            metadata = self.embedded_documents[idx]['unique_metadata']
            metadata_str = json.dumps(metadata)
            
            if metadata_str not in seen_metadata:
                seen_metadata.add(metadata_str)
                similar_phenotypes.append({
                    'term': metadata.get('info', ''),
                    'hp_id': metadata.get('hp_id', ''),
                    'similarity_score': 1.0 / (1.0 + distance)  # Convert distance to similarity
                })
                
                if len(similar_phenotypes) >= k:
                    break
                    
        return similar_phenotypes
    
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
    
    def pre_filter_term(self, entity: str, context: Optional[str] = None) -> Dict:
        """
        New stage: Pre-filter common non-phenotype terms before expensive verification steps.
        
        Args:
            entity: Entity text to filter
            context: Original sentence containing the entity
            
        Returns:
            Dictionary with filtering results
        """
        # Handle empty entities
        if not entity:
            return {
                'should_filter': True,
                'confidence': 1.0,
                'method': 'empty_entity'
            }
            
        # Create a cache key
        cache_key = f"filter::{entity}"
        
        # Check cache first
        if cache_key in self.term_filtering_cache:
            result = self.term_filtering_cache[cache_key]
            self._debug_print(f"Cache hit for term filtering '{entity}': {result['should_filter']}", level=1)
            return result
            
        self._debug_print(f"Pre-filtering term '{entity}'", level=1)
        
        # Simple rule-based filtering for common patterns
        lower_entity = entity.lower()
        
        # List of terms that are clearly not phenotypes
        common_non_phenotypes = [
            'medication', 'drug', 'dose', 'therapy', 'treatment', 'rx', 'prescription',
            'lab', 'test', 'scan', 'mri', 'ct', 'ultrasound', 'biopsy', 'culture',
            'consult', 'consultation', 'follow-up', 'follow up', 'evaluation', 'assessment', 'plan', 
            'history', 'hx', 'family history', 'social history', 'allergies',
            'monitor', 'monitoring', 'check'
        ]
        
        # Check for exact matches to common non-phenotypes
        for term in common_non_phenotypes:
            if lower_entity == term or f"{term}s" == lower_entity:  # Also check plural form
                self._debug_print(f"Term '{entity}' filtered by rule-based system (common non-phenotype)", level=2)
                result = {
                    'should_filter': True,
                    'confidence': 0.9,
                    'method': 'rule_based_filtering'
                }
                self.term_filtering_cache[cache_key] = result
                return result
        
        # Send to LLM for more nuanced filtering if not caught by rules
        prompt = (
            f"I need to determine if the term '{entity}' is of a category that is NOT a phenotype. "
            f"Categories that are NOT phenotypes include: "
            f"1. Normal anatomical structures (e.g., 'liver', 'kidney')\n"
            f"2. Lab tests or measurements without values (e.g., 'blood pressure')\n"
            f"3. Medications or treatments (e.g., 'insulin', 'antibiotic')\n"
            f"4. Medical procedures (e.g., 'biopsy', 'MRI')\n"
            f"5. General medical concepts (e.g., 'follow-up', 'assessment')\n\n"
            f"Does '{entity}' clearly fall into one of these categories?\n"
            f"Respond with ONLY 'FILTER_OUT' if it should be filtered out as a non-phenotype, "
            f"or 'KEEP' if it does not clearly fall into one of these categories."
        )
        
        # Query the LLM
        response = self.llm_client.query(prompt, self.term_filtering_system_message)
        
        # Parse the response
        should_filter = "filter_out" in response.lower() and "keep" not in response.lower()
        
        # Create result
        result = {
            'should_filter': should_filter,
            'confidence': 0.8,
            'method': 'llm_filtering'
        }
        
        # Cache the result
        self.term_filtering_cache[cache_key] = result
        
        self._debug_print(f"LLM says term '{entity}' should{' ' if should_filter else ' not '}be filtered", level=2)
        return result

    def verify_direct_phenotype(self, entity: str, context: Optional[str] = None) -> Dict:
        """
        Verify if an entity is a direct phenotype with configurable retrieval and context usage.
        
        Args:
            entity: Entity text to verify
            context: Original sentence containing the entity
            
        Returns:
            Dictionary with verification results
        """
        # Handle empty entities
        if not entity:
            return {
                'is_phenotype': False,
                'confidence': 1.0,
                'method': 'empty_entity'
            }
            
        # Create a cache key - only include context if configured to use it
        cache_key = f"direct::{entity}::{context if self.config.use_context_for_direct else ''}"
        
        # Check cache first
        if cache_key in self.verification_cache:
            result = self.verification_cache[cache_key]
            self._debug_print(f"Cache hit for direct phenotype '{entity}': {result['is_phenotype']}", level=1)
            return result
            
        self._debug_print(f"Verifying if '{entity}' is a direct phenotype", level=1)
        
        # Retrieve similar phenotypes for context if configured
        similar_phenotypes = []
        if self.config.use_retrieval_for_direct:
            similar_phenotypes = self._retrieve_similar_phenotypes(entity)
            
            # Check for exact matches first (optimization)
            for phenotype in similar_phenotypes:
                if self._normalize_text(phenotype['term']) == self._normalize_text(entity):
                    self._debug_print(f"Exact match found: '{entity}' matches '{phenotype['term']}' ({phenotype['hp_id']})", level=2)
                    result = {
                        'is_phenotype': True,
                        'confidence': 1.0,
                        'method': 'exact_match',
                        'hp_id': phenotype['hp_id'],
                        'matched_term': phenotype['term']
                    }
                    self.verification_cache[cache_key] = result
                    return result
        
        # Format context items if using retrieval
        context_items = []
        if self.config.use_retrieval_for_direct:
            for phenotype in similar_phenotypes[:10]:  # Use top 10 for context
                context_items.append(f"- {phenotype['term']} ({phenotype['hp_id']})")
            
        context_text = "\n".join(context_items)
        
        # Create prompt for the LLM, including the sentence context if configured
        context_part = ""
        if context and self.config.use_context_for_direct:
            context_part = f"Original sentence context: '{context}'\n\n"
        
        retrieval_part = ""
        if self.config.use_retrieval_for_direct and context_items:
            retrieval_part = (
                f"Here are some retrieved candidates from the Human Phenotype Ontology to help you make your decision:\n\n"
                f"{context_text}\n\n"
            )
        
        prompt = (
            f"I need to determine if ENTITY:'{entity}' is a valid human phenotype. "
            f"{context_part}"
            f"{retrieval_part}"
            f"Based on {'these examples and ' if retrieval_part else ''}{'the original context' if context_part else 'your knowledge'}, "
            f"is just the ENTITY: '{entity}' a valid human phenotype? "
            f"A phenotype must describe an abnormal characteristic, not just a normal anatomical structure or measurement."
            f"\nRespond with ONLY 'YES' if the term is a valid phenotype, or 'NO' if it's not a phenotype."
        )
        
        # Query the LLM
        response = self.llm_client.query(prompt, self.direct_verification_system_message)
        
        # Parse the response
        is_phenotype = "yes" in response.lower() and "no" not in response.lower()
        
        # Create result
        if is_phenotype:
            result = {
                'is_phenotype': True, 
                'confidence': 0.8,
                'method': 'llm_verification'
            }
        else:
            result = {
                'is_phenotype': False,
                'confidence': 0.8,
                'method': 'llm_verification'
            }
        
        # Cache the result
        self.verification_cache[cache_key] = result
        
        self._debug_print(f"LLM says '{entity}' is{'' if is_phenotype else ' not'} a phenotype", level=2)
        return result

    def check_implies_phenotype(self, entity: str, context: Optional[str] = None) -> Dict:
        """
        Check if an entity implies a phenotype with configurable retrieval and context usage.
        More conservative criteria compared to V1.
        
        Args:
            entity: Entity text to check
            context: Original sentence containing the entity
            
        Returns:
            Dictionary with results
        """
        # Handle empty entities
        if not entity:
            return {
                'implies_phenotype': False,
                'confidence': 1.0,
                'method': 'empty_entity'
            }
            
        # Create a cache key - only include context if configured to use it
        cache_key = f"implies::{entity}::{context if self.config.use_context_for_implies else ''}"
        
        # Check cache
        if cache_key in self.implied_phenotype_cache:
            result = self.implied_phenotype_cache[cache_key]
            self._debug_print(f"Cache hit for implied phenotype check '{entity}': {result['implies_phenotype']}", level=1)
            return result
            
        self._debug_print(f"Checking if '{entity}' implies a phenotype", level=1)
        
        # Apply more conservative matching for common lab tests and medications
        lower_entity = entity.lower()
        
        # Conservative filtering for lab tests without values
        lab_tests = [
            'blood count', 'cbc', 'white blood cell', 'wbc', 'hemoglobin', 'hgb', 
            'platelet', 'sodium', 'potassium', 'glucose', 'a1c', 'creatinine', 'bun',
            'liver enzyme', 'alt', 'ast', 'albumin', 'troponin', 'cholesterol'
        ]
        
        # Check if entity is just a lab test without a value
        for test in lab_tests:
            if test in lower_entity:
                # Check if there's no value or indication of abnormality
                if not any(term in lower_entity for term in ['elevated', 'high', 'low', 'abnormal', 'increased', 'decreased']):
                    if not re.search(r'\d+(?:\.\d+)?', lower_entity):  # No numbers in the entity
                        self._debug_print(f"Entity '{entity}' is a lab test without a value, does not imply phenotype", level=2)
                        result = {
                            'implies_phenotype': False,
                            'confidence': 0.9,
                            'method': 'rule_based_filtering'
                        }
                        self.implied_phenotype_cache[cache_key] = result
                        return result
        
        # Retrieve similar phenotypes for context if configured
        similar_phenotypes = []
        if self.config.use_retrieval_for_implies:
            similar_phenotypes = self._retrieve_similar_phenotypes(entity)
        
        # Format context items if using retrieval
        context_items = []
        if self.config.use_retrieval_for_implies:
            for phenotype in similar_phenotypes[:10]:  # Use top 10 for context
                context_items.append(f"- {phenotype['term']} ({phenotype['hp_id']})")
        
        context_text = "\n".join(context_items)
        
        # Create prompt for the LLM
        context_part = ""
        if context and self.config.use_context_for_implies:
            context_part = f"Original sentence context: '{context}'\n\n"
        
        retrieval_part = ""
        if self.config.use_retrieval_for_implies and context_items:
            retrieval_part = (
                f"Here are some phenotype terms from the Human Phenotype Ontology for context:\n\n"
                f"{context_text}\n\n"
            )
        
        prompt = (
            f"I need to determine if '{entity}' DIRECTLY AND UNAMBIGUOUSLY implies a specific phenotype. "
            f"Be extremely conservative - only say YES if the implication is clear and specific."
            f"\n{context_part}"
            f"{retrieval_part}"
            f"Laboratory values, medications, or procedures DO NOT imply phenotypes unless there is explicit abnormality mentioned."
            f"\nIf you're uncertain or the implication requires multiple assumptions, say NO."
            f"\n\nDoes '{entity}' directly imply a specific phenotype? "
            f"Respond with ONLY 'YES' if it directly implies a phenotype or 'NO' if it doesn't."
        )
        
        # Query the LLM
        response = self.llm_client.query(prompt, self.implied_phenotype_system_message)
        
        # Parse the response
        implies_phenotype = "yes" in response.lower() and "no" not in response.lower()
        
        # Create result
        result = {
            'implies_phenotype': implies_phenotype,
            'confidence': 0.8 if implies_phenotype else 0.9,  # Higher confidence for "no" to be conservative
            'method': 'llm_verification'
        }
        
        # Cache the result
        self.implied_phenotype_cache[cache_key] = result
        
        self._debug_print(f"LLM says '{entity}' does{'' if implies_phenotype else ' not'} imply a phenotype", level=2)
        return result

    def extract_implied_phenotype(self, entity: str, context: Optional[str] = None) -> Dict:
        """
        Extract the specific phenotype implied by an entity with configurable retrieval and context usage.
        Enhanced to allow for "no clear phenotype" as a result.
        
        Args:
            entity: Entity text that implies a phenotype
            context: Original sentence containing the entity
            
        Returns:
            Dictionary with results
        """
        # Handle empty entities
        if not entity:
            return {
                'implied_phenotype': None,
                'confidence': 0.0,
                'method': 'empty_entity'
            }
            
        # Create a cache key - only include context if configured to use it
        cache_key = f"extract::{entity}::{context if self.config.use_context_for_extract else ''}"
        
        # Check cache
        if cache_key in self.extracted_phenotype_cache:
            result = self.extracted_phenotype_cache[cache_key]
            self._debug_print(f"Cache hit for extracting implied phenotype from '{entity}': {result.get('implied_phenotype')}", level=1)
            return result
            
        self._debug_print(f"Extracting implied phenotype from '{entity}'", level=1)
        
        # Retrieve similar phenotypes for context if configured
        similar_phenotypes = []
        if self.config.use_retrieval_for_extract:
            similar_phenotypes = self._retrieve_similar_phenotypes(entity)
        
        # Format context items if using retrieval
        context_items = []
        if self.config.use_retrieval_for_extract:
            for phenotype in similar_phenotypes[:10]:  # Use top 10 for context
                context_items.append(f"- {phenotype['term']} ({phenotype['hp_id']})")
        
        context_text = "\n".join(context_items)
        
        # Create prompt for the LLM
        context_part = ""
        if context and self.config.use_context_for_extract:
            context_part = f"Original sentence context: '{context}'\n\n"
        
        retrieval_part = ""
        if self.config.use_retrieval_for_extract and context_items:
            retrieval_part = (
                f"Here are some phenotype terms for context:\n\n"
                f"{context_text}\n\n"
            )
        
        prompt = (
            f"The term '{entity}' might imply a phenotype. "
            f"{context_part}"
            f"{retrieval_part}"
            f"What specific phenotype is directly implied by '{entity}'? "
            f"For example, 'hemoglobin of 8 g/dL' implies 'anemia'."
            f"\n\nIf you cannot identify a specific phenotype that is DIRECTLY implied with high confidence, "
            f"respond with EXACTLY 'NO_CLEAR_PHENOTYPE_IMPLIED'."
            f"\n\nProvide ONLY the name of the implied phenotype, without any explanation, "
            f"or 'NO_CLEAR_PHENOTYPE_IMPLIED' if none is clear."
        )
        
        # Query the LLM
        response = self.llm_client.query(prompt, self.extract_phenotype_system_message)
        
        # Clean the response
        implied_phenotype = response.strip()
        implied_phenotype = re.sub(r'[.,;:]$', '', implied_phenotype)
        
        # Check for the special "no clear phenotype" response
        if "NO_CLEAR_PHENOTYPE_IMPLIED" in implied_phenotype.upper():
            result = {
                'implied_phenotype': None,
                'confidence': 0.9,
                'method': 'llm_extraction_no_clear_phenotype'
            }
        else:
            result = {
                'implied_phenotype': implied_phenotype,
                'confidence': 0.7,  # Lower confidence compared to V1 to be more conservative
                'method': 'llm_extraction'
            }
        
        # Cache the result
        self.extracted_phenotype_cache[cache_key] = result
        
        if result['implied_phenotype'] is None:
            self._debug_print(f"LLM could not extract a clear implied phenotype from '{entity}'", level=2)
        else:
            self._debug_print(f"LLM extracted implied phenotype from '{entity}': '{implied_phenotype}'", level=2)
        
        return result

    def validate_phenotype_exists(self, phenotype: str) -> Dict:
        """
        Validate if a phenotype exists as a recognized medical concept with configurable retrieval.
        
        Args:
            phenotype: The phenotype to validate
            
        Returns:
            Dictionary with validation results
        """
        # Skip empty inputs
        if not phenotype:
            return {
                'is_valid': False,
                'confidence': 1.0,
                'method': 'empty_input'
            }
            
        # Create a cache key
        cache_key = f"validate_phenotype::{phenotype}"
        
        # Check cache
        if cache_key in self.phenotype_validation_cache:
            result = self.phenotype_validation_cache[cache_key]
            self._debug_print(f"Cache hit for phenotype validation '{phenotype}': {result['is_valid']}", level=1)
            return result
            
        self._debug_print(f"Validating if phenotype '{phenotype}' exists", level=1)
        
        # Retrieve similar phenotypes for context if configured
        similar_phenotypes = []
        if self.config.use_retrieval_for_validation:
            similar_phenotypes = self._retrieve_similar_phenotypes(phenotype)
        
        # Format context items if using retrieval
        context_items = []
        if self.config.use_retrieval_for_validation:
            for similar_pheno in similar_phenotypes[:10]:  # Use top 10 for context
                context_items.append(f"- {similar_pheno['term']} ({similar_pheno['hp_id']})")
        
        context_text = "\n".join(context_items)
        
        # Create prompt for the LLM
        retrieval_part = ""
        if self.config.use_retrieval_for_validation and context_items:
            retrieval_part = (
                f"Here are some similar phenotype terms for context:\n\n"
                f"{context_text}\n\n"
            )
        
        prompt = (
            f"I need to validate whether '{phenotype}' is a valid, specific, recognized phenotype in clinical medicine.\n\n"
            f"{retrieval_part}"
            f"A valid phenotype must be specific enough to be clinically meaningful. "
            f"General terms like 'blood abnormality' or 'kidney disease' are too vague."
            f"\n\nBased on {'this context and ' if retrieval_part else ''}your clinical knowledge, "
            f"is '{phenotype}' a valid, specific medical phenotype concept?\n\n"
            f"Respond with ONLY 'YES' if it's a valid, specific phenotype or 'NO' if it's not."
        )
        
        # Query the LLM
        response = self.llm_client.query(prompt, self.phenotype_validation_system_message)
        
        # Parse the response
        is_valid = "yes" in response.lower() and "no" not in response.lower()
        
        # Create result
        result = {
            'is_valid': is_valid,
            'confidence': 0.9 if is_valid else 0.8,
            'method': 'llm_validation'
        }
        
        # Cache the result
        self.phenotype_validation_cache[cache_key] = result
        
        self._debug_print(f"Phenotype '{phenotype}' is{'' if is_valid else ' not'} valid", level=2)
        return result

    def validate_implication(self, entity: str, implied_phenotype: str, context: Optional[str] = None) -> Dict:
        """
        Validate if the implication from entity to phenotype is reasonable with configurable context usage.
        
        Args:
            entity: Original entity text
            implied_phenotype: Extracted implied phenotype 
            context: Original sentence containing the entity
            
        Returns:
            Dictionary with validation results
        """
        # Skip empty inputs
        if not entity or not implied_phenotype:
            return {
                'is_valid': False,
                'confidence': 1.0,
                'method': 'empty_input'
            }
            
        # Create a cache key - only include context if configured to use it
        cache_key = f"validate_implication::{entity}::{implied_phenotype}::{context if self.config.use_context_for_implication else ''}"
        
        # Check cache
        if cache_key in self.implication_validation_cache:
            result = self.implication_validation_cache[cache_key]
            self._debug_print(f"Cache hit for implication validation '{entity}' → '{implied_phenotype}': {result['is_valid']}", level=1)
            return result
            
        self._debug_print(f"Validating implication from '{entity}' to '{implied_phenotype}'", level=1)
        
        # Create prompt for the LLM
        context_part = ""
        if context and self.config.use_context_for_implication:
            context_part = f"Original sentence context: '{context}'\n\n"
        
        prompt = (
            f"I need to validate whether the following implication is reasonable:\n\n"
            f"Original entity: '{entity}'\n"
            f"Implied phenotype: '{implied_phenotype}'\n\n"
            f"{context_part}"
            f"Be extremely critical and conservative. Say NO unless there is an unambiguous, "
            f"direct clinical connection between the entity and the proposed phenotype."
            f"\nThe connection must be evident from the entity itself, not inferred from general knowledge."
            f"\n\nIs this a valid and reasonable implication? "
            f"Respond with ONLY 'YES' if the implication is valid or 'NO' if it's not valid."
        )
        
        # Query the LLM
        response = self.llm_client.query(prompt, self.implication_validation_system_message)
        
        # Parse the response
        is_valid = "yes" in response.lower() and "no" not in response.lower()
        
        # Create result
        result = {
            'is_valid': is_valid,
            'confidence': 0.9 if is_valid else 0.9,  # High confidence either way
            'method': 'llm_validation'
        }
        
        # Cache the result
        self.implication_validation_cache[cache_key] = result
        
        self._debug_print(f"Implication from '{entity}' to '{implied_phenotype}' is{'' if is_valid else ' not'} valid", level=2)
        return result

    def process_entity(self, entity: str, context: Optional[str] = None) -> Dict:
        """
        Process an entity through the enhanced multi-stage pipeline with improved precision.
        
        Args:
            entity: Entity text to process
            context: Original sentence containing the entity
            
        Returns:
            Dictionary with processing results
        """
        # Handle empty entities
        if not entity:
            return {
                'status': 'not_phenotype',
                'phenotype': None,
                'original_entity': entity,
                'confidence': 1.0,
                'method': 'empty_entity'
            }
            
        self._debug_print(f"Processing entity: '{entity}'", level=0)
        
        # Clean and preprocess the entity
        cleaned_entity = self.preprocess_entity(entity)
        if not cleaned_entity:
            return {
                'status': 'not_phenotype',
                'phenotype': None,
                'original_entity': entity,
                'confidence': 1.0,
                'method': 'empty_after_preprocessing'
            }
        
        # NEW STAGE 0: Pre-filter common non-phenotype terms
        filter_result = self.pre_filter_term(cleaned_entity, context)
        if filter_result.get('should_filter', False):
            self._debug_print(f"'{entity}' filtered out as a common non-phenotype term", level=1)
            return {
                'status': 'not_phenotype',
                'phenotype': None,
                'original_entity': entity,
                'confidence': filter_result['confidence'],
                'method': 'pre_filtering'
            }
        
        # STAGE 1: Check if it's a direct phenotype
        direct_result = self.verify_direct_phenotype(cleaned_entity, context)
        
        # If it's a direct phenotype, return it with details
        if direct_result.get('is_phenotype', False):
            self._debug_print(f"'{entity}' is a direct phenotype", level=1)
            result = {
                'status': 'direct_phenotype',
                'phenotype': cleaned_entity,
                'original_entity': entity,
                'confidence': direct_result['confidence'],
                'method': direct_result['method']
            }
            
            if 'hp_id' in direct_result:
                result['hp_id'] = direct_result['hp_id']
                result['matched_term'] = direct_result['matched_term']
                
            return result
        
        # STAGE 2: Check if it implies a phenotype - more conservative than V1
        implies_result = self.check_implies_phenotype(cleaned_entity, context)
        
        if not implies_result.get('implies_phenotype', False):
            self._debug_print(f"'{entity}' is not a phenotype and doesn't imply one", level=1)
            return {
                'status': 'not_phenotype',
                'phenotype': None,
                'original_entity': entity,
                'confidence': implies_result['confidence'],
                'method': implies_result.get('method', 'llm_verification')
            }
            
        # STAGE 3: Extract the implied phenotype - allows for "no clear phenotype" result
        extract_result = self.extract_implied_phenotype(cleaned_entity, context)
        implied_phenotype = extract_result.get('implied_phenotype')
        
        # If we couldn't extract an implied phenotype, not a phenotype
        if not implied_phenotype:
            self._debug_print(f"No clear phenotype could be extracted from '{entity}'", level=1)
            return {
                'status': 'not_phenotype',
                'phenotype': None,
                'original_entity': entity,
                'confidence': extract_result.get('confidence', 0.7),
                'method': extract_result.get('method', 'no_implied_phenotype_found')
            }
        
        # STAGE 4: Validate if the phenotype exists as a recognized medical concept
        # Moving this step before implication validation to fail faster
        phenotype_validation_result = self.validate_phenotype_exists(implied_phenotype)
        
        if not phenotype_validation_result.get('is_valid', False):
            self._debug_print(f"Implied phenotype '{implied_phenotype}' from '{entity}' is not valid", level=1)
            return {
                'status': 'not_phenotype',
                'phenotype': None,
                'original_entity': entity,
                'confidence': phenotype_validation_result['confidence'],
                'method': 'invalid_phenotype'
            }
        
        # STAGE 5: Validate if the implication is reasonable
        implication_validation_result = self.validate_implication(cleaned_entity, implied_phenotype, context)
        
        if not implication_validation_result.get('is_valid', False):
            self._debug_print(f"Implication from '{entity}' to '{implied_phenotype}' is not valid", level=1)
            return {
                'status': 'not_phenotype',
                'phenotype': None,
                'original_entity': entity,
                'confidence': implication_validation_result['confidence'],
                'method': 'invalid_implication'
            }
        
        # If implication and phenotype are both valid, return the implied phenotype
        self._debug_print(f"'{entity}' implies valid phenotype '{implied_phenotype}'", level=1)
        result = {
            'status': 'implied_phenotype',
            'phenotype': implied_phenotype,
            'original_entity': entity,
            'confidence': min(extract_result['confidence'], phenotype_validation_result['confidence']),
            'method': 'multi_stage_pipeline'
        }
            
        return result
    
    def batch_process(self, entities_with_context: List[Dict]) -> List[Dict]:
        """
        Process a batch of entities with their contexts through the multi-stage pipeline.
        
        Args:
            entities_with_context: List of dicts with 'entity' and 'context' keys
            
        Returns:
            List of dicts with processing results (phenotypes only)
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
            
            # Only include entities that are phenotypes (direct or implied)
            if result['status'] in ['direct_phenotype', 'implied_phenotype']:
                # Add original context
                result['context'] = context
                results.append(result)
        
        self._debug_print(f"Identified {len(results)} phenotypes (direct or implied)")
        return results


# For compatibility with the original implementation
class HPOVerifierConfig:
    """Configuration for when to use retrieval and context in the HPO verification pipeline."""
    
    def __init__(self, 
                 use_retrieval_for_direct=True,
                 use_retrieval_for_implies=True,
                 use_retrieval_for_extract=True,
                 use_retrieval_for_validation=True,
                 use_retrieval_for_implication=True,
                 use_context_for_direct=True,
                 use_context_for_implies=True,
                 use_context_for_extract=True,
                 use_context_for_validation=False,
                 use_context_for_implication=True):
        # Retrieval settings
        self.use_retrieval_for_direct = use_retrieval_for_direct
        self.use_retrieval_for_implies = use_retrieval_for_implies
        self.use_retrieval_for_extract = use_retrieval_for_extract
        self.use_retrieval_for_validation = use_retrieval_for_validation
        self.use_retrieval_for_implication = use_retrieval_for_implication
        
        # Context usage settings
        self.use_context_for_direct = use_context_for_direct
        self.use_context_for_implies = use_context_for_implies
        self.use_context_for_extract = use_context_for_extract
        self.use_context_for_validation = use_context_for_validation
        self.use_context_for_implication = use_context_for_implication
    
    def to_dict(self):
        """Convert configuration to a dictionary."""
        return {
            "retrieval": {
                "direct": self.use_retrieval_for_direct,
                "implies": self.use_retrieval_for_implies,
                "extract": self.use_retrieval_for_extract,
                "validation": self.use_retrieval_for_validation,
                "implication": self.use_retrieval_for_implication
            },
            "context": {
                "direct": self.use_context_for_direct,
                "implies": self.use_context_for_implies,
                "extract": self.use_context_for_extract,
                "validation": self.use_context_for_validation,
                "implication": self.use_context_for_implication
            }
        }
    
    @classmethod
    def from_dict(cls, config_dict):
        """Create configuration from a dictionary."""
        return cls(
            use_retrieval_for_direct=config_dict["retrieval"]["direct"],
            use_retrieval_for_implies=config_dict["retrieval"]["implies"],
            use_retrieval_for_extract=config_dict["retrieval"]["extract"],
            use_retrieval_for_validation=config_dict["retrieval"]["validation"],
            use_retrieval_for_implication=config_dict["retrieval"]["implication"],
            use_context_for_direct=config_dict["context"]["direct"],
            use_context_for_implies=config_dict["context"]["implies"],
            use_context_for_extract=config_dict["context"]["extract"],
            use_context_for_validation=config_dict["context"]["validation"],
            use_context_for_implication=config_dict["context"]["implication"]
        )
    
    def __str__(self):
        """String representation of the configuration."""
        return str(self.to_dict())






from typing import List, Dict, Any, Optional, Tuple, Union, Set
import json
import re
import numpy as np
from datetime import datetime
from fuzzywuzzy import fuzz


class MultiStageHPOVerifierV3:
    """
    An enhanced multi-stage HPO verifier that incorporates direct matching with HPO candidates
    at key decision points while maintaining the overall multi-stage pipeline structure.
    
    This version has:
    1. No pre-filtering of entities
    2. Binary YES/NO responses for direct matching and implication validation
    3. Simplified decision criteria at each stage
    """
    
    def __init__(self, embedding_manager, llm_client, config=None, debug=False):
        """Initialize with a specific configuration."""
        self.embedding_manager = embedding_manager
        self.llm_client = llm_client
        self.debug = debug
        self.index = None
        self.embedded_documents = None
        self.config = config or HPOVerifierConfig()
        self.candidate_count = 20  # Retrieve more candidates than V1/V2
        
        # Binary system messages for direct YES/NO responses
        
        # Direct verification with binary matching
        self.direct_verification_system_message = (
            "You are a clinical expert specializing in phenotype identification for Human Phenotype Ontology (HPO) mapping. "
            "Your task is to determine if the given entity represents a valid human phenotype based on the provided HPO candidates."
            "\n\nA valid phenotype must describe an abnormal characteristic or trait, not just a normal "
            "anatomical structure, physiological process, laboratory test, or medication."
            "\n\nAFTER REVIEWING THE CANDIDATES, respond with ONLY 'YES' if:"
            "\n- The entity EXACTLY or CLOSELY matches any HPO candidate, OR"
            "\n- The entity clearly describes a phenotype even if not in the candidates"
            "\n\nRespond with ONLY 'NO' if:"
            "\n- The entity is NOT a phenotype (e.g., normal anatomy, medication, lab test without value)"
            "\n- The entity does not represent any abnormal human trait or characteristic"
            "\n\nYOUR RESPONSE MUST BE EXACTLY 'YES' OR 'NO' WITH NO ADDITIONAL TEXT."
        )
        
        # Implied phenotype check - binary YES/NO
        self.implied_phenotype_system_message = (
            "You are a clinical expert specializing in phenotype identification for Human Phenotype Ontology (HPO) mapping. "
            "Determine if the given term DIRECTLY AND UNAMBIGUOUSLY implies a specific phenotype, even though it's not a direct phenotype itself. "
            "Be extremely conservative - only say YES if the implication is clear and specific."
            "\nLaboratory values, medications, or procedures DO NOT imply phenotypes unless there is explicit abnormality mentioned."
            "\nIf you're uncertain or the implication requires multiple assumptions, say NO."
            "\n\nEXAMPLES OF VALID IMPLICATIONS:"
            "\n- 'Elevated white blood cell count of 15,000' implies 'leukocytosis' (YES - explicit abnormality)"
            "\n- 'E. coli growing in urine culture' implies 'bacteriuria' (YES - specific finding)"
            "\n- 'Hemoglobin of 6.5 g/dL' implies 'anemia' (YES - clearly below normal range)"
            "\n\nEXAMPLES OF INVALID IMPLICATIONS:"
            "\n- 'White blood cell count' does NOT imply 'leukocytosis' (NO - no value specified)"
            "\n- 'Taking insulin' does NOT imply 'diabetes mellitus' (NO - medications alone don't imply diagnoses)"
            "\n- 'Kidney biopsy' does NOT imply 'nephropathy' (NO - diagnostic procedure without findings)"
            "\n- 'Heart murmur' does NOT imply 'congenital heart defect' (NO - too specific without evidence)"
            "\n\nRespond with ONLY 'YES' if the term directly implies a phenotype, or 'NO' if it doesn't."
            "\nYOUR RESPONSE MUST BE EXACTLY 'YES' OR 'NO' WITH NO ADDITIONAL TEXT."
        )
        
        # Extract phenotype with option for no clear phenotype
        self.extract_phenotype_system_message = (
            "You are a clinical expert specializing in phenotype identification for Human Phenotype Ontology (HPO) mapping. "
            "A previous analysis determined that a given term might imply a phenotype but is not a direct phenotype itself. "
            "Your task is to precisely identify what specific phenotype is implied by this term. "
            "\nProvide ONLY the name of the implied phenotype as it would appear in medical terminology. "
            "Be specific and concise. Do not include explanations or multiple options."
            "\n\nEXAMPLES:"
            "\n- 'Elevated white blood cell count of 15,000' implies 'leukocytosis'"
            "\n- 'E. coli growing in urine culture' implies 'bacteriuria'"
            "\n- 'Hemoglobin of 6.5 g/dL' implies 'anemia'"
            "\n\nIf you cannot identify a specific phenotype that is DIRECTLY implied with high confidence, "
            "respond with EXACTLY 'NO_CLEAR_PHENOTYPE_IMPLIED'."
        )
        
        # Phenotype validation - binary YES/NO
        self.phenotype_validation_system_message = (
            "You are a clinical expert specializing in phenotype identification for Human Phenotype Ontology (HPO) mapping. "
            "Your task is to determine if the proposed phenotype is a valid medical concept based on the provided HPO phenotype candidates."
            "\n\nAFTER REVIEWING THE CANDIDATES, respond with ONLY 'YES' if:"
            "\n- The phenotype EXACTLY or CLOSELY matches any HPO candidate, OR"
            "\n- The phenotype is a valid, recognized abnormal characteristic or condition in clinical medicine even if not in the candidates"
            "\n\nRespond with ONLY 'NO' if:"
            "\n- The proposed term is NOT a valid phenotype in clinical medicine"
            "\n- The term is too vague, general, or not recognized as a specific phenotype"
            "\n\nYOUR RESPONSE MUST BE EXACTLY 'YES' OR 'NO' WITH NO ADDITIONAL TEXT."
        )
        
        # Implication validation - strictly binary YES/NO
        self.implication_validation_system_message = (
            "You are a clinical expert specializing in phenotype identification for Human Phenotype Ontology (HPO) mapping. "
            "Your task is to validate whether an implied phenotype is reasonable given the original entity and its context. "
            "Be extremely critical and conservative in your assessment. Say YES only if there is an unambiguous, "
            "direct clinical connection between the entity and the proposed phenotype."
            "\nThe connection must be evident from the entity itself, not inferred from general knowledge."
            "\n\nEXAMPLES of VALID implications:"
            "\n- Entity: 'E. coli in urine culture' → Implied phenotype: 'bacteriuria' (VALID: specific finding)"
            "\n- Entity: 'Hemoglobin of 6.5 g/dL' → Implied phenotype: 'anemia' (VALID: specific abnormal value)"
            "\n\nEXAMPLES of INVALID implications:"
            "\n- Entity: 'white blood cell count' → Implied phenotype: 'leukocytosis' (INVALID: no value specified)"
            "\n- Entity: 'taking insulin daily' → Implied phenotype: 'diabetes mellitus' (INVALID: medication alone)"
            "\n- Entity: 'retina' → Implied phenotype: 'retinopathy' (INVALID: normal anatomy without abnormality)"
            "\n- Entity: 'renal tissue' → Implied phenotype: 'glomerulonephritis' (INVALID: too specific without evidence)"
            "\n\nRespond with ONLY 'YES' if the implication is valid, or 'NO' if it's not valid."
            "\nYOUR RESPONSE MUST BE EXACTLY 'YES' OR 'NO' WITH NO ADDITIONAL TEXT."
        )
        
        # Caches to avoid redundant API calls
        self.verification_cache = {}
        self.implied_phenotype_cache = {}
        self.extracted_phenotype_cache = {}
        self.implication_validation_cache = {}
        self.phenotype_validation_cache = {}
    
    def _debug_print(self, message: str, level: int = 0):
        """Print debug messages if debug mode is enabled."""
        if self.debug:
            indent = "  " * level
            print(f"{datetime.now().strftime('%H:%M:%S')} | {indent}{message}")
    
    def set_config(self, config):
        """Update the configuration."""
        self.config = config
        self.clear_caches()  # Clear caches when configuration changes
        
    def prepare_index(self, metadata: List[Dict]):
        """Prepare FAISS index from metadata for similarity search."""
        if self.index is None:
            self._debug_print("Preparing FAISS index for phenotype verification...")
            embeddings_array = self.embedding_manager.prepare_embeddings(metadata)
            self.index = self.embedding_manager.create_index(embeddings_array)
            self.embedded_documents = metadata
            self._debug_print(f"Index prepared with {len(metadata)} embedded documents")

    def clear_caches(self):
        """Clear all caches to prepare for a fresh evaluation run."""
        self.verification_cache = {}
        self.implied_phenotype_cache = {}
        self.extracted_phenotype_cache = {}
        self.implication_validation_cache = {}
        self.phenotype_validation_cache = {}
        self._debug_print("All caches cleared")

    def _retrieve_similar_phenotypes(self, entity: str, k: int = 20) -> List[Dict]:
        """Retrieve similar phenotypes from the HPO ontology for context, with increased count."""
        if self.index is None:
            raise ValueError("Index not prepared. Call prepare_index() first.")
            
        # Embed the query
        query_vector = self.embedding_manager.query_text(entity).reshape(1, -1)
        
        # Search for similar items
        distances, indices = self.embedding_manager.search(query_vector, self.index, k=min(800, len(self.embedded_documents)))
        
        # Extract unique metadata
        similar_phenotypes = []
        seen_metadata = set()
        
        for idx, distance in zip(indices[0], distances[0]):
            metadata = self.embedded_documents[idx]['unique_metadata']
            metadata_str = json.dumps(metadata)
            
            if metadata_str not in seen_metadata:
                seen_metadata.add(metadata_str)
                similar_phenotypes.append({
                    'term': metadata.get('info', ''),
                    'hp_id': metadata.get('hp_id', ''),
                    'similarity_score': 1.0 / (1.0 + distance)  # Convert distance to similarity
                })
                
                if len(similar_phenotypes) >= k:
                    break
                    
        return similar_phenotypes
    
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

    def verify_direct_phenotype(self, entity: str, context: Optional[str] = None) -> Dict:
        """
        Verify if an entity is a direct phenotype through binary YES/NO matching against HPO candidates.
        
        Args:
            entity: Entity text to verify
            context: Original sentence containing the entity
            
        Returns:
            Dictionary with verification results
        """
        # Handle empty entities
        if not entity:
            return {
                'is_phenotype': False,
                'confidence': 1.0,
                'method': 'empty_entity'
            }
            
        # Create a cache key - only include context if configured to use it
        cache_key = f"direct::{entity}::{context if self.config.use_context_for_direct else ''}"
        
        # Check cache first
        if cache_key in self.verification_cache:
            result = self.verification_cache[cache_key]
            self._debug_print(f"Cache hit for direct phenotype '{entity}': {result['is_phenotype']}", level=1)
            return result
            
        self._debug_print(f"Verifying if '{entity}' is a direct phenotype via binary matching", level=1)
        
        # Check for exact matches using fuzzy matching first (optimization)
        similar_phenotypes = self._retrieve_similar_phenotypes(entity, k=self.candidate_count)
        
        for phenotype in similar_phenotypes:
            normalized_term = self._normalize_text(phenotype['term'])
            normalized_entity = self._normalize_text(entity)
            
            # Check for exact match
            if normalized_term == normalized_entity:
                self._debug_print(f"Exact match found: '{entity}' matches '{phenotype['term']}' ({phenotype['hp_id']})", level=2)
                result = {
                    'is_phenotype': True,
                    'confidence': 1.0,
                    'method': 'exact_match',
                    'hp_id': phenotype['hp_id'],
                    'matched_term': phenotype['term']
                }
                self.verification_cache[cache_key] = result
                return result
                
            # Check for high similarity match (over 90%)
            similarity = fuzz.ratio(normalized_term, normalized_entity)
            if similarity > 93:
                self._debug_print(f"High similarity match ({similarity}%): '{entity}' matches '{phenotype['term']}' ({phenotype['hp_id']})", level=2)
                result = {
                    'is_phenotype': True,
                    'confidence': similarity / 100.0,
                    'method': 'high_similarity_match',
                    'hp_id': phenotype['hp_id'],
                    'matched_term': phenotype['term']
                }
                self.verification_cache[cache_key] = result
                return result
        
        # Format candidates for the LLM prompt
        candidate_items = []
        for i, phenotype in enumerate(similar_phenotypes, 1):
            candidate_items.append(f"{i}. '{phenotype['term']}' ({phenotype['hp_id']})")
        
        candidates_text = "\n".join(candidate_items)
        
        # Create context part if configured
        context_part = ""
        if context and self.config.use_context_for_direct:
            context_part = f"\nOriginal sentence context: '{context}'"
        
        # Create the binary YES/NO matching prompt
        prompt = (
            f"I need to determine if the entity '{entity}' is a valid human phenotype."
            f"\n\nHere are some HPO phenotype candidates for reference:"
            f"\n\n{candidates_text}\n"
            f"{context_part}\n\n"
            f"A valid phenotype must describe an abnormal characteristic or trait, not just a normal "
            f"anatomical structure, physiological process, laboratory test, or medication."
            f"\n\nBased on these candidates and criteria, is '{entity}' a valid human phenotype?"
            f"\nRespond with ONLY 'YES' or 'NO'."
        )
        
        # Query the LLM
        response = self.llm_client.query(prompt, self.direct_verification_system_message)
        
        # Parse the response - strictly look for "YES" or "NO"
        response_text = response.strip().upper()
        is_phenotype = "YES" in response_text and "NO" not in response_text
        
        # Create result based on binary response
        if is_phenotype:
            result = {
                'is_phenotype': True,
                'confidence': 0.9,
                'method': 'llm_binary_verification'
            }
        else:
            result = {
                'is_phenotype': False,
                'confidence': 0.9,
                'method': 'llm_binary_verification'
            }
        
        # Cache the result
        self.verification_cache[cache_key] = result
        
        self._debug_print(f"LLM binary verification: '{entity}' is{'' if is_phenotype else ' not'} a phenotype", level=2)
        return result

    def check_implies_phenotype(self, entity: str, context: Optional[str] = None) -> Dict:
        """
        Check if an entity implies a phenotype with configurable retrieval and context usage.
        
        Args:
            entity: Entity text to check
            context: Original sentence containing the entity
            
        Returns:
            Dictionary with results
        """
        # Handle empty entities
        if not entity:
            return {
                'implies_phenotype': False,
                'confidence': 1.0,
                'method': 'empty_entity'
            }
            
        # Create a cache key - only include context if configured to use it
        cache_key = f"implies::{entity}::{context if self.config.use_context_for_implies else ''}"
        
        # Check cache
        if cache_key in self.implied_phenotype_cache:
            result = self.implied_phenotype_cache[cache_key]
            self._debug_print(f"Cache hit for implied phenotype check '{entity}': {result['implies_phenotype']}", level=1)
            return result
            
        self._debug_print(f"Checking if '{entity}' implies a phenotype", level=1)
        
        # Retrieve similar phenotypes for context if configured
        similar_phenotypes = []
        if self.config.use_retrieval_for_implies:
            similar_phenotypes = self._retrieve_similar_phenotypes(entity)
        
        # Format context items if using retrieval
        context_items = []
        if self.config.use_retrieval_for_implies:
            for phenotype in similar_phenotypes[:10]:  # Use top 10 for context
                context_items.append(f"- {phenotype['term']} ({phenotype['hp_id']})")
        
        context_text = "\n".join(context_items)
        
        # Create prompt for the LLM
        context_part = ""
        if context and self.config.use_context_for_implies:
            context_part = f"Original sentence context: '{context}'\n\n"
        
        retrieval_part = ""
        if self.config.use_retrieval_for_implies and context_items:
            retrieval_part = (
                f"Here are some phenotype terms from the Human Phenotype Ontology for context:\n\n"
                f"{context_text}\n\n"
            )
        
        prompt = (
            f"I need to determine if '{entity}' DIRECTLY AND UNAMBIGUOUSLY implies a specific phenotype. "
            f"Be extremely conservative - only say YES if the implication is clear and specific."
            f"\n{context_part}"
            f"{retrieval_part}"
            f"Laboratory values, medications, or procedures DO NOT imply phenotypes unless there is explicit abnormality mentioned."
            f"\nIf you're uncertain or the implication requires multiple assumptions, say NO."
            f"\n\nDoes '{entity}' directly imply a specific phenotype? "
            f"Respond with ONLY 'YES' if it directly implies a phenotype or 'NO' if it doesn't."
        )
        
        # Query the LLM
        response = self.llm_client.query(prompt, self.implied_phenotype_system_message)
        
        # Parse the response - strictly look for "YES" or "NO"
        response_text = response.strip().upper()
        implies_phenotype = "YES" in response_text and "NO" not in response_text
        
        # Create result
        result = {
            'implies_phenotype': implies_phenotype,
            'confidence': 0.8 if implies_phenotype else 0.9,  # Higher confidence for "no" to be conservative
            'method': 'llm_binary_verification'
        }
        
        # Cache the result
        self.implied_phenotype_cache[cache_key] = result
        
        self._debug_print(f"LLM binary verification: '{entity}' does{'' if implies_phenotype else ' not'} imply a phenotype", level=2)
        return result

    def extract_implied_phenotype(self, entity: str, context: Optional[str] = None) -> Dict:
        """
        Extract the specific phenotype implied by an entity with configurable retrieval and context usage.
        
        Args:
            entity: Entity text that implies a phenotype
            context: Original sentence containing the entity
            
        Returns:
            Dictionary with results
        """
        # Handle empty entities
        if not entity:
            return {
                'implied_phenotype': None,
                'confidence': 0.0,
                'method': 'empty_entity'
            }
            
        # Create a cache key - only include context if configured to use it
        cache_key = f"extract::{entity}::{context if self.config.use_context_for_extract else ''}"
        
        # Check cache
        if cache_key in self.extracted_phenotype_cache:
            result = self.extracted_phenotype_cache[cache_key]
            self._debug_print(f"Cache hit for extracting implied phenotype from '{entity}': {result.get('implied_phenotype')}", level=1)
            return result
            
        self._debug_print(f"Extracting implied phenotype from '{entity}'", level=1)
        
        # Retrieve similar phenotypes for context if configured
        similar_phenotypes = []
        if self.config.use_retrieval_for_extract:
            similar_phenotypes = self._retrieve_similar_phenotypes(entity)
        
        # Format context items if using retrieval
        context_items = []
        if self.config.use_retrieval_for_extract:
            for phenotype in similar_phenotypes[:10]:  # Use top 10 for context
                context_items.append(f"- {phenotype['term']} ({phenotype['hp_id']})")
        
        context_text = "\n".join(context_items)
        
        # Create prompt for the LLM
        context_part = ""
        if context and self.config.use_context_for_extract:
            context_part = f"Original sentence context: '{context}'\n\n"
        
        retrieval_part = ""
        if self.config.use_retrieval_for_extract and context_items:
            retrieval_part = (
                f"Here are some phenotype terms for context:\n\n"
                f"{context_text}\n\n"
            )
        
        prompt = (
            f"The term '{entity}' might imply a phenotype. "
            f"{context_part}"
            f"{retrieval_part}"
            f"What specific phenotype is directly implied by '{entity}'? "
            f"For example, 'hemoglobin of 8 g/dL' implies 'anemia'."
            f"\n\nIf you cannot identify a specific phenotype that is DIRECTLY implied with high confidence, "
            f"respond with EXACTLY 'NO_CLEAR_PHENOTYPE_IMPLIED'."
            f"\n\nProvide ONLY the name of the implied phenotype, without any explanation, "
            f"or 'NO_CLEAR_PHENOTYPE_IMPLIED' if none is clear."
        )
        
        # Query the LLM
        response = self.llm_client.query(prompt, self.extract_phenotype_system_message)
        
        # Clean the response
        implied_phenotype = response.strip()
        implied_phenotype = re.sub(r'[.,;:]$', '', implied_phenotype)
        
        # Check for the special "no clear phenotype" response
        if "NO_CLEAR_PHENOTYPE_IMPLIED" in implied_phenotype.upper():
            result = {
                'implied_phenotype': None,
                'confidence': 0.9,
                'method': 'llm_extraction_no_clear_phenotype'
            }
        else:
            result = {
                'implied_phenotype': implied_phenotype,
                'confidence': 0.7,  # Lower confidence compared to V1 to be more conservative
                'method': 'llm_extraction'
            }
        
        # Cache the result
        self.extracted_phenotype_cache[cache_key] = result
        
        if result['implied_phenotype'] is None:
            self._debug_print(f"LLM could not extract a clear implied phenotype from '{entity}'", level=2)
        else:
            self._debug_print(f"LLM extracted implied phenotype from '{entity}': '{implied_phenotype}'", level=2)
        
        return result

    def validate_phenotype_exists(self, phenotype: str) -> Dict:
        """
        Validate if a phenotype exists by binary YES/NO matching against HPO candidates.
        
        Args:
            phenotype: The phenotype to validate
            
        Returns:
            Dictionary with validation results
        """
        # Skip empty inputs
        if not phenotype:
            return {
                'is_valid': False,
                'confidence': 1.0,
                'method': 'empty_input'
            }
            
        # Create a cache key
        cache_key = f"validate_phenotype::{phenotype}"
        
        # Check cache
        if cache_key in self.phenotype_validation_cache:
            result = self.phenotype_validation_cache[cache_key]
            self._debug_print(f"Cache hit for phenotype validation '{phenotype}': {result['is_valid']}", level=1)
            return result
            
        self._debug_print(f"Validating phenotype '{phenotype}' via binary matching", level=1)
        
        # Check for exact matches using fuzzy matching first (optimization)
        similar_phenotypes = self._retrieve_similar_phenotypes(phenotype, k=self.candidate_count)
        
        for pheno in similar_phenotypes:
            normalized_term = self._normalize_text(pheno['term'])
            normalized_phenotype = self._normalize_text(phenotype)
            
            # Check for exact match
            if normalized_term == normalized_phenotype:
                self._debug_print(f"Exact match found: '{phenotype}' matches '{pheno['term']}' ({pheno['hp_id']})", level=2)
                result = {
                    'is_valid': True,
                    'confidence': 1.0,
                    'method': 'exact_match',
                    'hp_id': pheno['hp_id'],
                    'matched_term': pheno['term']
                }
                self.phenotype_validation_cache[cache_key] = result
                return result
                
            # Check for high similarity match (over 90%)
            similarity = fuzz.ratio(normalized_term, normalized_phenotype)
            if similarity > 93:
                self._debug_print(f"High similarity match ({similarity}%): '{phenotype}' matches '{pheno['term']}' ({pheno['hp_id']})", level=2)
                result = {
                    'is_valid': True,
                    'confidence': similarity / 100.0,
                    'method': 'high_similarity_match',
                    'hp_id': pheno['hp_id'],
                    'matched_term': pheno['term']
                }
                self.phenotype_validation_cache[cache_key] = result
                return result
        
        # Format candidates for the LLM prompt
        candidate_items = []
        for i, pheno in enumerate(similar_phenotypes, 1):
            candidate_items.append(f"{i}. '{pheno['term']}' ({pheno['hp_id']})")
        
        candidates_text = "\n".join(candidate_items)
        
        # Create the binary YES/NO matching prompt
        prompt = (
            f"I need to determine if the phenotype '{phenotype}' is a valid medical concept."
            f"\n\nHere are some HPO phenotype candidates for reference:"
            f"\n\n{candidates_text}\n\n"
            f"Is '{phenotype}' a valid phenotype in clinical medicine? Consider both potential matches "
            f"in the candidates and your general knowledge of medical phenotypes."
            f"\nRespond with ONLY 'YES' or 'NO'."
        )
        
        # Query the LLM
        response = self.llm_client.query(prompt, self.phenotype_validation_system_message)
        
        # Parse the response - strictly look for "YES" or "NO"
        response_text = response.strip().upper()
        is_valid = "YES" in response_text and "NO" not in response_text
        
        # Create result based on binary response
        result = {
            'is_valid': is_valid,
            'confidence': 0.9,
            'method': 'llm_binary_validation'
        }
        
        # Cache the result
        self.phenotype_validation_cache[cache_key] = result
        
        self._debug_print(f"LLM binary validation: '{phenotype}' is{'' if is_valid else ' not'} a valid phenotype", level=2)
        return result

    def validate_implication(self, entity: str, implied_phenotype: str, context: Optional[str] = None) -> Dict:
        """
        Validate if the implication from entity to phenotype is reasonable with strictly binary YES/NO response.
        
        Args:
            entity: Original entity text
            implied_phenotype: Extracted implied phenotype 
            context: Original sentence containing the entity
            
        Returns:
            Dictionary with validation results
        """
        # Skip empty inputs
        if not entity or not implied_phenotype:
            return {
                'is_valid': False,
                'confidence': 1.0,
                'method': 'empty_input'
            }
            
        # Create a cache key - only include context if configured to use it
        cache_key = f"validate_implication::{entity}::{implied_phenotype}::{context if self.config.use_context_for_implication else ''}"
        
        # Check cache
        if cache_key in self.implication_validation_cache:
            result = self.implication_validation_cache[cache_key]
            self._debug_print(f"Cache hit for implication validation '{entity}' → '{implied_phenotype}': {result['is_valid']}", level=1)
            return result
            
        self._debug_print(f"Validating implication from '{entity}' to '{implied_phenotype}' with binary response", level=1)
        
        # Create prompt for the LLM
        context_part = ""
        if context and self.config.use_context_for_implication:
            context_part = f"Original sentence context: '{context}'\n\n"
        
        prompt = (
            f"I need to validate whether the following implication is reasonable:\n\n"
            f"Original entity: '{entity}'\n"
            f"Implied phenotype: '{implied_phenotype}'\n\n"
            f"{context_part}"
            f"Be extremely critical and conservative. Say YES only if there is an unambiguous, "
            f"direct clinical connection between the entity and the proposed phenotype."
            f"\nThe connection must be evident from the entity itself, not inferred from general knowledge."
            f"\n\nIs this a valid and reasonable implication? "
            f"Respond with ONLY 'YES' or 'NO'."
        )
        
        # Query the LLM
        response = self.llm_client.query(prompt, self.implication_validation_system_message)
        
        # Parse the response - strictly look for "YES" or "NO"
        response_text = response.strip().upper()
        is_valid = "YES" in response_text and "NO" not in response_text
        
        # Create result
        result = {
            'is_valid': is_valid,
            'confidence': 0.9,
            'method': 'llm_binary_validation'
        }
        
        # Cache the result
        self.implication_validation_cache[cache_key] = result
        
        self._debug_print(f"LLM binary validation: Implication from '{entity}' to '{implied_phenotype}' is{'' if is_valid else ' not'} valid", level=2)
        return result

    def process_entity(self, entity: str, context: Optional[str] = None) -> Dict:
        """
        Process an entity through the enhanced multi-stage pipeline with binary matching.
        
        Args:
            entity: Entity text to process
            context: Original sentence containing the entity
            
        Returns:
            Dictionary with processing results
        """
        # Handle empty entities
        if not entity:
            return {
                'status': 'not_phenotype',
                'phenotype': None,
                'original_entity': entity,
                'confidence': 1.0,
                'method': 'empty_entity'
            }
            
        self._debug_print(f"Processing entity: '{entity}'", level=0)
        
        # Clean and preprocess the entity
        cleaned_entity = self.preprocess_entity(entity)
        if not cleaned_entity:
            return {
                'status': 'not_phenotype',
                'phenotype': None,
                'original_entity': entity,
                'confidence': 1.0,
                'method': 'empty_after_preprocessing'
            }
        
        # STAGE 1: Check if it's a direct phenotype using binary matching
        direct_result = self.verify_direct_phenotype(cleaned_entity, context)
        
        # If it's a direct phenotype, return it with details
        if direct_result.get('is_phenotype', False):
            self._debug_print(f"'{entity}' is a direct phenotype", level=1)
            result = {
                'status': 'direct_phenotype',
                'phenotype': direct_result.get('matched_term', cleaned_entity),
                'original_entity': entity,
                'confidence': direct_result['confidence'],
                'method': direct_result['method']
            }
            
            if 'hp_id' in direct_result:
                result['hp_id'] = direct_result['hp_id']
                
            return result
        
        # STAGE 2: Check if it implies a phenotype with binary response
        implies_result = self.check_implies_phenotype(cleaned_entity, context)
        
        if not implies_result.get('implies_phenotype', False):
            self._debug_print(f"'{entity}' is not a phenotype and doesn't imply one", level=1)
            return {
                'status': 'not_phenotype',
                'phenotype': None,
                'original_entity': entity,
                'confidence': implies_result['confidence'],
                'method': implies_result.get('method', 'llm_verification')
            }
            
        # STAGE 3: Extract the implied phenotype
        extract_result = self.extract_implied_phenotype(cleaned_entity, context)
        implied_phenotype = extract_result.get('implied_phenotype')
        
        # If we couldn't extract an implied phenotype, not a phenotype
        if not implied_phenotype:
            self._debug_print(f"No clear phenotype could be extracted from '{entity}'", level=1)
            return {
                'status': 'not_phenotype',
                'phenotype': None,
                'original_entity': entity,
                'confidence': extract_result.get('confidence', 0.7),
                'method': extract_result.get('method', 'no_implied_phenotype_found')
            }
        
        # STAGE 4: Validate if the phenotype exists via binary matching
        phenotype_validation_result = self.validate_phenotype_exists(implied_phenotype)
        
        if not phenotype_validation_result.get('is_valid', False):
            self._debug_print(f"Implied phenotype '{implied_phenotype}' from '{entity}' is not valid", level=1)
            return {
                'status': 'not_phenotype',
                'phenotype': None,
                'original_entity': entity,
                'confidence': phenotype_validation_result['confidence'],
                'method': 'invalid_phenotype'
            }
        
        # If there's a matching HPO term, use it
        if 'hp_id' in phenotype_validation_result:
            implied_phenotype = phenotype_validation_result.get('matched_term', implied_phenotype)
            hp_id = phenotype_validation_result['hp_id']
        else:
            hp_id = None
        
        # STAGE 5: Validate if the implication is reasonable with binary response
        implication_validation_result = self.validate_implication(cleaned_entity, implied_phenotype, context)
        
        if not implication_validation_result.get('is_valid', False):
            self._debug_print(f"Implication from '{entity}' to '{implied_phenotype}' is not valid", level=1)
            return {
                'status': 'not_phenotype',
                'phenotype': None,
                'original_entity': entity,
                'confidence': implication_validation_result['confidence'],
                'method': 'invalid_implication'
            }
        
        # If implication and phenotype are both valid, return the implied phenotype
        self._debug_print(f"'{entity}' implies valid phenotype '{implied_phenotype}'", level=1)
        result = {
            'status': 'implied_phenotype',
            'phenotype': implied_phenotype,
            'original_entity': entity,
            'confidence': min(extract_result['confidence'], phenotype_validation_result['confidence']),
            'method': 'multi_stage_pipeline'
        }
        
        # Include HP ID if available
        if hp_id:
            result['hp_id'] = hp_id
            
        return result
    
    def batch_process(self, entities_with_context: List[Dict]) -> List[Dict]:
        """
        Process a batch of entities with their contexts through the enhanced multi-stage pipeline.
        
        Args:
            entities_with_context: List of dicts with 'entity' and 'context' keys
            
        Returns:
            List of dicts with processing results (phenotypes only)
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
            
            # Only include entities that are phenotypes (direct or implied)
            if result['status'] in ['direct_phenotype', 'implied_phenotype']:
                # Add original context
                result['context'] = context
                results.append(result)
        
        self._debug_print(f"Identified {len(results)} phenotypes (direct or implied)")
        return results

# For compatibility with the original implementation
class HPOVerifierConfig:
    """Configuration for when to use retrieval and context in the HPO verification pipeline."""
    
    def __init__(self, 
                 use_retrieval_for_direct=True,
                 use_retrieval_for_implies=True,
                 use_retrieval_for_extract=True,
                 use_retrieval_for_validation=True,
                 use_retrieval_for_implication=True,
                 use_context_for_direct=True,
                 use_context_for_implies=True,
                 use_context_for_extract=True,
                 use_context_for_validation=False,
                 use_context_for_implication=True):
        # Retrieval settings
        self.use_retrieval_for_direct = use_retrieval_for_direct
        self.use_retrieval_for_implies = use_retrieval_for_implies
        self.use_retrieval_for_extract = use_retrieval_for_extract
        self.use_retrieval_for_validation = use_retrieval_for_validation
        self.use_retrieval_for_implication = use_retrieval_for_implication
        
        # Context usage settings
        self.use_context_for_direct = use_context_for_direct
        self.use_context_for_implies = use_context_for_implies
        self.use_context_for_extract = use_context_for_extract
        self.use_context_for_validation = use_context_for_validation
        self.use_context_for_implication = use_context_for_implication
    
    def to_dict(self):
        """Convert configuration to a dictionary."""
        return {
            "retrieval": {
                "direct": self.use_retrieval_for_direct,
                "implies": self.use_retrieval_for_implies,
                "extract": self.use_retrieval_for_extract,
                "validation": self.use_retrieval_for_validation,
                "implication": self.use_retrieval_for_implication
            },
            "context": {
                "direct": self.use_context_for_direct,
                "implies": self.use_context_for_implies,
                "extract": self.use_context_for_extract,
                "validation": self.use_context_for_validation,
                "implication": self.use_context_for_implication
            }
        }
    
    @classmethod
    def from_dict(cls, config_dict):
        """Create configuration from a dictionary."""
        return cls(
            use_retrieval_for_direct=config_dict["retrieval"]["direct"],
            use_retrieval_for_implies=config_dict["retrieval"]["implies"],
            use_retrieval_for_extract=config_dict["retrieval"]["extract"],
            use_retrieval_for_validation=config_dict["retrieval"]["validation"],
            use_retrieval_for_implication=config_dict["retrieval"]["implication"],
            use_context_for_direct=config_dict["context"]["direct"],
            use_context_for_implies=config_dict["context"]["implies"],
            use_context_for_extract=config_dict["context"]["extract"],
            use_context_for_validation=config_dict["context"]["validation"],
            use_context_for_implication=config_dict["context"]["implication"]
        )
    
    def __str__(self):
        """String representation of the configuration."""
        return str(self.to_dict())
    import re
import json
import numpy as np
import torch
from datetime import datetime
from fuzzywuzzy import fuzz
import itertools
import time
from typing import List, Dict, Any, Optional, Tuple, Union, Set

from hporag.verify import HPOVerifierConfig
from utils.search_tools import ToolSearcher


class MultiStageHPOVerifierV4:
    """
    Enhanced multi-stage HPO verifier with simplified lab test analysis pipeline.
    
    This version implements a streamlined workflow:
    1. Direct phenotype verification with retrieved candidates
    2. Lab test detection (only if numbers are present in entity/context)
    3. Lab test analysis with reference ranges
    4. Implied phenotype determination
    
    Lab test abnormalities are directly translated to phenotypes.
    """
    
    def __init__(self, embedding_manager, llm_client, config=None, debug=False, 
                 lab_embeddings_file=None):
        """Initialize with specific configuration and optional lab embeddings file."""
        self.embedding_manager = embedding_manager
        self.llm_client = llm_client
        self.debug = debug
        self.index = None
        self.embedded_documents = None
        self.config = config or HPOVerifierConfig()
        self.candidate_count = 20
        
        # Lab test tools
        self.lab_embeddings_file = lab_embeddings_file
        self.lab_searcher = None
        if lab_embeddings_file:
            self.initialize_lab_searcher()
        
        # Direct verification with binary matching
        self.direct_verification_system_message = (
            "You are a clinical expert specializing in phenotype identification for Human Phenotype Ontology (HPO) mapping. "
            "Your task is to determine if the given entity represents a valid human phenotype based on the provided HPO candidates."
            "\n\nA valid phenotype must describe an abnormal characteristic or trait, not just a normal "
            "anatomical structure, physiological process, laboratory test, or medication."
            "\n\nAFTER REVIEWING THE CANDIDATES, respond with ONLY 'YES' if:"
            "\n- The entity EXACTLY or CLOSELY matches any HPO candidate, OR"
            "\n- The entity clearly describes a phenotype even if not in the candidates"
            "\n\nRespond with ONLY 'NO' if:"
            "\n- The entity is NOT a phenotype (e.g., normal anatomy, medication, lab test without value)"
            "\n- The entity does not represent any abnormal human trait or characteristic"
            "\n\nYOUR RESPONSE MUST BE EXACTLY 'YES' OR 'NO' WITH NO ADDITIONAL TEXT."
        )
        
        # Lab test identification (simplified)
        self.lab_identification_system_message = (
            "You are a clinical expert analyzing laboratory test information in clinical notes. "
            "Your task is to determine if the given entity contains information about a laboratory test with a measured value."
            "\n\nRespond with ONLY 'YES' if:"
            "\n- The entity contains a lab test name AND a numerical value/result"
            "\n- The entity clearly refers to a laboratory measurement with a value"
            "\n\nRespond with ONLY 'NO' if:"
            "\n- The entity only mentions a lab test without a specific value"
            "\n- The entity is not related to laboratory measurements"
            "\n\nYOUR RESPONSE MUST BE EXACTLY 'YES' OR 'NO' WITH NO ADDITIONAL TEXT."
        )
        
        # Lab test extraction and analysis (combined)
        self.lab_analysis_system_message = (
            "You are a clinical laboratory expert analyzing laboratory test results. "
            "Extract and analyze the lab test from the provided entity and determine if the value is abnormal."
            "\n\nFor ABNORMAL results, provide a medically precise description of the abnormality "
            "(e.g., 'elevated glucose', 'decreased hemoglobin', 'leukocytosis')."
            "\n\nFor NORMAL results, simply state 'normal'."
            "\n\nProvide your response in this EXACT JSON format:"
            "\n{"
            "\n  \"lab_name\": \"[extracted lab test name]\","
            "\n  \"value\": \"[extracted value with units if available]\","
            "\n  \"is_abnormal\": true/false,"
            "\n  \"abnormality\": \"[descriptive term for the abnormality, or 'normal' if not abnormal]\","
            "\n  \"direction\": \"[high/low/normal]\","
            "\n  \"confidence\": [0.0-1.0 value indicating your confidence]"
            "\n}"
            "\n\nReturn ONLY the JSON with no additional text."
        )
        
        # Implied phenotype check - binary YES/NO
        self.implied_phenotype_system_message = (
            "You are a clinical expert specializing in phenotype identification for Human Phenotype Ontology (HPO) mapping. "
            "Determine if the given term DIRECTLY AND UNAMBIGUOUSLY implies a specific phenotype, even though it's not a direct phenotype itself. "
            "Be extremely conservative - only say YES if the implication is clear and specific."
            "\nLaboratory values, medications, or procedures DO NOT imply phenotypes unless there is explicit abnormality mentioned."
            "\nIf you're uncertain or the implication requires multiple assumptions, say NO."
            "\n\nEXAMPLES OF VALID IMPLICATIONS:"
            "\n- 'Elevated white blood cell count of 15,000' implies 'leukocytosis' (YES - explicit abnormality)"
            "\n- 'E. coli growing in urine culture' implies 'bacteriuria' (YES - specific finding)"
            "\n- 'Hemoglobin of 6.5 g/dL' implies 'anemia' (YES - clearly below normal range)"
            "\n\nEXAMPLES OF INVALID IMPLICATIONS:"
            "\n- 'White blood cell count' does NOT imply 'leukocytosis' (NO - no value specified)"
            "\n- 'Taking insulin' does NOT imply 'diabetes mellitus' (NO - medications alone don't imply diagnoses)"
            "\n- 'Kidney biopsy' does NOT imply 'nephropathy' (NO - diagnostic procedure without findings)"
            "\n- 'Heart murmur' does NOT imply 'congenital heart defect' (NO - too specific without evidence)"
            "\n\nRespond with ONLY 'YES' if the term directly implies a phenotype, or 'NO' if it doesn't."
            "\nYOUR RESPONSE MUST BE EXACTLY 'YES' OR 'NO' WITH NO ADDITIONAL TEXT."
        )
        
        # Extract phenotype with option for no clear phenotype
        self.extract_phenotype_system_message = (
            "You are a clinical expert specializing in phenotype identification for Human Phenotype Ontology (HPO) mapping. "
            "A previous analysis determined that a given term might imply a phenotype but is not a direct phenotype itself. "
            "Your task is to precisely identify what specific phenotype is implied by this term. "
            "\nProvide ONLY the name of the implied phenotype as it would appear in medical terminology. "
            "Be specific and concise. Do not include explanations or multiple options."
            "\n\nEXAMPLES:"
            "\n- 'Elevated white blood cell count of 15,000' implies 'leukocytosis'"
            "\n- 'E. coli growing in urine culture' implies 'bacteriuria'"
            "\n- 'Hemoglobin of 6.5 g/dL' implies 'anemia'"
            "\n\nIf you cannot identify a specific phenotype that is DIRECTLY implied with high confidence, "
            "respond with EXACTLY 'NO_CLEAR_PHENOTYPE_IMPLIED'."
        )
        
        # Phenotype validation - binary YES/NO
        self.phenotype_validation_system_message = (
            "You are a clinical expert specializing in phenotype identification for Human Phenotype Ontology (HPO) mapping. "
            "Your task is to determine if the proposed phenotype is a valid medical concept based on the provided HPO phenotype candidates."
            "\n\nAFTER REVIEWING THE CANDIDATES, respond with ONLY 'YES' if:"
            "\n- The phenotype EXACTLY or CLOSELY matches any HPO candidate, OR"
            "\n- The phenotype is a valid, recognized abnormal characteristic or condition in clinical medicine even if not in the candidates"
            "\n\nRespond with ONLY 'NO' if:"
            "\n- The proposed term is NOT a valid phenotype in clinical medicine"
            "\n- The term is too vague, general, or not recognized as a specific phenotype"
            "\n\nYOUR RESPONSE MUST BE EXACTLY 'YES' OR 'NO' WITH NO ADDITIONAL TEXT."
        )
        
        # Implication validation - strictly binary YES/NO
        self.implication_validation_system_message = (
            "You are a clinical expert specializing in phenotype identification for Human Phenotype Ontology (HPO) mapping. "
            "Your task is to validate whether an implied phenotype is reasonable given the original entity and its context. "
            "Be extremely critical and conservative in your assessment. Say YES only if there is an unambiguous, "
            "direct clinical connection between the entity and the proposed phenotype."
            "\nThe connection must be evident from the entity itself, not inferred from general knowledge."
            "\n\nEXAMPLES of VALID implications:"
            "\n- Entity: 'E. coli in urine culture' → Implied phenotype: 'bacteriuria' (VALID: specific finding)"
            "\n- Entity: 'Hemoglobin of 6.5 g/dL' → Implied phenotype: 'anemia' (VALID: specific abnormal value)"
            "\n\nEXAMPLES of INVALID implications:"
            "\n- Entity: 'white blood cell count' → Implied phenotype: 'leukocytosis' (INVALID: no value specified)"
            "\n- Entity: 'taking insulin daily' → Implied phenotype: 'diabetes mellitus' (INVALID: medication alone)"
            "\n- Entity: 'retina' → Implied phenotype: 'retinopathy' (INVALID: normal anatomy without abnormality)"
            "\n- Entity: 'renal tissue' → Implied phenotype: 'glomerulonephritis' (INVALID: too specific without evidence)"
            "\n\nRespond with ONLY 'YES' if the implication is valid, or 'NO' if it's not valid."
            "\nYOUR RESPONSE MUST BE EXACTLY 'YES' OR 'NO' WITH NO ADDITIONAL TEXT."
        )
        
        # Caches to avoid redundant API calls
        self.verification_cache = {}
        self.lab_test_detection_cache = {}
        self.lab_analysis_cache = {}
        self.implied_phenotype_cache = {}
        self.extracted_phenotype_cache = {}
        self.implication_validation_cache = {}
        self.phenotype_validation_cache = {}
    
    def initialize_lab_searcher(self):
        """Initialize the lab test searcher with embeddings file."""
        try:
            if not self.lab_embeddings_file:
                self._debug_print("No lab embeddings file provided, lab test tools disabled")
                return
                
            self._debug_print(f"Initializing lab test searcher with {self.lab_embeddings_file}")
            self.lab_searcher = ToolSearcher(
                model_type=self.embedding_manager.model_type,
                model_name=self.embedding_manager.model_name,
                device="cpu",  # Use CPU for tool searching to avoid GPU conflicts
                top_k=5  # Get top 5 matches for lab tests
            )
            self.lab_searcher.load_embeddings(self.lab_embeddings_file)
            self._debug_print("Lab test searcher initialized successfully")
        except Exception as e:
            self._debug_print(f"Error initializing lab test searcher: {e}")
            self.lab_searcher = None
    
    def _debug_print(self, message: str, level: int = 0):
        """Print debug messages if debug mode is enabled."""
        if self.debug:
            indent = "  " * level
            print(f"{datetime.now().strftime('%H:%M:%S')} | {indent}{message}")
    
    def set_config(self, config):
        """Update the configuration."""
        self.config = config
        self.clear_caches()  # Clear caches when configuration changes
        
    def prepare_index(self, metadata: List[Dict]):
        """Prepare FAISS index from metadata for similarity search."""
        if self.index is None:
            self._debug_print("Preparing FAISS index for phenotype verification...")
            embeddings_array = self.embedding_manager.prepare_embeddings(metadata)
            self.index = self.embedding_manager.create_index(embeddings_array)
            self.embedded_documents = metadata
            self._debug_print(f"Index prepared with {len(metadata)} embedded documents")

    def clear_caches(self):
        """Clear all caches to prepare for a fresh evaluation run."""
        self.verification_cache = {}
        self.lab_test_detection_cache = {}
        self.lab_analysis_cache = {}
        self.implied_phenotype_cache = {}
        self.extracted_phenotype_cache = {}
        self.implication_validation_cache = {}
        self.phenotype_validation_cache = {}
        self._debug_print("All caches cleared")

    def _retrieve_similar_phenotypes(self, entity: str, k: int = 20) -> List[Dict]:
        """Retrieve similar phenotypes from the HPO ontology for context."""
        if self.index is None:
            raise ValueError("Index not prepared. Call prepare_index() first.")
            
        # Embed the query
        query_vector = self.embedding_manager.query_text(entity).reshape(1, -1)
        
        # Search for similar items
        distances, indices = self.embedding_manager.search(query_vector, self.index, k=min(800, len(self.embedded_documents)))
        
        # Extract unique metadata
        similar_phenotypes = []
        seen_metadata = set()
        
        for idx, distance in zip(indices[0], distances[0]):
            metadata = self.embedded_documents[idx]['unique_metadata']
            metadata_str = json.dumps(metadata)
            
            if metadata_str not in seen_metadata:
                seen_metadata.add(metadata_str)
                similar_phenotypes.append({
                    'term': metadata.get('info', ''),
                    'hp_id': metadata.get('hp_id', ''),
                    'similarity_score': 1.0 / (1.0 + distance)  # Convert distance to similarity
                })
                
                if len(similar_phenotypes) >= k:
                    break
                    
        return similar_phenotypes
    
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

    def verify_direct_phenotype(self, entity: str, context: Optional[str] = None) -> Dict:
        """
        Verify if an entity is a direct phenotype through binary YES/NO matching against HPO candidates.
        
        Args:
            entity: Entity text to verify
            context: Original sentence containing the entity
            
        Returns:
            Dictionary with verification results
        """
        # Handle empty entities
        if not entity:
            return {
                'is_phenotype': False,
                'confidence': 1.0,
                'method': 'empty_entity'
            }
            
        # Create a cache key - only include context if configured to use it
        cache_key = f"direct::{entity}::{context if self.config.use_context_for_direct else ''}"
        
        # Check cache first
        if cache_key in self.verification_cache:
            result = self.verification_cache[cache_key]
            self._debug_print(f"Cache hit for direct phenotype '{entity}': {result['is_phenotype']}", level=1)
            return result
            
        self._debug_print(f"Verifying if '{entity}' is a direct phenotype via binary matching", level=1)
        
        # Check for exact matches using fuzzy matching first (optimization)
        similar_phenotypes = self._retrieve_similar_phenotypes(entity, k=self.candidate_count)
        
        for phenotype in similar_phenotypes:
            normalized_term = self._normalize_text(phenotype['term'])
            normalized_entity = self._normalize_text(entity)
            
            # Check for exact match
            if normalized_term == normalized_entity:
                self._debug_print(f"Exact match found: '{entity}' matches '{phenotype['term']}' ({phenotype['hp_id']})", level=2)
                result = {
                    'is_phenotype': True,
                    'confidence': 1.0,
                    'method': 'exact_match',
                    'hp_id': phenotype['hp_id'],
                    'matched_term': phenotype['term']
                }
                self.verification_cache[cache_key] = result
                return result
                
            # Check for high similarity match (over 90%)
            similarity = fuzz.ratio(normalized_term, normalized_entity)
            if similarity > 93:
                self._debug_print(f"High similarity match ({similarity}%): '{entity}' matches '{phenotype['term']}' ({phenotype['hp_id']})", level=2)
                result = {
                    'is_phenotype': True,
                    'confidence': similarity / 100.0,
                    'method': 'high_similarity_match',
                    'hp_id': phenotype['hp_id'],
                    'matched_term': phenotype['term']
                }
                self.verification_cache[cache_key] = result
                return result
        
        # Format candidates for the LLM prompt
        candidate_items = []
        for i, phenotype in enumerate(similar_phenotypes, 1):
            candidate_items.append(f"{i}. '{phenotype['term']}' ({phenotype['hp_id']})")
        
        candidates_text = "\n".join(candidate_items)
        
        # Create context part if configured
        context_part = ""
        if context and self.config.use_context_for_direct:
            context_part = f"\nOriginal sentence context: '{context}'"
        
        # Create the binary YES/NO matching prompt
        prompt = (
            f"I need to determine if the entity '{entity}' is a valid human phenotype."
            f"\n\nHere are some HPO phenotype candidates for reference:"
            f"\n\n{candidates_text}\n"
            f"{context_part}\n\n"
            f"A valid phenotype must describe an abnormal characteristic or trait, not just a normal "
            f"anatomical structure, physiological process, laboratory test, or medication."
            f"\n\nBased on these candidates and criteria, is '{entity}' a valid human phenotype?"
            f"\nRespond with ONLY 'YES' or 'NO'."
        )
        
        # Query the LLM
        response = self.llm_client.query(prompt, self.direct_verification_system_message)
        
        # Parse the response - strictly look for "YES" or "NO"
        response_text = response.strip().upper()
        is_phenotype = "YES" in response_text and "NO" not in response_text
        
        # Create result based on binary response
        if is_phenotype:
            result = {
                'is_phenotype': True,
                'confidence': 0.9,
                'method': 'llm_binary_verification'
            }
        else:
            result = {
                'is_phenotype': False,
                'confidence': 0.9,
                'method': 'llm_binary_verification'
            }
        
        # Cache the result
        self.verification_cache[cache_key] = result
        
        self._debug_print(f"LLM binary verification: '{entity}' is{'' if is_phenotype else ' not'} a phenotype", level=2)
        return result

    def contains_number(self, text: str) -> bool:
        """
        Check if text contains any numerical values.
        
        Args:
            text: Text to check for numbers
            
        Returns:
            Boolean indicating if numbers are present
        """
        # Match any digit sequence, including decimal points, etc.
        return bool(re.search(r'\d+(?:\.\d+)?', text))

    def detect_lab_test(self, entity: str, context: Optional[str] = None) -> Dict:
        """
        Detect if an entity is a lab test with a numerical value.
        First checks if there's a number present before querying LLM.
        
        Args:
            entity: Entity text to check
            context: Original sentence containing the entity
            
        Returns:
            Dictionary with detection results
        """
        # Handle empty entities
        if not entity:
            return {
                'is_lab_test': False,
                'confidence': 1.0,
                'method': 'empty_entity'
            }
            
        # Create a cache key
        cache_key = f"lab_detect::{entity}::{context or ''}"
        
        # Check cache
        if cache_key in self.lab_test_detection_cache:
            result = self.lab_test_detection_cache[cache_key]
            self._debug_print(f"Cache hit for lab test detection '{entity}': {result['is_lab_test']}", level=1)
            return result
        
        # Quick check - if no numbers in entity or context, it's not a lab test with value
        has_number_in_entity = self.contains_number(entity)
        has_number_in_context = context and self.contains_number(context)
        
        if not has_number_in_entity and not has_number_in_context:
            self._debug_print(f"Quick check: '{entity}' is not a lab test (no numbers present)", level=1)
            result = {
                'is_lab_test': False,
                'confidence': 0.95,
                'method': 'quick_check_no_numbers'
            }
            self.lab_test_detection_cache[cache_key] = result
            return result
            
        self._debug_print(f"Detecting if '{entity}' is a lab test with measurement (numbers present)", level=1)
        
        # Create context part
        context_part = f"\nOriginal sentence context: '{context}'" if context else ""
        
        # Create the detection prompt
        prompt = (
            f"I need to determine if the entity '{entity}' contains information about a laboratory test with a measured value."
            f"{context_part}\n\n"
            f"Laboratory tests with measured values include examples like:"
            f"\n- 'Hemoglobin 8.5 g/dL'"
            f"\n- 'Elevated white blood cell count of 15,000/μL'"
            f"\n- 'Sodium 140 mEq/L'"
            f"\n\nDoes '{entity}' represent a lab test with a numerical value/result?"
            f"\nRespond with ONLY 'YES' or 'NO'."
        )
        
        # Query the LLM
        response = self.llm_client.query(prompt, self.lab_identification_system_message)
        
        # Parse the response - strictly look for "YES" or "NO"
        response_text = response.strip().upper()
        is_lab_test = "YES" in response_text and "NO" not in response_text
        
        # Create result
        result = {
            'is_lab_test': is_lab_test,
            'confidence': 0.9,
            'method': 'llm_binary_detection'
        }
        
        # Cache the result
        self.lab_test_detection_cache[cache_key] = result
        
        self._debug_print(f"LLM detection: '{entity}' is{'' if is_lab_test else ' not'} a lab test with measurement", level=2)
        return result

    def retrieve_lab_reference_ranges(self, lab_name: str) -> List[Dict]:
        """
        Retrieve reference ranges for a lab test using the lab searcher.
        
        Args:
            lab_name: Name of the lab test
            
        Returns:
            List of reference range information
        """
        if not self.lab_searcher:
            self._debug_print("Lab searcher not initialized, skipping reference range retrieval", level=1)
            return []
            
        try:
            # Search for lab test
            self._debug_print(f"Searching for lab test reference ranges: '{lab_name}'", level=2)
            search_results = self.lab_searcher.search(lab_name)
            
            # Process results
            if not search_results:
                self._debug_print("No reference ranges found", level=2)
                return []
                
            # Format results
            reference_ranges = []
            for result in search_results:
                try:
                    result_data = result.get('result', {})
                    
                    # Skip if no result data
                    if not result_data:
                        continue
                        
                    # Extract reference ranges
                    lab_id = result_data.get('lab_id', 'N/A')
                    name = result_data.get('name', lab_name)
                    ranges = result_data.get('reference_ranges', [])
                    units = result_data.get('units', '')
                    
                    formatted_result = {
                        'lab_id': lab_id,
                        'name': name,
                        'ranges': ranges,
                        'units': units,
                        'similarity': result.get('similarity', 0.0)
                    }
                    
                    reference_ranges.append(formatted_result)
                    
                except Exception as e:
                    self._debug_print(f"Error processing search result: {e}", level=2)
                    continue
            
            self._debug_print(f"Retrieved {len(reference_ranges)} reference range entries", level=2)
            return reference_ranges
            
        except Exception as e:
            self._debug_print(f"Error retrieving lab reference ranges: {e}", level=1)
            return []

    def analyze_lab_test(self, entity: str, context: Optional[str] = None, 
                        sample_data: Optional[Dict] = None) -> Dict:
        """
        Comprehensive analysis of a lab test entity, extracting and determining abnormality.
        
        Args:
            entity: Entity text containing lab test information
            context: Original sentence containing the entity
            sample_data: Optional additional data about the sample (e.g., demographics)
            
        Returns:
            Dictionary with comprehensive lab analysis results
        """
        # Handle empty entities
        if not entity:
            return {
                'lab_name': None,
                'value': None,
                'units': None,
                'is_abnormal': False,
                'abnormality': None,
                'direction': 'unknown',
                'confidence': 0.0,
                'method': 'empty_entity'
            }
            
        # Create a cache key
        sample_data_str = json.dumps(sample_data) if sample_data else ""
        cache_key = f"lab_analysis::{entity}::{context or ''}::{sample_data_str}"
        
        # Check cache
        if cache_key in self.lab_analysis_cache:
            result = self.lab_analysis_cache[cache_key]
            self._debug_print(f"Cache hit for lab analysis '{entity}'", level=1)
            return result
            
        self._debug_print(f"Analyzing lab test '{entity}' for extraction and abnormality", level=1)
        
        # Retrieve lab candidates
        lab_name_guess = entity.split()[0] if entity else ""  # Simple extraction of first word as lab name guess
        reference_ranges = self.retrieve_lab_reference_ranges(lab_name_guess)
        
        # Format reference range information
        reference_info = []
        if reference_ranges:
            for ref_range in reference_ranges[:3]:  # Limit to top 3
                name = ref_range.get('name', lab_name_guess)
                units = ref_range.get('units', '')
                ranges = ref_range.get('ranges', [])
                
                range_strings = []
                for range_item in ranges[:3]:  # Limit to top 3 ranges
                    age_group = range_item.get('age_group', 'Adult')
                    male = range_item.get('male', 'N/A')
                    female = range_item.get('female', 'N/A')
                    range_strings.append(f"  {age_group}: Male: {male}, Female: {female}")
                
                ref_str = f"Lab: {name} (Units: {units})\n"
                ref_str += "\n".join(range_strings)
                reference_info.append(ref_str)
        
        # Create reference part text
        reference_part = ""
        if reference_info:
            reference_part = "Reference Range Information:\n"
            reference_part += "\n\n".join(reference_info)
            reference_part += "\n\n"
        
        # Create context part
        context_part = f"Original Context: {context}\n\n" if context else ""
        
        # Create sample data part
        sample_part = ""
        if sample_data:
            sample_part = "Sample Data:\n"
            for key, value in sample_data.items():
                if value is not None:
                    sample_part += f"  {key}: {value}\n"
            sample_part += "\n"
        
        # Create the analysis prompt
        prompt = (
            f"Analyze this potential laboratory test entity: '{entity}'\n\n"
            f"{context_part}"
            f"{reference_part}"
            f"{sample_part}"
            f"Extract the lab test name, value (with units if available), and determine if the result is abnormal. "
            f"If abnormal, provide a clear medical description of the abnormality (e.g., 'elevated glucose', 'leukopenia')."
            f"\n\nProvide your response in this EXACT JSON format:"
            f"\n{{"
            f"\n  \"lab_name\": \"[extracted lab test name]\","
            f"\n  \"value\": \"[extracted value with units if available]\","
            f"\n  \"units\": \"[extracted units if separable from value]\","
            f"\n  \"is_abnormal\": true/false,"
            f"\n  \"abnormality\": \"[descriptive term for the abnormality, or 'normal' if not abnormal]\","
            f"\n  \"direction\": \"[high/low/normal]\","
            f"\n  \"confidence\": [0.0-1.0 value indicating your confidence]"
            f"\n}}"
            f"\n\nReturn ONLY the JSON with no additional text."
        )
        
        # Query the LLM
        response = self.llm_client.query(prompt, self.lab_analysis_system_message)
        
        # Parse the JSON response
        try:
            # Extract the JSON part from the response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                extracted_json = json_match.group(0)
                analysis_info = json.loads(extracted_json)
                
                # Create result
                result = {
                    'lab_name': analysis_info.get('lab_name', '').lower().strip() if analysis_info.get('lab_name') else None,
                    'value': analysis_info.get('value', '').strip() if analysis_info.get('value') else None,
                    'units': analysis_info.get('units', '').strip() if analysis_info.get('units') else None,
                    'is_abnormal': analysis_info.get('is_abnormal', False),
                    'abnormality': analysis_info.get('abnormality', 'normal').strip(),
                    'direction': analysis_info.get('direction', 'normal').lower().strip(),
                    'confidence': analysis_info.get('confidence', 0.5),
                    'method': 'llm_analysis'
                }
                
                # Cache the result
                self.lab_analysis_cache[cache_key] = result
                
                if result['is_abnormal']:
                    self._debug_print(f"Lab test analysis: '{entity}' is abnormal: {result['abnormality']} ({result['direction']})", level=2)
                else:
                    self._debug_print(f"Lab test analysis: '{entity}' is normal", level=2)
                    
                return result
            else:
                # Failed to find JSON in response
                self._debug_print(f"Failed to extract JSON from response: {response}", level=2)
                result = {
                    'lab_name': None,
                    'value': None,
                    'units': None,
                    'is_abnormal': False,
                    'abnormality': None,
                    'direction': 'unknown',
                    'confidence': 0.0,
                    'method': 'extraction_failed'
                }
                self.lab_analysis_cache[cache_key] = result
                return result
                
        except Exception as e:
            # Failed to parse JSON
            self._debug_print(f"Failed to parse JSON from response ({e}): {response}", level=2)
            result = {
                'lab_name': None,
                'value': None,
                'units': None,
                'is_abnormal': False,
                'abnormality': None,
                'direction': 'unknown',
                'confidence': 0.0,
                'method': 'json_parse_failed'
            }
            self.lab_analysis_cache[cache_key] = result
            return result

    def check_implies_phenotype(self, entity: str, context: Optional[str] = None) -> Dict:
        """
        Check if an entity implies a phenotype with configurable retrieval and context usage.
        
        Args:
            entity: Entity text to check
            context: Original sentence containing the entity
            
        Returns:
            Dictionary with results
        """
        # Handle empty entities
        if not entity:
            return {
                'implies_phenotype': False,
                'confidence': 1.0,
                'method': 'empty_entity'
            }
            
        # Create a cache key - only include context if configured to use it
        cache_key = f"implies::{entity}::{context if self.config.use_context_for_implies else ''}"
        
        # Check cache
        if cache_key in self.implied_phenotype_cache:
            result = self.implied_phenotype_cache[cache_key]
            self._debug_print(f"Cache hit for implied phenotype check '{entity}': {result['implies_phenotype']}", level=1)
            return result
            
        self._debug_print(f"Checking if '{entity}' implies a phenotype", level=1)
        
        # Retrieve similar phenotypes for context if configured
        similar_phenotypes = []
        if self.config.use_retrieval_for_implies:
            similar_phenotypes = self._retrieve_similar_phenotypes(entity)
        
        # Format context items if using retrieval
        context_items = []
        if self.config.use_retrieval_for_implies:
            for phenotype in similar_phenotypes[:10]:  # Use top 10 for context
                context_items.append(f"- {phenotype['term']} ({phenotype['hp_id']})")
        
        context_text = "\n".join(context_items)
        
        # Create prompt for the LLM
        context_part = ""
        if context and self.config.use_context_for_implies:
            context_part = f"Original sentence context: '{context}'\n\n"
        
        retrieval_part = ""
        if self.config.use_retrieval_for_implies and context_items:
            retrieval_part = (
                f"Here are some phenotype terms from the Human Phenotype Ontology for context:\n\n"
                f"{context_text}\n\n"
            )
        
        prompt = (
            f"I need to determine if '{entity}' DIRECTLY AND UNAMBIGUOUSLY implies a specific phenotype. "
            f"Be extremely conservative - only say YES if the implication is clear and specific."
            f"\n{context_part}"
            f"{retrieval_part}"
            f"Laboratory values, medications, or procedures DO NOT imply phenotypes unless there is explicit abnormality mentioned."
            f"\nIf you're uncertain or the implication requires multiple assumptions, say NO."
            f"\n\nDoes '{entity}' directly imply a specific phenotype? "
            f"Respond with ONLY 'YES' if it directly implies a phenotype or 'NO' if it doesn't."
        )
        
        # Query the LLM
        response = self.llm_client.query(prompt, self.implied_phenotype_system_message)
        
        # Parse the response - strictly look for "YES" or "NO"
        response_text = response.strip().upper()
        implies_phenotype = "YES" in response_text and "NO" not in response_text
        
        # Create result
        result = {
            'implies_phenotype': implies_phenotype,
            'confidence': 0.8 if implies_phenotype else 0.9,  # Higher confidence for "no" to be conservative
            'method': 'llm_binary_verification'
        }
        
        # Cache the result
        self.implied_phenotype_cache[cache_key] = result
        
        self._debug_print(f"LLM binary verification: '{entity}' does{'' if implies_phenotype else ' not'} imply a phenotype", level=2)
        return result

    def extract_implied_phenotype(self, entity: str, context: Optional[str] = None) -> Dict:
        """
        Extract the specific phenotype implied by an entity with configurable retrieval and context usage.
        
        Args:
            entity: Entity text that implies a phenotype
            context: Original sentence containing the entity
            
        Returns:
            Dictionary with results
        """
        # Handle empty entities
        if not entity:
            return {
                'implied_phenotype': None,
                'confidence': 0.0,
                'method': 'empty_entity'
            }
            
        # Create a cache key - only include context if configured to use it
        cache_key = f"extract::{entity}::{context if self.config.use_context_for_extract else ''}"
        
        # Check cache
        if cache_key in self.extracted_phenotype_cache:
            result = self.extracted_phenotype_cache[cache_key]
            self._debug_print(f"Cache hit for extracting implied phenotype from '{entity}': {result.get('implied_phenotype')}", level=1)
            return result
            
        self._debug_print(f"Extracting implied phenotype from '{entity}'", level=1)
        
        # Retrieve similar phenotypes for context if configured
        similar_phenotypes = []
        if self.config.use_retrieval_for_extract:
            similar_phenotypes = self._retrieve_similar_phenotypes(entity)
        
        # Format context items if using retrieval
        context_items = []
        if self.config.use_retrieval_for_extract:
            for phenotype in similar_phenotypes[:10]:  # Use top 10 for context
                context_items.append(f"- {phenotype['term']} ({phenotype['hp_id']})")
        
        context_text = "\n".join(context_items)
        
        # Create prompt for the LLM
        context_part = ""
        if context and self.config.use_context_for_extract:
            context_part = f"Original sentence context: '{context}'\n\n"
        
        retrieval_part = ""
        if self.config.use_retrieval_for_extract and context_items:
            retrieval_part = (
                f"Here are some phenotype terms for context:\n\n"
                f"{context_text}\n\n"
            )
        
        prompt = (
            f"The term '{entity}' might imply a phenotype. "
            f"{context_part}"
            f"{retrieval_part}"
            f"What specific phenotype is directly implied by '{entity}'? "
            f"For example, 'hemoglobin of 8 g/dL' implies 'anemia'."
            f"\n\nIf you cannot identify a specific phenotype that is DIRECTLY implied with high confidence, "
            f"respond with EXACTLY 'NO_CLEAR_PHENOTYPE_IMPLIED'."
            f"\n\nProvide ONLY the name of the implied phenotype, without any explanation, "
            f"or 'NO_CLEAR_PHENOTYPE_IMPLIED' if none is clear."
        )
        
        # Query the LLM
        response = self.llm_client.query(prompt, self.extract_phenotype_system_message)
        
        # Clean the response
        implied_phenotype = response.strip()
        implied_phenotype = re.sub(r'[.,;:]$', '', implied_phenotype)
        
        # Check for the special "no clear phenotype" response
        if "NO_CLEAR_PHENOTYPE_IMPLIED" in implied_phenotype.upper():
            result = {
                'implied_phenotype': None,
                'confidence': 0.9,
                'method': 'llm_extraction_no_clear_phenotype'
            }
        else:
            result = {
                'implied_phenotype': implied_phenotype,
                'confidence': 0.7,  # Lower confidence compared to V1 to be more conservative
                'method': 'llm_extraction'
            }
        
        # Cache the result
        self.extracted_phenotype_cache[cache_key] = result
        
        if result['implied_phenotype'] is None:
            self._debug_print(f"LLM could not extract a clear implied phenotype from '{entity}'", level=2)
        else:
            self._debug_print(f"LLM extracted implied phenotype from '{entity}': '{implied_phenotype}'", level=2)
        
        return result

    def validate_phenotype_exists(self, phenotype: str) -> Dict:
        """
        Validate if a phenotype exists by binary YES/NO matching against HPO candidates.
        
        Args:
            phenotype: The phenotype to validate
            
        Returns:
            Dictionary with validation results
        """
        # Skip empty inputs
        if not phenotype:
            return {
                'is_valid': False,
                'confidence': 1.0,
                'method': 'empty_input'
            }
            
        # Create a cache key
        cache_key = f"validate_phenotype::{phenotype}"
        
        # Check cache
        if cache_key in self.phenotype_validation_cache:
            result = self.phenotype_validation_cache[cache_key]
            self._debug_print(f"Cache hit for phenotype validation '{phenotype}': {result['is_valid']}", level=1)
            return result
            
        self._debug_print(f"Validating phenotype '{phenotype}' via binary matching", level=1)
        
        # Check for exact matches using fuzzy matching first (optimization)
        similar_phenotypes = self._retrieve_similar_phenotypes(phenotype, k=self.candidate_count)
        
        for pheno in similar_phenotypes:
            normalized_term = self._normalize_text(pheno['term'])
            normalized_phenotype = self._normalize_text(phenotype)
            
            # Check for exact match
            if normalized_term == normalized_phenotype:
                self._debug_print(f"Exact match found: '{phenotype}' matches '{pheno['term']}' ({pheno['hp_id']})", level=2)
                result = {
                    'is_valid': True,
                    'confidence': 1.0,
                    'method': 'exact_match',
                    'hp_id': pheno['hp_id'],
                    'matched_term': pheno['term']
                }
                self.phenotype_validation_cache[cache_key] = result
                return result
                
            # Check for high similarity match (over 90%)
            similarity = fuzz.ratio(normalized_term, normalized_phenotype)
            if similarity > 93:
                self._debug_print(f"High similarity match ({similarity}%): '{phenotype}' matches '{pheno['term']}' ({pheno['hp_id']})", level=2)
                result = {
                    'is_valid': True,
                    'confidence': similarity / 100.0,
                    'method': 'high_similarity_match',
                    'hp_id': pheno['hp_id'],
                    'matched_term': pheno['term']
                }
                self.phenotype_validation_cache[cache_key] = result
                return result
        
        # Format candidates for the LLM prompt
        candidate_items = []
        for i, pheno in enumerate(similar_phenotypes, 1):
            candidate_items.append(f"{i}. '{pheno['term']}' ({pheno['hp_id']})")
        
        candidates_text = "\n".join(candidate_items)
        
        # Create the binary YES/NO matching prompt
        prompt = (
            f"I need to determine if the phenotype '{phenotype}' is a valid medical concept."
            f"\n\nHere are some HPO phenotype candidates for reference:"
            f"\n\n{candidates_text}\n\n"
            f"Is '{phenotype}' a valid phenotype in clinical medicine? Consider both potential matches "
            f"in the candidates and your general knowledge of medical phenotypes."
            f"\nRespond with ONLY 'YES' or 'NO'."
        )
        
        # Query the LLM
        response = self.llm_client.query(prompt, self.phenotype_validation_system_message)
        
        # Parse the response - strictly look for "YES" or "NO"
        response_text = response.strip().upper()
        is_valid = "YES" in response_text and "NO" not in response_text
        
        # Create result based on binary response
        result = {
            'is_valid': is_valid,
            'confidence': 0.9,
            'method': 'llm_binary_validation'
        }
        
        # Cache the result
        self.phenotype_validation_cache[cache_key] = result
        
        self._debug_print(f"LLM binary validation: '{phenotype}' is{'' if is_valid else ' not'} a valid phenotype", level=2)
        return result

    def validate_implication(self, entity: str, implied_phenotype: str, context: Optional[str] = None) -> Dict:
        """
        Validate if the implication from entity to phenotype is reasonable with strictly binary YES/NO response.
        
        Args:
            entity: Original entity text
            implied_phenotype: Extracted implied phenotype 
            context: Original sentence containing the entity
            
        Returns:
            Dictionary with validation results
        """
        # Skip empty inputs
        if not entity or not implied_phenotype:
            return {
                'is_valid': False,
                'confidence': 1.0,
                'method': 'empty_input'
            }
            
        # Create a cache key - only include context if configured to use it
        cache_key = f"validate_implication::{entity}::{implied_phenotype}::{context if self.config.use_context_for_implication else ''}"
        
        # Check cache
        if cache_key in self.implication_validation_cache:
            result = self.implication_validation_cache[cache_key]
            self._debug_print(f"Cache hit for implication validation '{entity}' → '{implied_phenotype}': {result['is_valid']}", level=1)
            return result
            
        self._debug_print(f"Validating implication from '{entity}' to '{implied_phenotype}' with binary response", level=1)
        
        # Create prompt for the LLM
        context_part = ""
        if context and self.config.use_context_for_implication:
            context_part = f"Original sentence context: '{context}'\n\n"
        
        prompt = (
            f"I need to validate whether the following implication is reasonable:\n\n"
            f"Original entity: '{entity}'\n"
            f"Implied phenotype: '{implied_phenotype}'\n\n"
            f"{context_part}"
            f"Be extremely critical and conservative. Say YES only if there is an unambiguous, "
            f"direct clinical connection between the entity and the proposed phenotype."
            f"\nThe connection must be evident from the entity itself, not inferred from general knowledge."
            f"\n\nIs this a valid and reasonable implication? "
            f"Respond with ONLY 'YES' or 'NO'."
        )
        
        # Query the LLM
        response = self.llm_client.query(prompt, self.implication_validation_system_message)
        
        # Parse the response - strictly look for "YES" or "NO"
        response_text = response.strip().upper()
        is_valid = "YES" in response_text and "NO" not in response_text
        
        # Create result
        result = {
            'is_valid': is_valid,
            'confidence': 0.9,
            'method': 'llm_binary_validation'
        }
        
        # Cache the result
        self.implication_validation_cache[cache_key] = result
        
        self._debug_print(f"LLM binary validation: Implication from '{entity}' to '{implied_phenotype}' is{'' if is_valid else ' not'} valid", level=2)
        return result

    def process_entity(self, entity: str, context: Optional[str] = None, 
                     sample_data: Optional[Dict] = None) -> Dict:
        """
        Process an entity through the enhanced multi-stage pipeline with simplified lab test analysis.
        
        Args:
            entity: Entity text to process
            context: Original sentence containing the entity
            sample_data: Optional dictionary with sample-specific data (demographics, etc.)
            
        Returns:
            Dictionary with processing results
        """
        # Handle empty entities
        if not entity:
            return {
                'status': 'not_phenotype',
                'phenotype': None,
                'original_entity': entity,
                'confidence': 1.0,
                'method': 'empty_entity'
            }
            
        self._debug_print(f"Processing entity: '{entity}'", level=0)
        
        # Clean and preprocess the entity
        cleaned_entity = self.preprocess_entity(entity)
        if not cleaned_entity:
            return {
                'status': 'not_phenotype',
                'phenotype': None,
                'original_entity': entity,
                'confidence': 1.0,
                'method': 'empty_after_preprocessing'
            }
        
        # STAGE 1: Check if it's a direct phenotype using binary matching
        direct_result = self.verify_direct_phenotype(cleaned_entity, context)
        
        # If it's a direct phenotype, return it with details
        if direct_result.get('is_phenotype', False):
            self._debug_print(f"'{entity}' is a direct phenotype", level=1)
            result = {
                'status': 'direct_phenotype',
                'phenotype': direct_result.get('matched_term', cleaned_entity),
                'original_entity': entity,
                'confidence': direct_result['confidence'],
                'method': direct_result['method']
            }
            
            if 'hp_id' in direct_result:
                result['hp_id'] = direct_result['hp_id']
                
            return result
        
        # STAGE 2: Check if entity or context contains numbers (quick check for lab tests)
        has_numbers = self.contains_number(cleaned_entity) or (context and self.contains_number(context))
        
        # If numbers present, check if it's a lab test
        if has_numbers:
            lab_detection_result = self.detect_lab_test(cleaned_entity, context)
            
            if lab_detection_result.get('is_lab_test', False):
                self._debug_print(f"'{entity}' is a lab test, performing analysis", level=1)
                
                # Analyze lab test for abnormality
                lab_analysis_result = self.analyze_lab_test(cleaned_entity, context, sample_data)
                
                # If it's an abnormal lab test, use the abnormality as a phenotype
                if lab_analysis_result.get('is_abnormal', False) and lab_analysis_result.get('abnormality'):
                    abnormality = lab_analysis_result['abnormality']
                    
                    # Skip if abnormality is just "normal"
                    if abnormality.lower() == 'normal':
                        self._debug_print(f"Lab test '{entity}' is normal, not a phenotype", level=1)
                        return {
                            'status': 'not_phenotype',
                            'phenotype': None,
                            'original_entity': entity,
                            'confidence': lab_analysis_result['confidence'],
                            'method': 'normal_lab_value',
                            'lab_info': {
                                'lab_name': lab_analysis_result.get('lab_name'),
                                'value': lab_analysis_result.get('value'),
                                'units': lab_analysis_result.get('units')
                            }
                        }
                    
                    # Validate the abnormality as a phenotype
                    phenotype_validation_result = self.validate_phenotype_exists(abnormality)
                    
                    if phenotype_validation_result.get('is_valid', False):
                        self._debug_print(f"Lab abnormality '{abnormality}' is a valid phenotype", level=1)
                        result = {
                            'status': 'implied_phenotype',
                            'phenotype': phenotype_validation_result.get('matched_term', abnormality),
                            'original_entity': entity,
                            'confidence': min(lab_analysis_result['confidence'], phenotype_validation_result.get('confidence', 0.7)),
                            'method': 'lab_abnormality',
                            'lab_info': {
                                'lab_name': lab_analysis_result.get('lab_name'),
                                'value': lab_analysis_result.get('value'),
                                'units': lab_analysis_result.get('units'),
                                'direction': lab_analysis_result.get('direction')
                            }
                        }
                        
                        # Include HP ID if available
                        if 'hp_id' in phenotype_validation_result:
                            result['hp_id'] = phenotype_validation_result['hp_id']
                            
                        return result
                    
                    # Fall through if abnormality is not a valid phenotype
                    self._debug_print(f"Lab abnormality '{abnormality}' is not a valid phenotype", level=1)
                    
                # If lab test is normal or abnormality is not a valid phenotype, continue to implied phenotype check
                self._debug_print(f"Lab test '{entity}' analysis did not yield a valid phenotype", level=1)
        
        # STAGE 3: Check if it implies a phenotype (for non-lab tests or lab tests that didn't yield valid phenotypes)
        implies_result = self.check_implies_phenotype(cleaned_entity, context)
        
        if not implies_result.get('implies_phenotype', False):
            self._debug_print(f"'{entity}' is not a phenotype and doesn't imply one", level=1)
            return {
                'status': 'not_phenotype',
                'phenotype': None,
                'original_entity': entity,
                'confidence': implies_result['confidence'],
                'method': implies_result.get('method', 'llm_verification')
            }
            
        # STAGE 4: Extract the implied phenotype
        extract_result = self.extract_implied_phenotype(cleaned_entity, context)
        implied_phenotype = extract_result.get('implied_phenotype')
        
        # If we couldn't extract an implied phenotype, not a phenotype
        if not implied_phenotype:
            self._debug_print(f"No clear phenotype could be extracted from '{entity}'", level=1)
            return {
                'status': 'not_phenotype',
                'phenotype': None,
                'original_entity': entity,
                'confidence': extract_result.get('confidence', 0.7),
                'method': extract_result.get('method', 'no_implied_phenotype_found')
            }
        
        # STAGE 5: Validate if the phenotype exists via binary matching
        phenotype_validation_result = self.validate_phenotype_exists(implied_phenotype)
        
        if not phenotype_validation_result.get('is_valid', False):
            self._debug_print(f"Implied phenotype '{implied_phenotype}' from '{entity}' is not valid", level=1)
            return {
                'status': 'not_phenotype',
                'phenotype': None,
                'original_entity': entity,
                'confidence': phenotype_validation_result['confidence'],
                'method': 'invalid_phenotype'
            }
        
        # If there's a matching HPO term, use it
        if 'hp_id' in phenotype_validation_result:
            implied_phenotype = phenotype_validation_result.get('matched_term', implied_phenotype)
            hp_id = phenotype_validation_result['hp_id']
        else:
            hp_id = None
        
        # STAGE 6: Validate if the implication is reasonable with binary response
        implication_validation_result = self.validate_implication(cleaned_entity, implied_phenotype, context)
        
        if not implication_validation_result.get('is_valid', False):
            self._debug_print(f"Implication from '{entity}' to '{implied_phenotype}' is not valid", level=1)
            return {
                'status': 'not_phenotype',
                'phenotype': None,
                'original_entity': entity,
                'confidence': implication_validation_result['confidence'],
                'method': 'invalid_implication'
            }
        
        # If implication and phenotype are both valid, return the implied phenotype
        self._debug_print(f"'{entity}' implies valid phenotype '{implied_phenotype}'", level=1)
        result = {
            'status': 'implied_phenotype',
            'phenotype': implied_phenotype,
            'original_entity': entity,
            'confidence': min(extract_result['confidence'], phenotype_validation_result['confidence']),
            'method': 'multi_stage_pipeline'
        }
        
        # Include HP ID if available
        if hp_id:
            result['hp_id'] = hp_id
            
        return result
    
    def batch_process(self, entities_with_context: List[Dict], 
                     sample_data: Optional[Dict] = None) -> List[Dict]:
        """
        Process a batch of entities with their contexts through the enhanced multi-stage pipeline.
        
        Args:
            entities_with_context: List of dicts with 'entity' and 'context' keys
            sample_data: Optional dictionary with sample-specific data (demographics, etc.)
            
        Returns:
            List of dicts with processing results (phenotypes only)
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
            
            result = self.process_entity(entity, context, sample_data)
            
            # Only include entities that are phenotypes (direct or implied)
            if result['status'] in ['direct_phenotype', 'implied_phenotype']:
                # Add original context
                result['context'] = context
                results.append(result)
        
        self._debug_print(f"Identified {len(results)} phenotypes (direct or implied)")
        return results