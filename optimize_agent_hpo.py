from typing import List, Dict, Any, Optional, Tuple, Union, Set
import json
import re
import numpy as np
import torch
from datetime import datetime
from fuzzywuzzy import fuzz
import itertools
import time
from utils.embedding import EmbeddingsManager


def load_mistral_llm_client():
    """
    Load a Mistral 24B LLM client configured with default cache directories
    and assigned to cuda:0 device.
    
    Returns:
        LocalLLMClient: Initialized LLM client for Mistral 24B
    """
    from utils.llm_client import LocalLLMClient
    
    # Default cache directory from mine_hpo.py
    default_cache_dir = "/shared/rsaas/jw3/rare_disease/model_cache"
    
    # Initialize and return the client with specific configuration
    llm_client = LocalLLMClient( # for condor
        model_type="mistral_24b",  # Explicitly request mistral_24b model
        device="cuda",           # Assign to first GPU (cuda:0)
        cache_dir=default_cache_dir,
        temperature=0.1            # Default temperature from mine_hpo.py
    )
    
    return llm_client

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


from hporag.entity import BaseEntityExtractor
import json
import pandas as pd
import numpy as np
import re
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from fuzzywuzzy import fuzz
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



import re
from typing import List, Dict, Optional, Union, Any
from datetime import datetime


class ContextExtractor:
    """
    Extracts relevant context for entities from clinical notes.
    
    This class is responsible for finding the most relevant sentences or sections
    in a clinical note that provide context for extracted entities. It can be used
    by different components in the pipeline to ensure consistent context extraction.
    """
    
    def __init__(self, debug: bool = False):
        """
        Initialize the context extractor.
        
        Args:
            debug: Whether to print debug information
        """
        self.debug = debug
    
    def _debug_print(self, message: str, level: int = 0):
        """Print debug messages if debug mode is enabled."""
        if self.debug:
            indent = "  " * level
            print(f"{datetime.now().strftime('%H:%M:%S')} | {indent}{message}")
    
    def extract_sentences(self, text: str) -> List[str]:
        """
        Split clinical text into sentences.
        
        Args:
            text: Clinical note text
            
        Returns:
            List of sentences extracted from the text
        """
        # First split by common sentence terminators while preserving them
        sentence_parts = []
        for part in re.split(r'([.!?])', text):
            if part.strip():
                if part in '.!?':
                    if sentence_parts:
                        sentence_parts[-1] += part
                else:
                    sentence_parts.append(part.strip())
        
        # Then handle other clinical note delimiters like line breaks and semicolons
        sentences = []
        for part in sentence_parts:
            # Split by semicolons and newlines
            for subpart in re.split(r'[;\n]', part):
                if subpart.strip():
                    sentences.append(subpart.strip())
        
        self._debug_print(f"Extracted {len(sentences)} sentences from text")
        return sentences
    
    def find_entity_context(self, entity: str, sentences: List[str], 
                          window_size: int = 0) -> Optional[str]:
        """
        Find the most relevant context for a given entity.
        
        Args:
            entity: Entity to find context for
            sentences: List of sentences from the clinical note
            window_size: Number of additional sentences to include (default: 0 - just the matching sentence)
            
        Returns:
            The most relevant context or None if no match found
        """
        entity_lower = entity.lower()
        
        # Try exact matching first
        for i, sentence in enumerate(sentences):
            if entity_lower in sentence.lower():
                # Found exact match - just return the sentence for token efficiency
                return sentence.strip()
        
        # If no exact match, try fuzzy matching based on word overlap
        entity_words = set(re.findall(r'\b\w+\b', entity_lower))
        if not entity_words:
            return None
            
        best_match = None
        best_score = 0
        
        for sentence in sentences:
            sentence_words = set(re.findall(r'\b\w+\b', sentence.lower()))
            if not sentence_words:
                continue
                
            common_words = entity_words & sentence_words
            # Calculate Jaccard similarity
            similarity = len(common_words) / (len(entity_words) + len(sentence_words) - len(common_words))
            
            # Prioritize sentences with higher word overlap
            overlap_ratio = len(common_words) / len(entity_words) if entity_words else 0
            
            # Combined score giving more weight to overlap ratio
            score = (0.7 * overlap_ratio) + (0.3 * similarity)
            
            if score > best_score:
                best_score = score
                best_match = sentence
        
        # If we found a reasonably good match
        if best_score > 0.3 and best_match:
            return best_match.strip()
        
        return None
    
    def extract_contexts(self, entities: List[str], text: str, 
                        window_size: int = 0) -> List[Dict[str, str]]:
        """
        Extract contexts for a list of entities from a clinical note.
        
        Args:
            entities: List of entities to find context for
            text: Clinical note text
            window_size: Number of additional sentences to include (default: 0 - just the matching sentence)
            
        Returns:
            List of dictionaries with entity and context pairs
        """
        self._debug_print(f"Extracting contexts for {len(entities)} entities")
        
        # Extract sentences once
        sentences = self.extract_sentences(text)
        
        # Find context for each entity
        results = []
        for entity in entities:
            context = self.find_entity_context(entity, sentences, window_size)
            results.append({
                "entity": entity,
                "context": context or ""  # Empty string if no context found
            })
            
            self._debug_print(f"Entity: '{entity}'", level=1)
            self._debug_print(f"Context: '{context}'", level=2)
        
        return results
    
    def batch_extract_contexts(self, batch_entities: List[List[str]], 
                             texts: List[str], 
                             window_size: int = 0) -> List[List[Dict[str, str]]]:
        """
        Extract contexts for multiple batches of entities from multiple clinical notes.
        
        Args:
            batch_entities: List of lists of entities
            texts: List of clinical note texts
            window_size: Number of additional sentences to include (default: 0 - just the matching sentence)
            
        Returns:
            List of lists of dictionaries with entity and context pairs
        """
        if len(batch_entities) != len(texts):
            raise ValueError(f"Mismatch between number of entity batches ({len(batch_entities)}) and texts ({len(texts)})")
            
        results = []
        for entities, text in zip(batch_entities, texts):
            batch_results = self.extract_contexts(entities, text, window_size)
            results.append(batch_results)
            
        return results


# Standalone function for simpler usage
def extract_entity_contexts(entities: List[str], text: str, window_size: int = 0) -> List[Dict[str, str]]:
    """
    Utility function to extract contexts for entities from a clinical note.
    
    Args:
        entities: List of entities to find context for
        text: Clinical note text
        window_size: Number of additional sentences to include (default: 0 - just the matching sentence)
        
    Returns:
        List of dictionaries with entity and context pairs
    """
    extractor = ContextExtractor()
    return extractor.extract_contexts(entities, text, window_size)



if __name__ == "__main__":
    truth =  [
      {
        "phenotype_name": "3(+) proteinuria",
        "hpo_id": "HP:0000093"
      },
      {
        "phenotype_name": "focal segmental glomerulonephritis (FSGS)",
        "hpo_id": "HP:0000097"
      },
      {
        "phenotype_name": "bilateral hearing loss",
        "hpo_id": "HP:0000365"
      },
      {
        "phenotype_name": "severe bilateral sensorineural hearing loss",
        "hpo_id": "HP:0000407"
      },
      {
        "phenotype_name": "hypertension",
        "hpo_id": "HP:0000822"
      },
      {
        "phenotype_name": "pallor (in setting of anemia)",
        "hpo_id": "HP:0001017"
      },
      {
        "phenotype_name": "cardiomegaly",
        "hpo_id": "HP:0001640"
      },
      {
        "phenotype_name": "mild mitral regurgitation",
        "hpo_id": "HP:0001653"
      },
      {
        "phenotype_name": "left ventricular hypertrophy",
        "hpo_id": "HP:0001712"
      },
      {
        "phenotype_name": "progressive hearing loss",
        "hpo_id": "HP:0001730"
      },
      {
        "phenotype_name": "normochromic anaemia",
        "hpo_id": "HP:0001895"
      },
      {
        "phenotype_name": "normocytic anaemia",
        "hpo_id": "HP:0001897"
      },
      {
        "phenotype_name": "anemia",
        "hpo_id": "HP:0001903"
      },
      {
        "phenotype_name": "vomiting",
        "hpo_id": "HP:0002013"
      },
      {
        "phenotype_name": "suprapubic tenderness",
        "hpo_id": "HP:0002027"
      },
      {
        "phenotype_name": "breathlessness/dyspneic",
        "hpo_id": "HP:0002094"
      },
      {
        "phenotype_name": "elevated uric acid",
        "hpo_id": "HP:0002149"
      },
      {
        "phenotype_name": "elevated serum calcium",
        "hpo_id": "HP:0002901"
      },
      {
        "phenotype_name": "elevated blood urea",
        "hpo_id": "HP:0003138"
      },
      {
        "phenotype_name": "serum creatinine 11.2\u2005mg/dL",
        "hpo_id": "HP:0003259"
      },
      {
        "phenotype_name": "loss of appetite",
        "hpo_id": "HP:0004396"
      },
      {
        "phenotype_name": "loss of corticomedullary differentiation",
        "hpo_id": "HP:0005564"
      },
      {
        "phenotype_name": "bilateral sensorineural hearing loss",
        "hpo_id": "HP:0008625"
      },
      {
        "phenotype_name": "pedal oedema",
        "hpo_id": "HP:0010741"
      },
      {
        "phenotype_name": "low-grade fever",
        "hpo_id": "HP:0011134"
      },
      {
        "phenotype_name": "mild anisocytosis",
        "hpo_id": "HP:0011273"
      },
      {
        "phenotype_name": "bilateral anterior lentiglobus",
        "hpo_id": "HP:0011501"
      },
      {
        "phenotype_name": "posterior lenticonus",
        "hpo_id": "HP:0011502"
      },
      {
        "phenotype_name": "bilateral anterior lentiglobus",
        "hpo_id": "HP:0011527"
      },
      {
        "phenotype_name": "sinus tachycardia",
        "hpo_id": "HP:0011703"
      },
      {
        "phenotype_name": "82% polymorphs",
        "hpo_id": "HP:0011897"
      },
      {
        "phenotype_name": "pyuria",
        "hpo_id": "HP:0012085"
      },
      {
        "phenotype_name": "bacteriuria",
        "hpo_id": "HP:0012461"
      },
      {
        "phenotype_name": "bilateral contracted kidneys",
        "hpo_id": "HP:0012586"
      },
      {
        "phenotype_name": "haematuria",
        "hpo_id": "HP:0012587"
      },
      {
        "phenotype_name": "pus cells in urine",
        "hpo_id": "HP:0012614"
      },
      {
        "phenotype_name": "diffuse thickening of glomerular basement",
        "hpo_id": "HP:0025005"
      },
      {
        "phenotype_name": "blot haemorrhages in retina",
        "hpo_id": "HP:0025242"
      },
      {
        "phenotype_name": "peripheral retina revealed multiple yellowish white lesion-like flecks in the mid-periphery",
        "hpo_id": "HP:0030506"
      },
      {
        "phenotype_name": "bibasilar end-inspiratory crepitations in lungs",
        "hpo_id": "HP:0031998"
      },
      {
        "phenotype_name": "visual impairment",
        "hpo_id": "HP:0032037"
      },
      {
        "phenotype_name": "increased cortical echogenecity",
        "hpo_id": "HP:0033132"
      },
      {
        "phenotype_name": "disruption of glomerular basement membrane",
        "hpo_id": "HP:0033485"
      },
      {
        "phenotype_name": "discontinuity of lamina densa",
        "hpo_id": "HP:0033803"
      },
      {
        "phenotype_name": "dysuria",
        "hpo_id": "HP:0100518"
      },
      {
        "phenotype_name": "anuria",
        "hpo_id": "HP:0100519"
      },
      {
        "phenotype_name": "dilated left ventricular cavity",
        "hpo_id": "HP:4000141"
      },
      {
        "phenotype_name": "oil droplet sign",
        "hpo_id": "HP:6000027"
      }
    ]
    print(len(truth))

    ground_truth = []
    for item in truth:
        ground_truth.append(item["phenotype_name"])

    llm_client = load_mistral_llm_client()
    text =  "A 35-year-old woman presented to the medical emergency department with low-grade fever for 3 weeks, vomiting for 1 week and anuria for 3 days. She also reported dysuria and breathlessness for 1 week. There was no history of decreased urine output, dialysis, effort intolerance, chest pain or palpitation, dyspnoea and weight loss. Menstrual history was within normal limit but she reported gradually progressive loss of appetite. Family history included smoky urine in her younger brother in his childhood, who died in an accident. On general survey, the patient was conscious and alert. She was dyspnoeic and febrile. Severe pallor was present with mild pedal oedema. Blood pressure was 180/100 mm Hg and pulse rate of 116/min regular. No evidence of jaundice, clubbing cyanosis or lymphadenopathy was found. Physical examination revealed bibasilar end-inspiratory crepitations in lungs and suprapubic tenderness. There was no hepatosplenomegaly or ascites. Cardiac examination was normal. She was found to have severe bilateral hearing loss, which was gradually progressive for 5 years. The fundi were bilaterally pale. The patient was referred to the department of ophthalmology for a comprehensive eye examination. Her visual acuity was documented as 6/18 in both eyes with no obvious lenticular opacity. Slit-lamp examination showed bilateral anterior lentiglobus with posterior lenticonus. Distant direct ophthalmoscopy revealed oil droplet sign (a suggestive confirmation of the presence of lenticonus); and peripheral retina revealed multiple yellowish white lesion-like flecks in the mid-periphery, and few blot haemorrhages indicative of hypertensive changes. Haemogram showed haemoglobin (Hb) 5.7 g/dL, erythrocyte sedimentation rate 15 mm in first hour, white cell count 17 200/\u00b5L with 82% polymorphs, adequate platelets and mean corpuscular volume 83.3 fL. Peripheral smear showed normocytic normochromic anaemia with mild anisocytosis. Fasting sugar 78 mg/dL, blood urea 325 mg/dL, serum creatinine 11.2 mg/dL and uric acid 8.3 mg/dL. Liver function tests were within normal limit as were serum electrolytes, except serum calcium (conc.) 5.8 mg/dL (adjusted with serum albumin). Lipid profile and iron profile were also normal. HIV and viral markers for HbsAg and hepatitis C virus were non-reactive. ECG showed sinus tachycardia with features of left ventricular hypertrophy and chest X-ray posteroanterior view revealed cardiomegaly. Urinalysis showed full field of pus cells with 35\u201340 RBCs/hpf and 3(+) proteinuria. Urine samples for cultures were sent which reported pure growth of Escherichia coli. Spot urine for protein:creatinine ratio was 2.07 g/g Cr. She underwent pure tone audiometry which revealed features suggestive of severe bilateral sensorineural hearing loss (SHNL). Ultrasound of the abdomen showed bilateral contracted kidneys: right measured 6.7\u00d72.3 cm and left 7.8\u00d73 cm, with increased cortical echogenecity and loss of corticomedullary differentiation, suggestive of medical renal disease. Two-dimensional Echo reported dilated left ventricular cavity with mild mitral regurgitation and ejection fraction of 55%. Renal and skin biopsies were conducted and specimens were sent for light and electron microscopy (EM). Renal tissue on H&E stain was reported as focal segmental glomerulonephritis (FSGS). Ultrathin sections of EM study of renal tissue revealed disruption of glomerular basement membrane (GBM) with diffuse thickening of glomerular capillary wall. Dermal tissue depicted discontinuity of lamina densa with basket weaving pattern under EM."

    
    system_message_I = "You are a rare disease expert with extensive medical knowledge. Carefully review every sentence of the clinical passage to identify terms related to genetic inheritance patterns, anatomical anomalies, clinical symptoms, diagnostic findings, lab test results, and specific conditions or syndromes. Completely ignore negative findings, normal findings (i.e. 'normal' or 'no'), procedures and family history.  Return the extracted terms in a JSON object with a single key 'findings', which contains the list of extracted terms spelled correctly. Ensure the output is concise without any additional notes, commentary, or meta explanations."

    extractor = IterativeLLMEntityExtractor(
        llm_client=llm_client,
        # llm_client=llama_client,
        system_message=system_message_I,
        max_iterations=10 # iterative extractor
    )
    extracted = extractor.process_batch([text])
    extractor = ContextExtractor(debug=False)
    # Extract contexts for all entities
    extracted = extractor.batch_extract_contexts(extracted, [text], window_size=1)

    embeddings_manager = EmbeddingsManager(
        model_type="sentence_transformer",
        model_name="abhinand/MedEmbed-small-v0.1",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    

    
    # 4. Load the embeddings and prepare the index
    embedded_documents = embeddings_manager.load_documents(
    "data/vector_stores/G2GHPO_metadata_medembed.npy"
    )
    
    # we're only optimizing for a single sample here to save time
    best_config = run_optimization(embeddings_manager, llm_client, ground_truth, extracted[0], embedded_documents, debug=True)
    print("Best Config:")
    print(best_config)