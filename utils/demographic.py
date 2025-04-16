import re
import json
from typing import Dict, Optional, Any
from datetime import datetime

class DemographicsExtractor:
    """
    Module for extracting demographic information from clinical text.
    
    This module handles extraction of patient demographics including:
    - Age (exact value or estimated range)
    - Gender
    - Age group (infant, child, adolescent, adult, elderly)
    
    Features:
    - Caching to avoid redundant extractions
    - Standardized output format
    - Age group inference from exact age when possible
    """
    
    def __init__(self, llm_client, debug=False):
        """
        Initialize the demographics extractor.
        
        Args:
            llm_client: LLM client for querying the language model
            debug: Whether to print debug information
        """
        self.llm_client = llm_client
        self.debug = debug
        self.cache = {}
        
        # System message for demographic extraction
        self.system_message = (
            "You are a medical information extraction specialist. "
            "Your task is to extract demographic information (age and gender) from clinical notes. "
            "Be precise with the information you extract and return it in the requested JSON format. "
            "If the information is not present, return null for that field."
        )
    
    def _debug_print(self, message: str, level: int = 0):
        """Print debug messages if debug mode is enabled."""
        if self.debug:
            indent = "  " * level
            print(f"{datetime.now().strftime('%H:%M:%S')} | {indent}{message}")
    
    def extract(self, clinical_note: str) -> Dict[str, Any]:
        """
        Extract demographic information from a clinical note.
        
        Args:
            clinical_note: Full clinical note text
            
        Returns:
            Dictionary with demographic information (age, gender, age_group)
        """
        if not clinical_note:
            return {
                'age': None,
                'gender': None,
                'age_group': None
            }
            
        # Create a cache key
        cache_key = f"demographics::{hash(clinical_note[:1000])}"  # Use first 1000 chars for hashing
        
        # Check cache
        if cache_key in self.cache:
            result = self.cache[cache_key]
            self._debug_print(f"Cache hit for demographics extraction", level=1)
            return result
        
        self._debug_print(f"Extracting demographics from clinical note", level=1)
        
        # Create the extraction prompt
        prompt = (
            f"Extract age and gender information from the following clinical note. "
            f"If exact age is not mentioned, estimate age group (infant, child, adolescent, adult, elderly).\n\n"
            f"Clinical Note: {clinical_note[:3000]}...\n\n"  # Limit to first 3000 chars
            f"Provide your response in this EXACT JSON format:"
            f"\n{{"
            f"\n  \"age\": [exact age in years if mentioned, otherwise null],"
            f"\n  \"gender\": [\"male\", \"female\", or null if not mentioned],"
            f"\n  \"age_group\": [\"infant\" (0-1), \"child\" (2-11), \"adolescent\" (12-17), "
            f"\"adult\" (18-64), \"elderly\" (65+), or null if unknown]"
            f"\n}}"
            f"\n\nReturn ONLY the JSON with no additional text."
        )
        
        # Query the LLM
        response = self.llm_client.query(prompt, self.system_message)
        
        # Parse the JSON response
        try:
            # Extract the JSON part from the response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                extracted_json = json_match.group(0)
                demographics = json.loads(extracted_json)
                
                # Standardize values
                gender = demographics.get('gender', '').lower() if demographics.get('gender') else None
                age = demographics.get('age')
                age_group = demographics.get('age_group', '').lower() if demographics.get('age_group') else None
                
                # Validate gender
                if gender not in ['male', 'female', None]:
                    gender = None
                    
                # Validate age_group
                valid_age_groups = ['infant', 'child', 'adolescent', 'adult', 'elderly']
                if age_group not in valid_age_groups:
                    age_group = None
                    
                # Determine age_group from age if age_group is None but age is provided
                if age_group is None and age is not None:
                    try:
                        age_value = float(age)
                        if 0 <= age_value <= 1:
                            age_group = 'infant'
                        elif 2 <= age_value <= 11:
                            age_group = 'child'
                        elif 12 <= age_value <= 17:
                            age_group = 'adolescent'
                        elif 18 <= age_value <= 64:
                            age_group = 'adult'
                        elif age_value >= 65:
                            age_group = 'elderly'
                    except (ValueError, TypeError):
                        pass
                
                result = {
                    'age': age,
                    'gender': gender,
                    'age_group': age_group
                }
                
                # Cache the result
                self.cache[cache_key] = result
                
                self._debug_print(f"Extracted demographics: age={age}, gender={gender}, age_group={age_group}", level=2)
                return result
        
        except Exception as e:
            self._debug_print(f"Error extracting demographics: {e}", level=2)
        
        # Return empty demographics on any error
        default_result = {
            'age': None,
            'gender': None,
            'age_group': None
        }
        self.cache[cache_key] = default_result
        return default_result
    
    def clear_cache(self):
        """Clear the extraction cache."""
        self.cache = {}
        self._debug_print("Demographics extraction cache cleared")
    
    def get_age_group(self, age: float) -> str:
        """
        Convert a numeric age to an age group category.
        
        Args:
            age: Age in years
            
        Returns:
            Age group string ('infant', 'child', 'adolescent', 'adult', 'elderly')
        """
        if age is None:
            return None
            
        try:
            age_value = float(age)
            if 0 <= age_value <= 1:
                return 'infant'
            elif 2 <= age_value <= 11:
                return 'child'
            elif 12 <= age_value <= 17:
                return 'adolescent'
            elif 18 <= age_value <= 64:
                return 'adult'
            elif age_value >= 65:
                return 'elderly'
            else:
                return None
        except (ValueError, TypeError):
            return None