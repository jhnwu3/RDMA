#!/usr/bin/env python3
"""
Standalone Abbreviation Detector for Medical Text

This class wraps the abbreviation detection functionality from rdrag.verify.py
into a reusable component that can be used for any downstream task.

Author: Assistant
Created: Based on rdrag.verify.py functionality
"""

import os
import sys
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path

from rdma.utils.search_tools import ToolSearcher
from rdma.utils.embedding import EmbeddingsManager


class AbbreviationDetector:
    """
    Standalone medical abbreviation detector and expander.

    This class provides a simple interface for detecting and expanding clinical
    abbreviations using embedding-based similarity search.

    Key Features:
    - Uses MedEmbed (medical domain embeddings) by default
    - Caches results for improved performance
    - Case-sensitive exact matching with fallback to similarity
    - Configurable similarity thresholds
    - Debug mode for development and testing

    Example Usage:
        detector = AbbreviationDetector(abbreviations_file="abbreviations.npy")
        result = detector.check_abbreviation("MI")
        # Returns: {"is_abbreviation": True, "expanded_term": "myocardial infarction", ...}
    """

    def __init__(
        self,
        abbreviations_file: str = "/home/johnwu3/projects/rare_disease/workspace/repos/RDMA/data/tools/abbreviations_medembed_sm.npy",
        model_type: str = "sentence_transformer",
        model_name: str = "abhinand/MedEmbed-small-v0.1",  # Medical domain default
        device: str = "cpu",
        similarity_threshold: float = 0.96,
        top_k: int = 3,
        debug: bool = False,
        enable_caching: bool = True,
    ):
        """
        Initialize the abbreviation detector.

        Args:
            abbreviations_file: Path to the NPY file containing abbreviation embeddings
            model_type: Type of embedding model ('sentence_transformer', 'fastembed', 'medcpt')
            model_name: Name of the embedding model (ignored for medcpt)
            device: Device to use for embeddings ('cpu', 'cuda', 'cuda:0', etc.)
            similarity_threshold: Minimum similarity score to consider a match (0.0-1.0)
            top_k: Number of top results to retrieve from search
            debug: Enable debug output
            enable_caching: Enable result caching for performance
        """
        self.abbreviations_file = abbreviations_file
        self.model_type = model_type
        self.model_name = model_name
        self.device = device
        self.similarity_threshold = similarity_threshold
        self.top_k = top_k
        self.debug = debug
        self.enable_caching = enable_caching

        # Initialize caching
        self.abbreviation_cache = {} if enable_caching else None

        # Initialize the abbreviation searcher
        self.abbreviation_searcher = None
        self._initialize_searcher()

    def _initialize_searcher(self):
        """Initialize the abbreviation searcher with embeddings."""
        try:
            if not os.path.exists(self.abbreviations_file):
                raise FileNotFoundError(
                    f"Abbreviations file not found: {self.abbreviations_file}"
                )

            self._debug_print(
                f"Initializing abbreviation searcher with {self.abbreviations_file}"
            )
            self._debug_print(
                f"Model: {self.model_type} ({self.model_name if self.model_name else 'built-in'})"
            )
            self._debug_print(f"Device: {self.device}")

            # Handle model_name parameter based on model type
            if self.model_type == "medcpt":
                model_name_to_use = None  # MedCPT doesn't use external model names
                if (
                    self.model_name != "abhinand/MedEmbed-small-v0.1"
                ):  # If user specified something other than default
                    self._debug_print(
                        "Note: Ignoring model_name for MedCPT (uses built-in encoders)"
                    )
            else:
                model_name_to_use = self.model_name

            self.abbreviation_searcher = ToolSearcher(
                model_type=self.model_type,
                model_name=model_name_to_use,
                device=self.device,
                top_k=self.top_k,
            )

            self.abbreviation_searcher.load_embeddings(self.abbreviations_file)
            self._debug_print("Abbreviation searcher initialized successfully")

        except Exception as e:
            self._debug_print(f"Error initializing abbreviation searcher: {e}")
            self.abbreviation_searcher = None
            raise RuntimeError(f"Failed to initialize abbreviation detector: {e}")

    def _debug_print(self, message: str, level: int = 0):
        """Print debug messages if debug mode is enabled."""
        if self.debug:
            indent = "  " * level
            print(f"{datetime.now().strftime('%H:%M:%S')} | {indent}{message}")

    def check_abbreviation(
        self, text: str, context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Check if a text string is a medical abbreviation and expand it.

        This is the main interface function that takes any text and returns
        abbreviation detection results.

        Args:
            text: The text to check for abbreviations
            context: Optional context sentence (currently not used but kept for API compatibility)

        Returns:
            Dictionary containing:
            - is_abbreviation: Boolean indicating if text is an abbreviation
            - expanded_term: Full expansion if abbreviation, None otherwise
            - method: Method used for detection ('exact_match', 'similarity_match', 'quick_check', etc.)
            - similarity_score: Similarity score for similarity matches
            - matched_term: The abbreviation term that matched from database
            - all_matches: Top matches for debugging (if debug=True)

        Example:
            result = detector.check_abbreviation("MI")
            # Returns: {
            #     "is_abbreviation": True,
            #     "expanded_term": "myocardial infarction",
            #     "method": "exact_match",
            #     "matched_term": "MI"
            # }
        """
        # Handle empty input
        if not text or not text.strip():
            return {
                "is_abbreviation": False,
                "expanded_term": None,
                "method": "empty_input",
            }

        # Clean the input
        entity = text.strip()

        # Check cache first
        if self.enable_caching:
            cache_key = f"abbr::{entity}"
            if cache_key in self.abbreviation_cache:
                result = self.abbreviation_cache[cache_key]
                self._debug_print(
                    f"Cache hit for '{entity}': {result['is_abbreviation']}", level=1
                )
                return result

        self._debug_print(f"Checking if '{entity}' is an abbreviation", level=1)

        # Quick heuristic check - does it look like an abbreviation?
        looks_like_abbreviation = self._looks_like_abbreviation(entity)

        if not looks_like_abbreviation:
            result = {
                "is_abbreviation": False,
                "expanded_term": None,
                "method": "quick_check_failed",
                "reason": "Does not match abbreviation patterns",
            }
            self._cache_result(entity, result)
            return result

        # Search for abbreviation using the searcher
        if not self.abbreviation_searcher:
            result = {
                "is_abbreviation": False,
                "expanded_term": None,
                "method": "searcher_not_available",
            }
            self._cache_result(entity, result)
            return result

        return self._search_abbreviation(entity)

    def _looks_like_abbreviation(self, entity: str) -> bool:
        """
        Quick heuristic check if entity looks like an abbreviation.

        Returns True if:
        - All uppercase (e.g., "MI", "CHF")
        - Contains periods (e.g., "i.e.", "etc.")
        - Short length (5 characters or less)
        - Mix of uppercase and periods
        """
        if len(entity) <= 5:
            # Short terms are more likely to be abbreviations
            if entity.isupper():  # All caps like "MI", "CHF"
                return True
            if "." in entity:  # Contains periods like "i.e.", "etc."
                return True
            # Mixed case but short could still be abbreviation
            if len(entity) <= 3:
                return True

        # Longer terms with specific patterns
        if entity.isupper() and len(entity) <= 8:  # Longer all-caps terms
            return True
        if "." in entity:  # Any length with periods
            return True

        return False

    def _search_abbreviation(self, entity: str) -> Dict[str, Any]:
        """
        Search for abbreviation using the embedding-based searcher.
        """
        try:
            search_results = self.abbreviation_searcher.search(entity)

            if not search_results:
                result = {
                    "is_abbreviation": False,
                    "expanded_term": None,
                    "method": "no_match_found",
                }
                self._cache_result(entity, result)
                return result

            # Look for exact match (case-sensitive)
            exact_match = None
            for search_result in search_results:
                matched_term = search_result.get("matched_term", "")
                if matched_term == entity:  # Exact case-sensitive match
                    exact_match = search_result
                    self._debug_print(f"Found exact match for '{entity}'", level=2)
                    break

            # Use exact match if found, otherwise use top result
            if exact_match:
                top_result = exact_match
                is_exact_match = True
                self._debug_print(f"Using exact match for '{entity}'", level=2)
            else:
                top_result = search_results[0]
                is_exact_match = False
                self._debug_print(
                    f"No exact match, using top result for '{entity}'", level=2
                )

            # Extract information
            similarity = top_result.get("similarity", 0.0)
            matched_term = top_result.get("matched_term", "")
            expanded_term = top_result.get("result", "")

            # Determine if this is a good match
            is_abbreviation = is_exact_match or similarity > self.similarity_threshold

            result = {
                "is_abbreviation": is_abbreviation,
                "expanded_term": expanded_term if is_abbreviation else None,
                "method": "exact_match" if is_exact_match else "similarity_match",
                "matched_term": matched_term,
            }

            # Add similarity score for non-exact matches
            if not is_exact_match and is_abbreviation:
                result["similarity_score"] = similarity

            # Add debug information
            if self.debug:
                result["all_matches"] = search_results[: self.top_k]

            # Log result
            if is_abbreviation:
                self._debug_print(
                    f"'{entity}' → '{expanded_term}' (method: {result['method']})",
                    level=2,
                )
            else:
                self._debug_print(
                    f"'{entity}' is not a recognized abbreviation", level=2
                )

            self._cache_result(entity, result)
            return result

        except Exception as e:
            self._debug_print(f"Error searching abbreviation: {e}", level=2)
            result = {
                "is_abbreviation": False,
                "expanded_term": None,
                "method": "search_error",
                "error": str(e),
            }
            self._cache_result(entity, result)
            return result

    def _cache_result(self, entity: str, result: Dict[str, Any]):
        """Cache the result if caching is enabled."""
        if self.enable_caching:
            cache_key = f"abbr::{entity}"
            self.abbreviation_cache[cache_key] = result

    def batch_check(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Check multiple texts for abbreviations in batch.

        Args:
            texts: List of text strings to check

        Returns:
            List of dictionaries with abbreviation results for each text
        """
        results = []
        for text in texts:
            result = self.check_abbreviation(text)
            results.append(result)
        return results

    def get_cache_stats(self) -> Dict[str, int]:
        """Get statistics about the cache usage."""
        if not self.enable_caching:
            return {"caching_disabled": True}

        total_entries = len(self.abbreviation_cache)
        abbreviations_found = sum(
            1
            for result in self.abbreviation_cache.values()
            if result.get("is_abbreviation", False)
        )

        return {
            "total_cached_entries": total_entries,
            "abbreviations_found": abbreviations_found,
            "non_abbreviations": total_entries - abbreviations_found,
            "hit_rate": (
                abbreviations_found / total_entries if total_entries > 0 else 0.0
            ),
        }

    def clear_cache(self):
        """Clear the abbreviation cache."""
        if self.enable_caching:
            self.abbreviation_cache.clear()
            self._debug_print("Cache cleared")

    def set_similarity_threshold(self, threshold: float):
        """
        Update the similarity threshold for abbreviation matching.

        Args:
            threshold: New threshold value (0.0-1.0)
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Threshold must be between 0.0 and 1.0")
        self.similarity_threshold = threshold
        self._debug_print(f"Similarity threshold updated to {threshold}")

        # Clear cache since results may change with new threshold
        if self.enable_caching:
            self.clear_cache()

    def test_examples(self) -> Dict[str, Any]:
        """
        Test the abbreviation detector with common medical abbreviations.

        Returns:
            Dictionary with test results
        """
        test_cases = [
            "MI",  # myocardial infarction
            "CHF",  # congestive heart failure
            "COPD",  # chronic obstructive pulmonary disease
            "UTI",  # urinary tract infection
            "HTN",  # hypertension
            "DM",  # diabetes mellitus
            "CAD",  # coronary artery disease
            "hello",  # not an abbreviation
            "patient",  # not an abbreviation
            "i.e.",  # Latin abbreviation
            "etc.",  # Latin abbreviation
        ]

        results = {}
        self._debug_print("Running test cases...")

        for test_case in test_cases:
            result = self.check_abbreviation(test_case)
            results[test_case] = result

            # Print result summary
            if result["is_abbreviation"]:
                print(f"✓ '{test_case}' → '{result['expanded_term']}'")
            else:
                print(f"✗ '{test_case}' (not an abbreviation)")

        return results


# Convenience factory function
def create_abbreviation_detector(
    abbreviations_file: str, debug: bool = False, **kwargs
) -> AbbreviationDetector:
    """
    Factory function to create an AbbreviationDetector with sensible defaults.

    Args:
        abbreviations_file: Path to abbreviations NPY file
        debug: Enable debug output
        **kwargs: Additional parameters to pass to AbbreviationDetector

    Returns:
        Configured AbbreviationDetector instance
    """
    return AbbreviationDetector(
        abbreviations_file=abbreviations_file, debug=debug, **kwargs
    )


def main():
    """
    Example usage of the AbbreviationDetector.

    This demonstrates how to use the abbreviation detector in your code
    without command line arguments.
    """
    # Example 1: Basic usage with default settings
    print("=" * 60)
    print("Example 1: Basic Usage")
    print("=" * 60)

    # Initialize detector with default medical embeddings
    abbreviations_file = "/home/johnwu3/projects/rare_disease/workspace/repos/RDMA/data/tools/abbreviations_medembed_sm.npy"

    try:
        detector = AbbreviationDetector(
            abbreviations_file=abbreviations_file,
            debug=True,  # Enable debug output for demonstration
        )

        # Test some common medical abbreviations
        test_terms = ["MI", "CHF", "COPD", "UTI", "HTN", "patient", "hello"]

        print("\nChecking individual terms:")
        for term in test_terms:
            result = detector.check_abbreviation(term)
            if result["is_abbreviation"]:
                print(
                    f"✓ '{term}' → '{result['expanded_term']}' (method: {result['method']})"
                )
            else:
                print(f"✗ '{term}' is not an abbreviation (method: {result['method']})")

        print(f"\nCache stats: {detector.get_cache_stats()}")

    except Exception as e:
        print(f"Error with basic example: {e}")
        print("Make sure the abbreviations file path is correct for your system")

    # Example 2: Batch processing
    print("\n" + "=" * 60)
    print("Example 2: Batch Processing")
    print("=" * 60)

    try:
        # Create detector without debug output for cleaner batch results
        detector_batch = AbbreviationDetector(
            abbreviations_file=abbreviations_file,
            debug=False,
            similarity_threshold=0.95,  # Slightly lower threshold
        )

        batch_terms = ["DM", "CAD", "BP", "HR", "patient care", "medical history"]
        results = detector_batch.batch_check(batch_terms)

        print("\nBatch results:")
        for term, result in zip(batch_terms, results):
            if result["is_abbreviation"]:
                print(f"✓ '{term}' → '{result['expanded_term']}'")
            else:
                print(f"✗ '{term}' (not an abbreviation)")

    except Exception as e:
        print(f"Error with batch example: {e}")

    # Example 3: Custom configuration
    print("\n" + "=" * 60)
    print("Example 3: Custom Configuration")
    print("=" * 60)

    try:
        # Example with different model configuration
        detector_custom = AbbreviationDetector(
            abbreviations_file=abbreviations_file,
            model_type="sentence_transformer",
            model_name="abhinand/MedEmbed-small-v0.1",
            device="cpu",
            similarity_threshold=0.90,  # Lower threshold for more matches
            top_k=5,  # Get more results
            debug=False,
            enable_caching=True,
        )

        # Test with some edge cases
        edge_cases = ["i.e.", "etc.", "BP", "temp", "vs", "w/"]

        print("\nTesting edge cases:")
        for term in edge_cases:
            result = detector_custom.check_abbreviation(term)
            if result["is_abbreviation"]:
                expansion = result["expanded_term"]
                score = result.get("similarity_score", "N/A")
                print(f"✓ '{term}' → '{expansion}' (score: {score})")
            else:
                print(
                    f"✗ '{term}' (reason: {result.get('reason', 'not an abbreviation')})"
                )

        # Demonstrate threshold adjustment
        print(f"\nAdjusting similarity threshold to 0.98...")
        detector_custom.set_similarity_threshold(0.98)

        # Recheck a borderline case
        result_strict = detector_custom.check_abbreviation("BP")
        print(f"BP with stricter threshold: {result_strict['is_abbreviation']}")

    except Exception as e:
        print(f"Error with custom configuration: {e}")


# Example usage and testing
if __name__ == "__main__":
    main()
