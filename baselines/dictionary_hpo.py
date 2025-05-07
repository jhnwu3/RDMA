#!/usr/bin/env python3
import argparse
import json
import os
import sys
import numpy as np
import traceback
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from tqdm import tqdm
import re

# Add parent directory to path for module imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Import embedding utilities
from rdma.utils.embedding import EmbeddingsManager


def timestamp_print(message: str) -> None:
    """Print message with timestamp."""
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Retrieval-Enhanced Dictionary Matching for HPO Code Extraction"
    )

    # Input/output files
    parser.add_argument(
        "--input_file", required=True, help="Input JSON file with clinical notes"
    )
    parser.add_argument(
        "--output_file", required=True, help="Output JSON file for extraction results"
    )
    parser.add_argument(
        "--embeddings_file", required=True, help="NPY file containing HPO embeddings"
    )

    # Embedding configuration
    parser.add_argument(
        "--retriever",
        type=str,
        choices=["fastembed", "sentence_transformer", "medcpt"],
        default="sentence_transformer",
        help="Type of retriever/embedding model to use",
    )
    parser.add_argument(
        "--retriever_model",
        type=str,
        default="BAAI/bge-small-en-v1.5",
        help="Model name for retriever",
    )

    # Matching configuration
    parser.add_argument(
        "--top_k",
        type=int,
        default=20,
        help="Number of top candidates to retrieve per sentence",
    )
    parser.add_argument(
        "--similarity_threshold",
        type=float,
        default=0.9,
        help="Similarity threshold for fuzzy matching (0.0-1.0)",
    )

    # Processing configuration
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=10,
        help="Save intermediate results every N cases",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output file if it exists",
    )

    # Debug mode
    parser.add_argument("--debug", action="store_true", help="Enable debug output")

    return parser.parse_args()


def load_hpo_dictionary(embedded_documents: np.ndarray) -> Dict[str, str]:
    """
    Extract HPO dictionary from embedded documents.

    Args:
        embedded_documents: Numpy array of embedded HPO documents

    Returns:
        Dictionary mapping normalized terms to HPO codes
    """
    try:
        # Create dictionary from unique metadata
        hpo_dictionary = {}
        normalized_terms = set()

        for doc in embedded_documents:
            # Extract metadata
            metadata = doc.get("unique_metadata", {})
            term = metadata.get("info", "").lower()
            hp_id = metadata.get("hp_id", "")

            # Normalize term
            normalized_term = _normalize_text(term)

            # Skip empty or duplicate terms
            if not normalized_term or normalized_term in normalized_terms:
                continue

            hpo_dictionary[normalized_term] = hp_id
            normalized_terms.add(normalized_term)

        timestamp_print(
            f"Extracted HPO dictionary with {len(hpo_dictionary)} unique terms"
        )
        return hpo_dictionary

    except Exception as e:
        timestamp_print(f"Error extracting HPO dictionary from embeddings: {e}")
        raise


def _normalize_text(text: str) -> str:
    """
    Normalize text for consistent matching.

    Args:
        text: Input text to normalize

    Returns:
        Normalized text (lowercase, no punctuation)
    """
    if not text:
        return ""

    # Convert to lowercase
    normalized = text.lower()

    # Remove punctuation
    normalized = re.sub(r"[^\w\s]", "", normalized)

    # Remove extra whitespace
    normalized = re.sub(r"\s+", " ", normalized).strip()

    return normalized


def find_exact_match(term: str, hpo_dictionary: Dict[str, str]) -> Optional[str]:
    """
    Find exact match in HPO dictionary.

    Args:
        term: Term to match
        hpo_dictionary: Dictionary of normalized terms to HPO codes

    Returns:
        HPO code if exact match found, else None
    """
    normalized_term = _normalize_text(term)
    return hpo_dictionary.get(normalized_term)


def find_fuzzy_match(
    term: str, hpo_dictionary: Dict[str, str], similarity_threshold: float = 0.9
) -> Optional[str]:
    """
    Find fuzzy match in HPO dictionary using Levenshtein distance.

    Args:
        term: Term to match
        hpo_dictionary: Dictionary of normalized terms to HPO codes
        similarity_threshold: Minimum similarity score to consider a match

    Returns:
        HPO code if fuzzy match found, else None
    """
    from difflib import SequenceMatcher

    normalized_term = _normalize_text(term)

    # Compute edit distance similarity for each dictionary term
    for dict_term, hpo_code in hpo_dictionary.items():
        # Compute similarity ratio
        similarity = SequenceMatcher(None, normalized_term, dict_term).ratio()

        if similarity >= similarity_threshold:
            return hpo_code

    return None


def load_input_data(input_file: str) -> Dict[str, Dict[str, Any]]:
    """
    Load and validate the input JSON file.

    Args:
        input_file: Path to input JSON file

    Returns:
        Dictionary of cases with clinical texts
    """
    try:
        with open(input_file, "r") as f:
            data = json.load(f)

        # Handle different possible input formats
        if isinstance(data, dict) and "results" in data:
            data = data["results"]

        # Standardize data structure
        processed_data = {}
        for case_id, case_data in data.items():
            # Extract clinical text from different possible keys
            if "clinical_text" in case_data:
                clinical_text = case_data["clinical_text"]
            elif "note_details" in case_data and "text" in case_data["note_details"]:
                clinical_text = case_data["note_details"]["text"]
            else:
                timestamp_print(f"Skipping case {case_id}: No clinical text found")
                continue

            processed_data[str(case_id)] = {
                "clinical_text": clinical_text,
                "metadata": case_data.get("metadata", {}),
            }

        return processed_data

    except Exception as e:
        timestamp_print(f"Error loading input file: {e}")
        raise


def load_existing_results(output_file: str) -> Dict[str, Dict[str, Any]]:
    """
    Load existing results from output file if it exists.

    Args:
        output_file: Path to output JSON file

    Returns:
        Dictionary of existing results
    """
    if os.path.exists(output_file):
        try:
            with open(output_file, "r") as f:
                data = json.load(f)

            # Handle different possible result formats
            if "results" in data:
                data = data["results"]

            timestamp_print(f"Loaded existing results for {len(data)} cases")
            return data
        except Exception as e:
            timestamp_print(f"Error loading existing results: {e}")
            return {}
    return {}


def save_checkpoint(results: Dict, output_file: str, checkpoint_num: int) -> None:
    """
    Save intermediate results to a checkpoint file.

    Args:
        results: Results dictionary to save
        output_file: Path to output file
        checkpoint_num: Checkpoint number
    """
    checkpoint_file = (
        f"{os.path.splitext(output_file)[0]}_checkpoint{checkpoint_num}.json"
    )
    with open(checkpoint_file, "w") as f:
        json.dump(results, f, indent=2)
    timestamp_print(f"Saved checkpoint to {checkpoint_file}")


def process_clinical_text(
    clinical_text: str,
    embedding_manager: EmbeddingsManager,
    embedded_documents: np.ndarray,
    args: argparse.Namespace,
) -> List[Dict[str, str]]:
    """
    Process a single clinical text to extract HPO codes using exact and fuzzy matching.

    Args:
        clinical_text: Full clinical text
        embedding_manager: Embedding model for vector retrieval
        embedded_documents: Pre-embedded HPO documents
        args: Parsed command line arguments

    Returns:
        List of extracted HPO code dictionaries
    """
    # Split text into sentences
    import re
    from difflib import SequenceMatcher
    import numpy as np

    # Split into sentences, handling common medical abbreviations
    sentences = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s", clinical_text)
    sentences = [s.strip() for s in sentences if s.strip()]

    extracted_hpo_codes = []
    unique_hpo_codes = set()

    # Prepare FAISS index for searching
    embeddings_array = embedding_manager.prepare_embeddings(embedded_documents)
    faiss_index = embedding_manager.create_index(embeddings_array)

    # Process each sentence
    for sentence in sentences:
        # Normalize the sentence
        clean_sentence = sentence.lower()

        # Embed the sentence
        query_vector = embedding_manager.query_text(sentence).reshape(1, -1)

        # Retrieve top candidates using embeddings
        distances, indices = embedding_manager.search(
            query_vector, faiss_index, k=args.top_k
        )

        # Process retrieved candidates
        for idx, distance in zip(indices[0], distances[0]):
            # Get the candidate metadata
            metadata = embedded_documents[idx]["unique_metadata"]
            term = metadata.get("info", "")
            hp_id = metadata.get("hp_id", "")

            # Skip if no HPO code or already processed
            if not hp_id or hp_id in unique_hpo_codes:
                continue

            # Normalize the term
            clean_term = term.lower()

            # Method 1: Check for exact term match in sentence (case-insensitive)
            if clean_term in clean_sentence:
                extracted_hpo_codes.append(
                    {
                        "entity": term,
                        "context": sentence,
                        "hpo_code": hp_id,
                        "match_method": "exact",
                        "confidence_score": 1.0,
                    }
                )
                unique_hpo_codes.add(hp_id)
                continue

            # Method 2: Advanced n-gram based fuzzy matching
            best_match_score = 0
            best_match_text = ""

            # Get sentence tokens and term tokens
            sentence_tokens = re.findall(r"\b\w+\b", clean_sentence)
            term_tokens = re.findall(r"\b\w+\b", clean_term)

            # Skip terms that are too short for proper comparison
            if len(term_tokens) <= 1:
                continue

            # Generate n-grams from the sentence text
            # For each possible n-gram size from 2 up to the number of tokens in the term
            max_ngram_size = min(len(term_tokens) + 1, 5)  # Up to 4-grams

            for n in range(2, max_ngram_size):
                # Create all possible n-grams from the sentence
                for i in range(len(sentence_tokens) - n + 1):
                    ngram = " ".join(sentence_tokens[i : i + n])

                    # Calculate similarity with the term using SequenceMatcher
                    similarity = SequenceMatcher(None, ngram, clean_term).ratio()

                    # Update best match if this is better
                    if similarity > best_match_score:
                        best_match_score = similarity
                        best_match_text = ngram

            # If best match score exceeds threshold, add to results
            if best_match_score >= args.similarity_threshold:
                extracted_hpo_codes.append(
                    {
                        "entity": term,
                        "matched_text": best_match_text,
                        "context": sentence,
                        "hpo_code": hp_id,
                        "match_method": "fuzzy",
                        "confidence_score": best_match_score,
                    }
                )
                unique_hpo_codes.add(hp_id)

    # Sort results by confidence score
    extracted_hpo_codes.sort(key=lambda x: x["confidence_score"], reverse=True)

    return extracted_hpo_codes


def process_cases(
    cases: Dict[str, Dict[str, Any]],
    embedding_manager: EmbeddingsManager,
    hpo_dictionary: Dict[str, str],
    embedded_documents: np.ndarray,
    args: argparse.Namespace,
    existing_results: Dict = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Process all cases to extract HPO codes.

    Args:
        cases: Dictionary of clinical cases
        embedding_manager: Embedding model for vector retrieval
        hpo_dictionary: Dictionary mapping terms to HPO codes
        embedded_documents: Pre-embedded HPO documents
        args: Parsed command line arguments
        existing_results: Existing results to resume from

    Returns:
        Dictionary of results for each case
    """
    results = existing_results or {}
    checkpoint_counter = 0

    # Determine which cases need processing
    pending_cases = {
        case_id: case_data
        for case_id, case_data in cases.items()
        if case_id not in results or not results[case_id].get("matched_phenotypes")
    }

    timestamp_print(
        f"Processing {len(pending_cases)} cases out of {len(cases)} total cases"
    )

    # Convert to list for progress tracking
    case_items = list(pending_cases.items())

    for i, (case_id, case_data) in enumerate(tqdm(case_items, desc="Processing cases")):
        try:
            if args.debug:
                timestamp_print(
                    f"Processing case {i+1}/{len(pending_cases)} (ID: {case_id})"
                )

            clinical_text = case_data["clinical_text"]

            # Extract raw HPO matches
            raw_matches = process_clinical_text(
                clinical_text,
                embedding_manager,
                embedded_documents,
                args,
            )

            # Map raw output into enriched matched_phenotypes structure
            matched_phenotypes = []
            for match in raw_matches:
                matched_phenotypes.append(
                    {
                        "status": match.get("status", "direct_phenotype"),
                        "phenotype": match.get("entity"),
                        "original_entity": match.get("entity"),
                        "confidence": match.get("confidence", 0.9),
                        "method": match.get("match_method", "retrieval"),
                        "context": match.get("context", clinical_text),
                        "hpo_term": match.get("hpo_code"),
                        "hp_id": match.get("hpo_code"),
                        "match_method": match.get("match_method", "retrieval"),
                        "confidence_score": match.get("confidence_score", 1.0),
                        "top_candidates": match.get("top_candidates", []),
                    }
                )

            # Store results in new output format
            results[case_id] = {
                "original_text": clinical_text,
                "matched_phenotypes": matched_phenotypes,
            }

            checkpoint_counter += 1
            if checkpoint_counter >= args.checkpoint_interval:
                save_checkpoint(results, args.output_file, i + 1)
                checkpoint_counter = 0

        except Exception as e:
            timestamp_print(f"Error processing case {case_id}: {e}")
            if args.debug:
                traceback.print_exc()

            results[case_id] = {
                "original_text": case_data.get("clinical_text", ""),
                "matched_phenotypes": [],
                "error": str(e),
            }

    return results


def main():
    """Main function to run the Retrieval-Enhanced Dictionary Matching pipeline."""
    try:
        # Parse command line arguments
        args = parse_arguments()

        timestamp_print(
            "Starting Retrieval-Enhanced Dictionary Matching for HPO Code Extraction"
        )

        # Load embeddings
        timestamp_print(f"Loading embeddings from {args.embeddings_file}")
        try:
            embedded_documents = np.load(args.embeddings_file, allow_pickle=True)
            timestamp_print(f"Loaded {len(embedded_documents)} embedded documents")
        except Exception as e:
            timestamp_print(f"Error loading embeddings file: {e}")
            raise

        # Extract HPO dictionary from embeddings
        hpo_dictionary = load_hpo_dictionary(embedded_documents)

        # Initialize embedding manager
        timestamp_print(f"Initializing {args.retriever} embedding manager")
        embedding_manager = EmbeddingsManager(
            model_type=args.retriever,
            model_name=(
                args.retriever_model
                if args.retriever in ["fastembed", "sentence_transformer"]
                else None
            ),
        )

        # Load input data
        timestamp_print(f"Loading clinical notes from {args.input_file}")
        cases = load_input_data(args.input_file)
        timestamp_print(f"Loaded {len(cases)} cases")

        # Check for existing results if resuming
        existing_results = {}
        if args.resume:
            existing_results = load_existing_results(args.output_file)

        # Process cases
        timestamp_print(
            "Extracting HPO codes using retrieval-enhanced dictionary matching"
        )
        results = process_cases(
            cases,
            embedding_manager,
            hpo_dictionary,
            embedded_documents,
            args,
            existing_results,
        )

        # Prepare final output with metadata
        final_output = {
            "metadata": {
                "extraction_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "input_file": args.input_file,
                "embeddings_file": args.embeddings_file,
                "matching_parameters": {
                    "retriever": args.retriever,
                    "retriever_model": args.retriever_model,
                    "top_k": args.top_k,
                    "similarity_threshold": args.similarity_threshold,
                },
            },
            "results": results,
        }

        # Save results to JSON
        timestamp_print(f"Saving extraction results to {args.output_file}")
        os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
        with open(args.output_file, "w") as f:
            json.dump(final_output, f, indent=2)

        # Print summary statistics
        total_cases = len(results)
        total_hpo_codes = sum(
            len(case_data.get("hpo_codes", [])) for case_data in results.values()
        )
        cases_with_hpo_codes = sum(
            1 for case_data in results.values() if case_data.get("hpo_codes")
        )

        timestamp_print("Retrieval-Enhanced Dictionary Matching complete:")
        timestamp_print(f"  Total cases processed: {total_cases}")
        timestamp_print(
            f"  Cases with HPO codes: {cases_with_hpo_codes} ({cases_with_hpo_codes/total_cases*100:.1f}%)"
        )
        timestamp_print(f"  Total HPO codes extracted: {total_hpo_codes}")
        timestamp_print(
            f"  Average HPO codes per case: {total_hpo_codes/total_cases:.2f}"
        )

    except Exception as e:
        timestamp_print(f"Critical error: {e}")
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()
