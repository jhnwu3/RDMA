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
        description="Retrieval-Enhanced Dictionary Matching for Rare Disease Extraction"
    )

    # Input/output files
    parser.add_argument(
        "--input_file", required=True, help="Input JSON file with clinical notes"
    )
    parser.add_argument(
        "--output_file", required=True, help="Output JSON file for extraction results"
    )
    parser.add_argument(
        "--embeddings_file",
        required=True,
        help="NPY file containing rare disease embeddings",
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
        default=5,
        help="Number of top candidates to include in results",
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

    # GPU configuration
    gpu_group = parser.add_mutually_exclusive_group()
    gpu_group.add_argument("--gpu_id", type=int, help="Specific GPU ID to use")
    gpu_group.add_argument(
        "--condor",
        action="store_true",
        help="Use generic CUDA device without specific GPU ID (for job schedulers)",
    )
    gpu_group.add_argument(
        "--cpu", action="store_true", help="Force CPU usage even if GPU is available"
    )

    # Debug mode
    parser.add_argument("--debug", action="store_true", help="Enable debug output")

    return parser.parse_args()


def setup_device(args: argparse.Namespace) -> Dict[str, str]:
    """Configure devices for embeddings based on command line arguments."""
    import torch

    # Initialize result dictionary
    devices = {}

    # Configure device
    if args.cpu:
        devices["retriever"] = "cpu"
    elif args.condor:
        if torch.cuda.is_available():
            timestamp_print(
                "Using generic CUDA device in condor/job scheduler environment"
            )
            devices["retriever"] = "cuda"
        else:
            timestamp_print(
                "Warning: CUDA requested but not available. Falling back to CPU."
            )
            devices["retriever"] = "cpu"
    elif args.gpu_id is not None:
        if torch.cuda.is_available():
            if args.gpu_id < torch.cuda.device_count():
                devices["retriever"] = f"cuda:{args.gpu_id}"
            else:
                timestamp_print(
                    f"Warning: GPU {args.gpu_id} requested but only {torch.cuda.device_count()} GPUs available. Using GPU 0."
                )
                devices["retriever"] = "cuda:0"
        else:
            timestamp_print(
                f"Warning: GPU {args.gpu_id} requested but no CUDA available. Falling back to CPU."
            )
            devices["retriever"] = "cpu"
    else:
        devices["retriever"] = "cuda" if torch.cuda.is_available() else "cpu"

    return devices


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

    # Remove punctuation except hyphens (important for disease names)
    normalized = re.sub(r"[^\w\s-]", "", normalized)

    # Remove extra whitespace
    normalized = re.sub(r"\s+", " ", normalized).strip()

    return normalized


def load_verification_results(input_file: str) -> Dict:
    """Load clinical notes from input file with handling for various formats."""
    try:
        with open(input_file, "r") as f:
            data = json.load(f)

        # Check if the data has the expected structure with "metadata" and "results" keys
        if isinstance(data, dict) and "results" in data:
            timestamp_print(f"Found structured output with 'results' key")
            results = data["results"]
            metadata = data.get("metadata", {})

            return {"metadata": metadata, "results": results}
        else:
            # Fallback for older format where the entire JSON is the results
            timestamp_print(f"Using legacy format - treating entire JSON as results")
            results = {}

            # Process the data to extract clinical text from note_details
            for case_id, case_data in data.items():
                # Extract clinical text from note_details structure
                if (
                    isinstance(case_data, dict)
                    and "note_details" in case_data
                    and "text" in case_data["note_details"]
                ):
                    clinical_text = case_data["note_details"]["text"]
                    results[case_id] = {
                        "clinical_text": clinical_text,
                        "metadata": {
                            "patient_id": case_data.get("patient_id", ""),
                            "admission_id": case_data.get("admission_id", ""),
                            "category": case_data.get("category", ""),
                            "chart_date": case_data.get("chart_date", ""),
                        },
                    }
                # Handle case where clinical_text is directly available
                elif isinstance(case_data, dict) and "clinical_text" in case_data:
                    results[case_id] = case_data
                # Handle other formats
                else:
                    timestamp_print(
                        f"Warning: Could not extract clinical text for case {case_id}"
                    )
                    results[case_id] = {"clinical_text": "", "metadata": {}}

            return {"metadata": {}, "results": results}
    except Exception as e:
        timestamp_print(f"Error loading input file: {e}")
        raise


def load_existing_results(output_file: str) -> Dict:
    """Load existing matching results if available."""
    if os.path.exists(output_file):
        try:
            with open(output_file, "r") as f:
                data = json.load(f)

            # Handle case where the results are wrapped in a metadata structure
            if "results" in data:
                data = data["results"]

            timestamp_print(
                f"Loaded existing results for {len(data)} cases from {output_file}"
            )
            return data
        except Exception as e:
            timestamp_print(f"Error loading existing results: {e}")
            return {}
    return {}


def save_checkpoint(results: Dict, output_file: str, checkpoint_num: int) -> None:
    """Save intermediate results to a checkpoint file."""
    checkpoint_file = (
        f"{os.path.splitext(output_file)[0]}_checkpoint{checkpoint_num}.json"
    )
    with open(checkpoint_file, "w") as f:
        json.dump(results, f, indent=2)
    timestamp_print(f"Saved checkpoint to {checkpoint_file}")


def convert_to_serializable(obj):
    """Convert all non-serializable types to serializable ones."""
    import numpy as np

    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list) or isinstance(obj, tuple):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    else:
        # For any other types, convert to string
        return str(obj)


def process_clinical_text(
    clinical_text: str,
    embedding_manager: EmbeddingsManager,
    embedded_documents: np.ndarray,
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    """
    Process a single clinical text to extract rare disease entities.

    Args:
        clinical_text: Full clinical text
        embedding_manager: Embedding model for vector retrieval
        embedded_documents: Pre-embedded rare disease documents
        args: Parsed command line arguments

    Returns:
        List of extracted rare disease entities
    """
    # Split text into sentences
    import re
    from difflib import SequenceMatcher
    import numpy as np

    # Split into sentences, handling common medical abbreviations
    sentences = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s", clinical_text)
    sentences = [s.strip() for s in sentences if s.strip()]

    extracted_entities = []
    unique_entities = set()

    # Prepare FAISS index for searching
    embeddings_array = embedding_manager.prepare_embeddings(embedded_documents)
    faiss_index = embedding_manager.create_index(embeddings_array)

    # Process each sentence
    for sentence in sentences:
        # Skip very short sentences
        if len(sentence) < 10:
            continue

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
            if "name" in embedded_documents[idx] and "id" in embedded_documents[idx]:
                # Direct structure
                term = embedded_documents[idx].get("name", "")
                orpha_id = embedded_documents[idx].get("id", "")
                definition = embedded_documents[idx].get("definition", "")
            else:
                # Nested structure with unique_metadata
                metadata = embedded_documents[idx].get("unique_metadata", {})
                term = metadata.get("name", "")
                orpha_id = metadata.get("id", "")
                definition = metadata.get("definition", "")

            # Skip if no ORPHA code or already processed
            entity_key = f"{term}::{orpha_id}"
            if not orpha_id or entity_key in unique_entities:
                continue

            # Normalize the term
            clean_term = term.lower()

            # Method 1: Check for exact term match in sentence (case-insensitive)
            if clean_term in clean_sentence:
                extracted_entities.append(
                    {
                        "entity": term,
                        "rd_term": term,
                        "orpha_id": orpha_id,
                        "context": sentence,
                        "match_method": "exact",
                        "confidence_score": 1.0,
                        "top_candidates": [
                            {"name": term, "id": orpha_id, "similarity": 1.0}
                        ],
                    }
                )
                unique_entities.add(entity_key)
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
                extracted_entities.append(
                    {
                        "entity": best_match_text,
                        "rd_term": term,
                        "orpha_id": orpha_id,
                        "context": sentence,
                        "match_method": "fuzzy",
                        "confidence_score": best_match_score,
                        "top_candidates": [
                            {
                                "name": term,
                                "id": orpha_id,
                                "similarity": best_match_score,
                            }
                        ],
                    }
                )
                unique_entities.add(entity_key)

    # Sort results by confidence score
    extracted_entities.sort(key=lambda x: x["confidence_score"], reverse=True)

    return extracted_entities


def match_cases(
    verification_results: Dict,
    embedding_manager: EmbeddingsManager,
    embedded_documents: np.ndarray,
    args: argparse.Namespace,
    existing_results: Dict = None,
) -> Dict:
    """
    Match verified entities to rare disease terms using dictionary-based approach.
    """
    results = existing_results or {}
    checkpoint_counter = 0

    # Extract the actual verification results
    cases = verification_results.get("results", verification_results)

    # Determine which cases need processing
    pending_cases = {
        case_id: case_data
        for case_id, case_data in cases.items()
        if case_id not in results or not results[case_id].get("matched_diseases")
    }

    timestamp_print(
        f"Matching rare diseases for {len(pending_cases)} cases out of {len(cases)} total cases"
    )

    # Convert to list for progress tracking
    case_items = list(pending_cases.items())

    # Use tqdm for progress tracking
    for i, (case_id, case_data) in enumerate(
        tqdm(case_items, desc="Matching rare diseases")
    ):
        try:
            if args.debug:
                timestamp_print(
                    f"Processing case {i+1}/{len(pending_cases)} (ID: {case_id})"
                )

            # Get clinical text for processing - handle different possible structures
            clinical_text = ""
            if "clinical_text" in case_data:
                clinical_text = case_data["clinical_text"]
            elif "note_details" in case_data and "text" in case_data["note_details"]:
                clinical_text = case_data["note_details"]["text"]

            if not clinical_text:
                timestamp_print(f"No clinical text found for case {case_id}")
                results[case_id] = {
                    "clinical_text": "",
                    "metadata": case_data.get("metadata", {}),
                    "matched_diseases": [],
                    "note": "No clinical text found",
                }
                continue

            # Process clinical text to extract rare disease entities
            matched_diseases = process_clinical_text(
                clinical_text, embedding_manager, embedded_documents, args
            )

            # Store results in output format
            results[case_id] = {
                "clinical_text": clinical_text,
                "metadata": case_data.get("metadata", {}),
                "matched_diseases": matched_diseases,
                "stats": {
                    "verified_diseases_count": (
                        len(case_data.get("verified_rare_diseases", []))
                        if "verified_rare_diseases" in case_data
                        else 0
                    ),
                    "matched_diseases_count": len(matched_diseases),
                },
            }

            # Save checkpoint if interval reached
            checkpoint_counter += 1
            if checkpoint_counter >= args.checkpoint_interval:
                save_checkpoint(results, args.output_file, i + 1)
                checkpoint_counter = 0

        except Exception as e:
            timestamp_print(f"Error processing case {case_id}: {e}")
            if args.debug:
                traceback.print_exc()
            # Still add the case to results but mark as failed
            results[case_id] = {
                "clinical_text": case_data.get("clinical_text", ""),
                "metadata": case_data.get("metadata", {}),
                "matched_diseases": [],
                "stats": {
                    "verified_diseases_count": (
                        len(case_data.get("verified_rare_diseases", []))
                        if "verified_rare_diseases" in case_data
                        else 0
                    ),
                    "matched_diseases_count": 0,
                },
                "error": str(e),
            }

    return results


def main():
    """Main function to run the retrieval-enhanced dictionary matching for rare diseases."""
    import torch

    try:
        # Parse command line arguments
        args = parse_arguments()

        timestamp_print(f"Starting rare disease matching process")

        # Setup device
        device = setup_device(args)
        timestamp_print(f"Using device: {device}")

        # Initialize embedding manager
        timestamp_print(f"Initializing {args.retriever} embedding manager")
        embedding_manager = EmbeddingsManager(
            model_type=args.retriever,
            model_name=(
                args.retriever_model
                if args.retriever in ["fastembed", "sentence_transformer"]
                else None
            ),
            device=device["retriever"],
        )

        # Load embeddings
        timestamp_print(f"Loading embeddings from {args.embeddings_file}")
        try:
            embedded_documents = np.load(args.embeddings_file, allow_pickle=True)
            timestamp_print(f"Loaded {len(embedded_documents)} embedded documents")
        except Exception as e:
            timestamp_print(f"Error loading embeddings file: {e}")
            raise

        # Load verification results from input file
        timestamp_print(f"Loading clinical notes from {args.input_file}")
        verification_data = load_verification_results(args.input_file)

        # Extract results and metadata
        verification_results = verification_data.get("results", verification_data)
        verification_metadata = verification_data.get("metadata", {})

        timestamp_print(
            f"Loaded verification results for {len(verification_results)} cases"
        )

        # Check for existing results if resuming
        existing_results = {}
        if args.resume:
            existing_results = load_existing_results(args.output_file)

        # Match verified entities to rare disease terms
        timestamp_print(f"Starting rare disease matching")
        results = match_cases(
            verification_results,
            embedding_manager,
            embedded_documents,
            args,
            existing_results,
        )

        # Save results to JSON
        timestamp_print(f"Saving matching results to {args.output_file}")
        os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)

        # Add metadata about the matching run
        metadata = {
            "verification_metadata": verification_metadata,
            "matching_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model_info": {
                "retriever": args.retriever,
                "retriever_model": args.retriever_model,
            },
        }

        # Create final output with metadata
        final_output = {"metadata": metadata, "results": results}

        # Convert to serializable format
        serializable_output = convert_to_serializable(final_output)

        with open(args.output_file, "w") as f:
            json.dump(serializable_output, f, indent=2)

        # Print summary
        total_verified_entities = sum(
            case_data.get("stats", {}).get("verified_diseases_count", 0)
            for case_data in results.values()
        )
        total_matched_entities = sum(
            case_data.get("stats", {}).get("matched_diseases_count", 0)
            for case_data in results.values()
        )

        # Calculate match rate
        match_rate = (
            (total_matched_entities / total_verified_entities * 100)
            if total_verified_entities > 0
            else 0
        )

        timestamp_print(f"Matching complete:")
        timestamp_print(f"  Total verified rare diseases: {total_verified_entities}")
        timestamp_print(
            f"  Successfully matched to ORPHA codes: {total_matched_entities} ({match_rate:.1f}%)"
        )

        timestamp_print(f"Rare disease matching completed successfully.")

    except Exception as e:
        timestamp_print(f"Critical error: {e}")
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()
