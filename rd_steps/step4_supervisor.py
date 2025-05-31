#!/usr/bin/env python3
import argparse
import json
import os
import sys
import pandas as pd
import numpy as np
import traceback
import torch
from datetime import datetime
from typing import Dict, List, Any, Optional, Set, Tuple
from pathlib import Path
from tqdm import tqdm

# Set correct directory pathing
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Import project modules
from rdma.rdrag.rd_match import RAGRDMatcher
from rdma.rdrag.verify import MultiStageRDVerifier
from rdma.utils.embedding import EmbeddingsManager
from rdma.utils.llm_client import LocalLLMClient, APILLMClient
from rdma.utils.setup import setup_device
from rdma.hporag.context import ContextExtractor


def timestamp_print(message: str) -> None:
    """Print message with timestamp."""
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Supervise evaluation results for rare disease matching"
    )

    # Input/output files
    parser.add_argument(
        "--predictions",
        required=True,
        help="Path to JSON file with predictions from step 3",
    )
    parser.add_argument(
        "--ground-truth", required=True, help="Path to JSON file with ground truth"
    )
    parser.add_argument(
        "--evaluation",
        required=True,
        help="Path to JSON file with evaluation results from step 3",
    )
    parser.add_argument(
        "--output", required=True, help="Path to save supervision results JSON"
    )
    parser.add_argument(
        "--embeddings_file",
        required=True,
        help="NPY file containing rare disease embeddings",
    )
    parser.add_argument(
        "--abbreviations_file",
        type=str,
        help="NPY file containing abbreviation embeddings for resolution",
    )
    # System prompt configuration
    parser.add_argument(
        "--system_prompt",
        type=str,
        default="You are a medical expert specializing in rare diseases with extensive knowledge of Orphanet classifications.",
        help="System prompt for LLM verification",
    )

    # LLM configuration
    parser.add_argument(
        "--llm_type",
        type=str,
        choices=["local", "api"],
        default="local",
        help="Type of LLM to use",
    )
    parser.add_argument(
        "--model_type", type=str, default="llama3_70b", help="Model type for local LLM"
    )
    parser.add_argument(
        "--api_config", type=str, help="Path to API configuration file for API LLM"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Temperature for LLM inference (lower for more consistency)",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="/shared/rsaas/jw3/rare_disease/model_cache",
        help="Directory for caching models",
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
        default="abhinand/MedEmbed-small-v0.1",
        help="Model name for retriever (if using fastembed or sentence_transformer)",
    )

    # Processing configuration
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=10,
        help="Save intermediate results every N cases",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of top candidates to include in verification",
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


def initialize_llm_client(args: argparse.Namespace, device: Dict[str, str]):
    """Initialize appropriate LLM client based on arguments."""
    # Extract the LLM device from the device_info dictionary
    llm_device = device["llm"]

    if args.llm_type == "api":
        if args.api_config:
            return APILLMClient.from_config(args.api_config)
        else:
            return APILLMClient.initialize_from_input()
    else:  # local
        return LocalLLMClient(
            model_type=args.model_type,
            device=llm_device,
            cache_dir=args.cache_dir,
            temperature=args.temperature,
        )


def load_data(
    predictions_file: str, ground_truth_file: str, evaluation_file: str
) -> Tuple[Dict, Dict, Dict]:
    """
    Load prediction, ground truth, and evaluation data from files.

    Args:
        predictions_file: Path to predictions JSON file
        ground_truth_file: Path to ground truth JSON file
        evaluation_file: Path to evaluation results JSON file

    Returns:
        Tuple of (predictions_data, ground_truth_data, evaluation_data)
    """
    # Load predictions
    try:
        with open(predictions_file, "r") as f:
            predictions = json.load(f)

        # Handle nested structure with metadata and results keys
        if isinstance(predictions, dict) and "results" in predictions:
            predictions_data = predictions.get("results", {})
        else:
            predictions_data = predictions

        timestamp_print(f"Loaded predictions for {len(predictions_data)} cases")
    except Exception as e:
        timestamp_print(f"Error loading predictions file: {e}")
        raise

    # Load ground truth
    try:
        with open(ground_truth_file, "r") as f:
            ground_truth = json.load(f)

        timestamp_print(f"Loaded ground truth with {len(ground_truth)} entries")
    except Exception as e:
        timestamp_print(f"Error loading ground truth file: {e}")
        raise

    # Load evaluation results
    try:
        with open(evaluation_file, "r") as f:
            evaluation_data = json.load(f)

        # Check if we have the 'exact_match' key for newer evaluation format
        if "exact_match" in evaluation_data:
            timestamp_print("Found newer evaluation format with 'exact_match' key")
            evaluation_data = evaluation_data["exact_match"]
        elif "set_based_exact_match" in evaluation_data:
            timestamp_print(
                "Found older evaluation format with 'set_based_exact_match' key"
            )
            evaluation_data = evaluation_data["set_based_exact_match"]

        timestamp_print(f"Loaded evaluation results")
    except Exception as e:
        timestamp_print(f"Error loading evaluation file: {e}")
        raise

    return predictions_data, ground_truth, evaluation_data


def extract_entities_with_context(
    predictions_data: Dict, ground_truth_data: Dict, evaluation_data: Dict
) -> Dict[str, List[Dict]]:
    """
    Extract entities and their contexts categorized as false negatives, false positives, and true positives.
    Also preserves top_candidates from matched_diseases when available.

    Args:
        predictions_data: Dictionary containing prediction results
        ground_truth_data: Dictionary containing ground truth data
        evaluation_data: Dictionary containing evaluation results

    Returns:
        Dictionary with categorized entities and their contexts
    """
    # Initialize containers for categorized entities
    entities = {"false_negatives": [], "false_positives": [], "true_positives": []}
    context_extractor = ContextExtractor()
    # Extract false negatives from evaluation results
    if "corpus_false_negatives" in evaluation_data:
        for fn in evaluation_data["corpus_false_negatives"]:
            sample_id = fn.get("sample_id")
            code = fn.get("code")

            if not sample_id or not code:
                continue

            # Find corresponding ground truth entry
            gt_entity = None
            gt_context = None

            # Look up the sample in ground truth
            document_text = ""
            if sample_id in ground_truth_data:
                gt_sample = ground_truth_data[sample_id]

                # Extract from annotations
                if isinstance(gt_sample, dict) and "annotations" in gt_sample:
                    for ann in gt_sample["annotations"]:

                        # gt_entity = ann.get("mention", "")
                        # Get context if available
                        document_text = gt_sample.get("note_details", {}).get(
                            "text", ""
                        )

                        # Extract context around the entity if possible
                        # if (
                        #     document_text
                        #     and gt_entity
                        #     and gt_entity.lower() in document_text.lower()
                        # ):
                        #     gt_context = context_extractor.extract_contexts(
                        #         [gt_entity.lower()],
                        #         document_text.lower(),
                        #         window_size=2,
                        #     )[0]["context"]
            gt_entity = fn.get("name")
            gt_context = context_extractor.extract_contexts(
                                [gt_entity.lower()],
                                document_text.lower(),
                                window_size=2,
                            )[0]["context"]
            # Only include entities with context
            if gt_entity and gt_context:
                entities["false_negatives"].append(
                    {
                        "entity": gt_entity,
                        "context": gt_context,
                        "orpha_code": code,
                        "document_id": sample_id,
                    }
                )
            elif gt_entity:
                # If no context is found, still include the entity
                entities["false_negatives"].append(
                    {
                        "entity": gt_entity,
                        "context": "No context found.",
                        "orpha_code": code,
                        "document_id": sample_id,
                    }
                )

    # Extract false positives and true positives from evaluation results
    for category, eval_key in [
        ("false_positives", "corpus_false_positives"),
        ("true_positives", "corpus_true_positives"),
    ]:
        if eval_key in evaluation_data:
            for item in evaluation_data[eval_key]:
                sample_id = item.get("sample_id")
                code = item.get("code")

                if not sample_id or not code:
                    continue

                # Find corresponding prediction
                pred_entity = None
                pred_context = None
                top_candidates = None  # Store top_candidates from matched entity
                expanded_term = None  # Store expanded abbreviation if available

                # Look up sample in predictions
                if sample_id in predictions_data:
                    pred_sample = predictions_data[sample_id]
                    clinical_text = pred_sample.get("clinical_text", "")

                    # Look for the entity in matched_diseases
                    if "matched_diseases" in pred_sample:
                        for match in pred_sample["matched_diseases"]:
                            # Check if this match corresponds to our code
                            orpha_id = match.get("orpha_id", "")
                            if orpha_id == code or code in orpha_id:
                                pred_entity = match.get("entity", "")

                                # Preserve expanded term if available
                                if "expanded_term" in match:
                                    expanded_term = match["expanded_term"]

                                # Preserve top_candidates if available
                                if "top_candidates" in match:
                                    top_candidates = match["top_candidates"]

                                # Extract context from the clinical text
                                if (
                                    clinical_text
                                    and pred_entity
                                    and pred_entity.lower() in clinical_text.lower()
                                ):
                                    pos = clinical_text.find(pred_entity)
                                    start = max(0, pos - 100)
                                    end = min(
                                        len(clinical_text), pos + len(pred_entity) + 100
                                    )
                                    pred_context = clinical_text[start:end]
                                # If entity not found in clinical text but context is available in match
                                if "context" in match:
                                    pred_context = match["context"]
                                break

                # Only include entities with context
                if pred_entity and pred_context:
                    entity_data = {
                        "entity": pred_entity,
                        "context": pred_context,
                        "orpha_code": code,
                        "document_id": sample_id,
                    }

                    # Add expanded term if available
                    if expanded_term:
                        entity_data["expanded_term"] = expanded_term

                    # Add top_candidates if available
                    if top_candidates:
                        entity_data["top_candidates"] = top_candidates

                    entities[category].append(entity_data)

    # Print statistics
    timestamp_print(f"Extracted entities with context:")
    timestamp_print(f"  False Negatives: {len(entities['false_negatives'])}")
    timestamp_print(f"  False Positives: {len(entities['false_positives'])}")
    timestamp_print(f"  True Positives: {len(entities['true_positives'])}")

    return entities


def process_entities(
    entities: Dict[str, List[Dict]],
    verifier: MultiStageRDVerifier,
    args: argparse.Namespace,
) -> Dict[str, List[Dict]]:
    """
    Process and verify all categorized entities using the MultiStageRDVerifier.

    Args:
        entities: Dictionary with categorized entities
        verifier: MultiStageRDVerifier instance for verification
        args: Command line arguments

    Returns:
        Dictionary with verification results for each category
    """
    results = {"false_negatives": [], "false_positives": [], "true_positives": []}

    checkpoint_counter = 0

    # Process each category
    for category in ["false_negatives", "false_positives", "true_positives"]:
        category_entities = entities[category]

        timestamp_print(f"Processing {len(category_entities)} {category}...")

        # Track progress with tqdm
        for i, entity_data in enumerate(
            tqdm(category_entities, desc=f"Verifying {category}")
        ):
            try:
                entity = entity_data["entity"]
                context = entity_data["context"]
                orpha_code = entity_data["orpha_code"]
                document_id = entity_data["document_id"]

                if args.debug:
                    timestamp_print(
                        f"Processing {category} entity: '{entity}' (Document: {document_id})"
                    )

                if entity is None and "original_entity" in entity_data:
                    entity = entity_data["original_entity"]
                # Process the entity through MultiStageRDVerifier
                verification_result = verifier.process_entity(entity, context)
                print(entity)
                print(verification_result)
                # Check if it's an abbreviation
                is_abbreviation = False
                expanded_term = None

                if "expanded_term" in verification_result:
                    is_abbreviation = True
                    expanded_term = verification_result["expanded_term"]
                    if args.debug:
                        timestamp_print(
                            f"  Detected abbreviation: '{entity}' expands to '{expanded_term}'"
                        )

                # Determine if we should flag for review based on category and verification result
                flag_for_review = False
                explanation = verification_result.get("method", "")

                # Determine if the entity is a rare disease according to the verifier
                is_rare_disease = (
                    verification_result.get("status") == "verified_rare_disease"
                )

                # Logic for flagging based on category and verification result
                if category == "false_positives" and is_rare_disease:
                    # Contradiction: Verifier thinks it's a rare disease but it's marked as false positive
                    flag_for_review = True
                    explanation += " [FLAGGED: Entity verified as a rare disease despite being categorized as false positive]"
                elif category == "false_negatives" and not is_rare_disease:
                    # Contradiction: Verifier thinks it's not a rare disease but it's marked as false negative
                    flag_for_review = True
                    explanation += " [FLAGGED: Entity verified as not a rare disease despite being categorized as false negative]"
                elif category == "true_positives" and not is_rare_disease:
                    # Contradiction: Verifier thinks it's not a rare disease but it's marked as true positive
                    flag_for_review = True
                    explanation += " [FLAGGED: Entity verified as not a rare disease despite being categorized as true positive]"

                # Create the final result
                result = {
                    "entity": entity,
                    "context": context,
                    "is_rare_disease": is_rare_disease,
                    "flag_for_review": flag_for_review,
                    "explanation": explanation,
                    "category": category,
                    "document_id": document_id,
                    "orpha_code": orpha_code,
                    "verification_method": verification_result.get("method", ""),
                }

                # Add expanded term if it's an abbreviation
                if is_abbreviation and expanded_term:
                    result["expanded_term"] = expanded_term

                # Add candidates if available (from top_candidates or from new retrieval)
                if "top_candidates" in entity_data and entity_data["top_candidates"]:
                    result["orpha_candidates"] = entity_data["top_candidates"]
                else:
                    # Use query for retrieval based on expanded term if available
                    query_term = expanded_term if expanded_term else entity
                    candidates = verifier._retrieve_similar_diseases(query_term)
                    if candidates:
                        # FIXED: The candidates from _retrieve_similar_diseases have a different structure
                        result["orpha_candidates"] = [
                            {
                                "name": c.get("name", ""),
                                "id": c.get("id", ""),
                                "similarity": float(c.get("similarity_score", 0.0)),
                            }
                            for c in candidates[:5]  # Include up to 5 candidates
                        ]

                # Add to results
                results[category].append(result)

                # Debug output
                if args.debug:
                    timestamp_print(
                        f"  Result: is_rare_disease={result['is_rare_disease']}, "
                        f"flag_for_review={result['flag_for_review']}"
                    )
                    timestamp_print(f"  Explanation: {result['explanation']}")

                # Save checkpoint if interval reached
                checkpoint_counter += 1
                if checkpoint_counter >= args.checkpoint_interval:
                    save_checkpoint(results, args.output, i, category)
                    checkpoint_counter = 0

            except Exception as e:
                timestamp_print(
                    f"Error processing {category} entity '{entity_data.get('entity', '')}': {e}"
                )
                if args.debug:
                    traceback.print_exc()

                # Add error result
                error_result = {
                    "entity": entity_data.get("entity", ""),
                    "context": entity_data.get("context", ""),
                    "document_id": entity_data.get("document_id", ""),
                    "orpha_code": entity_data.get("orpha_code", ""),
                    "is_rare_disease": False,
                    "flag_for_review": True,  # Always flag for review on errors
                    "explanation": f"Error during processing: {str(e)}",
                    "category": category,
                    "error": str(e),
                }

                results[category].append(error_result)

    return results


def save_checkpoint(
    results: Dict[str, List[Dict]], output_file: str, checkpoint_num: int, category: str
) -> None:
    """
    Save intermediate results to a checkpoint file.

    Args:
        results: Dictionary with verification results
        output_file: Path to output file
        checkpoint_num: Current checkpoint number
        category: Current category being processed
    """
    checkpoint_file = f"{os.path.splitext(output_file)[0]}_checkpoint_{category}_{checkpoint_num}.json"

    # Create metadata for checkpoint
    checkpoint_data = {
        "metadata": {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "checkpoint_num": checkpoint_num,
            "category": category,
        },
        "results": results,
    }

    os.makedirs(os.path.dirname(os.path.abspath(checkpoint_file)), exist_ok=True)
    with open(checkpoint_file, "w") as f:
        json.dump(checkpoint_data, f, indent=2)

    timestamp_print(f"Saved checkpoint to {checkpoint_file}")


def prepare_summary(verification_results: Dict[str, List[Dict]]) -> Dict[str, Any]:
    """
    Prepare a summary of the verification results.

    Args:
        verification_results: Dictionary with verification results for each category

    Returns:
        Dictionary with summary statistics
    """
    summary = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "categories": {},
    }

    # Initialize arrays for overall statistics
    all_flagged_for_review = []

    # Calculate statistics for each category
    for category in ["false_negatives", "false_positives", "true_positives"]:
        results = verification_results[category]

        # Initialize category statistics
        category_stats = {
            "total": len(results),
            "confirmed_rare_disease_count": 0,
            "confirmed_rare_disease_percentage": 0.0,
            "flagged_for_review_count": 0,
            "flagged_for_review_percentage": 0.0,
            "confirmation_status": {},
        }

        # Count by confirmation and review status
        is_rare_counts = {"YES": 0, "NO": 0}
        flag_for_review_counts = {"YES": 0, "NO": 0}

        for result in results:
            # Count by rare disease status
            if result.get("is_rare_disease", False):
                is_rare_counts["YES"] += 1
            else:
                is_rare_counts["NO"] += 1

            # Count by review flag status
            if result.get("flag_for_review", False):
                flag_for_review_counts["YES"] += 1
                # Add to overall flagged list
                all_flagged_for_review.append(
                    {
                        "entity": result.get("entity", ""),
                        "document_id": result.get("document_id", ""),
                        "orpha_code": result.get("orpha_code", ""),
                        "category": category,
                        "explanation": result.get("explanation", ""),
                    }
                )
            else:
                flag_for_review_counts["NO"] += 1

        # Calculate percentages
        if category_stats["total"] > 0:
            category_stats["confirmed_rare_disease_count"] = is_rare_counts["YES"]
            category_stats["confirmed_rare_disease_percentage"] = (
                is_rare_counts["YES"] / category_stats["total"] * 100
            )

            category_stats["flagged_for_review_count"] = flag_for_review_counts["YES"]
            category_stats["flagged_for_review_percentage"] = (
                flag_for_review_counts["YES"] / category_stats["total"] * 100
            )

        # Store detailed status counts
        category_stats["confirmation_status"] = {
            "is_rare_disease": is_rare_counts,
            "flag_for_review": flag_for_review_counts,
        }

        # Add to summary
        summary["categories"][category] = category_stats

    # Add overall statistics
    summary["total_entities"] = sum(
        summary["categories"][cat]["total"] for cat in summary["categories"]
    )
    summary["total_flagged_for_review"] = len(all_flagged_for_review)
    summary["flagged_for_review_percentage"] = (
        (summary["total_flagged_for_review"] / summary["total_entities"] * 100)
        if summary["total_entities"] > 0
        else 0.0
    )

    # Add flagged entities list for easy access
    summary["flagged_entities"] = all_flagged_for_review

    return summary


def main():
    """Main function for the supervisor script."""
    try:
        # Parse command line arguments
        args = parse_arguments()

        timestamp_print(f"Starting rare disease supervision process")

        # Setup device
        from rdma.utils.setup import setup_device

        devices = setup_device(args)
        timestamp_print(f"Using device: {devices}")

        # Initialize LLM client
        timestamp_print(f"Initializing {args.llm_type} LLM client")
        llm_client = initialize_llm_client(args, devices)

        # Initialize embedding manager
        timestamp_print(f"Initializing {args.retriever} embedding manager")
        embedding_manager = EmbeddingsManager(
            model_type=args.retriever,
            model_name=(
                args.retriever_model
                if args.retriever in ["fastembed", "sentence_transformer"]
                else None
            ),
            device=devices["retriever"],
        )

        # Load embeddings
        timestamp_print(f"Loading embeddings from {args.embeddings_file}")
        try:
            embedded_documents = np.load(args.embeddings_file, allow_pickle=True)
            timestamp_print(f"Loaded {len(embedded_documents)} embedded documents")
        except Exception as e:
            timestamp_print(f"Error loading embeddings file: {e}")
            raise

        # Initialize MultiStageRDVerifier instead of RAGRDMatcher
        timestamp_print(f"Initializing MultiStageRDVerifier")

        # Import the MultiStageRDVerifier class
        from rdma.rdrag.verify import MultiStageRDVerifier

        verifier = MultiStageRDVerifier(
            embedding_manager=embedding_manager,
            llm_client=llm_client,
            config=None,  # No specific config needed
            debug=args.debug,
            abbreviations_file=(
                args.abbreviations_file if hasattr(args, "abbreviations_file") else None
            ),
            use_abbreviations=True,  # Enable abbreviation resolution
        )

        # Prepare verifier index
        timestamp_print(f"Preparing verifier index")
        verifier.prepare_index(embedded_documents)

        # Load data files
        timestamp_print(f"Loading data files")
        predictions_data, ground_truth_data, evaluation_data = load_data(
            args.predictions, args.ground_truth, args.evaluation
        )

        # Extract entities with contexts
        timestamp_print(f"Extracting entities with contexts")
        entities = extract_entities_with_context(
            predictions_data, ground_truth_data, evaluation_data
        )

        # Process entities with the verifier
        timestamp_print(f"Starting entity verification")
        verification_results = process_entities(entities, verifier, args)

        # Prepare summary
        timestamp_print(f"Preparing results summary")
        summary = prepare_summary(verification_results)

        # Save results to file
        timestamp_print(f"Saving results to {args.output}")
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

        # Create final output with all results
        final_output = {
            "metadata": {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "predictions_file": args.predictions,
                "ground_truth_file": args.ground_truth,
                "evaluation_file": args.evaluation,
                "model_info": {
                    "llm_type": args.llm_type,
                    "model_type": args.model_type,
                    "temperature": args.temperature,
                    "retriever": args.retriever,
                    "retriever_model": args.retriever_model,
                },
            },
            "summary": summary,
            "results": verification_results,
        }

        with open(args.output, "w") as f:
            json.dump(final_output, f, indent=2)

        # Output summary to console
        timestamp_print("\n=== Supervision Summary ===")
        timestamp_print(f"Total entities processed: {summary['total_entities']}")
        timestamp_print(
            f"Total entities flagged for review: {summary['total_flagged_for_review']} "
            f"({summary['flagged_for_review_percentage']:.1f}%)"
        )

        for category in ["false_negatives", "false_positives", "true_positives"]:
            stats = summary["categories"][category]
            timestamp_print(f"\n{category.replace('_', ' ').title()}:")
            timestamp_print(f"  Total: {stats['total']}")
            timestamp_print(
                f"  Confirmed as rare disease: {stats['confirmed_rare_disease_count']} "
                f"({stats['confirmed_rare_disease_percentage']:.1f}%)"
            )
            timestamp_print(
                f"  Flagged for review: {stats['flagged_for_review_count']} "
                f"({stats['flagged_for_review_percentage']:.1f}%)"
            )

        # Create simple CSV of flagged entities for human review
        csv_path = os.path.splitext(args.output)[0] + "_flagged_for_review.csv"
        if summary["flagged_entities"]:
            timestamp_print(f"\nSaving flagged entities to CSV: {csv_path}")
            df = pd.DataFrame(summary["flagged_entities"])
            df.to_csv(csv_path, index=False)

        timestamp_print(f"Supervision process completed successfully.")

    except Exception as e:
        timestamp_print(f"Critical error: {e}")
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()
