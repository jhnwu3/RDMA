#!/usr/bin/env python3
import argparse
import json
import os
import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz
from typing import List, Dict, Tuple, Optional, Set, Any
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

# so I can call it outside of the directory it's stuck in. 
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)


def normalize_text(text: str) -> str:
    """Normalize text by converting to lowercase and removing extra whitespace."""
    return ' '.join(text.lower().split())


def calculate_similarity_matrix(predictions: List[str], ground_truth: List[str], 
                               similarity_threshold: float = 80.0) -> np.ndarray:
    """
    Calculate the similarity matrix between predictions and ground truth using fuzzy matching.
    
    Args:
        predictions: List of predicted entities
        ground_truth: List of ground truth entities
        similarity_threshold: Minimum similarity score (0-100) to consider a potential match
        
    Returns:
        2D numpy array of similarity scores
    """
    # Create similarity matrix
    similarity_matrix = np.zeros((len(predictions), len(ground_truth)))
    
    for i, pred in enumerate(predictions):
        for j, truth in enumerate(ground_truth):
            similarity = fuzz.ratio(normalize_text(pred), normalize_text(truth))
            # Only consider matches above threshold
            if similarity >= similarity_threshold:
                similarity_matrix[i, j] = similarity
    
    return similarity_matrix


def find_best_matches(similarity_matrix: np.ndarray) -> List[Tuple[int, int, float]]:
    """
    Find the best matches between predictions and ground truth using a greedy approach.
    
    Args:
        similarity_matrix: 2D array of similarity scores
        
    Returns:
        List of (pred_idx, truth_idx, similarity) tuples representing the matches
    """
    matches = []
    
    # Create a copy to modify
    sim_matrix = similarity_matrix.copy()
    
    # While there are non-zero elements in the matrix
    while np.max(sim_matrix) > 0:
        # Find the highest similarity
        max_val = np.max(sim_matrix)
        max_pos = np.unravel_index(np.argmax(sim_matrix), sim_matrix.shape)
        
        # Add the match
        pred_idx, truth_idx = max_pos
        matches.append((pred_idx, truth_idx, max_val))
        
        # Mark the row and column as used
        sim_matrix[pred_idx, :] = 0
        sim_matrix[:, truth_idx] = 0
    
    return matches


def set_based_evaluation(predictions: List[str], ground_truth: List[str],
                         similarity_threshold: float = 80.0) -> Dict:
    """
    Evaluates predictions against ground truth using set operations and fuzzy matching.
    
    Args:
        predictions: List of predicted entities (can contain duplicates)
        ground_truth: List of ground truth entities (can contain duplicates)
        similarity_threshold: Minimum similarity score to consider a potential match
        
    Returns:
        Dictionary with precision, recall, F1 scores and detailed match information
    """
    # First, deduplicate both lists
    unique_predictions = list(set(predictions))
    unique_ground_truth = list(set(ground_truth))
    
    # Store original counts for reference
    pred_counter = Counter(predictions)
    truth_counter = Counter(ground_truth)
    
    # Default empty result with all required fields
    result = {
        "precision": 0.0,
        "recall": 0.0,
        "f1_score": 0.0,
        "matches": [],
        "true_positives": [],
        "false_positives": [],
        "false_negatives": [],
        "tp_count": 0,
        "fp_count": 0,
        "fn_count": 0,
        "weighted_tp": 0.0,
        "pred_counter": dict(pred_counter),
        "truth_counter": dict(truth_counter),
        "unique_pred_count": len(unique_predictions),
        "unique_truth_count": len(unique_ground_truth),
        "total_pred_count": len(predictions),
        "total_truth_count": len(ground_truth)
    }
    
    # Handle empty sets
    if not unique_predictions or not unique_ground_truth:
        # Set precision to 1.0 if no predictions (no false positives)
        if not unique_predictions:
            result["precision"] = 1.0
        # Populate false positives and false negatives
        result["false_positives"] = [{"text": p, "count": pred_counter[p]} for p in unique_predictions]
        result["false_negatives"] = [{"text": t, "count": truth_counter[t]} for t in unique_ground_truth]
        result["fp_count"] = len(unique_predictions)
        result["fn_count"] = len(unique_ground_truth)
        return result
    
    # Calculate similarity matrix
    similarity_matrix = calculate_similarity_matrix(unique_predictions, unique_ground_truth, similarity_threshold)
    
    # Find best matches
    matches = find_best_matches(similarity_matrix)
    
    # Create match details
    match_details = []
    matched_pred_indices = set()
    matched_truth_indices = set()
    
    for pred_idx, truth_idx, similarity in matches:
        pred_text = unique_predictions[pred_idx]
        truth_text = unique_ground_truth[truth_idx]
        
        match_details.append({
            "prediction": pred_text,
            "ground_truth": truth_text,
            "similarity": similarity,
            "pred_count": pred_counter[pred_text],
            "truth_count": truth_counter[truth_text]
        })
        
        result["true_positives"].append({
            "prediction": pred_text,
            "ground_truth": truth_text,
            "similarity": similarity
        })
        
        matched_pred_indices.add(pred_idx)
        matched_truth_indices.add(truth_idx)
    
    # Identify false positives and negatives using set operations
    false_positive_indices = set(range(len(unique_predictions))) - matched_pred_indices
    false_negative_indices = set(range(len(unique_ground_truth))) - matched_truth_indices
    
    result["false_positives"] = [{"text": unique_predictions[i], "count": pred_counter[unique_predictions[i]]} for i in false_positive_indices]
    result["false_negatives"] = [{"text": unique_ground_truth[i], "count": truth_counter[unique_ground_truth[i]]} for i in false_negative_indices]
    
    # Store matches
    result["matches"] = match_details
    
    # Calculate metrics based on unique items (set-based)
    result["tp_count"] = len(matched_pred_indices)
    result["fp_count"] = len(false_positive_indices)
    result["fn_count"] = len(false_negative_indices)
    
    # Calculate weighted true positives based on similarity scores
    result["weighted_tp"] = sum(similarity / 100.0 for _, _, similarity in matches)
    
    # Calculate precision and recall
    if unique_predictions:
        result["precision"] = result["weighted_tp"] / len(unique_predictions)
    if unique_ground_truth:
        result["recall"] = result["weighted_tp"] / len(unique_ground_truth)
    
    # Calculate F1 score
    if result["precision"] + result["recall"] > 0:
        result["f1_score"] = 2 * (result["precision"] * result["recall"]) / (result["precision"] + result["recall"])
    
    return result


def load_data(entities_file: str, ground_truth_file: str) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Load entity extraction results and ground truth data from files.
    
    Args:
        entities_file: Path to entities JSON file
        ground_truth_file: Path to ground truth JSON file
        
    Returns:
        Tuple of (entities_data, ground_truth_data) where each dict maps
        sample_id to a list of entities
    """
    with open(entities_file, 'r') as f:
        entities_data = json.load(f)
    
    with open(ground_truth_file, 'r') as f:
        ground_truth_data = json.load(f)
    
    return entities_data, ground_truth_data


def extract_entities(entities_data: Dict) -> Dict[str, List[str]]:
    """
    Extract entities from the step1 output data structure.
    
    Args:
        entities_data: Loaded entity extraction results
        
    Returns:
        Dictionary mapping sample_id to list of extracted entities
    """
    result = {}
    
    # Handle case where entities_data is already a dict mapping case_id -> data
    if isinstance(entities_data, dict):
        for case_id, case_data in entities_data.items():
            # Ensure case_id is treated as a string
            case_id_str = str(case_id)
            
            # Extract entities from entities_with_contexts
            entities = []
            if isinstance(case_data, dict) and "entities_with_contexts" in case_data:
                for entity_data in case_data["entities_with_contexts"]:
                    if isinstance(entity_data, dict) and "entity" in entity_data:
                        entities.append(entity_data["entity"])
            
            # Add non-empty lists to result
            if entities:
                result[case_id_str] = entities
    
    return result


def extract_ground_truth_entities(ground_truth_data: Dict) -> Dict[str, List[str]]:
    """
    Extract ground truth entities from the ground truth data structure.
    
    Args:
        ground_truth_data: Loaded ground truth data
        
    Returns:
        Dictionary mapping sample_id to list of ground truth entities
    """
    result = {}
    
    # Handle different possible formats
    if isinstance(ground_truth_data, dict):
        # Format 1: {case_id: {"phenotypes": [...]}, ...}
        for case_id, case_data in ground_truth_data.items():
            # Ensure case_id is treated as a string
            case_id_str = str(case_id)
            entities = []
            
            # Try different possible field names as used in evaluate_step2.py
            for field in ["phenotypes", "ground_truth", "hpo_terms", "entities", "ground_truth_entities", "annotated_entities"]:
                if field in case_data:
                    field_data = case_data[field]
                    if isinstance(field_data, list):
                        for item in field_data:
                            # Check for different possible field names in nested dictionaries
                            if isinstance(item, dict):
                                # Try different field names for the phenotype/entity
                                for entity_field in ["phenotype", "phenotype_name", "name", "text", "entity"]:
                                    if entity_field in item:
                                        entities.append(item[entity_field])
                                        break
                            elif isinstance(item, str):
                                entities.append(item)
                    break
            
            # Add non-empty lists to result
            if entities:
                result[case_id_str] = entities
                
            # Print debug info for this case
            print(f"Case {case_id}: Found {len(entities)} ground truth entities")
    
    return result


def evaluate_corpus(entities_dict: Dict[str, List[str]], 
                   ground_truth_dict: Dict[str, List[str]],
                   similarity_threshold: float = 80.0) -> Dict:
    """
    Evaluate entity extraction against ground truth across the entire corpus.
    
    Args:
        entities_dict: Dictionary mapping sample_id to extracted entities
        ground_truth_dict: Dictionary mapping sample_id to ground truth entities
        similarity_threshold: Minimum similarity score to consider a potential match
        
    Returns:
        Dictionary with corpus-level and per-sample evaluation results
    """
    # Initialize result structure
    result = {
        "corpus_metrics": {},
        "per_sample_metrics": {},
        "corpus_true_positives": [],
        "corpus_false_positives": [],
        "corpus_false_negatives": []
    }
    
    # Initialize corpus-level counters for cases with ground truth
    all_entities_with_gt = []
    all_ground_truth = []
    
    # Initialize corpus-level counters for all entities (including those without ground truth)
    all_entities_total = []
    
    # Track cases with and without ground truth
    cases_with_ground_truth = []
    cases_without_ground_truth = []
    
    # Evaluate each sample
    for sample_id in sorted(set(list(entities_dict.keys()) + list(ground_truth_dict.keys()))):
        # Get entities for this sample (empty list if sample not in dict)
        entities = entities_dict.get(sample_id, [])
        ground_truth = ground_truth_dict.get(sample_id, [])
        
        # Add to all entities counter
        all_entities_total.extend(entities)
        
        # Track if this case has ground truth
        has_ground_truth = len(ground_truth) > 0
        
        # Skip empty samples
        if not entities and not ground_truth:
            continue
        
        # Evaluate this sample
        sample_result = set_based_evaluation(entities, ground_truth, similarity_threshold)
        
        # Store per-sample metrics
        result["per_sample_metrics"][sample_id] = {
            "precision": sample_result["precision"],
            "recall": sample_result["recall"],
            "f1_score": sample_result["f1_score"],
            "tp_count": sample_result["tp_count"],
            "fp_count": sample_result["fp_count"],
            "fn_count": sample_result["fn_count"],
            "weighted_tp": sample_result["weighted_tp"],
            "unique_pred_count": sample_result["unique_pred_count"],
            "unique_truth_count": sample_result["unique_truth_count"],
            "total_pred_count": sample_result["total_pred_count"],
            "total_truth_count": sample_result["total_truth_count"],
            "has_ground_truth": has_ground_truth
        }
        
        # Add detailed match information
        result["per_sample_metrics"][sample_id]["true_positives"] = sample_result["true_positives"]
        result["per_sample_metrics"][sample_id]["false_positives"] = sample_result["false_positives"]
        result["per_sample_metrics"][sample_id]["false_negatives"] = sample_result["false_negatives"]
        
        # Add false positives to corpus-level list regardless of ground truth availability
        for fp in sample_result["false_positives"]:
            fp_with_sample_id = fp.copy()
            fp_with_sample_id["sample_id"] = sample_id
            result["corpus_false_positives"].append(fp_with_sample_id)
        
        # For cases with ground truth, add to metrics for F1 calculation
        if has_ground_truth:
            cases_with_ground_truth.append(sample_id)
            # Add to corpus-level lists for F1 calculation
            all_entities_with_gt.extend(entities)
            all_ground_truth.extend(ground_truth)
            
            # Add true positives and false negatives to corpus-level
            for tp in sample_result["true_positives"]:
                tp_with_sample_id = tp.copy()
                tp_with_sample_id["sample_id"] = sample_id
                result["corpus_true_positives"].append(tp_with_sample_id)
            
            for fn in sample_result["false_negatives"]:
                fn_with_sample_id = fn.copy()
                fn_with_sample_id["sample_id"] = sample_id
                result["corpus_false_negatives"].append(fn_with_sample_id)
        else:
            cases_without_ground_truth.append(sample_id)
    
    # Calculate corpus-level metrics ONLY for cases with ground truth
    corpus_result = set_based_evaluation(all_entities_with_gt, all_ground_truth, similarity_threshold)
    
    # Store corpus-level metrics
    result["corpus_metrics"] = {
        "precision": corpus_result["precision"],
        "recall": corpus_result["recall"],
        "f1_score": corpus_result["f1_score"],
        "tp_count": corpus_result["tp_count"],
        "fp_count": corpus_result["fp_count"],
        "fn_count": corpus_result["fn_count"],
        "weighted_tp": corpus_result["weighted_tp"],
        "unique_pred_count": corpus_result["unique_pred_count"],
        "unique_truth_count": corpus_result["unique_truth_count"],
        "total_pred_count": corpus_result["total_pred_count"],
        "total_truth_count": corpus_result["total_truth_count"],
        "cases_with_ground_truth": len(cases_with_ground_truth),
        "cases_without_ground_truth": len(cases_without_ground_truth),
        "total_cases": len(cases_with_ground_truth) + len(cases_without_ground_truth),
        "total_entities_all_cases": len(all_entities_total)
    }
    
    # Track counts and lists of cases
    result["cases_with_ground_truth"] = cases_with_ground_truth
    result["cases_without_ground_truth"] = cases_without_ground_truth
    
    # Add additional info about false positives from cases without ground truth
    fps_no_ground_truth = [fp for fp in result["corpus_false_positives"] 
                          if fp["sample_id"] in cases_without_ground_truth]
    result["corpus_metrics"]["fps_from_cases_without_ground_truth"] = len(fps_no_ground_truth)
    
    # Add statistical summaries for cases WITH ground truth only
    valid_metrics = {k: v for k, v in result["per_sample_metrics"].items() 
                    if v["has_ground_truth"]}
    result["statistics"] = calculate_statistics(valid_metrics)
    
    # Add explanation of how metrics were calculated
    result["notes"] = [
        "Corpus-level precision, recall, and F1 scores only include cases with ground truth",
        f"Cases without ground truth ({len(cases_without_ground_truth)}) are tracked but excluded from F1 calculations",
        "False positives are tracked for all cases, including those without ground truth"
    ]
    
    return result


def calculate_statistics(per_sample_metrics: Dict[str, Dict]) -> Dict:
    """
    Calculate statistical summaries of per-sample metrics.
    
    Args:
        per_sample_metrics: Dictionary mapping sample_id to metric dictionaries
        
    Returns:
        Dictionary with statistical summaries
    """
    if not per_sample_metrics:
        return {}
    
    # Extract metrics into lists
    precision_values = [m["precision"] for m in per_sample_metrics.values() if "precision" in m]
    recall_values = [m["recall"] for m in per_sample_metrics.values() if "recall" in m]
    f1_values = [m["f1_score"] for m in per_sample_metrics.values() if "f1_score" in m]
    
    tp_counts = [m["tp_count"] for m in per_sample_metrics.values() if "tp_count" in m]
    fp_counts = [m["fp_count"] for m in per_sample_metrics.values() if "fp_count" in m]
    fn_counts = [m["fn_count"] for m in per_sample_metrics.values() if "fn_count" in m]
    
    # Calculate statistics
    stats = {}
    
    # Helper function for basic stats
    def calc_stats(values, name):
        if not values:
            return {}
        
        values_array = np.array(values)
        return {
            f"{name}_mean": float(np.mean(values_array)),
            f"{name}_median": float(np.median(values_array)),
            f"{name}_min": float(np.min(values_array)),
            f"{name}_max": float(np.max(values_array)),
            f"{name}_std": float(np.std(values_array)),
            f"{name}_samples": len(values_array)
        }
    
    # Calculate stats for each metric
    stats.update(calc_stats(precision_values, "precision"))
    stats.update(calc_stats(recall_values, "recall"))
    stats.update(calc_stats(f1_values, "f1"))
    stats.update(calc_stats(tp_counts, "tp"))
    stats.update(calc_stats(fp_counts, "fp"))
    stats.update(calc_stats(fn_counts, "fn"))
    
    # Count samples with precision/recall/f1 of 0 or 1
    stats["perfect_precision_count"] = sum(1 for p in precision_values if p == 1.0)
    stats["zero_precision_count"] = sum(1 for p in precision_values if p == 0.0)
    stats["perfect_recall_count"] = sum(1 for r in recall_values if r == 1.0)
    stats["zero_recall_count"] = sum(1 for r in recall_values if r == 0.0)
    stats["perfect_f1_count"] = sum(1 for f in f1_values if f == 1.0)
    stats["zero_f1_count"] = sum(1 for f in f1_values if f == 0.0)
    
    return stats


def analyze_corpus_errors(result: Dict) -> Dict:
    """
    Analyze common error patterns across the corpus.
    
    Args:
        result: Dictionary with corpus evaluation results
        
    Returns:
        Dictionary with error analysis
    """
    analysis = {
        "most_common_false_positives": [],
        "most_common_false_negatives": []
    }
    
    # Count frequency of false positives and negatives
    fp_counter = Counter()
    fn_counter = Counter()
    
    for fp in result["corpus_false_positives"]:
        fp_counter[fp["text"]] += 1
    
    for fn in result["corpus_false_negatives"]:
        fn_counter[fn["text"]] += 1
    
    # Get most common errors
    analysis["most_common_false_positives"] = [
        {"text": text, "count": count}
        for text, count in fp_counter.most_common(20)
    ]
    
    analysis["most_common_false_negatives"] = [
        {"text": text, "count": count}
        for text, count in fn_counter.most_common(20)
    ]
    
    return analysis


def print_evaluation_summary(result: Dict) -> None:
    """
    Print a summary of the evaluation results.
    
    Args:
        result: Dictionary with corpus evaluation results
    """
    print("\n=== Corpus-Level Entity Extraction Evaluation Summary ===")
    
    # Print corpus metrics
    corpus = result["corpus_metrics"]
    
    # Print case counts
    print(f"Cases with ground truth: {corpus.get('cases_with_ground_truth', 0)}")
    print(f"Cases without ground truth: {corpus.get('cases_without_ground_truth', 0)}")
    print(f"Total cases evaluated: {corpus.get('total_cases', 0)}")
    
    # Print overall metrics (only for cases with ground truth)
    print(f"\nMetrics (calculated only on cases with ground truth):")
    print(f"  Precision: {corpus['precision']:.4f}")
    print(f"  Recall: {corpus['recall']:.4f}")
    print(f"  F1 Score: {corpus['f1_score']:.4f}")
    print(f"  True Positives: {corpus['tp_count']} (weighted: {corpus['weighted_tp']:.2f})")
    print(f"  False Positives: {corpus['fp_count']}")
    print(f"  False Negatives: {corpus['fn_count']}")
    
    # Print additional information about false positives
    if 'fps_from_cases_without_ground_truth' in corpus:
        fps_no_gt = corpus['fps_from_cases_without_ground_truth']
        print(f"\nAdditional metrics:")
        print(f"  False positives from cases without ground truth: {fps_no_gt}")
        print(f"  Total false positives (all cases): {corpus['fp_count'] + fps_no_gt}")
    
    # Print counts of entities and ground truth
    print(f"\nCounts:")
    print(f"  Total Entities (all cases): {corpus.get('total_entities_all_cases', 0)}")
    print(f"  Entities in cases with ground truth: {corpus.get('total_pred_count', 0)}")
    print(f"  Unique entities in cases with ground truth: {corpus.get('unique_pred_count', 0)}")
    print(f"  Total Ground Truth entities: {corpus.get('total_truth_count', 0)}")
    print(f"  Unique Ground Truth entities: {corpus.get('unique_truth_count', 0)}")
    
    # Print statistics
    stats = result["statistics"]
    if stats:
        print("\nPer-Sample Statistics (cases with ground truth only):")
        print(f"  Samples with ground truth: {stats.get('precision_samples', 0)}")
        print(f"  F1 Score: mean={stats.get('f1_mean', 0):.4f}, median={stats.get('f1_median', 0):.4f}, std={stats.get('f1_std', 0):.4f}")
        print(f"  Precision: mean={stats.get('precision_mean', 0):.4f}, median={stats.get('precision_median', 0):.4f}")
        print(f"  Recall: mean={stats.get('recall_mean', 0):.4f}, median={stats.get('recall_median', 0):.4f}")
    else:
        print("\nNo per-sample statistics available (no samples with both entities and ground truth)")
    
    # Print error analysis
    if "error_analysis" in result:
        analysis = result["error_analysis"]
        
        if analysis["most_common_false_positives"]:
            print("\nMost Common False Positives (all cases):")
            for i, item in enumerate(analysis["most_common_false_positives"][:10]):
                print(f"  {i+1}. '{item['text']}' ({item['count']} occurrences)")
        
        if analysis["most_common_false_negatives"]:
            print("\nMost Common False Negatives (cases with ground truth):")
            for i, item in enumerate(analysis["most_common_false_negatives"][:10]):
                print(f"  {i+1}. '{item['text']}' ({item['count']} occurrences)")
                
    # Print notes if available
    if "notes" in result:
        print("\nNotes:")
        for note in result["notes"]:
            print(f"  - {note}")
        
    # Provide guidance if no matches
    if corpus.get('tp_count', 0) == 0 and (corpus.get('fp_count', 0) > 0 or corpus.get('fn_count', 0) > 0):
        print("\nWARNING: No true positive matches found between extracted entities and ground truth.")
        print("Possible reasons:")
        print("  1. Ground truth format might not match expected structure")
        print("  2. Similarity threshold might be too high (current: {:.1f})".format(result.get("similarity_threshold", 80.0)))
        print("  3. Extracted entities and ground truth might use different terminology")
        print("Try:")
        print("  - Checking the ground truth data format")
        print("  - Lowering the similarity threshold (--similarity-threshold)")
        print("  - Adding more detailed debug output to see extracted entities")


def main():
    parser = argparse.ArgumentParser(description="Evaluate entity extraction results against ground truth")
    
    parser.add_argument("--entities", required=True,
                       help="Path to JSON file with entity extraction results from step1")
    parser.add_argument("--ground-truth", required=True,
                       help="Path to JSON file with ground truth entities")
    parser.add_argument("--output", required=True,
                       help="Path to save evaluation results JSON")
    parser.add_argument("--similarity-threshold", type=float, default=80.0,
                       help="Similarity threshold for fuzzy matching (0-100, default: 80.0)")
    parser.add_argument("--summary-only", action="store_true",
                       help="Only print summary, don't save detailed JSON")
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading entity extraction results from {args.entities}")
    print(f"Loading ground truth from {args.ground_truth}")
    entities_data, ground_truth_data = load_data(args.entities, args.ground_truth)
    
    # Extract entities from each input file structure
    entities_dict = extract_entities(entities_data)
    ground_truth_dict = extract_ground_truth_entities(ground_truth_data)
    
    print(f"Found entities for {len(entities_dict)} cases")
    print(f"Found ground truth for {len(ground_truth_dict)} cases")
    
    # Evaluate corpus
    print(f"Evaluating with similarity threshold: {args.similarity_threshold}")
    result = evaluate_corpus(entities_dict, ground_truth_dict, args.similarity_threshold)
    
    # Add error analysis
    result["error_analysis"] = analyze_corpus_errors(result)
    
    # Store similarity threshold in results
    result["similarity_threshold"] = args.similarity_threshold
    
    # Print summary
    print_evaluation_summary(result)
    
    # Save results
    if not args.summary_only:
        print(f"Saving evaluation results to {args.output}")
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Results saved successfully")


if __name__ == "__main__":
    main()