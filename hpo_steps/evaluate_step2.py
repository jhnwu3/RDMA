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
        predictions: List of predicted phenotypes
        ground_truth: List of ground truth phenotypes
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
        predictions: List of predicted phenotypes (can contain duplicates)
        ground_truth: List of ground truth phenotypes (can contain duplicates)
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

def load_data(predictions_file: str, ground_truth_file: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Load prediction and ground truth data from files.
    
    Args:
        predictions_file: Path to predictions JSON file
        ground_truth_file: Path to ground truth JSON file
        
    Returns:
        Tuple of (predictions_data, ground_truth_data) where each is the raw loaded JSON
    """
    with open(predictions_file, 'r') as f:
        predictions = json.load(f)
    
    with open(ground_truth_file, 'r') as f:
        ground_truth = json.load(f)
    
    return predictions, ground_truth


def extract_predictions(predictions_data: Dict) -> Dict[str, List[Dict]]:
    """
    Extract prediction phenotypes from the predictions data structure.
    
    Args:
        predictions_data: Loaded predictions data
        
    Returns:
        Dictionary mapping sample_id to list of predicted phenotypes with their status
    """
    result = {}
    
    # Handle different possible formats
    if isinstance(predictions_data, dict):
        # Format: {case_id: {"verified_phenotypes": [...]}, ...}
        for case_id, case_data in predictions_data.items():
            phenotypes = []
            # Different possible structures for verified_phenotypes
            if "verified_phenotypes" in case_data:
                verified_list = case_data["verified_phenotypes"]
                if isinstance(verified_list, list):
                    for item in verified_list:
                        if isinstance(item, dict) and "phenotype" in item:
                            phenotype_info = {
                                "phenotype": item["phenotype"],
                                "status": item.get("status", "unknown")
                            }
                            phenotypes.append(phenotype_info)
                        elif isinstance(item, str):
                            phenotype_info = {
                                "phenotype": item,
                                "status": "unknown"
                            }
                            phenotypes.append(phenotype_info)
            
            # Add non-empty lists to result
            if phenotypes:
                result[str(case_id)] = phenotypes
    
    # Return dictionary mapping case_id to phenotype list
    return result


def extract_ground_truth(ground_truth_data: Dict) -> Dict[str, List[str]]:
    """
    Extract ground truth phenotypes from the ground truth data structure.
    
    Args:
        ground_truth_data: Loaded ground truth data
        
    Returns:
        Dictionary mapping sample_id to list of ground truth phenotypes
    """
    result = {}
    
    # Handle different possible formats
    if isinstance(ground_truth_data, dict):
        # Format 1: {case_id: {"phenotypes": [...]}, ...}
        for case_id, case_data in ground_truth_data.items():
            phenotypes = []
            
            # Try different possible field names
            for field in ["phenotypes", "ground_truth", "hpo_terms"]:
                if field in case_data:
                    field_data = case_data[field]
                    if isinstance(field_data, list):
                        for item in field_data:
                            # Check for different possible field names in nested dictionaries
                            if isinstance(item, dict):
                                # Try different field names for the phenotype
                                for phenotype_field in ["phenotype", "phenotype_name", "name"]:
                                    if phenotype_field in item:
                                        phenotypes.append(item[phenotype_field])
                                        break
                            elif isinstance(item, str):
                                phenotypes.append(item)
                    break
            
            # Add non-empty lists to result
            if phenotypes:
                result[str(case_id)] = phenotypes
                
            # Print debug info for this case
            print(f"Case {case_id}: Found {len(phenotypes)} phenotypes")
    
    # Return dictionary mapping case_id to phenotype list
    return result


def evaluate_corpus(predictions_dict: Dict[str, List[Dict]], 
                    ground_truth_dict: Dict[str, List[str]],
                    phenotype_types: str = "all",
                    similarity_threshold: float = 80.0) -> Dict:
    """
    Evaluate predictions against ground truth across the entire corpus using three approaches:
    1. Micro-averaging (corpus-level): All phenotypes from all cases are pooled together
    2. Macro-averaging (case-level): Metrics are calculated per case, then averaged
    3. Count-based: All TP, FP, FN counts are summed across cases, then metrics are calculated
    
    Args:
        predictions_dict: Dictionary mapping sample_id to predicted phenotypes
        ground_truth_dict: Dictionary mapping sample_id to ground truth phenotypes
        phenotype_types: Filter for phenotype types (all, direct, implied)
        similarity_threshold: Minimum similarity score to consider a potential match
        
    Returns:
        Dictionary with corpus-level and per-sample evaluation results
    """
    # Initialize result structure
    result = {
        "corpus_metrics": {},
        "micro_averaging_metrics": {},
        "macro_averaging_metrics": {},
        "count_based_metrics": {},
        "per_sample_metrics": {},
        "corpus_true_positives": [],
        "corpus_false_positives": [],
        "corpus_false_negatives": [],
        "direct_phenotype_metrics": {},
        "implied_phenotype_metrics": {}
    }
    
    # Initialize corpus-level counters for cases with ground truth
    all_predictions_with_gt = []
    all_ground_truth = []
    
    # Initialize corpus-level counters for all predictions (including those without ground truth)
    all_predictions_total = []
    
    # Initialize type-specific lists for cases with ground truth
    direct_phenotypes_with_gt = []
    implied_phenotypes_with_gt = []
    
    # Initialize type-specific counters for tracking
    direct_phenotypes_count = 0
    implied_phenotypes_count = 0
    
    # Track cases with and without ground truth
    cases_with_ground_truth = []
    cases_without_ground_truth = []
    
    # Initialize counters for count-based approach
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_weighted_tp = 0.0
    
    # Initialize lists for macro-averaging
    case_precision_values = []
    case_recall_values = []
    case_f1_values = []
    
    # Evaluate each sample (only those in predictions_dict)
    for sample_id in sorted(predictions_dict.keys()):
        # Get phenotypes for this sample
        predictions_info = predictions_dict.get(sample_id, [])
        ground_truth = ground_truth_dict.get(sample_id, [])
        
        # Filter predictions based on phenotype type
        if phenotype_types != "all":
            predictions_info = [p for p in predictions_info if p.get("status", "unknown") == (phenotype_types + "_phenotype")]
            
        # Extract just the phenotype strings for evaluation
        predictions = [p["phenotype"] for p in predictions_info]
        
        # Add to all predictions counter
        all_predictions_total.extend(predictions)
        
        # Track if this case has ground truth
        has_ground_truth = len(ground_truth) > 0
        
        # Skip empty samples
        if not predictions and not ground_truth:
            continue
        
        # Evaluate this sample
        sample_result = set_based_evaluation(predictions, ground_truth, similarity_threshold)
        
        # Create phenotype type tracking dictionaries
        direct_phenotypes = []
        implied_phenotypes = []
        
        # Track phenotypes by type
        for pred_info in predictions_info:
            phenotype = pred_info["phenotype"]
            status = pred_info.get("status", "unknown")
            
            if status == "direct_phenotype":
                direct_phenotypes.append(phenotype)
                direct_phenotypes_count += 1
                if has_ground_truth:
                    direct_phenotypes_with_gt.append(phenotype)
            elif status == "implied_phenotype":
                implied_phenotypes.append(phenotype)
                implied_phenotypes_count += 1
                if has_ground_truth:
                    implied_phenotypes_with_gt.append(phenotype)
        
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
            "has_ground_truth": has_ground_truth,
            "direct_phenotypes": direct_phenotypes,
            "implied_phenotypes": implied_phenotypes
        }
        
        # Add detailed match information
        result["per_sample_metrics"][sample_id]["true_positives"] = sample_result["true_positives"]
        result["per_sample_metrics"][sample_id]["false_positives"] = sample_result["false_positives"]
        result["per_sample_metrics"][sample_id]["false_negatives"] = sample_result["false_negatives"]
        
        # Add false positives to corpus-level list regardless of ground truth availability
        for fp in sample_result["false_positives"]:
            fp_with_sample_id = fp.copy()
            fp_with_sample_id["sample_id"] = sample_id
            # Add type information if possible
            if fp["text"] in direct_phenotypes:
                fp_with_sample_id["type"] = "direct_phenotype"
            elif fp["text"] in implied_phenotypes:
                fp_with_sample_id["type"] = "implied_phenotype"
            else:
                fp_with_sample_id["type"] = "unknown"
            result["corpus_false_positives"].append(fp_with_sample_id)
        
        # For cases with ground truth, add to metrics for F1 calculation
        if has_ground_truth:
            cases_with_ground_truth.append(sample_id)
            # Add to corpus-level lists for F1 calculation
            all_predictions_with_gt.extend(predictions)
            all_ground_truth.extend(ground_truth)
            
            # Approach 2: Count-based - accumulate counts
            total_tp += sample_result["tp_count"]
            total_fp += sample_result["fp_count"]
            total_fn += sample_result["fn_count"]
            total_weighted_tp += sample_result["weighted_tp"]
            
            # Approach 3: Macro-averaging - collect metrics for averaging
            case_precision_values.append(sample_result["precision"])
            case_recall_values.append(sample_result["recall"])
            case_f1_values.append(sample_result["f1_score"])
            
            # Add true positives and false negatives to corpus-level
            for tp in sample_result["true_positives"]:
                tp_with_sample_id = tp.copy()
                tp_with_sample_id["sample_id"] = sample_id
                # Add type information if possible
                if tp["prediction"] in direct_phenotypes:
                    tp_with_sample_id["type"] = "direct_phenotype"
                elif tp["prediction"] in implied_phenotypes:
                    tp_with_sample_id["type"] = "implied_phenotype"
                else:
                    tp_with_sample_id["type"] = "unknown"
                result["corpus_true_positives"].append(tp_with_sample_id)
            
            for fn in sample_result["false_negatives"]:
                fn_with_sample_id = fn.copy()
                fn_with_sample_id["sample_id"] = sample_id
                result["corpus_false_negatives"].append(fn_with_sample_id)
        else:
            cases_without_ground_truth.append(sample_id)
    
    # Approach 1: Micro-averaging metrics (corpus-level pooling)
    micro_result = set_based_evaluation(all_predictions_with_gt, all_ground_truth, similarity_threshold)
    
    # Approach 2: Count-based metrics
    count_based_metrics = {}
    if total_tp + total_fp > 0:
        count_based_metrics["precision"] = total_weighted_tp / (total_tp + total_fp)
    else:
        count_based_metrics["precision"] = 1.0 if total_tp > 0 else 0.0
        
    if total_tp + total_fn > 0:
        count_based_metrics["recall"] = total_weighted_tp / (total_tp + total_fn)
    else:
        count_based_metrics["recall"] = 1.0 if total_tp > 0 else 0.0
        
    if count_based_metrics["precision"] + count_based_metrics["recall"] > 0:
        count_based_metrics["f1_score"] = 2 * (count_based_metrics["precision"] * count_based_metrics["recall"]) / (count_based_metrics["precision"] + count_based_metrics["recall"])
    else:
        count_based_metrics["f1_score"] = 0.0
        
    count_based_metrics["tp_count"] = total_tp
    count_based_metrics["fp_count"] = total_fp
    count_based_metrics["fn_count"] = total_fn
    count_based_metrics["weighted_tp"] = total_weighted_tp
    
    # Approach 3: Macro-averaging metrics
    macro_metrics = {}
    if case_precision_values:
        macro_metrics["precision"] = np.mean(case_precision_values)
        macro_metrics["recall"] = np.mean(case_recall_values)
        macro_metrics["f1_score"] = np.mean(case_f1_values)
        macro_metrics["precision_std"] = np.std(case_precision_values)
        macro_metrics["recall_std"] = np.std(case_recall_values)
        macro_metrics["f1_score_std"] = np.std(case_f1_values)
        macro_metrics["case_count"] = len(case_precision_values)
    else:
        macro_metrics["precision"] = 0.0
        macro_metrics["recall"] = 0.0
        macro_metrics["f1_score"] = 0.0
        macro_metrics["precision_std"] = 0.0
        macro_metrics["recall_std"] = 0.0
        macro_metrics["f1_score_std"] = 0.0
        macro_metrics["case_count"] = 0
    
    # Store all three metric approaches
    result["micro_averaging_metrics"] = {
        "precision": micro_result["precision"],
        "recall": micro_result["recall"],
        "f1_score": micro_result["f1_score"],
        "tp_count": micro_result["tp_count"],
        "fp_count": micro_result["fp_count"],
        "fn_count": micro_result["fn_count"],
        "weighted_tp": micro_result["weighted_tp"],
        "description": "Micro-averaging: All phenotypes pooled together across cases"
    }
    
    result["macro_averaging_metrics"] = macro_metrics
    result["macro_averaging_metrics"]["description"] = "Macro-averaging: Metrics calculated per case, then averaged"
    
    result["count_based_metrics"] = count_based_metrics
    result["count_based_metrics"]["description"] = "Count-based: TP, FP, FN counts summed across cases, then metrics calculated"
    
    # Store corpus-level metrics (backward compatibility, same as micro-averaging)
    result["corpus_metrics"] = {
        "precision": micro_result["precision"],
        "recall": micro_result["recall"],
        "f1_score": micro_result["f1_score"],
        "tp_count": micro_result["tp_count"],
        "fp_count": micro_result["fp_count"],
        "fn_count": micro_result["fn_count"],
        "weighted_tp": micro_result["weighted_tp"],
        "unique_pred_count": micro_result["unique_pred_count"],
        "unique_truth_count": micro_result["unique_truth_count"],
        "total_pred_count": micro_result["total_pred_count"],
        "total_truth_count": micro_result["total_truth_count"],
        "cases_with_ground_truth": len(cases_with_ground_truth),
        "cases_without_ground_truth": len(cases_without_ground_truth),
        "total_cases": len(cases_with_ground_truth) + len(cases_without_ground_truth),
        "total_predictions_all_cases": len(all_predictions_total),
        "direct_phenotypes_count": direct_phenotypes_count,
        "implied_phenotypes_count": implied_phenotypes_count
    }
    
    # Calculate separate metrics for direct phenotypes if available
    if direct_phenotypes_with_gt:
        direct_result = set_based_evaluation(direct_phenotypes_with_gt, all_ground_truth, similarity_threshold)
        result["direct_phenotype_metrics"] = {
            "precision": direct_result["precision"],
            "recall": direct_result["recall"],
            "f1_score": direct_result["f1_score"],
            "tp_count": direct_result["tp_count"],
            "fp_count": direct_result["fp_count"],
            "fn_count": direct_result["fn_count"],
            "count": len(direct_phenotypes_with_gt)
        }
    
    # Calculate separate metrics for implied phenotypes if available
    if implied_phenotypes_with_gt:
        implied_result = set_based_evaluation(implied_phenotypes_with_gt, all_ground_truth, similarity_threshold)
        result["implied_phenotype_metrics"] = {
            "precision": implied_result["precision"],
            "recall": implied_result["recall"],
            "f1_score": implied_result["f1_score"],
            "tp_count": implied_result["tp_count"],
            "fp_count": implied_result["fp_count"],
            "fn_count": implied_result["fn_count"],
            "count": len(implied_phenotypes_with_gt)
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
        "Three approaches to metric calculation are provided:",
        " 1. Micro-averaging: All phenotypes pooled together across cases",
        " 2. Macro-averaging: Metrics calculated per case, then averaged",
        " 3. Count-based: TP, FP, FN counts summed across cases, then metrics calculated",
        f"Only cases in predictions file were evaluated ({len(predictions_dict)} cases)",
        f"Cases without ground truth ({len(cases_without_ground_truth)}) are tracked but excluded from F1 calculations",
        "False positives are tracked for all cases, including those without ground truth",
        "Separate metrics are provided for direct_phenotype and implied_phenotype",
        f"Phenotype filter applied: {phenotype_types}"
    ]
    
    # Add similarity threshold to result
    result["similarity_threshold"] = similarity_threshold
    
    # Add phenotype type info to result
    result["phenotype_types"] = phenotype_types
    
    # Add total cases evaluated
    result["total_cases_evaluated"] = len(predictions_dict)
    
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
    
    # Count direct and implied phenotypes
    direct_counts = [len(m.get("direct_phenotypes", [])) for m in per_sample_metrics.values()]
    implied_counts = [len(m.get("implied_phenotypes", [])) for m in per_sample_metrics.values()]
    
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
    stats.update(calc_stats(direct_counts, "direct_phenotype"))
    stats.update(calc_stats(implied_counts, "implied_phenotype"))
    
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
        "most_common_false_negatives": [],
        "direct_phenotype_errors": {
            "most_common_false_positives": [],
            "most_common_false_negatives": []
        },
        "implied_phenotype_errors": {
            "most_common_false_positives": [],
            "most_common_false_negatives": []
        }
    }
    
    # Count frequency of false positives and negatives
    fp_counter = Counter()
    fn_counter = Counter()
    
    # Separate counters for direct and implied phenotypes
    direct_fp_counter = Counter()
    implied_fp_counter = Counter()
    
    for fp in result["corpus_false_positives"]:
        fp_counter[fp["text"]] += 1
        
        # Check phenotype type if available
        if "type" in fp:
            if fp["type"] == "direct_phenotype":
                direct_fp_counter[fp["text"]] += 1
            elif fp["type"] == "implied_phenotype":
                implied_fp_counter[fp["text"]] += 1
    
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
    
    # Get type-specific errors
    analysis["direct_phenotype_errors"]["most_common_false_positives"] = [
        {"text": text, "count": count}
        for text, count in direct_fp_counter.most_common(10)
    ]
    
    analysis["implied_phenotype_errors"]["most_common_false_positives"] = [
        {"text": text, "count": count}
        for text, count in implied_fp_counter.most_common(10)
    ]
    
    return analysis


def print_evaluation_summary(result: Dict) -> None:
    """
    Print a summary of the evaluation results.
    
    Args:
        result: Dictionary with corpus evaluation results
    """
    print("\n=== Corpus-Level Evaluation Summary ===")
    print(f"Evaluating {result.get('total_cases_evaluated', 0)} cases from predictions file")
    
    # Print case counts
    corpus = result["corpus_metrics"]
    print(f"Cases with ground truth: {corpus.get('cases_with_ground_truth', 0)}")
    print(f"Cases without ground truth: {corpus.get('cases_without_ground_truth', 0)}")
    print(f"Total cases evaluated: {corpus.get('total_cases', 0)}")
    
    # Print metrics using all three approaches
    print("\n=== Metrics Using Three Different Approaches ===")
    
    # 1. Print Micro-averaging metrics
    micro = result["micro_averaging_metrics"]
    print("\n1. MICRO-AVERAGING (pooling all phenotypes across cases):")
    print(f"  Precision: {micro['precision']:.4f}")
    print(f"  Recall: {micro['recall']:.4f}")
    print(f"  F1 Score: {micro['f1_score']:.4f}")
    print(f"  True Positives: {micro['tp_count']} (weighted: {micro['weighted_tp']:.2f})")
    print(f"  False Positives: {micro['fp_count']}")
    print(f"  False Negatives: {micro['fn_count']}")
    
    # 2. Print Macro-averaging metrics
    macro = result["macro_averaging_metrics"]
    print("\n2. MACRO-AVERAGING (averaging metrics across cases):")
    print(f"  Precision: {macro['precision']:.4f} (±{macro['precision_std']:.4f})")
    print(f"  Recall: {macro['recall']:.4f} (±{macro['recall_std']:.4f})")
    print(f"  F1 Score: {macro['f1_score']:.4f} (±{macro['f1_score_std']:.4f})")
    print(f"  Cases included: {macro['case_count']}")
    
    # 3. Print Count-based metrics
    count = result["count_based_metrics"]
    print("\n3. COUNT-BASED (summing TP, FP, FN across cases):")
    print(f"  Precision: {count['precision']:.4f}")
    print(f"  Recall: {count['recall']:.4f}")
    print(f"  F1 Score: {count['f1_score']:.4f}")
    print(f"  True Positives: {count['tp_count']} (weighted: {count['weighted_tp']:.2f})")
    print(f"  False Positives: {count['fp_count']}")
    print(f"  False Negatives: {count['fn_count']}")
    
    # Print phenotype type counts
    print(f"\nPhenotype Type Breakdown:")
    print(f"  Direct Phenotypes: {corpus.get('direct_phenotypes_count', 0)}")
    print(f"  Implied Phenotypes: {corpus.get('implied_phenotypes_count', 0)}")
    
    # Print direct phenotype metrics if available
    if "direct_phenotype_metrics" in result and result["direct_phenotype_metrics"]:
        direct = result["direct_phenotype_metrics"]
        print(f"\nDirect Phenotype Metrics:")
        print(f"  Count: {direct.get('count', 0)}")
        print(f"  Precision: {direct['precision']:.4f}")
        print(f"  Recall: {direct['recall']:.4f}")
        print(f"  F1 Score: {direct['f1_score']:.4f}")
        print(f"  True Positives: {direct['tp_count']}")
        print(f"  False Positives: {direct['fp_count']}")
        print(f"  False Negatives: {direct['fn_count']}")
    
    # Print implied phenotype metrics if available
    if "implied_phenotype_metrics" in result and result["implied_phenotype_metrics"]:
        implied = result["implied_phenotype_metrics"]
        print(f"\nImplied Phenotype Metrics:")
        print(f"  Count: {implied.get('count', 0)}")
        print(f"  Precision: {implied['precision']:.4f}")
        print(f"  Recall: {implied['recall']:.4f}")
        print(f"  F1 Score: {implied['f1_score']:.4f}")
        print(f"  True Positives: {implied['tp_count']}")
        print(f"  False Positives: {implied['fp_count']}")
        print(f"  False Negatives: {implied['fn_count']}")
    
    # Print additional information about false positives
    if 'fps_from_cases_without_ground_truth' in corpus:
        fps_no_gt = corpus['fps_from_cases_without_ground_truth']
        print(f"\nAdditional metrics:")
        print(f"  False positives from cases without ground truth: {fps_no_gt}")
        print(f"  Total false positives (all cases): {corpus['fp_count'] + fps_no_gt}")
    
    # Print counts of predictions and ground truth
    print(f"\nCounts:")
    print(f"  Total Predictions (all cases): {corpus.get('total_predictions_all_cases', 0)}")
    print(f"  Predictions in cases with ground truth: {corpus.get('total_pred_count', 0)}")
    print(f"  Unique predictions in cases with ground truth: {corpus.get('unique_pred_count', 0)}")
    print(f"  Total Ground Truth items: {corpus.get('total_truth_count', 0)}")
    print(f"  Unique Ground Truth items: {corpus.get('unique_truth_count', 0)}")
    
    # Print statistics
    stats = result["statistics"]
    if stats:
        print("\nPer-Sample Statistics (cases with ground truth only):")
        print(f"  Samples with ground truth: {stats.get('precision_samples', 0)}")
        print(f"  F1 Score: mean={stats.get('f1_mean', 0):.4f}, median={stats.get('f1_median', 0):.4f}, std={stats.get('f1_std', 0):.4f}")
        print(f"  Precision: mean={stats.get('precision_mean', 0):.4f}, median={stats.get('precision_median', 0):.4f}")
        print(f"  Recall: mean={stats.get('recall_mean', 0):.4f}, median={stats.get('recall_median', 0):.4f}")
        
        # Print phenotype type statistics if available
        if 'direct_phenotype_mean' in stats:
            print(f"  Direct Phenotypes per sample: mean={stats.get('direct_phenotype_mean', 0):.2f}, median={stats.get('direct_phenotype_median', 0):.2f}")
        if 'implied_phenotype_mean' in stats:
            print(f"  Implied Phenotypes per sample: mean={stats.get('implied_phenotype_mean', 0):.2f}, median={stats.get('implied_phenotype_median', 0):.2f}")
    else:
        print("\nNo per-sample statistics available (no samples with both predictions and ground truth)")
    
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
        
        # Print direct phenotype errors
        if analysis["direct_phenotype_errors"]["most_common_false_positives"]:
            print("\nMost Common Direct Phenotype False Positives:")
            for i, item in enumerate(analysis["direct_phenotype_errors"]["most_common_false_positives"][:5]):
                print(f"  {i+1}. '{item['text']}' ({item['count']} occurrences)")
        
        # Print implied phenotype errors
        if analysis["implied_phenotype_errors"]["most_common_false_positives"]:
            print("\nMost Common Implied Phenotype False Positives:")
            for i, item in enumerate(analysis["implied_phenotype_errors"]["most_common_false_positives"][:5]):
                print(f"  {i+1}. '{item['text']}' ({item['count']} occurrences)")
                
    # Print notes if available
    if "notes" in result:
        print("\nNotes:")
        for note in result["notes"]:
            print(f"  - {note}")
        
    # Provide guidance if no matches
    if corpus.get('tp_count', 0) == 0 and (corpus.get('fp_count', 0) > 0 or corpus.get('fn_count', 0) > 0):
        print("\nWARNING: No true positive matches found between predictions and ground truth.")
        print("Possible reasons:")
        print("  1. Ground truth format might not match expected structure")
        print("  2. Similarity threshold might be too high (current: {:.1f})".format(result.get("similarity_threshold", 80.0)))
        print("  3. Predictions and ground truth might use different terminology")
        print("Try:")
        print("  - Checking the ground truth data format")
        print("  - Lowering the similarity threshold (--similarity-threshold)")
        print("  - Adding more detailed debug output to see extracted phenotypes")


def main():
    parser = argparse.ArgumentParser(description="Evaluate phenotype predictions across a corpus of clinical notes")
    
    parser.add_argument("--predictions", required=True,
                       help="Path to JSON file with predictions")
    parser.add_argument("--ground-truth", required=True,
                       help="Path to JSON file with ground truth")
    parser.add_argument("--output", required=True,
                       help="Path to save evaluation results JSON")
    parser.add_argument("--phenotype-types", choices=["all", "direct", "implied"], default="all",
                       help="Filter phenotypes by type: 'direct' for only direct phenotypes, 'implied' for only implied phenotypes, 'all' for both (default: all)")
    parser.add_argument("--similarity-threshold", type=float, default=80.0,
                       help="Similarity threshold for fuzzy matching (0-100, default: 80.0)")
    parser.add_argument("--summary-only", action="store_true",
                       help="Only print summary, don't save detailed JSON")
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading predictions from {args.predictions}")
    print(f"Loading ground truth from {args.ground_truth}")
    predictions_data, ground_truth_data = load_data(args.predictions, args.ground_truth)
    
    # Extract phenotypes
    predictions_dict = extract_predictions(predictions_data)
    ground_truth_dict = extract_ground_truth(ground_truth_data)
    
    print(f"Found predictions for {len(predictions_dict)} samples")
    print(f"Found ground truth for {len(ground_truth_dict)} samples")
    
    # Evaluate corpus
    print(f"Evaluating with similarity threshold: {args.similarity_threshold}")
    print(f"Phenotype filter: {args.phenotype_types}")
    result = evaluate_corpus(predictions_dict, ground_truth_dict, args.phenotype_types, args.similarity_threshold)
    
    # Add error analysis
    result["error_analysis"] = analyze_corpus_errors(result)
    
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