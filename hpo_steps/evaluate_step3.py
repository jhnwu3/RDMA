#!/usr/bin/env python3
import argparse
import json
import os
import pandas as pd
import numpy as np
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


def code_based_evaluation(predictions: List[str], ground_truth: List[str]) -> Dict:
    """
    Evaluates predictions against ground truth using exact matching of HPO codes.
    
    Args:
        predictions: List of predicted HPO codes
        ground_truth: List of ground truth HPO codes
        
    Returns:
        Dictionary with precision, recall, F1 scores and match information
    """
    # Create sets for exact matching
    unique_predictions = set(predictions)
    unique_ground_truth = set(ground_truth)
    
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
        result["false_positives"] = [{"code": p, "count": pred_counter[p]} for p in unique_predictions]
        result["false_negatives"] = [{"code": t, "count": truth_counter[t]} for t in unique_ground_truth]
        result["fp_count"] = len(unique_predictions)
        result["fn_count"] = len(unique_ground_truth)
        return result
    
    # Find matches (true positives)
    true_positives = unique_predictions & unique_ground_truth
    false_positives = unique_predictions - unique_ground_truth
    false_negatives = unique_ground_truth - unique_predictions
    
    # Record matches and errors
    result["true_positives"] = [{"code": code} for code in true_positives]
    result["false_positives"] = [{"code": code, "count": pred_counter[code]} for code in false_positives]
    result["false_negatives"] = [{"code": code, "count": truth_counter[code]} for code in false_negatives]
    
    # Calculate metrics based on unique items (set-based)
    result["tp_count"] = len(true_positives)
    result["fp_count"] = len(false_positives)
    result["fn_count"] = len(false_negatives)
    
    # Calculate precision and recall
    if unique_predictions:
        result["precision"] = result["tp_count"] / len(unique_predictions)
    if unique_ground_truth:
        result["recall"] = result["tp_count"] / len(unique_ground_truth)
    
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
    
    # Handle nested structure with metadata and results keys
    if isinstance(predictions, dict) and "results" in predictions:
        print("Detected nested structure with 'results' and 'metadata' keys")
        predictions_data = predictions.get("results", {})
        # Keep the metadata for reference, but work with the results for evaluation
    else:
        predictions_data = predictions
    
    return predictions_data, ground_truth


def extract_predictions(predictions_data: Dict) -> Dict[str, List[str]]:
    """
    Extract HPO codes from the predictions data structure.
    
    Args:
        predictions_data: Loaded predictions data
        
    Returns:
        Dictionary mapping sample_id to list of predicted HPO codes
    """
    result = {}
    
    # Handle different possible formats
    if isinstance(predictions_data, dict):
        # Format: {case_id: {"verified_phenotypes" or "matched_phenotypes": [...]}, ...}
        for case_id, case_data in predictions_data.items():
            hpo_codes = []
            
            # Check for different possible field names for phenotype lists
            phenotype_fields = ["matched_phenotypes", "verified_phenotypes"]
            for field in phenotype_fields:
                if field in case_data:
                    phenotype_list = case_data[field]
                    if isinstance(phenotype_list, list):
                        for item in phenotype_list:
                            if not isinstance(item, dict):
                                continue
                                
                            # Check for different possible field names for HPO codes
                            # (both uppercase and lowercase variants)
                            for code_field in ["HPO_Term", "hpo_term", "hpo_id", "hp_id"]:
                                if code_field in item:
                                    code = item[code_field]
                                    # Ensure code starts with HP:
                                    if code and isinstance(code, str) and code.startswith("HP:"):
                                        hpo_codes.append(code)
                                        break
            
            # Add non-empty lists to result
            if hpo_codes:
                result[str(case_id)] = hpo_codes
    
    # Return dictionary mapping case_id to HPO code list
    return result


def extract_ground_truth(ground_truth_data: Dict) -> Dict[str, List[str]]:
    """
    Extract HPO codes from the ground truth data structure.
    
    Args:
        ground_truth_data: Loaded ground truth data
        
    Returns:
        Dictionary mapping sample_id to list of ground truth HPO codes
    """
    result = {}
    
    # Handle different possible formats
    if isinstance(ground_truth_data, dict):
        # Format: {case_id: {"phenotypes": [...]}, ...}
        for case_id, case_data in ground_truth_data.items():
            hpo_codes = []
            
            # Try different possible field names
            for field in ["phenotypes", "ground_truth", "hpo_terms"]:
                if field in case_data:
                    field_data = case_data[field]
                    if isinstance(field_data, list):
                        for item in field_data:
                            # Check for different possible field names in nested dictionaries
                            if isinstance(item, dict):
                                # Try different field names for the HPO code
                                for code_field in ["hpo_id", "HPO_Term", "hpo_code", "hp_id"]:
                                    if code_field in item:
                                        code = item[code_field]
                                        # Ensure code starts with HP:
                                        if code and isinstance(code, str) and code.startswith("HP:"):
                                            hpo_codes.append(code)
                                        break
                            elif isinstance(item, str) and item.startswith("HP:"):
                                # Direct HPO code string
                                hpo_codes.append(item)
                    break
            
            # Add non-empty lists to result
            if hpo_codes:
                result[str(case_id)] = hpo_codes
                
            # Print debug info for this case
            print(f"Case {case_id}: Found {len(hpo_codes)} HPO codes")
    
    # Return dictionary mapping case_id to HPO code list
    return result


def evaluate_corpus(predictions_dict: Dict[str, List[str]], 
                    ground_truth_dict: Dict[str, List[str]]) -> Dict:
    """
    Evaluate predictions against ground truth across the entire corpus using three approaches:
    1. Micro-averaging (corpus-level): All HPO codes from all cases are pooled together
    2. Macro-averaging (case-level): Metrics are calculated per case, then averaged
    3. Count-based: All TP, FP, FN counts are summed across cases, then metrics are calculated
    
    Args:
        predictions_dict: Dictionary mapping sample_id to predicted HPO codes
        ground_truth_dict: Dictionary mapping sample_id to ground truth HPO codes
        
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
        "corpus_false_negatives": []
    }
    
    # Initialize corpus-level counters for cases with ground truth
    all_predictions_with_gt = []
    all_ground_truth = []
    
    # Initialize corpus-level counters for all predictions (including those without ground truth)
    all_predictions_total = []
    
    # Track cases with and without ground truth
    cases_with_ground_truth = []
    cases_without_ground_truth = []
    
    # Initialize counters for count-based approach
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    # Initialize lists for macro-averaging
    case_precision_values = []
    case_recall_values = []
    case_f1_values = []
    
    # Evaluate each sample
    for sample_id in sorted(predictions_dict.keys()):
        # Get HPO codes for this sample
        predictions = predictions_dict.get(sample_id, [])
        ground_truth = ground_truth_dict.get(sample_id, [])
        
        # Add to all predictions counter
        all_predictions_total.extend(predictions)
        
        # Track if this case has ground truth
        has_ground_truth = len(ground_truth) > 0
        
        # Skip empty samples
        if not predictions and not ground_truth:
            continue
        
        # Evaluate this sample
        sample_result = code_based_evaluation(predictions, ground_truth)
        
        # Store per-sample metrics
        result["per_sample_metrics"][sample_id] = {
            "precision": sample_result["precision"],
            "recall": sample_result["recall"],
            "f1_score": sample_result["f1_score"],
            "tp_count": sample_result["tp_count"],
            "fp_count": sample_result["fp_count"],
            "fn_count": sample_result["fn_count"],
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
            all_predictions_with_gt.extend(predictions)
            all_ground_truth.extend(ground_truth)
            
            # Approach 2: Count-based - accumulate counts
            total_tp += sample_result["tp_count"]
            total_fp += sample_result["fp_count"]
            total_fn += sample_result["fn_count"]
            
            # Approach 3: Macro-averaging - collect metrics for averaging
            case_precision_values.append(sample_result["precision"])
            case_recall_values.append(sample_result["recall"])
            case_f1_values.append(sample_result["f1_score"])
            
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
    
    # Approach 1: Micro-averaging metrics (corpus-level pooling)
    micro_result = code_based_evaluation(all_predictions_with_gt, all_ground_truth)
    
    # Approach 2: Count-based metrics
    count_based_metrics = {}
    if total_tp + total_fp > 0:
        count_based_metrics["precision"] = total_tp / (total_tp + total_fp)
    else:
        count_based_metrics["precision"] = 1.0 if total_tp > 0 else 0.0
        
    if total_tp + total_fn > 0:
        count_based_metrics["recall"] = total_tp / (total_tp + total_fn)
    else:
        count_based_metrics["recall"] = 1.0 if total_tp > 0 else 0.0
        
    if count_based_metrics["precision"] + count_based_metrics["recall"] > 0:
        count_based_metrics["f1_score"] = 2 * (count_based_metrics["precision"] * count_based_metrics["recall"]) / (count_based_metrics["precision"] + count_based_metrics["recall"])
    else:
        count_based_metrics["f1_score"] = 0.0
        
    count_based_metrics["tp_count"] = total_tp
    count_based_metrics["fp_count"] = total_fp
    count_based_metrics["fn_count"] = total_fn
    
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
        "description": "Micro-averaging: All HPO codes pooled together across cases"
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
        "unique_pred_count": micro_result["unique_pred_count"],
        "unique_truth_count": micro_result["unique_truth_count"],
        "total_pred_count": micro_result["total_pred_count"],
        "total_truth_count": micro_result["total_truth_count"],
        "cases_with_ground_truth": len(cases_with_ground_truth),
        "cases_without_ground_truth": len(cases_without_ground_truth),
        "total_cases": len(cases_with_ground_truth) + len(cases_without_ground_truth),
        "total_predictions_all_cases": len(all_predictions_total)
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
        " 1. Micro-averaging: All HPO codes pooled together across cases",
        " 2. Macro-averaging: Metrics calculated per case, then averaged",
        " 3. Count-based: TP, FP, FN counts summed across cases, then metrics calculated",
        f"Only cases in predictions file were evaluated ({len(predictions_dict)} cases)",
        f"Cases without ground truth ({len(cases_without_ground_truth)}) are tracked but excluded from F1 calculations",
        "False positives are tracked for all cases, including those without ground truth",
        "NOTE: This evaluation uses EXACT MATCHING of HPO codes, not fuzzy string matching of phenotype names"
    ]
    
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
        fp_counter[fp["code"]] += 1
    
    for fn in result["corpus_false_negatives"]:
        fn_counter[fn["code"]] += 1
    
    # Get most common errors
    analysis["most_common_false_positives"] = [
        {"code": code, "count": count}
        for code, count in fp_counter.most_common(20)
    ]
    
    analysis["most_common_false_negatives"] = [
        {"code": code, "count": count}
        for code, count in fn_counter.most_common(20)
    ]
    
    return analysis



def extract_predictions(predictions_data: Dict, only_direct: bool = False) -> Dict[str, List[str]]:
    """Extract HPO codes, optionally filtering for direct phenotypes only."""
    result = {}
    
    if isinstance(predictions_data, dict):
        for case_id, case_data in predictions_data.items():
            hpo_codes = []
            
            phenotype_fields = ["matched_phenotypes", "verified_phenotypes"]
            for field in phenotype_fields:
                if field in case_data:
                    phenotype_list = case_data[field]
                    if isinstance(phenotype_list, list):
                        for item in phenotype_list:
                            if not isinstance(item, dict):
                                continue
                            
                            # Filter for direct phenotypes if requested
                            if only_direct and item.get("status") != "direct_phenotype":
                                continue
                            
                            # Extract the HPO code
                            for code_field in ["HPO_Term", "hpo_term", "hpo_id", "hp_id"]:
                                if code_field in item and item[code_field]:
                                    code = item[code_field]
                                    if isinstance(code, str) and code.startswith("HP:"):
                                        hpo_codes.append(code)
                                        break
            
            if hpo_codes:
                result[str(case_id)] = hpo_codes
    
    return result

def print_evaluation_summary(result: Dict) -> None:
    """
    Print a summary of the evaluation results.
    
    Args:
        result: Dictionary with corpus evaluation results
    """
    print("\n=== HPO Code Evaluation Summary ===")
    print("NOTE: This evaluation uses EXACT MATCHING of HPO codes, not fuzzy string matching of phenotype names")
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
    print("\n1. MICRO-AVERAGING (pooling all HPO codes across cases):")
    print(f"  Precision: {micro['precision']:.4f}")
    print(f"  Recall: {micro['recall']:.4f}")
    print(f"  F1 Score: {micro['f1_score']:.4f}")
    print(f"  True Positives: {micro['tp_count']}")
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
    print(f"  True Positives: {count['tp_count']}")
    print(f"  False Positives: {count['fp_count']}")
    print(f"  False Negatives: {count['fn_count']}")
    
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
    else:
        print("\nNo per-sample statistics available (no samples with both predictions and ground truth)")
    
    # Print error analysis
    if "error_analysis" in result:
        analysis = result["error_analysis"]
        
        if analysis["most_common_false_positives"]:
            print("\nMost Common False Positive HPO Codes (all cases):")
            for i, item in enumerate(analysis["most_common_false_positives"][:10]):
                print(f"  {i+1}. {item['code']} ({item['count']} occurrences)")
        
        if analysis["most_common_false_negatives"]:
            print("\nMost Common False Negative HPO Codes (cases with ground truth):")
            for i, item in enumerate(analysis["most_common_false_negatives"][:10]):
                print(f"  {i+1}. {item['code']} ({item['count']} occurrences)")
                
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
        print("  2. HPO codes in predictions may not match those in ground truth")
        print("  3. Check for HP: prefix differences or formatting issues")


def main():
    parser = argparse.ArgumentParser(description="Evaluate HPO code predictions across a corpus")
    
    parser.add_argument("--predictions", required=True,
                       help="Path to JSON file with predictions")
    parser.add_argument("--ground-truth", required=True,
                       help="Path to JSON file with ground truth")
    parser.add_argument("--output", required=True,
                       help="Path to save evaluation results JSON")
    parser.add_argument("--summary-only", action="store_true",
                       help="Only print summary, don't save detailed JSON")
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading predictions from {args.predictions}")
    print(f"Loading ground truth from {args.ground_truth}")
    predictions_data, ground_truth_data = load_data(args.predictions, args.ground_truth)
    
    # Extract HPO codes
    predictions_dict = extract_predictions(predictions_data)
    ground_truth_dict = extract_ground_truth(ground_truth_data)
    
    print(f"Found predictions for {len(predictions_dict)} samples")
    print(f"Found ground truth for {len(ground_truth_dict)} samples")
    
    # Evaluate corpus
    print(f"Evaluating with exact matching of HPO codes")
    result = evaluate_corpus(predictions_dict, ground_truth_dict)
    
    # Add error analysis
    result["error_analysis"] = analyze_corpus_errors(result)
    
    # Print summary
    print_evaluation_summary(result)
    
    # Original evaluation
    predictions_dict = extract_predictions(predictions_data)
    result = evaluate_corpus(predictions_dict, ground_truth_dict)

    # Direct-match only evaluation
    predictions_dict_direct = extract_predictions(predictions_data, only_direct=True)
    result_direct = evaluate_corpus(predictions_dict_direct, ground_truth_dict)

    # Compare results
    print("\n=== Comparison: All Matches vs. Direct Matches Only ===")
    print("\nCount-based metrics:")
    print(f"  Precision - All: {result['count_based_metrics']['precision']:.4f}, Direct: {result_direct['count_based_metrics']['precision']:.4f}")
    print(f"  Recall - All: {result['count_based_metrics']['recall']:.4f}, Direct: {result_direct['count_based_metrics']['recall']:.4f}")
    print(f"  F1 Score - All: {result['count_based_metrics']['f1_score']:.4f}, Direct: {result_direct['count_based_metrics']['f1_score']:.4f}")
    print(f"  TP/FP/FN - All: {result['count_based_metrics']['tp_count']}/{result['count_based_metrics']['fp_count']}/{result['count_based_metrics']['fn_count']}")
    print(f"  TP/FP/FN - Direct: {result_direct['count_based_metrics']['tp_count']}/{result_direct['count_based_metrics']['fp_count']}/{result_direct['count_based_metrics']['fn_count']}")
    # Save results
    if not args.summary_only:
        print(f"Saving evaluation results to {args.output}")
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Results saved successfully")


if __name__ == "__main__":
    main()