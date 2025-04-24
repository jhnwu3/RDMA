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

# Add parent directory to path
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

def code_based_evaluation(predictions: List[str], ground_truth: List[str]) -> Dict:
    """
    Evaluates predictions against ground truth using exact matching of ORPHA codes.
    Only compares the numeric portion of the ORPHA codes.
    
    Args:
        predictions: List of predicted ORPHA codes
        ground_truth: List of ground truth ORPHA codes
        
    Returns:
        Dictionary with precision, recall, F1 scores and match information
    """
    import re
    
    # Extract only numeric part of ORPHA codes
    def normalize_code(code):
        if not code:
            return ""
        # Use regex to extract only the digits
        match = re.search(r'(\d+)', code)
        if match:
            return match.group(1)
        return code
    
    # Normalize predictions and ground truth
    normalized_predictions = [normalize_code(p) for p in predictions if p]
    normalized_ground_truth = [normalize_code(g) for g in ground_truth if g]
    
    # Print some debug info
    if predictions and ground_truth:
        print(f"\nDebug: Sample normalization:")
        print(f"  Original prediction: '{predictions[0]}'")
        print(f"  Normalized prediction: '{normalize_code(predictions[0])}'")
        print(f"  Original ground truth: '{ground_truth[0]}'")
        print(f"  Normalized ground truth: '{normalize_code(ground_truth[0])}'")
    
    # Create sets for exact matching with normalized codes
    unique_predictions = set(normalized_predictions)
    unique_ground_truth = set(normalized_ground_truth)
    
    # Store original counts for reference
    pred_counter = Counter(normalized_predictions)
    truth_counter = Counter(normalized_ground_truth)
    
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
    try:
        with open(predictions_file, 'r') as f:
            predictions = json.load(f)
            print(f"Successfully loaded predictions from {predictions_file}")
    except Exception as e:
        print(f"Error loading predictions file: {e}")
        raise
    
    try:
        with open(ground_truth_file, 'r') as f:
            ground_truth = json.load(f)
            print(f"Successfully loaded ground truth from {ground_truth_file}")
    except Exception as e:
        print(f"Error loading ground truth file: {e}")
        raise
    
    # Handle nested structure with metadata and results keys
    if isinstance(predictions, dict) and "results" in predictions:
        print("Detected nested structure with 'results' and 'metadata' keys")
        predictions_data = predictions.get("results", {})
        # Keep the metadata for reference, but work with the results for evaluation
    else:
        predictions_data = predictions
    
    return predictions_data, ground_truth


def extract_predictions(predictions_data: Dict, 
                        match_method: Optional[str] = None, 
                        confidence_threshold: Optional[float] = None) -> Dict[str, List[str]]:
    """
    Extract ORPHA codes from the predictions data structure with filtering options.
    """
    result = {}
    
    if not isinstance(predictions_data, dict):
        print(f"Warning: Unexpected predictions data format: {type(predictions_data)}")
        return result
    
    for case_id, case_data in predictions_data.items():
        orpha_codes = []
        
        # Check for matched diseases field
        if "matched_diseases" in case_data and isinstance(case_data["matched_diseases"], list):
            matched_diseases = case_data["matched_diseases"]
            
            for item in matched_diseases:
                if not isinstance(item, dict):
                    continue
                    
                # Apply filters if specified
                if match_method and item.get("match_method") != match_method:
                    continue
                
                if confidence_threshold is not None:
                    confidence = item.get("confidence_score", 0.0)
                    if confidence < confidence_threshold:
                        continue
                    
                # Extract the ORPHA code
                orpha_id = item.get("orpha_id")
                if orpha_id and isinstance(orpha_id, str):
                    # Keep original format for now, normalization happens in evaluation
                    orpha_codes.append(orpha_id)
        
        # Add non-empty lists to result
        if orpha_codes:
            result[str(case_id)] = orpha_codes
    
    print(f"Extracted predictions for {len(result)} cases with {sum(len(codes) for codes in result.values())} total ORPHA codes")
    return result

def extract_prediction_entities(predictions_data: Dict) -> Tuple[Dict[str, List[str]], Dict[str, Dict[str, str]]]:
    """
    Extract ORPHA codes and entity names from the predictions data structure.
    
    Returns:
        Tuple of (
            Dictionary mapping sample_id to list of ORPHA codes,
            Dictionary mapping {doc_id: {orpha_code: entity_name}}
        )
    """
    orpha_codes_dict = {}
    entity_names_dict = {}
    
    if isinstance(predictions_data, dict):
        for case_id, case_data in predictions_data.items():
            orpha_codes = []
            entity_names = {}
            
            if "matched_diseases" in case_data and isinstance(case_data["matched_diseases"], list):
                for item in case_data["matched_diseases"]:
                    if not isinstance(item, dict):
                        continue
                    
                    orpha_id = item.get("orpha_id")
                    entity = item.get("entity")
                    
                    if orpha_id and isinstance(orpha_id, str):
                        # Normalize format
                        if not orpha_id.startswith("ORPHA:"):
                            orpha_id = f"ORPHA:{orpha_id}"
                            
                        orpha_codes.append(orpha_id)
                        
                        # Store entity name mapping
                        if entity:
                            normalized_id = orpha_id.replace("ORPHA:", "").strip().lower()
                            entity_names[normalized_id] = entity
            
            if orpha_codes:
                orpha_codes_dict[str(case_id)] = orpha_codes
                entity_names_dict[str(case_id)] = entity_names
    
    return orpha_codes_dict, entity_names_dict

def extract_ground_truth_entities(ground_truth_data: Dict) -> Tuple[Dict[str, List[str]], Dict[str, Dict[str, str]]]:
    """
    Extract ORPHA codes and entity names from the ground truth data structure.
    
    Returns:
        Tuple of (
            Dictionary mapping sample_id to list of ORPHA codes,
            Dictionary mapping {doc_id: {orpha_code: entity_name}}
        )
    """
    orpha_codes_dict = {}
    entity_names_dict = {}
    
    if isinstance(ground_truth_data, dict):
        for case_id, case_data in ground_truth_data.items():
            orpha_codes = []
            entity_names = {}
            
            if isinstance(case_data, dict) and "annotations" in case_data:
                annotations = case_data["annotations"]
                
                for annotation in annotations:
                    if isinstance(annotation, dict):
                        ordo_field = annotation.get("ordo_with_desc", "")
                        mention = annotation.get("mention", "")
                        
                        if ordo_field and isinstance(ordo_field, str):
                            # Split by space to separate ID from description
                            ordo_parts = ordo_field.split(' ', 1)
                            orpha_id = ordo_parts[0] if ordo_parts else ''
                            
                            if orpha_id:
                                # Normalize format
                                if not orpha_id.startswith("ORPHA:"):
                                    orpha_id = f"ORPHA:{orpha_id}"
                                
                                orpha_codes.append(orpha_id)
                                
                                # Store entity name mapping
                                if mention:
                                    normalized_id = orpha_id.replace("ORPHA:", "").strip().lower()
                                    entity_names[normalized_id] = mention
            
            if orpha_codes:
                orpha_codes_dict[str(case_id)] = orpha_codes
                entity_names_dict[str(case_id)] = entity_names
    
    return orpha_codes_dict, entity_names_dict

def extract_ground_truth(ground_truth_data: Dict) -> Dict[str, List[str]]:
    """
    Extract ORPHA codes from the ground truth data structure in MIMIC-style format.
    """
    result = {}
    total_annotations = 0
    
    # Handle MIMIC-style format
    if isinstance(ground_truth_data, dict):
        for case_id, case_data in ground_truth_data.items():
            orpha_codes = []
            
            # Check for MIMIC-style format with note_details and annotations
            if isinstance(case_data, dict) and "annotations" in case_data:
                annotations = case_data["annotations"]
                total_annotations += len(annotations)
                
                if isinstance(annotations, list):
                    for annotation in annotations:
                        if isinstance(annotation, dict) and "ordo_with_desc" in annotation:
                            ordo_field = annotation["ordo_with_desc"]
                            
                            # Extract ORPHA ID from the ordo_with_desc field
                            if ordo_field and isinstance(ordo_field, str):
                                # Keep original format for now, normalization happens in evaluation
                                orpha_codes.append(f"ORPHA:{ordo_field.split(' ', 1)[0]}")
            
            # Add non-empty lists to result
            if orpha_codes:
                result[str(case_id)] = orpha_codes
    
    print(f"Found {total_annotations} total annotations in data")
    print(f"Extracted ground truth for {len(result)} cases with {sum(len(codes) for codes in result.values())} total ORPHA codes")
    return result

def evaluate_corpus(predictions_dict: Dict[str, List[str]], 
                    ground_truth_dict: Dict[str, List[str]]) -> Dict:
    """
    Evaluate predictions against ground truth across the entire corpus using three approaches:
    1. Micro-averaging (corpus-level): All ORPHA codes from all cases are pooled together
    2. Macro-averaging (case-level): Metrics are calculated per case, then averaged
    3. Count-based: All TP, FP, FN counts are summed across cases, then metrics are calculated
    
    Args:
        predictions_dict: Dictionary mapping sample_id to predicted ORPHA codes
        ground_truth_dict: Dictionary mapping sample_id to ground truth ORPHA codes
        
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
        # Get ORPHA codes for this sample
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
        "description": "Micro-averaging: All ORPHA codes pooled together across cases"
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
        " 1. Micro-averaging: All ORPHA codes pooled together across cases",
        " 2. Macro-averaging: Metrics calculated per case, then averaged",
        " 3. Count-based: TP, FP, FN counts summed across cases, then metrics calculated",
        f"Only cases in predictions file were evaluated ({len(predictions_dict)} cases)",
        f"Cases without ground truth ({len(cases_without_ground_truth)}) are tracked but excluded from F1 calculations",
        "False positives are tracked for all cases, including those without ground truth",
        "NOTE: This evaluation uses EXACT MATCHING of ORPHA codes, not fuzzy string matching of disease names"
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


def print_evaluation_summary(result: Dict) -> None:
    """
    Print a summary of the evaluation results.
    
    Args:
        result: Dictionary with corpus evaluation results
    """
    print("\n=== ORPHA Code Evaluation Summary ===")
    print("NOTE: This evaluation uses EXACT MATCHING of ORPHA codes, not fuzzy string matching of disease names")
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
    print("\n1. MICRO-AVERAGING (pooling all ORPHA codes across cases):")
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
            print("\nMost Common False Positive ORPHA Codes (all cases):")
            for i, item in enumerate(analysis["most_common_false_positives"][:10]):
                print(f"  {i+1}. {item['code']} ({item['count']} occurrences)")
        
        if analysis["most_common_false_negatives"]:
            print("\nMost Common False Negative ORPHA Codes (cases with ground truth):")
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
        print("  2. ORPHA codes in predictions may not match those in ground truth")
        print("  3. Check for ORPHA: prefix differences or formatting issues")
def evaluate_fuzzy_match(
    predictions_dict: Dict[str, List[str]],
    ground_truth_dict: Dict[str, List[str]],
    prediction_entities: Dict[str, Dict[str, str]],
    ground_truth_entities: Dict[str, Dict[str, str]],
    threshold: int = 90,
    debug: bool = False
) -> Dict[str, Any]:
    """
    Evaluate predictions using fuzzy matching on entity names.
    
    Args:
        predictions_dict: Dictionary mapping sample_id to predicted ORPHA codes
        ground_truth_dict: Dictionary mapping sample_id to ground truth ORPHA codes
        prediction_entities: Dictionary mapping {doc_id: {numeric_orpha_id: entity_name}}
        ground_truth_entities: Dictionary mapping {doc_id: {numeric_orpha_id: entity_name}}
        threshold: Threshold for fuzzy matching (0-100)
        debug: Whether to print detailed debug information
        
    Returns:
        Dictionary with evaluation metrics and unmatched entity details
    """
    from fuzzywuzzy import fuzz
    import re
    import numpy as np
    
    # Helper function to extract numeric ID
    def get_numeric_id(code: str) -> str:
        match = re.search(r'(\d+)', code)
        return match.group(1) if match else ""
    
    # Initialize result structure
    result = {
        "per_sample_metrics": {},
        "corpus_metrics": {},
        "micro_averaging_metrics": {},
        "macro_averaging_metrics": {},
        "count_based_metrics": {},
        "cases_with_ground_truth": [],
        "cases_without_ground_truth": [],
        "unmatched_details": {}
    }
    
    # Initialize counters for aggregate metrics
    all_tp_count = 0
    all_fp_count = 0
    all_fn_count = 0
    
    # Initialize lists for macro-averaging
    case_precision_values = []
    case_recall_values = []
    case_f1_values = []
    
    # Detailed match tracking for debugging
    all_fuzzy_matches = []
    
    if debug:
        print("\n===== DEBUG: FUZZY MATCHING DETAILS =====")
        print(f"Threshold for fuzzy matching: {threshold}")
    
    # Process each document
    for doc_id in set(predictions_dict.keys()) & set(ground_truth_dict.keys()):
        pred_codes = predictions_dict.get(doc_id, [])
        gt_codes = ground_truth_dict.get(doc_id, [])
        
        # Skip if no ground truth
        if not gt_codes:
            result["cases_without_ground_truth"].append(doc_id)
            continue
        
        result["cases_with_ground_truth"].append(doc_id)
        
        # Initialize unmatched details for this document
        result["unmatched_details"][doc_id] = {
            "false_negatives": [],
            "false_positives": []
        }
        
        # Check if we have entity names for fuzzy matching
        if (doc_id not in prediction_entities or doc_id not in ground_truth_entities):
            if debug:
                print(f"  No entity names available for document {doc_id} - skipping fuzzy matching")
            
            # Mark all predictions as false positives and all ground truth as false negatives
            result["unmatched_details"][doc_id]["false_positives"] = [
                {"name": code, "orpha_code": code} for code in pred_codes
            ]
            result["unmatched_details"][doc_id]["false_negatives"] = [
                {"name": code, "orpha_code": code} for code in gt_codes
            ]
            
            # Count metrics
            sample_result = {
                "tp_count": 0,
                "fp_count": len(pred_codes),
                "fn_count": len(gt_codes),
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0
            }
            result["per_sample_metrics"][doc_id] = sample_result
            
            # Update aggregate counts
            all_fp_count += sample_result["fp_count"]
            all_fn_count += sample_result["fn_count"]
            
            continue
        
        # Get entity names
        pred_entities = prediction_entities[doc_id]
        gt_entities = ground_truth_entities[doc_id]
        
        # Track matches
        matched_pred_nums = set()
        matched_gt_nums = set()
        
        # Track fuzzy matches for this document
        doc_fuzzy_matches = []
        
        # Try to match each prediction to ground truth
        for pred_num, pred_name in pred_entities.items():
            pred_name = pred_name.lower()
            best_match = None
            best_score = 0
            
            # Try to find best match among unmatched ground truth entities
            for gt_num, gt_name in gt_entities.items():
                # Skip already matched ground truth
                if gt_num in matched_gt_nums:
                    continue
                
                gt_name = gt_name.lower()
                score = fuzz.ratio(pred_name, gt_name)
                
                if score > best_score:
                    best_score = score
                    best_match = gt_num
            
            # If good match found
            if best_match and best_score >= threshold:
                matched_pred_nums.add(pred_num)
                matched_gt_nums.add(best_match)
                
                if debug or len(doc_fuzzy_matches) < 10:
                    doc_fuzzy_matches.append({
                        'pred_name': pred_name,
                        'gt_name': gt_entities[best_match],
                        'score': best_score
                    })
        
        # Store unmatched details
        result["unmatched_details"][doc_id]["false_positives"] = [
            {"name": pred_entities[num], "orpha_code": f"ORPHA:{num}"} 
            for num in set(pred_entities.keys()) - matched_pred_nums
        ]
        result["unmatched_details"][doc_id]["false_negatives"] = [
            {"name": gt_entities[num], "orpha_code": f"ORPHA:{num}"} 
            for num in set(gt_entities.keys()) - matched_gt_nums
        ]
        
        # Calculate metrics for this document
        tp_count = len(matched_pred_nums)
        fp_count = len(pred_entities) - tp_count
        fn_count = len(gt_entities) - len(matched_gt_nums)
        
        # Compute precision, recall, F1
        precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0.0
        recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Store sample-level metrics
        sample_result = {
            "tp_count": tp_count,
            "fp_count": fp_count,
            "fn_count": fn_count,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "fuzzy_matches": doc_fuzzy_matches
        }
        result["per_sample_metrics"][doc_id] = sample_result
        
        # Accumulate for aggregate metrics
        all_tp_count += tp_count
        all_fp_count += fp_count
        all_fn_count += fn_count
        
        # For macro-averaging
        case_precision_values.append(precision)
        case_recall_values.append(recall)
        case_f1_values.append(f1_score)
        
        # Collect debug matches
        if debug and doc_fuzzy_matches:
            all_fuzzy_matches.extend(doc_fuzzy_matches)
    
    # Compute micro-averaging metrics
    micro_precision = all_tp_count / (all_tp_count + all_fp_count) if (all_tp_count + all_fp_count) > 0 else 0.0
    micro_recall = all_tp_count / (all_tp_count + all_fn_count) if (all_tp_count + all_fn_count) > 0 else 0.0
    micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0
    
    result["micro_averaging_metrics"] = {
        "precision": micro_precision,
        "recall": micro_recall,
        "f1_score": micro_f1,
        "tp_count": all_tp_count,
        "fp_count": all_fp_count,
        "fn_count": all_fn_count,
        "description": "Micro-averaging: All matches pooled together across cases"
    }
    
    # Compute macro-averaging metrics
    macro_precision = np.mean(case_precision_values) if case_precision_values else 0.0
    macro_recall = np.mean(case_recall_values) if case_recall_values else 0.0
    macro_f1 = np.mean(case_f1_values) if case_f1_values else 0.0
    
    result["macro_averaging_metrics"] = {
        "precision": macro_precision,
        "recall": macro_recall,
        "f1_score": macro_f1,
        "precision_std": np.std(case_precision_values) if case_precision_values else 0.0,
        "recall_std": np.std(case_recall_values) if case_recall_values else 0.0,
        "f1_score_std": np.std(case_f1_values) if case_f1_values else 0.0,
        "case_count": len(case_precision_values),
        "description": "Macro-averaging: Metrics calculated per case, then averaged"
    }
    
    # Compute count-based metrics
    result["count_based_metrics"] = {
        "precision": all_tp_count / (all_tp_count + all_fp_count) if (all_tp_count + all_fp_count) > 0 else 0.0,
        "recall": all_tp_count / (all_tp_count + all_fn_count) if (all_tp_count + all_fn_count) > 0 else 0.0,
        "f1_score": 2 * (all_tp_count / (all_tp_count + all_fp_count) * all_tp_count / (all_tp_count + all_fn_count)) / 
                    (all_tp_count / (all_tp_count + all_fp_count) + all_tp_count / (all_tp_count + all_fn_count)) 
                    if (all_tp_count + all_fp_count + all_fn_count) > 0 else 0.0,
        "tp_count": all_tp_count,
        "fp_count": all_fp_count,
        "fn_count": all_fn_count,
        "description": "Count-based: TP, FP, FN counts summed across cases, then metrics calculated"
    }
    
    # Add sample-level unmatched details to the result structure
    result["total_false_positives"] = sum(
        len(details["false_positives"]) 
        for details in result["unmatched_details"].values()
    )
    result["total_false_negatives"] = sum(
        len(details["false_negatives"]) 
        for details in result["unmatched_details"].values()
    )
    
    # Corpus metrics (using micro-averaging)
    result["corpus_metrics"] = result["micro_averaging_metrics"].copy()
    result["corpus_metrics"].update({
        "cases_with_ground_truth": len(result["cases_with_ground_truth"]),
        "cases_without_ground_truth": len(result["cases_without_ground_truth"]),
        "total_cases": len(result["cases_with_ground_truth"]) + len(result["cases_without_ground_truth"])
    })
    
    # Top-level metrics for compatibility
    result["precision"] = micro_precision
    result["recall"] = micro_recall
    result["f1_score"] = micro_f1
    result["tp_count"] = all_tp_count
    result["fp_count"] = all_fp_count
    result["fn_count"] = all_fn_count
    result["total_matches_found"] = len(all_fuzzy_matches)
    
    # Notes about the evaluation
    result["notes"] = [
        f"Fuzzy matching approach using threshold of {threshold}",
        "Three approaches to metric calculation:",
        " 1. Micro-averaging: All matches pooled together across cases",
        " 2. Macro-averaging: Metrics calculated per case, then averaged",
        " 3. Count-based: TP, FP, FN counts summed across cases, then metrics calculated",
        "NOTE: Only documents with entity names are processed for fuzzy matching"
    ]
    
    # Debug output if requested
    if debug and all_fuzzy_matches:
        print("\n===== FUZZY MATCHING SUMMARY =====")
        print("Top 10 fuzzy matches:")
        sorted_matches = sorted(all_fuzzy_matches, key=lambda x: x['score'], reverse=True)
        for i, match in enumerate(sorted_matches[:10], 1):
            print(f"  {i}. '{match['pred_name']}' matched with '{match['gt_name']}' (score: {match['score']})")
    
    return result

def main():
    parser = argparse.ArgumentParser(description="Evaluate ORPHA code predictions across a corpus")
    
    # Required arguments
    parser.add_argument("--predictions", required=True,
                       help="Path to JSON file with predictions")
    parser.add_argument("--ground-truth", required=True,
                       help="Path to JSON file with ground truth")
    parser.add_argument("--output", required=True,
                       help="Path to save evaluation results JSON")
    parser.add_argument("--debug-fuzzy", action="store_true",
                       help="Enable detailed debug output for fuzzy matching")
    # Optional filtering arguments
    parser.add_argument("--match-method", type=str, choices=["exact", "llm"],
                       help="Filter predictions by match method ('exact' or 'llm')")
    parser.add_argument("--confidence-threshold", type=float,
                       help="Filter predictions by minimum confidence score (0.0-1.0)")
    
    # Fuzzy matching configuration
    parser.add_argument("--fuzzy-threshold", type=int, default=90,
                       help="Threshold for fuzzy matching (0-100, default: 90)")
    parser.add_argument("--enable-fuzzy", action="store_true",
                       help="Enable fuzzy matching of disease entity names")
    
    # Output control arguments
    parser.add_argument("--summary-only", action="store_true",
                       help="Only print summary, don't save detailed JSON")
    parser.add_argument("--save-csv", type=str,
                       help="Path to save summary results as CSV")
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading predictions from {args.predictions}")
    print(f"Loading ground truth from {args.ground_truth}")
    predictions_data, ground_truth_data = load_data(args.predictions, args.ground_truth)
    
    # Apply any filtering from command-line arguments
    predictions_dict = extract_predictions(
        predictions_data, 
        match_method=args.match_method,
        confidence_threshold=args.confidence_threshold
    )
    
    ground_truth_dict = extract_ground_truth(ground_truth_data)
    
    # Print debug info about document ID matching
    pred_ids = set(predictions_dict.keys())
    gt_ids = set(ground_truth_dict.keys())
    common_ids = pred_ids & gt_ids
    
    print(f"\nDocument ID matching:")
    print(f"  Prediction document IDs: {len(pred_ids)}")
    print(f"  Ground truth document IDs: {len(gt_ids)}")
    print(f"  Common document IDs: {len(common_ids)}")
    
    if len(common_ids) == 0:
        print("\nWARNING: No common document IDs found between predictions and ground truth!")
        print("Sample prediction IDs:", list(pred_ids)[:5])
        print("Sample ground truth IDs:", list(gt_ids)[:5])
    
    # Run exact match evaluation with numeric-only normalization
    print(f"\nRunning exact match evaluation (numeric ORPHA ID matching)...")
    exact_result = evaluate_corpus(predictions_dict, ground_truth_dict)
    
    # Add error analysis
    exact_result["error_analysis"] = analyze_corpus_errors(exact_result)
    
    # Extract entity names for fuzzy matching if enabled
    if args.enable_fuzzy:
        print(f"\nExtracting entity names for fuzzy matching...")
        
        # Extract entity names mapped to ORPHA codes
        prediction_entities = {}
        ground_truth_entities = {}
        
        # Process predictions
        for doc_id, doc_data in predictions_data.get("results", predictions_data).items():
            if "matched_diseases" in doc_data:
                entities = {}
                for match in doc_data["matched_diseases"]:
                    if "entity" in match and "orpha_id" in match:
                        # Extract just the numeric part for the key
                        import re
                        match_id = re.search(r'(\d+)', match["orpha_id"])
                        if match_id:
                            if "original_entity" in match:
                                entities[match_id.group(1)] = match["original_entity"]
                            else:
                                entities[match_id.group(1)] = match["entity"]
                
                if entities:
                    prediction_entities[doc_id] = entities
        
        # Process ground truth
        for doc_id, doc_data in ground_truth_data.items():
            if "annotations" in doc_data:
                entities = {}
                for ann in doc_data["annotations"]:
                    if "mention" in ann and "ordo_with_desc" in ann:
                        # Get first word as ORPHA ID
                        orpha_id = ann["ordo_with_desc"].split(' ', 1)[0]
                        # Extract just the numeric part for the key
                        import re
                        match_id = re.search(r'(\d+)', orpha_id)
                        if match_id:
                            entities[match_id.group(1)] = ann["mention"]
                
                if entities:
                    ground_truth_entities[doc_id] = entities
        
        print(f"Extracted entity names for {len(prediction_entities)} prediction docs and {len(ground_truth_entities)} ground truth docs")
        
        # Run fuzzy matching evaluation
        print(f"\nRunning fuzzy match evaluation (disease entity name matching)...")
        fuzzy_result = evaluate_fuzzy_match(
            predictions_dict, 
            ground_truth_dict,
            prediction_entities,
            ground_truth_entities,
            threshold=args.fuzzy_threshold,
            debug=args.debug_fuzzy  # Pass the debug flag
        )
        
        # Combine results
        combined_result = {
            "exact_match": exact_result,
            "fuzzy_match": fuzzy_result,
            "metadata": {
                "evaluation_timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "fuzzy_threshold": args.fuzzy_threshold,
                "match_method_filter": args.match_method,
                "confidence_threshold": args.confidence_threshold
            }
        }
    else:
        # Just use exact match results
        combined_result = {
            "exact_match": exact_result,
            "metadata": {
                "evaluation_timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "match_method_filter": args.match_method,
                "confidence_threshold": args.confidence_threshold
            }
        }
    
    # Print evaluation summary
    print("\n=== Evaluation Summary ===")
    print("\nExact Match Metrics (Numeric ORPHA ID):")
    print(f"  Precision: {exact_result['count_based_metrics']['precision']:.4f}")
    print(f"  Recall: {exact_result['count_based_metrics']['recall']:.4f}")
    print(f"  F1 Score: {exact_result['count_based_metrics']['f1_score']:.4f}")
    print(f"  TP/FP/FN: {exact_result['count_based_metrics']['tp_count']}/{exact_result['count_based_metrics']['fp_count']}/{exact_result['count_based_metrics']['fn_count']}")
    
    if args.enable_fuzzy:
        print("\nFuzzy Match Metrics (Disease Entity Names):")
        print(f"  Precision: {fuzzy_result['precision']:.4f}")
        print(f"  Recall: {fuzzy_result['recall']:.4f}")
        print(f"  F1 Score: {fuzzy_result['f1_score']:.4f}")
        print(f"  TP/FP/FN: {fuzzy_result['tp_count']}/{fuzzy_result['fp_count']}/{fuzzy_result['fn_count']}")
        print(f"  Entity matches found: {fuzzy_result['total_matches_found']}")
    
    # Save results if not summary-only
    if not args.summary_only:
        print(f"\nSaving evaluation results to {args.output}")
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(combined_result, f, indent=2)
    
    # Save summary as CSV if requested
    if args.save_csv:
        print(f"Saving summary as CSV to {args.save_csv}")
        
        # Create DataFrame for CSV
        data = {
            'metric': ['precision', 'recall', 'f1_score', 'tp_count', 'fp_count', 'fn_count'],
            'exact_match': [
                exact_result['count_based_metrics']['precision'],
                exact_result['count_based_metrics']['recall'],
                exact_result['count_based_metrics']['f1_score'],
                exact_result['count_based_metrics']['tp_count'],
                exact_result['count_based_metrics']['fp_count'],
                exact_result['count_based_metrics']['fn_count']
            ]
        }
        
        if args.enable_fuzzy:
            data['fuzzy_match'] = [
                fuzzy_result['precision'],
                fuzzy_result['recall'],
                fuzzy_result['f1_score'],
                fuzzy_result['tp_count'],
                fuzzy_result['fp_count'],
                fuzzy_result['fn_count']
            ]
        
        metrics_df = pd.DataFrame(data)
        metrics_df.to_csv(args.save_csv, index=False)
        print(f"CSV summary saved successfully")

if __name__ == "__main__":
    main()