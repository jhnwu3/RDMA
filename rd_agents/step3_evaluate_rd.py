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
from fuzzywuzzy import fuzz
import re
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
    threshold: int = 90
) -> Dict[str, Any]:
    """
    Evaluate predictions using fuzzy matching on entity names.
    
    Args:
        predictions_dict: Dictionary mapping sample_id to predicted ORPHA codes
        ground_truth_dict: Dictionary mapping sample_id to ground truth ORPHA codes
        prediction_entities: Dictionary mapping {doc_id: {numeric_orpha_id: entity_name}}
        ground_truth_entities: Dictionary mapping {doc_id: {numeric_orpha_id: entity_name}}
        threshold: Threshold for fuzzy matching (0-100)
        
    Returns:
        Dictionary with evaluation metrics
    """
    from fuzzywuzzy import fuzz
    import re
    
    # Helper function to extract numeric ID
    def get_numeric_id(code: str) -> str:
        match = re.search(r'(\d+)', code)
        return match.group(1) if match else ""
    
    # Initialize counters
    tp_count = 0
    fp_count = 0
    fn_count = 0
    fuzzy_matches_found = 0
    
    # Process each document with ground truth
    for doc_id in set(predictions_dict.keys()) & set(ground_truth_dict.keys()):
        pred_codes = predictions_dict.get(doc_id, [])
        gt_codes = ground_truth_dict.get(doc_id, [])
        
        # Skip if no ground truth
        if not gt_codes:
            continue
            
        # Extract numeric IDs
        pred_nums = [get_numeric_id(code) for code in pred_codes]
        gt_nums = [get_numeric_id(code) for code in gt_codes]
        
        # Find exact matches first
        exact_matches = set(pred_nums) & set(gt_nums)
        tp_count += len(exact_matches)
        
        # Remaining predictions and ground truth after removing exact matches
        remaining_preds = [num for num in pred_nums if num not in exact_matches]
        remaining_gts = [num for num in gt_nums if num not in exact_matches]
        
        # Skip fuzzy matching if no entity names available for this document
        if doc_id not in prediction_entities or doc_id not in ground_truth_entities:
            fp_count += len(remaining_preds)
            fn_count += len(remaining_gts)
            continue
        
        # Get entity names
        pred_entities = prediction_entities[doc_id]
        gt_entities = ground_truth_entities[doc_id]
        
        # Track matched prediction and ground truth IDs to avoid double counting
        matched_preds = set()
        matched_gts = set()
        
        # Try to find fuzzy matches for each remaining prediction
        for pred_num in remaining_preds:
            if pred_num not in pred_entities:
                continue
                
            pred_name = pred_entities[pred_num].lower()
            best_score = 0
            best_match = None
            
            # Compare against all ground truth entities
            for gt_num in remaining_gts:
                if gt_num not in gt_entities or gt_num in matched_gts:
                    continue
                    
                gt_name = gt_entities[gt_num].lower()
                score = fuzz.ratio(pred_name, gt_name)
                
                if score > best_score:
                    best_score = score
                    best_match = gt_num
            
            # If good match found, count as true positive
            if best_match and best_score >= threshold:
                tp_count += 1
                matched_preds.add(pred_num)
                matched_gts.add(best_match)
                fuzzy_matches_found += 1
                print(f"Fuzzy match: '{pred_name}' matched with '{gt_entities[best_match]}' (score: {best_score})")
        
        # Count remaining as false positives and false negatives
        fp_count += len(remaining_preds) - len(matched_preds)
        fn_count += len(remaining_gts) - len(matched_gts)
    
    # Calculate metrics
    precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0
    recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "tp_count": tp_count,
        "fp_count": fp_count,
        "fn_count": fn_count,
        "fuzzy_threshold": threshold,
        "fuzzy_matches_found": fuzzy_matches_found,
        "description": f"Fuzzy matching of disease names with threshold {threshold}"
    }

def evaluate_entity_extraction(step1_file: str, step2_file: str, ground_truth_file: str, fuzzy_threshold: int = 90) -> Dict:
    """
    Evaluate entity extraction quality from step 1 and step 2 using fuzzy matching against ground truth mentions.
    
    Args:
        step1_file: Path to step 1 extraction results
        step2_file: Path to step 2 verification results
        ground_truth_file: Path to ground truth data
        fuzzy_threshold: Threshold for fuzzy matching (0-100)
        
    Returns:
        Evaluation metrics for both steps
    """
    # Load files
    print(f"Loading step 1 extraction results from {step1_file}")
    with open(step1_file, 'r') as f:
        step1_data = json.load(f)
    
    print(f"Loading step 2 verification results from {step2_file}")
    with open(step2_file, 'r') as f:
        step2_data = json.load(f)
        
    print(f"Loading ground truth from {ground_truth_file}")
    with open(ground_truth_file, 'r') as f:
        ground_truth_data = json.load(f)
    
    # Handle nested structure if present
    if isinstance(step1_data, dict) and "results" in step1_data:
        step1_data = step1_data["results"]
    if isinstance(step2_data, dict) and "results" in step2_data:
        step2_data = step2_data["results"]
    
    # Extract entities from each file
    extracted_entities = extract_step1_entities(step1_data)
    verified_entities = extract_step2_entities(step2_data)
    ground_truth_entities = extract_ground_truth_entities(ground_truth_data)
    
    # Print statistics
    print(f"\nStatistics:")
    print(f"  Step 1 extracted entities: {sum(len(entities) for entities in extracted_entities.values())} across {len(extracted_entities)} documents")
    print(f"  Step 2 verified entities: {sum(len(entities) for entities in verified_entities.values())} across {len(verified_entities)} documents")
    print(f"  Ground truth entities: {sum(len(entities) for entities in ground_truth_entities.values())} across {len(ground_truth_entities)} documents")
    
    # Evaluate both steps
    step1_metrics = evaluate_entities_fuzzy_match(extracted_entities, ground_truth_entities, fuzzy_threshold)
    step2_metrics = evaluate_entities_fuzzy_match(verified_entities, ground_truth_entities, fuzzy_threshold)
    
    # Return combined results
    return {
        "step1_extraction": step1_metrics,
        "step2_verification": step2_metrics,
        "metadata": {
            "fuzzy_threshold": fuzzy_threshold,
            "evaluation_timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    }

def extract_step1_entities(data: Dict) -> Dict[str, List[str]]:
    """Extract entities from step 1 extraction results."""
    entities_dict = {}
    
    for doc_id, doc_data in data.items():
        if "entities_with_contexts" in doc_data and isinstance(doc_data["entities_with_contexts"], list):
            entities = []
            
            for item in doc_data["entities_with_contexts"]:
                # Different possible entity formats
                if isinstance(item, dict):
                    if "entity" in item:
                        entities.append(item["entity"])
                    elif "term" in item:
                        entities.append(item["term"])
                elif isinstance(item, str):
                    entities.append(item)
            
            if entities:
                entities_dict[str(doc_id)] = entities
    
    return entities_dict

def extract_step2_entities(data: Dict) -> Dict[str, List[str]]:
    """Extract entities from step 2 verification results."""
    entities_dict = {}
    
    for doc_id, doc_data in data.items():
        if "verified_rare_diseases" in doc_data and isinstance(doc_data["verified_rare_diseases"], list):
            entities = []
            
            for item in doc_data["verified_rare_diseases"]:
                if isinstance(item, dict):
                    if "entity" in item:
                        entities.append(item["entity"])
                    elif "term" in item:
                        entities.append(item["term"])
                elif isinstance(item, str):
                    entities.append(item)
            
            if entities:
                entities_dict[str(doc_id)] = entities
    
    return entities_dict

def extract_ground_truth_entities(data: Dict) -> Dict[str, List[str]]:
    """Extract ground truth entity mentions for fuzzy matching."""
    entities_dict = {}
    
    for doc_id, doc_data in data.items():
        if isinstance(doc_data, dict) and "annotations" in doc_data:
            entities = []
            
            for annotation in doc_data["annotations"]:
                if isinstance(annotation, dict) and "mention" in annotation:
                    entities.append(annotation["mention"])
            
            if entities:
                entities_dict[str(doc_id)] = entities
    
    return entities_dict

def evaluate_entities_fuzzy_match(
    pred_entities: Dict[str, List[str]], 
    gt_entities: Dict[str, List[str]],
    threshold: int = 90
) -> Dict:
    """
    Evaluate entity extraction using fuzzy matching.
    
    Args:
        pred_entities: Dictionary mapping doc_id to list of predicted entities
        gt_entities: Dictionary mapping doc_id to list of ground truth entities
        threshold: Threshold for fuzzy matching (0-100)
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Initialize counters
    tp_count = 0
    fp_count = 0
    fn_count = 0
    fuzzy_matches_found = 0
    
    # Process each document with ground truth
    processed_docs = 0
    for doc_id in sorted(set(pred_entities.keys()) & set(gt_entities.keys())):
        pred_list = pred_entities.get(doc_id, [])
        gt_list = gt_entities.get(doc_id, [])
        
        # Skip if either is empty
        if not pred_list or not gt_list:
            continue
            
        processed_docs += 1
        
        # Track matches to avoid double counting
        matched_preds = set()
        matched_gts = set()
        
        # Try to find fuzzy matches for each prediction
        for i, pred in enumerate(pred_list):
            pred_normalized = pred.lower()
            best_score = 0
            best_match_idx = -1
            
            # Compare against all ground truth entities
            for j, gt in enumerate(gt_list):
                if j in matched_gts:
                    continue
                    
                gt_normalized = gt.lower()
                score = fuzz.ratio(pred_normalized, gt_normalized)
                
                if score > best_score:
                    best_score = score
                    best_match_idx = j
            
            # If good match found, count as true positive
            if best_match_idx >= 0 and best_score >= threshold:
                tp_count += 1
                matched_preds.add(i)
                matched_gts.add(best_match_idx)
                fuzzy_matches_found += 1
                
                if fuzzy_matches_found <= 10:  # Limit printing to avoid excessive output
                    print(f"Fuzzy match: '{pred}' matched with '{gt_list[best_match_idx]}' (score: {best_score})")
        
        # Count remaining as false positives and false negatives
        fp_count += len(pred_list) - len(matched_preds)
        fn_count += len(gt_list) - len(matched_gts)
    
    # Calculate metrics
    precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0
    recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "tp_count": tp_count,
        "fp_count": fp_count,
        "fn_count": fn_count,
        "fuzzy_threshold": threshold,
        "fuzzy_matches_found": fuzzy_matches_found,
        "processed_documents": processed_docs,
        "description": f"Fuzzy matching of entity mentions with threshold {threshold}"
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate rare disease entity extraction and verification")
    
    # Required arguments
    parser.add_argument("--step1", required=True,
                       help="Path to step 1 extraction results JSON file")
    parser.add_argument("--step2", required=True,
                       help="Path to step 2 verification results JSON file")
    parser.add_argument("--ground-truth", required=True,
                       help="Path to ground truth JSON file")
    parser.add_argument("--output", required=True,
                       help="Path to save evaluation results JSON")
    
    # Optional arguments
    parser.add_argument("--fuzzy-threshold", type=int, default=90,
                       help="Threshold for fuzzy matching (0-100, default: 90)")
    parser.add_argument("--summary-only", action="store_true",
                       help="Only print summary, don't save detailed JSON")
    parser.add_argument("--save-csv", type=str,
                       help="Path to save summary results as CSV")
    
    args = parser.parse_args()
    
    # Run entity extraction evaluation
    results = evaluate_entity_extraction(
        args.step1, 
        args.step2, 
        args.ground_truth, 
        args.fuzzy_threshold
    )
    
    # Print evaluation summary
    print("\n=== Entity Extraction Evaluation Summary ===")
    
    print("\nStep 1 Extraction Results:")
    step1 = results["step1_extraction"]
    print(f"  Precision: {step1['precision']:.4f}")
    print(f"  Recall: {step1['recall']:.4f}")
    print(f"  F1 Score: {step1['f1_score']:.4f}")
    print(f"  TP/FP/FN: {step1['tp_count']}/{step1['fp_count']}/{step1['fn_count']}")
    print(f"  Fuzzy matches found: {step1['fuzzy_matches_found']}")
    
    print("\nStep 2 Verification Results:")
    step2 = results["step2_verification"]
    print(f"  Precision: {step2['precision']:.4f}")
    print(f"  Recall: {step2['recall']:.4f}")
    print(f"  F1 Score: {step2['f1_score']:.4f}")
    print(f"  TP/FP/FN: {step2['tp_count']}/{step2['fp_count']}/{step2['fn_count']}")
    print(f"  Fuzzy matches found: {step2['fuzzy_matches_found']}")
    
    # Print improvement from step 1 to step 2
    print("\nImprovement from Step 1 to Step 2:")
    print(f"  Precision: {step2['precision'] - step1['precision']:.4f}")
    print(f"  Recall: {step2['recall'] - step1['recall']:.4f}")
    print(f"  F1 Score: {step2['f1_score'] - step1['f1_score']:.4f}")
    
    # Save results if not summary-only
    if not args.summary_only:
        print(f"\nSaving entity evaluation results to {args.output}")
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
    
    # Save summary as CSV if requested
    if args.save_csv:
        print(f"Saving summary as CSV to {args.save_csv}")
        
        # Create DataFrame for CSV
        data = {
            'metric': ['precision', 'recall', 'f1_score', 'tp_count', 'fp_count', 'fn_count', 'fuzzy_matches'],
            'step1_extraction': [
                step1['precision'],
                step1['recall'],
                step1['f1_score'],
                step1['tp_count'],
                step1['fp_count'],
                step1['fn_count'],
                step1['fuzzy_matches_found']
            ],
            'step2_verification': [
                step2['precision'],
                step2['recall'],
                step2['f1_score'],
                step2['tp_count'],
                step2['fp_count'],
                step2['fn_count'],
                step2['fuzzy_matches_found']
            ],
            'improvement': [
                step2['precision'] - step1['precision'],
                step2['recall'] - step1['recall'],
                step2['f1_score'] - step1['f1_score'],
                'N/A', 'N/A', 'N/A', 'N/A'
            ]
        }
        
        metrics_df = pd.DataFrame(data)
        metrics_df.to_csv(args.save_csv, index=False)
        print(f"CSV summary saved successfully")


if __name__ == "__main__":
    main()