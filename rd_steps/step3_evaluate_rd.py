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


def set_based_code_evaluation(
    predictions_dict: Dict[str, List[str]],
    ground_truth_dict: Dict[str, List[str]],
    prediction_entities: Optional[Dict[str, Dict[str, str]]] = None,
    ground_truth_entities: Optional[Dict[str, Dict[str, str]]] = None,
) -> Dict:
    """
    Evaluates predictions against ground truth using set-based exact matching of ORPHA codes.
    Each document's predictions and ground truth are first deduplicated before evaluation.

    Args:
        predictions_dict: Dictionary mapping sample_id to lists of predicted ORPHA codes
        ground_truth_dict: Dictionary mapping sample_id to lists of ground truth ORPHA codes
        prediction_entities: Dictionary mapping {doc_id: {numeric_orpha_id: entity_name}}
        ground_truth_entities: Dictionary mapping {doc_id: {numeric_orpha_id: entity_name}}

    Returns:
        Dictionary with precision, recall, F1 scores and match information
    """
    import re
    import numpy as np

    # Initialize result structure
    result = {
        "corpus_metrics": {},
        "micro_averaging_metrics": {},
        "macro_averaging_metrics": {},
        "per_sample_metrics": {},
        "corpus_true_positives": [],
        "corpus_false_positives": [],
        "corpus_false_negatives": [],
        "cases_with_ground_truth": [],
        "cases_without_ground_truth": [],
    }

    # Function to normalize ORPHA codes (extract only the numeric part)
    def normalize_code(code):
        if not code:
            return ""
        # Extract numeric part
        match = re.search(r"(\d+)", code)
        return match.group(1) if match else code

    # Track samples with and without ground truth
    cases_with_ground_truth = []
    cases_without_ground_truth = []

    # Initialize counters for aggregation
    all_tp_count = 0
    all_fp_count = 0
    all_fn_count = 0

    # Initialize lists for macro-averaging
    case_precision_values = []
    case_recall_values = []
    case_f1_values = []

    # Process each sample
    for sample_id in sorted(
        set(predictions_dict.keys()) | set(ground_truth_dict.keys())
    ):
        # Get codes for this sample
        pred_codes = predictions_dict.get(sample_id, [])
        gt_codes = ground_truth_dict.get(sample_id, [])

        # Normalize and deduplicate - convert to sets to remove duplicates
        unique_pred_codes = {normalize_code(code) for code in pred_codes if code}
        unique_gt_codes = {normalize_code(code) for code in gt_codes if code}

        # Track if this case has ground truth
        has_ground_truth = bool(unique_gt_codes)
        if has_ground_truth:
            cases_with_ground_truth.append(sample_id)
        else:
            cases_without_ground_truth.append(sample_id)

        # Calculate set intersections for this sample
        true_positives = unique_pred_codes & unique_gt_codes
        false_positives = unique_pred_codes - unique_gt_codes
        false_negatives = unique_gt_codes - unique_pred_codes

        # Calculate metrics for this sample
        tp_count = len(true_positives)
        fp_count = len(false_positives)
        fn_count = len(false_negatives)

        # Only calculate precision/recall/F1 if there's something to evaluate
        if tp_count + fp_count > 0:
            precision = tp_count / (tp_count + fp_count)
        else:
            precision = 1.0 if tp_count > 0 else 0.0

        if tp_count + fn_count > 0:
            recall = tp_count / (tp_count + fn_count)
        else:
            recall = 1.0 if tp_count > 0 else 0.0

        if precision + recall > 0:
            f1_score = 2 * (precision * recall) / (precision + recall)
        else:
            f1_score = 0.0

        # Store per-sample metrics
        result["per_sample_metrics"][sample_id] = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "tp_count": tp_count,
            "fp_count": fp_count,
            "fn_count": fn_count,
            "unique_pred_count": len(unique_pred_codes),
            "unique_truth_count": len(unique_gt_codes),
            "has_ground_truth": has_ground_truth,
        }

        # Get entity names for this sample
        pred_entity_names = {}
        gt_entity_names = {}

        if prediction_entities and sample_id in prediction_entities:
            pred_entity_names = prediction_entities[sample_id]

        if ground_truth_entities and sample_id in ground_truth_entities:
            gt_entity_names = ground_truth_entities[sample_id]

        # Add to corpus-level lists with entity names
        for tp in true_positives:
            # Try to get entity names from both sources
            pred_name = pred_entity_names.get(tp, "")
            gt_name = gt_entity_names.get(tp, "")

            # Use prediction name by default, fall back to ground truth name if needed
            entity_name = pred_name if pred_name else gt_name

            result["corpus_true_positives"].append(
                {
                    "code": tp,
                    "sample_id": sample_id,
                    "name": entity_name,
                    "pred_name": pred_name,
                    "gt_name": gt_name,
                }
            )

        for fp in false_positives:
            # Get entity name from predictions
            entity_name = pred_entity_names.get(fp, "")

            result["corpus_false_positives"].append(
                {"code": fp, "sample_id": sample_id, "name": entity_name}
            )

        for fn in false_negatives:
            # Get entity name from ground truth
            entity_name = gt_entity_names.get(fn, "")

            result["corpus_false_negatives"].append(
                {"code": fn, "sample_id": sample_id, "name": entity_name}
            )

        # Update counters
        all_tp_count += tp_count
        all_fp_count += fp_count
        all_fn_count += fn_count

        # For macro-averaging - include only samples with ground truth
        if has_ground_truth:
            case_precision_values.append(precision)
            case_recall_values.append(recall)
            case_f1_values.append(f1_score)

    # Calculate corpus-level metrics based on aggregated counts
    if all_tp_count + all_fp_count > 0:
        corpus_precision = all_tp_count / (all_tp_count + all_fp_count)
    else:
        corpus_precision = 1.0 if all_tp_count > 0 else 0.0

    if all_tp_count + all_fn_count > 0:
        corpus_recall = all_tp_count / (all_tp_count + all_fn_count)
    else:
        corpus_recall = 1.0 if all_tp_count > 0 else 0.0

    if corpus_precision + corpus_recall > 0:
        corpus_f1 = (
            2 * (corpus_precision * corpus_recall) / (corpus_precision + corpus_recall)
        )
    else:
        corpus_f1 = 0.0

    # Store corpus-level metrics
    result["corpus_metrics"] = {
        "precision": corpus_precision,
        "recall": corpus_recall,
        "f1_score": corpus_f1,
        "tp_count": all_tp_count,
        "fp_count": all_fp_count,
        "fn_count": all_fn_count,
        "cases_with_ground_truth": len(cases_with_ground_truth),
        "cases_without_ground_truth": len(cases_without_ground_truth),
        "total_cases": len(
            set(predictions_dict.keys()) | set(ground_truth_dict.keys())
        ),
    }

    # Same as corpus metrics for micro-averaging
    result["micro_averaging_metrics"] = {
        "precision": corpus_precision,
        "recall": corpus_recall,
        "f1_score": corpus_f1,
        "tp_count": all_tp_count,
        "fp_count": all_fp_count,
        "fn_count": all_fn_count,
        "description": "Micro-averaging: Sum of TP, FP, FN across all cases",
    }

    # Calculate macro-averaging metrics
    if case_precision_values:
        result["macro_averaging_metrics"] = {
            "precision": float(np.mean(case_precision_values)),
            "recall": float(np.mean(case_recall_values)),
            "f1_score": float(np.mean(case_f1_values)),
            "precision_std": float(np.std(case_precision_values)),
            "recall_std": float(np.std(case_recall_values)),
            "f1_score_std": float(np.std(case_f1_values)),
            "case_count": len(case_precision_values),
            "description": "Macro-averaging: Metrics calculated per case, then averaged",
        }
    else:
        result["macro_averaging_metrics"] = {
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "precision_std": 0.0,
            "recall_std": 0.0,
            "f1_score_std": 0.0,
            "case_count": 0,
            "description": "Macro-averaging: Metrics calculated per case, then averaged",
        }

    # Store lists of cases
    result["cases_with_ground_truth"] = cases_with_ground_truth
    result["cases_without_ground_truth"] = cases_without_ground_truth

    return result


def fuzzy_set_based_evaluation(
    predictions_dict: Dict[str, List[str]],
    ground_truth_dict: Dict[str, List[str]],
    prediction_entities: Dict[str, Dict[str, str]],
    ground_truth_entities: Dict[str, Dict[str, str]],
    threshold: int = 90,
) -> Dict:
    """
    Evaluates predictions using set-based fuzzy matching of entity names.
    Each document's predictions and ground truth are first deduplicated before evaluation.

    Args:
        predictions_dict: Dictionary mapping sample_id to lists of predicted ORPHA codes
        ground_truth_dict: Dictionary mapping sample_id to lists of ground truth ORPHA codes
        prediction_entities: Dictionary mapping {doc_id: {orpha_id: entity_name}}
        ground_truth_entities: Dictionary mapping {doc_id: {orpha_id: entity_name}}
        threshold: Threshold for fuzzy matching (0-100)

    Returns:
        Dictionary with evaluation metrics
    """
    from fuzzywuzzy import fuzz
    import re
    import numpy as np

    # Helper function to extract numeric ID
    def get_numeric_id(code: str) -> str:
        if not code:
            return ""
        match = re.search(r"(\d+)", code)
        return match.group(1) if match else ""

    # Initialize result structure
    result = {
        "corpus_metrics": {},
        "micro_averaging_metrics": {},
        "macro_averaging_metrics": {},
        "per_sample_metrics": {},
        "corpus_true_positives": [],
        "corpus_false_positives": [],
        "corpus_false_negatives": [],
        "cases_with_ground_truth": [],
        "cases_without_ground_truth": [],
        "fuzzy_matches": [],
    }

    # Initialize counters for aggregation
    all_tp_count = 0
    all_fp_count = 0
    all_fn_count = 0

    # Initialize lists for macro-averaging
    case_precision_values = []
    case_recall_values = []
    case_f1_values = []

    # Track samples with and without ground truth
    cases_with_ground_truth = []
    cases_without_ground_truth = []

    # Process each sample that exists in either predictions or ground truth
    all_doc_ids = set(predictions_dict.keys()) | set(ground_truth_dict.keys())

    for doc_id in sorted(all_doc_ids):
        # Get ORPHA codes for this sample
        pred_codes = predictions_dict.get(doc_id, [])
        gt_codes = ground_truth_dict.get(doc_id, [])

        # Deduplicate - convert to sets to remove duplicates
        unique_pred_codes = set(pred_codes)
        unique_gt_codes = set(gt_codes)

        # Track if this case has ground truth
        has_ground_truth = bool(unique_gt_codes)
        if has_ground_truth:
            cases_with_ground_truth.append(doc_id)
        else:
            cases_without_ground_truth.append(doc_id)

        # Skip if no predictions or no ground truth
        if not unique_pred_codes or not unique_gt_codes:
            # Special case: No predictions or no ground truth
            tp_count = 0
            fp_count = len(unique_pred_codes)
            fn_count = len(unique_gt_codes)

            # Calculate metrics for this sample
            precision = 1.0 if fp_count == 0 else 0.0
            recall = 1.0 if fn_count == 0 else 0.0
            f1_score = (
                2 * (precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )

            # Store per-sample metrics
            result["per_sample_metrics"][doc_id] = {
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "tp_count": tp_count,
                "fp_count": fp_count,
                "fn_count": fn_count,
                "has_ground_truth": has_ground_truth,
                "method": "empty_set_comparison",
            }

            # Update counters
            all_fp_count += fp_count
            all_fn_count += fn_count

            # Only add to macro-averaging if has ground truth
            if has_ground_truth:
                case_precision_values.append(precision)
                case_recall_values.append(recall)
                case_f1_values.append(f1_score)

            # Add false positives and false negatives to corpus lists
            for code in unique_pred_codes:
                result["corpus_false_positives"].append(
                    {"code": code, "sample_id": doc_id}
                )

            for code in unique_gt_codes:
                result["corpus_false_negatives"].append(
                    {"code": code, "sample_id": doc_id}
                )

            continue

        # Check if we have entity names for fuzzy matching
        have_entity_names = (doc_id in prediction_entities) and (
            doc_id in ground_truth_entities
        )

        if have_entity_names:
            # Perform fuzzy matching using entity names
            pred_entities = prediction_entities[doc_id]
            gt_entities = ground_truth_entities[doc_id]

            # Extract entity names for each code
            pred_names = {}
            for code in unique_pred_codes:
                numeric_id = get_numeric_id(code)
                if numeric_id in pred_entities:
                    pred_names[code] = pred_entities[numeric_id].lower()
                else:
                    pred_names[code] = code  # Fallback to code itself

            gt_names = {}
            for code in unique_gt_codes:
                numeric_id = get_numeric_id(code)
                if numeric_id in gt_entities:
                    gt_names[code] = gt_entities[numeric_id].lower()
                else:
                    gt_names[code] = code  # Fallback to code itself

            # Match entities using fuzzy matching
            matched_pairs = []
            matched_pred_codes = set()
            matched_gt_codes = set()

            # For each prediction, find best match in ground truth
            for pred_code in unique_pred_codes:
                best_match = None
                best_score = 0
                pred_name = pred_names[pred_code]

                for gt_code in unique_gt_codes:
                    if gt_code in matched_gt_codes:
                        continue  # Skip already matched ground truth

                    gt_name = gt_names[gt_code]
                    score = fuzz.ratio(pred_name, gt_name)

                    if score > threshold and score > best_score:
                        best_match = gt_code
                        best_score = score

                if best_match:
                    # Found a match
                    matched_pairs.append(
                        {
                            "pred_code": pred_code,
                            "gt_code": best_match,
                            "pred_name": pred_name,
                            "gt_name": gt_names[best_match],
                            "score": best_score,
                        }
                    )
                    matched_pred_codes.add(pred_code)
                    matched_gt_codes.add(best_match)

                    # Add to result fuzzy matches
                    result["fuzzy_matches"].append(
                        {
                            "pred_code": pred_code,
                            "gt_code": best_match,
                            "pred_name": pred_name,
                            "gt_name": gt_names[best_match],
                            "score": best_score,
                            "sample_id": doc_id,
                        }
                    )

                    # Add to corpus true positives
                    result["corpus_true_positives"].append(
                        {
                            "code": pred_code,
                            "matched_code": best_match,
                            "sample_id": doc_id,
                            "score": best_score,
                        }
                    )

            # Calculate metrics based on matches
            tp_count = len(matched_pairs)
            fp_count = len(unique_pred_codes) - tp_count
            fn_count = len(unique_gt_codes) - tp_count

            # Add false positives
            for code in unique_pred_codes - matched_pred_codes:
                result["corpus_false_positives"].append(
                    {
                        "code": code,
                        "name": pred_names.get(code, code),
                        "sample_id": doc_id,
                    }
                )

            # Add false negatives
            for code in unique_gt_codes - matched_gt_codes:
                result["corpus_false_negatives"].append(
                    {
                        "code": code,
                        "name": gt_names.get(code, code),
                        "sample_id": doc_id,
                    }
                )

        else:
            # No entity names - fall back to exact code matching
            matched_codes = unique_pred_codes.intersection(unique_gt_codes)
            tp_count = len(matched_codes)
            fp_count = len(unique_pred_codes) - tp_count
            fn_count = len(unique_gt_codes) - tp_count

            # Add true positives to corpus list
            for code in matched_codes:
                result["corpus_true_positives"].append(
                    {"code": code, "sample_id": doc_id}
                )

            # Add false positives to corpus list
            for code in unique_pred_codes - matched_codes:
                result["corpus_false_positives"].append(
                    {"code": code, "sample_id": doc_id}
                )

            # Add false negatives to corpus list
            for code in unique_gt_codes - matched_codes:
                result["corpus_false_negatives"].append(
                    {"code": code, "sample_id": doc_id}
                )

        # Calculate metrics for this sample
        if tp_count + fp_count > 0:
            precision = tp_count / (tp_count + fp_count)
        else:
            precision = 1.0 if tp_count > 0 else 0.0

        if tp_count + fn_count > 0:
            recall = tp_count / (tp_count + fn_count)
        else:
            recall = 1.0 if tp_count > 0 else 0.0

        if precision + recall > 0:
            f1_score = 2 * (precision * recall) / (precision + recall)
        else:
            f1_score = 0.0

        # Store per-sample metrics
        result["per_sample_metrics"][doc_id] = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "tp_count": tp_count,
            "fp_count": fp_count,
            "fn_count": fn_count,
            "has_ground_truth": has_ground_truth,
            "has_entity_names": have_entity_names,
            "method": "fuzzy_matching" if have_entity_names else "code_matching",
        }

        # Update counters
        all_tp_count += tp_count
        all_fp_count += fp_count
        all_fn_count += fn_count

        # Only add to macro-averaging if has ground truth
        if has_ground_truth:
            case_precision_values.append(precision)
            case_recall_values.append(recall)
            case_f1_values.append(f1_score)

    # Calculate corpus-level metrics based on aggregated counts
    if all_tp_count + all_fp_count > 0:
        corpus_precision = all_tp_count / (all_tp_count + all_fp_count)
    else:
        corpus_precision = 1.0 if all_tp_count > 0 else 0.0

    if all_tp_count + all_fn_count > 0:
        corpus_recall = all_tp_count / (all_tp_count + all_fn_count)
    else:
        corpus_recall = 1.0 if all_tp_count > 0 else 0.0

    if corpus_precision + corpus_recall > 0:
        corpus_f1 = (
            2 * (corpus_precision * corpus_recall) / (corpus_precision + corpus_recall)
        )
    else:
        corpus_f1 = 0.0

    # Store corpus-level metrics
    result["corpus_metrics"] = {
        "precision": corpus_precision,
        "recall": corpus_recall,
        "f1_score": corpus_f1,
        "tp_count": all_tp_count,
        "fp_count": all_fp_count,
        "fn_count": all_fn_count,
        "cases_with_ground_truth": len(cases_with_ground_truth),
        "cases_without_ground_truth": len(cases_without_ground_truth),
        "total_cases": len(all_doc_ids),
        "fuzzy_matches_count": len(result["fuzzy_matches"]),
    }

    # Same as corpus metrics for micro-averaging
    result["micro_averaging_metrics"] = {
        "precision": corpus_precision,
        "recall": corpus_recall,
        "f1_score": corpus_f1,
        "tp_count": all_tp_count,
        "fp_count": all_fp_count,
        "fn_count": all_fn_count,
        "description": "Micro-averaging: Sum of TP, FP, FN across all cases",
    }

    # Calculate macro-averaging metrics
    if case_precision_values:
        result["macro_averaging_metrics"] = {
            "precision": float(np.mean(case_precision_values)),
            "recall": float(np.mean(case_recall_values)),
            "f1_score": float(np.mean(case_f1_values)),
            "precision_std": float(np.std(case_precision_values)),
            "recall_std": float(np.std(case_recall_values)),
            "f1_score_std": float(np.std(case_f1_values)),
            "case_count": len(case_precision_values),
            "description": "Macro-averaging: Metrics calculated per case, then averaged",
        }
    else:
        result["macro_averaging_metrics"] = {
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "precision_std": 0.0,
            "recall_std": 0.0,
            "f1_score_std": 0.0,
            "case_count": 0,
            "description": "Macro-averaging: Metrics calculated per case, then averaged",
        }

    # Store lists of cases
    result["cases_with_ground_truth"] = cases_with_ground_truth
    result["cases_without_ground_truth"] = cases_without_ground_truth

    return result


def extract_entity_mappings(
    predictions_data: Dict, ground_truth_data: Dict
) -> Tuple[Dict[str, Dict[str, str]], Dict[str, Dict[str, str]]]:
    """
    Extract entity name to ORPHA code mappings from both predictions and ground truth.

    Args:
        predictions_data: Dictionary containing prediction results
        ground_truth_data: Dictionary containing ground truth data

    Returns:
        Tuple of (
            Dictionary mapping {doc_id: {numeric_orpha_id: entity_name}} for predictions,
            Dictionary mapping {doc_id: {numeric_orpha_id: entity_name}} for ground truth
        )
    """
    import re

    # Helper function to extract numeric ID
    def get_numeric_id(code: str) -> str:
        if not code:
            return ""
        match = re.search(r"(\d+)", code)
        return match.group(1) if match else code

    # Initialize mappings
    prediction_entities = {}
    ground_truth_entities = {}

    # Process predictions
    if predictions_data:
        # Handle direct predictions format or nested results format
        pred_results = predictions_data.get("results", predictions_data)
        for doc_id, doc_data in pred_results.items():
            if isinstance(doc_data, dict) and "matched_diseases" in doc_data:
                entities = {}
                for match in doc_data["matched_diseases"]:
                    if isinstance(match, dict):
                        # Extract entity and ORPHA ID
                        orpha_id = match.get("orpha_id", "")
                        entity = match.get("entity", "")

                        if orpha_id and entity:
                            # Extract just the numeric part for the key
                            numeric_id = get_numeric_id(orpha_id)
                            if numeric_id:
                                # Use original_entity if available, otherwise entity
                                if (
                                    "original_entity" in match
                                    and match["original_entity"]
                                ):
                                    entities[numeric_id] = match["original_entity"]
                                else:
                                    entities[numeric_id] = entity

                if entities:
                    prediction_entities[str(doc_id)] = entities

    # Process ground truth
    if ground_truth_data:
        # Handle both old and new formats
        gt_data = ground_truth_data
        if "documents" in ground_truth_data:
            gt_data = ground_truth_data["documents"]

        for doc_id, doc_data in gt_data.items():
            entities = {}

            # Format 1: documents with annotations
            if isinstance(doc_data, dict) and "annotations" in doc_data:
                for ann in doc_data["annotations"]:
                    if isinstance(ann, dict) and "mention" in ann:
                        mention = ann.get("mention", "")

                        # Get ORPHA ID from ordo_with_desc field
                        orpha_id = ""
                        if "ordo_with_desc" in ann:
                            ordo_parts = ann["ordo_with_desc"].split(" ", 1)
                            orpha_id = ordo_parts[0] if ordo_parts else ""
                        elif "orpha_id" in ann:
                            orpha_id = ann["orpha_id"]

                        if orpha_id and mention:
                            # Extract just the numeric part for the key
                            numeric_id = get_numeric_id(orpha_id)
                            if numeric_id:
                                entities[numeric_id] = mention

            # Format 2: documents with gold_annotations
            elif isinstance(doc_data, dict) and "gold_annotations" in doc_data:
                gold_anns = doc_data["gold_annotations"]

                if isinstance(gold_anns, list):
                    for ann in gold_anns:
                        if isinstance(ann, dict) and "mention" in ann:
                            mention = ann.get("mention", "")
                            orpha_id = ann.get("orpha_id", "")

                            if orpha_id and mention:
                                numeric_id = get_numeric_id(orpha_id)
                                if numeric_id:
                                    entities[numeric_id] = mention

            if entities:
                ground_truth_entities[str(doc_id)] = entities

    # Debug info about extraction
    print(f"Extracted entity mappings:")
    print(f"  Prediction documents with entity mappings: {len(prediction_entities)}")
    print(
        f"  Ground truth documents with entity mappings: {len(ground_truth_entities)}"
    )

    # Sample of entity mappings (first document for each)
    if prediction_entities:
        first_pred_doc = next(iter(prediction_entities))
        first_pred_entities = prediction_entities[first_pred_doc]
        print(f"  Sample prediction entity mappings (doc {first_pred_doc}):")
        for i, (code, name) in enumerate(list(first_pred_entities.items())[:3]):
            print(f"    {i+1}. ORPHA:{code} -> '{name}'")

    if ground_truth_entities:
        first_gt_doc = next(iter(ground_truth_entities))
        first_gt_entities = ground_truth_entities[first_gt_doc]
        print(f"  Sample ground truth entity mappings (doc {first_gt_doc}):")
        for i, (code, name) in enumerate(list(first_gt_entities.items())[:3]):
            print(f"    {i+1}. ORPHA:{code} -> '{name}'")

    return prediction_entities, ground_truth_entities


def code_based_evaluation(predictions: List[str], ground_truth: List[str]) -> Dict:
    """
    Evaluates predictions against ground truth using count-based exact matching of ORPHA codes.
    Counts all occurrences rather than using set-based deduplication.

    Args:
        predictions: List of predicted ORPHA codes
        ground_truth: List of ground truth ORPHA codes

    Returns:
        Dictionary with precision, recall, F1 scores and match information
    """
    import re
    from collections import Counter

    # Extract only numeric part of ORPHA codes
    def normalize_code(code):
        if not code:
            return ""
        # Use regex to extract only the digits
        match = re.search(r"(\d+)", code)
        if match:
            return match.group(1)
        return code

    # Normalize predictions and ground truth
    normalized_predictions = [normalize_code(p) for p in predictions if p]
    normalized_ground_truth = [normalize_code(g) for g in ground_truth if g]

    # Debug info if available
    if predictions and ground_truth:
        print(f"\nDebug: Sample normalization:")
        print(f"  Original prediction: '{predictions[0]}'")
        print(f"  Normalized prediction: '{normalize_code(predictions[0])}'")
        print(f"  Original ground truth: '{ground_truth[0]}'")
        print(f"  Normalized ground truth: '{normalize_code(ground_truth[0])}'")

    # Count all occurrences using Counter
    pred_counter = Counter(normalized_predictions)
    truth_counter = Counter(normalized_ground_truth)

    # Initialize result structure
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
        "total_pred_count": len(normalized_predictions),
        "total_truth_count": len(normalized_ground_truth),
        "unique_pred_count": len(set(normalized_predictions)),
        "unique_truth_count": len(set(normalized_ground_truth)),
    }

    # Handle empty cases
    if not normalized_predictions or not normalized_ground_truth:
        if not normalized_predictions:
            result["precision"] = 1.0  # No false positives

        # Add all predictions as false positives
        for code, count in pred_counter.items():
            result["false_positives"].append({"code": code, "count": count})
            result["fp_count"] += count

        # Add all ground truth as false negatives
        for code, count in truth_counter.items():
            result["false_negatives"].append({"code": code, "count": count})
            result["fn_count"] += count

        return result

    # Count-based matching for true positives, false positives, and false negatives
    tp_count = 0
    for code, truth_count in truth_counter.items():
        pred_count = pred_counter.get(code, 0)
        # True positives: minimum of prediction count and truth count for this code
        match_count = min(pred_count, truth_count)
        if match_count > 0:
            result["true_positives"].append({"code": code, "count": match_count})
            tp_count += match_count

    # Count total false positives (predictions - matches)
    fp_count = 0
    for code, pred_count in pred_counter.items():
        truth_count = truth_counter.get(code, 0)
        # False positives: prediction count - match count (which is min of pred and truth)
        excess_count = max(0, pred_count - truth_count)
        if excess_count > 0:
            result["false_positives"].append({"code": code, "count": excess_count})
            fp_count += excess_count

    # Count total false negatives (truth - matches)
    fn_count = 0
    for code, truth_count in truth_counter.items():
        pred_count = pred_counter.get(code, 0)
        # False negatives: truth count - match count (which is min of pred and truth)
        excess_count = max(0, truth_count - pred_count)
        if excess_count > 0:
            result["false_negatives"].append({"code": code, "count": excess_count})
            fn_count += excess_count

    # Store total counts
    result["tp_count"] = tp_count
    result["fp_count"] = fp_count
    result["fn_count"] = fn_count

    # Calculate metrics
    total_predictions = tp_count + fp_count
    total_ground_truth = tp_count + fn_count

    # Calculate precision and recall
    if total_predictions > 0:
        result["precision"] = tp_count / total_predictions
    else:
        result["precision"] = 1.0  # No predictions means no false positives

    if total_ground_truth > 0:
        result["recall"] = tp_count / total_ground_truth
    else:
        result["recall"] = 1.0  # No ground truth means no false negatives

    # Calculate F1 score
    if result["precision"] + result["recall"] > 0:
        result["f1_score"] = (
            2
            * (result["precision"] * result["recall"])
            / (result["precision"] + result["recall"])
        )

    return result


def load_data(
    predictions_file: str, ground_truth_file: str
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Load prediction and ground truth data from files.

    Args:
        predictions_file: Path to predictions JSON file
        ground_truth_file: Path to ground truth JSON file

    Returns:
        Tuple of (predictions_data, ground_truth_data) where each is the raw loaded JSON
    """
    try:
        with open(predictions_file, "r") as f:
            predictions = json.load(f)
            print(f"Successfully loaded predictions from {predictions_file}")
    except Exception as e:
        print(f"Error loading predictions file: {e}")
        raise

    try:
        with open(ground_truth_file, "r") as f:
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


def extract_predictions(
    predictions_data: Dict,
    match_method: Optional[str] = None,
    confidence_threshold: Optional[float] = None,
) -> Dict[str, List[str]]:
    """
    Extract ORPHA codes from the predictions data structure with filtering options.

    This function will include all cases in the result, even those with no matched diseases,
    to ensure comprehensive evaluation coverage.

    Args:
        predictions_data (Dict): Dictionary of prediction results
        match_method (Optional[str]): Optional filter for match method (e.g., 'exact', 'llm')
        confidence_threshold (Optional[float]): Optional minimum confidence score threshold

    Returns:
        Dict[str, List[str]]: Dictionary mapping case IDs to lists of ORPHA codes
    """
    # Input validation
    if not isinstance(predictions_data, dict):
        print(f"Warning: Unexpected predictions data format: {type(predictions_data)}")
        return {}

    result = {}
    total_cases_processed = 0
    total_orpha_codes_extracted = 0

    for case_id, case_data in predictions_data.items():
        total_cases_processed += 1
        orpha_codes = []

        # Check for matched diseases field, but proceed even if it's empty
        if "matched_diseases" in case_data and isinstance(
            case_data["matched_diseases"], list
        ):
            matched_diseases = case_data["matched_diseases"]

            for item in matched_diseases:

                # Validate item type
                if not isinstance(item, dict):
                    continue

                # Apply match method filter if specified
                if match_method and item.get("match_method") != match_method:
                    continue

                # Apply confidence threshold filter if specified
                if confidence_threshold is not None:
                    confidence = item.get("confidence_score", 0.0)
                    if confidence < confidence_threshold:
                        continue

                # Extract the ORPHA code
                orpha_id = item.get("orpha_id")
                if orpha_id and isinstance(orpha_id, str):
                    orpha_codes.append(orpha_id)
                    total_orpha_codes_extracted += 1

        # Always add the case to the result, even with an empty list of ORPHA codes
        result[str(case_id)] = orpha_codes

    # Print comprehensive statistics
    print(f"Processed {total_cases_processed} total cases")
    print(f"Extracted predictions for {len(result)} cases")
    print(f"Total ORPHA codes extracted: {total_orpha_codes_extracted}")
    print(
        f"Cases with matched ORPHA codes: {sum(1 for codes in result.values() if codes)}"
    )
    print(
        f"Cases with zero matched ORPHA codes: {sum(1 for codes in result.values() if not codes)}"
    )

    return result


def extract_prediction_entities(
    predictions_data: Dict,
) -> Tuple[Dict[str, List[str]], Dict[str, Dict[str, str]]]:
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

            if "matched_diseases" in case_data and isinstance(
                case_data["matched_diseases"], list
            ):
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
                            normalized_id = (
                                orpha_id.replace("ORPHA:", "").strip().lower()
                            )
                            entity_names[normalized_id] = entity

            if orpha_codes:
                orpha_codes_dict[str(case_id)] = orpha_codes
                entity_names_dict[str(case_id)] = entity_names

    return orpha_codes_dict, entity_names_dict


def extract_ground_truth_entities(
    ground_truth_data: Dict,
) -> Tuple[Dict[str, List[str]], Dict[str, Dict[str, str]]]:
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
        if "documents" in ground_truth_data:
            ground_truth_data = ground_truth_data["documents"]
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
                            ordo_parts = ordo_field.split(" ", 1)
                            orpha_id = ordo_parts[0] if ordo_parts else ""

                            if orpha_id:
                                # Normalize format
                                if not orpha_id.startswith("ORPHA:"):
                                    orpha_id = f"ORPHA:{orpha_id}"

                                orpha_codes.append(orpha_id)

                                # Store entity name mapping
                                if mention:
                                    normalized_id = (
                                        orpha_id.replace("ORPHA:", "").strip().lower()
                                    )
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
        if "documents" in ground_truth_data.keys():
            ground_truth_data = ground_truth_data["documents"]
        for case_id, case_data in ground_truth_data.items():
            orpha_codes = []

            # Check for MIMIC-style format with note_details and annotations
            if isinstance(case_data, dict) and "annotations" in case_data:
                annotations = case_data["annotations"]
                total_annotations += len(annotations)

                if isinstance(annotations, list):
                    for annotation in annotations:
                        if (
                            isinstance(annotation, dict)
                            and "ordo_with_desc" in annotation
                        ):
                            ordo_field = annotation["ordo_with_desc"]

                            # Extract ORPHA ID from the ordo_with_desc field
                            if ordo_field and isinstance(ordo_field, str):
                                # Keep original format for now, normalization happens in evaluation
                                orpha_codes.append(
                                    f"ORPHA:{ordo_field.split(' ', 1)[0]}"
                                )

            # Add non-empty lists to result

            result[str(case_id)] = orpha_codes

    print(f"Found {total_annotations} total annotations in data")
    print(
        f"Extracted ground truth for {len(result)} cases with {sum(len(codes) for codes in result.values())} total ORPHA codes"
    )
    return result


def evaluate_corpus(
    predictions_dict: Dict[str, List[str]], ground_truth_dict: Dict[str, List[str]]
) -> Dict:
    """
    Evaluate predictions against ground truth across the entire corpus using count-based evaluation.

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
        "corpus_false_negatives": [],
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

        # Evaluate this sample using count-based evaluation
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
            "has_ground_truth": has_ground_truth,
        }

        # Add detailed match information
        result["per_sample_metrics"][sample_id]["true_positives"] = sample_result[
            "true_positives"
        ]
        result["per_sample_metrics"][sample_id]["false_positives"] = sample_result[
            "false_positives"
        ]
        result["per_sample_metrics"][sample_id]["false_negatives"] = sample_result[
            "false_negatives"
        ]

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

            # Count-based - accumulate raw counts
            total_tp += sample_result["tp_count"]
            total_fp += sample_result["fp_count"]
            total_fn += sample_result["fn_count"]

            # Macro-averaging - collect metrics for averaging
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

    # Corpus-level pooling (micro-averaging)
    micro_result = code_based_evaluation(all_predictions_with_gt, all_ground_truth)

    # Count-based metrics (these are now actual count-based, not set-based)
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
        count_based_metrics["f1_score"] = (
            2
            * (count_based_metrics["precision"] * count_based_metrics["recall"])
            / (count_based_metrics["precision"] + count_based_metrics["recall"])
        )
    else:
        count_based_metrics["f1_score"] = 0.0

    count_based_metrics["tp_count"] = total_tp
    count_based_metrics["fp_count"] = total_fp
    count_based_metrics["fn_count"] = total_fn

    # Macro-averaging metrics
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
        "description": "Micro-averaging: All ORPHA codes pooled together across cases",
    }

    result["macro_averaging_metrics"] = macro_metrics
    result["macro_averaging_metrics"][
        "description"
    ] = "Macro-averaging: Metrics calculated per case, then averaged"

    result["count_based_metrics"] = count_based_metrics
    result["count_based_metrics"][
        "description"
    ] = "Count-based: TP, FP, FN counts summed across cases, then metrics calculated"

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
        "total_predictions_all_cases": len(all_predictions_total),
    }

    # Track counts and lists of cases
    result["cases_with_ground_truth"] = cases_with_ground_truth
    result["cases_without_ground_truth"] = cases_without_ground_truth

    # Add additional info about false positives from cases without ground truth
    fps_no_ground_truth = [
        fp
        for fp in result["corpus_false_positives"]
        if fp["sample_id"] in cases_without_ground_truth
    ]
    result["corpus_metrics"]["fps_from_cases_without_ground_truth"] = len(
        fps_no_ground_truth
    )

    # Add statistical summaries for cases WITH ground truth only
    valid_metrics = {
        k: v for k, v in result["per_sample_metrics"].items() if v["has_ground_truth"]
    }
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
        "NOTE: This evaluation now uses COUNT-BASED matching of ORPHA codes, including all occurrences",
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
    precision_values = [
        m["precision"] for m in per_sample_metrics.values() if "precision" in m
    ]
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
            f"{name}_samples": len(values_array),
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
    analysis = {"most_common_false_positives": [], "most_common_false_negatives": []}

    # Count frequency of false positives and negatives
    fp_counter = Counter()
    fn_counter = Counter()

    for fp in result["corpus_false_positives"]:
        fp_counter[fp["code"]] += 1

    for fn in result["corpus_false_negatives"]:
        fn_counter[fn["code"]] += 1

    # Get most common errors
    analysis["most_common_false_positives"] = [
        {"code": code, "count": count} for code, count in fp_counter.most_common(20)
    ]

    analysis["most_common_false_negatives"] = [
        {"code": code, "count": count} for code, count in fn_counter.most_common(20)
    ]

    return analysis


def print_evaluation_summary(result: Dict) -> None:
    """
    Print a summary of the evaluation results.

    Args:
        result: Dictionary with corpus evaluation results
    """
    print("\n=== ORPHA Code Evaluation Summary ===")
    print(
        "NOTE: This evaluation uses EXACT MATCHING of ORPHA codes, not fuzzy string matching of disease names"
    )
    print(
        f"Evaluating {result.get('total_cases_evaluated', 0)} cases from predictions file"
    )

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
    print(
        f"  Total Predictions (all cases): {corpus.get('total_predictions_all_cases', 0)}"
    )
    print(
        f"  Predictions in cases with ground truth: {corpus.get('total_pred_count', 0)}"
    )
    print(
        f"  Unique predictions in cases with ground truth: {corpus.get('unique_pred_count', 0)}"
    )
    print(f"  Total Ground Truth items: {corpus.get('total_truth_count', 0)}")
    print(f"  Unique Ground Truth items: {corpus.get('unique_truth_count', 0)}")

    # Print statistics
    stats = result["statistics"]
    if stats:
        print("\nPer-Sample Statistics (cases with ground truth only):")
        print(f"  Samples with ground truth: {stats.get('precision_samples', 0)}")
        print(
            f"  F1 Score: mean={stats.get('f1_mean', 0):.4f}, median={stats.get('f1_median', 0):.4f}, std={stats.get('f1_std', 0):.4f}"
        )
        print(
            f"  Precision: mean={stats.get('precision_mean', 0):.4f}, median={stats.get('precision_median', 0):.4f}"
        )
        print(
            f"  Recall: mean={stats.get('recall_mean', 0):.4f}, median={stats.get('recall_median', 0):.4f}"
        )
    else:
        print(
            "\nNo per-sample statistics available (no samples with both predictions and ground truth)"
        )

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
    if corpus.get("tp_count", 0) == 0 and (
        corpus.get("fp_count", 0) > 0 or corpus.get("fn_count", 0) > 0
    ):
        print(
            "\nWARNING: No true positive matches found between predictions and ground truth."
        )
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
    debug: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate predictions using count-based fuzzy matching on entity names.
    Includes all documents from ground truth, even if predictions don't exist.

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
    from collections import Counter

    # Helper function to extract numeric ID
    def get_numeric_id(code: str) -> str:
        match = re.search(r"(\d+)", code)
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
        "unmatched_details": {},
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

    # Process all documents from both predictions and ground truth
    all_doc_ids = set(predictions_dict.keys()) | set(ground_truth_dict.keys())

    if debug:
        print(f"Total documents to evaluate: {len(all_doc_ids)}")
        print(f"Documents in predictions: {len(predictions_dict)}")
        print(f"Documents in ground truth: {len(ground_truth_dict)}")
        print(
            f"Documents only in ground truth: {len(set(ground_truth_dict.keys()) - set(predictions_dict.keys()))}"
        )
        print(
            f"Documents only in predictions: {len(set(predictions_dict.keys()) - set(ground_truth_dict.keys()))}"
        )

    # Process each document
    for doc_id in sorted(all_doc_ids):
        pred_codes = predictions_dict.get(doc_id, [])
        gt_codes = ground_truth_dict.get(doc_id, [])

        # Create Counters to track occurrence counts
        pred_counter = Counter(pred_codes)
        gt_counter = Counter(gt_codes)

        # Track if this case has ground truth
        has_ground_truth = len(gt_codes) > 0

        # Update tracking lists
        if has_ground_truth:
            result["cases_with_ground_truth"].append(doc_id)
        else:
            result["cases_without_ground_truth"].append(doc_id)

        # Initialize unmatched details for this document
        result["unmatched_details"][doc_id] = {
            "false_negatives": [],
            "false_positives": [],
        }

        # Special case: No ground truth for this document
        if not gt_codes:
            # All predictions are false positives
            if pred_codes and doc_id in prediction_entities:
                # Add all predictions as false positives with their counts
                pred_entities_for_doc = prediction_entities.get(doc_id, {})

                for code, count in pred_counter.items():
                    result["unmatched_details"][doc_id]["false_positives"].append(
                        {
                            "name": pred_entities_for_doc.get(
                                get_numeric_id(code), code
                            ),
                            "orpha_code": code,
                            "count": count,
                        }
                    )

                # Count metrics - all false positives
                sample_result = {
                    "tp_count": 0,
                    "fp_count": len(pred_codes),  # Count all occurrences
                    "fn_count": 0,
                    "precision": 0.0,
                    "recall": 1.0,  # No false negatives, so recall is 1.0
                    "f1_score": 0.0,
                }

                all_fp_count += len(pred_codes)
            else:
                # No predictions and no ground truth
                sample_result = {
                    "tp_count": 0,
                    "fp_count": 0,
                    "fn_count": 0,
                    "precision": 1.0,  # No false positives
                    "recall": 1.0,  # No false negatives
                    "f1_score": (
                        1.0 if pred_codes else 0.0
                    ),  # 1.0 if predictions exist, 0.0 otherwise
                }

            result["per_sample_metrics"][doc_id] = sample_result
            continue

        # Special case: No predictions for this document
        if not pred_codes:
            # All ground truth items are false negatives
            if gt_codes and doc_id in ground_truth_entities:
                # Add all ground truth items as false negatives with their counts
                gt_entities_for_doc = ground_truth_entities.get(doc_id, {})

                for code, count in gt_counter.items():
                    result["unmatched_details"][doc_id]["false_negatives"].append(
                        {
                            "name": gt_entities_for_doc.get(get_numeric_id(code), code),
                            "orpha_code": code,
                            "count": count,
                        }
                    )

                # Count metrics - all false negatives
                sample_result = {
                    "tp_count": 0,
                    "fp_count": 0,
                    "fn_count": len(gt_codes),  # Count all occurrences
                    "precision": 1.0,  # No false positives
                    "recall": 0.0,  # All false negatives
                    "f1_score": 0.0,
                }

                all_fn_count += len(gt_codes)
            else:
                # Empty case
                sample_result = {
                    "tp_count": 0,
                    "fp_count": 0,
                    "fn_count": 0,
                    "precision": 1.0,
                    "recall": 1.0,
                    "f1_score": 0.0,
                }

            result["per_sample_metrics"][doc_id] = sample_result

            # For macro-averaging
            if has_ground_truth:
                case_precision_values.append(sample_result["precision"])
                case_recall_values.append(sample_result["recall"])
                case_f1_values.append(sample_result["f1_score"])

            continue

        # Check if we have entity names for fuzzy matching
        have_entity_names = (doc_id in prediction_entities) and (
            doc_id in ground_truth_entities
        )

        if not have_entity_names:
            if debug:
                print(
                    f"  No entity names available for document {doc_id} - using code-based evaluation"
                )

            # Fallback to count-based code evaluation for this document
            code_result = code_based_evaluation(pred_codes, gt_codes)

            # Add fallback results with counts
            for fp in code_result["false_positives"]:
                result["unmatched_details"][doc_id]["false_positives"].append(
                    {"name": fp["code"], "orpha_code": fp["code"], "count": fp["count"]}
                )

            for fn in code_result["false_negatives"]:
                result["unmatched_details"][doc_id]["false_negatives"].append(
                    {"name": fn["code"], "orpha_code": fn["code"], "count": fn["count"]}
                )

            # Count metrics
            sample_result = {
                "tp_count": code_result["tp_count"],
                "fp_count": code_result["fp_count"],
                "fn_count": code_result["fn_count"],
                "precision": code_result["precision"],
                "recall": code_result["recall"],
                "f1_score": code_result["f1_score"],
            }

            result["per_sample_metrics"][doc_id] = sample_result

            # Update aggregate counts
            all_tp_count += sample_result["tp_count"]
            all_fp_count += sample_result["fp_count"]
            all_fn_count += sample_result["fn_count"]

            # For macro-averaging
            case_precision_values.append(sample_result["precision"])
            case_recall_values.append(sample_result["recall"])
            case_f1_values.append(sample_result["f1_score"])

            continue

        # Get entity names
        pred_entities_for_doc = prediction_entities.get(doc_id, {})
        gt_entities_for_doc = ground_truth_entities.get(doc_id, {})

        # Map entity names to ORPHA codes with counts
        pred_entity_to_codes = {}
        for code in pred_codes:
            numeric_id = get_numeric_id(code)
            if numeric_id in pred_entities_for_doc:
                entity_name = pred_entities_for_doc[numeric_id].lower()
                if entity_name not in pred_entity_to_codes:
                    pred_entity_to_codes[entity_name] = []
                pred_entity_to_codes[entity_name].append(code)

        gt_entity_to_codes = {}
        for code in gt_codes:
            numeric_id = get_numeric_id(code)
            if numeric_id in gt_entities_for_doc:
                entity_name = gt_entities_for_doc[numeric_id].lower()
                if entity_name not in gt_entity_to_codes:
                    gt_entity_to_codes[entity_name] = []
                gt_entity_to_codes[entity_name].append(code)

        # Create entity name counters
        pred_entity_counter = {
            name: len(codes) for name, codes in pred_entity_to_codes.items()
        }
        gt_entity_counter = {
            name: len(codes) for name, codes in gt_entity_to_codes.items()
        }

        # Initialize counters for fuzzy matching
        tp_count = 0
        fp_count = 0
        fn_count = 0

        # Track fuzzy matches
        fuzzy_matches = []
        matched_gt_entities = set()

        # For each predicted entity, try to find matching ground truth entity
        for pred_entity, pred_count in pred_entity_counter.items():
            best_match = None
            best_score = 0
            best_match_count = 0

            # Try each ground truth entity for fuzzy matching
            for gt_entity, gt_count in gt_entity_counter.items():
                if gt_entity in matched_gt_entities:
                    # Skip already fully matched ground truth entities
                    continue

                # Calculate fuzzy match score
                score = fuzz.ratio(pred_entity, gt_entity)

                if score > threshold and score > best_score:
                    best_score = score
                    best_match = gt_entity
                    best_match_count = gt_count

            # Found a match
            if best_match:
                # Calculate match count based on counts of both entities
                match_count = min(pred_count, best_match_count)
                tp_count += match_count

                # Record false positives (excess predictions)
                if pred_count > match_count:
                    excess_count = pred_count - match_count
                    fp_count += excess_count

                    # Add to unmatched details
                    result["unmatched_details"][doc_id]["false_positives"].append(
                        {
                            "name": pred_entity,
                            "orpha_code": pred_entity_to_codes[pred_entity][
                                0
                            ],  # Take first code as representative
                            "count": excess_count,
                        }
                    )

                # Record match for debugging
                fuzzy_matches.append(
                    {
                        "pred_name": pred_entity,
                        "gt_name": best_match,
                        "score": best_score,
                        "count": match_count,
                    }
                )

                # Mark ground truth entity as matched for its matched count
                gt_entity_counter[best_match] -= match_count

                # If fully matched, add to matched set
                if gt_entity_counter[best_match] <= 0:
                    matched_gt_entities.add(best_match)
            else:
                # No match found - all predicted occurrences are false positives
                fp_count += pred_count

                # Add to unmatched details
                result["unmatched_details"][doc_id]["false_positives"].append(
                    {
                        "name": pred_entity,
                        "orpha_code": pred_entity_to_codes[pred_entity][
                            0
                        ],  # Take first code as representative
                        "count": pred_count,
                    }
                )

        # Count remaining ground truth entities as false negatives
        for gt_entity, remaining_count in gt_entity_counter.items():
            if remaining_count > 0:
                fn_count += remaining_count

                # Add to unmatched details
                result["unmatched_details"][doc_id]["false_negatives"].append(
                    {
                        "name": gt_entity,
                        "orpha_code": gt_entity_to_codes[gt_entity][
                            0
                        ],  # Take first code as representative
                        "count": remaining_count,
                    }
                )

        # Calculate metrics
        precision = (
            tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 1.0
        )
        recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 1.0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        # Store results
        sample_result = {
            "tp_count": tp_count,
            "fp_count": fp_count,
            "fn_count": fn_count,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "fuzzy_matches": fuzzy_matches[:10],  # Store up to 10 matches for brevity
        }

        result["per_sample_metrics"][doc_id] = sample_result

        # Update aggregate metrics
        all_tp_count += tp_count
        all_fp_count += fp_count
        all_fn_count += fn_count

        # For macro-averaging
        case_precision_values.append(precision)
        case_recall_values.append(recall)
        case_f1_values.append(f1)

        # Store some matches for debugging
        all_fuzzy_matches.extend(fuzzy_matches[:5])

    # Compute micro-averaging metrics
    micro_precision = (
        all_tp_count / (all_tp_count + all_fp_count)
        if (all_tp_count + all_fp_count) > 0
        else 1.0
    )
    micro_recall = (
        all_tp_count / (all_tp_count + all_fn_count)
        if (all_tp_count + all_fn_count) > 0
        else 1.0
    )
    micro_f1 = (
        2 * (micro_precision * micro_recall) / (micro_precision + micro_recall)
        if (micro_precision + micro_recall) > 0
        else 0.0
    )

    result["micro_averaging_metrics"] = {
        "precision": micro_precision,
        "recall": micro_recall,
        "f1_score": micro_f1,
        "tp_count": all_tp_count,
        "fp_count": all_fp_count,
        "fn_count": all_fn_count,
        "description": "Micro-averaging: All matches pooled together across cases",
    }

    # Compute macro-averaging metrics
    macro_precision = np.mean(case_precision_values) if case_precision_values else 0.0
    macro_recall = np.mean(case_recall_values) if case_recall_values else 0.0
    macro_f1 = np.mean(case_f1_values) if case_f1_values else 0.0

    result["macro_averaging_metrics"] = {
        "precision": macro_precision,
        "recall": macro_recall,
        "f1_score": macro_f1,
        "precision_std": (
            np.std(case_precision_values) if case_precision_values else 0.0
        ),
        "recall_std": np.std(case_recall_values) if case_recall_values else 0.0,
        "f1_score_std": np.std(case_f1_values) if case_f1_values else 0.0,
        "case_count": len(case_precision_values),
        "description": "Macro-averaging: Metrics calculated per case, then averaged",
    }

    # Count-based metrics
    result["count_based_metrics"] = {
        "precision": micro_precision,
        "recall": micro_recall,
        "f1_score": micro_f1,
        "tp_count": all_tp_count,
        "fp_count": all_fp_count,
        "fn_count": all_fn_count,
        "description": "Count-based: TP, FP, FN counts summed across cases, then metrics calculated",
    }

    # Corpus metrics (using micro-averaging)
    result["corpus_metrics"] = result["micro_averaging_metrics"].copy()
    result["corpus_metrics"].update(
        {
            "cases_with_ground_truth": len(result["cases_with_ground_truth"]),
            "cases_without_ground_truth": len(result["cases_without_ground_truth"]),
            "total_cases": len(all_doc_ids),
            "docs_only_in_ground_truth": len(
                set(ground_truth_dict.keys()) - set(predictions_dict.keys())
            ),
            "docs_only_in_predictions": len(
                set(predictions_dict.keys()) - set(ground_truth_dict.keys())
            ),
        }
    )

    # Top-level metrics for compatibility
    result["precision"] = micro_precision
    result["recall"] = micro_recall
    result["f1_score"] = micro_f1
    result["tp_count"] = all_tp_count
    result["fp_count"] = all_fp_count
    result["fn_count"] = all_fn_count
    result["total_matches_found"] = sum(
        match.get("count", 1) for match in all_fuzzy_matches
    )
    result["total_documents_evaluated"] = len(all_doc_ids)

    # Notes about the evaluation
    result["notes"] = [
        f"Count-based fuzzy matching approach using threshold of {threshold}",
        "Three approaches to metric calculation:",
        " 1. Micro-averaging: All matches pooled together across cases",
        " 2. Macro-averaging: Metrics calculated per case, then averaged",
        " 3. Count-based: TP, FP, FN counts summed across cases, then metrics calculated",
        "All documents from both predictions and ground truth are included in evaluation",
        "Documents without entity names fall back to code-based matching",
        "Entity counts are preserved for more accurate count-based evaluation",
    ]

    # Debug output if requested
    if debug and all_fuzzy_matches:
        print("\n===== FUZZY MATCHING SUMMARY =====")
        print("Top 10 fuzzy matches:")
        sorted_matches = sorted(
            all_fuzzy_matches, key=lambda x: x["score"], reverse=True
        )
        for i, match in enumerate(sorted_matches[:10], 1):
            count_str = f", count: {match.get('count', 1)}" if "count" in match else ""
            print(
                f"  {i}. '{match['pred_name']}' matched with '{match['gt_name']}' (score: {match['score']}{count_str})"
            )

    return result


def main():
    """Main function to run the set-based evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate ORPHA code predictions across a corpus using set-based evaluation"
    )

    # Required arguments
    parser.add_argument(
        "--predictions", required=True, help="Path to JSON file with predictions"
    )
    parser.add_argument(
        "--ground-truth", required=True, help="Path to JSON file with ground truth"
    )
    parser.add_argument(
        "--output", required=True, help="Path to save evaluation results JSON"
    )

    # Optional filtering arguments
    parser.add_argument(
        "--match-method",
        type=str,
        choices=["exact", "llm"],
        help="Filter predictions by match method ('exact' or 'llm')",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        help="Filter predictions by minimum confidence score (0.0-1.0)",
    )

    # Fuzzy matching configuration
    parser.add_argument(
        "--fuzzy-threshold",
        type=int,
        default=90,
        help="Threshold for fuzzy matching (0-100, default: 90)",
    )
    parser.add_argument(
        "--enable-fuzzy",
        action="store_true",
        help="Enable fuzzy matching of disease entity names",
    )
    parser.add_argument(
        "--debug-fuzzy",
        action="store_true",
        help="Enable detailed debug output for fuzzy matching",
    )

    # Output control arguments
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only print summary, don't save detailed JSON",
    )
    parser.add_argument(
        "--save-csv", type=str, help="Path to save summary results as CSV"
    )

    args = parser.parse_args()

    # Load data
    print(f"Loading predictions from {args.predictions}")
    print(f"Loading ground truth from {args.ground_truth}")
    predictions_data, ground_truth_data = load_data(args.predictions, args.ground_truth)

    # Handle nested structure in ground truth
    if isinstance(ground_truth_data, dict) and "documents" in ground_truth_data:
        ground_truth_data = ground_truth_data["documents"]

    # Apply any filtering from command-line arguments
    predictions_dict = extract_predictions(
        predictions_data,
        match_method=args.match_method,
        confidence_threshold=args.confidence_threshold,
    )

    ground_truth_dict = extract_ground_truth(ground_truth_data)

    # Extract entity names for both exact and fuzzy evaluations
    print(f"\nExtracting entity names from predictions and ground truth...")
    prediction_entities, _ = extract_entity_mappings(predictions_data, {})
    _, ground_truth_entities = extract_entity_mappings({}, ground_truth_data)

    print(
        f"Extracted entity names for {len(prediction_entities)} prediction docs and {len(ground_truth_entities)} ground truth docs"
    )

    # Print debug info about document ID matching
    pred_ids = set(predictions_dict.keys())
    gt_ids = set(ground_truth_dict.keys())
    common_ids = pred_ids & gt_ids

    print(f"\nDocument ID matching:")
    print(f"  Prediction document IDs: {len(pred_ids)}")
    print(f"  Ground truth document IDs: {len(gt_ids)}")
    print(f"  Common document IDs: {len(common_ids)}")

    if len(common_ids) == 0:
        print(
            "\nWARNING: No common document IDs found between predictions and ground truth!"
        )
        print("Sample prediction IDs:", list(pred_ids)[:5])
        print("Sample ground truth IDs:", list(gt_ids)[:5])

    # Run set-based code evaluation with entity names
    print(f"\nRunning set-based code evaluation (numeric ORPHA ID matching)...")
    exact_result = set_based_code_evaluation(
        predictions_dict, ground_truth_dict, prediction_entities, ground_truth_entities
    )

    # Run fuzzy matching evaluation if enabled
    if args.enable_fuzzy:
        print(f"\nRunning fuzzy match evaluation (disease entity name matching)...")
        fuzzy_result = fuzzy_set_based_evaluation(
            predictions_dict,
            ground_truth_dict,
            prediction_entities,
            ground_truth_entities,
            threshold=args.fuzzy_threshold,
        )

        # Combine results
        combined_result = {
            "set_based_exact_match": exact_result,
            "set_based_fuzzy_match": fuzzy_result,
            "metadata": {
                "evaluation_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "fuzzy_threshold": args.fuzzy_threshold,
                "match_method_filter": args.match_method,
                "confidence_threshold": args.confidence_threshold,
                "evaluation_type": "set_based",
            },
        }
    else:
        # Just use exact match results
        combined_result = {
            "set_based_exact_match": exact_result,
            "metadata": {
                "evaluation_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "match_method_filter": args.match_method,
                "confidence_threshold": args.confidence_threshold,
                "evaluation_type": "set_based",
            },
        }

    # Print evaluation summary
    print("\n=== Set-Based Evaluation Summary ===")
    print("\nExact Match Metrics (Numeric ORPHA ID):")
    print(f"  Precision: {exact_result['corpus_metrics']['precision']:.4f}")
    print(f"  Recall: {exact_result['corpus_metrics']['recall']:.4f}")
    print(f"  F1 Score: {exact_result['corpus_metrics']['f1_score']:.4f}")
    print(
        f"  TP/FP/FN: {exact_result['corpus_metrics']['tp_count']}/{exact_result['corpus_metrics']['fp_count']}/{exact_result['corpus_metrics']['fn_count']}"
    )

    if args.enable_fuzzy:
        print("\nFuzzy Match Metrics (Disease Entity Names):")
        print(f"  Precision: {fuzzy_result['corpus_metrics']['precision']:.4f}")
        print(f"  Recall: {fuzzy_result['corpus_metrics']['recall']:.4f}")
        print(f"  F1 Score: {fuzzy_result['corpus_metrics']['f1_score']:.4f}")
        print(
            f"  TP/FP/FN: {fuzzy_result['corpus_metrics']['tp_count']}/{fuzzy_result['corpus_metrics']['fp_count']}/{fuzzy_result['corpus_metrics']['fn_count']}"
        )
        print(
            f"  Fuzzy matches found: {fuzzy_result['corpus_metrics']['fuzzy_matches_count']}"
        )

    # Save results if not summary-only
    if not args.summary_only:
        print(f"\nSaving evaluation results to {args.output}")
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(combined_result, f, indent=2)

    # Save summary as CSV if requested
    if args.save_csv:
        print(f"Saving summary as CSV to {args.save_csv}")

        # Create DataFrame for CSV
        data = {
            "metric": [
                "precision",
                "recall",
                "f1_score",
                "tp_count",
                "fp_count",
                "fn_count",
            ],
            "set_based_exact_match": [
                exact_result["corpus_metrics"]["precision"],
                exact_result["corpus_metrics"]["recall"],
                exact_result["corpus_metrics"]["f1_score"],
                exact_result["corpus_metrics"]["tp_count"],
                exact_result["corpus_metrics"]["fp_count"],
                exact_result["corpus_metrics"]["fn_count"],
            ],
        }

        if args.enable_fuzzy:
            data["set_based_fuzzy_match"] = [
                fuzzy_result["corpus_metrics"]["precision"],
                fuzzy_result["corpus_metrics"]["recall"],
                fuzzy_result["corpus_metrics"]["f1_score"],
                fuzzy_result["corpus_metrics"]["tp_count"],
                fuzzy_result["corpus_metrics"]["fp_count"],
                fuzzy_result["corpus_metrics"]["fn_count"],
            ]

        metrics_df = pd.DataFrame(data)
        metrics_df.to_csv(args.save_csv, index=False)
        print(f"CSV summary saved successfully")


if __name__ == "__main__":
    main()
