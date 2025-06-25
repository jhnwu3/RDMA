"""
Fixed script to analyze error rates by categorizing them as implied vs. direct phenotypes.

The key fix: True positives should be calculated based on PREDICTION status, not ground truth status.
- If model predicts something as "direct" and it exists in ground truth (direct OR implied), it's a direct TP
- If model predicts something as "implied" and it exists in ground truth (direct OR implied), it's an implied TP
"""

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import json
import argparse
import os
import sys
from typing import Dict, List, Set, Tuple, Any
from collections import defaultdict, Counter
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def load_json_file(filepath: str) -> Dict:
    """Load JSON file and return data."""
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return {}


def extract_hpo_codes_from_phenotypes(phenotype_list: List) -> Set[str]:
    """Extract HPO codes from a list of phenotype objects."""
    hpo_codes = set()
    for item in phenotype_list:
        if isinstance(item, dict):
            for code_field in ["hpo_code", "hpo_id", "hp_id", "HPO_Term", "hpo_term"]:
                if code_field in item and item[code_field]:
                    code = item[code_field]
                    if isinstance(code, str) and code.startswith("HP:"):
                        hpo_codes.add(code)
                        break
        elif isinstance(item, str) and item.startswith("HP:"):
            hpo_codes.add(item)
    return hpo_codes


def extract_predictions_with_status(
    predictions_data: Dict,
) -> Tuple[
    Dict[str, Dict[str, Set[str]]], Dict[str, Dict[str, str]], Dict[str, Dict[str, str]]
]:
    """Extract HPO codes from predictions data, categorized by their predicted status."""
    hpo_result = {}
    phenotype_names = {}
    prediction_contexts = {}

    # Handle nested structure with results key
    if isinstance(predictions_data, dict) and "results" in predictions_data:
        predictions_data = predictions_data["results"]

    for case_id, case_data in predictions_data.items():
        direct_hpo = set()
        implied_hpo = set()
        all_hpo = set()
        case_phenotype_names = {}
        case_contexts = {}

        # Check for different possible field names for phenotype lists
        phenotype_fields = ["matched_phenotypes", "verified_phenotypes", "phenotypes"]
        for field in phenotype_fields:
            if field in case_data and isinstance(case_data[field], list):
                for item in case_data[field]:
                    if isinstance(item, dict):
                        # Extract HPO code
                        hpo_code = None
                        for code_field in ["hpo_id", "HPO_Term", "hpo_code", "hp_id"]:
                            if code_field in item and item[code_field]:
                                code = item[code_field]
                                if isinstance(code, str) and code.startswith("HP:"):
                                    hpo_code = code
                                    break

                        # Extract phenotype name and context
                        phenotype_name = item.get("phenotype", "Unknown")
                        context = item.get("context", "No context available")

                        if hpo_code:
                            all_hpo.add(hpo_code)
                            case_phenotype_names[hpo_code] = phenotype_name
                            case_contexts[hpo_code] = context

                            # Check PREDICTED status to categorize
                            status = item.get("status", "unknown")
                            if status == "direct_phenotype":
                                direct_hpo.add(hpo_code)
                            elif status == "implied_phenotype":
                                implied_hpo.add(hpo_code)

        if all_hpo:
            hpo_result[str(case_id)] = {
                "direct": direct_hpo,
                "implied": implied_hpo,
                "all": all_hpo,
            }
            phenotype_names[str(case_id)] = case_phenotype_names
            prediction_contexts[str(case_id)] = case_contexts

    return hpo_result, phenotype_names, prediction_contexts


def extract_ground_truth_from_implied_data(
    implied_data: Dict,
) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]], Dict[str, Set[str]]]:
    """Extract ground truth from the implied phenotypes dataset."""
    all_ground_truth = {}
    direct_ground_truth = {}
    implied_ground_truth = {}

    for case_id, case_data in implied_data.items():
        case_id = str(case_id)

        # Extract direct phenotypes
        direct_hpo = set()
        if "direct_phenotypes" in case_data and isinstance(
            case_data["direct_phenotypes"], list
        ):
            direct_hpo.update(
                extract_hpo_codes_from_phenotypes(case_data["direct_phenotypes"])
            )

        # Extract implied phenotypes
        implied_hpo = set()
        if "implied_phenotypes_with_context" in case_data and isinstance(
            case_data["implied_phenotypes_with_context"], list
        ):
            for item in case_data["implied_phenotypes_with_context"]:
                if isinstance(item, dict) and "hpo_code" in item and item["hpo_code"]:
                    code = item["hpo_code"]
                    if isinstance(code, str) and code.startswith("HP:"):
                        implied_hpo.add(code)

        # Combine all phenotypes as ground truth
        all_hpo = direct_hpo | implied_hpo

        if all_hpo:
            all_ground_truth[case_id] = all_hpo
        if direct_hpo:
            direct_ground_truth[case_id] = direct_hpo
        if implied_hpo:
            implied_ground_truth[case_id] = implied_hpo

    return all_ground_truth, direct_ground_truth, implied_ground_truth


def categorize_errors_by_predicted_status(
    predictions_dict: Dict[str, Dict[str, Set[str]]],
    prediction_phenotype_names: Dict[str, Dict[str, str]],
    prediction_contexts: Dict[str, Dict[str, str]],
    all_ground_truth_dict: Dict[str, Set[str]],
    direct_ground_truth_dict: Dict[str, Set[str]],
    implied_ground_truth_dict: Dict[str, Set[str]],
) -> Dict:
    """
    FIXED: Categorize errors by PREDICTED status vs ground truth.

    Key insight: We care about whether our model's direct/implied predictions are correct,
    not whether ground truth items are correctly categorized by status.
    """
    results = {
        "case_analysis": {},
        "aggregate_stats": {
            "predictions_as_direct": {"tp": 0, "fp": 0, "total_predicted": 0},
            "predictions_as_implied": {"tp": 0, "fp": 0, "total_predicted": 0},
            "ground_truth_direct_missed": 0,  # Direct GT items not predicted at all
            "ground_truth_implied_missed": 0,  # Implied GT items not predicted at all
        },
        "error_patterns": {
            "fp_predicted_as_direct": Counter(),  # Direct predictions that are wrong
            "fp_predicted_as_implied": Counter(),  # Implied predictions that are wrong
            "fn_ground_truth_direct": Counter(),  # Direct GT items missed
            "fn_ground_truth_implied": Counter(),  # Implied GT items missed
            "fp_direct_details": [],
            "fp_implied_details": [],
        },
        "case_counts": {
            "total_cases_evaluated": 0,
            "cases_with_predictions": 0,
            "cases_with_ground_truth": 0,
            "cases_with_direct_gt": 0,
            "cases_with_implied_gt": 0,
        },
    }

    # Process each case with predictions
    for case_id in predictions_dict.keys():
        case_analysis = {
            "predicted_all": list(predictions_dict[case_id]["all"]),
            "predicted_direct": list(predictions_dict[case_id]["direct"]),
            "predicted_implied": list(predictions_dict[case_id]["implied"]),
            "ground_truth_all": list(all_ground_truth_dict.get(case_id, set())),
            "ground_truth_direct": list(direct_ground_truth_dict.get(case_id, set())),
            "ground_truth_implied": list(implied_ground_truth_dict.get(case_id, set())),
            # FIXED: Organize by prediction status
            "direct_predictions": {
                "correct": [],  # Direct predictions that are in GT (regardless of GT status)
                "incorrect": [],  # Direct predictions that are NOT in GT
            },
            "implied_predictions": {
                "correct": [],  # Implied predictions that are in GT (regardless of GT status)
                "incorrect": [],  # Implied predictions that are NOT in GT
            },
            "missed_ground_truth": {
                "direct": [],  # Direct GT items not predicted at all
                "implied": [],  # Implied GT items not predicted at all
            },
        }

        # Get predictions and ground truth
        all_predictions = predictions_dict[case_id]["all"]
        direct_predictions = predictions_dict[case_id]["direct"]
        implied_predictions = predictions_dict[case_id]["implied"]

        all_ground_truth = all_ground_truth_dict.get(case_id, set())
        direct_ground_truth = direct_ground_truth_dict.get(case_id, set())
        implied_ground_truth = implied_ground_truth_dict.get(case_id, set())

        # Update case counts
        results["case_counts"]["cases_with_predictions"] += 1
        if all_ground_truth:
            results["case_counts"]["cases_with_ground_truth"] += 1
        if direct_ground_truth:
            results["case_counts"]["cases_with_direct_gt"] += 1
        if implied_ground_truth:
            results["case_counts"]["cases_with_implied_gt"] += 1

        # FIXED: Analyze by prediction status

        # 1. Direct predictions: Are they correct regardless of their GT status?
        direct_correct = direct_predictions & all_ground_truth
        direct_incorrect = direct_predictions - all_ground_truth

        case_analysis["direct_predictions"]["correct"] = list(direct_correct)
        case_analysis["direct_predictions"]["incorrect"] = list(direct_incorrect)

        # 2. Implied predictions: Are they correct regardless of their GT status?
        implied_correct = implied_predictions & all_ground_truth
        implied_incorrect = implied_predictions - all_ground_truth

        case_analysis["implied_predictions"]["correct"] = list(implied_correct)
        case_analysis["implied_predictions"]["incorrect"] = list(implied_incorrect)

        # 3. Missed ground truth items (False negatives by GT status)
        missed_direct_gt = direct_ground_truth - all_predictions
        missed_implied_gt = implied_ground_truth - all_predictions

        case_analysis["missed_ground_truth"]["direct"] = list(missed_direct_gt)
        case_analysis["missed_ground_truth"]["implied"] = list(missed_implied_gt)

        # Update aggregate stats
        results["aggregate_stats"]["predictions_as_direct"]["tp"] += len(direct_correct)
        results["aggregate_stats"]["predictions_as_direct"]["fp"] += len(
            direct_incorrect
        )
        results["aggregate_stats"]["predictions_as_direct"]["total_predicted"] += len(
            direct_predictions
        )

        results["aggregate_stats"]["predictions_as_implied"]["tp"] += len(
            implied_correct
        )
        results["aggregate_stats"]["predictions_as_implied"]["fp"] += len(
            implied_incorrect
        )
        results["aggregate_stats"]["predictions_as_implied"]["total_predicted"] += len(
            implied_predictions
        )

        results["aggregate_stats"]["ground_truth_direct_missed"] += len(
            missed_direct_gt
        )
        results["aggregate_stats"]["ground_truth_implied_missed"] += len(
            missed_implied_gt
        )

        # Update error patterns
        case_prediction_names = prediction_phenotype_names.get(case_id, {})
        case_prediction_contexts = prediction_contexts.get(case_id, {})

        for code in direct_incorrect:
            results["error_patterns"]["fp_predicted_as_direct"][code] += 1
            phenotype_name = case_prediction_names.get(code, "Unknown")
            context = case_prediction_contexts.get(code, "No context available")
            results["error_patterns"]["fp_direct_details"].append(
                {
                    "code": code,
                    "phenotype_name": phenotype_name,
                    "context": context,
                    "case_id": case_id,
                }
            )

        for code in implied_incorrect:
            results["error_patterns"]["fp_predicted_as_implied"][code] += 1
            phenotype_name = case_prediction_names.get(code, "Unknown")
            context = case_prediction_contexts.get(code, "No context available")
            results["error_patterns"]["fp_implied_details"].append(
                {
                    "code": code,
                    "phenotype_name": phenotype_name,
                    "context": context,
                    "case_id": case_id,
                }
            )

        for code in missed_direct_gt:
            results["error_patterns"]["fn_ground_truth_direct"][code] += 1

        for code in missed_implied_gt:
            results["error_patterns"]["fn_ground_truth_implied"][code] += 1

        results["case_analysis"][case_id] = case_analysis

    results["case_counts"]["total_cases_evaluated"] = len(predictions_dict)
    return results


def calculate_prediction_accuracy_metrics(results: Dict) -> Dict:
    """Calculate accuracy metrics for direct vs implied predictions."""
    stats = results["aggregate_stats"]
    metrics = {}

    # Direct predictions accuracy
    direct_tp = stats["predictions_as_direct"]["tp"]
    direct_fp = stats["predictions_as_direct"]["fp"]
    direct_total = stats["predictions_as_direct"]["total_predicted"]

    if direct_total > 0:
        metrics["direct_predictions"] = {
            "precision": direct_tp / direct_total,
            "tp_count": direct_tp,
            "fp_count": direct_fp,
            "total_predicted": direct_total,
            "accuracy_pct": (direct_tp / direct_total) * 100,
        }
    else:
        metrics["direct_predictions"] = {
            "precision": 0.0,
            "tp_count": 0,
            "fp_count": 0,
            "total_predicted": 0,
            "accuracy_pct": 0.0,
        }

    # Implied predictions accuracy
    implied_tp = stats["predictions_as_implied"]["tp"]
    implied_fp = stats["predictions_as_implied"]["fp"]
    implied_total = stats["predictions_as_implied"]["total_predicted"]

    if implied_total > 0:
        metrics["implied_predictions"] = {
            "precision": implied_tp / implied_total,
            "tp_count": implied_tp,
            "fp_count": implied_fp,
            "total_predicted": implied_total,
            "accuracy_pct": (implied_tp / implied_total) * 100,
        }
    else:
        metrics["implied_predictions"] = {
            "precision": 0.0,
            "tp_count": 0,
            "fp_count": 0,
            "total_predicted": 0,
            "accuracy_pct": 0.0,
        }

    # Overall metrics
    total_tp = direct_tp + implied_tp
    total_fp = direct_fp + implied_fp
    total_predictions = direct_total + implied_total

    if total_predictions > 0:
        metrics["overall"] = {
            "precision": total_tp / total_predictions,
            "tp_count": total_tp,
            "fp_count": total_fp,
            "total_predicted": total_predictions,
            "accuracy_pct": (total_tp / total_predictions) * 100,
        }
    else:
        metrics["overall"] = {
            "precision": 0.0,
            "tp_count": 0,
            "fp_count": 0,
            "total_predicted": 0,
            "accuracy_pct": 0.0,
        }

    # Ground truth coverage
    metrics["ground_truth_coverage"] = {
        "direct_missed": stats["ground_truth_direct_missed"],
        "implied_missed": stats["ground_truth_implied_missed"],
        "total_missed": stats["ground_truth_direct_missed"]
        + stats["ground_truth_implied_missed"],
    }

    return metrics


def analyze_error_composition(
    results: Dict,
    all_ground_truth_dict: Dict[str, Set[str]],
    direct_ground_truth_dict: Dict[str, Set[str]],
    implied_ground_truth_dict: Dict[str, Set[str]],
) -> Dict:
    """
    Analyze the composition of true positives, false positives and false negatives.
    """
    composition = {
        "true_positives_by_prediction_status": {
            "predicted_as_direct": 0,
            "predicted_as_implied": 0,
            "total": 0,
        },
        "false_positives_by_prediction_status": {
            "predicted_as_direct": 0,
            "predicted_as_implied": 0,
            "total": 0,
        },
        "false_negatives_by_ground_truth_status": {
            "direct_in_gt": 0,
            "implied_in_gt": 0,
            "total": 0,
        },
        "true_positives_cross_classification": {
            "predicted_direct_was_direct_in_gt": 0,
            "predicted_direct_was_implied_in_gt": 0,
            "predicted_implied_was_direct_in_gt": 0,
            "predicted_implied_was_implied_in_gt": 0,
            "total": 0,
        },
    }

    # Analyze each case
    for case_id, case_data in results["case_analysis"].items():
        # True Positives composition (by prediction status)
        tp_direct_count = len(case_data["direct_predictions"]["correct"])
        tp_implied_count = len(case_data["implied_predictions"]["correct"])

        composition["true_positives_by_prediction_status"][
            "predicted_as_direct"
        ] += tp_direct_count
        composition["true_positives_by_prediction_status"][
            "predicted_as_implied"
        ] += tp_implied_count

        # False Positives composition (by prediction status)
        fp_direct_count = len(case_data["direct_predictions"]["incorrect"])
        fp_implied_count = len(case_data["implied_predictions"]["incorrect"])

        composition["false_positives_by_prediction_status"][
            "predicted_as_direct"
        ] += fp_direct_count
        composition["false_positives_by_prediction_status"][
            "predicted_as_implied"
        ] += fp_implied_count

        # False Negatives composition (by ground truth status)
        fn_direct_count = len(case_data["missed_ground_truth"]["direct"])
        fn_implied_count = len(case_data["missed_ground_truth"]["implied"])

        composition["false_negatives_by_ground_truth_status"][
            "direct_in_gt"
        ] += fn_direct_count
        composition["false_negatives_by_ground_truth_status"][
            "implied_in_gt"
        ] += fn_implied_count

        # True Positives cross-classification analysis
        # For each correct prediction, check if the prediction status matches GT status
        case_direct_gt = direct_ground_truth_dict.get(case_id, set())
        case_implied_gt = implied_ground_truth_dict.get(case_id, set())

        for hpo_code in case_data["direct_predictions"]["correct"]:
            if hpo_code in case_direct_gt:
                composition["true_positives_cross_classification"][
                    "predicted_direct_was_direct_in_gt"
                ] += 1
            elif hpo_code in case_implied_gt:
                composition["true_positives_cross_classification"][
                    "predicted_direct_was_implied_in_gt"
                ] += 1

        for hpo_code in case_data["implied_predictions"]["correct"]:
            if hpo_code in case_direct_gt:
                composition["true_positives_cross_classification"][
                    "predicted_implied_was_direct_in_gt"
                ] += 1
            elif hpo_code in case_implied_gt:
                composition["true_positives_cross_classification"][
                    "predicted_implied_was_implied_in_gt"
                ] += 1

    # Calculate totals
    composition["true_positives_by_prediction_status"]["total"] = (
        composition["true_positives_by_prediction_status"]["predicted_as_direct"]
        + composition["true_positives_by_prediction_status"]["predicted_as_implied"]
    )

    composition["false_positives_by_prediction_status"]["total"] = (
        composition["false_positives_by_prediction_status"]["predicted_as_direct"]
        + composition["false_positives_by_prediction_status"]["predicted_as_implied"]
    )

    composition["false_negatives_by_ground_truth_status"]["total"] = (
        composition["false_negatives_by_ground_truth_status"]["direct_in_gt"]
        + composition["false_negatives_by_ground_truth_status"]["implied_in_gt"]
    )

    composition["true_positives_cross_classification"]["total"] = (
        composition["true_positives_cross_classification"][
            "predicted_direct_was_direct_in_gt"
        ]
        + composition["true_positives_cross_classification"][
            "predicted_direct_was_implied_in_gt"
        ]
        + composition["true_positives_cross_classification"][
            "predicted_implied_was_direct_in_gt"
        ]
        + composition["true_positives_cross_classification"][
            "predicted_implied_was_implied_in_gt"
        ]
    )

    # Calculate percentages
    composition["percentages"] = {}

    # TP percentages
    tp_total = composition["true_positives_by_prediction_status"]["total"]
    if tp_total > 0:
        composition["percentages"]["true_positives"] = {
            "predicted_as_direct_pct": (
                composition["true_positives_by_prediction_status"][
                    "predicted_as_direct"
                ]
                / tp_total
            )
            * 100,
            "predicted_as_implied_pct": (
                composition["true_positives_by_prediction_status"][
                    "predicted_as_implied"
                ]
                / tp_total
            )
            * 100,
        }
    else:
        composition["percentages"]["true_positives"] = {
            "predicted_as_direct_pct": 0.0,
            "predicted_as_implied_pct": 0.0,
        }

    # FP percentages
    fp_total = composition["false_positives_by_prediction_status"]["total"]
    if fp_total > 0:
        composition["percentages"]["false_positives"] = {
            "predicted_as_direct_pct": (
                composition["false_positives_by_prediction_status"][
                    "predicted_as_direct"
                ]
                / fp_total
            )
            * 100,
            "predicted_as_implied_pct": (
                composition["false_positives_by_prediction_status"][
                    "predicted_as_implied"
                ]
                / fp_total
            )
            * 100,
        }
    else:
        composition["percentages"]["false_positives"] = {
            "predicted_as_direct_pct": 0.0,
            "predicted_as_implied_pct": 0.0,
        }

    # FN percentages
    fn_total = composition["false_negatives_by_ground_truth_status"]["total"]
    if fn_total > 0:
        composition["percentages"]["false_negatives"] = {
            "direct_in_gt_pct": (
                composition["false_negatives_by_ground_truth_status"]["direct_in_gt"]
                / fn_total
            )
            * 100,
            "implied_in_gt_pct": (
                composition["false_negatives_by_ground_truth_status"]["implied_in_gt"]
                / fn_total
            )
            * 100,
        }
    else:
        composition["percentages"]["false_negatives"] = {
            "direct_in_gt_pct": 0.0,
            "implied_in_gt_pct": 0.0,
        }

    # TP cross-classification percentages
    tp_total = composition["true_positives_cross_classification"]["total"]
    if tp_total > 0:
        composition["percentages"]["true_positives_cross_classification"] = {
            "predicted_direct_was_direct_in_gt_pct": (
                composition["true_positives_cross_classification"][
                    "predicted_direct_was_direct_in_gt"
                ]
                / tp_total
            )
            * 100,
            "predicted_direct_was_implied_in_gt_pct": (
                composition["true_positives_cross_classification"][
                    "predicted_direct_was_implied_in_gt"
                ]
                / tp_total
            )
            * 100,
            "predicted_implied_was_direct_in_gt_pct": (
                composition["true_positives_cross_classification"][
                    "predicted_implied_was_direct_in_gt"
                ]
                / tp_total
            )
            * 100,
            "predicted_implied_was_implied_in_gt_pct": (
                composition["true_positives_cross_classification"][
                    "predicted_implied_was_implied_in_gt"
                ]
                / tp_total
            )
            * 100,
        }
    else:
        composition["percentages"]["true_positives_cross_classification"] = {
            "predicted_direct_was_direct_in_gt_pct": 0.0,
            "predicted_direct_was_implied_in_gt_pct": 0.0,
            "predicted_implied_was_direct_in_gt_pct": 0.0,
            "predicted_implied_was_implied_in_gt_pct": 0.0,
        }

    return composition


def show_detailed_examples(
    results: Dict,
    implied_data: Dict,
    prediction_phenotype_names: Dict[str, Dict[str, str]],
    prediction_contexts: Dict[str, Dict[str, str]],
) -> None:
    """Show detailed examples for each category of predictions."""
    print(f"\n" + "=" * 80)
    print("DETAILED EXAMPLES BY PREDICTION CATEGORY")
    print("=" * 80)

    def get_clinical_text(case_id: str) -> str:
        """Get clinical text for a case."""
        case_id = str(case_id)
        if case_id in implied_data:
            text = implied_data[case_id].get("text", "")
            if text:
                return text[:200] + "..." if len(text) > 200 else text
        return "No text available"

    def get_ground_truth_status(hpo_code: str, case_id: str) -> str:
        """Determine if HPO code is direct/implied/missing in ground truth."""
        case_id = str(case_id)
        if case_id in implied_data:
            case_data = implied_data[case_id]

            # Check direct phenotypes
            if "direct_phenotypes" in case_data:
                for item in case_data["direct_phenotypes"]:
                    if isinstance(item, dict) and item.get("hpo_code") == hpo_code:
                        return "direct_in_gt"

            # Check implied phenotypes
            if "implied_phenotypes_with_context" in case_data:
                for item in case_data["implied_phenotypes_with_context"]:
                    if isinstance(item, dict) and item.get("hpo_code") == hpo_code:
                        return "implied_in_gt"

        return "not_in_gt"

    def get_gt_context(hpo_code: str, case_id: str) -> str:
        """Get context from ground truth if available."""
        case_id = str(case_id)
        if case_id in implied_data:
            case_data = implied_data[case_id]

            # Check implied phenotypes for context
            if "implied_phenotypes_with_context" in case_data:
                for item in case_data["implied_phenotypes_with_context"]:
                    if isinstance(item, dict) and item.get("hpo_code") == hpo_code:
                        return item.get("context", "No context in GT")

        return "No context in GT (direct phenotype or not found)"

    # Collect examples for each category
    examples = {
        "direct_predictions": {"tp": [], "fp": []},
        "implied_predictions": {"tp": [], "fp": []},
        "missed_gt": {"direct": [], "implied": []},
    }

    # Gather examples from case analysis
    for case_id, case_data in results["case_analysis"].items():
        clinical_text = get_clinical_text(case_id)

        # Direct prediction examples
        for hpo_code in case_data["direct_predictions"]["correct"][
            :3
        ]:  # Limit per case
            gt_status = get_ground_truth_status(hpo_code, case_id)
            gt_context = get_gt_context(hpo_code, case_id)
            pred_name = prediction_phenotype_names.get(case_id, {}).get(
                hpo_code, "Unknown"
            )
            pred_context = prediction_contexts.get(case_id, {}).get(
                hpo_code, "No context"
            )

            examples["direct_predictions"]["tp"].append(
                {
                    "case_id": case_id,
                    "hpo_code": hpo_code,
                    "predicted_name": pred_name,
                    "predicted_context": pred_context,
                    "gt_status": gt_status,
                    "gt_context": gt_context,
                    "clinical_text": clinical_text,
                }
            )

        for hpo_code in case_data["direct_predictions"]["incorrect"][
            :3
        ]:  # Limit per case
            pred_name = prediction_phenotype_names.get(case_id, {}).get(
                hpo_code, "Unknown"
            )
            pred_context = prediction_contexts.get(case_id, {}).get(
                hpo_code, "No context"
            )

            examples["direct_predictions"]["fp"].append(
                {
                    "case_id": case_id,
                    "hpo_code": hpo_code,
                    "predicted_name": pred_name,
                    "predicted_context": pred_context,
                    "clinical_text": clinical_text,
                }
            )

        # Implied prediction examples
        for hpo_code in case_data["implied_predictions"]["correct"][
            :3
        ]:  # Limit per case
            gt_status = get_ground_truth_status(hpo_code, case_id)
            gt_context = get_gt_context(hpo_code, case_id)
            pred_name = prediction_phenotype_names.get(case_id, {}).get(
                hpo_code, "Unknown"
            )
            pred_context = prediction_contexts.get(case_id, {}).get(
                hpo_code, "No context"
            )

            examples["implied_predictions"]["tp"].append(
                {
                    "case_id": case_id,
                    "hpo_code": hpo_code,
                    "predicted_name": pred_name,
                    "predicted_context": pred_context,
                    "gt_status": gt_status,
                    "gt_context": gt_context,
                    "clinical_text": clinical_text,
                }
            )

        for hpo_code in case_data["implied_predictions"]["incorrect"][
            :3
        ]:  # Limit per case
            pred_name = prediction_phenotype_names.get(case_id, {}).get(
                hpo_code, "Unknown"
            )
            pred_context = prediction_contexts.get(case_id, {}).get(
                hpo_code, "No context"
            )

            examples["implied_predictions"]["fp"].append(
                {
                    "case_id": case_id,
                    "hpo_code": hpo_code,
                    "predicted_name": pred_name,
                    "predicted_context": pred_context,
                    "clinical_text": clinical_text,
                }
            )

        # Missed ground truth examples
        for hpo_code in case_data["missed_ground_truth"]["direct"][
            :2
        ]:  # Limit per case
            # Get GT info for missed items
            gt_name = "Unknown"
            if case_id in implied_data and "direct_phenotypes" in implied_data[case_id]:
                for item in implied_data[case_id]["direct_phenotypes"]:
                    if isinstance(item, dict) and item.get("hpo_code") == hpo_code:
                        gt_name = item.get("phenotype", "Unknown")
                        break

            examples["missed_gt"]["direct"].append(
                {
                    "case_id": case_id,
                    "hpo_code": hpo_code,
                    "gt_name": gt_name,
                    "clinical_text": clinical_text,
                }
            )

        for hpo_code in case_data["missed_ground_truth"]["implied"][
            :2
        ]:  # Limit per case
            # Get GT info for missed items
            gt_name = "Unknown"
            gt_context = "No context"
            if (
                case_id in implied_data
                and "implied_phenotypes_with_context" in implied_data[case_id]
            ):
                for item in implied_data[case_id]["implied_phenotypes_with_context"]:
                    if isinstance(item, dict) and item.get("hpo_code") == hpo_code:
                        gt_name = item.get("implied_phenotype", "Unknown")
                        gt_context = item.get("context", "No context")
                        break

            examples["missed_gt"]["implied"].append(
                {
                    "case_id": case_id,
                    "hpo_code": hpo_code,
                    "gt_name": gt_name,
                    "gt_context": gt_context,
                    "clinical_text": clinical_text,
                }
            )

    # Display examples
    print(f"\n🎯 DIRECT PREDICTIONS (predicted as 'direct'):")
    print("-" * 60)

    print(
        f"\n✅ TRUE POSITIVES: Direct predictions that exist in ground truth ({len(examples['direct_predictions']['tp'][:5])} shown):"
    )
    for i, ex in enumerate(examples["direct_predictions"]["tp"][:5], 1):
        print(f"\n  {i}. Case: {ex['case_id']} | HPO: {ex['hpo_code']}")
        print(f"     Predicted: '{ex['predicted_name']}'")
        print(f"     Pred Context: {ex['predicted_context']}")
        print(f"     GT Status: {ex['gt_status']} | GT Context: {ex['gt_context']}")
        print(f"     Clinical: {ex['clinical_text']}")

    print(
        f"\n❌ FALSE POSITIVES: Direct predictions NOT in ground truth ({len(examples['direct_predictions']['fp'][:5])} shown):"
    )
    for i, ex in enumerate(examples["direct_predictions"]["fp"][:5], 1):
        print(f"\n  {i}. Case: {ex['case_id']} | HPO: {ex['hpo_code']}")
        print(f"     Predicted: '{ex['predicted_name']}'")
        print(f"     Pred Context: {ex['predicted_context']}")
        print(f"     Clinical: {ex['clinical_text']}")

    print(f"\n🔍 IMPLIED PREDICTIONS (predicted as 'implied'):")
    print("-" * 60)

    print(
        f"\n✅ TRUE POSITIVES: Implied predictions that exist in ground truth ({len(examples['implied_predictions']['tp'][:5])} shown):"
    )
    for i, ex in enumerate(examples["implied_predictions"]["tp"][:5], 1):
        print(f"\n  {i}. Case: {ex['case_id']} | HPO: {ex['hpo_code']}")
        print(f"     Predicted: '{ex['predicted_name']}'")
        print(f"     Pred Context: {ex['predicted_context']}")
        print(f"     GT Status: {ex['gt_status']} | GT Context: {ex['gt_context']}")
        print(f"     Clinical: {ex['clinical_text']}")

    print(
        f"\n❌ FALSE POSITIVES: Implied predictions NOT in ground truth ({len(examples['implied_predictions']['fp'][:5])} shown):"
    )
    for i, ex in enumerate(examples["implied_predictions"]["fp"][:5], 1):
        print(f"\n  {i}. Case: {ex['case_id']} | HPO: {ex['hpo_code']}")
        print(f"     Predicted: '{ex['predicted_name']}'")
        print(f"     Pred Context: {ex['predicted_context']}")
        print(f"     Clinical: {ex['clinical_text']}")

    print(f"\n⚠️  MISSED GROUND TRUTH (False Negatives):")
    print("-" * 60)

    print(
        f"\n📋 DIRECT GT MISSED: Direct phenotypes in GT that we didn't predict at all ({len(examples['missed_gt']['direct'][:5])} shown):"
    )
    for i, ex in enumerate(examples["missed_gt"]["direct"][:5], 1):
        print(f"\n  {i}. Case: {ex['case_id']} | HPO: {ex['hpo_code']}")
        print(f"     GT Name: '{ex['gt_name']}'")
        print(f"     Clinical: {ex['clinical_text']}")

    print(
        f"\n🔍 IMPLIED GT MISSED: Implied phenotypes in GT that we didn't predict at all ({len(examples['missed_gt']['implied'][:5])} shown):"
    )
    for i, ex in enumerate(examples["missed_gt"]["implied"][:5], 1):
        print(f"\n  {i}. Case: {ex['case_id']} | HPO: {ex['hpo_code']}")
        print(f"     GT Name: '{ex['gt_name']}'")
        print(f"     GT Context: {ex['gt_context']}")
        print(f"     Clinical: {ex['clinical_text']}")


def print_composition_analysis(composition: Dict):
    """Print detailed composition analysis of errors and true positives."""
    print(f"\n" + "=" * 80)
    print("📊 ERROR COMPOSITION ANALYSIS")
    print("=" * 80)

    # False Positives composition
    fp_stats = composition["false_positives_by_prediction_status"]
    fp_pcts = composition["percentages"]["false_positives"]

    print(f"\n❌ FALSE POSITIVES COMPOSITION (by prediction status):")
    print(f"  Total False Positives: {fp_stats['total']}")
    print(
        f"  Predicted as Direct:   {fp_stats['predicted_as_direct']} ({fp_pcts['predicted_as_direct_pct']:.1f}%)"
    )
    print(
        f"  Predicted as Implied:  {fp_stats['predicted_as_implied']} ({fp_pcts['predicted_as_implied_pct']:.1f}%)"
    )

    if fp_stats["total"] > 0:
        print(
            f"\n  💡 Insight: {fp_pcts['predicted_as_direct_pct']:.1f}% of false positives were incorrectly predicted as direct phenotypes"
        )

    # False Negatives composition
    fn_stats = composition["false_negatives_by_ground_truth_status"]
    fn_pcts = composition["percentages"]["false_negatives"]

    print(f"\n⚠️  FALSE NEGATIVES COMPOSITION (by ground truth status):")
    print(f"  Total False Negatives: {fn_stats['total']}")
    print(
        f"  Direct in GT:   {fn_stats['direct_in_gt']} ({fn_pcts['direct_in_gt_pct']:.1f}%)"
    )
    print(
        f"  Implied in GT:  {fn_stats['implied_in_gt']} ({fn_pcts['implied_in_gt_pct']:.1f}%)"
    )

    if fn_stats["total"] > 0:
        print(
            f"\n  💡 Insight: {fn_pcts['direct_in_gt_pct']:.1f}% of missed phenotypes were direct in ground truth"
        )

    # True Positives cross-classification
    tp_stats = composition["true_positives_cross_classification"]
    tp_pcts = composition["percentages"]["true_positives_cross_classification"]

    print(f"\n✅ TRUE POSITIVES CROSS-CLASSIFICATION:")
    print(f"  Total True Positives: {tp_stats['total']}")
    print(
        f"  📋→📋 Predicted Direct, was Direct in GT:   {tp_stats['predicted_direct_was_direct_in_gt']} ({tp_pcts['predicted_direct_was_direct_in_gt_pct']:.1f}%)"
    )
    print(
        f"  📋→🔍 Predicted Direct, was Implied in GT:  {tp_stats['predicted_direct_was_implied_in_gt']} ({tp_pcts['predicted_direct_was_implied_in_gt_pct']:.1f}%)"
    )
    print(
        f"  🔍→📋 Predicted Implied, was Direct in GT:  {tp_stats['predicted_implied_was_direct_in_gt']} ({tp_pcts['predicted_implied_was_direct_in_gt_pct']:.1f}%)"
    )
    print(
        f"  🔍→🔍 Predicted Implied, was Implied in GT: {tp_stats['predicted_implied_was_implied_in_gt']} ({tp_pcts['predicted_implied_was_implied_in_gt_pct']:.1f}%)"
    )

    if tp_stats["total"] > 0:
        # Calculate status agreement
        perfect_agreement = (
            tp_stats["predicted_direct_was_direct_in_gt"]
            + tp_stats["predicted_implied_was_implied_in_gt"]
        )
        status_agreement_pct = (perfect_agreement / tp_stats["total"]) * 100

        print(
            f"\n  💡 Status Classification Accuracy: {perfect_agreement}/{tp_stats['total']} ({status_agreement_pct:.1f}%) predictions had correct direct/implied classification"
        )

        if tp_pcts["predicted_direct_was_implied_in_gt_pct"] > 20:
            print(
                f"  ⚠️  Warning: {tp_pcts['predicted_direct_was_implied_in_gt_pct']:.1f}% of correct predictions were classified as direct when they were implied in GT"
            )

        if tp_pcts["predicted_implied_was_direct_in_gt_pct"] > 20:
            print(
                f"  ⚠️  Warning: {tp_pcts['predicted_implied_was_direct_in_gt_pct']:.1f}% of correct predictions were classified as implied when they were direct in GT"
            )


def print_fixed_analysis_summary(results: Dict, metrics: Dict):
    """Print the corrected analysis summary."""
    print("\n" + "=" * 80)
    print("FIXED: PREDICTION STATUS ACCURACY ANALYSIS")
    print("Question: How accurate are our direct vs implied predictions?")
    print("=" * 80)

    # Case counts
    counts = results["case_counts"]
    print(f"\nCASE STATISTICS:")
    print(f"  Total cases evaluated: {counts['total_cases_evaluated']}")
    print(f"  Cases with predictions: {counts['cases_with_predictions']}")
    print(f"  Cases with ground truth: {counts['cases_with_ground_truth']}")
    print(f"  Cases with direct ground truth: {counts['cases_with_direct_gt']}")
    print(f"  Cases with implied ground truth: {counts['cases_with_implied_gt']}")

    # Prediction accuracy
    print(f"\nPREDICTION ACCURACY BY STATUS:")

    direct = metrics["direct_predictions"]
    print(f"\n🎯 DIRECT PREDICTIONS:")
    print(f"  Total predicted as direct: {direct['total_predicted']}")
    print(
        f"  ✅ Correct (in ground truth): {direct['tp_count']} ({direct['accuracy_pct']:.1f}%)"
    )
    print(
        f"  ❌ Incorrect (not in ground truth): {direct['fp_count']} ({100-direct['accuracy_pct']:.1f}%)"
    )
    print(f"  Precision: {direct['precision']:.4f}")

    implied = metrics["implied_predictions"]
    print(f"\n🔍 IMPLIED PREDICTIONS:")
    print(f"  Total predicted as implied: {implied['total_predicted']}")
    print(
        f"  ✅ Correct (in ground truth): {implied['tp_count']} ({implied['accuracy_pct']:.1f}%)"
    )
    print(
        f"  ❌ Incorrect (not in ground truth): {implied['fp_count']} ({100-implied['accuracy_pct']:.1f}%)"
    )
    print(f"  Precision: {implied['precision']:.4f}")

    overall = metrics["overall"]
    print(f"\n📊 OVERALL:")
    print(f"  Total predictions: {overall['total_predicted']}")
    print(
        f"  Correct predictions: {overall['tp_count']} ({overall['accuracy_pct']:.1f}%)"
    )
    print(
        f"  Incorrect predictions: {overall['fp_count']} ({100-overall['accuracy_pct']:.1f}%)"
    )
    print(f"  Overall precision: {overall['precision']:.4f}")

    # Ground truth coverage
    coverage = metrics["ground_truth_coverage"]
    print(f"\n⚠️  GROUND TRUTH COVERAGE (what we missed):")
    print(f"  Direct phenotypes missed: {coverage['direct_missed']}")
    print(f"  Implied phenotypes missed: {coverage['implied_missed']}")
    print(f"  Total phenotypes missed: {coverage['total_missed']}")

    # Error patterns
    print(f"\n🔍 MOST COMMON INCORRECT PREDICTIONS:")

    fp_direct = results["error_patterns"]["fp_predicted_as_direct"].most_common(5)
    if fp_direct:
        print(f"\nIncorrect Direct Predictions:")
        for code, count in fp_direct:
            example_name = "Unknown"
            for detail in results["error_patterns"]["fp_direct_details"]:
                if detail["code"] == code and detail["phenotype_name"] != "Unknown":
                    example_name = detail["phenotype_name"]
                    break
            print(f"  {code}: {count} occurrences (e.g., '{example_name}')")

    fp_implied = results["error_patterns"]["fp_predicted_as_implied"].most_common(5)
    if fp_implied:
        print(f"\nIncorrect Implied Predictions:")
        for code, count in fp_implied:
            example_name = "Unknown"
            for detail in results["error_patterns"]["fp_implied_details"]:
                if detail["code"] == code and detail["phenotype_name"] != "Unknown":
                    example_name = detail["phenotype_name"]
                    break
            print(f"  {code}: {count} occurrences (e.g., '{example_name}')")


import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict


def create_composition_pie_charts(composition: Dict, output_dir: str = None) -> Dict:
    """
    Create individual pie charts for the composition analysis and save as PNG files.
    Charts have no titles or legends for clean appearance.

    Args:
        composition: Composition analysis results
        output_dir: Directory to save charts (optional)

    Returns:
        Dictionary with chart figures
    """
    # Color palette
    color_palette = [
        "rgb(103,122,165)",  # Grayish-blue (for Direct)
        "rgb(252,141,98)",  # Orange (for Implied)
    ]

    charts = {}

    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # 1. True Positives by Prediction Status
    tp_stats = composition["true_positives_by_prediction_status"]
    if tp_stats["total"] > 0:
        tp_data = pd.DataFrame(
            {
                "Status": ["Predicted as Direct", "Predicted as Implied"],
                "Count": [
                    tp_stats["predicted_as_direct"],
                    tp_stats["predicted_as_implied"],
                ],
                "Percentage": [
                    composition["percentages"]["true_positives"][
                        "predicted_as_direct_pct"
                    ],
                    round(
                        composition["percentages"]["true_positives"][
                            "predicted_as_implied_pct"
                        ],
                        1,
                    ),
                ],
            }
        )

        # Remove rows with 0 count
        tp_data = tp_data[tp_data["Count"] > 0]

        if not tp_data.empty:
            fig_tp = px.pie(
                tp_data,
                values="Count",
                names="Status",
                color_discrete_sequence=color_palette[: len(tp_data)],
            )
            fig_tp.update_traces(
                textinfo="percent", texttemplate="%{percent:.1%}", textfont_size=36
            )
            fig_tp.update_layout(
                font=dict(size=16),
                width=500,
                height=500,
                showlegend=False,
                margin=dict(t=20, b=20, l=20, r=20),
            )
            charts["true_positives"] = fig_tp

            # Save as PNG
            if output_dir:
                png_path = os.path.join(
                    output_dir, "true_positives_by_prediction_status.png"
                )
                fig_tp.write_image(png_path, width=500, height=500, scale=2)
                print(f"Saved True Positives chart to: {png_path}")

    # 2. False Positives by Prediction Status
    fp_stats = composition["false_positives_by_prediction_status"]
    if fp_stats["total"] > 0:
        fp_data = pd.DataFrame(
            {
                "Status": ["Predicted as Direct", "Predicted as Implied"],
                "Count": [
                    fp_stats["predicted_as_direct"],
                    fp_stats["predicted_as_implied"],
                ],
                "Percentage": [
                    composition["percentages"]["false_positives"][
                        "predicted_as_direct_pct"
                    ],
                    composition["percentages"]["false_positives"][
                        "predicted_as_implied_pct"
                    ],
                ],
            }
        )

        # Remove rows with 0 count
        fp_data = fp_data[fp_data["Count"] > 0]

        if not fp_data.empty:
            fig_fp = px.pie(
                fp_data,
                values="Count",
                names="Status",
                color_discrete_sequence=color_palette[: len(fp_data)],
            )
            fig_fp.update_traces(
                textinfo="percent", texttemplate="%{percent}", textfont_size=36
            )
            fig_fp.update_layout(
                font=dict(size=16),
                width=500,
                height=500,
                showlegend=False,
                margin=dict(t=20, b=20, l=20, r=20),
            )
            charts["false_positives"] = fig_fp

            # Save as PNG
            if output_dir:
                png_path = os.path.join(
                    output_dir, "false_positives_by_prediction_status.png"
                )
                fig_fp.write_image(png_path, width=500, height=500, scale=2)
                print(f"Saved False Positives chart to: {png_path}")

    # 3. False Negatives by Ground Truth Status
    fn_stats = composition["false_negatives_by_ground_truth_status"]
    if fn_stats["total"] > 0:
        fn_data = pd.DataFrame(
            {
                "Status": ["Direct in GT", "Implied in GT"],
                "Count": [fn_stats["direct_in_gt"], fn_stats["implied_in_gt"]],
                "Percentage": [
                    composition["percentages"]["false_negatives"]["direct_in_gt_pct"],
                    composition["percentages"]["false_negatives"]["implied_in_gt_pct"],
                ],
            }
        )

        # Remove rows with 0 count
        fn_data = fn_data[fn_data["Count"] > 0]

        if not fn_data.empty:
            fig_fn = px.pie(
                fn_data,
                values="Count",
                names="Status",
                color_discrete_sequence=color_palette[: len(fn_data)],
            )
            fig_fn.update_traces(
                textinfo="percent", texttemplate="%{percent}", textfont_size=36
            )
            fig_fn.update_layout(
                font=dict(size=16),
                width=500,
                height=500,
                showlegend=False,
                margin=dict(t=20, b=20, l=20, r=20),
            )
            charts["false_negatives"] = fig_fn

            # Save as PNG
            if output_dir:
                png_path = os.path.join(
                    output_dir, "false_negatives_by_ground_truth_status.png"
                )
                fig_fn.write_image(png_path, width=500, height=500, scale=2)
                print(f"Saved False Negatives chart to: {png_path}")

    return charts


def create_cross_classification_chart(
    composition: Dict, output_dir: str = None
) -> go.Figure:
    """
    Create a separate 4-way cross-classification pie chart and save as PNG.
    Chart has no title or legend for clean appearance.

    Args:
        composition: Composition analysis results
        output_dir: Directory to save chart (optional)

    Returns:
        Plotly figure object
    """
    cross_stats = composition["true_positives_cross_classification"]

    if cross_stats["total"] == 0:
        print("No cross-classification data available")
        return None

    cross_data = pd.DataFrame(
        {
            "Classification": [
                "Direct → Direct",
                "Direct → Implied",
                "Implied → Direct",
                "Implied → Implied",
            ],
            "Count": [
                cross_stats["predicted_direct_was_direct_in_gt"],
                cross_stats["predicted_direct_was_implied_in_gt"],
                cross_stats["predicted_implied_was_direct_in_gt"],
                cross_stats["predicted_implied_was_implied_in_gt"],
            ],
            "Percentage": [
                composition["percentages"]["true_positives_cross_classification"][
                    "predicted_direct_was_direct_in_gt_pct"
                ],
                composition["percentages"]["true_positives_cross_classification"][
                    "predicted_direct_was_implied_in_gt_pct"
                ],
                composition["percentages"]["true_positives_cross_classification"][
                    "predicted_implied_was_direct_in_gt_pct"
                ],
                composition["percentages"]["true_positives_cross_classification"][
                    "predicted_implied_was_implied_in_gt_pct"
                ],
            ],
        }
    )

    # Remove rows with 0 count
    cross_data = cross_data[cross_data["Count"] > 0]

    if cross_data.empty:
        print("No non-zero cross-classification data available")
        return None

    # Use a 4-color palette for cross-classification
    cross_colors = [
        "rgb(103,122,165)",  # Direct→Direct (Grayish-blue)
        "rgb(151,172,195)",  # Direct→Implied (Light grayish-blue)
        "rgb(252,201,158)",  # Implied→Direct (Light orange)
        "rgb(252,141,98)",  # Implied→Implied (Orange)
    ][: len(cross_data)]

    fig_cross = px.pie(
        cross_data,
        values="Count",
        names="Classification",
        color_discrete_sequence=cross_colors,
    )
    fig_cross.update_traces(
        textinfo="percent", texttemplate="%{percent}", textfont_size=48
    )
    fig_cross.update_layout(
        font=dict(size=14),
        width=600,
        height=600,
        showlegend=False,
        margin=dict(t=20, b=20, l=20, r=20),
    )

    # Save as PNG
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        png_path = os.path.join(output_dir, "true_positives_cross_classification.png")
        fig_cross.write_image(png_path, width=600, height=600, scale=2)
        print(f"Saved Cross-Classification chart to: {png_path}")

    return fig_cross


def create_legend_charts(composition: Dict, output_dir: str = None) -> Dict:
    """
    Create separate legend charts for the pie charts.

    Args:
        composition: Composition analysis results
        output_dir: Directory to save legend charts (optional)

    Returns:
        Dictionary with legend figures
    """
    if not output_dir:
        return {}

    os.makedirs(output_dir, exist_ok=True)
    legend_charts = {}

    # Color palette
    color_palette = [
        "rgb(103,122,165)",  # Grayish-blue (for Direct)
        "rgb(252,141,98)",  # Orange (for Implied)
    ]

    # Legend for True Positives and False Positives (Direct vs Implied predictions)
    fig_legend1 = go.Figure()

    # Add invisible traces just for legend
    fig_legend1.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(size=15, color=color_palette[0]),
            name="Direct Phenotype",
            showlegend=True,
        )
    )
    fig_legend1.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(size=15, color=color_palette[1]),
            name="Implied Phenotype",
            showlegend=True,
        )
    )

    fig_legend1.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        width=300,
        height=100,
        margin=dict(t=10, b=10, l=10, r=10),
        legend=dict(orientation="v", x=0, y=1, font=dict(size=14)),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    legend_charts["prediction_status_legend"] = fig_legend1
    legend_path1 = os.path.join(output_dir, "legend_prediction_status.png")
    fig_legend1.write_image(legend_path1, width=300, height=100, scale=2)
    print(f"Saved Prediction Status legend to: {legend_path1}")

    # Legend for False Negatives (Ground Truth status)
    fig_legend2 = go.Figure()

    fig_legend2.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(size=15, color=color_palette[0]),
            name="Direct Phenotype",
            showlegend=True,
        )
    )
    fig_legend2.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(size=15, color=color_palette[1]),
            name="Implied Phenotype",
            showlegend=True,
        )
    )

    fig_legend2.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        width=300,
        height=100,
        margin=dict(t=10, b=10, l=10, r=10),
        legend=dict(orientation="v", x=0, y=1, font=dict(size=14)),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    legend_charts["ground_truth_status_legend"] = fig_legend2
    legend_path2 = os.path.join(output_dir, "legend_ground_truth_status.png")
    fig_legend2.write_image(legend_path2, width=300, height=100, scale=2)
    print(f"Saved Ground Truth Status legend to: {legend_path2}")

    # Legend for Cross-Classification (4-way)
    cross_colors = [
        "rgb(103,122,165)",  # Direct→Direct (Grayish-blue)
        "rgb(151,172,195)",  # Direct→Implied (Light grayish-blue)
        "rgb(252,201,158)",  # Implied→Direct (Light orange)
        "rgb(252,141,98)",  # Implied→Implied (Orange)
    ]

    cross_labels = [
        "Direct → Direct",
        "Direct → Implied",
        "Implied → Direct",
        "Implied → Implied",
    ]

    fig_legend3 = go.Figure()

    for i, (color, label) in enumerate(zip(cross_colors, cross_labels)):
        fig_legend3.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(size=15, color=color),
                name=label,
                showlegend=True,
            )
        )

    fig_legend3.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        width=400,
        height=150,
        margin=dict(t=10, b=10, l=10, r=10),
        legend=dict(orientation="v", x=0, y=1, font=dict(size=12)),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    legend_charts["cross_classification_legend"] = fig_legend3
    legend_path3 = os.path.join(output_dir, "legend_cross_classification.png")
    fig_legend3.write_image(legend_path3, width=400, height=150, scale=2)
    print(f"Saved Cross-Classification legend to: {legend_path3}")

    return legend_charts


def create_all_composition_charts(composition: Dict, output_dir: str = None) -> Dict:
    """
    Create all composition charts and save them as individual PNG files.

    Args:
        composition: Composition analysis results
        output_dir: Directory to save charts (optional)

    Returns:
        Dictionary with all chart figures
    """
    print(f"\n📊 Creating composition pie charts...")

    # Create the three main charts (no titles, no legends)
    charts = create_composition_pie_charts(composition, output_dir)

    # Create the cross-classification chart separately (no title, no legend)
    cross_chart = create_cross_classification_chart(composition, output_dir)
    if cross_chart:
        charts["cross_classification"] = cross_chart

    # Create separate legend charts
    if output_dir:
        legend_charts = create_legend_charts(composition, output_dir)
        charts.update(legend_charts)

    if charts:
        print(f"\nCreated {len(charts)} charts (including legends):")
        for chart_name in charts.keys():
            if "legend" in chart_name:
                print(f"  - {chart_name.replace('_', ' ').title()}")
            else:
                print(f"  - {chart_name.replace('_', ' ').title()} Chart")

        if output_dir:
            print(f"\nAll charts saved as PNG files to: {output_dir}")
            print("\nNote: To use these charts, you'll need to install kaleido:")
            print("  pip install kaleido")
    else:
        print("No charts created (insufficient data)")

    return charts


def print_composition_analysis_with_charts(composition: Dict, output_dir: str = None):
    """Print detailed composition analysis with rounded percentages and create PNG charts."""
    print(f"\n" + "=" * 80)
    print("📊 ERROR COMPOSITION ANALYSIS")
    print("=" * 80)

    # True Positives composition
    tp_stats = composition["true_positives_by_prediction_status"]
    tp_pcts = composition["percentages"]["true_positives"]

    print(f"\n✅ TRUE POSITIVES COMPOSITION (by prediction status):")
    print(f"  Total True Positives: {tp_stats['total']}")
    print(
        f"  Predicted as Direct:   {tp_stats['predicted_as_direct']} ({tp_pcts['predicted_as_direct_pct']:.1f}%)"
    )
    print(
        f"  Predicted as Implied:  {tp_stats['predicted_as_implied']} ({tp_pcts['predicted_as_implied_pct']:.1f}%)"
    )

    if tp_stats["total"] > 0:
        print(
            f"\n  💡 Insight: {tp_pcts['predicted_as_direct_pct']:.1f}% of correct predictions were classified as direct phenotypes"
        )

    # False Positives composition
    fp_stats = composition["false_positives_by_prediction_status"]
    fp_pcts = composition["percentages"]["false_positives"]

    print(f"\n❌ FALSE POSITIVES COMPOSITION (by prediction status):")
    print(f"  Total False Positives: {fp_stats['total']}")
    print(
        f"  Predicted as Direct:   {fp_stats['predicted_as_direct']} ({fp_pcts['predicted_as_direct_pct']:.1f}%)"
    )
    print(
        f"  Predicted as Implied:  {fp_stats['predicted_as_implied']} ({fp_pcts['predicted_as_implied_pct']:.1f}%)"
    )

    if fp_stats["total"] > 0:
        print(
            f"\n  💡 Insight: {fp_pcts['predicted_as_direct_pct']:.1f}% of false positives were incorrectly predicted as direct phenotypes"
        )

    # False Negatives composition
    fn_stats = composition["false_negatives_by_ground_truth_status"]
    fn_pcts = composition["percentages"]["false_negatives"]

    print(f"\n⚠️  FALSE NEGATIVES COMPOSITION (by ground truth status):")
    print(f"  Total False Negatives: {fn_stats['total']}")
    print(
        f"  Direct in GT:   {fn_stats['direct_in_gt']} ({fn_pcts['direct_in_gt_pct']:.1f}%)"
    )
    print(
        f"  Implied in GT:  {fn_stats['implied_in_gt']} ({fn_pcts['implied_in_gt_pct']:.1f}%)"
    )

    if fn_stats["total"] > 0:
        print(
            f"\n  💡 Insight: {fn_pcts['direct_in_gt_pct']:.1f}% of missed phenotypes were direct in ground truth"
        )

    # True Positives cross-classification
    tp_cross_stats = composition["true_positives_cross_classification"]
    tp_cross_pcts = composition["percentages"]["true_positives_cross_classification"]

    print(f"\n✅ TRUE POSITIVES CROSS-CLASSIFICATION:")
    print(f"  Total True Positives: {tp_cross_stats['total']}")
    print(
        f"  📋→📋 Predicted Direct, was Direct in GT:   {tp_cross_stats['predicted_direct_was_direct_in_gt']} ({tp_cross_pcts['predicted_direct_was_direct_in_gt_pct']:.1f}%)"
    )
    print(
        f"  📋→🔍 Predicted Direct, was Implied in GT:  {tp_cross_stats['predicted_direct_was_implied_in_gt']} ({tp_cross_pcts['predicted_direct_was_implied_in_gt_pct']:.1f}%)"
    )
    print(
        f"  🔍→📋 Predicted Implied, was Direct in GT:  {tp_cross_stats['predicted_implied_was_direct_in_gt']} ({tp_cross_pcts['predicted_implied_was_direct_in_gt_pct']:.1f}%)"
    )
    print(
        f"  🔍→🔍 Predicted Implied, was Implied in GT: {tp_cross_stats['predicted_implied_was_implied_in_gt']} ({tp_cross_pcts['predicted_implied_was_implied_in_gt_pct']:.1f}%)"
    )

    if tp_cross_stats["total"] > 0:
        # Calculate status agreement
        perfect_agreement = (
            tp_cross_stats["predicted_direct_was_direct_in_gt"]
            + tp_cross_stats["predicted_implied_was_implied_in_gt"]
        )
        status_agreement_pct = (perfect_agreement / tp_cross_stats["total"]) * 100

        print(
            f"\n  💡 Status Classification Accuracy: {perfect_agreement}/{tp_cross_stats['total']} ({status_agreement_pct:.1f}%) predictions had correct direct/implied classification"
        )

        if tp_cross_pcts["predicted_direct_was_implied_in_gt_pct"] > 20:
            print(
                f"  ⚠️  Warning: {tp_cross_pcts['predicted_direct_was_implied_in_gt_pct']:.1f}% of correct predictions were classified as direct when they were implied in GT"
            )

        if tp_cross_pcts["predicted_implied_was_direct_in_gt_pct"] > 20:
            print(
                f"  ⚠️  Warning: {tp_cross_pcts['predicted_implied_was_direct_in_gt_pct']:.1f}% of correct predictions were classified as implied when they were direct in GT"
            )

    # Create and save charts
    if output_dir:
        charts = create_all_composition_charts(composition, output_dir)
        return charts
    else:
        print(
            f"\n📊 To create PNG charts, provide an output directory when calling this function."
        )
        return {}


def main():
    """Main function with updated chart creation."""
    parser = argparse.ArgumentParser(
        description="FIXED: Analyze prediction accuracy by direct vs implied status"
    )

    parser.add_argument(
        "--predictions",
        default="data/results/agents/hpo/step3/verified_lab_test_matched.json",
        help="Path to predictions JSON file",
    )
    parser.add_argument(
        "--implied-phenotypes",
        default="data/dataset/implied_phenotypes.json",
        help="Path to implied phenotypes JSON file (used as ground truth)",
    )
    parser.add_argument(
        "--output",
        default="data/results/agents/hpo/step3/fixed_prediction_accuracy_analysis.json",
        help="Path to save analysis results",
    )
    parser.add_argument(
        "--charts-dir",
        default="figs/",
        help="Directory to save pie charts (optional)",
    )

    args = parser.parse_args()

    # Load data files
    print("Loading data files...")
    predictions_data = load_json_file(args.predictions)
    implied_data = load_json_file(args.implied_phenotypes)

    if not all([predictions_data, implied_data]):
        print("Error: One or more files could not be loaded. Exiting.")
        return

    # Extract data structures
    print("Extracting phenotype data...")
    predictions_dict, prediction_phenotype_names, prediction_contexts = (
        extract_predictions_with_status(predictions_data)
    )
    all_ground_truth_dict, direct_ground_truth_dict, implied_ground_truth_dict = (
        extract_ground_truth_from_implied_data(implied_data)
    )

    print(f"Loaded {len(predictions_dict)} cases with predictions")
    print(f"Loaded {len(all_ground_truth_dict)} cases with ground truth")

    # Perform FIXED analysis
    print("Analyzing prediction accuracy by status...")
    results = categorize_errors_by_predicted_status(
        predictions_dict,
        prediction_phenotype_names,
        prediction_contexts,
        all_ground_truth_dict,
        direct_ground_truth_dict,
        implied_ground_truth_dict,
    )

    # Calculate metrics
    metrics = calculate_prediction_accuracy_metrics(results)
    results["prediction_accuracy_metrics"] = metrics

    # Calculate composition analysis
    composition = analyze_error_composition(
        results,
        all_ground_truth_dict,
        direct_ground_truth_dict,
        implied_ground_truth_dict,
    )
    results["error_composition"] = composition

    # Print summary with rounded percentages
    print_fixed_analysis_summary(results, metrics)

    # Print composition analysis and create charts
    charts = print_composition_analysis_with_charts(composition, args.charts_dir)

    # Show detailed examples
    show_detailed_examples(
        results, implied_data, prediction_phenotype_names, prediction_contexts
    )

    # Save results
    print(f"\nSaving results to {args.output}")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    # Convert Counter objects to regular dicts for JSON serialization
    for pattern_type in results["error_patterns"]:
        if isinstance(results["error_patterns"][pattern_type], Counter):
            results["error_patterns"][pattern_type] = dict(
                results["error_patterns"][pattern_type]
            )

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print("Fixed analysis complete!")


if __name__ == "__main__":
    main()
