#!/usr/bin/env python3
"""
LLM-based Evaluation Script for Rare Disease Diagnosis Benchmark Results
Uses an LLM as a judge to evaluate disease matches more robustly than string matching.
"""

import argparse
import json
import os
import sys
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import re

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from rdma.utils.llm_client import LocalLLMClient


class LLMEvaluator:
    """LLM-based evaluator for disease diagnosis matches"""

    def __init__(self, llm_client, verbose: bool = False):
        self.llm_client = llm_client
        self.verbose = verbose

    def extract_diseases_from_response(
        self, raw_response: str, observed_diseases: List[str]
    ) -> List[str]:
        """
        Extract predicted diseases from raw LLM response when parsing failed
        """
        extraction_prompt = f"""You are tasked with extracting rare disease names from an LLM's response about medical diagnosis.

The LLM was asked to predict rare diseases based on patient phenotypes. Here is the raw response:

"{raw_response}"

Please extract ONLY the disease names that the LLM mentioned as predictions/diagnoses. Do not include:
- Phenotypes or symptoms
- General medical terms
- Explanatory text

Return your response as a JSON list of disease names in the order they appear:
["Disease Name 1", "Disease Name 2", ...]

If no diseases can be identified, return: []

Your response:"""

        try:
            response = self.llm_client.query(
                system_message="You are a medical text extraction expert. Extract only disease names from medical text.",
                user_input=extraction_prompt,
            )

            # Parse the JSON response
            response = response.strip()
            if response.startswith("[") and response.endswith("]"):
                diseases = json.loads(response)
                if isinstance(diseases, list):
                    return diseases

            # If not properly formatted, try regex extraction
            match = re.search(r"\[(.*?)\]", response, re.DOTALL)
            if match:
                diseases_str = match.group(1)
                diseases = []
                for item in diseases_str.split(","):
                    item = item.strip().strip("\"'")
                    if item:
                        diseases.append(item)
                return diseases

        except Exception as e:
            if self.verbose:
                print(f"Error extracting diseases: {e}")

        return []

    def evaluate_disease_match(
        self, predicted_diseases: List[str], observed_diseases: List[str], k: int
    ) -> Tuple[bool, str]:
        """
        Use LLM to evaluate if any of the top-k predicted diseases match any observed diseases

        Returns:
            Tuple of (has_match, explanation)
        """
        # Take only top-k predicted diseases
        top_k_predicted = (
            predicted_diseases[:k]
            if len(predicted_diseases) >= k
            else predicted_diseases
        )

        if not top_k_predicted or not observed_diseases:
            return False, "Empty disease lists"

        evaluation_prompt = f"""You are a medical expert evaluating whether predicted rare diseases match observed/actual diagnoses.

PREDICTED DISEASES (Top {k}):
{chr(10).join([f"- {disease}" for disease in top_k_predicted])}

OBSERVED/ACTUAL DISEASES:
{chr(10).join([f"- {disease}" for disease in observed_diseases])}

Question: Do ANY of the predicted diseases match ANY of the observed diseases?

Consider diseases as matching if they:
1. Are exactly the same disease name
2. Refer to the same medical condition but with different terminology (e.g., "Marfan Syndrome" vs "Marfan's Disease")
3. Are synonyms or alternative names for the same condition
4. One is a more specific subtype of the other (e.g., "Type 1 Diabetes" matches "Diabetes Mellitus Type 1")

Do NOT consider as matches:
1. Diseases that are completely different conditions
2. General categories vs specific diseases (unless clearly the same condition)
3. Similar-sounding but medically distinct conditions

Respond with EXACTLY this format:
MATCH: [YES/NO]
EXPLANATION: [Brief explanation of your reasoning]

Your response:"""

        try:
            response = self.llm_client.query(
                system_message="You are a medical expert specializing in rare disease diagnosis evaluation.",
                user_input=evaluation_prompt,
            )

            # Parse the response
            match_pattern = r"MATCH:\s*(YES|NO)"
            explanation_pattern = r"EXPLANATION:\s*(.*?)(?:\n|$)"

            match_result = re.search(match_pattern, response, re.IGNORECASE)
            explanation_result = re.search(
                explanation_pattern, response, re.IGNORECASE | re.DOTALL
            )

            if match_result:
                has_match = match_result.group(1).upper() == "YES"
                explanation = (
                    explanation_result.group(1).strip()
                    if explanation_result
                    else "No explanation provided"
                )
                return has_match, explanation
            else:
                # Fallback: look for yes/no anywhere in the response
                if "yes" in response.lower():
                    return True, "Fallback parsing - found 'yes' in response"
                elif "no" in response.lower():
                    return False, "Fallback parsing - found 'no' in response"
                else:
                    return False, "Could not parse LLM evaluation response"

        except Exception as e:
            if self.verbose:
                print(f"Error in disease evaluation: {e}")
            return False, f"Evaluation error: {str(e)}"


def load_benchmark_results(filepath: str) -> Dict[str, Any]:
    """Load benchmark results from JSON file"""
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
        return data
    except Exception as e:
        raise Exception(f"Error loading benchmark results: {e}")


def evaluate_benchmark_with_llm(
    benchmark_data: Dict[str, Any],
    llm_client,
    k_values: List[int] = [1, 5, 10],
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Evaluate benchmark results using LLM as judge

    Args:
        benchmark_data: Loaded benchmark results
        llm_client: LLM client for evaluation
        k_values: List of k values to evaluate (e.g., [1, 5, 10])
        verbose: Whether to print detailed results

    Returns:
        Dictionary with evaluation results
    """

    evaluator = LLMEvaluator(llm_client, verbose)
    patient_results = benchmark_data.get("patient_results", {})

    if not patient_results:
        raise Exception("No patient results found in benchmark data")

    # Initialize counters for each k value
    results = {}
    for k in k_values:
        results[f"hit_at_{k}"] = 0
        results[f"total_at_{k}"] = 0
        results[f"patients_with_hit_at_{k}"] = 0

    results["total_patients"] = 0
    results["extraction_needed"] = 0
    results["extraction_successful"] = 0
    results["evaluation_errors"] = 0

    # Store detailed results for each patient
    detailed_results = {}

    if verbose:
        print(f"Evaluating {len(patient_results)} patients with LLM judge...")
        print("=" * 80)

    for patient_id, patient_data in patient_results.items():
        results["total_patients"] += 1

        observed_diseases = patient_data.get("observed_diseases", [])
        if not observed_diseases:
            if verbose:
                print(f"Patient {patient_id}: No observed diseases, skipping")
            continue

        # Get predicted diseases
        predicted_diseases = patient_data.get("predicted_diseases", [])
        raw_response = patient_data.get("raw_llm_response", "")

        # If predicted diseases is empty or parsing failed, try extraction
        if not predicted_diseases and raw_response:
            if verbose:
                print(f"Patient {patient_id}: Extracting diseases from raw response")

            results["extraction_needed"] += 1
            predicted_diseases = evaluator.extract_diseases_from_response(
                raw_response, observed_diseases
            )

            if predicted_diseases:
                results["extraction_successful"] += 1
                if verbose:
                    print(f"  Extracted diseases: {predicted_diseases}")
            else:
                if verbose:
                    print(f"  Extraction failed")

        if not predicted_diseases:
            if verbose:
                print(
                    f"Patient {patient_id}: No predicted diseases available, skipping"
                )
            continue

        if verbose:
            print(f"\nPatient {patient_id}:")
            print(f"  Observed diseases: {observed_diseases}")
            print(f"  Predicted diseases: {predicted_diseases}")

        # Evaluate for each k value
        patient_hits = {}
        patient_evaluations = {}

        for k in k_values:
            results[f"total_at_{k}"] += len(observed_diseases)

            # Use LLM to evaluate match
            has_match, explanation = evaluator.evaluate_disease_match(
                predicted_diseases, observed_diseases, k
            )

            patient_evaluations[f"k_{k}"] = {
                "has_match": has_match,
                "explanation": explanation,
                "top_k_predicted": predicted_diseases[:k],
            }

            if has_match:
                results[f"hit_at_{k}"] += len(
                    observed_diseases
                )  # Count all observed diseases as hits
                patient_hits[f"hit_at_{k}"] = True
                if verbose:
                    print(f"  Hit@{k}: YES - {explanation}")
            else:
                patient_hits[f"hit_at_{k}"] = False
                if verbose:
                    print(f"  Hit@{k}: NO - {explanation}")

        # Count patients with at least one hit at each k
        for k in k_values:
            if patient_hits.get(f"hit_at_{k}", False):
                results[f"patients_with_hit_at_{k}"] += 1

        # Store detailed results
        detailed_results[patient_id] = {
            "observed_diseases": observed_diseases,
            "predicted_diseases": predicted_diseases,
            "evaluations": patient_evaluations,
            "hits": patient_hits,
            "extraction_used": not bool(patient_data.get("predicted_diseases", [])),
        }

        if verbose:
            print("-" * 80)

    # Calculate rates
    for k in k_values:
        total = results[f"total_at_{k}"]
        if total > 0:
            results[f"hit_at_{k}_rate"] = results[f"hit_at_{k}"] / total
            results[f"patient_hit_at_{k}_rate"] = (
                results[f"patients_with_hit_at_{k}"] / results["total_patients"]
            )
        else:
            results[f"hit_at_{k}_rate"] = 0
            results[f"patient_hit_at_{k}_rate"] = 0

    # Add extraction success rate
    if results["extraction_needed"] > 0:
        results["extraction_success_rate"] = (
            results["extraction_successful"] / results["extraction_needed"]
        )
    else:
        results["extraction_success_rate"] = 0

    results["detailed_results"] = detailed_results

    return results


def print_evaluation_results(results: Dict[str, Any], model_type: str) -> None:
    """Print evaluation results in a formatted way"""

    print("\n" + "=" * 80)
    print(f"LLM-BASED EVALUATION RESULTS - {model_type.upper()}")
    print("=" * 80)
    print(f"Total patients evaluated: {results['total_patients']}")
    print(f"Cases requiring extraction: {results['extraction_needed']}")
    print(f"Extraction success rate: {results['extraction_success_rate']:.2%}")
    print("-" * 80)

    # Print hit rates for each k
    k_values = [1, 5, 10]  # Assuming these are the k values used
    for k in k_values:
        if f"hit_at_{k}_rate" in results:
            disease_hits = results[f"hit_at_{k}"]
            total_diseases = results[f"total_at_{k}"]
            disease_rate = results[f"hit_at_{k}_rate"]

            patient_hits = results[f"patients_with_hit_at_{k}"]
            total_patients = results["total_patients"]
            patient_rate = results[f"patient_hit_at_{k}_rate"]

            print(
                f"Hit@{k} Rate (Disease-level): {disease_rate:.2%} ({disease_hits}/{total_diseases})"
            )
            print(
                f"Hit@{k} Rate (Patient-level): {patient_rate:.2%} ({patient_hits}/{total_patients})"
            )
            print("-" * 40)

    print("=" * 80)


def save_evaluation_results(
    results: Dict[str, Any], model_type: str, output_dir: str = "llm_evaluation_results"
) -> str:
    """Save evaluation results to JSON file"""

    os.makedirs(output_dir, exist_ok=True)

    try:
        import pandas as pd

        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    except ImportError:
        import datetime

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    filename = f"{model_type}_llm_evaluation_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)

    # Add metadata
    results_with_metadata = {
        "model_type": model_type,
        "evaluation_method": "LLM-based",
        "timestamp": timestamp,
        "summary_metrics": {
            k: v for k, v in results.items() if k != "detailed_results"
        },
        "detailed_results": results["detailed_results"],
    }

    with open(filepath, "w") as f:
        json.dump(results_with_metadata, f, indent=2)

    return filepath


def main():
    """Main function to run LLM-based evaluation"""

    parser = argparse.ArgumentParser(
        description="Evaluate rare disease diagnosis benchmark results using LLM as judge",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script takes benchmark results from the benchmarking script and uses an LLM
to evaluate whether predicted diseases match observed diseases. This approach is
more robust than exact string matching as it can identify semantic matches.

Examples:
  python llm_evaluation.py --results_file benchmark_results/llama3_8b_benchmark_20241201_123456.json --evaluator_model llama3_70b --verbose
  python llm_evaluation.py --results_file results.json --evaluator_model qwen_70b --k_values 1 5 10 --save_results
        """,
    )

    # Required arguments
    parser.add_argument(
        "--results_file",
        type=str,
        required=True,
        help="Path to the benchmark results JSON file",
    )

    parser.add_argument(
        "--evaluator_model",
        type=str,
        required=True,
        help="Model type to use as evaluator/judge",
    )

    # Evaluation parameters
    parser.add_argument(
        "--k_values",
        type=int,
        nargs="+",
        default=[1, 5, 10],
        help="K values to evaluate (e.g., --k_values 1 5 10)",
    )

    # Model configuration
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to run the evaluator model on",
    )

    parser.add_argument(
        "--cache_dir",
        type=str,
        default="/shared/rsaas/jw3/rare_disease/model_cache",
        help="Directory to cache models",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0001,
        help="Temperature for evaluator model",
    )

    # Output options
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed evaluation for each patient",
    )

    parser.add_argument(
        "--save_results",
        action="store_true",
        help="Save evaluation results to JSON file",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="llm_evaluation_results",
        help="Directory to save evaluation results",
    )

    args = parser.parse_args()

    # Validate results file exists
    if not os.path.exists(args.results_file):
        print(f"Error: Results file '{args.results_file}' not found.")
        sys.exit(1)

    # Load benchmark results
    print(f"Loading benchmark results from {args.results_file}...")
    try:
        benchmark_data = load_benchmark_results(args.results_file)
        patient_count = len(benchmark_data.get("patient_results", {}))
        original_model = benchmark_data.get("model_type", "unknown")
        print(
            f"Loaded results for {patient_count} patients (original model: {original_model})"
        )
    except Exception as e:
        print(f"Error loading results: {e}")
        sys.exit(1)

    # Initialize evaluator LLM
    print(f"Initializing evaluator model {args.evaluator_model} on {args.device}...")
    try:
        llm_client = LocalLLMClient(
            model_type=args.evaluator_model,
            device=args.device,
            cache_dir=args.cache_dir,
            temperature=args.temperature,
        )
        print("Evaluator model initialized successfully!")
    except Exception as e:
        print(f"Error initializing evaluator model: {e}")
        sys.exit(1)

    # Run LLM-based evaluation
    print(f"\nStarting LLM-based evaluation with k values: {args.k_values}...")
    try:
        results = evaluate_benchmark_with_llm(
            benchmark_data=benchmark_data,
            llm_client=llm_client,
            k_values=args.k_values,
            verbose=args.verbose,
        )
    except Exception as e:
        print(f"Error during evaluation: {e}")
        sys.exit(1)

    # Print results
    print_evaluation_results(
        results, f"{original_model}_evaluated_by_{args.evaluator_model}"
    )

    # Save results if requested
    if args.save_results:
        try:
            filepath = save_evaluation_results(
                results,
                f"{original_model}_evaluated_by_{args.evaluator_model}",
                args.output_dir,
            )
            print(f"\nEvaluation results saved to: {filepath}")
        except Exception as e:
            print(f"Warning: Could not save results: {e}")


if __name__ == "__main__":
    main()
