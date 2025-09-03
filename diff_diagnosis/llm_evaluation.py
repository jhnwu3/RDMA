#!/usr/bin/env python3
"""
Command Line LLM-based Evaluation Script for Rare Disease Diagnosis Benchmark Results
Uses an LLM as a judge to evaluate disease matches more robustly than string matching.
Supports processing multiple benchmark files.
"""

import argparse
import json
import os
import sys
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import re
import glob
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from rdma.utils.llm_client import LocalLLMClient
from rdma.utils.abbreviation_detector import AbbreviationDetector


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
        raise Exception(f"Error loading benchmark results from {filepath}: {e}")


def expand_all_abbreviation_terms(
    abbreviation_detector: AbbreviationDetector, disease_list: List[str]
):
    """Expand abbreviations in disease list"""
    new_dis_list = []
    for dis in disease_list:
        abbr_res = abbreviation_detector.check_abbreviation(dis)
        if abbr_res["is_abbreviation"]:
            new_dis_list.append(abbr_res["expanded_term"])
        else:
            new_dis_list.append(dis)
    return new_dis_list


def evaluate_benchmark_with_llm(
    benchmark_data: Dict[str, Any],
    llm_client,
    k_values: List[int] = [1, 5, 10],
    verbose: bool = False,
    abbreviation_detector: AbbreviationDetector = None,
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
        if abbreviation_detector:
            observed_diseases = expand_all_abbreviation_terms(
                abbreviation_detector, observed_diseases
            )

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

        if abbreviation_detector:
            predicted_diseases = expand_all_abbreviation_terms(
                abbreviation_detector, predicted_diseases
            )

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


def print_evaluation_results(
    results: Dict[str, Any], model_type: str, filename: str = None
) -> None:
    """Print evaluation results in a formatted way"""

    print("\n" + "=" * 80)
    if filename:
        print(f"LLM-BASED EVALUATION RESULTS - {model_type.upper()} ({filename})")
    else:
        print(f"LLM-BASED EVALUATION RESULTS - {model_type.upper()}")
    print("=" * 80)
    print(f"Total patients evaluated: {results['total_patients']}")
    print(f"Cases requiring extraction: {results['extraction_needed']}")
    print(f"Extraction success rate: {results['extraction_success_rate']:.2%}")
    print("-" * 80)

    # Get k values from results
    k_values = []
    for key in results.keys():
        if key.startswith("hit_at_") and key.endswith("_rate"):
            k = key.replace("hit_at_", "").replace("_rate", "")
            if k.isdigit():
                k_values.append(int(k))
    k_values.sort()

    # Print hit rates for each k
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


def find_benchmark_files(input_path: str) -> List[str]:
    """Find benchmark files from input path (file, directory, or glob pattern)"""
    files = []

    if os.path.isfile(input_path):
        # Single file
        files.append(input_path)
    elif os.path.isdir(input_path):
        # Directory - find all JSON files
        files.extend(glob.glob(os.path.join(input_path, "*.json")))
    else:
        # Glob pattern
        files.extend(glob.glob(input_path))

    # Filter out evaluation result files to avoid processing them
    files = [f for f in files if "llm_evaluation" not in os.path.basename(f)]

    if not files:
        raise Exception(f"No benchmark files found at: {input_path}")

    return sorted(files)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="LLM-based evaluation of rare disease diagnosis benchmark results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input arguments
    parser.add_argument(
        "input",
        help="Benchmark results file, directory, or glob pattern (e.g., 'data/*.json')",
    )

    # Model configuration
    parser.add_argument(
        "--evaluator-model", default="mistral_24b", help="Model to use for evaluation"
    )
    parser.add_argument("--device", default="cuda:0", help="Device to run the model on")
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0001,
        help="Temperature for model inference",
    )

    # Evaluation configuration
    parser.add_argument(
        "--k-values",
        nargs="+",
        type=int,
        default=[1, 5, 10],
        help="K values to evaluate (e.g., --k-values 1 5 10)",
    )

    # Output configuration
    parser.add_argument(
        "--output-dir",
        default="data/differential_diagnosis/eval",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--no-save", action="store_true", help="Don't save evaluation results to file"
    )

    # Debugging options
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )
    parser.add_argument(
        "--no-abbreviation-expansion",
        action="store_true",
        help="Disable abbreviation expansion",
    )

    return parser.parse_args()


def main():
    """Main function"""
    args = parse_args()

    # Find all benchmark files to process
    try:
        benchmark_files = find_benchmark_files(args.input)
        print(f"Found {len(benchmark_files)} benchmark files to process:")
        for f in benchmark_files:
            print(f"  - {f}")
        print()
    except Exception as e:
        print(f"Error finding benchmark files: {e}")
        sys.exit(1)

    # Initialize LLM client
    print(f"Initializing LLM client with model: {args.evaluator_model}")
    try:
        llm_client = LocalLLMClient(
            model_type=args.evaluator_model,
            device=args.device,
            temperature=args.temperature,
        )
        print("LLM client initialized successfully\n")
    except Exception as e:
        print(f"Error initializing LLM client: {e}")
        sys.exit(1)

    # Initialize abbreviation detector if needed
    abbreviation_detector = None
    if not args.no_abbreviation_expansion:
        try:
            abbreviation_detector = AbbreviationDetector(debug=args.verbose)
            if args.verbose:
                print("Abbreviation detector initialized")
        except Exception as e:
            print(f"Warning: Could not initialize abbreviation detector: {e}")
            print("Continuing without abbreviation expansion...")

    # Process each benchmark file
    all_results = []

    for benchmark_file in benchmark_files:
        print(f"Processing: {benchmark_file}")
        print("-" * 60)

        try:
            # Load benchmark results
            benchmark_data = load_benchmark_results(benchmark_file)
            patient_count = len(benchmark_data.get("patient_results", {}))
            original_model = benchmark_data.get("model_type", "unknown")

            if args.verbose:
                print(
                    f"Loaded results for {patient_count} patients (original model: {original_model})"
                )

            # Run evaluation
            print(f"Starting LLM-based evaluation with k values: {args.k_values}...")
            results = evaluate_benchmark_with_llm(
                benchmark_data=benchmark_data,
                llm_client=llm_client,
                k_values=args.k_values,
                verbose=args.verbose,
                abbreviation_detector=abbreviation_detector,
            )

            # Print results
            model_identifier = f"{original_model}_evaluated_by_{args.evaluator_model}"
            filename = os.path.basename(benchmark_file)
            print_evaluation_results(results, model_identifier, filename)

            # Save results if requested
            if not args.no_save:
                try:
                    filepath = save_evaluation_results(
                        results,
                        model_identifier,
                        args.output_dir,
                    )
                    print(f"Evaluation results saved to: {filepath}")
                except Exception as e:
                    print(f"Warning: Could not save results: {e}")

            # Store results for summary
            all_results.append(
                {
                    "file": benchmark_file,
                    "original_model": original_model,
                    "results": results,
                }
            )

        except Exception as e:
            print(f"Error processing {benchmark_file}: {e}")
            continue

        print("\n" + "=" * 80 + "\n")

    # Print summary if multiple files were processed
    if len(all_results) > 1:
        print("SUMMARY OF ALL EVALUATIONS")
        print("=" * 80)

        for result_data in all_results:
            filename = os.path.basename(result_data["file"])
            results = result_data["results"]
            original_model = result_data["original_model"]

            print(f"File: {filename} (Model: {original_model})")
            print(f"  Patients: {results['total_patients']}")

            for k in args.k_values:
                if f"patient_hit_at_{k}_rate" in results:
                    rate = results[f"patient_hit_at_{k}_rate"]
                    print(f"  Hit@{k} (Patient-level): {rate:.2%}")
            print("-" * 60)

        print("=" * 80)

    print(f"Evaluation complete! Processed {len(all_results)} files successfully.")


if __name__ == "__main__":
    main()
