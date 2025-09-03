#!/usr/bin/env python3
"""
LLM Benchmarking Script for Rare Disease Diagnosis
Supports multiple model types from the RDMA framework.
Includes evaluation-only mode for processing existing benchmark files.
"""

import argparse
import json
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import os
import sys
import glob

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
# Import the benchmark functions from your existing code
import re
from rdma.utils.llm_client import LocalLLMClient
import pandas as pd


def parse_diseases_json(response: str) -> Tuple[List[str], str]:
    """
    Robust function to parse LLM response into explanation and diseases list

    Returns:
        Tuple of (diseases_list, explanation)
    """
    try:
        # First, try direct JSON parsing
        cleaned = response.strip()
        response_json = json.loads(cleaned)

        # Validate it has the expected structure
        if isinstance(response_json, dict):
            explanation = response_json.get("explanation", "")
            diseases_list = response_json.get("rare_diseases", [])

            # Ensure diseases_list is actually a list
            if isinstance(diseases_list, list):
                return diseases_list, explanation
            else:
                return [], explanation
        else:
            return [], ""

    except json.JSONDecodeError:
        # Try to extract JSON object using regex
        json_pattern = r"\{.*?\}"
        match = re.search(json_pattern, response, re.DOTALL)

        if match:
            try:
                json_content = match.group(0)
                response_json = json.loads(json_content)
                explanation = response_json.get("explanation", "")
                diseases_list = response_json.get("rare_diseases", [])

                if isinstance(diseases_list, list):
                    return diseases_list, explanation
                else:
                    return [], explanation
            except json.JSONDecodeError:
                pass

        # Last resort: try to extract from the response manually
        try:
            # Look for explanation and diseases patterns
            explanation_match = re.search(
                r'"explanation":\s*"([^"]*)"', response, re.DOTALL
            )
            diseases_match = re.search(
                r'"rare_diseases":\s*\[(.*?)\]', response, re.DOTALL
            )

            explanation = explanation_match.group(1) if explanation_match else ""

            if diseases_match:
                diseases_str = diseases_match.group(1)
                # Parse the diseases list
                diseases_list = []
                for item in diseases_str.split(","):
                    item = item.strip().strip("\"'")
                    if item:
                        diseases_list.append(item)
                return diseases_list, explanation
            else:
                return [], explanation

        except Exception:
            return [], ""


def normalize_disease_name(disease: str) -> str:
    """
    Normalize disease name for comparison: lowercase and strip whitespace
    """
    return disease.lower().strip()


def benchmark_rare_disease_diagnosis(
    data: Dict[str, Any],
    llm_client: Any,
    num_samples: int = None,
    verbose: bool = False,
) -> Tuple[Dict[str, float], Dict[str, Dict[str, Any]]]:
    """
    Benchmark rare disease diagnosis performance

    Args:
        data: Dictionary containing patient data with phenotypes and disease entities
        llm_client: LLM client with query method
        num_samples: Number of samples to evaluate (None for all)
        verbose: Whether to print detailed results for each case

    Returns:
        Tuple of (benchmark metrics, patient-level results)
    """

    # System prompt for LLM
    diff_diag_sys_prompt = """Given the following phenotypes, identify the top 10 most likely rare diseases and provide your reasoning.

CRITICAL: Your response must be EXACTLY in this JSON format with no additional text:

{
  "explanation": "Brief explanation of your diagnostic reasoning based on the phenotypes presented",
  "rare_diseases": ["Disease Name 1", "Disease Name 2", "Disease Name 3", "Disease Name 4", "Disease Name 5", "Disease Name 6", "Disease Name 7", "Disease Name 8", "Disease Name 9", "Disease Name 10"]
}

Rules:
1. Return ONLY the JSON object - no explanations outside the JSON, no additional text
2. Use double quotes around all strings
3. The "explanation" should be a concise summary of your diagnostic reasoning
4. The "rare_diseases" array should list diseases in order of likelihood (most likely first)

Example of correct format:
{
  "explanation": "The combination of tall stature, arachnodactyly, and lens dislocation strongly suggests connective tissue disorders, with Marfan syndrome being most likely.",
  "rare_diseases": ["Marfan Syndrome", "Ehlers-Danlos Syndrome", "Homocystinuria"]
}

Your response:"""

    # Initialize counters
    total_diseases = 0
    hits = 0
    hits_at_1 = 0
    hits_at_5 = 0
    hits_at_10 = 0
    total_patients = 0
    patients_with_hits = 0
    parsing_failures = 0

    # Dictionary to store patient-level results
    patient_results = {}

    # Get samples to process
    items_to_process = list(data.items())
    if num_samples is not None:
        items_to_process = items_to_process[:num_samples]

    if verbose:
        print(f"Evaluating {len(items_to_process)} patients...")
        print("=" * 60)

    for patient_id, patient_data in items_to_process:
        if "matched_phenotypes" not in patient_data:
            continue

        total_patients += 1
        patient_hits = 0

        # Build phenotypes string
        phenotypes = ""
        for phenotype in patient_data["matched_phenotypes"]:
            phenotypes += phenotype["phenotype"] + ", "

        # Query LLM
        phenotypes_prompt = f"Phenotypes: {phenotypes}"
        print(phenotypes_prompt)
        llm_response = llm_client.query(
            system_message=diff_diag_sys_prompt, user_input=phenotypes_prompt
        )
        print("Test:", llm_response)
        # Parse response
        predicted_diseases, explanation = parse_diseases_json(llm_response)

        if not predicted_diseases and not explanation:
            parsing_failures += 1
            if verbose:
                print(f"Patient {patient_id}: Failed to parse LLM response")
                print(f"Raw response: '{llm_response}'")

            # Still store the failed case
            patient_results[patient_id] = {
                "phenotypes": phenotypes.strip(", "),
                "predicted_diseases": [],
                "explanation": "",
                "observed_diseases": patient_data.get("disease_entities", []),
                "raw_llm_response": llm_response,
                "parsing_failed": True,
                "hits": 0,
                "hit_at_1": False,
                "hit_at_5": False,
                "hit_at_10": False,
            }
            continue

        # Normalize predicted diseases for comparison
        predicted_normalized = [normalize_disease_name(d) for d in predicted_diseases]

        # Get ground truth diseases
        observed_diseases = patient_data.get("disease_entities", [])

        if verbose:
            print(f"Patient ID: {patient_id}")
            print(f"Phenotypes: {phenotypes.strip(', ')}")
            print(f"Explanation: {explanation}")
            print(f"Predicted diseases: {predicted_diseases}")
            print(f"Observed diseases: {observed_diseases}")

        # Calculate hits for this patient
        patient_hit_details = []
        patient_hit_at_1 = False
        patient_hit_at_5 = False
        patient_hit_at_10 = False

        for observed_disease in observed_diseases:
            observed_normalized = normalize_disease_name(observed_disease)
            total_diseases += 1

            # Check if disease is in top-10 predictions
            hit_in_top10 = observed_normalized in predicted_normalized
            hit_rank = None

            if hit_in_top10:
                hits += 1
                patient_hits += 1
                hit_rank = predicted_normalized.index(observed_normalized) + 1
                hits_at_10 += 1  # If it's in the list, it's automatically in top-10

                patient_hit_details.append(
                    {"disease": observed_disease, "hit": True, "rank": hit_rank}
                )

                if verbose:
                    print(f"  ✓ Hit: '{observed_disease}' found at rank {hit_rank}")

                # Check hit@1
                if hit_rank == 1:
                    hits_at_1 += 1
                    patient_hit_at_1 = True
                    if verbose:
                        print(f"  ✓ Hit@1: '{observed_disease}' is top prediction")

                # Check hit@5
                if hit_rank <= 5:
                    hits_at_5 += 1
                    patient_hit_at_5 = True
                    if verbose and hit_rank <= 5:
                        print(
                            f"  ✓ Hit@5: '{observed_disease}' is in top 5 (rank {hit_rank})"
                        )

                # Patient hit@10 is already handled above
                patient_hit_at_10 = True
                if verbose:
                    print(
                        f"  ✓ Hit@10: '{observed_disease}' is in top 10 (rank {hit_rank})"
                    )

            else:
                patient_hit_details.append(
                    {"disease": observed_disease, "hit": False, "rank": None}
                )
                if verbose:
                    print(f"  ✗ Miss: '{observed_disease}' not found in predictions")

        if patient_hits > 0:
            patients_with_hits += 1

        # Store patient-level results
        patient_results[patient_id] = {
            "phenotypes": phenotypes.strip(", "),
            "predicted_diseases": predicted_diseases,
            "explanation": explanation,
            "observed_diseases": observed_diseases,
            "raw_llm_response": llm_response,
            "parsing_failed": False,
            "hits": patient_hits,
            "hit_at_1": patient_hit_at_1,
            "hit_at_5": patient_hit_at_5,
            "hit_at_10": patient_hit_at_10,
            "hit_details": patient_hit_details,
            "total_observed_diseases": len(observed_diseases),
        }

        if verbose:
            print(f"  Patient hits: {patient_hits}/{len(observed_diseases)}")
            print("-" * 60)

    # Calculate final metrics
    hit_rate = hits / total_diseases if total_diseases > 0 else 0
    hit_at_1_rate = hits_at_1 / total_diseases if total_diseases > 0 else 0
    hit_at_5_rate = hits_at_5 / total_diseases if total_diseases > 0 else 0
    hit_at_10_rate = hits_at_10 / total_diseases if total_diseases > 0 else 0
    patient_hit_rate = patients_with_hits / total_patients if total_patients > 0 else 0
    parsing_success_rate = (
        1 - (parsing_failures / total_patients) if total_patients > 0 else 0
    )

    results = {
        "hit_rate": hit_rate,  # Same as hit_at_10_rate for backwards compatibility
        "hit_at_1_rate": hit_at_1_rate,
        "hit_at_5_rate": hit_at_5_rate,
        "hit_at_10_rate": hit_at_10_rate,
        "patient_hit_rate": patient_hit_rate,
        "parsing_success_rate": parsing_success_rate,
        "total_diseases": total_diseases,
        "total_patients": total_patients,
        "hits": hits,
        "hits_at_1": hits_at_1,
        "hits_at_5": hits_at_5,
        "hits_at_10": hits_at_10,
        "patients_with_hits": patients_with_hits,
        "parsing_failures": parsing_failures,
    }


def evaluate_existing_benchmark(
    benchmark_file: str, verbose: bool = False
) -> Tuple[Dict[str, float], Dict[str, Dict[str, Any]]]:
    """
    Evaluate existing benchmark results file for metrics calculation
    
    Args:
        benchmark_file: Path to the JSON benchmark results file
        verbose: Whether to print detailed results for each case
        
    Returns:
        Tuple of (benchmark metrics, patient-level results)
    """
    # Load the benchmark file
    try:
        with open(benchmark_file, 'r') as f:
            benchmark_data = json.load(f)
    except Exception as e:
        raise Exception(f"Error loading benchmark file {benchmark_file}: {e}")
    
    # Extract patient results
    if "patient_results" not in benchmark_data:
        raise Exception(f"Benchmark file {benchmark_file} missing 'patient_results' key")
    
    patient_results = benchmark_data["patient_results"]
    
    # Initialize counters
    total_diseases = 0
    hits = 0
    hits_at_1 = 0
    hits_at_5 = 0
    hits_at_10 = 0
    total_patients = 0
    patients_with_hits = 0
    parsing_failures = 0
    
    # Process each patient result
    for patient_id, result in patient_results.items():
        total_patients += 1
        patient_hits = 0
        
        # Check if parsing failed
        if result.get("parsing_failed", False):
            parsing_failures += 1
            
        predicted_diseases = result.get("predicted_diseases", [])
        observed_diseases = result.get("observed_diseases", [])
        
        # Update total diseases counter
        total_diseases += len(observed_diseases)
        
        # If no observed diseases, skip hit calculation
        if not observed_diseases:
            continue
            
        # Normalize disease names for comparison
        predicted_normalized = [normalize_disease_name(d) for d in predicted_diseases]
        observed_normalized = [normalize_disease_name(d) for d in observed_diseases]
        
        # Check for hits in observed diseases
        for observed_disease in observed_diseases:
            observed_normalized_single = normalize_disease_name(observed_disease)
            
            # Check if this disease is in predicted list
            if observed_normalized_single in predicted_normalized:
                hit_rank = predicted_normalized.index(observed_normalized_single) + 1
                hits += 1
                patient_hits += 1
                
                # Check hit@k metrics
                if hit_rank == 1:
                    hits_at_1 += 1
                if hit_rank <= 5:
                    hits_at_5 += 1
                if hit_rank <= 10:
                    hits_at_10 += 1
                    
                if verbose:
                    print(f"  ✓ Hit: '{observed_disease}' found at rank {hit_rank}")
        
        # Update patient-level hit counter
        if patient_hits > 0:
            patients_with_hits += 1
            
        if verbose:
            print(f"Patient {patient_id}:")
            print(f"  Predicted: {predicted_diseases}")
            print(f"  Observed: {observed_diseases}")
            print(f"  Hits: {patient_hits}")
            print("-" * 40)
    
    # Calculate final metrics
    hit_rate = hits / total_diseases if total_diseases > 0 else 0
    hit_at_1_rate = hits_at_1 / total_diseases if total_diseases > 0 else 0
    hit_at_5_rate = hits_at_5 / total_diseases if total_diseases > 0 else 0
    hit_at_10_rate = hits_at_10 / total_diseases if total_diseases > 0 else 0
    patient_hit_rate = patients_with_hits / total_patients if total_patients > 0 else 0
    parsing_success_rate = (total_patients - parsing_failures) / total_patients if total_patients > 0 else 0
    
    results = {
        "hit_rate": hit_rate,  # Same as hit_at_10_rate for backwards compatibility
        "hit_at_1_rate": hit_at_1_rate,
        "hit_at_5_rate": hit_at_5_rate,
        "hit_at_10_rate": hit_at_10_rate,
        "patient_hit_rate": patient_hit_rate,
        "parsing_success_rate": parsing_success_rate,
        "total_diseases": total_diseases,
        "total_patients": total_patients,
        "hits": hits,
        "hits_at_1": hits_at_1,
        "hits_at_5": hits_at_5,
        "hits_at_10": hits_at_10,
        "patients_with_hits": patients_with_hits,
        "parsing_failures": parsing_failures,
    }
    
    return results, patient_results


def find_benchmark_files(input_path: str) -> List[str]:
    """Find benchmark files from input path (file, directory, or glob pattern)"""
    files = []
    
    if os.path.isfile(input_path):
        # Single file
        files.append(input_path)
    elif os.path.isdir(input_path):
        # Directory - find all JSON files that look like benchmark files
        pattern = os.path.join(input_path, "*_benchmark_*.json")
        files.extend(glob.glob(pattern))
    else:
        # Glob pattern
        files.extend(glob.glob(input_path))
    
    # Filter out evaluation result files to avoid processing them
    files = [f for f in files if "llm_evaluation" not in os.path.basename(f)]
    
    if not files:
        raise Exception(f"No benchmark files found at: {input_path}")
    
    return sorted(files)

    return results, patient_results


def print_benchmark_results(results: Dict[str, float], model_type: str) -> None:
    """
    Print benchmark results in a formatted way
    """
    print("\n" + "=" * 60)
    print(f"RARE DISEASE DIAGNOSIS BENCHMARK RESULTS - {model_type.upper()}")
    print("=" * 60)
    print(f"Total patients evaluated: {results['total_patients']}")
    print(f"Total diseases to predict: {results['total_diseases']}")
    print(f"LLM parsing success rate: {results['parsing_success_rate']:.2%}")
    print("-" * 60)
    print(
        f"Hit@1 Rate: {results['hit_at_1_rate']:.2%} ({results['hits_at_1']}/{results['total_diseases']})"
    )
    print(
        f"Hit@5 Rate: {results['hit_at_5_rate']:.2%} ({results['hits_at_5']}/{results['total_diseases']})"
    )
    print(
        f"Hit@10 Rate: {results['hit_at_10_rate']:.2%} ({results['hits_at_10']}/{results['total_diseases']})"
    )
    print(
        f"Patient Hit Rate: {results['patient_hit_rate']:.2%} ({results['patients_with_hits']}/{results['total_patients']})"
    )
    print("=" * 60)


def save_results_to_file(
    results: Dict[str, float],
    patient_results: Dict[str, Dict[str, Any]],
    model_type: str,
    output_dir: str = "benchmark_results",
) -> str:
    """
    Save benchmark results and patient-level results to a JSON file
    """
    os.makedirs(output_dir, exist_ok=True)

    # Add metadata to results
    results_with_metadata = {
        "model_type": model_type,
        "timestamp": str(pd.Timestamp.now()),
        "summary_metrics": results,
        "patient_results": patient_results,
    }

    # Create filename with timestamp
    filename = (
        f"{model_type}_benchmark_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w") as f:
        json.dump(results_with_metadata, f, indent=2)

    return filepath


def main():
    """
    Main function to run the benchmarking script
    """
    parser = argparse.ArgumentParser(
        description="Benchmark LLM performance on rare disease diagnosis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Common model types (any string accepted, will be passed to LocalLLMClient):
  llama3_70b    - Llama 3 70B Chat (70B parameters)
  llama3_8b     - Llama 3 8B Chat (8B parameters)  
  llama3_70b_2b - Llama 3 70B Chat (70B parameters)
  mistral_24b   - Mixtral 8x7B Instruct (24B parameters)
  llama3_70b_r1 - Llama 3 70B Chat R1 (70B parameters)
  qwen_70b      - Qwen2 72B Instruct (72B parameters)
  mixtral_70b   - Mixtral 8x7B Instruct (70B parameters)

Examples (Normal benchmarking mode):
  python benchmark_llm.py --model_type llama3_8b --num_samples 10 --verbose
  python benchmark_llm.py --model_type mistral_24b --data_file custom_data.json
  python benchmark_llm.py --model_type qwen_70b --device cuda:1 --save_results

Examples (Evaluation-only mode):
  python benchmark_llm.py --eval_only --benchmark_file data/differential_diagnosis/ --save_results
  python benchmark_llm.py --eval_only --benchmark_file llama3_70b_benchmark_*.json --verbose
  python benchmark_llm.py --eval_only --benchmark_file single_benchmark.json
        """,
    )

    # Required arguments (only for normal benchmarking mode)
    parser.add_argument(
        "--model_type",
        type=str,
        help="Type of LLM model to benchmark (any string, passed to LocalLLMClient) - not required for --eval_only mode",
    )

    # Data arguments
    parser.add_argument(
        "--data_file",
        type=str,
        default="data/medical_students_data/high_agreement_with_phenotypes.json",
        help="Path to the JSON data file containing patient data",
    )

    # Model configuration arguments
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to run the model on (e.g., 'cuda:0', 'cpu', 'auto')",
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
        help="Temperature for model sampling (default: 0.0001 for deterministic results)",
    )

    # Evaluation arguments
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of patient samples to evaluate (default: all)",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed results for each patient case",
    )

    # Output arguments
    parser.add_argument(
        "--save_results", action="store_true", help="Save results to a JSON file"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="benchmark_results",
        help="Directory to save benchmark results",
    )

    # List models option
    parser.add_argument(
        "--list_models",
        action="store_true",
        help="List common model types and exit (for reference only)",
    )

    # Evaluation-only mode
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Evaluation-only mode: process existing benchmark files instead of running new predictions",
    )

    parser.add_argument(
        "--benchmark_file",
        type=str,
        help="Path to benchmark file, directory, or glob pattern (required for --eval_only mode)",
    )

    args = parser.parse_args()

    # Handle list models option

    # Check if evaluation-only mode
    if args.eval_only:
        # Evaluation-only mode: process existing benchmark files
        if not args.benchmark_file:
            print("Error: --benchmark_file is required when using --eval_only mode")
            print("Example: --benchmark_file data/differential_diagnosis/")
            sys.exit(1)
        
        # Find benchmark files to process
        try:
            benchmark_files = find_benchmark_files(args.benchmark_file)
            print(f"Found {len(benchmark_files)} benchmark files to process:")
            for f in benchmark_files:
                print(f"  - {f}")
            print()
        except Exception as e:
            print(f"Error finding benchmark files: {e}")
            sys.exit(1)
        
        # Process each benchmark file
        all_results = []
        for benchmark_file in benchmark_files:
            print(f"Processing: {benchmark_file}")
            print("-" * 60)
            
            try:
                # Extract model type from filename
                basename = os.path.basename(benchmark_file)
                model_match = re.match(r"(.+?)_benchmark_", basename)
                original_model = model_match.group(1) if model_match else "unknown"
                
                # Evaluate existing benchmark
                results, patient_results = evaluate_existing_benchmark(
                    benchmark_file=benchmark_file,
                    verbose=args.verbose
                )
                
                # Print results
                print_benchmark_results(results, original_model)
                
                # Save results if requested
                if args.save_results:
                    try:
                        eval_filename = f"{original_model}_evaluation_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
                        eval_filepath = os.path.join(args.output_dir, eval_filename)
                        os.makedirs(args.output_dir, exist_ok=True)
                        
                        eval_results = {
                            "original_model": original_model,
                            "original_file": benchmark_file,
                            "timestamp": str(pd.Timestamp.now()),
                            "evaluation_metrics": results,
                        }
                        
                        with open(eval_filepath, "w") as f:
                            json.dump(eval_results, f, indent=2)
                        print(f"\nEvaluation results saved to: {eval_filepath}")
                    except Exception as e:
                        print(f"Warning: Could not save evaluation results: {e}")
                
                all_results.append({
                    "file": benchmark_file,
                    "original_model": original_model,
                    "results": results
                })
                
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
                
                print(f"Model: {original_model} (File: {filename})")
                print(f"  Patients: {results['total_patients']}")
                print(f"  Hit@1 Rate: {results['hit_at_1_rate']:.2%}")
                print(f"  Hit@5 Rate: {results['hit_at_5_rate']:.2%}")
                print(f"  Hit@10 Rate: {results['hit_at_10_rate']:.2%}")
                print(f"  Patient Hit Rate: {results['patient_hit_rate']:.2%}")
                print("-" * 60)
            
            print("=" * 80)
        
        print(f"Evaluation complete! Processed {len(all_results)} files successfully.")
        return
    
    # Normal benchmarking mode (original functionality)
    # Check that model_type is provided for normal mode
    if not args.model_type:
        print("Error: --model_type is required for normal benchmarking mode")
        print("Use --eval_only for evaluation-only mode, or provide --model_type for benchmarking")
        sys.exit(1)
    
    # Validate data file exists
    if not os.path.exists(args.data_file):
        print(f"Error: Data file '{args.data_file}' not found.")
        sys.exit(1)

    # Load data
    print(f"Loading data from {args.data_file}...")
    try:
        with open(args.data_file, "r") as f:
            data = json.load(f)
        print(f"Loaded data for {len(data)} patients")
    except Exception as e:
        print(f"Error loading data file: {e}")
        sys.exit(1)

    # Initialize LLM client
    print(f"Initializing {args.model_type} model on {args.device}...")
    try:
        llm_client = LocalLLMClient(
            model_type=args.model_type,
            device=args.device,
            cache_dir=args.cache_dir,
            temperature=args.temperature,
        )
        print("Model initialized successfully!")
    except Exception as e:
        print(f"Error initializing model: {e}")
        print(
            f"Make sure '{args.model_type}' is a valid model type supported by LocalLLMClient"
        )
        sys.exit(1)

    # Run benchmark
    print(f"\nStarting benchmark for {args.model_type}...")
    try:
        results, patient_results = benchmark_rare_disease_diagnosis(
            data=data,
            llm_client=llm_client,
            num_samples=args.num_samples,
            verbose=args.verbose,
        )
    except Exception as e:
        print(f"Error during benchmarking: {e}")
        sys.exit(1)

    # Print results
    print_benchmark_results(results, args.model_type)

    # Save results if requested
    if args.save_results:
        try:
            filepath = save_results_to_file(
                results, patient_results, args.model_type, args.output_dir
            )
            print(f"\nResults saved to: {filepath}")
            print(f"Patient-level results included for {len(patient_results)} patients")
        except ImportError:
            # Fallback without pandas
            import datetime

            os.makedirs(args.output_dir, exist_ok=True)
            results_with_metadata = {
                "model_type": args.model_type,
                "timestamp": datetime.datetime.now().isoformat(),
                "summary_metrics": results,
                "patient_results": patient_results,
            }
            filename = f"{args.model_type}_benchmark_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            filepath = os.path.join(args.output_dir, filename)
            with open(filepath, "w") as f:
                json.dump(results_with_metadata, f, indent=2)
            print(f"\nResults saved to: {filepath}")
            print(f"Patient-level results included for {len(patient_results)} patients")
        except Exception as e:
            print(f"Warning: Could not save results to file: {e}")


if __name__ == "__main__":
    main()
