#!/usr/bin/env python3
import argparse
import json
import os
import torch
import traceback
import re
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path for module imports
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Import project modules
from utils.llm_client import LocalLLMClient, APILLMClient


def timestamp_print(message: str) -> None:
    """Print message with timestamp."""
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Zero-shot extraction of phenotypes and HPO codes from clinical notes"
    )
    
    # Input/output files
    parser.add_argument(
        "--input_file", required=True,
        help="Input JSON file with clinical notes"
    )
    parser.add_argument(
        "--output_file", required=True, 
        help="Output JSON file for extraction results"
    )
    
    # LLM configuration
    parser.add_argument(
        "--llm_type", type=str, 
        choices=["local", "api"],
        default="local", 
        help="Type of LLM to use (default: local)"
    )
    parser.add_argument(
        "--model", type=str, 
        default="llama3-70b",
        help="Model type for local LLM (default: llama3-70b)"
    )
    parser.add_argument(
        "--api_config", type=str, 
        help="Path to API configuration file for API LLM"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.2,
        help="Temperature for LLM inference (default: 0.2)"
    )
    parser.add_argument(
        "--max_tokens", type=int, default=4000,
        help="Maximum tokens for LLM response (default: 4000)"
    )
    parser.add_argument(
        "--cache_dir", type=str, 
        default="/shared/rsaas/jw3/rare_disease/model_cache",
        help="Directory for caching models"
    )
    
    # Processing configuration
    parser.add_argument(
        "--checkpoint_interval", type=int, default=10,
        help="Save intermediate results every N cases (default: 10)"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from existing output file if it exists"
    )
    
    # GPU configuration
    gpu_group = parser.add_mutually_exclusive_group()
    gpu_group.add_argument(
        "--gpu_id", type=int,
        help="Specific GPU ID to use"
    )
    gpu_group.add_argument(
        "--condor", action="store_true",
        help="Use generic CUDA device without specific GPU ID (for job schedulers)"
    )
    gpu_group.add_argument(
        "--cpu", action="store_true",
        help="Force CPU usage even if GPU is available"
    )
    
    # Debug mode
    parser.add_argument(
        "--debug", action="store_true", 
        help="Enable debug output"
    )
    
    return parser.parse_args()


def setup_device(args: argparse.Namespace) -> str:
    """Configure the device based on command line arguments."""
    if args.cpu:
        return "cpu"
    elif args.condor:
        if torch.cuda.is_available():
            timestamp_print("Using generic CUDA device for condor/job scheduler environment")
            return "cuda"
        else:
            timestamp_print("Warning: CUDA requested but not available. Falling back to CPU.")
            return "cpu"
    elif args.gpu_id is not None:
        if torch.cuda.is_available():
            if args.gpu_id < torch.cuda.device_count():
                return f"cuda:{args.gpu_id}"
            else:
                timestamp_print(f"Warning: GPU {args.gpu_id} requested but only {torch.cuda.device_count()} GPUs available. Using GPU 0.")
                return "cuda:0"
        else:
            timestamp_print(f"Warning: GPU {args.gpu_id} requested but no CUDA available. Falling back to CPU.")
            return "cpu"
    else:
        return "cuda" if torch.cuda.is_available() else "cpu"


def initialize_llm_client(args: argparse.Namespace, device: str):
    """Initialize appropriate LLM client based on arguments."""
    if args.llm_type == "api":
        if args.api_config:
            return APILLMClient.from_config(args.api_config)
        else:
            return APILLMClient.initialize_from_input()
    else:  # local
        return LocalLLMClient(
            model_type=args.model,
            device=device,
            cache_dir=args.cache_dir,
            temperature=args.temperature
        )


def load_input_data(input_file: str) -> Dict[int, Dict[str, Any]]:
    """Load and validate the input JSON file."""
    try:
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        # Validate structure
        if not isinstance(data, dict):
            raise ValueError(f"Input file {input_file} must contain a JSON object")
        
        # Convert string keys to integers if necessary
        if data and all(isinstance(k, str) for k in data.keys()):
            data = {int(k): v for k, v in data.items()}
        
        # Basic validation of required fields
        for case_id, case_data in data.items():
            if not isinstance(case_data, dict):
                raise ValueError(f"Case {case_id} data must be a dictionary")
            if "clinical_text" not in case_data:
                raise ValueError(f"Case {case_id} missing required 'clinical_text' field")
        
        return data
    
    except (json.JSONDecodeError, ValueError) as e:
        timestamp_print(f"Error loading input file: {e}")
        raise


def load_existing_results(output_file: str) -> Dict[int, Dict[str, Any]]:
    """Load existing results from output file if it exists."""
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r') as f:
                data = json.load(f)
            
            # Convert string keys to integers if necessary
            if data and all(isinstance(k, str) for k in data.keys()):
                data = {int(k): v for k, v in data.items()}
                
            timestamp_print(f"Loaded existing results for {len(data)} cases from {output_file}")
            return data
        except Exception as e:
            timestamp_print(f"Error loading existing results: {e}")
            return {}
    return {}


def save_checkpoint(results: Dict[int, Dict[str, Any]], output_file: str, checkpoint_num: int) -> None:
    """Save intermediate results to a checkpoint file."""
    checkpoint_file = f"{os.path.splitext(output_file)[0]}_checkpoint{checkpoint_num}.json"
    with open(checkpoint_file, 'w') as f:
        json.dump(results, f, indent=2)
    timestamp_print(f"Saved checkpoint to {checkpoint_file}")


def get_zero_shot_system_prompt() -> str:
    """Return the system prompt for zero-shot HPO code extraction."""
    return """You are a clinical expert specializing in identifying human phenotypes from clinical notes and mapping them to Human Phenotype Ontology (HPO) codes.

When analyzing clinical text, your task is to:
1. Identify all phenotypes (observable abnormalities, clinical symptoms, or conditions) in the clinical note
2. Map each phenotype to the most specific and accurate HPO code
3. Return a structured response with both the phenotypes and their HPO codes

Guidelines:
- Focus on abnormal phenotypes, not normal findings or anatomical structures
- Identify specific phenotypic abnormalities, not just medical procedures or lab tests
- Map to the most specific HPO term available
- Use the standard "HP:NNNNNNN" format for HPO codes (e.g., HP:0001250 for seizures)
- If you're uncertain about the exact HPO code, provide your best match
- Only include phenotypes that are explicitly mentioned or strongly implied in the text

Your response must be in JSON format with these fields:
1. "phenotypes": A list of identified phenotype terms as they appear in the text
2. "hpo_codes": A corresponding list of HPO codes in the same order as the phenotypes

Do not include explanations, interpretations, or additional commentary - just the JSON output."""


def get_zero_shot_prompt(clinical_text: str) -> str:
    """Create a prompt for zero-shot HPO code extraction."""
    return f"""Please analyze the following clinical note to identify phenotypes and map them to HPO codes:

CLINICAL NOTE:
{clinical_text}

Extract all phenotypes from this note and provide the corresponding HPO codes in JSON format.
Include only phenotypes that are explicitly mentioned or strongly implied in the text.
Your response should be structured as a JSON object with two lists: "phenotypes" and "hpo_codes".
"""


def extract_json_from_response(response: str) -> Dict:
    """Extract the JSON object from the LLM response."""
    # Look for JSON content between triple backticks
    json_match = re.search(r'```(?:json)?(.*?)```', response, re.DOTALL)
    if json_match:
        json_str = json_match.group(1).strip()
    else:
        # Try to find JSON content without backticks
        # Look for the first { and last } in the response
        start_idx = response.find('{')
        end_idx = response.rfind('}')
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            json_str = response[start_idx:end_idx + 1].strip()
        else:
            # Fallback: assume the entire response is JSON
            json_str = response.strip()
    
    try:
        # Try to parse the extracted string as JSON
        result = json.loads(json_str)
        return result
    except json.JSONDecodeError:
        # If parsing fails, try to clean the string further
        cleaned_str = re.sub(r'[^\x00-\x7F]+', '', json_str)  # Remove non-ASCII
        cleaned_str = re.sub(r',\s*}', '}', cleaned_str)  # Fix trailing commas
        
        try:
            return json.loads(cleaned_str)
        except json.JSONDecodeError:
            # Return error structure
            return {
                "error": "Failed to parse JSON from LLM response", 
                "raw_response": response
            }


def process_case_with_llm(case_id: int, clinical_text: str, llm_client, debug: bool = False) -> Dict:
    """Process a single clinical note to extract phenotypes and HPO codes."""
    if debug:
        timestamp_print(f"Processing clinical note for case {case_id}")
    
    # Create system prompt and user prompt
    system_prompt = get_zero_shot_system_prompt()
    user_prompt = get_zero_shot_prompt(clinical_text)
    
    # Query the LLM
    llm_response = llm_client.query(user_prompt, system_prompt)
    
    if debug:
        timestamp_print(f"LLM response received for case {case_id} (length: {len(llm_response)})")
    
    # Extract JSON from the response
    try:
        extracted_data = extract_json_from_response(llm_response)
        
        # Check if extraction failed
        if "error" in extracted_data:
            timestamp_print(f"Failed to extract JSON from response for case {case_id}")
            return {
                "clinical_text": clinical_text,
                "error": extracted_data["error"],
                "phenotypes": [],
                "hpo_codes": [],
                "raw_response": llm_response
            }
        
        # Validate expected fields
        if "phenotypes" not in extracted_data or "hpo_codes" not in extracted_data:
            timestamp_print(f"Missing expected fields in response for case {case_id}")
            # Try to reconstruct from what we have
            phenotypes = extracted_data.get("phenotypes", [])
            hpo_codes = extracted_data.get("hpo_codes", [])
            
            # If we have a list of dictionaries, try to extract from that
            if not phenotypes and isinstance(extracted_data, list):
                phenotypes = []
                hpo_codes = []
                for item in extracted_data:
                    if isinstance(item, dict):
                        if "phenotype" in item:
                            phenotypes.append(item["phenotype"])
                            hpo_codes.append(item.get("hpo_code", ""))
        else:
            phenotypes = extracted_data["phenotypes"]
            hpo_codes = extracted_data["hpo_codes"]
        
        # Normalize lists if needed
        if not isinstance(phenotypes, list):
            phenotypes = [phenotypes]
        if not isinstance(hpo_codes, list):
            hpo_codes = [hpo_codes]
        
        # Ensure lists are the same length
        if len(phenotypes) != len(hpo_codes):
            timestamp_print(f"Warning: Mismatch in phenotypes and HPO codes length for case {case_id}")
            # Adjust lists to match
            if len(phenotypes) > len(hpo_codes):
                hpo_codes.extend([""] * (len(phenotypes) - len(hpo_codes)))
            else:
                phenotypes.extend([""] * (len(hpo_codes) - len(phenotypes)))
        
        # Build the result
        result = {
            "clinical_text": clinical_text,
            "phenotypes": phenotypes,
            "hpo_codes": hpo_codes
        }
        
        if debug:
            timestamp_print(f"Found {len(phenotypes)} phenotypes for case {case_id}")
            
        return result
    
    except Exception as e:
        timestamp_print(f"Error processing case {case_id}: {str(e)}")
        if debug:
            traceback.print_exc()
        
        return {
            "clinical_text": clinical_text,
            "error": str(e),
            "phenotypes": [],
            "hpo_codes": [],
            "raw_response": llm_response
        }


def process_cases(cases: Dict[int, Dict[str, Any]], llm_client, args: argparse.Namespace, 
                 existing_results: Dict = None) -> Dict[int, Dict[str, Any]]:
    """Process all cases to extract phenotypes and HPO codes."""
    results = existing_results or {}
    checkpoint_counter = 0
    
    # Determine which cases need processing
    pending_cases = {case_id: case_data for case_id, case_data in cases.items() 
                   if case_id not in results or not results[case_id].get('phenotypes')}
    
    timestamp_print(f"Processing {len(pending_cases)} cases out of {len(cases)} total cases")
    
    # Convert to list for progress tracking
    case_items = list(pending_cases.items())
    
    # Use tqdm for progress tracking
    for i, (case_id, case_data) in enumerate(tqdm(case_items, desc="Processing cases")):
        try:
            clinical_text = case_data["clinical_text"]
            
            # Process the clinical note
            case_result = process_case_with_llm(case_id, clinical_text, llm_client, args.debug)
            
            # Store the result
            results[case_id] = case_result
            
            # Save checkpoint if interval reached
            checkpoint_counter += 1
            if checkpoint_counter >= args.checkpoint_interval:
                save_checkpoint(results, args.output_file, i+1)
                checkpoint_counter = 0
                
        except Exception as e:
            timestamp_print(f"Error processing case {case_id}: {e}")
            if args.debug:
                traceback.print_exc()
            
            # Still add the case to results but mark as failed
            results[case_id] = {
                "clinical_text": case_data.get("clinical_text", ""),
                "error": str(e),
                "phenotypes": [],
                "hpo_codes": []
            }
    
    return results


def format_results_for_evaluation(results: Dict[int, Dict[str, Any]]) -> Dict[str, Dict[str, List[Dict[str, str]]]]:
    """Format results in a structure compatible with evaluation scripts."""
    evaluation_format = {}
    
    for case_id, case_data in results.items():
        phenotypes = case_data.get("phenotypes", [])
        hpo_codes = case_data.get("hpo_codes", [])
        
        # Create entries list with phenotypes and HPO codes
        verified_phenotypes = []
        for i, (phenotype, hpo_code) in enumerate(zip(phenotypes, hpo_codes)):
            if phenotype and hpo_code:
                verified_phenotypes.append({
                    "phenotype": phenotype,
                    "HPO_Term": hpo_code
                })
        
        evaluation_format[str(case_id)] = {
            "verified_phenotypes": verified_phenotypes
        }
    
    return evaluation_format


def main():
    """Main function to run the zero-shot HPO extraction pipeline."""
    try:
        # Parse command line arguments
        args = parse_arguments()
        
        timestamp_print(f"Starting zero-shot phenotype and HPO code extraction")
        
        # Setup device
        device = setup_device(args)
        timestamp_print(f"Using device: {device}")
        
        # Initialize LLM client
        timestamp_print(f"Initializing {args.llm_type} LLM client with {args.model} model")
        llm_client = initialize_llm_client(args, device)
        
        # Load input data
        timestamp_print(f"Loading clinical notes from {args.input_file}")
        cases = load_input_data(args.input_file)
        timestamp_print(f"Loaded {len(cases)} cases")
        
        # Check for existing results if resuming
        existing_results = {}
        if args.resume:
            existing_results = load_existing_results(args.output_file)
        
        # Process cases
        timestamp_print(f"Extracting phenotypes and HPO codes")
        results = process_cases(cases, llm_client, args, existing_results)
        
        # Format results for evaluation
        timestamp_print(f"Formatting results for evaluation")
        evaluation_format = format_results_for_evaluation(results)
        
        # Save raw results
        raw_output_file = f"{os.path.splitext(args.output_file)[0]}_raw.json"
        timestamp_print(f"Saving raw extraction results to {raw_output_file}")
        os.makedirs(os.path.dirname(os.path.abspath(raw_output_file)), exist_ok=True)
        with open(raw_output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save evaluation-ready results
        timestamp_print(f"Saving evaluation-ready results to {args.output_file}")
        os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
        with open(args.output_file, 'w') as f:
            json.dump(evaluation_format, f, indent=2)
        
        # Print statistics
        total_cases = len(results)
        cases_with_phenotypes = sum(1 for case_data in results.values() 
                                  if case_data.get("phenotypes") and len(case_data.get("phenotypes", [])) > 0)
        total_phenotypes = sum(len(case_data.get("phenotypes", [])) for case_data in results.values())
        
        timestamp_print(f"Extraction complete:")
        timestamp_print(f"  Total cases processed: {total_cases}")
        timestamp_print(f"  Cases with phenotypes: {cases_with_phenotypes}")
        timestamp_print(f"  Total phenotypes identified: {total_phenotypes}")
        timestamp_print(f"  Average phenotypes per case: {total_phenotypes / total_cases:.2f}")
        
    except Exception as e:
        timestamp_print(f"Critical error: {e}")
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()