#!/usr/bin/env python3
import argparse
import json
import os
import torch
import traceback
from datetime import datetime
from typing import Dict, Any, List
from pathlib import Path
from tqdm import tqdm

# so I can call it outside of the directory it's stuck in. 
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Import utils.setup for shared device setup
from utils.setup import setup_device

# Import project modules
from utils.demographic import DemographicsExtractor
from utils.llm_client import LocalLLMClient, APILLMClient

# Import project modules
from utils.demographic import DemographicsExtractor
from utils.llm_client import LocalLLMClient, APILLMClient


def timestamp_print(message: str) -> None:
    """Print message with timestamp."""
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Extract demographic information from clinical notes")
    
    # Input/output files
    parser.add_argument("--input_file", required=True,
                       help="Input JSON file with clinical notes (output from step1.py)")
    parser.add_argument("--output_file", required=True, 
                       help="Output JSON file for extraction results with added demographics")
    
    # LLM configuration
    parser.add_argument("--llm_type", type=str, choices=["local", "api"],
                       default="local", help="Type of LLM to use (default: local)")
    parser.add_argument("--model_type", type=str, 
                       default="mistral_24b",
                       help="Model type for local LLM (default: mistral_24b)")
    parser.add_argument("--api_config", type=str, 
                       help="Path to API configuration file for API LLM")
    parser.add_argument("--temperature", type=float, default=0.2,
                       help="Temperature for LLM inference (default: 0.2)")
    parser.add_argument("--cache_dir", type=str, 
                       default="/shared/rsaas/jw3/rare_disease/model_cache",
                       help="Directory for caching models")
    
    # Processing configuration
    parser.add_argument("--checkpoint_interval", type=int, default=10,
                       help="Save intermediate results every N cases (default: 10)")
    parser.add_argument("--resume", action="store_true",
                       help="Resume from existing output file if it exists")
    
    # GPU configuration
    gpu_group = parser.add_mutually_exclusive_group()
    gpu_group.add_argument("--gpu_id", type=int,
                          help="Specific GPU ID to use")
    gpu_group.add_argument("--condor", action="store_true",
                          help="Use generic CUDA device without specific GPU ID (for job schedulers)")
    gpu_group.add_argument("--cpu", action="store_true",
                          help="Force CPU usage even if GPU is available")
    
    # Add retriever arguments needed by utils.setup
    parser.add_argument("--retriever_cpu", action="store_true",
                       help="Force CPU usage for retriever even if GPU is available")
    parser.add_argument("--retriever_gpu_id", type=int,
                       help="Specific GPU ID to use for retriever")
    parser.add_argument("--stanza_cpu", action="store_true",
                       help="Force CPU usage for Stanza even if GPU is available")
    parser.add_argument("--stanza_gpu_id", type=int,
                       help="Specific GPU ID to use for Stanza")
    
    # Debug mode
    parser.add_argument("--debug", action="store_true", 
                       help="Enable debug output")
    
    return parser.parse_args()


# We're now using the imported setup_device, no need to define our own


def initialize_llm_client(args: argparse.Namespace, device: str):
    """Initialize appropriate LLM client based on arguments."""
    timestamp_print(f"Initializing {args.llm_type} LLM client")
    
    if args.llm_type == "api":
        if args.api_config:
            timestamp_print(f"Loading API configuration from {args.api_config}")
            return APILLMClient.from_config(args.api_config)
        else:
            timestamp_print("No API config provided. Initializing from user input.")
            return APILLMClient.initialize_from_input()
    else:  # local
        timestamp_print(f"Loading local model {args.model_type} on {device}")
        return LocalLLMClient(
            model_type=args.model_type,
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
                timestamp_print(f"Warning: Case {case_id} missing 'clinical_text' field")
        
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


def process_cases(cases: Dict[int, Dict[str, Any]], demographics_extractor: DemographicsExtractor, 
                args: argparse.Namespace, existing_results: Dict = None) -> Dict[int, Dict[str, Any]]:
    """Process all cases to extract demographic information."""
    results = existing_results or {}
    checkpoint_counter = 0
    
    # Determine which cases need processing
    pending_cases = {case_id: case_data for case_id, case_data in cases.items() 
                   if case_id not in results or not results[case_id].get('demographics')}
    
    timestamp_print(f"Processing {len(pending_cases)} cases out of {len(cases)} total cases")
    
    # Convert to list for progress tracking
    case_items = list(pending_cases.items())
    
    # Use tqdm for progress tracking
    for i, (case_id, case_data) in enumerate(tqdm(case_items, desc="Processing cases")):
        try:
            if args.debug:
                timestamp_print(f"Processing case {i+1}/{len(pending_cases)} (ID: {case_id})")
            
            clinical_text = case_data.get("clinical_text", "")
            
            # Extract demographics
            demographics = demographics_extractor.extract(clinical_text)
            
            if args.debug:
                timestamp_print(f"  Extracted demographics: age={demographics.get('age')}, "
                              f"gender={demographics.get('gender')}, "
                              f"age_group={demographics.get('age_group')}")
            
            # Create a copy of the original case data
            result_data = case_data.copy()
            
            # Add demographics information
            result_data["demographics"] = demographics
            
            # Store results
            results[case_id] = result_data
            
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
            results[case_id] = case_data.copy()
            results[case_id]["demographics"] = {
                "age": None,
                "gender": None,
                "age_group": None,
                "error": str(e)
            }
    
    return results


def main():
    """Main function to run the demographic extraction pipeline."""
    try:
        # Parse command line arguments
        args = parse_arguments()
        
        timestamp_print(f"Starting demographic extraction process")
        
        # Setup device using shared utils.setup function
        devices = setup_device(args)
        # Extract just the LLM device from the dictionary
        device = devices['llm']
        timestamp_print(f"Using device: {device}")
        
        # Initialize LLM client
        llm_client = initialize_llm_client(args, device)
        
        # Initialize demographics extractor
        timestamp_print(f"Initializing demographics extractor")
        demographics_extractor = DemographicsExtractor(llm_client, debug=args.debug)
        
        # Load input data
        timestamp_print(f"Loading clinical notes from {args.input_file}")
        cases = load_input_data(args.input_file)
        timestamp_print(f"Loaded {len(cases)} cases")
        
        # Check for existing results if resuming
        existing_results = {}
        if args.resume:
            existing_results = load_existing_results(args.output_file)
        
        # Process cases
        timestamp_print(f"Extracting demographics")
        results = process_cases(cases, demographics_extractor, args, existing_results)
        
        # Save results to JSON
        timestamp_print(f"Saving extraction results to {args.output_file}")
        os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Compute summary statistics
        total_cases = len(results)
        gender_counts = {"male": 0, "female": 0, "unknown": 0}
        age_group_counts = {"infant": 0, "child": 0, "adolescent": 0, "adult": 0, "elderly": 0, "unknown": 0}
        
        for case_id, case_data in results.items():
            demographics = case_data.get("demographics", {})
            gender = demographics.get("gender")
            age_group = demographics.get("age_group")
            
            if gender == "male":
                gender_counts["male"] += 1
            elif gender == "female":
                gender_counts["female"] += 1
            else:
                gender_counts["unknown"] += 1
                
            if age_group in ["infant", "child", "adolescent", "adult", "elderly"]:
                age_group_counts[age_group] += 1
            else:
                age_group_counts["unknown"] += 1
        
        # Print summary statistics
        timestamp_print(f"Demographic extraction complete. Processed {total_cases} cases.")
        timestamp_print(f"Gender distribution:")
        timestamp_print(f"  Male: {gender_counts['male']} ({gender_counts['male']/total_cases*100:.1f}%)")
        timestamp_print(f"  Female: {gender_counts['female']} ({gender_counts['female']/total_cases*100:.1f}%)")
        timestamp_print(f"  Unknown: {gender_counts['unknown']} ({gender_counts['unknown']/total_cases*100:.1f}%)")
        
        timestamp_print(f"Age group distribution:")
        timestamp_print(f"  Infant (0-1): {age_group_counts['infant']} ({age_group_counts['infant']/total_cases*100:.1f}%)")
        timestamp_print(f"  Child (2-11): {age_group_counts['child']} ({age_group_counts['child']/total_cases*100:.1f}%)")
        timestamp_print(f"  Adolescent (12-17): {age_group_counts['adolescent']} ({age_group_counts['adolescent']/total_cases*100:.1f}%)")
        timestamp_print(f"  Adult (18-64): {age_group_counts['adult']} ({age_group_counts['adult']/total_cases*100:.1f}%)")
        timestamp_print(f"  Elderly (65+): {age_group_counts['elderly']} ({age_group_counts['elderly']/total_cases*100:.1f}%)")
        timestamp_print(f"  Unknown: {age_group_counts['unknown']} ({age_group_counts['unknown']/total_cases*100:.1f}%)")
        
    except Exception as e:
        timestamp_print(f"Critical error: {e}")
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()