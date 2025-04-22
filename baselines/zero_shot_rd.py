#!/usr/bin/env python3
import argparse
import json
import os
from datetime import datetime
from typing import Dict, List, Any
import traceback
from pathlib import Path
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Import necessary LLM client
from utils.llm_client import LocalLLMClient, APILLMClient
from utils.setup import setup_device, initialize_llm_client

def timestamp_print(message: str) -> None:
    """Print message with timestamp."""
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}")

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Zero-shot rare disease identification from clinical notes")
    
    # Input/output files
    parser.add_argument("--input_file", required=True, 
                       help="Input JSON file with clinical notes")
    parser.add_argument("--output_file", required=True, 
                       help="Output JSON file for results")
    
    # LLM configuration
    parser.add_argument("--llm_type", type=str, choices=["local", "api"],
                       default="local", help="Type of LLM to use")
    parser.add_argument("--model_type", type=str, 
                       default="llama3_70b",
                       help="Model type for local LLM")
    parser.add_argument("--api_config", type=str, 
                       help="Path to API configuration file for API LLM")
    parser.add_argument("--temperature", type=float, default=0.2,
                       help="Temperature for LLM inference")
    parser.add_argument("--cache_dir", type=str, 
                       default="/shared/rsaas/jw3/rare_disease/model_cache",
                       help="Directory for caching models")
    
    # Processing configuration
    parser.add_argument("--checkpoint_interval", type=int, default=10,
                       help="Save intermediate results every N cases")
    
    # GPU configuration
    gpu_group = parser.add_mutually_exclusive_group()
    gpu_group.add_argument("--gpu_id", type=int,
                          help="Specific GPU ID to use")
    gpu_group.add_argument("--condor", action="store_true",
                          help="Use generic CUDA device without specific GPU ID (for job schedulers)")
    gpu_group.add_argument("--cpu", action="store_true",
                          help="Force CPU usage even if GPU is available")
    
    # Debug mode
    parser.add_argument("--debug", action="store_true", 
                       help="Enable debug output")
    
    return parser.parse_args()

def load_input_data(input_file: str) -> Dict[str, Dict[str, Any]]:
    """Load and validate the input JSON file."""
    try:
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        processed_data = {}
        
        # Handle nested structure with metadata and results keys
        if isinstance(data, dict) and "results" in data:
            timestamp_print("Found structured input with 'results' key")
            data = data["results"]
        
        for doc_id, doc_data in data.items():
            # Check if data format is from previous steps
            if isinstance(doc_data, dict):
                if "clinical_text" in doc_data:
                    # Direct format with clinical_text field
                    processed_data[doc_id] = {
                        "clinical_text": doc_data["clinical_text"],
                        "metadata": doc_data.get("metadata", {})
                    }
                elif "note_details" in doc_data and "text" in doc_data["note_details"]:
                    # MIMIC format
                    note_details = doc_data["note_details"]
                    processed_data[doc_id] = {
                        "clinical_text": note_details["text"],
                        "metadata": {
                            "patient_id": note_details.get("subject_id", ""),
                            "admission_id": note_details.get("hadm_id", ""),
                            "category": note_details.get("category", ""),
                            "chart_date": note_details.get("chartdate", "")
                        }
                    }
            elif isinstance(doc_data, str):
                # Assume the string itself is the clinical text
                processed_data[doc_id] = {
                    "clinical_text": doc_data,
                    "metadata": {}
                }
        
        if not processed_data:
            raise ValueError(f"No valid clinical notes found in {input_file}")
        
        return processed_data
    
    except Exception as e:
        timestamp_print(f"Error loading input file: {e}")
        traceback.print_exc()
        raise

def save_checkpoint(results: Dict, output_file: str, checkpoint_num: int) -> None:
    """Save intermediate results to a checkpoint file."""
    checkpoint_file = f"{os.path.splitext(output_file)[0]}_checkpoint{checkpoint_num}.json"
    with open(checkpoint_file, 'w') as f:
        json.dump(results, f, indent=2)
    timestamp_print(f"Saved checkpoint to {checkpoint_file}")

def format_orpha_id(code: str) -> str:
    """Format ORPHA ID to standard format (ORPHA:12345)."""
    if not code:
        return ""
        
    # Handle case where code already has ORPHA: prefix
    if code.upper().startswith("ORPHA:"):
        return code
        
    # Extract numeric part if it's just a number
    if code.isdigit():
        return f"ORPHA:{code}"
        
    # Default case
    return code

def extract_and_match_zero_shot(clinical_text: str, llm_client) -> List[Dict[str, Any]]:
    """
    Extract and match rare diseases in one zero-shot step.
    
    Args:
        clinical_text: The clinical note text
        llm_client: LLM client for querying the model
        
    Returns:
        List of dictionaries with matched disease information
    """
    system_message = (
        "You are a medical expert specialized in diagnosing rare diseases from clinical notes. "
        "Your task is to identify rare diseases mentioned in a clinical note and provide their "
        "ORPHA codes. Be precise and thorough in your analysis."
    )
    
    prompt = f"""Given the following clinical note, identify all rare diseases mentioned or implied, and provide their ORPHA codes.

Clinical Note:
{clinical_text}

Instructions:
1. Identify all rare diseases that are explicitly mentioned or can be inferred from the symptoms and findings
2. For each disease, provide the ORPHA code (in format ORPHA:12345)
3. Provide your confidence in each diagnosis (0.0-1.0)
4. Only include rare diseases (conditions affecting fewer than 1 in 2,000 people)

Return your answer as a JSON array with this structure:
[
  {{
    "entity": "mention in the text",
    "rd_term": "formal rare disease name",
    "orpha_id": "ORPHA:12345",
    "confidence_score": 0.9
  }},
  ...
]

IMPORTANT: 
- Only include entries where you're confident the disease is mentioned
- Do not include differential diagnoses that are explicitly ruled out
- Use the actual ORPHA codes from Orphanet, not made-up codes
- If you're uncertain about the exact ORPHA code but confident about the disease, still include it
- Return only the JSON array with no additional text or explanation
"""
    
    # Query LLM
    response = llm_client.query(prompt, system_message)
    
    # Extract JSON array from response
    try:
        # Find anything that looks like a JSON array
        import re
        json_match = re.search(r'\[.*\]', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            matches = json.loads(json_str)
            
            # Validate and clean up extracted matches
            valid_matches = []
            for match in matches:
                # Ensure required fields are present
                if "entity" in match and "rd_term" in match and "orpha_id" in match:
                    # Format ORPHA ID
                    match["orpha_id"] = format_orpha_id(match["orpha_id"])
                    
                    # Add match method
                    match["match_method"] = "zero_shot"
                    
                    # Ensure confidence score is present and valid
                    if "confidence_score" not in match or not isinstance(match["confidence_score"], (int, float)):
                        match["confidence_score"] = 0.7  # Default confidence
                    
                    valid_matches.append(match)
            
            return valid_matches
        
        return []
    except json.JSONDecodeError:
        timestamp_print(f"Failed to parse LLM response as JSON: {response}")
        return []
    except Exception as e:
        timestamp_print(f"Error processing LLM response: {str(e)}")
        return []

def process_cases(cases: Dict[str, Dict[str, Any]], args: argparse.Namespace, llm_client) -> Dict[str, Dict[str, Any]]:
    """Process all cases to extract and match rare diseases."""
    results = {}
    checkpoint_counter = 0
    
    timestamp_print(f"Processing {len(cases)} cases")
    
    for i, (case_id, case_data) in enumerate(cases.items()):
        try:
            timestamp_print(f"Processing case {i+1}/{len(cases)} (ID: {case_id})")
            
            clinical_text = case_data["clinical_text"]
            if not clinical_text:
                timestamp_print(f"  Empty clinical text for case {case_id}, skipping")
                results[case_id] = {
                    "clinical_text": "",
                    "metadata": case_data.get("metadata", {}),
                    "matched_diseases": [],
                    "note": "Empty clinical text"
                }
                continue
            
            # Extract and match rare diseases in one step
            matched_diseases = extract_and_match_zero_shot(clinical_text, llm_client)
            
            timestamp_print(f"  Found {len(matched_diseases)} rare diseases")
            
            # Store results
            results[case_id] = {
                "clinical_text": clinical_text,
                "metadata": case_data.get("metadata", {}),
                "matched_diseases": matched_diseases,
                "stats": {
                    "matched_diseases_count": len(matched_diseases)
                }
            }
            
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
                "metadata": case_data.get("metadata", {}),
                "matched_diseases": [],
                "error": str(e)
            }
    
    return results

def main():
    """Main function to run the zero-shot rare disease identification pipeline."""
    try:
        # Parse command line arguments
        args = parse_arguments()
        
        timestamp_print(f"Starting zero-shot rare disease identification")
        
        # Setup device
        from utils.setup import setup_device
        devices = setup_device(args)
        timestamp_print(f"Using device: {devices}")
        
        # Initialize LLM client
        timestamp_print(f"Initializing {args.llm_type} LLM client")
        llm_client = initialize_llm_client(args, devices)
        
        # Load input data
        timestamp_print(f"Loading clinical notes from {args.input_file}")
        cases = load_input_data(args.input_file)
        timestamp_print(f"Loaded {len(cases)} cases")
        
        # Process cases
        timestamp_print(f"Identifying rare diseases with zero-shot approach")
        results = process_cases(cases, args, llm_client)
        
        # Save results to JSON
        timestamp_print(f"Saving results to {args.output_file}")
        os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
        
        # Add metadata about the run
        metadata = {
            "identification_timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "approach": "zero_shot",
            "model_info": {
                "llm_type": args.llm_type,
                "model_type": args.model_type,
                "temperature": args.temperature
            }
        }
        
        # Create final output with metadata
        final_output = {
            "metadata": metadata,
            "results": results
        }
        
        with open(args.output_file, 'w') as f:
            json.dump(final_output, f, indent=2)
        
        # Print summary
        total_diseases = sum(case_data.get("stats", {}).get("matched_diseases_count", 0) for case_data in results.values())
        
        timestamp_print(f"Zero-shot identification complete:")
        timestamp_print(f"  Total cases: {len(cases)}")
        timestamp_print(f"  Total rare diseases identified: {total_diseases}")
        timestamp_print(f"  Average diseases per case: {total_diseases / len(cases):.2f}")
        
        timestamp_print(f"Zero-shot rare disease identification completed successfully.")
    
    except Exception as e:
        timestamp_print(f"Critical error: {e}")
        traceback.print_exc()
        exit(1)

if __name__ == "__main__":
    main()