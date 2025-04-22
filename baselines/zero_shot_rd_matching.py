#!/usr/bin/env python3
import argparse
import json
import os
import torch
import re
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from tqdm import tqdm
import sys

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Import LLM client
from utils.llm_client import LocalLLMClient, APILLMClient
from utils.setup import setup_device

def timestamp_print(message: str) -> None:
    """Print message with timestamp."""
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}")

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Extract rare disease ORPHA codes using zero-shot LLM approach")
    
    # Input/output files
    parser.add_argument("--input_file", required=True, 
                       help="Input JSON file with clinical notes")
    parser.add_argument("--output_file", required=True, 
                       help="Output JSON file for extraction results")
    parser.add_argument("--ground_truth", 
                       help="Optional ground truth file for evaluation")
    parser.add_argument("--csv_output", 
                       help="Optional CSV file for formatted results")
    
    # LLM configuration
    parser.add_argument("--llm_type", type=str, choices=["local", "api"],
                       default="local", help="Type of LLM to use")
    parser.add_argument("--model_type", type=str, 
                       default="llama3_70b",
                       help="Model type for local LLM")
    parser.add_argument("--api_config", type=str, 
                       help="Path to API configuration file for API LLM")
    parser.add_argument("--temperature", type=float, default=0.1,
                       help="Temperature for LLM inference")
    parser.add_argument("--cache_dir", type=str, 
                       default="/shared/rsaas/jw3/rare_disease/model_cache",
                       help="Directory for caching models")
    
    # Processing configuration
    parser.add_argument("--max_cases", type=int, 
                       help="Maximum number of cases to process (for testing)")
    parser.add_argument("--checkpoint_interval", type=int, default=10,
                       help="Save intermediate results every N cases")
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
    
    # Debug mode
    parser.add_argument("--debug", action="store_true", 
                       help="Enable debug output")
    
    return parser.parse_args()

def initialize_llm_client(args: argparse.Namespace, device: str):
    """Initialize appropriate LLM client based on arguments."""
    if args.llm_type == "api":
        if args.api_config:
            return APILLMClient.from_config(args.api_config)
        else:
            return APILLMClient.initialize_from_input()
    else:  # local
        return LocalLLMClient(
            model_type=args.model_type,
            device=device,
            cache_dir=args.cache_dir,
            temperature=args.temperature
        )

def load_input_data(input_file: str) -> Dict[str, Dict]:
    """Load clinical notes from input file."""
    try:
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        # Check for different possible formats
        processed_data = {}
        
        if isinstance(data, dict):
            for doc_id, doc_data in data.items():
                # Check if this is MIMIC format with note_details
                if isinstance(doc_data, dict) and "note_details" in doc_data:
                    note_text = doc_data["note_details"].get("text", "")
                    if note_text:
                        processed_data[doc_id] = {
                            "clinical_text": note_text,
                            "metadata": {
                                "patient_id": doc_data["note_details"].get("subject_id", ""),
                                "admission_id": doc_data["note_details"].get("hadm_id", ""),
                                "category": doc_data["note_details"].get("category", ""),
                                "chart_date": doc_data["note_details"].get("chartdate", "")
                            }
                        }
                # Check if already in expected format with clinical_text
                elif isinstance(doc_data, dict) and "clinical_text" in doc_data:
                    processed_data[doc_id] = doc_data
                # Assume string is the clinical text
                elif isinstance(doc_data, str):
                    processed_data[doc_id] = {"clinical_text": doc_data}
        
        if not processed_data:
            raise ValueError(f"No valid clinical notes found in {input_file}")
        
        return processed_data
    
    except Exception as e:
        timestamp_print(f"Error loading input file: {e}")
        raise

def load_existing_results(output_file: str) -> Dict:
    """Load existing results if available."""
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r') as f:
                data = json.load(f)
            
            # Handle case where the results are wrapped in a metadata structure
            if "results" in data:
                data = data["results"]
                
            timestamp_print(f"Loaded existing results for {len(data)} cases from {output_file}")
            return data
        except Exception as e:
            timestamp_print(f"Error loading existing results: {e}")
            return {}
    return {}

def save_checkpoint(results: Dict, output_file: str, checkpoint_num: int) -> None:
    """Save intermediate results to a checkpoint file."""
    checkpoint_file = f"{os.path.splitext(output_file)[0]}_checkpoint{checkpoint_num}.json"
    with open(checkpoint_file, 'w') as f:
        json.dump(results, f, indent=2)
    timestamp_print(f"Saved checkpoint to {checkpoint_file}")

def extract_orpha_codes_with_llm(clinical_text: str, llm_client, debug: bool = False) -> List[Dict]:
    """
    Extract rare diseases and ORPHA codes directly using the LLM in zero-shot mode.
    
    Args:
        clinical_text: Clinical note text
        llm_client: LLM client for querying the language model
        debug: Enable detailed debugging output
        
    Returns:
        List of dictionaries with extracted entities and ORPHA codes
    """
    if debug:
        timestamp_print(f"Extracting rare diseases from text of length {len(clinical_text)}")
    
    # Truncate text if too long
    max_text_length = 8000
    if len(clinical_text) > max_text_length:
        truncated_text = clinical_text[:max_text_length]
        if debug:
            timestamp_print(f"Text truncated from {len(clinical_text)} to {len(truncated_text)} characters")
    else:
        truncated_text = clinical_text
    
    # Define the system message
    system_message = (
        "You are a medical expert specializing in rare diseases with comprehensive knowledge of the "
        "ORPHANET database and ORPHA codes. Your task is to identify rare diseases mentioned in the "
        "clinical text and assign them the correct ORPHA codes."
    )
    
    # Create the extraction prompt
    prompt = f"""Analyze the following clinical note and identify all rare diseases mentioned in the text.
For each rare disease, provide the disease name exactly as it appears in the text and its corresponding ORPHA code.

Clinical Note:
{truncated_text}

Important instructions:
1. Only include diseases that are actually rare (affecting less than 1 in 2,000 people)
2. Do not include common diseases, symptoms, or conditions
3. Ignore any diseases that are negated (e.g., "no evidence of...", "ruled out...", etc.)
4. If you don't know the exact ORPHA code, use "unknown" instead
5. Format your response as a JSON list of objects with "entity" and "orpha_id" fields

Example response format:
```json
[
  {
    "entity": "Fabry disease", 
    "orpha_id": "ORPHA:324"
  },
  {
    "entity": "Gaucher disease",
    "orpha_id": "ORPHA:355"
  }
]
if debug:
    timestamp_print("Sending prompt to LLM")

# Query LLM
response = llm_client.query(prompt, system_message)

if debug:
    timestamp_print(f"Received response of length {len(response)}")

# Extract JSON from response
try:
    # Look for JSON content within markdown code blocks
    json_match = re.search(r'```(?:json)?\s*(\[[\s\S]*?\])\s*```', response)
    if json_match:
        json_content = json_match.group(1)
    else:
        # If no code block, try to extract anything that looks like a JSON array
        json_match = re.search(r'\[\s*\{[\s\S]*\}\s*\]', response)
        if json_match:
            json_content = json_match.group(0)
        else:
            if debug:
                timestamp_print(f"No JSON-like content found in response: {response}")
            return []
    
    # Parse JSON content
    extracted_data = json.loads(json_content)
    
    if debug:
        timestamp_print(f"Successfully parsed JSON with {len(extracted_data)} entities")
    
    # Ensure consistent field names
    normalized_results = []
    for item in extracted_data:
        # Ensure the entity field exists
        if "entity" not in item and "name" in item:
            item["entity"] = item["name"]
        
        # Ensure the orpha_id field exists
        if "orpha_id" not in item and "id" in item:
            item["orpha_id"] = item["id"]
        
        # Only include items that have both entity and orpha_id
        if "entity" in item and "orpha_id" in item:
            normalized_results.append({
                "entity": item["entity"],
                "orpha_id": item["orpha_id"],
                "extraction_method": "llm_zeroshot"
            })
    
    return normalized_results
    
except Exception as e:
    if debug:
        timestamp_print(f"Error parsing LLM response: {e}")
        timestamp_print(f"Raw response: {response}")
    return []

# Determine which cases need processing
pending_cases = {case_id: case_data for case_id, case_data in cases.items() 
               if case_id not in results or not results[case_id].get('matched_diseases')}

# Limit cases if max_cases specified
if args.max_cases is not None:
    pending_case_ids = list(pending_cases.keys())[:args.max_cases]
    pending_cases = {case_id: pending_cases[case_id] for case_id in pending_case_ids}

timestamp_print(f"Processing {len(pending_cases)} cases")

# Convert to list for progress tracking
case_items = list(pending_cases.items())

# Use tqdm for progress tracking
for i, (case_id, case_data) in enumerate(tqdm(case_items, desc="Processing cases")):
    try:
        if args.debug:
            timestamp_print(f"Processing case {i+1}/{len(pending_cases)} (ID: {case_id})")
        
        clinical_text = case_data["clinical_text"]
        
        # Extract rare diseases and ORPHA codes
        extracted_results = extract_orpha_codes_with_llm(clinical_text, llm_client, args.debug)
        
        if args.debug:
            timestamp_print(f"  Extracted {len(extracted_results)} entities with ORPHA codes")
        
        # Store results
        results[case_id] = {
            "clinical_text": clinical_text,
            "metadata": case_data.get("metadata", {}),
            "matched_diseases": extracted_results
        }
        
        # Save checkpoint if interval reached
        checkpoint_counter += 1
        if checkpoint_counter >= args.checkpoint_interval:
            save_checkpoint(results, args.output_file, i+1)
            checkpoint_counter = 0
            
    except Exception as e:
        timestamp_print(f"Error processing case {case_id}: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        
        # Still add the case to results but mark as failed
        results[case_id] = {
            "clinical_text": case_data.get("clinical_text", ""),
            "metadata": case_data.get("metadata", {}),
            "matched_diseases": [],
            "error": str(e)
        }

        

        # Extract ORPHA codes
    pred_codes = {}
    for doc_id, doc_data in results.items():
        doc_codes = []
        for disease in doc_data.get("matched_diseases", []):
            orpha_id = disease.get("orpha_id", "")
            if orpha_id and orpha_id.lower() != "unknown":
                doc_codes.append(orpha_id)
        if doc_codes:
            pred_codes[doc_id] = doc_codes
    
    # Extract ground truth ORPHA codes
    gt_codes = {}
    for doc_id, doc_data in ground_truth_data.items():
        doc_codes = []
        if "annotations" in doc_data and isinstance(doc_data["annotations"], list):
            for annotation in doc_data["annotations"]:
                if "ordo_with_desc" in annotation:
                    ordo_field = annotation["ordo_with_desc"]
                    if ordo_field:
                        # Extract first word as ORPHA ID
                        orpha_id = ordo_field.split(' ', 1)[0]
                        doc_codes.append(f"ORPHA:{orpha_id}")
        if doc_codes:
            gt_codes[doc_id] = doc_codes
    
    # Normalize and evaluate
    metrics = evaluate_numeric_codes(pred_codes, gt_codes)
    return metrics
    
except Exception as e:
    timestamp_print(f"Error during evaluation: {e}")
    import traceback
    traceback.print_exc()
    return {
        "error": str(e),
        "precision": 0.0,
        "recall": 0.0,
        "f1_score": 0.0
    }
return results

python zero_shot_rd_extraction.py \
    --input_file data/dataset/filtered_rd_annos_updated_adam.json \
    --output_file data/results/agents/rd/zeroshot/direct_extraction_results.json \
    --ground_truth data/dataset/filtered_rd_annos_updated_adam.json \
    --csv_output data/results/agents/rd/zeroshot/direct_extraction_results.csv \
    --llm_type local \
    --model_type llama3_70b \
    --temperature 0.1 \
    --gpu_id 0 \
    --checkpoint_interval 10