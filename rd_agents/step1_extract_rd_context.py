#!/usr/bin/env python3
import argparse
import json
import os
import torch
import traceback
from datetime import datetime
from typing import Dict, List, Any
from pathlib import Path
from tqdm import tqdm

# Set correct directory pathing
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Import project modules
from rdrag.entity import LLMRDExtractor, BaseRDExtractor
from hporag.context import ContextExtractor
from utils.llm_client import LocalLLMClient, APILLMClient
from utils.setup import setup_device

def timestamp_print(message: str) -> None:
    """Print message with timestamp."""
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Extract rare disease entities and contexts from clinical notes")
    
    # Input/output files
    parser.add_argument("--input_file", required=True,
                       help="Input JSON file with clinical notes (MIMIC-style format)")
    parser.add_argument("--output_file", required=True, 
                       help="Output JSON file for extraction results")
    
    # System prompt configuration
    parser.add_argument("--system_prompt", type=str, default="You are a medical expert specializing in rare diseases.", 
                       help="System prompt for LLM extraction")
    
    # Entity extractor configuration
    parser.add_argument("--extractor_type", type=str, choices=["llm"],
                       default="llm", help="Entity extraction method (default: llm)")
    
    # Context extractor configuration
    parser.add_argument("--window_size", type=int, default=0,
                       help="Context window size for sentences (default: 0)")
    
    # LLM configuration
    parser.add_argument("--llm_type", type=str, choices=["local", "api"],
                       default="local", help="Type of LLM to use (default: local)")
    parser.add_argument("--model_type", type=str, 
                       default="llama3_70b",
                       help="Model type for local LLM (default: llama3_70b)")
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
    """Load and validate the input JSON file."""
    try:
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        # Validate structure
        if not isinstance(data, dict):
            raise ValueError(f"Input file {input_file} must contain a JSON object mapping IDs to clinical notes")
        
        processed_data = {}
        for doc_id, doc_data in data.items():
            # Check if this is MIMIC format
            if isinstance(doc_data, dict) and 'note_details' in doc_data:
                note_details = doc_data['note_details']
                # Extract clinical text
                clinical_text = note_details.get('text', '')
                if clinical_text:
                    processed_data[doc_id] = {
                        'clinical_text': clinical_text,
                        'patient_id': note_details.get('subject_id', ''),
                        'admission_id': note_details.get('hadm_id', ''),
                        'category': note_details.get('category', ''),
                        'chart_date': note_details.get('chartdate', '')
                    }
            else:
                # Fallback to simpler format
                if isinstance(doc_data, dict) and 'clinical_text' in doc_data:
                    processed_data[doc_id] = doc_data
                elif isinstance(doc_data, str):
                    # Assume the string itself is the clinical text
                    processed_data[doc_id] = {
                        'clinical_text': doc_data
                    }
        
        if not processed_data:
            raise ValueError(f"No valid clinical notes found in {input_file}")
        
        return processed_data
    
    except (json.JSONDecodeError, ValueError) as e:
        timestamp_print(f"Error loading input file: {e}")
        raise


def load_existing_results(output_file: str) -> Dict[str, Dict[str, Any]]:
    """Load existing results from output file if it exists."""
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r') as f:
                data = json.load(f)
            
            timestamp_print(f"Loaded existing results for {len(data)} cases from {output_file}")
            return data
        except Exception as e:
            timestamp_print(f"Error loading existing results: {e}")
            return {}
    return {}


def save_checkpoint(results: Dict[str, Dict[str, Any]], output_file: str, checkpoint_num: int) -> None:
    """Save intermediate results to a checkpoint file."""
    checkpoint_file = f"{os.path.splitext(output_file)[0]}_checkpoint{checkpoint_num}.json"
    with open(checkpoint_file, 'w') as f:
        json.dump(results, f, indent=2)
    timestamp_print(f"Saved checkpoint to {checkpoint_file}")


def process_cases(cases: Dict[str, Dict[str, Any]], args: argparse.Namespace, 
                 entity_extractor, context_extractor, existing_results: Dict = None) -> Dict[str, Dict[str, Any]]:
    """Process all cases to extract entities and contexts."""
    results = existing_results or {}
    checkpoint_counter = 0
    
    # Determine which cases need processing
    pending_cases = {case_id: case_data for case_id, case_data in cases.items() 
                   if case_id not in results or not results[case_id].get('entities_with_contexts')}
    
    timestamp_print(f"Processing {len(pending_cases)} cases out of {len(cases)} total cases")
    
    # Convert to list for progress tracking
    case_items = list(pending_cases.items())
    
    # Use tqdm for progress tracking
    for i, (case_id, case_data) in enumerate(tqdm(case_items, desc="Processing cases")):
        try:
            if args.debug:
                timestamp_print(f"Processing case {i+1}/{len(pending_cases)} (ID: {case_id})")
            
            clinical_text = case_data["clinical_text"]
            
            # Extract entities (possible rare disease mentions)
            entities = entity_extractor.extract_entities(clinical_text)
            
            if args.debug:
                timestamp_print(f"  Extracted {len(entities)} potential rare disease entities")
            
            # Find contexts for entities
            entity_contexts = context_extractor.extract_contexts(entities, clinical_text, window_size=args.window_size)
            
            # Store results
            results[case_id] = {
                "clinical_text": clinical_text,
                "entities_with_contexts": entity_contexts,
                "metadata": {
                    "patient_id": case_data.get("patient_id", ""),
                    "admission_id": case_data.get("admission_id", ""),
                    "category": case_data.get("category", ""),
                    "chart_date": case_data.get("chart_date", "")
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
                "entities_with_contexts": [],
                "metadata": {
                    "patient_id": case_data.get("patient_id", ""),
                    "admission_id": case_data.get("admission_id", ""),
                    "category": case_data.get("category", ""),
                    "chart_date": case_data.get("chart_date", "")
                },
                "error": str(e)
            }
    
    return results


def main():
    """Main function to run the entity and context extraction pipeline."""
    try:
        # Parse command line arguments
        args = parse_arguments()
        
        timestamp_print(f"Starting rare disease entity extraction process")
        
        # Setup device
        device = setup_device(args)
        timestamp_print(f"Using device: {device}")
        
        # Initialize LLM client
        timestamp_print(f"Initializing {args.llm_type} LLM client")
        llm_client = initialize_llm_client(args, device)
        
        # Initialize entity extractor
        timestamp_print(f"Initializing entity extractor ({args.extractor_type})")
        if args.extractor_type == "llm":
            entity_extractor = LLMRDExtractor(llm_client, args.system_prompt)
        else:
            raise ValueError(f"Unsupported extractor type: {args.extractor_type}")
        
        # Initialize context extractor
        timestamp_print(f"Initializing context extractor (window_size={args.window_size})")
        context_extractor = ContextExtractor(debug=args.debug)
        
        # Load input data
        timestamp_print(f"Loading clinical notes from {args.input_file}")
        cases = load_input_data(args.input_file)
        timestamp_print(f"Loaded {len(cases)} cases")
        
        # Check for existing results if resuming
        existing_results = {}
        if args.resume:
            existing_results = load_existing_results(args.output_file)
        
        # Process cases
        timestamp_print(f"Extracting entities and contexts")
        results = process_cases(cases, args, entity_extractor, context_extractor, existing_results)
        
        # Save results to JSON
        timestamp_print(f"Saving extraction results to {args.output_file}")
        os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        timestamp_print(f"Extraction complete. Processed {len(cases)} cases.")
        
    except Exception as e:
        timestamp_print(f"Critical error: {e}")
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()