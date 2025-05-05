#!/usr/bin/env python3
import argparse
import json
import os
import pandas as pd
import numpy as np
import torch
import traceback
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
from tqdm import tqdm

# Set correct directory pathing
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Import project modules
from hporag.hpo_match import RAGHPOMatcher
from utils.embedding import EmbeddingsManager
from utils.llm_client import LocalLLMClient, APILLMClient
from utils.setup import setup_device, timestamp_print

def convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    import numpy as np
    
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list) or isinstance(obj, tuple):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    else:
        # For any other types, convert to string
        return str(obj)

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Match verified phenotypes to HPO terms")
    
    # Input/output files
    parser.add_argument("--input_file", required=True, 
                       help="Input JSON file from verification step (step 2)")
    parser.add_argument("--output_file", required=True, 
                       help="Output JSON file for matching results")
    parser.add_argument("--embeddings_file", required=True,
                       help="NPY file containing HPO embeddings (G2GHPO_metadata.npy)")
    parser.add_argument("--csv_output", 
                       help="Optional CSV file for formatted results")
    
    # System prompt configuration
    parser.add_argument("--system_prompt_file", default="data/prompts/system_prompts.json", 
                       help="File containing system prompts")
    
    # Matching configuration
    parser.add_argument("--top_k", type=int, default=5,
                       help="Number of top candidates to include in results")
    
    # Embedding configuration
    parser.add_argument("--retriever", type=str,
                       choices=["fastembed", "sentence_transformer", "medcpt"],
                       default="fastembed",
                       help="Type of retriever/embedding model to use")
    parser.add_argument("--retriever_model", type=str,
                       default="BAAI/bge-small-en-v1.5",
                       help="Model name for retriever (if using fastembed or sentence_transformer)")
    
    # Legacy parameter names (for backward compatibility)
    parser.add_argument("--model_type", type=str,
                       choices=["fastembed", "sentence_transformer", "medcpt"],
                       help=argparse.SUPPRESS)  # Hidden parameter
    parser.add_argument("--model_name", type=str,
                       help=argparse.SUPPRESS)  # Hidden parameter
    
    # LLM configuration
    parser.add_argument("--llm_type", type=str, choices=["local", "api"],
                       default="local", help="Type of LLM to use")
    parser.add_argument("--model", type=str, 
                       default="mistral_24b",
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
                          
    # Retriever GPU configuration
    retriever_group = parser.add_mutually_exclusive_group()
    retriever_group.add_argument("--retriever_gpu_id", type=int,
                                help="Specific GPU ID to use for retriever/embeddings")
    retriever_group.add_argument("--retriever_cpu", action="store_true",
                                help="Force CPU usage for retriever even if GPU is available")
    
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
            model_type=args.model,
            device=device,
            cache_dir=args.cache_dir,
            temperature=args.temperature
        )

def load_verification_results(input_file: str) -> Dict:
    """Load verification results from step 2."""
    try:
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        # Check if the data has the expected structure with "metadata" and "results" keys
        if isinstance(data, dict) and "results" in data:
            timestamp_print(f"Found structured output with 'results' key")
            results = data["results"]
            metadata = data.get("metadata", {})
            
            # Ensure results keys are strings
            if isinstance(results, dict):
                results = {str(k): v for k, v in results.items()}
            
            return {
                "metadata": metadata,
                "results": results
            }
        else:
            # Fallback for older format where the entire JSON is the results
            timestamp_print(f"Using legacy format - treating entire JSON as results")
            # Ensure keys are strings
            if isinstance(data, dict):
                data = {str(k): v for k, v in data.items()}
            
            return {
                "metadata": {},
                "results": data
            }
    except Exception as e:
        timestamp_print(f"Error loading verification results: {e}")
        raise

def format_entities_for_matching(verified_phenotypes: List[Dict]) -> List[Dict]:
    """Format verified phenotypes for HPO matching."""
    formatted_entities = []
    
    for phenotype in verified_phenotypes:
        # The RAGHPOMatcher expects entities with entity and context fields
        entity_text = phenotype.get("phenotype", "")  # Primary phenotype name
        original_entity = phenotype.get("original_entity", "")  # Original entity text
        context = phenotype.get("context", "")  # Context from verification
        
        if entity_text:  # Skip empty entities
            formatted_entities.append({
                "entity": entity_text,
                "original_entity": original_entity,
                "context": context,
                # Store original phenotype data for later
                "original_data": phenotype
            })
    
    return formatted_entities

def format_csv_results(results: Dict[str, Dict]) -> pd.DataFrame:
    """Format results as a CSV-friendly DataFrame."""
    csv_data = []
    
    for patient_id, case_data in results.items():
        matched_phenotypes = case_data.get("matched_phenotypes", [])
        
        for phenotype in matched_phenotypes:
            # Extract relevant fields for CSV
            entry = {
                "patient_id": patient_id,
                "phenotype": phenotype.get("phenotype", ""),
                "original_entity": phenotype.get("original_entity", ""),
                "hpo_term": phenotype.get("hpo_term", ""),
                "hpo_id": phenotype.get("hp_id", ""),
                "match_method": phenotype.get("match_method", ""),
                "confidence_score": phenotype.get("confidence_score", 0.0),
                "status": phenotype.get("status", "")
            }
            csv_data.append(entry)
    
    return pd.DataFrame(csv_data)

def main():
    """Main function to run HPO term matching pipeline."""
    try:
        # Parse command line arguments
        args = parse_arguments()
        
        timestamp_print(f"Starting HPO term matching process")
        
        # Setup devices - use the imported function
        devices = setup_device(args)
        llm_device = devices['llm']
        retriever_device = devices['embeddings']
        timestamp_print(f"Using device for LLM: {llm_device}")
        timestamp_print(f"Using device for embeddings: {retriever_device}")
        
        # Load system prompts
        timestamp_print(f"Loading system prompts from {args.system_prompt_file}")
        try:
            with open(args.system_prompt_file, 'r') as f:
                prompts = json.load(f)
                system_message_matching = prompts.get("system_message_II", "")
        except Exception as e:
            timestamp_print(f"Error loading system prompts: {e}")
            raise
        
        # Initialize LLM client - pass the correct string device
        timestamp_print(f"Initializing {args.llm_type} LLM client")
        llm_client = initialize_llm_client(args, llm_device)
        
        # Initialize embedding manager - pass the correct string device
        timestamp_print(f"Initializing {args.retriever} embedding manager")
        
        # Get model type (use new parameter name, fall back to legacy parameter if needed)
        model_type = args.retriever if args.retriever else args.model_type
        if not model_type:
            model_type = "fastembed"  # Default if neither parameter is provided
            
        # Get model name (use new parameter name, fall back to legacy parameter if needed)
        model_name = args.retriever_model if args.retriever_model else args.model_name
        if not model_name and model_type in ['fastembed', 'sentence_transformer']:
            model_name = "BAAI/bge-small-en-v1.5"  # Default if needed
            
        embedding_manager = EmbeddingsManager(
            model_type=model_type,
            model_name=model_name if model_type in ['fastembed', 'sentence_transformer'] else None,
            device=retriever_device
        )
        
        # Initialize HPO matcher
        timestamp_print(f"Initializing RAGHPOMatcher")
        matcher = RAGHPOMatcher(
            embeddings_manager=embedding_manager,
            llm_client=llm_client,
            system_message=system_message_matching
        )
        
       # Load verification results from step 2
        timestamp_print(f"Loading verification results from {args.input_file}")
        verification_data = load_verification_results(args.input_file)
        
        # Extract results and metadata
        verification_results = verification_data["results"]
        verification_metadata = verification_data["metadata"]
        
        timestamp_print(f"Loaded verification results for {len(verification_results)} cases")
        
        # Load embeddings
        timestamp_print(f"Loading embeddings from {args.embeddings_file}")
        try:
            embedded_documents = np.load(args.embeddings_file, allow_pickle=True)
            timestamp_print(f"Loaded {len(embedded_documents)} embedded documents")
        except Exception as e:
            timestamp_print(f"Error loading embeddings file: {e}")
            raise
        
        # Prepare matcher index
        timestamp_print(f"Preparing matcher index")
        matcher.prepare_index(embedded_documents)
        
        # Match verified phenotypes to HPO terms
        timestamp_print(f"Matching verified phenotypes to HPO terms")
        matched_results = {}
        checkpoint_counter = 0
        
        # Process each case
        for patient_id, case_data in tqdm(verification_results.items(), desc="Matching HPO terms"):
            if args.debug:
                timestamp_print(f"Processing case {patient_id}")
            
            # Get verified phenotypes
            verified_phenotypes = case_data.get("verified_phenotypes", [])
            
            if not verified_phenotypes:
                if args.debug:
                    timestamp_print(f"  No verified phenotypes for case {patient_id}")
                
                # Still add to results, just with empty matches
                matched_results[patient_id] = {
                    "original_text": case_data.get("original_text", ""),
                    "matched_phenotypes": [],
                    "stats": case_data.get("stats", {})
                }
                continue
            
            if args.debug:
                timestamp_print(f"  Matching {len(verified_phenotypes)} verified phenotypes")
            
            # Format entities for matching
            formatted_entities = format_entities_for_matching(verified_phenotypes)
            
            # Create lists for RAGHPOMatcher input
            entities = [entity["entity"] for entity in formatted_entities]
            contexts = [entity.get("context", "") for entity in formatted_entities]
            
            # Match entities to HPO terms
            matches = matcher.match_hpo_terms(entities, embedded_documents, contexts)
            
            # Combine match results with original data and format for output
            matched_phenotypes = []
            for i, match in enumerate(matches):
                if i < len(formatted_entities):
                    # Get original data
                    original_data = formatted_entities[i].get("original_data", {}).copy()
                    
                    # Add match data
                    phenotype_match = {
                        # Original phenotype data
                        **original_data,
                        
                        # Add HPO match information
                        "hpo_term": match.get("hpo_term"),
                        "hp_id": match.get("hpo_term"),  # HPO ID is the same as hpo_term for now
                        "match_method": match.get("match_method"),
                        "confidence_score": match.get("confidence_score"),
                        "top_candidates": match.get("top_candidates", [])[:args.top_k]  # Limit to top_k
                    }
                    
                    matched_phenotypes.append(phenotype_match)
            
            # Store result for this case
            matched_results[patient_id] = {
                # Preserve original data
                "original_text": case_data.get("original_text", ""),
                "matched_phenotypes": matched_phenotypes,
                "stats": case_data.get("stats", {})
            }
            
            # Add matching stats
            matched_results[patient_id]["stats"]["matched_phenotypes_count"] = len(matched_phenotypes)
            
            # Save checkpoint if interval reached
            checkpoint_counter += 1
            if checkpoint_counter >= args.checkpoint_interval:
                timestamp_print(f"Saving checkpoint after {checkpoint_counter} cases")
                with open(f"{os.path.splitext(args.output_file)[0]}_checkpoint.json", 'w') as f:
                    serializable_checkpoint = convert_to_serializable(matched_results)
                    json.dump(serializable_checkpoint, f, indent=2)
                checkpoint_counter = 0
        
        # Create the final output structure
        final_output = {
            "metadata": verification_metadata,  # Preserve original metadata
            "results": matched_results
        }
        
        # Save results to JSON with type conversion
        timestamp_print(f"Saving HPO match results to {args.output_file}")
        os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
        with open(args.output_file, 'w') as f:
            # Convert numpy types to Python native types before serializing
            serializable_results = convert_to_serializable(final_output)
            json.dump(serializable_results, f, indent=2)
    
        
        # Optionally save as CSV
        if args.csv_output:
            timestamp_print(f"Saving results as CSV to {args.csv_output}")
            csv_df = format_csv_results(matched_results)
            csv_df.to_csv(args.csv_output, index=False)
        
        # Print summary statistics
        total_cases = len(matched_results)
        total_phenotypes = sum(len(case.get("matched_phenotypes", [])) for case in matched_results.values())
        cases_with_phenotypes = sum(1 for case in matched_results.values() if case.get("matched_phenotypes", []))
        
        timestamp_print(f"HPO term matching complete:")
        timestamp_print(f"  Total cases: {total_cases}")
        timestamp_print(f"  Cases with matched phenotypes: {cases_with_phenotypes} ({cases_with_phenotypes/total_cases*100:.1f}%)")
        timestamp_print(f"  Total matched phenotypes: {total_phenotypes}")
        timestamp_print(f"  Average phenotypes per case: {total_phenotypes/total_cases:.2f}")
        
    except Exception as e:
        timestamp_print(f"Error: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()