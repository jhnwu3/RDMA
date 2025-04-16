#!/usr/bin/env python3
import argparse
import json
import os
import pandas as pd
import numpy as np
import torch
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
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


def timestamp_print(message: str) -> None:
    """Print message with timestamp."""
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Direct HPO Term Matching without Verification")
    
    # Input/output files
    parser.add_argument("--input_file", required=True, 
                       help="Input JSON file from extraction step (step 1)")
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
    
    # Debug mode
    parser.add_argument("--debug", action="store_true", 
                       help="Enable debug output")
    
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


def load_input_data(input_file: str) -> Dict[int, Dict]:
    """Load and validate the extraction results."""
    try:
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        # Handle different possible input formats
        if isinstance(data, dict):
            # Format is {case_id: case_data}
            # Convert string keys to integers if necessary
            if all(isinstance(k, str) for k in data.keys()):
                data = {int(k): v for k, v in data.items()}
        elif isinstance(data, list):
            # Format is [{note_id: id, entities_with_contexts: [...], ...}, ...]
            # Convert to {id: {entities_with_contexts: [...], ...}, ...}
            data_dict = {}
            for item in data:
                if "note_id" in item:
                    case_id = item.pop("note_id")
                    data_dict[case_id] = item
                else:
                    timestamp_print(f"Warning: Found item without note_id in input file. Skipping: {item}")
            data = data_dict
        else:
            raise ValueError(f"Unsupported input format in {input_file}")
            
        # Validate structure with entities_with_contexts in each case
        for case_id, case_data in data.items():
            if not isinstance(case_data, dict):
                raise ValueError(f"Case {case_id} data must be a dictionary")
            if "entities_with_contexts" not in case_data:
                timestamp_print(f"Warning: Case {case_id} missing 'entities_with_contexts' field")
                case_data["entities_with_contexts"] = []
        
        return data
    
    except (json.JSONDecodeError, ValueError) as e:
        timestamp_print(f"Error loading input file: {e}")
        raise


def format_entities_for_matching(entities_with_contexts: List[Dict]) -> List[Dict]:
    """Format the entities with contexts for HPO matching.
    
    Args:
        entities_with_contexts: Raw entities from extraction step
        
    Returns:
        Formatted entities ready for HPO matching
    """
    formatted_entities = []
    
    for entity_data in entities_with_contexts:
        # Ensure we have the expected fields
        entity = entity_data.get("entity", "") or entity_data.get("phrase", "")
        context = entity_data.get("context", "") or entity_data.get("original_sentence", "")
        
        if entity:  # Skip empty entities
            formatted_entities.append({
                "entity": entity,
                "context": context
            })
    
    return formatted_entities


def match_cases(cases: Dict[int, Dict], matcher, 
                args, embedded_documents, system_message_matching) -> Dict[int, Dict]:
    """Process each case to match entities to HPO terms."""
    results = {}
    checkpoint_counter = 0
    
    # Convert to list for progress tracking
    case_items = list(cases.items())
    
    # Use tqdm for progress tracking
    for i, (case_id, case_data) in enumerate(tqdm(case_items, desc="Matching HPO Terms")):
        try:
            if args.debug:
                timestamp_print(f"Processing case {i+1}/{len(cases)} (ID: {case_id})")
            
            # Format entities for matching
            entities_with_contexts = case_data.get("entities_with_contexts", [])
            
            if args.debug:
                timestamp_print(f"  Processing {len(entities_with_contexts)} raw entities")
            
            formatted_entities = format_entities_for_matching(entities_with_contexts)
            
            if args.debug:
                timestamp_print(f"  Matching {len(formatted_entities)} entities")
            
            # Skip processing if no entities
            if not formatted_entities:
                timestamp_print(f"  No entities to match for case {case_id}")
                results[case_id] = {
                    "original_text": case_data.get("clinical_text", case_data.get("clinical_note", "")),
                    "matched_phenotypes": [],
                    "stats": {
                        "original_entity_count": len(entities_with_contexts),
                        "entities_matched": 0,
                        "phenotypes_found": 0
                    }
                }
                continue
            
            # Prepare lists for matching
            entities = [entity["entity"] for entity in formatted_entities]
            contexts = [entity.get("context", "") for entity in formatted_entities]
            
            # Match entities to HPO terms
            matches = matcher.match_hpo_terms(entities, embedded_documents, contexts)
            
            # Combine match results with original data
            matched_phenotypes = []
            for i, match in enumerate(matches):
                if i < len(formatted_entities):
                    original_entity = formatted_entities[i]
                    
                    # Add match data
                    phenotype_match = {
                        # Original entity data
                        "entity": original_entity["entity"],
                        "context": original_entity["context"],
                        
                        # Add HPO match information
                        "hpo_term": match.get("hpo_term"),
                        "hp_id": match.get("hpo_term"),  # HPO ID is the same as hpo_term for now
                        "match_method": match.get("match_method"),
                        "confidence_score": match.get("confidence_score"),
                        "top_candidates": match.get("top_candidates", [])[:args.top_k]  # Limit to top_k
                    }
                    
                    matched_phenotypes.append(phenotype_match)
            
            # Store result for this case
            results[case_id] = {
                # Preserve original data
                "original_text": case_data.get("clinical_text", case_data.get("clinical_note", "")),
                "matched_phenotypes": matched_phenotypes,
                "stats": {
                    "original_entity_count": len(entities_with_contexts),
                    "entities_matched": len(formatted_entities),
                    "phenotypes_found": len(matched_phenotypes)
                }
            }
            
            # Save checkpoint if interval reached
            checkpoint_counter += 1
            if checkpoint_counter >= args.checkpoint_interval:
                timestamp_print(f"Saving checkpoint after {checkpoint_counter} cases")
                checkpoint_file = f"{os.path.splitext(args.output_file)[0]}_checkpoint.json"
                with open(checkpoint_file, 'w') as f:
                    json.dump(results, f, indent=2)
                checkpoint_counter = 0
                
        except Exception as e:
            timestamp_print(f"Error processing case {case_id}: {e}")
            if args.debug:
                traceback.print_exc()
            # Still add the case to results but mark as failed
            results[case_id] = {
                "original_text": case_data.get("clinical_text", case_data.get("clinical_note", "")),
                "matched_phenotypes": [],
                "stats": {
                    "original_entity_count": len(case_data.get("entities_with_contexts", [])),
                    "entities_matched": 0,
                    "phenotypes_found": 0
                },
                "error": str(e)
            }
    
    return results


def format_csv_results(results: Dict[str, Dict]) -> pd.DataFrame:
    """Format results as a CSV-friendly DataFrame."""
    csv_data = []
    
    for patient_id, case_data in results.items():
        matched_phenotypes = case_data.get("matched_phenotypes", [])
        
        for phenotype in matched_phenotypes:
            # Extract relevant fields for CSV
            entry = {
                "patient_id": patient_id,
                "entity": phenotype.get("entity", ""),
                "context": phenotype.get("context", ""),
                "hpo_term": phenotype.get("hpo_term", ""),
                "hpo_id": phenotype.get("hp_id", ""),
                "match_method": phenotype.get("match_method", ""),
                "confidence_score": phenotype.get("confidence_score", 0.0)
            }
            csv_data.append(entry)
    
    return pd.DataFrame(csv_data)


def main():
    """Main function to run the direct HPO matching pipeline."""
    try:
        # Parse command line arguments
        args = parse_arguments()
        
        timestamp_print(f"Starting direct HPO term matching process")
        
        # Setup device
        device = setup_device(args)
        timestamp_print(f"Using device: {device}")
        
        # Load system prompts
        timestamp_print(f"Loading system prompts from {args.system_prompt_file}")
        try:
            with open(args.system_prompt_file, 'r') as f:
                prompts = json.load(f)
                system_message_matching = prompts.get("system_message_II", "")
        except Exception as e:
            timestamp_print(f"Error loading system prompts: {e}")
            raise
        
        # Initialize LLM client
        timestamp_print(f"Initializing {args.llm_type} LLM client")
        llm_client = initialize_llm_client(args, device)
        
        # Initialize embedding manager
        timestamp_print(f"Initializing {args.retriever} embedding manager")
        
        # Get model type (use new parameter name)
        model_type = args.retriever
        model_name = args.retriever_model if args.retriever in ['fastembed', 'sentence_transformer'] else None
        
        embedding_manager = EmbeddingsManager(
            model_type=model_type,
            model_name=model_name,
            device=device
        )
        
        # Initialize HPO matcher
        timestamp_print(f"Initializing RAGHPOMatcher")
        matcher = RAGHPOMatcher(
            embeddings_manager=embedding_manager,
            llm_client=llm_client,
            system_message=system_message_matching
        )
        
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
        
        # Load input data from extraction step
        timestamp_print(f"Loading extraction results from {args.input_file}")
        cases = load_input_data(args.input_file)
        timestamp_print(f"Loaded {len(cases)} cases with extracted entities")
        
        # Check for existing results if resuming
        existing_results = {}
        if args.resume and os.path.exists(args.output_file):
            try:
                with open(args.output_file, 'r') as f:
                    existing_data = json.load(f)
                    existing_results = existing_data.get('results', {}) if 'results' in existing_data else existing_data
                timestamp_print(f"Loaded existing results for {len(existing_results)} cases")
            except Exception as e:
                timestamp_print(f"Error loading existing results: {e}")
        
        # Match HPO terms
        timestamp_print(f"Starting direct HPO term matching")
        results = match_cases(cases, matcher, args, embedded_documents, system_message_matching)
        
        # Prepare final output with metadata
        final_output = {
            "metadata": {
                "matching_timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "input_file": args.input_file,
                "embeddings_file": args.embeddings_file,
                "model_info": {
                    "llm_type": args.llm_type,
                    "model": args.model,
                    "retriever": model_type,
                    "retriever_model": model_name,
                    "temperature": args.temperature
                }
            },
            "results": results
        }
        
        # Save results to JSON
        timestamp_print(f"Saving matching results to {args.output_file}")
        os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
        with open(args.output_file, 'w') as f:
            json.dump(final_output, f, indent=2)
        
        # Optionally save as CSV
        if args.csv_output:
            timestamp_print(f"Saving results as CSV to {args.csv_output}")
            csv_df = format_csv_results(results)
            csv_df.to_csv(args.csv_output, index=False)
        
        # Print summary statistics
        total_input_entities = sum(len(case_data.get("entities_with_contexts", [])) for case_data in cases.values())
        total_matched_phenotypes = sum(len(results[case_id].get("matched_phenotypes", [])) for case_id in results)
        
        timestamp_print("Direct HPO Term Matching Summary:")
        timestamp_print(f"  Total input entities: {total_input_entities}")
        timestamp_print(f"  Total matched phenotypes: {total_matched_phenotypes}")
        matching_rate = total_matched_phenotypes / total_input_entities * 100 if total_input_entities > 0 else 0
        timestamp_print(f"  Matching rate: {matching_rate:.1f}%")
        
        timestamp_print("Matching process completed successfully.")
    
    except Exception as e:
        timestamp_print(f"Critical error: {e}")
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()