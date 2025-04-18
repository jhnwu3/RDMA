#!/usr/bin/env python3
import argparse
import json
import os
import torch
import numpy as np
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
from rdrag.rd_match import RAGRDMatcher
from utils.embedding import EmbeddingsManager
from utils.llm_client import LocalLLMClient, APILLMClient
from utils.setup import setup_device

# Import the new MultiStageRDVerifier class
from rdrag.verify import MultiStageRDVerifier

def timestamp_print(message: str) -> None:
    """Print message with timestamp."""
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Verify extracted entities as rare diseases")
    
    # Input/output files
    parser.add_argument("--input_file", required=True, 
                       help="Input JSON file from extraction step")
    parser.add_argument("--output_file", required=True, 
                       help="Output JSON file for verification results")
    parser.add_argument("--embeddings_file", required=True,
                       help="NPY file containing rare disease embeddings")
    
    # System prompt configuration
    parser.add_argument("--system_prompt", type=str, 
                       default="You are a medical expert specializing in rare diseases.",
                       help="System prompt for LLM verification")
    
    # Verifier configuration
    parser.add_argument("--verifier_type", type=str, 
                       choices=["simple", "multi_stage"],
                       default="simple",
                       help="Type of verifier to use (default: simple)")
    
    # Multi-stage verifier specific configuration
    parser.add_argument("--abbreviations_file", type=str,
                       help="Path to abbreviations embeddings file for multi-stage verifier")
    parser.add_argument("--use_abbreviations", action="store_true",
                       help="Enable abbreviation lookup for multi-stage verifier")
    parser.add_argument("--config_file", 
                       help="Optional JSON file with verifier configuration parameters")
    
    # Verification configuration
    parser.add_argument("--min_context_length", type=int, default=1,
                       help="Minimum context length to consider valid (filter out shorter contexts)")
    
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
    
    # GPU configuration for retriever separately
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
            model_type=args.model_type,
            device=device,
            cache_dir=args.cache_dir,
            temperature=args.temperature
        )


def load_input_data(input_file: str) -> Dict:
    """Load and validate the extraction results from previous step."""
    try:
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        # Validate basic structure
        if not isinstance(data, dict):
            raise ValueError(f"Invalid input format: Expected a dictionary mapping case IDs to case data")
        
        # Validate presence of entities_with_contexts
        for case_id, case_data in data.items():
            if not isinstance(case_data, dict):
                raise ValueError(f"Invalid case data for {case_id}: Expected a dictionary")
            if "entities_with_contexts" not in case_data:
                timestamp_print(f"Warning: Case {case_id} missing 'entities_with_contexts' field")
                case_data["entities_with_contexts"] = []
        
        return data
    
    except (json.JSONDecodeError, ValueError) as e:
        timestamp_print(f"Error loading input file: {e}")
        raise


def load_existing_results(output_file: str) -> Dict:
    """Load existing verification results if available."""
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


def filter_hallucinated_entities(entities_with_contexts: List[Dict], min_context_length: int = 1) -> List[Dict]:
    """Filter out potentially hallucinated entities (those without valid contexts)."""
    initial_count = len(entities_with_contexts)
    
    # Keep only entities with non-empty context
    filtered_entities = [
        entity for entity in entities_with_contexts
        if entity.get("context") and len(entity.get("context", "").strip()) >= min_context_length
    ]
    
    removed_count = initial_count - len(filtered_entities)
    
    if removed_count > 0:
        timestamp_print(f"  Filtered out {removed_count} potentially hallucinated entities with no context")
    
    return filtered_entities


def format_entities_for_verification(entities_with_contexts: List[Dict], min_context_length: int = 1) -> List[Dict]:
    """Format the entities with contexts for verification, filtering out potential hallucinations."""
    # First, filter out potentially hallucinated entities
    filtered_entities = filter_hallucinated_entities(entities_with_contexts, min_context_length)
    
    # Then format the remaining entities for verification
    formatted_entities = []
    
    for entity_data in filtered_entities:
        # Ensure we have the expected fields
        entity = entity_data.get("entity", "")
        context = entity_data.get("context", "")
        
        if entity:  # Skip empty entities
            formatted_entities.append({
                "entity": entity,
                "context": context
            })
    
    return formatted_entities


def verify_cases(cases: Dict, verifier, args: argparse.Namespace, 
                existing_results: Dict = None) -> Dict:
    """Process each case to verify entities as rare diseases."""
    results = existing_results or {}
    checkpoint_counter = 0
    
    # Determine which cases need processing
    pending_cases = {case_id: case_data for case_id, case_data in cases.items() 
                   if case_id not in results or not results[case_id].get('verified_rare_diseases')}
    
    timestamp_print(f"Verifying rare diseases for {len(pending_cases)} cases out of {len(cases)} total cases")
    
    # Convert to list for progress tracking
    case_items = list(pending_cases.items())
    
    # Use tqdm for progress tracking
    for i, (case_id, case_data) in enumerate(tqdm(case_items, desc="Verifying rare diseases")):
        try:
            if args.debug:
                timestamp_print(f"Processing case {i+1}/{len(pending_cases)} (ID: {case_id})")
            
            # Get entities with contexts
            entities_with_contexts = case_data.get("entities_with_contexts", [])
            
            if args.debug:
                timestamp_print(f"  Processing {len(entities_with_contexts)} raw entities")
            
            # Format entities for verification
            formatted_entities = []
            for ewc in entities_with_contexts:
                # Handle different possible formats
                if "entity" in ewc:
                    entity = ewc["entity"]
                    context = ewc.get("context", "")
                elif "term" in ewc:
                    entity = ewc["term"]
                    context = ewc.get("context", "")
                else:
                    continue
                
                formatted_entities.append({
                    "entity": entity,
                    "context": context
                })
            
            # Filter entities based on context length
            formatted_entities = filter_hallucinated_entities(formatted_entities, args.min_context_length)
            
            if args.debug:
                timestamp_print(f"  Verifying {len(formatted_entities)} entities after filtering")
            
            # Skip processing if all entities were filtered out
            if not formatted_entities:
                timestamp_print(f"  All entities for case {case_id} were filtered out as potential hallucinations")
                results[case_id] = {
                    "clinical_text": case_data.get("clinical_text", ""),
                    "verified_rare_diseases": [],
                    "filtered_entities_count": len(entities_with_contexts),
                    "note": "All entities filtered as potential hallucinations (no valid contexts)",
                    "verifier_type": args.verifier_type
                }
                continue
                
            # Use the appropriate verification method based on verifier type
            if args.verifier_type == "multi_stage":
                # Use batch_process method for MultiStageRDVerifier
                verified_rare_diseases = verifier.batch_process(formatted_entities)
            else:
                # For the simple verifier, process each entity individually
                verified_rare_diseases = []
                
                for entity_data in formatted_entities:
                    entity = entity_data["entity"]
                    context = entity_data["context"]
                    
                    # Get candidates for verification
                    candidates = verifier._retrieve_candidates(entity)
                    
                    # Verify if it's a rare disease
                    is_rare_disease = verifier._verify_rare_disease(entity, candidates[:5])
                    
                    if is_rare_disease:
                        # Add to verified list
                        verified_entity = {
                            "entity": entity,
                            "context": context,
                            "is_verified": True,
                            "status": "rare_disease",
                        }
                        verified_rare_diseases.append(verified_entity)
                        
                        if args.debug:
                            timestamp_print(f"  ✓ Verified '{entity}' as a rare disease")
                    else:
                        if args.debug:
                            timestamp_print(f"  ✗ '{entity}' is not a rare disease")
            
            # Store results with tracking of filtered entities
            original_count = len(entities_with_contexts)
            filtered_count = original_count - len(formatted_entities)
            
            results[case_id] = {
                "clinical_text": case_data.get("clinical_text", ""),
                "metadata": case_data.get("metadata", {}),
                "verified_rare_diseases": verified_rare_diseases,
                "stats": {
                    "original_entity_count": original_count,
                    "filtered_hallucinations": filtered_count,
                    "entities_verified": len(formatted_entities),
                    "rare_diseases_found": len(verified_rare_diseases)
                },
                "verifier_type": args.verifier_type
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
                "verified_rare_diseases": [],
                "stats": {
                    "original_entity_count": len(case_data.get("entities_with_contexts", [])),
                    "filtered_hallucinations": 0,
                    "entities_verified": 0,
                    "rare_diseases_found": 0
                },
                "error": str(e),
                "verifier_type": args.verifier_type
            }
    
    return results


def main():
    """Main function to run the rare disease verification pipeline."""
    try:
        # Parse command line arguments
        args = parse_arguments()
        
        timestamp_print(f"Starting rare disease verification process with {args.verifier_type} verifier")
        
        # Setup device
        devices = setup_device(args)
        timestamp_print(f"Using device for LLM: {devices['llm']}")
        timestamp_print(f"Using device for embeddings: {devices['retriever']}")
        
        # Initialize LLM client
        timestamp_print(f"Initializing {args.llm_type} LLM client")
        llm_client = initialize_llm_client(args, devices['llm'])
        
        # Initialize embedding manager
        timestamp_print(f"Initializing {args.retriever} embedding manager")
        embedding_manager = EmbeddingsManager(
            model_type=args.retriever,
            model_name=args.retriever_model if args.retriever in ['fastembed', 'sentence_transformer'] else None,
            device=devices['retriever']
        )
        
        # Load embeddings
        timestamp_print(f"Loading embeddings from {args.embeddings_file}")
        try:
            embedded_documents = np.load(args.embeddings_file, allow_pickle=True)
            timestamp_print(f"Loaded {len(embedded_documents)} embedded documents")
        except Exception as e:
            timestamp_print(f"Error loading embeddings file: {e}")
            raise
        
        # Initialize the appropriate verifier based on type argument
        timestamp_print(f"Initializing {args.verifier_type} verifier")
        if args.verifier_type == "multi_stage":
            # Create verifier configuration
            config = None
            
            # Initialize multi-stage verifier
            verifier = MultiStageRDVerifier(
                embedding_manager=embedding_manager,
                llm_client=llm_client,
                config=config,
                debug=args.debug,
                abbreviations_file=args.abbreviations_file,
                use_abbreviations=args.use_abbreviations
            )
            timestamp_print("Using multi-stage rare disease verifier")
        else:
            # Initialize simple RAGRDMatcher verifier
            verifier = RAGRDMatcher(
                embeddings_manager=embedding_manager,
                llm_client=llm_client,
                system_message=args.system_prompt
            )
            timestamp_print("Using simple rare disease verifier")
        
        # Prepare verifier index
        timestamp_print(f"Preparing verifier index")
        verifier.prepare_index(embedded_documents)
        
        # Load input data from extraction step
        timestamp_print(f"Loading extraction results from {args.input_file}")
        cases = load_input_data(args.input_file)
        timestamp_print(f"Loaded {len(cases)} cases with extracted entities")
        
        # Check for existing results if resuming
        existing_results = {}
        if args.resume:
            existing_results = load_existing_results(args.output_file)
        
        # Verify rare diseases
        timestamp_print(f"Starting rare disease verification")
        results = verify_cases(cases, verifier, args, existing_results)
        
        # Save results to JSON
        timestamp_print(f"Saving verification results to {args.output_file}")
        os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
        
        # Add metadata about the verification run
        metadata = {
            "verifier_type": args.verifier_type,
            "verification_timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "model_info": {
                "llm_type": args.llm_type,
                "model_type": args.model_type,
                "temperature": args.temperature,
                "retriever": args.retriever,
                "retriever_model": args.retriever_model
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
        total_input_entities = sum(len(case_data.get("entities_with_contexts", [])) for case_data in cases.values())
        total_filtered_entities = sum(results[case_id].get("stats", {}).get("filtered_hallucinations", 0) for case_id in results)
        total_verified_entities = sum(results[case_id].get("stats", {}).get("entities_verified", 0) for case_id in results)
        total_rare_diseases = sum(len(results[case_id].get("verified_rare_diseases", [])) for case_id in results)
        
        # Calculate rates
        filtered_rate = (total_filtered_entities / total_input_entities * 100) if total_input_entities > 0 else 0
        verification_rate = (total_rare_diseases / total_verified_entities * 100) if total_verified_entities > 0 else 0
        overall_rate = (total_rare_diseases / total_input_entities * 100) if total_input_entities > 0 else 0
        
        timestamp_print(f"Verification complete:")
        timestamp_print(f"  Total input entities: {total_input_entities}")
        timestamp_print(f"  Filtered out as potential hallucinations: {total_filtered_entities} ({filtered_rate:.1f}%)")
        timestamp_print(f"  Entities after filtering: {total_verified_entities}")
        timestamp_print(f"  Verified as rare diseases: {total_rare_diseases} ({verification_rate:.1f}% of filtered entities)")
        timestamp_print(f"  Overall yield: {overall_rate:.1f}% (rare diseases from original entities)")
        
        timestamp_print(f"Rare disease verification completed successfully.")
    
    except Exception as e:
        timestamp_print(f"Critical error: {e}")
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()