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
from utils.data import parse_case_range

# Import project modules
from hporag.verify import HPOVerifierConfig, MultiStageHPOVerifierV2, MultiStageHPOVerifierV3, MultiStageHPOVerifierV4
# Import the V3 verifier
# You'll need to ensure MultiStageHPOVerifierV3.py is in the correct directory
# and properly imported in your project
from utils.embedding import EmbeddingsManager
from utils.llm_client import LocalLLMClient, APILLMClient
from utils.setup import setup_device, timestamp_print

def filter_hallucinated_entities(entities_with_contexts: List[Dict], min_context_length: int = 1) -> List[Dict]:
    """Filter out potentially hallucinated entities (those without valid contexts).
    
    Args:
        entities_with_contexts: List of entities with context information
        min_context_length: Minimum length of context to be considered valid
        
    Returns:
        Filtered list with only entities that have valid contexts
    """
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

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Verify extracted entities as phenotypes")
    
    # Input/output files
    parser.add_argument("--input_file", required=True, 
                       help="Input JSON file from extraction step")
    parser.add_argument("--output_file", required=True, 
                       help="Output JSON file for verification results")
    parser.add_argument("--embeddings_file", required=True,
                       help="NPY file containing HPO embeddings (G2GHPO_metadata.npy)")
    parser.add_argument("--config_file", 
                       help="Optional JSON file with verifier configuration parameters")
    # Add the new cases argument
    parser.add_argument("--cases", type=str,
                       help="Range of case IDs to process (format: start,end) e.g. '1,10'")
    # A/B/C Testing parameter
    parser.add_argument("--verifier_version", type=str, choices=["v1", "v2", "v3", "v4"], default="v1",
                       help="Verifier version to use: v1 (original), v2 (improved), or v3 (binary)")
    
    # System prompt configuration
    parser.add_argument("--system_prompt_file", default="data/prompts/system_prompts.json", 
                       help="File containing system prompts")
    
    # Verification configuration
    verification_group = parser.add_argument_group('Verification Configuration (overrides embedded config)')
    verification_group.add_argument("--use_retrieval_direct", type=lambda x: (str(x).lower() == 'true'),
                                   help="Use retrieval for direct phenotype verification")
    verification_group.add_argument("--use_retrieval_implies", type=lambda x: (str(x).lower() == 'true'),
                                   help="Use retrieval for implied phenotype check")
    verification_group.add_argument("--use_retrieval_extract", type=lambda x: (str(x).lower() == 'true'),
                                   help="Use retrieval for extracting implied phenotypes")
    verification_group.add_argument("--use_retrieval_validation", type=lambda x: (str(x).lower() == 'true'),
                                   help="Use retrieval for phenotype validation")
    verification_group.add_argument("--use_retrieval_implication", type=lambda x: (str(x).lower() == 'true'),
                                   help="Use retrieval for implication validation")
    verification_group.add_argument("--use_context_direct", type=lambda x: (str(x).lower() == 'true'),
                                   help="Use context for direct phenotype verification")
    verification_group.add_argument("--use_context_implies", type=lambda x: (str(x).lower() == 'true'),
                                   help="Use context for implied phenotype check")
    verification_group.add_argument("--use_context_extract", type=lambda x: (str(x).lower() == 'true'),
                                   help="Use context for extracting implied phenotypes")
    verification_group.add_argument("--use_context_validation", type=lambda x: (str(x).lower() == 'true'),
                                   help="Use context for phenotype validation")
    verification_group.add_argument("--use_context_implication", type=lambda x: (str(x).lower() == 'true'),
                                   help="Use context for implication validation")
    parser.add_argument("--lab_embeddings_file", type=str, default=None,
                   help="Path to lab test embeddings file for V4 verifier (lab_table_vectors.npy)")
    # Add this missing parameter
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
    
        # Add retriever arguments needed by utils.setup
    # Add these after the GPU configuration section
    retriever_group = parser.add_mutually_exclusive_group()
    retriever_group.add_argument("--retriever_gpu_id", type=int,
                            help="Specific GPU ID to use for retriever/embeddings")
    retriever_group.add_argument("--retriever_cpu", action="store_true",
                            help="Force CPU usage for retriever even if GPU is available")

    # If you need lab embeddings configuration
    lab_group = parser.add_mutually_exclusive_group()
    lab_group.add_argument("--lab_gpu_id", type=int,
                        help="Specific GPU ID to use for lab embeddings")
    lab_group.add_argument("--lab_cpu", action="store_true",
                        help="Force CPU usage for lab embeddings even if GPU is available")
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
    
    # Debug mode
    parser.add_argument("--debug", action="store_true", 
                       help="Enable debug output")
    
    return parser.parse_args()

def get_optimized_config() -> HPOVerifierConfig:
    """Return the optimized configuration based on previous experiments.
    
    This configuration achieved:
    - Precision: 0.68
    - Recall: 0.88
    - F1 Score: 0.77
    """
    optimized_config = {
        "retrieval": {
            "direct": True,
            "implies": True,
            "extract": True,
            "validation": True,
            "implication": True
        },
        "context": {
            "direct": True,
            "implies": True,
            "extract": True,
            "validation": True,
            "implication": True
        }
    }
    return HPOVerifierConfig.from_dict(optimized_config)


def create_verifier_config(args: argparse.Namespace) -> HPOVerifierConfig:
    """Create a HPOVerifierConfig from command line arguments or config file."""
    # Priority order:
    # 1. Command line argument --config_file (if provided)
    # 2. Command line arguments for individual settings
    # 3. Embedded optimized configuration
    
    # Check for config file first
    if args.config_file and os.path.exists(args.config_file):
        try:
            with open(args.config_file, 'r') as f:
                config_dict = json.load(f)
            timestamp_print(f"Loaded verifier configuration from {args.config_file}")
            return HPOVerifierConfig.from_dict(config_dict)
        except Exception as e:
            timestamp_print(f"Error loading config file: {e}. Falling back to other options.")
    
    # Check if any command line arguments were explicitly set
    cmd_args_provided = False
    for arg_name in vars(args):
        if arg_name.startswith('use_retrieval_') or arg_name.startswith('use_context_'):
            if getattr(args, arg_name) is not None:
                cmd_args_provided = True
                break
    
    # If command line arguments were explicitly provided, use them
    if cmd_args_provided:
        timestamp_print("Using configuration from command line arguments")
        return HPOVerifierConfig(
            use_retrieval_for_direct=args.use_retrieval_direct,
            use_retrieval_for_implies=args.use_retrieval_implies,
            use_retrieval_for_extract=args.use_retrieval_extract,
            use_retrieval_for_validation=args.use_retrieval_validation,
            use_retrieval_for_implication=args.use_retrieval_implication,
            use_context_for_direct=args.use_context_direct,
            use_context_for_implies=args.use_context_implies,
            use_context_for_extract=args.use_context_extract,
            use_context_for_validation=args.use_context_validation,
            use_context_for_implication=args.use_context_implication
        )
    
    # Otherwise, use the embedded optimized configuration
    timestamp_print("Using embedded optimized configuration")
    return get_optimized_config()


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


# 3. Modify the load_input_data function to filter cases
def load_input_data(input_file: str, start_id: int = None, end_id: int = None) -> Dict[int, Dict]:
    """Load and validate the extraction results from previous step, optionally filtering by case ID."""
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
            
        # Filter by case ID range if specified
        if start_id is not None and end_id is not None:
            filtered_data = {}
            for case_id, case_data in data.items():
                # Convert case_id to int if necessary
                case_id_int = int(case_id) if isinstance(case_id, str) else case_id
                
                if start_id <= case_id_int <= end_id:
                    filtered_data[case_id] = case_data
                    
            timestamp_print(f"Filtered data to {len(filtered_data)} cases (IDs {start_id}-{end_id})")
            data = filtered_data
            
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



def load_existing_results(output_file: str) -> Dict[int, Dict]:
    """Load existing verification results if available."""
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r') as f:
                data = json.load(f)
            
            # Handle case where the results are wrapped in a metadata structure
            if "results" in data:
                data = data["results"]
                
            # Convert string keys to integers if necessary
            if isinstance(data, dict) and all(isinstance(k, str) for k in data.keys()):
                data = {int(k): v for k, v in data.items()}
                
            timestamp_print(f"Loaded existing results for {len(data)} cases from {output_file}")
            return data
        except Exception as e:
            timestamp_print(f"Error loading existing results: {e}")
            return {}
    return {}


def save_checkpoint(results: Dict[int, Dict], output_file: str, checkpoint_num: int) -> None:
    """Save intermediate results to a checkpoint file."""
    checkpoint_file = f"{os.path.splitext(output_file)[0]}_checkpoint{checkpoint_num}.json"
    with open(checkpoint_file, 'w') as f:
        json.dump(results, f, indent=2)
    timestamp_print(f"Saved checkpoint to {checkpoint_file}")


def format_entities_for_verification(entities_with_contexts: List[Dict], min_context_length: int = 1) -> List[Dict]:
    """Format the entities with contexts for the verifier, filtering out potential hallucinations.
    
    Args:
        entities_with_contexts: Raw entities from extraction step
        min_context_length: Minimum context length to consider valid (filter out shorter contexts)
        
    Returns:
        Formatted entities ready for verification (with hallucinations filtered out)
    """
    # First, filter out potentially hallucinated entities (those without contexts)
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


def verify_cases(cases: Dict[int, Dict], verifier, 
                args: argparse.Namespace, existing_results: Dict = None) -> Dict[int, Dict]:
    """Process each case to verify entities as phenotypes."""
    results = existing_results or {}
    checkpoint_counter = 0
    
    # Determine which cases need processing
    pending_cases = {case_id: case_data for case_id, case_data in cases.items() 
                   if case_id not in results or not results[case_id].get('verified_phenotypes')}
    
    timestamp_print(f"Verifying phenotypes for {len(pending_cases)} cases out of {len(cases)} total cases")
    
    # Convert to list for progress tracking
    case_items = list(pending_cases.items())
    
    # Use tqdm for progress tracking
    for i, (case_id, case_data) in enumerate(tqdm(case_items, desc="Verifying phenotypes")):
        try:
            if args.debug:
                timestamp_print(f"Processing case {i+1}/{len(pending_cases)} (ID: {case_id})")
            
            # Format entities for verification with anti-hallucination filter
            entities_with_contexts = case_data.get("entities_with_contexts", [])
            
            if args.debug:
                timestamp_print(f"  Processing {len(entities_with_contexts)} raw entities")
            
            formatted_entities = format_entities_for_verification(
                entities_with_contexts, 
                min_context_length=args.min_context_length
            )
            
            if args.debug:
                timestamp_print(f"  Verifying {len(formatted_entities)} entities after filtering")
            
            # Skip processing if all entities were filtered out
            if not formatted_entities:
                timestamp_print(f"  All entities for case {case_id} were filtered out as potential hallucinations")
                results[case_id] = {
                    "original_text": case_data.get("clinical_text", ""),
                    "verified_phenotypes": [],
                    "filtered_entities_count": len(entities_with_contexts),
                    "note": "All entities filtered as potential hallucinations (no valid contexts)",
                    "verifier_version": args.verifier_version  # Record which verifier was used
                }
                continue
            
            # Verify entities
            verified_phenotypes = verifier.batch_process(formatted_entities)
            
            if args.debug:
                timestamp_print(f"  Identified {len(verified_phenotypes)} phenotypes (direct or implied)")
            
            # Store results with tracking of filtered entities
            original_count = len(entities_with_contexts)
            filtered_count = original_count - len(formatted_entities)
            
            results[case_id] = {
                "original_text": case_data.get("clinical_text", ""),  # Include original text if available
                "verified_phenotypes": verified_phenotypes,
                "stats": {
                    "original_entity_count": original_count,
                    "filtered_hallucinations": filtered_count,
                    "entities_verified": len(formatted_entities),
                    "phenotypes_found": len(verified_phenotypes)
                },
                "verifier_version": args.verifier_version  # Record which verifier was used
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
                "original_text": case_data.get("clinical_text", ""),
                "verified_phenotypes": [],
                "stats": {
                    "original_entity_count": len(case_data.get("entities_with_contexts", [])),
                    "filtered_hallucinations": 0,
                    "entities_verified": 0,
                    "phenotypes_found": 0
                },
                "error": str(e),
                "verifier_version": args.verifier_version  # Record which verifier was used
            }
    
    return results


def main():
    """Main function to run the phenotype verification pipeline."""
    try:
        # Parse command line arguments
        args = parse_arguments()
        
        timestamp_print(f"Starting phenotype verification process using {args.verifier_version}")
        # Parse case range if provided
        start_id, end_id = None, None
        if args.cases:
            start_id, end_id = parse_case_range(args.cases)
            timestamp_print(f"Processing case ID range: {start_id} to {end_id}")
        # Setup devices
        devices = setup_device(args)
        timestamp_print(f"Using device for LLM: {devices['llm']}")
        timestamp_print(f"Using device for embeddings: {devices['embeddings']}")
        
        # Load system prompts
        timestamp_print(f"Loading system prompts from {args.system_prompt_file}")
        try:
            with open(args.system_prompt_file, 'r') as f:
                prompts = json.load(f)
        except Exception as e:
            timestamp_print(f"Error loading system prompts: {e}")
            raise
        
        # Initialize LLM client
        timestamp_print(f"Initializing {args.llm_type} LLM client")
        llm_client = initialize_llm_client(args, devices['llm'])
        
        # Initialize embedding manager
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
            device=devices['embeddings']
        )
        
        # Create verifier configuration
        timestamp_print(f"Creating verifier configuration")
        verifier_config = create_verifier_config(args)
        
        # Always display the configuration being used
        config_dict = verifier_config.to_dict()
        timestamp_print(f"Using verifier config:")
        timestamp_print(f"  Retrieval settings: {json.dumps(config_dict['retrieval'], indent=2)}")
        timestamp_print(f"  Context settings: {json.dumps(config_dict['context'], indent=2)}")
        
        # Initialize the appropriate verifier based on version argument
        timestamp_print(f"Initializing {args.verifier_version.upper()} verifier")
        if args.verifier_version == "v1":
            # verifier = ConfigurableHPOVerifier(
            #     embedding_manager=embedding_manager,
            #     llm_client=llm_client,
            #     config=verifier_config,
            #     debug=args.debug
            # )
            raise(NotImplementedError("V1 verifier is not implemented in this version."))
            timestamp_print("Using original HPO verifier (V1)")
        elif args.verifier_version == "v2":  
            verifier = MultiStageHPOVerifierV2(
                embedding_manager=embedding_manager,
                llm_client=llm_client,
                config=verifier_config,
                debug=args.debug
            )
            timestamp_print("Using improved HPO verifier (V2)")
        elif args.verifier_version == "v3":  # v3
            verifier = MultiStageHPOVerifierV3(
                embedding_manager=embedding_manager,
                llm_client=llm_client,
                config=verifier_config,
                debug=args.debug
            )
            timestamp_print("Using binary HPO verifier (V3)")
        else:
            # Add this to the section that initializes the verifier
            timestamp_print(f"Initializing MultiStageHPOVerifierV4 with lab test analysis")
            verifier = MultiStageHPOVerifierV4(
                embedding_manager=embedding_manager,
                llm_client=llm_client,
                config=verifier_config,
                debug=args.debug,
                lab_embeddings_file=args.lab_embeddings_file
            )
        # Load embeddings
        timestamp_print(f"Loading embeddings from {args.embeddings_file}")
        try:
            embedded_documents = np.load(args.embeddings_file, allow_pickle=True)
            timestamp_print(f"Loaded {len(embedded_documents)} embedded documents")
        except Exception as e:
            timestamp_print(f"Error loading embeddings file: {e}")
            raise
        
        # Prepare verifier index
        timestamp_print(f"Preparing verifier index")
        verifier.prepare_index(embedded_documents)
        
        # Load input data from extraction step
        timestamp_print(f"Loading extraction results from {args.input_file}")
        cases = load_input_data(args.input_file, start_id, end_id)
        timestamp_print(f"Loaded {len(cases)} cases with extracted entities")
        
        # Check for existing results if resuming
        existing_results = {}
        if args.resume:
            existing_results = load_existing_results(args.output_file)
        
        # Modify output filename to include verifier version if not resuming
        output_file = args.output_file
        if not args.resume:
            base, ext = os.path.splitext(args.output_file)
            output_file = f"{base}_{args.verifier_version}{ext}"
            timestamp_print(f"Output will be saved to {output_file}")
        
        # Verify phenotypes
        timestamp_print(f"Starting phenotype verification with {args.verifier_version}")
        results = verify_cases(cases, verifier, args, existing_results)
        
        # Save results to JSON
        timestamp_print(f"Saving verification results to {output_file}")
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        
        # Add metadata about the verification run
        metadata = {
            "verifier_version": args.verifier_version,
            "verification_timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "configuration": config_dict,
            "model_info": {
                "llm_type": args.llm_type,
                "model": args.model,
                "temperature": args.temperature,
                "retriever": model_type,
                "retriever_model": model_name
            }
        }
        
        # Create final output with metadata
        final_output = {
            "metadata": metadata,
            "results": results
        }
        
        with open(output_file, 'w') as f:
            json.dump(final_output, f, indent=2)
        
        # Print summary
        # FIXED: Changed the total_input_entities calculation to avoid the int + list error
        total_input_entities = sum(len(case_data.get("entities_with_contexts", [])) for case_data in cases.values())
        total_filtered_entities = sum(results[case_id].get("stats", {}).get("filtered_hallucinations", 0) for case_id in results)
        total_verified_entities = sum(results[case_id].get("stats", {}).get("entities_verified", 0) for case_id in results)
        total_verified_phenotypes = sum(len(results[case_id].get("verified_phenotypes", [])) for case_id in results)
        
        # Calculate rates
        filtered_rate = (total_filtered_entities / total_input_entities * 100) if total_input_entities > 0 else 0
        verification_rate = (total_verified_phenotypes / total_verified_entities * 100) if total_verified_entities > 0 else 0
        overall_rate = (total_verified_phenotypes / total_input_entities * 100) if total_input_entities > 0 else 0
        
        timestamp_print(f"Verification complete with {args.verifier_version}:")
        timestamp_print(f"  Total input entities: {total_input_entities}")
        timestamp_print(f"  Filtered out as potential hallucinations: {total_filtered_entities} ({filtered_rate:.1f}%)")
        timestamp_print(f"  Entities after filtering: {total_verified_entities}")
        timestamp_print(f"  Verified as phenotypes: {total_verified_phenotypes} ({verification_rate:.1f}% of filtered entities)")
        timestamp_print(f"  Overall yield: {overall_rate:.1f}% (phenotypes from original entities)")
        
        timestamp_print(f"Phenotype verification with {args.verifier_version} completed successfully.")
    
    except Exception as e:
        timestamp_print(f"Critical error: {e}")
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()