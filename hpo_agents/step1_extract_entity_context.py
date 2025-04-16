#!/usr/bin/env python3
import argparse
import json
import os
import torch
import numpy as np
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
# Import project modules
from hporag.entity import IterativeLLMEntityExtractor, LLMEntityExtractor, MultiIterativeExtractor, RetrievalEnhancedEntityExtractor
from hporag.context import ContextExtractor
from utils.llm_client import LocalLLMClient, APILLMClient
from utils.embedding import EmbeddingsManager
from utils.data import parse_case_range
from utils.setup import setup_device, timestamp_print
from hporag.phenogpt import PhenoGPT
from hporag.entity import PhenoGPTEntityExtractor

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Verify extracted entities as phenotypes")
    
    # Input/output files
    parser.add_argument("--input_file", default="data/dataset/mine_hpo.json",
                       help="Input JSON file with clinical notes (default: data/dataset/mine_hpo.json)")
    parser.add_argument("--output_file", required=True, 
                       help="Output JSON file for verification results")
    
    # Add the new cases argument
    parser.add_argument("--cases", type=str,
                       help="Range of case IDs to process (format: start,end) e.g. '1,10'")
    

    phenogpt_group = parser.add_argument_group('PhenoGPT Configuration')
    phenogpt_group.add_argument("--phenogpt_base_model", type=str,
                       default="meta-llama/Llama-2-7b-chat-hf",
                       help="Base model for PhenoGPT (default: meta-llama/Llama-2-7b-chat-hf)")
    phenogpt_group.add_argument("--phenogpt_lora_weights", type=str,
                       default="/home/johnwu3/projects/rare_disease/workspace/repos/PhenoGPT/model/llama2/llama2_lora_weights",
                       help="LoRA weights path for PhenoGPT")
    phenogpt_group.add_argument("--phenogpt_load_8bit", action="store_true",
                       help="Load PhenoGPT model in 8-bit precision")
    phenogpt_group.add_argument("--phenogpt_custom_prompt", type=str,
                       help="Custom system prompt for PhenoGPT")
    phenogpt_group.add_argument("--hf_api_key", type=str,
                       help="HuggingFace API key for accessing gated models")
    
    # System prompt configuration
    parser.add_argument("--system_prompt_file", default="data/prompts/system_prompts.json", 
                       help="File containing system prompts (default: system_prompts.json)")
    
    # Entity extractor configuration
    parser.add_argument("--entity_extractor", type=str, 
                       choices=["iterative", "simple", "retrieval", "multi", "phenogpt"],
                       default="iterative", 
                       help="Entity extraction method (default: iterative)")
    parser.add_argument("--max_iterations", type=int, default=3,
                       help="Maximum iterations for iterative extractor (default: 3)")
    
    # Retrieval-enhanced extractor configuration
    parser.add_argument("--embeddings_file", type=str,
                       help="Path to embeddings file (G2GHPO_metadata.npy) for retrieval-enhanced extraction")
    parser.add_argument("--retriever", type=str,
                       choices=["fastembed", "sentence_transformer", "medcpt"],
                       default="fastembed",
                       help="Type of retriever/embedding model to use (default: fastembed)")
    parser.add_argument("--retriever_model", type=str,
                       default="BAAI/bge-small-en-v1.5",
                       help="Model name for retriever (default: BAAI/bge-small-en-v1.5)")
    parser.add_argument("--top_k", type=int, default=10,
                       help="Number of top candidates to retrieve per sentence (default: 10)")
    
    # Context extractor configuration
    parser.add_argument("--window_size", type=int, default=0,
                       help="Context window size for sentences (default: 0)")
    
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
    
    # Add these to the existing argument parser, for multi-pass self consistency attempts
    parser.add_argument("--multi_temp", action="store_true",
                    help="Use multi-temperature extraction approach")
    parser.add_argument("--temperatures", type=str, default="0.01,0.1,0.3,0.7,0.9",
                    help="Comma-separated list of temperatures for multi-temp extraction")
    parser.add_argument("--aggregation_type", type=str, 
                    choices=["union", "intersection", "hybrid"],
                    default="hybrid",
                    help="Method to aggregate results from multiple temperature runs")
    parser.add_argument("--hybrid_threshold", type=int, default=2,
                    help="Minimum number of runs an entity must appear in for hybrid aggregation")
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


# 3. Modify the load_input_data function to filter cases
def load_input_data(input_file: str, start_id: int = None, end_id: int = None) -> Dict[int, Dict[str, Any]]:
    """Load and validate the input JSON file, optionally filtering by case ID range."""
    try:
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        # Validate structure
        if not isinstance(data, dict):
            raise ValueError(f"Input file {input_file} must contain a JSON object")
        
        # Convert string keys to integers if necessary
        if data and all(isinstance(k, str) for k in data.keys()):
            data = {int(k): v for k, v in data.items()}
        
        # Filter by case ID range if specified
        if start_id is not None and end_id is not None:
            filtered_data = {}
            for case_id, case_data in data.items():
                # Convert string case_id to int if necessary
                case_id_int = int(case_id) if isinstance(case_id, str) else case_id
                
                if start_id <= case_id_int <= end_id:
                    filtered_data[case_id] = case_data
            
            timestamp_print(f"Filtered data to {len(filtered_data)} cases (IDs {start_id}-{end_id})")
            data = filtered_data
        
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


def process_cases(cases: Dict[int, Dict[str, Any]], args: argparse.Namespace, 
                 entity_extractor, context_extractor, existing_results: Dict = None) -> Dict[int, Dict[str, Any]]:
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
            
            # Extract entities
            entities = entity_extractor.extract_entities(clinical_text)
            
            if args.debug:
                timestamp_print(f"  Extracted {len(entities)} entities")
            
            # Find contexts for entities
            entity_contexts = context_extractor.extract_contexts(entities, clinical_text, window_size=args.window_size)
            
            # Store results
            results[case_id] = {
                "clinical_text": clinical_text,
                "entities_with_contexts": entity_contexts
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
                "error": str(e)
            }
    
    return results


def main():
    """Main function to run the entity and context extraction pipeline."""
    try:
        # Parse command line arguments
        # Parse command line arguments
        args = parse_arguments()
        
        timestamp_print(f"Starting entity extraction process")
        
        # Parse case range if provided
        start_id, end_id = None, None
        if args.cases:
            start_id, end_id = parse_case_range(args.cases)
            timestamp_print(f"Processing case ID range: {start_id} to {end_id}")
        
        # Setup device
        devices = setup_device(args)
        timestamp_print(f"Using device for LLM: {devices['llm']}")
        if args.entity_extractor == "retrieval":
            timestamp_print(f"Using device for retriever: {devices['retriever']}")
        
        # Load system prompts
        timestamp_print(f"Loading system prompts from {args.system_prompt_file}")
        try:
            with open(args.system_prompt_file, 'r') as f:
                prompts = json.load(f)
                system_message_extraction = prompts.get("system_message_I", "")
        except Exception as e:
            timestamp_print(f"Error loading system prompts: {e}")
            raise
        
        
        # Initialize entity extractor
        timestamp_print(f"Initializing entity extractor ({args.entity_extractor})")
        if args.entity_extractor == "phenogpt":
            # Initialize PhenoGPT
            timestamp_print(f"Initializing PhenoGPT with base model {args.phenogpt_base_model}")
            try:
                phenogpt = PhenoGPT(
                    base_model_path=args.phenogpt_base_model,
                    lora_weights_path=args.phenogpt_lora_weights,
                    load_8bit=args.phenogpt_load_8bit,
                    device=devices['llm'],  # Use the device configured for LLM
                    hf_api_key=args.hf_api_key
                )
                
                # Create the entity extractor using PhenoGPT
                entity_extractor = PhenoGPTEntityExtractor(
                    phenogpt_instance=phenogpt,
                    custom_prompt=args.phenogpt_custom_prompt
                )
                
                timestamp_print("PhenoGPT initialized successfully")
            except Exception as e:
                timestamp_print(f"Error initializing PhenoGPT: {e}")
                traceback.print_exc()
                exit(1)
             # Initialize LLM client
        else:
            timestamp_print(f"Initializing {args.llm_type} LLM client")
            llm_client = initialize_llm_client(args, devices['llm'])
            if args.entity_extractor == "retrieval":
                # Validate embeddings file is provided
                if not args.embeddings_file:
                    raise ValueError("--embeddings_file is required when using retrieval-enhanced entity extraction")
                
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
                
                # Initialize retrieval-enhanced entity extractor
                entity_extractor = RetrievalEnhancedEntityExtractor(
                    llm_client=llm_client,
                    embedding_manager=embedding_manager,
                    embedded_documents=embedded_documents,
                    system_message=system_message_extraction,
                    top_k=args.top_k
                )
                
            elif args.multi_temp or args.entity_extractor == "multi":
                # Parse temperatures from string
                temperatures = [float(t) for t in args.temperatures.split(',')]
                entity_extractor = MultiIterativeExtractor(
                    llm_client,
                    system_message_extraction, 
                    temperatures=temperatures,
                    max_iterations=args.max_iterations,
                    aggregation_type=args.aggregation_type,
                    hybrid_threshold=args.hybrid_threshold
                )
            elif args.entity_extractor == "iterative":
                entity_extractor = IterativeLLMEntityExtractor(
                    llm_client, 
                    system_message_extraction, 
                    max_iterations=args.max_iterations
                )
            else:  # simple
                entity_extractor = LLMEntityExtractor(llm_client, system_message_extraction)
        
        # Initialize context extractor
        timestamp_print(f"Initializing context extractor (window_size={args.window_size})")
        context_extractor = ContextExtractor(debug=args.debug)
        
        # Load input data
        timestamp_print(f"Loading clinical notes from {args.input_file}")
        cases = load_input_data(args.input_file, start_id, end_id)
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