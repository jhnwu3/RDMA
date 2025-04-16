#!/usr/bin/env python3
import argparse
import os
import sys
import pandas as pd
import numpy as np
import traceback
import json
import torch
import stanza
from datetime import datetime
from typing import Dict, Any, Union
from hporag.config import Config
from hporag.entity import LLMEntityExtractor, StanzaEntityExtractor, IterativeLLMEntityExtractor
from hporag.hpo_match import RAGHPOMatcher, OptimizedRAGHPOMatcher
from hporag.pipeline import HPORAG
from utils.embedding import EmbeddingsManager
from utils.llm_client import APILLMClient, LocalLLMClient, VLLMClient

def setup_stanza_pipeline(device='cpu'):
    """Initialize and return a Stanza pipeline with MIMIC package and i2b2 NER model."""
    try:
        # Download and initialize the MIMIC pipeline with the i2b2 NER model
        stanza.download('en', package='mimic', processors={'ner': 'i2b2'})
        return stanza.Pipeline('en', package='mimic', processors={'ner': 'i2b2'}, device=device)
    except Exception as e:
        print(f"Error initializing Stanza pipeline: {e}")
        sys.exit(1)

def validate_input_csv(filepath: str) -> pd.DataFrame:
    """Validate and prepare input CSV format.
    
    Args:
        filepath: Path to the input CSV file
        
    Returns:
        pd.DataFrame: Validated and prepared DataFrame with clinical notes
        
    Raises:
        ValueError: If required columns are missing
        SystemExit: If file reading fails
    """
    try:
        df = pd.read_csv(filepath)
        if 'clinical_note' not in df.columns:
            raise ValueError('Input CSV must have a "clinical_note" column')
        
        # Handle patient IDs
        if 'patient_id' not in df.columns and 'case number' not in df.columns:
            df['patient_id'] = range(1, len(df) + 1)
        elif 'case number' in df.columns:
            df['patient_id'] = df['case number']
            df = df.drop('case number', axis=1)
            
        # Clean notes
        df['clinical_note'] = df['clinical_note'].astype(str)
        df = df.dropna(subset=['clinical_note'])
        
        # Print validation summary
        print(f"Validated CSV file:")
        print(f"- Total rows: {len(df)}")
        print(f"- Columns: {', '.join(df.columns)}")
        print(f"- Patient ID range: {df['patient_id'].min()} to {df['patient_id'].max()}")
        
        return df
        
    except Exception as e:
        print(f"Error reading input CSV: {e}")
        sys.exit(1)

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='RAG-HPO: Extract HPO terms from clinical notes using modular pipeline')
    
    # Previous required arguments remain the same...
    
    # GPU configuration for LLM
    gpu_group = parser.add_mutually_exclusive_group()
    gpu_group.add_argument('--gpu_id', type=int,
                          help='Specific GPU ID to use for LLM (e.g., 0, 1, 2)')
    gpu_group.add_argument('--multi_gpu', action='store_true',
                          help='Use multiple GPUs with automatic device mapping')
    gpu_group.add_argument('--gpu_ids', type=str,
                          help='Comma-separated list of GPU IDs to use (e.g., "0,1,2")')
    gpu_group.add_argument('--condor', action='store_true',
                          help='Use generic CUDA device without specific GPU ID')
    
    # Model configuration
    parser.add_argument('--model', type=str,
                       choices=['llama3_8b', 'llama3_70b', 'llama3_70b_2b', 'llama3_70b_full', 'llama3-groq-70b', 'llama3-groq-8b', 'openbio_70b', 'openbio_8b', 'llama3_70b_groq', 'llama3_70b_r1', 'qwen_70b', 'mixtral_70b', 'mistral_24b'],
                       default='llama3_70b',
                       help='Model to use for inference (default: llama3_70b 4-bit)')
    
    # Required arguments
    parser.add_argument('--input_file', required=True,
                       help='Path to input CSV file containing clinical notes. Must have a "clinical_note" column.')
    parser.add_argument('--output_file', required=True,
                       help='Path for output CSV file with extracted HPO terms')
    
    # LLM client configuration
    llm_group = parser.add_argument_group('LLM Configuration')
    llm_group.add_argument('--llm_type', type=str,
                          choices=['local', 'api'],
                          default='local',
                          help='Type of LLM client to use (default: local)')
    
    # API-specific arguments
    api_group = parser.add_argument_group('API Client Configuration')
    api_group.add_argument('--api_key', type=str,
                          help='API key for remote LLM service')
    api_group.add_argument('--api_base_url', type=str,
                          default="https://api.groq.com/openai/v1/chat/completions",
                          help='Base URL for API endpoint')
    api_group.add_argument('--api_config', type=str,
                          help='Path to API configuration file')
    
    # Entity extractor configuration
    parser.add_argument('--entity_extractor', type=str,
                       choices=['llm', 'stanza', 'iterative'],
                       default='llm',
                       help='Entity extraction method to use (default: llm)')
    
    # GPU configuration for LLM
    gpu_group = parser.add_mutually_exclusive_group()
    
    # GPU configuration for retriever
    retriever_group = parser.add_mutually_exclusive_group()
    retriever_group.add_argument('--retriever_gpu_id', type=int,
                                help='Specific GPU ID to use for retriever/embeddings')
    retriever_group.add_argument('--retriever_cpu', action='store_true',
                                help='Force CPU usage for retriever even if GPU is available')
    
    # GPU configuration for Stanza
    stanza_group = parser.add_mutually_exclusive_group()
    stanza_group.add_argument('--stanza_gpu_id', type=int,
                             help='Specific GPU ID to use for Stanza pipeline')
    stanza_group.add_argument('--stanza_cpu', action='store_true',
                             help='Force CPU usage for Stanza even if GPU is available')
                             
    # Model configuration
    parser.add_argument('--retriever', type=str,
                       choices=['fastembed', 'medcpt', 'sentence_transformer'],
                       default='fastembed',
                       help='Type of retriever to use (default: fastembed)')
    parser.add_argument('--retriever_model', type=str,
                       default="BAAI/bge-small-en-v1.5",
                       help='Model name for fastembed retriever')
    parser.add_argument('--cache_dir', type=str,
                       default="/shared/rsaas/jw3/rare_disease/model_cache",
                       help='Directory for caching models')
    parser.add_argument('--embeddings_file', type=str,
                       default='data/vector_stores/G2GHPO_metadata_test.npy',
                       help='Path to embeddings file')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size for processing')
    parser.add_argument('--temperature', type=float, default=0.1,
                       help='Temperature for LLM inference')
    parser.add_argument('--no_cleanup', action='store_true',
                       help='Do not remove temporary files after completion')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode for detailed processing information')
    
    return parser.parse_args()

def setup_device(args: argparse.Namespace) -> tuple[str, str, str]:
    """Configure the devices based on command line arguments.
    
    Returns:
        tuple[str, str, str]: Tuple containing (llm_device, retriever_device, stanza_device)
    """
    # Setup LLM device
    if args.multi_gpu:
        llm_device = "auto"
    elif args.gpu_ids:
        # Convert comma-separated string to list of cuda devices
        gpu_list = [f"cuda:{int(gpu_id.strip())}" for gpu_id in args.gpu_ids.split(',')]
        llm_device = gpu_list
    elif args.condor:
        llm_device = "cuda"
    elif args.gpu_id is not None:
        llm_device = f"cuda:{args.gpu_id}"
    else:
        llm_device = "cpu"

    # Setup retriever device
    if args.retriever_gpu_id is not None:
        retriever_device = f"cuda:{args.retriever_gpu_id}"
    elif args.retriever_cpu:
        retriever_device = "cpu"
    else:
        # If using multi_gpu or gpu_ids for LLM, use the first GPU for retriever
        if isinstance(llm_device, list):
            retriever_device = llm_device[0]
        elif llm_device == "auto":
            # When LLM is using auto, default retriever to first available GPU
            retriever_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            retriever_device = llm_device
        
    # Setup Stanza device
    if args.stanza_gpu_id is not None:
        stanza_device = f"cuda:{args.stanza_gpu_id}"
    elif args.stanza_cpu:
        stanza_device = "cpu"
    else:
        # Similar to retriever, if using multi_gpu or gpu_ids, use the first GPU
        if isinstance(llm_device, list):
            stanza_device = llm_device[0]
        elif llm_device == "auto":
            # When LLM is using auto, default stanza to first available GPU
            stanza_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            stanza_device = llm_device

    # Verify GPU availability
    if any(isinstance(device, str) and 'cuda' in device for device in [llm_device, retriever_device, stanza_device]):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but no GPU available")
        
        specified_gpus = set()
        # Handle the llm_device which might be a list or "auto"
        if isinstance(llm_device, list):
            for device in llm_device:
                if 'cuda:' in device:
                    specified_gpus.add(int(device.split(':')[1]))
        elif isinstance(llm_device, str) and 'cuda:' in llm_device:
            specified_gpus.add(int(llm_device.split(':')[1]))
            
        # Add retriever and stanza devices
        for device in [retriever_device, stanza_device]:
            if isinstance(device, str) and 'cuda:' in device:
                specified_gpus.add(int(device.split(':')[1]))
                
        # Verify all specified GPUs exist
        for gpu_id in specified_gpus:
            if gpu_id >= torch.cuda.device_count():
                raise ValueError(f"GPU {gpu_id} not found. Available GPUs: {torch.cuda.device_count()}")

    # Print device configuration for debugging
    print(f"Device configuration:")
    print(f"  LLM device: {llm_device}")
    print(f"  Retriever device: {retriever_device}")
    print(f"  Stanza device: {stanza_device}")

    return llm_device, retriever_device, stanza_device

def initialize_llm_client(args: argparse.Namespace, device: str) -> Union[LocalLLMClient, APILLMClient, VLLMClient]:
    """Initialize appropriate LLM client based on arguments."""
    if args.llm_type == 'api':
        from utils.llm_client import APILLMClient
        
        if args.api_config:
            return APILLMClient.from_config(args.api_config)
        elif args.api_key:
            return APILLMClient(
                api_key=args.api_key,
                model_type=args.model,
                device=device,  # kept for interface compatibility
                cache_dir=args.cache_dir,
                temperature=args.temperature
            )
        else:
            return APILLMClient.initialize_from_input()
            
    elif args.llm_type == 'vllm':
        try:
            from utils.llm_client import VLLMClient
            
            # Convert quantization argument
            quantization = None if args.quantization == 'none' else args.quantization
            
            # Determine tensor parallel size from GPU configuration
            tensor_parallel_size = args.tensor_parallel_size
            if args.multi_gpu:
                tensor_parallel_size = torch.cuda.device_count()
            elif args.gpu_ids:
                tensor_parallel_size = len(args.gpu_ids.split(','))
            
            # Set devices based on configuration
            if args.multi_gpu:
                device = "auto"
            elif args.gpu_ids:
                device = [f"cuda:{gpu_id.strip()}" for gpu_id in args.gpu_ids.split(',')]
            
            return VLLMClient(
                model_type=args.model,
                device=device,
                cache_dir=args.cache_dir,
                temperature=args.temperature,
                tensor_parallel_size=tensor_parallel_size,
                max_model_len=args.max_model_len,
                quantization=quantization
            )
        except ImportError as e:
            print(f"VLLM not installed. Error: {e}. Falling back to HuggingFace.")
            args.llm_type = 'local'  # Fall back to local
    
    # Default: local
    from utils.llm_client import LocalLLMClient
    
    return LocalLLMClient(
        model_type=args.model,
        device=device,
        cache_dir=args.cache_dir,
        temperature=args.temperature
    )

def main_rag_pipeline():
    """Main function using the HPORAG pipeline."""
    try:
        # Parse command line arguments
        args = parse_arguments()

        # Initialize Config and load prompts
        config = Config()
        
        # Setup devices
        llm_device, retriever_device, stanza_device = setup_device(args)
        
        # Load prompts
        with open('data/prompts/system_prompts.json', 'r') as f:
            prompts = json.load(f)
        
        # Initialize LLM client
        config.timestamped_print(f"Initializing {args.llm_type} LLM client...")
        llm_client = initialize_llm_client(args, llm_device)
        
        # Initialize components based on entity extractor choice
        if args.entity_extractor == 'stanza':
            # Initialize Stanza pipeline
            config.timestamped_print("Initializing Stanza pipeline...")
            stanza_pipeline = setup_stanza_pipeline(device=stanza_device)
            entity_extractor = StanzaEntityExtractor(stanza_pipeline)
        elif args.entity_extractor == "iterative":
            entity_extractor = IterativeLLMEntityExtractor(
                llm_client=llm_client,
                system_message=prompts['system_message_I'],
                max_iterations=3
            )
        else:  # 'llm'
            entity_extractor = LLMEntityExtractor(llm_client, prompts['system_message_I'])
        
        # Initialize embeddings manager
        embeddings_manager = EmbeddingsManager(
            model_type=args.retriever,
            model_name=args.retriever_model if args.retriever in ['fastembed', 'sentence_transformer'] else None,
            device=retriever_device
        )
        
        # Initialize HPO matcher (always uses LLM)
        hpo_matcher = RAGHPOMatcher(embeddings_manager, llm_client, prompts['system_message_II'])
        # hpo_matcher = OptimizedRAGHPOMatcher(
        #     embeddings_manager=embeddings_manager,
        #     llm_client=llm_client,
        #     system_message=prompts['system_message_II'],
        #     debug=args.debug,                 # Enable timing information
        #     fuzzy_threshold=80,               # Threshold for fuzzy matching (0-100)
        #     early_term_threshold=90,          # Early termination threshold (0-100)
        #     max_candidates=20,                # Max candidates to process (default: 20)
        #     faiss_k=50                       # FAISS k value (reduced from 800)
        # )
                
        # Create pipeline
        pipeline = HPORAG(entity_extractor, hpo_matcher, args.debug)

        # Process clinical notes
        config.timestamped_print("Validating input CSV...")
        notes_df = validate_input_csv(args.input_file)
        config.timestamped_print(f"Found {len(notes_df)} valid clinical notes to process")
        
        # Load embeddings
        config.timestamped_print("Loading embeddings...")
        embedded_documents = embeddings_manager.load_documents(args.embeddings_file)
        
        # Process through pipeline
        config.timestamped_print("Processing clinical notes through pipeline...")
        results_df = pipeline.process_dataframe(
            notes_df,
            embedded_documents,
            batch_size=args.batch_size
        )
        
        # Save results
        pipeline.save_results(results_df, args.output_file)
        config.timestamped_print(f"Final results saved to: {args.output_file}")
        
        # Cleanup
        if not args.no_cleanup:
            for temp_file in Config.TEMP_FILES:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    config.timestamped_print(f"Removed temporary file: {temp_file}")
        
        config.timestamped_print("Processing completed successfully")

    except Exception as e:
        config = Config()
        config.timestamped_print(f"An error occurred: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main_rag_pipeline()