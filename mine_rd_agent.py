#!/usr/bin/env python3
import argparse
import os
import sys
import pandas as pd
import numpy as np
import traceback
import json
import torch
from datetime import datetime
from typing import Dict, Any, List, Optional, Union, Tuple
from rdrag.pipeline import RDPipeline
from rdrag.entity import LLMRDExtractor
from rdrag.rd_match import RAGRDMatcher
from utils.embedding import EmbeddingsManager
from utils.llm_client import APILLMClient, LocalLLMClient
from utils.data import NumpyJSONEncoder, process_mimic_json, evaluate_predictions
from rdrag.supervisor import IterativeSupervisor, SupervisedEvaluator

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Extract rare disease mentions from clinical notes')
    
    # Add evaluate mode
    parser.add_argument('--evaluate', action='store_true',
                      help='Run only evaluation on existing intermediate results')
    
    # Add supervised evaluation mode
    parser.add_argument('--supervised', action='store_true',
                      help='Run supervised evaluation on false positives')
    parser.add_argument('--supervised_output', type=str,
                      help='Path for supervised evaluation results (JSON)')
    
    # Add iterative supervision mode
    parser.add_argument('--iterative', action='store_true',
                      help='Run iterative supervised enhancement of gold standard annotations')
    parser.add_argument('--max_iterations', type=int, default=3,
                      help='Maximum number of iterations for iterative mode')
    parser.add_argument('--iterative_output', type=str,
                      help='Path for enhanced gold standard output (JSON)')

    # Model configuration (only needed for non-evaluate mode or for supervised/iterative modes)
    model_group = parser.add_argument_group('Model Configuration')
    model_group.add_argument('--llm_type', type=str,
                      choices=['local', 'api'],
                      default='local',
                      help='Type of LLM client to use')
    
    model_group.add_argument('--model', type=str,
                      choices=['llama3_8b', 'llama3_70b', 'llama3_70b_2b', 'llama3_70b_full', 
                              'llama3-groq-70b', 'llama3-groq-8b', 'openbio_70b', 'openbio_8b', 'llama3_70b_groq', 'mixtral_70b', 'mistral_24b'],
                      default='llama3_70b',
                      help='Model to use for inference (default: llama3_70b 4-bit)')
    
    # GPU configuration
    gpu_group = model_group.add_mutually_exclusive_group()
    gpu_group.add_argument('--gpu_id', type=int,
                          help='Specific GPU ID to use')
    gpu_group.add_argument('--multi_gpu', action='store_true',
                          help='Use multiple GPUs with automatic device mapping')
    gpu_group.add_argument('--gpu_ids', type=str,
                          help='Comma-separated list of GPU IDs (e.g., "0,1,2")')
    gpu_group.add_argument('--cpu', action='store_true',
                          help='Force CPU usage')
    gpu_group.add_argument('--condor', action='store_true',
                          help='Use generic CUDA device without specific GPU ID')
    
    # Retriever configuration
    retriever_group = parser.add_argument_group('Retriever Configuration')
    retriever_group.add_argument('--retriever', type=str,
                      choices=['fastembed', 'medcpt', 'sentence_transformer'],
                      default='fastembed',
                      help='Type of retriever to use')
    retriever_group.add_argument('--retriever_model', type=str,
                      default="BAAI/bge-small-en-v1.5",
                      help='Model name for retriever')
    
    # API configuration
    api_group = parser.add_argument_group('API Configuration')
    api_group.add_argument('--api_key', type=str,
                      help='API key for remote LLM service')
    api_group.add_argument('--api_config', type=str,
                      help='Path to API configuration file')
    
    # Input/Output configuration
    io_group = parser.add_argument_group('Input/Output Configuration')
    io_group.add_argument('--input_file',
                      help='Path to input JSON file with clinical notes and annotations')
    io_group.add_argument('--output_file',
                      help='Path for final output CSV file')
    io_group.add_argument('--intermediate_file', type=str,
                      help='Path to intermediate results file (required for --evaluate, optional otherwise)')
    io_group.add_argument('--embedded_documents',
                      help='Path to pre-embedded rare disease documents')
    io_group.add_argument('--evaluation_file', type=str,
                      help='Optional path for detailed evaluation results')
    
    # Processing configuration
    proc_group = parser.add_argument_group('Processing Configuration')
    proc_group.add_argument('--batch_size', type=int, default=1,
                      help='Batch size for processing')
    proc_group.add_argument('--temperature', type=float, default=0.1,
                      help='Temperature for LLM inference')
    proc_group.add_argument('--cache_dir', type=str,
                      default="/shared/rsaas/jw3/rare_disease/model_cache",
                      help='Directory for caching models')
    proc_group.add_argument('--debug', action='store_true',
                      help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Validate arguments based on mode
    if args.evaluate:
        if not args.intermediate_file:
            parser.error("--intermediate_file is required when using --evaluate")
        if not args.input_file:
            parser.error("--input_file is required for evaluation (contains gold standard)")
    else:
        required_args = ['input_file', 'output_file', 'embedded_documents']
        missing_args = [arg for arg in required_args if not getattr(args, arg)]
        if missing_args:
            parser.error(f"The following arguments are required: {', '.join(missing_args)}")
    
    # Supervised and iterative modes require embedded_documents
    if (args.supervised or args.iterative) and not args.embedded_documents:
        parser.error("--supervised and --iterative require --embedded_documents for candidate retrieval")
    
    # Iterative mode can be used with or without evaluate mode
    # If used with evaluate mode, it enhances gold standard based on intermediate results
    # If used without evaluate mode, it runs pipeline first, then enhances gold standard
    
    return args

def setup_device(args: argparse.Namespace) -> Tuple[str, str]:
    """Configure devices for LLM and retriever."""
    # Setup LLM device
    if args.multi_gpu:
        llm_device = "auto"
    elif args.gpu_ids:
        gpu_list = [f"cuda:{int(gpu_id.strip())}" for gpu_id in args.gpu_ids.split(',')]
        llm_device = gpu_list
    elif args.cpu:
        llm_device = "cpu"
    elif args.gpu_id is not None:
        llm_device = f"cuda:{args.gpu_id}"
    elif args.condor:
        llm_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        llm_device = "cuda" if torch.cuda.is_available() else "cpu"

    # Setup retriever device (uses first GPU if multiple available)
    if isinstance(llm_device, list):
        retriever_device = llm_device[0]
    elif llm_device == "auto":
        retriever_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        retriever_device = llm_device

    # Verify GPU availability
    if any('cuda' in str(device) for device in [llm_device, retriever_device]):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but no GPU available")

    print(f"Device configuration:")
    print(f"  LLM device: {llm_device}")
    print(f"  Retriever device: {retriever_device}")

    return llm_device, retriever_device

def initialize_llm_client(args: argparse.Namespace, device: str) -> Union[LocalLLMClient, APILLMClient]:
    """Initialize appropriate LLM client."""
    if args.llm_type == 'api':
        if args.api_config:
            return APILLMClient.from_config(args.api_config)
        elif args.api_key:
            return APILLMClient(
                api_key=args.api_key,
                model_type=args.model,
                device=device,
                cache_dir=args.cache_dir,
                temperature=args.temperature
            )
        else:
            return APILLMClient.initialize_from_input()
    else:
        return LocalLLMClient(
            model_type=args.model,
            device=device,
            cache_dir=args.cache_dir,
            temperature=args.temperature
        )

def load_system_prompts() -> Dict[str, str]:
    """Load system prompts from configuration."""
    prompts = {
        'extraction': """You are a rare disease expert. Follow the instructions below.""",
        
        'matching': """You are a rare disease expert. Follow the instructions below."""
    }
    return prompts


def main():
    """Main function for processing MIMIC-style JSON dataset."""
    start_time = datetime.now()
    
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Process evaluation-only mode
        if args.evaluate and not args.iterative:
            print("\nRunning in evaluation-only mode")
            print(f"Loading intermediate results from: {args.intermediate_file}")
            
            try:
                results_df = pd.read_csv(args.intermediate_file)
                print(f"Loaded results for {len(results_df)} documents")
            except Exception as e:
                print(f"Error loading intermediate results: {str(e)}")
                sys.exit(1)
                
            print(f"\nLoading gold standard from: {args.input_file}")
            dataset_df = process_mimic_json(args.input_file)
            
            if dataset_df.empty:
                raise ValueError("No valid data found in gold standard file")
            
            if 'gold_annotations' not in dataset_df.columns:
                raise ValueError("No gold annotations found in dataset")
                
            # Run evaluation
            eval_start = datetime.now()
            print("\nEvaluating results against gold standard...")
            
            try:
                metrics = evaluate_predictions(results_df, dataset_df)
                print("\nEvaluation Metrics:")
                print(f"Precision: {metrics['precision']:.3f}")
                print(f"Recall: {metrics['recall']:.3f}")
                print(f"F1 Score: {metrics['f1']:.3f}")
                print(f"True Positives: {metrics['true_positives']}")
                print(f"False Positives: {metrics['false_positives']}")
                print(f"False Negatives: {metrics['false_negatives']}")
                
                if args.evaluation_file:
                    os.makedirs(os.path.dirname(args.evaluation_file), exist_ok=True)
                    with open(args.evaluation_file, 'w') as f:
                        json.dump(metrics, f, indent=2)
                        
                print(f"Evaluation completed in: {datetime.now() - eval_start}")
                
            except Exception as eval_error:
                print(f"Evaluation failed: {str(eval_error)}")
                sys.exit(1)
            
            # Run supervised evaluation if requested
            if args.supervised:
                run_supervised_evaluation(args, results_df, dataset_df)
                
        # Process full pipeline mode
        elif not args.evaluate:
            # Set up intermediate file path
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if args.intermediate_file:
                intermediate_file = args.intermediate_file
                os.makedirs(os.path.dirname(args.intermediate_file), exist_ok=True)
            else:
                output_dir = os.path.dirname(args.output_file) or '.'
                os.makedirs(output_dir, exist_ok=True)
                intermediate_file = os.path.join(
                    output_dir, 
                    f"rare_disease_results_{timestamp}_{args.retriever}_{args.model}.csv"
                )
            
            # Log configuration
            print(f"\nStarting processing at: {start_time}")
            print(f"Configuration:")
            print(f"  Model: {args.model}")
            print(f"  Retriever: {args.retriever} ({args.retriever_model})")
            print(f"  Batch size: {args.batch_size}")
            print(f"  Intermediate file: {intermediate_file}")
            
            # Initialize components
            llm_device, retriever_device = setup_device(args)
            prompts = load_system_prompts()
            
            # Initialize LLM
            llm_start = datetime.now()
            print(f"\nInitializing {args.llm_type} LLM client...")
            llm_client = initialize_llm_client(args, llm_device)
            print(f"LLM initialization completed in: {datetime.now() - llm_start}")
            
            # Initialize embeddings
            embed_start = datetime.now()
            print("\nInitializing embeddings manager...")
            embeddings_manager = EmbeddingsManager(
                model_type=args.retriever,
                model_name=args.retriever_model,
                device=retriever_device
            )
            print(f"Embeddings initialization completed in: {datetime.now() - embed_start}")
            
            # Load and process data
            process_start = datetime.now()
            print(f"\nProcessing input dataset: {args.input_file}")
            dataset_df = process_mimic_json(args.input_file)
            if dataset_df.empty:
                raise ValueError("No valid data found in input file")
            print(f"Dataset processing completed in: {datetime.now() - process_start}")
            
            # Initialize pipeline
            pipeline = RDPipeline(
                entity_extractor=LLMRDExtractor(
                    llm_client=llm_client,
                    system_message=prompts['extraction']
                ),
                rd_matcher=RAGRDMatcher(
                    embeddings_manager=embeddings_manager,
                    llm_client=llm_client,
                    system_message=prompts['matching']
                ),
                debug=args.debug
            )
            
            # Load embeddings
            embed_load_start = datetime.now()
            print("\nLoading embeddings...")
            embedded_documents = embeddings_manager.load_documents(args.embedded_documents)
            print(f"Embeddings loaded in: {datetime.now() - embed_load_start}")
            
            # Run pipeline
            pipeline_start = datetime.now()
            print("\nProcessing clinical notes...")
            results_df = pipeline.process_dataframe(
                df=dataset_df,
                embedded_documents=embedded_documents,
                text_column='clinical_note',
                id_column='document_id',
                batch_size=args.batch_size
            )
            print(f"Pipeline processing completed in: {datetime.now() - pipeline_start}")
            
            # Save intermediate results
            print(f"\nSaving intermediate results to: {intermediate_file}")
            results_df.to_csv(intermediate_file, index=False)
            
            # Run evaluation if gold standard exists
            if 'gold_annotations' in dataset_df.columns:
                eval_start = datetime.now()
                print("\nEvaluating results against gold standard...")
                try:
                    metrics = evaluate_predictions(results_df, dataset_df)
                    print("\nEvaluation Metrics:")
                    print(f"Precision: {metrics['precision']:.3f}")
                    print(f"Recall: {metrics['recall']:.3f}")
                    print(f"F1 Score: {metrics['f1']:.3f}")
                    
                    if args.evaluation_file:
                        os.makedirs(os.path.dirname(args.evaluation_file), exist_ok=True)
                        with open(args.evaluation_file, 'w') as f:
                            json.dump(metrics, f, indent=2)
                    
                    print(f"Evaluation completed in: {datetime.now() - eval_start}")
                except Exception as eval_error:
                    print(f"Evaluation failed: {str(eval_error)}")
                    print("Continuing to save final results...")
            
            # Run iterative supervision if requested
            if args.iterative:
                iterative_start = datetime.now()
                enhanced_df, final_results_df = run_iterative_supervision(
                    args, pipeline, dataset_df, results_df, embedded_documents
                )
                
                # Update results_df for final output
                results_df = final_results_df
                print(f"Iterative supervision completed in: {datetime.now() - iterative_start}")
            
            # Save final results
            print(f"\nSaving final results to: {args.output_file}")
            results_df.to_csv(args.output_file, index=False)
            
        # Run iterative supervision on existing results
        elif args.evaluate and args.iterative:
            print("\nRunning iterative supervision on existing results")
            
            # Load intermediate results
            print(f"Loading intermediate results from: {args.intermediate_file}")
            try:
                results_df = pd.read_csv(args.intermediate_file)
                print(f"Loaded results for {len(results_df)} documents")
            except Exception as e:
                print(f"Error loading intermediate results: {str(e)}")
                sys.exit(1)
                
            # Load gold standard
            print(f"\nLoading gold standard from: {args.input_file}")
            dataset_df = process_mimic_json(args.input_file)
            
            if dataset_df.empty:
                raise ValueError("No valid data found in gold standard file")
            
            if 'gold_annotations' not in dataset_df.columns:
                raise ValueError("No gold annotations found in dataset")
            
            # Initialize components for iterative supervision
            llm_device, retriever_device = setup_device(args)
            prompts = load_system_prompts()
            
            # Initialize LLM
            llm_start = datetime.now()
            print(f"\nInitializing {args.llm_type} LLM client...")
            llm_client = initialize_llm_client(args, llm_device)
            print(f"LLM initialization completed in: {datetime.now() - llm_start}")
            
            # Initialize embeddings
            embed_start = datetime.now()
            print("\nInitializing embeddings manager...")
            embeddings_manager = EmbeddingsManager(
                model_type=args.retriever,
                model_name=args.retriever_model,
                device=retriever_device
            )
            print(f"Embeddings initialization completed in: {datetime.now() - embed_start}")
            
            # Load embeddings
            embed_load_start = datetime.now()
            print("\nLoading embeddings...")
            embedded_documents = embeddings_manager.load_documents(args.embedded_documents)
            print(f"Embeddings loaded in: {datetime.now() - embed_load_start}")
            
            # Initialize pipeline
            pipeline = RDPipeline(
                entity_extractor=LLMRDExtractor(
                    llm_client=llm_client,
                    system_message=prompts['extraction']
                ),
                rd_matcher=RAGRDMatcher(
                    embeddings_manager=embeddings_manager,
                    llm_client=llm_client,
                    system_message=prompts['matching']
                ),
                debug=args.debug
            )
            
            # Run iterative supervision
            iterative_start = datetime.now()
            enhanced_df, final_results_df = run_iterative_supervision(
                args, pipeline, dataset_df, results_df, embedded_documents
            )
            
            # Save final results
            output_file = args.output_file if args.output_file else args.intermediate_file.replace('.csv', '_enhanced.csv')
            print(f"\nSaving final enhanced results to: {output_file}")
            final_results_df.to_csv(output_file, index=False)
            
            print(f"Iterative supervision completed in: {datetime.now() - iterative_start}")
        
        # Print final summary
        end_time = datetime.now()
        print("\nProcessing Summary:")
        print(f"Started at:  {start_time}")
        print(f"Finished at: {end_time}")
        print(f"Total time:  {end_time - start_time}")
        print("Processing completed successfully")
        
    except Exception as e:
        end_time = datetime.now()
        print(f"\nError occurred after running for: {end_time - start_time}")
        print(f"Error: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

def run_supervised_evaluation(args, results_df, dataset_df):
    """Run supervised evaluation to verify false positives."""
    print("\nRunning supervised evaluation on false positives...")
    
    # Initialize components for supervised evaluation
    llm_device, retriever_device = setup_device(args)
    prompts = load_system_prompts()
    
    # Initialize LLM
    llm_start = datetime.now()
    print(f"\nInitializing {args.llm_type} LLM client...")
    llm_client = initialize_llm_client(args, llm_device)
    print(f"LLM initialization completed in: {datetime.now() - llm_start}")
    
    # Initialize embeddings
    embed_start = datetime.now()
    print("\nInitializing embeddings manager...")
    embeddings_manager = EmbeddingsManager(
        model_type=args.retriever,
        model_name=args.retriever_model,
        device=retriever_device
    )
    print(f"Embeddings initialization completed in: {datetime.now() - embed_start}")
    
    # Load embeddings
    embed_load_start = datetime.now()
    print("\nLoading embedded documents...")
    embedded_documents = embeddings_manager.load_documents(args.embedded_documents)
    print(f"Embeddings loaded in: {datetime.now() - embed_load_start}")
    
    # Initialize matcher
    rd_matcher = RAGRDMatcher(
        embeddings_manager=embeddings_manager,
        llm_client=llm_client,
        system_message=prompts['matching']
    )
    
    # Create supervised evaluator
    supervised_output = args.supervised_output
    if not supervised_output and args.evaluation_file:
        # Default supervised output path based on evaluation file
        supervised_output = args.evaluation_file.replace('.json', '_supervised.json')
        if supervised_output == args.evaluation_file:  # If no .json extension
            supervised_output = args.evaluation_file + '_supervised'
    
    supervisor = SupervisedEvaluator(
        predictions_df=results_df,
        gold_df=dataset_df,
        rd_matcher=rd_matcher,
        embedded_documents=embedded_documents,
        output_file=supervised_output
    )
    
    # Run supervised evaluation
    supervised_start = datetime.now()
    supervised_metrics = supervisor.evaluate()
    
    # Print summary
    supervisor.print_summary()
    
    print(f"Supervised evaluation completed in: {datetime.now() - supervised_start}")

def run_iterative_supervision(args, pipeline, dataset_df, results_df, embedded_documents):
    """Run iterative supervision to enhance gold standard over multiple iterations."""
    print("\nInitializing iterative supervision...")
    
    # Define output file for enhanced gold standard
    iterative_output = args.iterative_output
    if not iterative_output:
        # Create default output path based on input file
        input_base = os.path.basename(args.input_file)
        input_dir = os.path.dirname(args.input_file)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        iterative_output = os.path.join(
            input_dir, 
            f"{os.path.splitext(input_base)[0]}_enhanced_{timestamp}.json"
        )
    
    # Verify that pipeline.rd_matcher is properly set up
    print("Verifying RAG matcher setup...")
    
    # Ensure the matcher has an LLM client
    if not hasattr(pipeline.rd_matcher, 'llm_client') or pipeline.rd_matcher.llm_client is None:
        print("WARNING: RAG matcher has no LLM client. False positive verification will fail!")
    else:
        print("RAG matcher has LLM client: ✓")
    
    # Ensure the matcher has an embeddings manager
    if not hasattr(pipeline.rd_matcher, 'embeddings_manager') or pipeline.rd_matcher.embeddings_manager is None:
        print("WARNING: RAG matcher has no embeddings manager. Candidate retrieval will fail!")
    else:
        print("RAG matcher has embeddings manager: ✓")
    
    # Create iterative supervisor with debug flag
    supervisor = IterativeSupervisor(
        gold_df=dataset_df,
        rd_matcher=pipeline.rd_matcher,
        embedded_documents=embedded_documents,
        max_iterations=args.max_iterations,
        output_file=iterative_output,
        debug=args.debug  # Pass debug flag from args
    )
    
    # Run iterations
    enhanced_df, final_results_df = supervisor.run_iterations(
        pipeline=pipeline,
        initial_predictions_df=results_df
    )
    
    # Print summary
    initial_count = sum(len(row.get('gold_annotations', [])) for _, row in dataset_df.iterrows())
    final_count = sum(len(row.get('gold_annotations', [])) for _, row in enhanced_df.iterrows())
    print("\n===== Iterative Supervision Summary =====")
    print(f"Initial gold standard annotations: {initial_count}")
    print(f"Final gold standard annotations: {final_count}")
    print(f"Total additions: {final_count - initial_count}")
    print(f"Enhanced gold standard saved to: {iterative_output}")
    
    return enhanced_df, final_results_df

def initialize_llm_client(args: argparse.Namespace, device: str) -> Union[LocalLLMClient, APILLMClient]:
    """Initialize appropriate LLM client with validation."""
    print("Initializing LLM client...")
    try:
        if args.llm_type == 'api':
            if args.api_config:
                client = APILLMClient.from_config(args.api_config)
                print(f"Initialized API LLM client from config: {args.api_config}")
            elif args.api_key:
                client = APILLMClient(
                    api_key=args.api_key,
                    model_type=args.model,
                    device=device,
                    cache_dir=args.cache_dir,
                    temperature=args.temperature
                )
                print(f"Initialized API LLM client with key and model: {args.model}")
            else:
                client = APILLMClient.initialize_from_input()
                print("Initialized API LLM client from user input")
        else:
            client = LocalLLMClient(
                model_type=args.model,
                device=device,
                cache_dir=args.cache_dir,
                temperature=args.temperature
            )
            print(f"Initialized Local LLM client with model: {args.model}")
        
        # Validate the client with a simple query
        system_msg = "You are a helpful assistant."
        test_response = client.query("Return only the word 'SUCCESS' if you can read this message.", system_msg)
        if "SUCCESS" in test_response:
            print("LLM client successfully validated!")
        else:
            print(f"WARNING: LLM client validation returned unexpected response: {test_response[:50]}...")
            
        return client
    except Exception as e:
        print(f"ERROR initializing LLM client: {str(e)}")
        raise
        
if __name__ == "__main__":
    main()