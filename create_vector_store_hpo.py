#!/usr/bin/env python3
import argparse
import torch
from rdma.hporag.vectorize import create_vectorizer, VectorizationConfig

def parse_args():
    parser = argparse.ArgumentParser(description='Create vector database from HPO data')
    
    # Required arguments
    parser.add_argument('--model_type', 
                   type=str, 
                   choices=['fastembed', 'medcpt', 'sentence_transformer'],
                   required=True,
                   help='Type of embedding model to use')
    # GPU configuration
    gpu_group = parser.add_mutually_exclusive_group()
    gpu_group.add_argument('--gpu_id', 
                          type=int,
                          help='Specific GPU ID to use (e.g., 0, 1, 2)')
    gpu_group.add_argument('--cpu', 
                          action='store_true',
                          help='Force CPU usage even if GPU is available')
    
    # Optional arguments
    parser.add_argument('--model_name',
                       type=str,
                       help='Name of the model (required for fastembed, e.g., "BAAI/bge-small-en-v1.5")')
    
    parser.add_argument('--json_file',
                       type=str,
                       default='data/ontology/hpo_data_with_lineage.json',
                       help='Path to input JSON file containing HPO data')
    
    parser.add_argument('--csv_file',
                       type=str,
                       default='data/ontology/HPO_addons.csv',
                       help='Path to input CSV file containing additional HPO data')
    
    parser.add_argument('--output_file',
                       type=str,
                       default='data/vector_stores/G2GHPO_metadata.npy',
                       help='Path for output NPY file containing embeddings')
    
    parser.add_argument('--csv_output',
                       type=str,
                       default='HP_DB.csv',
                       help='Path for output CSV file for inspection')
    
    parser.add_argument('--batch_size',
                       type=int,
                       default=100,
                       help='Batch size for processing')

    args = parser.parse_args()
    
    # Validate model_name is provided if using fastembed
    if args.model_type == 'fastembed' and not args.model_name:
        parser.error("--model_name is required when using fastembed")
    
    return args

def setup_device(args):
    """Configure the device based on command line arguments."""
    if args.cpu:
        return "cpu"
    elif args.gpu_id is not None:
        return f"cuda:{args.gpu_id}"
    else:
        return "cuda" if torch.cuda.is_available() else "cpu"

def main():
    args = parse_args()
    
    # Setup device
    device = setup_device(args)
    
    # Create configuration
    config = VectorizationConfig(
        model_type=args.model_type,
        model_name=args.model_name,
        batch_size=args.batch_size,
        json_file_path=args.json_file,
        csv_file_path=args.csv_file,
        output_file=args.output_file,
        csv_output_file=args.csv_output,
        device=device  # Add device to config
    )
    
    # Create and run vectorizer
    vectorizer = create_vectorizer(
        model_type=args.model_type,
        model_name=args.model_name,
        config=config
    )
    
    # Run vectorization
    vectorizer.vectorize()

if __name__ == "__main__":
    main()