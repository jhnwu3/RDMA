#!/usr/bin/env python3
import argparse
import torch
from rdma.rdrag.vectorize import create_rare_disease_vectorizer

def parse_args():
    parser = argparse.ArgumentParser(description='Create vector database from rare disease data')
    
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
    
    parser.add_argument('--model_name',
                       type=str,
                       default='BAAI/bge-small-en-v1.5',
                       help='Name of the embedding model')
    
    parser.add_argument('--ontology_file',
                       type=str,
                       default='data/ontology/rare_disease_ontology.jsonl',
                       help='Path to input JSONL file containing rare disease ontology')
    
    parser.add_argument('--triples_file',
                       type=str,
                       default='data/ontology/RareDisease_Phenotype_Triples.json',
                       help='Path to input JSON file containing rare disease-phenotype triples')
    
    parser.add_argument('--output_file',
                       type=str,
                       default='data/vector_stores/rare_disease_embeddings.npy',
                       help='Path for output NPY file containing embeddings')
    
    parser.add_argument('--csv_output',
                       type=str,
                       default='rare_disease_db.csv',
                       help='Path for output CSV file for inspection')
    
    parser.add_argument('--batch_size',
                       type=int,
                       default=100,
                       help='Batch size for processing')

    args = parser.parse_args()
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
    print(f"Using device: {device}")
    
    # Create and run vectorizer
    vectorizer = create_rare_disease_vectorizer(
        model_type=args.model_type,
        model_name=args.model_name,
        ontology_path=args.ontology_file,
        triples_path=args.triples_file,
        batch_size=args.batch_size,
        output_file=args.output_file,
        csv_output_file=args.csv_output
    )
    
    # Run vectorization
    vectorizer.vectorize()

if __name__ == "__main__":
    main()