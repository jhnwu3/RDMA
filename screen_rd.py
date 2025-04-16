#!/usr/bin/env python3
import argparse
import json
import csv
import os
from pathlib import Path
from typing import Dict, List
import pandas as pd

from utils.llm_client import LocalLLMClient
from screening.data import DataLoader

def parse_args():
    parser = argparse.ArgumentParser(description='Analyze rare disease likelihood from phenotypes')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID to use')
    parser.add_argument('--model_type', type=str, default='llama3_8b', help='Type of model to use')
    parser.add_argument('--data_path', type=str, default='data/dataset/rd_phenotype_simulated_data.jsonl',
                        help='Path to the dataset')
    parser.add_argument('--hpo_json', type=str, default='data/ontology/hpo_data_with_lineage.json',
                        help='Path to HPO JSON file')
    parser.add_argument('--output_dir', type=str, default='data/results',
                        help='Directory to store results')
    return parser.parse_args()

def load_hpo_mapping(json_file: str) -> Dict[str, str]:
    """Load HPO ID to name mapping from JSON file."""
    with open(json_file, 'r') as f:
        hpo_data = json.load(f)
    
    # Create mapping from HPO ID to name (label)
    hpo_mapping = {}
    for hpo_id, details in hpo_data.items():
        if 'label' in details:
            # Remove underscore from HPO ID to match format in dataset
            hpo_id_clean = hpo_id.replace('_', ':')
            hpo_mapping[hpo_id_clean] = details['label']
        
    return hpo_mapping

def create_prompt(phenotypes: List[str], phenotype_names: List[str]) -> str:
    """Create a prompt for the LLM using phenotype information."""
    phenotype_text = "\n".join([f"- {name} (Code: {code})" for code, name in zip(phenotypes, phenotype_names)])
    
    prompt = f"""Given a patient with the following phenotypes:

{phenotype_text}

Based on these phenotypes, does this patient likely have a rare disease? Please answer with only 'yes' or 'no'.
"""
    return prompt

def main():
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = Path(args.output_dir) / 'rare_disease_analysis.csv'
    
    # Initialize model
    device = f"cuda:{args.gpu_id}"
    llm_client = LocalLLMClient(
        model_type=args.model_type,
        device=device,
        temperature=0.1  # Lower temperature for more consistent yes/no answers
    )
    
    # Load data
    data_loader = DataLoader(args.data_path)
    samples = data_loader.get_sample(100)  # Get first 100 samples
    
    # Load HPO mapping
    hpo_mapping = load_hpo_mapping(args.hpo_json)
    
    # System message for the LLM
    system_message = """You are an expert medical professional specializing in rare diseases. 
    Your task is to analyze patient phenotypes and determine if they likely indicate a rare disease.
    Please answer only with 'yes' or 'no'."""
    
    # Process samples and store results
    results = []
    for i, sample in enumerate(samples):
        phenotype_names = [hpo_mapping.get(p, "Unknown phenotype") for p in sample['phenotypes']]
        prompt = create_prompt(sample['phenotypes'], phenotype_names)
        print(f"\nSample {i + 1}:\n{prompt}")
        # Get model response
        response = llm_client.query(prompt, system_message).strip().lower()
        
        # Clean response to ensure it's yes/no
        is_rare = 'yes' in response[:3]  # Look at first 3 chars to catch "yes." etc.
        
        results.append({
            'sample_id': i,
            'disease_id': sample['disease_id'],
            'phenotypes': ','.join(sample['phenotypes']),
            'phenotype_names': ','.join(phenotype_names),
            'age': sample['age'],
            'is_rare_disease': 'yes' if is_rare else 'no'
        })
        
        # Print progress
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/100 samples")
    
    # Save results to CSV
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    
    # Calculate proportion of rare disease cases
    df = pd.DataFrame(results)
    rare_proportion = (df['is_rare_disease'] == 'yes').mean()
    
    print(f"\nAnalysis complete!")
    print(f"Results saved to: {output_file}")
    print(f"Proportion of rare disease cases: {rare_proportion:.2%}")

if __name__ == "__main__":
    main()