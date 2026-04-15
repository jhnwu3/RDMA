#!/usr/bin/env python3
import argparse
import json
import os
import sys
import traceback
import pandas as pd
import torch
from datetime import datetime
from typing import Dict, List, Union

# Import LLM clients
from utils.llm_client import LocalLLMClient, APILLMClient

def process_mimic_json(filepath: str) -> pd.DataFrame:
    """Process MIMIC-style JSON with clinical notes."""
    try:
        # Load JSON data
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        # Process each document
        records = []
        for doc_id, doc_data in data.items():
            if 'note_details' not in doc_data:
                continue
                
            note_details = doc_data['note_details']
            
            # Extract relevant fields
            record = {
                'document_id': doc_id,
                'patient_id': note_details.get('subject_id'),
                'admission_id': note_details.get('hadm_id'),
                'category': note_details.get('category'),
                'chart_date': note_details.get('chartdate'),
                'clinical_note': note_details.get('text', ''),
                'gold_annotations': doc_data.get('annotations', [])
            }
            
            records.append(record)
            
        # Create DataFrame
        df = pd.DataFrame(records)
        
        # Basic validation and cleaning
        df['clinical_note'] = df['clinical_note'].astype(str)
        df = df.dropna(subset=['clinical_note'])
        
        # Print dataset statistics
        print(f"\nDataset Statistics:")
        print(f"Total documents: {len(df)}")
        print(f"Documents with gold annotations: {len(df[df['gold_annotations'].str.len() > 0])}")
        print(f"Total gold annotations: {sum(df['gold_annotations'].str.len())}")
        print(f"Document categories: {df['category'].value_counts().to_dict()}")
        
        return df
        
    except Exception as e:
        print(f"Error processing JSON file: {str(e)}")
        traceback.print_exc()
        return pd.DataFrame()

def zero_shot_rd_extraction(llm_client, text: str) -> List[Dict]:
    """
    Use LLM to extract rare disease mentions in a zero-shot manner.
    
    Args:
        llm_client: Initialized LLM client
        text: Clinical note text
    
    Returns:
        List of extracted rare disease mentions
    """
    system_message = """You are an expert medical information extraction assistant specializing in rare disease detection. 
    Extract ALL potential rare disease mentions from the given clinical text. 
    
    Important guidelines:
    1. Focus on rare diseases (affecting less than 1 in 2000 people)
    2. Include diseases, syndromes, and specific rare medical conditions
    3. Do NOT include common diseases or general medical conditions
    4. Provide a structured JSON response with the following format:
       [
         {
           "mention": "Extracted disease name",
           "context": "Brief surrounding context",
           "confidence": 0.7  # Estimated confidence (0.0 to 1.0)
         },
         ...
       ]
    5. If no rare diseases are found, return an empty list
    6. Be precise and conservative in your extractions"""
    
    prompt = f"""Extract rare disease mentions from the following clinical text:

Text: ```{text}```

Respond ONLY with a JSON-formatted list of rare disease mentions."""
    
    try:
        response = llm_client.query(prompt, system_message)
        
        # Try to parse the JSON response
        try:
            # First, try to find JSON between the outermost square brackets
            import re
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                response_json = json_match.group(0)
            else:
                response_json = response
            
            # Parse the JSON
            mentions = json.loads(response_json)
            
            # Basic validation
            if not isinstance(mentions, list):
                return []
            
            # Clean and validate each mention
            cleaned_mentions = []
            for mention in mentions:
                if not isinstance(mention, dict):
                    continue
                
                # Ensure required keys exist with valid values
                cleaned_mention = {
                    'mention': str(mention.get('mention', '')).strip(),
                    'context': str(mention.get('context', '')).strip(),
                    'confidence': float(mention.get('confidence', 0.7))
                }
                
                # Only add if mention is not empty
                if cleaned_mention['mention']:
                    cleaned_mentions.append(cleaned_mention)
            
            return cleaned_mentions
        
        except json.JSONDecodeError:
            print("Could not parse LLM response as JSON. Returning empty list.")
            return []
    
    except Exception as e:
        print(f"Error in zero-shot extraction: {str(e)}")
        return []

def zero_shot_rd_pipeline(
    input_file: str, 
    output_file: str, 
    llm_client,
    batch_size: int = 1
) -> pd.DataFrame:
    """
    Process the entire dataset using zero-shot LLM extraction.
    
    Args:
        input_file: Path to input JSON file
        output_file: Path to output CSV file
        llm_client: Initialized LLM client
        batch_size: Number of documents to process in parallel
    
    Returns:
        DataFrame with rare disease mentions
    """
    # Load dataset
    dataset_df = process_mimic_json(input_file)
    
    # Prepare results list
    results = []
    
    # Process documents in batches
    for i in range(0, len(dataset_df), batch_size):
        batch = dataset_df.iloc[i:i + batch_size]
        
        for _, row in batch.iterrows():
            # Perform zero-shot extraction
            mentions = zero_shot_rd_extraction(llm_client, row['clinical_note'])
            
            # Prepare results for this document
            for mention in mentions:
                result = {
                    'document_id': row['document_id'],
                    'entity': mention['mention'],
                    'context': mention['context'],
                    'confidence': mention['confidence'],
                    'clinical_note': row['clinical_note']
                }
                results.append(result)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    results_df.to_csv(output_file, index=False)
    
    return results_df

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Zero-shot Rare Disease Mention Extraction')
    
    # LLM Configuration
    parser.add_argument('--llm_type', type=str,
                       choices=['local', 'api'],
                       default='local',
                       help='Type of LLM client to use')
    
    parser.add_argument('--model', type=str,
                       choices=['llama3_8b', 'llama3_70b', 'llama3_70b_2b', 'llama3_70b_full', 
                               'llama3-groq-70b', 'llama3-groq-8b', 'openbio_70b', 'openbio_8b', 'mistral_24b'], 
                       default='llama3_70b',
                       help='Model to use for inference')
    
    # GPU Configuration
    gpu_group = parser.add_mutually_exclusive_group()
    gpu_group.add_argument('--gpu_id', type=int,
                          help='Specific GPU ID to use')
    gpu_group.add_argument('--cpu', action='store_true',
                          help='Force CPU usage')
    
    # Input/Output Configuration
    parser.add_argument('--input_file', required=True,
                       help='Path to input JSON file with clinical notes')
    parser.add_argument('--output_file', required=True,
                       help='Path for output CSV file')
    
    # Processing Configuration
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size for processing')
    parser.add_argument('--temperature', type=float, default=0.1,
                       help='Temperature for LLM inference')
    parser.add_argument('--cache_dir', type=str,
                       default="/shared/rsaas/jw3/rare_disease/model_cache",
                       help='Directory for caching models')
    
    return parser.parse_args()

def setup_device(args):
    """Configure device based on arguments."""
    if args.cpu:
        return "cpu"
    elif args.gpu_id is not None:
        return f"cuda:{args.gpu_id}"
    else:
        return "cuda" if torch.cuda.is_available() else "cpu"

def main():
    """Main function for zero-shot rare disease extraction."""
    start_time = datetime.now()
    
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Setup device
        device = setup_device(args)
        print(f"Using device: {device}")
        
        # Initialize LLM
        if args.llm_type == 'local':
            llm_client = LocalLLMClient(
                model_type=args.model,
                device=device,
                cache_dir=args.cache_dir,
                temperature=args.temperature
            )
        else:
            llm_client = APILLMClient(
                model_type=args.model,
                device=device,
                cache_dir=args.cache_dir,
                temperature=args.temperature
            )
        
        # Run zero-shot extraction pipeline
        print(f"\nProcessing input dataset: {args.input_file}")
        results_df = zero_shot_rd_pipeline(
            input_file=args.input_file,
            output_file=args.output_file,
            llm_client=llm_client,
            batch_size=args.batch_size
        )
        
        # Print summary
        print("\nExtraction Summary:")
        print(f"Total documents processed: {len(results_df['document_id'].unique())}")
        print(f"Total rare disease mentions extracted: {len(results_df)}")
        print(f"Results saved to: {args.output_file}")
        
        # Print processing time
        end_time = datetime.now()
        print(f"\nStarted at:  {start_time}")
        print(f"Finished at: {end_time}")
        print(f"Total time:  {end_time - start_time}")
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()