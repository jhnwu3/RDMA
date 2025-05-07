#!/usr/bin/env python3
import argparse
import json
import os
import sys
import pandas as pd
import numpy as np
import traceback
import torch
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
from rdma.rdrag.rd_match import RAGRDMatcher, SimpleRDMatcher
from rdma.utils.embedding import EmbeddingsManager
from rdma.utils.llm_client import LocalLLMClient, APILLMClient
from rdma.utils.setup import setup_device, initialize_llm_client

def timestamp_print(message: str) -> None:
    """Print message with timestamp."""
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Match verified rare disease entities to ORPHA codes")
    
    # Input/output files
    parser.add_argument("--input_file", required=True, 
                       help="Input JSON file from verification step")
    parser.add_argument("--output_file", required=True, 
                       help="Output JSON file for matched results")
    parser.add_argument("--embeddings_file", required=True,
                       help="NPY file containing rare disease embeddings")
    parser.add_argument("--csv_output", 
                       help="Optional CSV file for formatted results")
    
    # System prompt configuration
    parser.add_argument("--system_prompt", type=str, 
                       default="You are a medical expert specializing in rare diseases.",
                       help="System prompt for LLM matching")
    
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
    
    # Debug mode
    parser.add_argument("--debug", action="store_true", 
                       help="Enable debug output")
    
    return parser.parse_args()


# In step3_match_rd.py, change this function:

def load_verification_results(input_file: str) -> Dict:
    """Load verification results from step 2."""
    try:
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        # Check if the data has the expected structure with "metadata" and "results" keys
        if isinstance(data, dict) and "results" in data:
            timestamp_print(f"Found structured output with 'results' key")
            results = data["results"]
            metadata = data.get("metadata", {})
            
            return {
                "metadata": metadata,
                "results": results
            }
        else:
            # Fallback for older format where the entire JSON is the results
            timestamp_print(f"Using legacy format - treating entire JSON as results")
            
            return {
                "metadata": {},
                "results": data
            }
    except Exception as e:
        timestamp_print(f"Error loading verification results: {e}")
        raise


def load_existing_results(output_file: str) -> Dict:
    """Load existing matching results if available."""
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


def format_entities_for_matching(verified_entities: List[Dict]) -> List[Dict]:
    formatted_entities = []
    for entity_item in verified_entities:
        # Preserve the full entity dictionary
        if entity_item.get("status") == "verified_rare_disease":
            formatted_entities.append(entity_item)
    return formatted_entities

def convert_to_serializable(obj):
    """Convert all non-serializable types to serializable ones."""
    import numpy as np
    
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list) or isinstance(obj, tuple):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    else:
        # For any other types, convert to string
        return str(obj)
    
def match_cases(verification_results: Dict, matcher, args: argparse.Namespace, 
               embedded_documents: List[Dict], existing_results: Dict = None) -> Dict:
    """Match verified entities to rare disease terms using the matcher's interface."""
    results = existing_results or {}
    checkpoint_counter = 0
    
    # Extract the actual verification results
    cases = verification_results.get("results", verification_results)
    
    # Determine which cases need processing
    pending_cases = {case_id: case_data for case_id, case_data in cases.items() 
                   if case_id not in results or not results[case_id].get('matched_diseases')}
    
    timestamp_print(f"Matching rare diseases for {len(pending_cases)} cases out of {len(cases)} total cases")
    
    # Convert to list for progress tracking
    case_items = list(pending_cases.items())
    
    # Use tqdm for progress tracking
    for i, (case_id, case_data) in enumerate(tqdm(case_items, desc="Matching rare diseases")):
        try:
            if args.debug:
                timestamp_print(f"Processing case {i+1}/{len(pending_cases)} (ID: {case_id})")
            
            # Get entities to match - check for verified_rare_diseases first,
            # then fall back to entities_with_contexts from extraction step
            if "verified_rare_diseases" in case_data and case_data["verified_rare_diseases"]:
                verified_entities = case_data["verified_rare_diseases"]
                timestamp_print(f"  Found {len(verified_entities)} verified rare diseases for case {case_id}")
            elif "entities_with_contexts" in case_data and case_data["entities_with_contexts"]:
                # Using entities directly from extraction step
                verified_entities = case_data["entities_with_contexts"]
                timestamp_print(f"  Using {len(verified_entities)} extracted entities for case {case_id}")
            else:
                verified_entities = []
                timestamp_print(f"  Warning: No entities found for case {case_id}")
            
            if args.debug:
                timestamp_print(f"  Processing {len(verified_entities)} verified rare diseases")
            
            # Skip processing if no verified entities
            if not verified_entities:
                results[case_id] = {
                    "clinical_text": case_data.get("clinical_text", ""),
                    "metadata": case_data.get("metadata", {}),
                    "matched_diseases": [],
                    "note": "No verified rare diseases to match"
                }
                continue
            
            # Extract just the entity text for matching but preserve full metadata
            entity_texts = []
            entity_metadata = {}
            
            for entity_item in verified_entities:
                # print(entity_item)
                if isinstance(entity_item, str):
                    entity_texts.append(entity_item)
                    entity_metadata[entity_item] = {"original_entity": entity_item}
                    
            # Remove empty entries and duplicates while preserving order
            # Extract entities while preserving full metadata
            unique_entities = []

            for entity_item in verified_entities:
                # Skip completely empty or invalid entries
                if not isinstance(entity_item, dict):
                    continue
                
                # Specifically check for verified rare disease status or presence of an entity
                if entity_item.get("status") == "verified_rare_disease" or entity_item.get("entity"):
                    unique_entities.append(entity_item)

            matched_results = matcher.match_rd_terms(unique_entities, embedded_documents)
            if args.debug:
                timestamp_print(f"  Matching {len(unique_entities)} unique entities")
            
            # Ensure matcher's index is initialized
            if not hasattr(matcher, 'index') or matcher.index is None:
                matcher.prepare_index(embedded_documents)
            
            # Call matcher's match_rd_terms directly
            matched_results = matcher.match_rd_terms(unique_entities, embedded_documents)
            
            # Enrich the matches with preserved metadata
            matched_diseases = []
            for match in matched_results:
                entity = match.get("entity", "")
                if entity:
                    # Preserve the original metadata from verification step
                    preserved_metadata = entity_metadata.get(entity, {})
                    
                    # Create enriched match with all metadata
                    enriched_match = match.copy()
                    
                    # Add preserved metadata from verification
                    for key, value in preserved_metadata.items():
                        if value is not None and (key not in enriched_match or not enriched_match[key]):
                            enriched_match[key] = value
                    
                    matched_diseases.append(enriched_match)
                    
                    if args.debug:
                        if "rd_term" in enriched_match:
                            entity_desc = f"'{entity}'"
                            if "original_entity" in enriched_match and enriched_match["original_entity"] != entity:
                                entity_desc += f" (original: '{enriched_match['original_entity']}')"
                            if "expanded_term" in enriched_match:
                                entity_desc += f" (expanded: '{enriched_match['expanded_term']}')"
                                
                            timestamp_print(f"  ✓ Matched {entity_desc} to {enriched_match['rd_term']} ({enriched_match['orpha_id']})")
            
            # Store results
            results[case_id] = {
                "clinical_text": case_data.get("clinical_text", ""),
                "metadata": case_data.get("metadata", {}),
                "matched_diseases": matched_diseases,
                "stats": {
                    "verified_diseases_count": len(verified_entities),
                    "matched_diseases_count": len(matched_diseases)
                }
            }
            # print(results[case_id])
            
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
                "metadata": case_data.get("metadata", {}),
                "matched_diseases": [],
                "stats": {
                    "verified_diseases_count": len(case_data.get("verified_rare_diseases", [])),
                    "matched_diseases_count": 0
                },
                "error": str(e)
            }
    
    return results


def format_csv_results(results: Dict) -> pd.DataFrame:
    """Format results as a CSV-friendly DataFrame."""
    csv_data = []
    
    for case_id, case_data in results.items():
        matched_diseases = case_data.get("matched_diseases", [])
        metadata = case_data.get("metadata", {})
        
        for disease in matched_diseases:
            # Extract relevant fields for CSV
            entry = {
                "document_id": case_id,
                "patient_id": metadata.get("patient_id", ""),
                "admission_id": metadata.get("admission_id", ""),
                "category": metadata.get("category", ""),
                "chart_date": metadata.get("chart_date", ""),
                "entity": disease.get("entity", ""),
                "rd_term": disease.get("rd_term", ""),
                "orpha_id": disease.get("orpha_id", ""),
                "match_method": disease.get("match_method", ""),
                "confidence_score": disease.get("confidence_score", 0.0)
            }
            csv_data.append(entry)
    
    return pd.DataFrame(csv_data)


def main():
    """Main function to run the rare disease matching pipeline."""
    try:
        # Parse command line arguments
        args = parse_arguments()
        
        timestamp_print(f"Starting rare disease matching process")
        
        # Setup device
        device = setup_device(args)
        timestamp_print(f"Using device: {device}")
        
        # Initialize LLM client
        timestamp_print(f"Initializing {args.llm_type} LLM client")
        llm_client = initialize_llm_client(args, device)
        
        # Initialize embedding manager
        # Initialize embedding manager
        timestamp_print(f"Initializing {args.retriever} embedding manager")
        embedding_manager = EmbeddingsManager(
            model_type=args.retriever,
            model_name=args.retriever_model if args.retriever in ['fastembed', 'sentence_transformer'] else None,
            device=device['retriever']  # Access the specific device string
        )
        
        # Initialize matcher
        timestamp_print(f"Initializing RAG matcher")
        matcher = SimpleRDMatcher(
            embeddings_manager=embedding_manager,
            llm_client=llm_client,
            system_message=args.system_prompt
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
        
        # Load verification results from step 2
        timestamp_print(f"Loading verification results from {args.input_file}")
        verification_data = load_verification_results(args.input_file)
        
        # Extract results and metadata
        verification_results = verification_data.get("results", verification_data)
        verification_metadata = verification_data.get("metadata", {})
        
        timestamp_print(f"Loaded verification results for {len(verification_results)} cases")
        
        # Check for existing results if resuming
        existing_results = {}
        if args.resume:
            existing_results = load_existing_results(args.output_file)
        
        # Match verified entities to rare disease terms
        timestamp_print(f"Starting rare disease matching")
        results = match_cases(verification_results, matcher, args, embedded_documents, existing_results)
        
        # Save results to JSON
        timestamp_print(f"Saving matching results to {args.output_file}")
        os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
        
        # Add metadata about the matching run
        metadata = {
            "verification_metadata": verification_metadata,
            "matching_timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
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
        
        # Convert to serializable format
        serializable_output = convert_to_serializable(final_output)
        
        with open(args.output_file, 'w') as f:
            json.dump(serializable_output, f, indent=2)
        
        # Optionally save as CSV
        if args.csv_output:
            timestamp_print(f"Saving results as CSV to {args.csv_output}")
            csv_df = format_csv_results(results)
            csv_df.to_csv(args.csv_output, index=False)
        
        # Print summary
        total_verified_entities = sum(case_data.get("stats", {}).get("verified_diseases_count", 0) for case_data in results.values())
        total_matched_entities = sum(case_data.get("stats", {}).get("matched_diseases_count", 0) for case_data in results.values())
        
        # Calculate match rate
        match_rate = (total_matched_entities / total_verified_entities * 100) if total_verified_entities > 0 else 0
        
        timestamp_print(f"Matching complete:")
        timestamp_print(f"  Total verified rare diseases: {total_verified_entities}")
        timestamp_print(f"  Successfully matched to ORPHA codes: {total_matched_entities} ({match_rate:.1f}%)")
        
        timestamp_print(f"Rare disease matching completed successfully.")
    
    except Exception as e:
        timestamp_print(f"Critical error: {e}")
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()