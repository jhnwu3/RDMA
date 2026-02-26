#!/usr/bin/env python3
import argparse
import os
import sys
import pandas as pd
import numpy as np
import json
import traceback
from typing import Dict, List, Any, Set, Tuple
from datetime import datetime
import re

class NumpyJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles NumPy data types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def load_human_corrections(corrections_file: str) -> List[str]:
    """
    Load the list of human-annotated terms that should be true positives.
    
    Args:
        corrections_file: Path to the file containing human corrections
        
    Returns:
        List of corrected terms
    """
    try:
        with open(corrections_file, 'r') as f:
            # Try loading as JSON first
            try:
                data = json.load(f)
                if isinstance(data, list):
                    return [term.lower().strip() for term in data]
                elif isinstance(data, dict) and 'terms' in data:
                    return [term.lower().strip() for term in data['terms']]
                else:
                    print("Warning: JSON file structure not recognized. Expected list or dict with 'terms' key.")
                    return []
            except json.JSONDecodeError:
                # If not JSON, try loading as a simple text file (one term per line)
                f.seek(0)
                return [line.lower().strip() for line in f.readlines() if line.strip()]
    except Exception as e:
        print(f"Error loading corrections file: {str(e)}")
        traceback.print_exc()
        return []

def load_mimic_json(filepath: str) -> Dict:
    """
    Load MIMIC-style JSON dataset.
    
    Args:
        filepath: Path to the JSON file
        
    Returns:
        Dictionary containing the dataset
    """
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        print(f"Loaded dataset with {len(data)} documents")
        return data
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

def load_intermediate_results(filepath: str) -> pd.DataFrame:
    """
    Load intermediate results from CSV file.
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        DataFrame containing the results
    """
    try:
        df = pd.read_csv(filepath)
        print(f"Loaded intermediate results with {len(df)} rows")
        return df
    except Exception as e:
        print(f"Error loading intermediate results: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

def find_term_in_text(term: str, text: str) -> List[Tuple[int, int]]:
    """
    Find all occurrences of a term in text with start and end indices.
    Uses case-insensitive matching.
    
    Args:
        term: Term to search for
        text: Text to search in
        
    Returns:
        List of (start, end) position tuples for each occurrence
    """
    positions = []
    pattern = re.compile(re.escape(term), re.IGNORECASE)
    
    for match in pattern.finditer(text):
        positions.append((match.start(), match.end()))
        
    return positions

def correct_annotations(dataset: Dict, 
                        corrected_terms: List[str], 
                        results_df: pd.DataFrame = None) -> Dict:
    """
    Add human-corrected terms to gold annotations in the dataset.
    
    Args:
        dataset: Original MIMIC-style dataset
        corrected_terms: List of human-corrected terms
        results_df: Optional DataFrame with intermediate results
        
    Returns:
        Updated dataset with corrected annotations
    """
    print(f"Processing {len(corrected_terms)} human-corrected terms")
    
    # Lowercase all terms for consistent matching
    corrected_terms = [term.lower() for term in corrected_terms]
    
    # Keep track of corrections made
    corrections_count = 0
    documents_corrected = 0
    
    # Create a dictionary to store ORPHA IDs for corrected terms (if available)
    term_to_orpha = {}
    
    # If we have intermediate results, extract ORPHA IDs for corrected terms
    if results_df is not None:
        for term in corrected_terms:
            # Find matching rows in results (case-insensitive)
            matches = results_df[results_df['entity'].str.lower() == term.lower()]
            
            if not matches.empty:
                # Get the most common ORPHA ID for this term
                orpha_ids = matches['orpha_id'].dropna().value_counts()
                if not orpha_ids.empty:
                    term_to_orpha[term] = orpha_ids.index[0]
    
    # Process each document
    for doc_id, doc_data in dataset.items():
        if 'note_details' not in doc_data:
            continue
            
        # Get document text
        text = doc_data['note_details'].get('text', '')
        if not text:
            continue
            
        # Initialize annotations list if not present
        if 'annotations' not in doc_data:
            doc_data['annotations'] = []
            
        # Get existing annotations
        annotations = doc_data['annotations']
        
        # Get existing mentions to avoid duplicates
        existing_mentions = {ann.get('mention', '').lower() for ann in annotations}
        
        document_corrected = False
        
        # Check for each corrected term
        for term in corrected_terms:
            # Skip if term is already in annotations
            if term in existing_mentions:
                continue
                
            # Find positions of term in text
            positions = find_term_in_text(term, text)
            
            if positions:
                # Term found in document text
                document_corrected = True
                
                # Create new annotation for each occurrence
                for start, end in positions:
                    # Extract the exact term as it appears in text
                    exact_term = text[start:end]
                    
                    # Create annotation
                    new_annotation = {
                        'mention': exact_term,
                        'document_structure': 'corrected'
                    }
                    
                    # Add ORPHA ID if available
                    if term in term_to_orpha:
                        orpha_id = term_to_orpha[term]
                        new_annotation['ordo_with_desc'] = f"{orpha_id} (human corrected)"
                    
                    # Add annotation
                    annotations.append(new_annotation)
                    existing_mentions.add(exact_term.lower())
                    corrections_count += 1
        
        if document_corrected:
            documents_corrected += 1
    
    print(f"Added {corrections_count} new annotations to {documents_corrected} documents")
    return dataset

def process_mimic_json(data: Dict) -> pd.DataFrame:
    """
    Process MIMIC-style JSON with annotations.
    
    Args:
        data: Dictionary containing the dataset
        
    Returns:
        pd.DataFrame: DataFrame with processed notes and annotations
    """
    try:
        # Process each document
        records = []
        for doc_id, doc_data in data.items():
            if 'note_details' not in doc_data:
                continue
                
            note_details = doc_data['note_details']
            annotations = doc_data.get('annotations', [])
            
            # Extract relevant fields
            record = {
                'document_id': doc_id,
                'patient_id': note_details.get('subject_id'),
                'admission_id': note_details.get('hadm_id'),
                'category': note_details.get('category'),
                'chart_date': note_details.get('chartdate'),
                'clinical_note': note_details.get('text', ''),
                'gold_annotations': []
            }
            
            # Process all annotations that have a mention
            for ann in annotations:
                if ann.get('mention'):  # Include any annotation with a mention
                    gold_annotation = {
                        'mention': ann['mention'],
                        'orpha_id': ann.get('ordo_with_desc', '').split()[0] if ann.get('ordo_with_desc') else '',
                        'orpha_desc': ' '.join(ann.get('ordo_with_desc', '').split()[1:]) if ann.get('ordo_with_desc') else '',
                        'document_section': ann.get('document_structure'),
                        'confidence': 1.0
                    }
                    record['gold_annotations'].append(gold_annotation)
            
            records.append(record)
            
        # Create DataFrame
        df = pd.DataFrame(records)
        
        # Basic validation and cleaning
        df['clinical_note'] = df['clinical_note'].astype(str)
        df = df.dropna(subset=['clinical_note'])
        
        return df
        
    except Exception as e:
        print(f"Error processing JSON data: {str(e)}")
        traceback.print_exc()
        return pd.DataFrame()

def evaluate_predictions(predictions_df: pd.DataFrame, gold_df: pd.DataFrame) -> Dict[str, float]:
    """
    Evaluate predictions against gold standard annotations.
    
    Args:
        predictions_df: DataFrame with pipeline predictions
        gold_df: DataFrame with gold standard annotations
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Convert document_id to string in both DataFrames for consistent comparison
    predictions_df = predictions_df.copy()
    gold_df = gold_df.copy()
    predictions_df['document_id'] = predictions_df['document_id'].astype(str)
    gold_df['document_id'] = gold_df['document_id'].astype(str)
    
    # Initialize counters for entity matching
    entity_true_positives = 0
    entity_false_positives = 0
    entity_false_negatives = 0
    
    # Initialize counters for ORPHA ID matching
    orpha_true_positives = 0
    orpha_false_positives = 0
    orpha_false_negatives = 0
    
    # Process all documents
    for _, gold_row in gold_df.iterrows():
        doc_id = gold_row['document_id']
        gold_anns = gold_row['gold_annotations']
        
        # Create gold standard sets
        gold_entities = {ann['mention'].lower() for ann in gold_anns}
        gold_pairs = {(ann['mention'].lower(), ann['orpha_id']) for ann in gold_anns if ann['orpha_id']}
        
        # Get all predictions for this document
        doc_preds = predictions_df[predictions_df['document_id'] == doc_id]
        
        if doc_preds.empty:
            entity_false_negatives += len(gold_entities)
            orpha_false_negatives += len(gold_pairs)
            continue
        
        # Collect all predictions for this document
        pred_entities = set()
        pred_pairs = set()
        
        for _, pred_row in doc_preds.iterrows():
            entity = pred_row.get('entity', '').lower() if pd.notna(pred_row.get('entity')) else ''
            orpha_id = pred_row.get('orpha_id', '') if pd.notna(pred_row.get('orpha_id')) else ''
            
            if entity:
                pred_entities.add(entity)
                if orpha_id:
                    pred_pairs.add((entity, orpha_id))
        
        # Update entity metrics
        entity_true_positives += len(gold_entities.intersection(pred_entities))
        entity_false_positives += len(pred_entities - gold_entities)
        entity_false_negatives += len(gold_entities - pred_entities)
        
        # Update ORPHA ID metrics
        orpha_true_positives += len(gold_pairs.intersection(pred_pairs))
        orpha_false_positives += len(pred_pairs - gold_pairs)
        orpha_false_negatives += len(gold_pairs - pred_pairs)
    
    # Calculate metrics
    metrics = {
        'entity_metrics': calculate_metrics(entity_true_positives, entity_false_positives, entity_false_negatives),
        'orpha_metrics': calculate_metrics(orpha_true_positives, orpha_false_positives, orpha_false_negatives)
    }
    
    return metrics

def calculate_metrics(true_positives, false_positives, false_negatives):
    """Helper function to calculate precision, recall, and F1 score."""
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives
    }

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Correct rare disease annotations with human-verified terms')
    
    # Required arguments
    parser.add_argument('--input_file', type=str, required=True,
                        help='Path to original input JSON file with clinical notes and annotations')
    
    parser.add_argument('--corrections_file', type=str, required=True,
                        help='Path to file containing human-verified true positive terms')
    
    parser.add_argument('--intermediate_file', type=str, required=True,
                        help='Path to intermediate results file with predictions')
    
    parser.add_argument('--output_file', type=str, required=True,
                        help='Path for corrected output JSON file')
    
    # Optional arguments
    parser.add_argument('--metrics_file', type=str,
                        help='Path for corrected metrics JSON file')
    
    parser.add_argument('--original_metrics_file', type=str,
                        help='Path to original metrics JSON file for comparison')
    
    return parser.parse_args()

def main():
    """Main function for correcting annotations and recalculating metrics."""
    start_time = datetime.now()
    
    try:
        # Parse arguments
        args = parse_arguments()
        
        print(f"\nStarting correction process at: {start_time}")
        print(f"Configuration:")
        print(f"  Input file: {args.input_file}")
        print(f"  Corrections file: {args.corrections_file}")
        print(f"  Intermediate file: {args.intermediate_file}")
        print(f"  Output file: {args.output_file}")
        
        # Load human corrections
        corrected_terms = load_human_corrections(args.corrections_file)
        if not corrected_terms:
            print("No corrected terms found. Exiting.")
            sys.exit(1)
        
        # Load original dataset
        dataset = load_mimic_json(args.input_file)
        
        # Load intermediate results
        results_df = load_intermediate_results(args.intermediate_file)
        
        # Correct annotations
        corrected_dataset = correct_annotations(dataset, corrected_terms, results_df)
        
        # Process dataset into DataFrame for evaluation
        dataset_df = process_mimic_json(corrected_dataset)
        if dataset_df.empty:
            raise ValueError("No valid data found in processed dataset")
        
        # Print dataset statistics
        print(f"\nCorrected Dataset Statistics:")
        print(f"Total documents: {len(dataset_df)}")
        print(f"Documents with annotations: {len(dataset_df[dataset_df['gold_annotations'].str.len() > 0])}")
        print(f"Total annotations: {sum(dataset_df['gold_annotations'].str.len())}")
        
        # Save corrected dataset
        print(f"\nSaving corrected dataset to: {args.output_file}")
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        with open(args.output_file, 'w') as f:
            json.dump(corrected_dataset, f, cls=NumpyJSONEncoder, indent=2)
        
        # Calculate updated metrics
        print("\nCalculating updated metrics...")
        metrics = evaluate_predictions(results_df, dataset_df)
        
        # Print metrics
        print("\nUpdated Entity Metrics:")
        print(f"Precision: {metrics['entity_metrics']['precision']:.3f}")
        print(f"Recall: {metrics['entity_metrics']['recall']:.3f}")
        print(f"F1 Score: {metrics['entity_metrics']['f1']:.3f}")
        print(f"True Positives: {metrics['entity_metrics']['true_positives']}")
        print(f"False Positives: {metrics['entity_metrics']['false_positives']}")
        print(f"False Negatives: {metrics['entity_metrics']['false_negatives']}")
        
        # Save metrics if requested
        if args.metrics_file:
            print(f"\nSaving updated metrics to: {args.metrics_file}")
            os.makedirs(os.path.dirname(args.metrics_file), exist_ok=True)
            with open(args.metrics_file, 'w') as f:
                json.dump(metrics, f, cls=NumpyJSONEncoder, indent=2)
        
        # Compare with original metrics if available
        if args.original_metrics_file and os.path.exists(args.original_metrics_file):
            try:
                with open(args.original_metrics_file, 'r') as f:
                    original_metrics = json.load(f)
                
                print("\nComparison with Original Metrics:")
                original_f1 = original_metrics.get('entity_metrics', {}).get('f1', 0)
                new_f1 = metrics['entity_metrics']['f1']
                improvement = new_f1 - original_f1
                
                print(f"Original F1 Score: {original_f1:.3f}")
                print(f"Updated F1 Score: {new_f1:.3f}")
                print(f"Improvement: {improvement:.3f} ({improvement*100:.1f}%)")
            except Exception as e:
                print(f"Error comparing with original metrics: {str(e)}")
        
        # Print final summary
        end_time = datetime.now()
        print("\nProcessing Summary:")
        print(f"Started at:  {start_time}")
        print(f"Finished at: {end_time}")
        print(f"Total time:  {end_time - start_time}")
        print("Correction process completed successfully")
        
    except Exception as e:
        end_time = datetime.now()
        print(f"\nError occurred after running for: {end_time - start_time}")
        print(f"Error: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()