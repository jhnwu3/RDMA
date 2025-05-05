import json
import pandas as pd
import numpy as np
import traceback
from typing import Dict, Any, List, Optional, Union, Tuple
def read_json_file(file_path):
    """
    Read and parse a JSON file.
    
    Args:
        file_path (str): Path to the JSON file to be read.
    
    Returns:
        dict or list: The parsed JSON data.
    
    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        raise
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in file {file_path}")
        raise

def print_json_structure(data, indent=0):
    """
    Recursively print the structure of a JSON object or list, showing only types and keys.
    
    Args:
        data (dict or list): The JSON data to analyze.
        indent (int, optional): Current indentation level. Defaults to 0.
    """
    indent_str = "  " * indent
    
    if isinstance(data, dict):
        print(f"{indent_str}Dictionary:")
        for key, value in data.items():
            print(f"{indent_str}  {key} ({type(value).__name__}): ", end="")
            
            if isinstance(value, (dict, list)):
                print()
                print_json_structure(value, indent + 1)
            else:
                print()  # Newline for primitive types
    
    elif isinstance(data, list):
        print(f"{indent_str}List:")
        for i, item in enumerate(data):
            print(f"{indent_str}  Item {i} ({type(item).__name__}): ", end="")
            
            if isinstance(item, (dict, list)):
                print()
                print_json_structure(item, indent + 1)
            else:
                print()  # Newline for primitive types
    
    else:
        print(f"{indent_str}Primitive value: ({type(data).__name__})")
        
from typing import Tuple, Optional

def parse_case_range(case_range: str) -> Tuple[Optional[int], Optional[int]]:
    """
    Parse case ID range string into start and end IDs.
    
    Args:
        case_range: String in format 'start,end' (e.g. '1,10')
        
    Returns:
        Tuple of (start_id, end_id) or (None, None) if no range provided
        
    Raises:
        ValueError: If range format is invalid
    """
    if not case_range:
        return None, None
        
    try:
        parts = case_range.strip().split(',')
        if len(parts) != 2:
            raise ValueError("Case range must be in format 'start,end'")
            
        start_id = int(parts[0])
        end_id = int(parts[1])
        
        if start_id > end_id:
            raise ValueError(f"Start ID ({start_id}) must be less than or equal to End ID ({end_id})")
            
        return start_id, end_id
    except Exception as e:
        # Let the caller handle logging/errors
        raise ValueError(f"Error parsing case range: {e}")
    
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
    





def process_mimic_json(filepath: str) -> pd.DataFrame:
    """Process MIMIC-style JSON with annotations.
    
    Args:
        filepath: Path to JSON file containing clinical notes with annotations
        
    Returns:
        pd.DataFrame: DataFrame with processed notes and annotations
    """
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
        
        # Print dataset statistics
        print(f"\nDataset Statistics:")
        print(f"Total documents: {len(df)}")
        print(f"Documents with annotations: {len(df[df['gold_annotations'].str.len() > 0])}")
        print(f"Total annotations: {sum(df['gold_annotations'].str.len())}")
        print(f"Document categories: {df['category'].value_counts().to_dict()}")
        
        return df
        
    except Exception as e:
        print(f"Error processing JSON file: {str(e)}")
        traceback.print_exc()
        return pd.DataFrame()


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



def evaluate_predictions(predictions_df: pd.DataFrame, gold_df: pd.DataFrame) -> Dict[str, float]:
    """Evaluate predictions against gold standard annotations."""
    
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
    
    # Process first document for debugging
    if not predictions_df.empty and not gold_df.empty:
        doc_id = gold_df.iloc[0]['document_id']
        gold_anns = gold_df.iloc[0]['gold_annotations']
        
        print(f"\nProcessing first document (ID: {doc_id})")
        print("\nGold annotations:")
        print(json.dumps(gold_anns, indent=2))
        
        # Get all predictions for this document
        doc_preds = predictions_df[predictions_df['document_id'] == doc_id]
        print(f"\nFound {len(doc_preds)} prediction rows for this document")
        
        if not doc_preds.empty:
            # Create gold standard sets
            gold_entities = {ann['mention'].lower() for ann in gold_anns}
            gold_pairs = {(ann['mention'].lower(), ann['orpha_id']) for ann in gold_anns}
            
            print("\nGold entities:", gold_entities)
            
            # Collect all predictions for this document
            all_pred_entities = set()
            all_pred_pairs = set()
            
            for _, pred_row in doc_preds.iterrows():
                # Get entity and orpha_id directly from the row
                entity = pred_row.get('entity', '').lower() if pd.notna(pred_row.get('entity')) else ''
                orpha_id = pred_row.get('orpha_id', '') if pd.notna(pred_row.get('orpha_id')) else ''
                
                if entity:
                    all_pred_entities.add(entity)
                    if orpha_id:
                        all_pred_pairs.add((entity, orpha_id))
            
            print("\nPredicted entities:", all_pred_entities)
            print("Predicted pairs:", all_pred_pairs)
            
            # Calculate metrics for first document
            print("\nMetrics for first document:")
            print("Entity matching:")
            print(f"Correct entities: {gold_entities.intersection(all_pred_entities)}")
            print(f"Missed entities: {gold_entities - all_pred_entities}")
            print(f"Extra entities: {all_pred_entities - gold_entities}")
    
    # Process all documents
    for _, gold_row in gold_df.iterrows():
        doc_id = gold_row['document_id']
        gold_anns = gold_row['gold_annotations']
        
        # Create gold standard sets
        gold_entities = {ann['mention'].lower() for ann in gold_anns}
        gold_pairs = {(ann['mention'].lower(), ann['orpha_id']) for ann in gold_anns}
        
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
    print(metrics)
    
    return metrics['entity_metrics']