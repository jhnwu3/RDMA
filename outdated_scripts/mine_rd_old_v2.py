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

class SupervisedEvaluator:
    """Evaluator that uses LLM supervision to re-evaluate false positives in rare disease detection."""
    
    def __init__(
        self, 
        predictions_df: pd.DataFrame, 
        gold_df: pd.DataFrame, 
        rd_matcher: RAGRDMatcher,
        embedded_documents: List[Dict],
        output_file: str = None
    ):
        """
        Initialize the supervised evaluator.
        
        Args:
            predictions_df: DataFrame with pipeline predictions
            gold_df: DataFrame with gold standard annotations
            rd_matcher: Rare disease matcher for verification
            embedded_documents: Embedded rare disease documents
            output_file: Path to save detailed results
        """
        self.predictions_df = predictions_df.copy()
        self.gold_df = gold_df.copy()
        self.rd_matcher = rd_matcher
        self.embedded_documents = embedded_documents
        self.output_file = output_file
        
        # Convert document_id to string in both DataFrames for consistent comparison
        self.predictions_df['document_id'] = self.predictions_df['document_id'].astype(str)
        self.gold_df['document_id'] = self.gold_df['document_id'].astype(str)
        
        # Initialize the matcher's index with the embedded documents
        # This is crucial - without this, the matcher's index will be None
        if self.rd_matcher.index is None:
            print("Initializing matcher's search index...")
            self.rd_matcher.prepare_index(self.embedded_documents)
        
        # Initialize result storage
        self.results = {}
        
    def evaluate(self) -> Dict[str, Any]:
        """
        Run supervised evaluation on false positives.
        
        Returns:
            Dict with evaluation metrics and details
        """
        print("\nRunning supervised evaluation...")
        
        # Process all documents
        for _, gold_row in self.gold_df.iterrows():
            doc_id = gold_row['document_id']
            gold_anns = gold_row['gold_annotations']
            original_text = gold_row.get('clinical_note', '')
            
            # Initialize document results
            self.results[doc_id] = {
                'document_id': doc_id,
                'old_true_positives': [],
                'old_false_positives': [],
                'false_negatives': [],
                'new_true_positives': [],
                'new_false_positives': [],
                'original_text': original_text
            }
            
            # Create gold standard sets
            gold_entities = {ann['mention'].lower() for ann in gold_anns}
            
            # Get all predictions for this document
            doc_preds = self.predictions_df[self.predictions_df['document_id'] == doc_id]
            
            if doc_preds.empty:
                # All gold entities are false negatives
                self.results[doc_id]['false_negatives'] = [
                    {'mention': ann['mention'], 'orpha_id': ann.get('orpha_id', '')}
                    for ann in gold_anns
                ]
                continue
            
            # Collect all predictions for this document
            pred_entities = []
            
            for _, pred_row in doc_preds.iterrows():
                entity = pred_row.get('entity', '').lower() if pd.notna(pred_row.get('entity')) else ''
                orpha_id = pred_row.get('orpha_id', '') if pd.notna(pred_row.get('orpha_id')) else ''
                rd_term = pred_row.get('rd_term', '') if pd.notna(pred_row.get('rd_term')) else ''
                
                if entity:
                    # Collect prediction details
                    pred_info = {
                        'entity': entity,
                        'orpha_id': orpha_id,
                        'rd_term': rd_term,
                    }
                    
                    pred_entities.append(pred_info)
            
            # Categorize predictions
            for pred in pred_entities:
                entity = pred['entity']
                
                if entity in gold_entities:
                    # This is a true positive for entity
                    self.results[doc_id]['old_true_positives'].append(pred)
                else:
                    # This is a false positive for entity
                    # Check if it exists in the original text
                    if original_text and self._entity_exists_in_text(entity, original_text):
                        self.results[doc_id]['old_false_positives'].append(pred)
                    else:
                        # It's a false positive that doesn't exist in text, 
                        # adding to a separate category to track these
                        if 'nonexistent_fps' not in self.results[doc_id]:
                            self.results[doc_id]['nonexistent_fps'] = []
                        self.results[doc_id]['nonexistent_fps'].append(pred)
                        print(f"Note: Entity '{entity}' not found in original text of doc {doc_id}")
            
            # Find false negatives
            for ann in gold_anns:
                if ann['mention'].lower() not in {pred['entity'] for pred in pred_entities}:
                    self.results[doc_id]['false_negatives'].append({
                        'mention': ann['mention'], 
                        'orpha_id': ann.get('orpha_id', '')
                    })
            
            # Re-evaluate false positives
            self._reevaluate_false_positives(doc_id)
        
        # Calculate overall metrics
        metrics = self._calculate_metrics()
        
        # Print info about nonexistent entities
        nonexistent_count = sum(len(doc.get('nonexistent_fps', [])) for doc in self.results.values())
        if nonexistent_count > 0:
            print(f"\nFound {nonexistent_count} false positive entities that don't exist in original text")
        
        # Save detailed results if output file specified
        if self.output_file:
            # Convert all results to JSON-serializable format
            serializable_results = self._prepare_for_serialization(self.results)
            serializable_metrics = self._prepare_for_serialization(metrics)
            
            os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
            try:
                with open(self.output_file, 'w') as f:
                    json.dump({
                        'metrics': serializable_metrics,
                        'document_results': serializable_results
                    }, f, indent=2, cls=NumpyJSONEncoder)
                print(f"Results saved to {self.output_file}")
            except Exception as e:
                print(f"Error saving results to file: {str(e)}")
                print("Continuing with evaluation...")
        
        return metrics
    
    def _entity_exists_in_text(self, entity: str, text: str) -> bool:
        """
        Check if an entity exists in the original text.
        
        Args:
            entity: Entity text to search for
            text: Original text to search in
            
        Returns:
            True if entity is found in text, False otherwise
        """
        # Normalize both entity and text for more accurate matching
        entity_normalized = entity.lower()
        text_normalized = text.lower()
        
        # Check for exact match
        if entity_normalized in text_normalized:
            return True
            
        # Check for case where entity has punctuation that might be different in text
        # Strip punctuation and check again
        import string
        entity_no_punct = entity_normalized.translate(str.maketrans('', '', string.punctuation))
        if entity_no_punct and entity_no_punct in text_normalized:
            return True
            
        return False
    
    def _prepare_for_serialization(self, data):
        """
        Recursively convert all numpy types to Python native types for JSON serialization.
        
        Args:
            data: Data structure to convert
            
        Returns:
            JSON-serializable version of the data
        """
        if isinstance(data, np.integer):
            return int(data)
        elif isinstance(data, np.floating):
            return float(data)
        elif isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, dict):
            return {k: self._prepare_for_serialization(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._prepare_for_serialization(item) for item in data]
        elif isinstance(data, tuple):
            return tuple(self._prepare_for_serialization(item) for item in data)
        elif isinstance(data, set):
            return {self._prepare_for_serialization(item) for item in data}
        else:
            return data
    
    def _reevaluate_false_positives(self, doc_id: str):
        """
        Re-evaluate false positives using the RAGRDMatcher.
        
        Args:
            doc_id: Document ID to process
        """
        print(f"\nRe-evaluating false positives for document {doc_id}...")
        
        false_positives = self.results[doc_id]['old_false_positives']
        if not false_positives:
            print("  No false positives to re-evaluate")
            return
        
        print(f"  Found {len(false_positives)} false positives to re-evaluate")
        
        for fp in false_positives:
            entity = fp['entity']
            print(f"\n  Re-evaluating: {entity}")
            
            # Make sure we're using a valid matcher with an initialized index
            if self.rd_matcher.index is None:
                print("  Warning: Matcher index not initialized, initializing now...")
                self.rd_matcher.prepare_index(self.embedded_documents)
            
            try:
                # Always re-retrieve candidates to ensure we have fresh data
                candidates = self.rd_matcher._retrieve_candidates(entity)
                
                # Convert NumPy types to native Python types for JSON serialization
                fp['top_candidates'] = [
                    {
                        'name': c['metadata']['name'],
                        'id': c['metadata']['id'],
                        'similarity': float(c['similarity_score'])  # Convert to Python float
                    }
                    for c in candidates[:5]
                ]
                
                # Re-verify if it's a rare disease
                is_rare_disease = self.rd_matcher._verify_rare_disease(entity, candidates[:5])
                
                if is_rare_disease:
                    print(f"  ✓ LLM confirms '{entity}' is a rare disease")
                    
                    # Try to match it to ORPHA ID
                    if not fp['orpha_id'] or not fp['rd_term']:
                        rd_term = self.rd_matcher._try_llm_match(entity, candidates[:5])
                        if rd_term:
                            fp['orpha_id'] = rd_term['id']
                            fp['rd_term'] = rd_term['name']
                            print(f"    Matched to {rd_term['name']} ({rd_term['id']})")
                    
                    # Move to new true positives
                    self.results[doc_id]['new_true_positives'].append(fp)
                else:
                    print(f"  ✗ LLM confirms '{entity}' is NOT a rare disease")
                    # Keep as false positive
                    self.results[doc_id]['new_false_positives'].append(fp)
                    
            except Exception as e:
                print(f"  Error processing '{entity}': {str(e)}")
                # Keep as false positive in case of error
                self.results[doc_id]['new_false_positives'].append(fp)
    
    def _calculate_metrics(self) -> Dict[str, Any]:
        """
        Calculate evaluation metrics with and without supervision.
        
        Returns:
            Dict with metrics
        """
        # Initialize counters for original metrics
        old_entity_tp = 0
        old_entity_fp = 0
        entity_fn = 0
        
        # Initialize counters for supervised metrics
        new_entity_tp = 0
        new_entity_fp = 0
        
        # Process all documents
        for doc_id, doc_results in self.results.items():
            # Count old metrics
            old_entity_tp += len(doc_results['old_true_positives'])
            old_entity_fp += len(doc_results['old_false_positives'])
            entity_fn += len(doc_results['false_negatives'])
            
            # Count new metrics
            new_entity_tp += len(doc_results['old_true_positives']) + len(doc_results['new_true_positives'])
            new_entity_fp += len(doc_results['new_false_positives'])
        
        # Calculate original metrics
        old_metrics = self._calculate_metric_values(old_entity_tp, old_entity_fp, entity_fn)
        
        # Calculate supervised metrics
        new_metrics = self._calculate_metric_values(new_entity_tp, new_entity_fp, entity_fn)
        
        # Calculate improvement
        precision_improvement = new_metrics['precision'] - old_metrics['precision']
        recall_improvement = new_metrics['recall'] - old_metrics['recall']
        f1_improvement = new_metrics['f1'] - old_metrics['f1']
        
        return {
            'original': old_metrics,
            'supervised': new_metrics,
            'improvement': {
                'precision': precision_improvement,
                'recall': recall_improvement,
                'f1': f1_improvement,
                'reclassified_count': sum(len(doc_results['new_true_positives']) for doc_results in self.results.values())
            }
        }
    
    def _calculate_metric_values(self, tp, fp, fn) -> Dict[str, float]:
        """Helper function to calculate precision, recall, and F1 score."""
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn
        }
    
    def print_summary(self):
        """Print a summary of the supervised evaluation results."""
        metrics = self._calculate_metrics()
        
        print("\n===== Supervised Evaluation Summary =====")
        
        # Original metrics
        print("\nOriginal Metrics:")
        print(f"  Precision: {metrics['original']['precision']:.3f}")
        print(f"  Recall: {metrics['original']['recall']:.3f}")
        print(f"  F1 Score: {metrics['original']['f1']:.3f}")
        print(f"  True Positives: {metrics['original']['tp']}")
        print(f"  False Positives: {metrics['original']['fp']}")
        print(f"  False Negatives: {metrics['original']['fn']}")
        
        # Supervised metrics
        print("\nSupervised Metrics:")
        print(f"  Precision: {metrics['supervised']['precision']:.3f}")
        print(f"  Recall: {metrics['supervised']['recall']:.3f}")
        print(f"  F1 Score: {metrics['supervised']['f1']:.3f}")
        print(f"  True Positives: {metrics['supervised']['tp']}")
        print(f"  False Positives: {metrics['supervised']['fp']}")
        print(f"  False Negatives: {metrics['supervised']['fn']}")
        
        # Improvement
        print("\nImprovement:")
        print(f"  Precision: {metrics['improvement']['precision']:.3f}")
        print(f"  Recall: {metrics['improvement']['recall']:.3f}")
        print(f"  F1 Score: {metrics['improvement']['f1']:.3f}")
        print(f"  Reclassified False Positives: {metrics['improvement']['reclassified_count']}")
        
        # Document statistics
        total_docs = len(self.results)
        docs_with_reclassification = sum(1 for doc in self.results.values() if doc['new_true_positives'])
        
        print(f"\nDocuments with Reclassifications: {docs_with_reclassification}/{total_docs} "
              f"({docs_with_reclassification/total_docs*100:.1f}%)")
        
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
    parser = argparse.ArgumentParser(description='Extract rare disease mentions from clinical notes')
    
    # Add evaluate mode
    parser.add_argument('--evaluate', action='store_true',
                       help='Run only evaluation on existing intermediate results')
    
    # Add supervised evaluation mode
    parser.add_argument('--supervised', action='store_true',
                       help='Run supervised evaluation on false positives')
    parser.add_argument('--supervised_output', type=str,
                       help='Path for supervised evaluation results (JSON)')

    # Model configuration (only needed for non-evaluate mode)
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
    
    # GPU configuration (only needed for non-evaluate mode)
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
    # Retriever configuration (only needed for non-evaluate mode)
    retriever_group = parser.add_argument_group('Retriever Configuration')
    retriever_group.add_argument('--retriever', type=str,
                       choices=['fastembed', 'medcpt', 'sentence_transformer'],
                       default='fastembed',
                       help='Type of retriever to use')
    retriever_group.add_argument('--retriever_model', type=str,
                       default="BAAI/bge-small-en-v1.5",
                       help='Model name for retriever')
    
    # API configuration (only needed for non-evaluate mode)
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
    
    # Processing configuration (only needed for non-evaluate mode)
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
    
    # Supervised mode requires evaluate mode
    if args.supervised and not args.evaluate:
        parser.error("--supervised requires --evaluate mode")
        
    # Supervised mode requires embedded_documents for re-evaluation
    if args.supervised and not args.embedded_documents:
        parser.error("--supervised requires --embedded_documents for candidate retrieval")
    
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
        if args.evaluate:
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
                
        # Process full pipeline mode
        else:
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
            
            # Save final results
            print(f"\nSaving final results to: {args.output_file}")
            results_df.to_csv(args.output_file, index=False)
        
        # Print final summary
        end_time = datetime.now()
        print("\nProcessing Summary:")
        print(f"Started at:  {start_time}")
        print(f"Finished at: {end_time}")
        print(f"Total time:  {end_time - start_time}")
        print("Processing completed successfully")

        # Add this supervised evaluation section inside the main() function,
    # specifically in the evaluate-only mode branch where regular evaluation is performed.
    # This should be inserted after the regular evaluation metrics are printed
    # and before the args.evaluation_file check.

    # Run supervised evaluation if requested
        if args.supervised:
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
    except Exception as e:
        end_time = datetime.now()
        print(f"\nError occurred after running for: {end_time - start_time}")
        print(f"Error: {str(e)}")
        traceback.print_exc()
        sys.exit(1)
        
if __name__ == "__main__":
    main()