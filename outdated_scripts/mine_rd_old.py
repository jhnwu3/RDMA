import json
from typing import Dict, List, Set
from collections import defaultdict
import torch
import numpy as np
from tqdm import tqdm
import argparse
from pathlib import Path
from rdrag.AutoRD import SimpleAutoRD, Entity
from utils.llm import ModelLoader

def load_mimic_annotations(file_path: str) -> Dict:
    """Load and parse the MIMIC annotations file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def normalize_mention(mention: str) -> str:
    """Normalize mention text for comparison."""
    return mention.lower().strip()

from typing import Dict, List, Set, Optional
from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



def validate_annotation(annotation: Dict) -> bool:
    """Validate that an annotation contains all required fields."""
    if not isinstance(annotation, dict):
        return False
    if 'annotations' not in annotation:
        return False
    for anno in annotation['annotations']:
        if not isinstance(anno, dict) or 'mention' not in anno:
            return False
    if 'note_details' not in annotation or 'text' not in annotation['note_details']:
        return False
    return True

def normalize_text(text: str, case_sensitive: bool = False) -> str:
    """Normalize text for comparison."""
    normalized = text.strip()
    if not case_sensitive:
        normalized = normalized.lower()
    return normalized

def evaluate_predictions(predictions: List[Entity], 
                        gold_annotations: Dict, 
                        case_sensitive: bool = False) -> Dict:
    """
    Evaluate AutoRD predictions against gold annotations using set-based comparison per document.
    
    Args:
        predictions: List of predicted entities from AutoRD
        gold_annotations: Dictionary of gold standard annotations by document ID
        case_sensitive: Whether to perform case-sensitive comparison
        
    Returns:
        Dict containing evaluation metrics and detailed statistics
    """
    doc_metrics = defaultdict(lambda: {
        'predicted_set': set(),
        'gold_set': set(),
        'true_positives': 0,
        'false_positives': 0,
        'false_negatives': 0
    })
    
    # Validate input data
    if not predictions:
        logger.warning("No predictions provided for evaluation")
    if not gold_annotations:
        logger.warning("No gold annotations provided for evaluation")
    
    # Group predictions by document ID
    pred_by_doc = defaultdict(list)
    for pred in predictions:
        if not isinstance(pred, Entity):
            logger.warning(f"Skipping invalid prediction: {pred}")
            continue
        if pred.entity_type == 'rare_disease' and pred.orpha_id is not None:
            normalized_phrase = normalize_text(pred.extracted_phrase, case_sensitive)
            pred_by_doc[pred.doc_id].append(normalized_phrase)
    
    # Process gold annotations by document
    for doc_id, doc in gold_annotations.items():
        try:
            if not validate_annotation(doc):
                logger.warning(f"Skipping invalid annotation document: {doc_id}")
                continue
                
            # Get gold mentions for this document
            gold_mentions = {
                normalize_text(anno['mention'], case_sensitive) 
                for anno in doc['annotations']
            }
            doc_metrics[doc_id]['gold_set'] = gold_mentions
            
            # Get predicted mentions for this document
            pred_mentions = set(pred_by_doc[doc_id])
            doc_metrics[doc_id]['predicted_set'] = pred_mentions
            
            # Calculate intersections
            true_positives = gold_mentions.intersection(pred_mentions)
            doc_metrics[doc_id]['true_positives'] = len(true_positives)
            doc_metrics[doc_id]['false_positives'] = len(pred_mentions - gold_mentions)
            doc_metrics[doc_id]['false_negatives'] = len(gold_mentions - pred_mentions)
            
        except Exception as e:
            logger.error(f"Error processing document {doc_id}: {str(e)}")
            continue
    
    # Aggregate metrics across all documents
    total_tp = sum(m['true_positives'] for m in doc_metrics.values())
    total_fp = sum(m['false_positives'] for m in doc_metrics.values())
    total_fn = sum(m['false_negatives'] for m in doc_metrics.values())
    
    # Calculate overall metrics
    try:
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    except ZeroDivisionError:
        logger.error("Error calculating metrics due to zero division")
        precision = recall = f1 = 0
    
    return {
        'overall_metrics': {
            'precision': precision,
            'recall': recall,
            'f1': f1
        },
        'overall_statistics': {
            'true_positives': total_tp,
            'false_positives': total_fp,
            'false_negatives': total_fn
        },
        'document_metrics': {
            doc_id: {
                'metrics': {
                    'precision': m['true_positives'] / (m['true_positives'] + m['false_positives']) 
                               if (m['true_positives'] + m['false_positives']) > 0 else 0,
                    'recall': m['true_positives'] / (m['true_positives'] + m['false_negatives'])
                             if (m['true_positives'] + m['false_negatives']) > 0 else 0
                },
                'statistics': {
                    'true_positives': m['true_positives'],
                    'false_positives': m['false_positives'],
                    'false_negatives': m['false_negatives']
                },
                'analysis': {
                    'correct_mentions': list(m['predicted_set'].intersection(m['gold_set'])),
                    'spurious_mentions': list(m['predicted_set'] - m['gold_set']),
                    'missed_mentions': list(m['gold_set'] - m['predicted_set'])
                }
            }
            for doc_id, m in doc_metrics.items()
        }
    }

def evaluate_single_document(predictions: List[Dict], gold_annotations: List[Dict], doc_id: str = "single_doc") -> Dict:
    """
    Evaluate predictions for a single document using set-based comparison.
    
    Args:
        predictions: List of predicted Entity objects from AutoRD
        gold_annotations: List of gold standard annotations for the document
        doc_id: Optional document identifier
        
    Returns:
        Dict containing evaluation metrics for the single document
    """
    # Get predicted mentions set
    pred_mentions = {
        pred.extracted_phrase.lower().strip() 
        for pred in predictions 
        if pred.entity_type == 'rare_disease' and pred.orpha_id is not None
    }
    
    # Get gold mentions set
    gold_mentions = {anno['mention'].lower().strip() for anno in gold_annotations}
    
    # Calculate set intersections
    true_positives = pred_mentions.intersection(gold_mentions)
    false_positives = pred_mentions - gold_mentions
    false_negatives = gold_mentions - pred_mentions
    
    # Calculate metrics
    tp_count = len(true_positives)
    fp_count = len(false_positives)
    fn_count = len(false_negatives)
    
    precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0
    recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'metrics': {
            'precision': precision,
            'recall': recall,
            'f1': f1
        },
        'statistics': {
            'true_positives': tp_count,
            'false_positives': fp_count,
            'false_negatives': fn_count
        },
        'analysis': {
            'correct_mentions': true_positives,
            'spurious_mentions': false_positives,
            'missed_mentions': false_negatives
        }
    }

def get_device(gpu_id: str, condor: bool) -> str:
    """Determine the device to use based on arguments."""
    if not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        return "cpu"
    
    if condor:
        print("Running in Condor mode, using 'cuda'")
        return "cuda"
    
    if gpu_id is not None:
        device = f"cuda:{gpu_id}"
        print(f"Using GPU {gpu_id}")
        return device
    
    print("No GPU specified, using 'cuda'")
    return "cuda"

def main():
    parser = argparse.ArgumentParser(description='Evaluate AutoRD on MIMIC annotations')
    # Add debug argument
    parser.add_argument('--debug', action='store_true',
                       help='Run in debug mode on a single document')
    parser.add_argument('--gpu_id', type=str, help='GPU ID to use (e.g., "0", "1")', default=None)
    parser.add_argument('--condor', action='store_true', help='Use generic "cuda" device for Condor compatibility')
    parser.add_argument('--model', type=str, choices=['llama3_8b', 'llama3_70b'], 
                       default='llama3_8b', help='Model to use for evaluation')
    parser.add_argument('--cache_dir', type=str, 
                       default="/shared/rsaas/jw3/rare_disease/model_cache",
                       help='Directory for model cache')
    parser.add_argument('--rare_disease_ontology', type=str,
                       default="data/ontology/rare_disease_ontology.jsonl",
                       help='Path to rare disease ontology file')
    parser.add_argument('--annotations_file', type=str,
                       default="/home/johnwu3/projects/rare_disease/workspace/repos/RareDiseaseMention/filtered_rd_annos.json",
                       help='Path to annotations file')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size for processing documents')
    parser.add_argument('--output_dir', type=str, 
                       default="results",
                       help='Directory to save evaluation results')
    parser.add_argument('--case_sensitive', action='store_true',
                       help='Perform case-sensitive comparison')
    
    args = parser.parse_args()
    
    # Create output directory
    try:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(f"Error creating output directory: {e}")
        return
    
    # Set up device
    try:
        device = get_device(args.gpu_id, args.condor)
    except Exception as e:
        logger.error(f"Error setting up device: {e}")
        return
    
    # Initialize model and pipeline
    try:
        logger.info(f"Initializing ModelLoader with cache directory: {args.cache_dir}")
        model_loader = ModelLoader(cache_dir=args.cache_dir)
        logger.info(f"Loading {args.model} model...")
        pipeline = model_loader.get_llm_pipeline(device, args.model)
    except Exception as e:
        logger.error(f"Error initializing model: {e}")
        return
    
    # Initialize AutoRD
    try:
        logger.info("Initializing AutoRD...")
        auto_rd = SimpleAutoRD(
            pipeline=pipeline,
            rare_disease_ontology_path=args.rare_disease_ontology
        )
    except Exception as e:
        logger.error(f"Error initializing AutoRD: {e}")
        return

    # Load annotations
    try:
        logger.info(f"Loading annotations from {args.annotations_file}...")
        with open(args.annotations_file, 'r') as f:
            annotations = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Error loading annotations file: {e}")
        return
    
    # Dictionary to store predictions by document ID
    predictions_by_doc = {}
    processed_count = 0
    error_count = 0
    
    # Process documents in batches
    logger.info("Processing documents...")
    doc_ids = list(annotations.keys())
    
    # If in debug mode, only process the first document
    if args.debug:
        logger.info("Running in debug mode - processing single document")
        doc_ids = [doc_ids[0]]  # Take just the first document
        args.batch_size = 1  # Force batch size to 1
    
    for i in tqdm(range(0, len(doc_ids), args.batch_size)):
        batch_ids = doc_ids[i:i + args.batch_size]
        for doc_id in batch_ids:
            try:
                doc = annotations[doc_id]
                if not validate_annotation(doc):
                    logger.warning(f"Skipping invalid annotation document: {doc_id}")
                    continue
                
                text = doc['note_details']['text']
                entities = auto_rd.process_text(text)
                
                # Add document ID to each entity
                for entity in entities:
                    entity.doc_id = doc_id
                predictions_by_doc[doc_id] = entities
                processed_count += 1
                
            except Exception as e:
                logger.error(f"Error processing document {doc_id}: {str(e)}")
                error_count += 1
                continue
    
    logger.info(f"Processed {processed_count} documents successfully")
    if error_count > 0:
        logger.warning(f"Encountered errors in {error_count} documents")
    
    # Flatten predictions for evaluation
    all_predictions = [
        entity 
        for doc_predictions in predictions_by_doc.values() 
        for entity in doc_predictions
    ]
    
    # Evaluate
    logger.info("Computing metrics...")
    evaluation_results = evaluate_predictions(
        all_predictions, 
        annotations,
        case_sensitive=args.case_sensitive
    )
    
    # Save results
    try:
        results = {
            'evaluation_results': evaluation_results,
            'args': vars(args),
            'device': str(device),
            'processing_stats': {
                'total_documents': len(doc_ids),
                'processed_successfully': processed_count,
                'errors': error_count
            }
        }
        
        results_file = output_dir / f"results_{args.model}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=lambda x: list(x) if isinstance(x, set) else str(x))
        logger.info(f"Results saved to: {results_file}")
        
    except Exception as e:
        logger.error(f"Error saving results: {e}")
        return
    
    # Print results
    if args.debug:
        print("\nDebug Mode Results (Single Document):")
        # Print document-specific results for the debug document
        doc_id = doc_ids[0]
        doc_metrics = evaluation_results['document_metrics'][doc_id]
        print(f"\nDocument ID: {doc_id}")
        print("\nDocument-level metrics:")
        print(f"Precision: {doc_metrics['metrics']['precision']:.3f}")
        print(f"Recall: {doc_metrics['metrics']['recall']:.3f}")
        print(f"\nDetailed Analysis:")
        print(f"Correct mentions: {doc_metrics['analysis']['correct_mentions']}")
        print(f"Spurious mentions: {doc_metrics['analysis']['spurious_mentions']}")
        print(f"Missed mentions: {doc_metrics['analysis']['missed_mentions']}")
    else:
        print("\nOverall Evaluation Results:")
    print(f"Model: {args.model}")
    print(f"Device: {device}")
    print(f"Precision: {evaluation_results['overall_metrics']['precision']:.3f}")
    print(f"Recall: {evaluation_results['overall_metrics']['recall']:.3f}")
    print(f"F1 Score: {evaluation_results['overall_metrics']['f1']:.3f}")
    print(f"\nDetailed Statistics:")
    print(f"True Positives: {evaluation_results['overall_statistics']['true_positives']}")
    print(f"False Positives: {evaluation_results['overall_statistics']['false_positives']}")
    print(f"False Negatives: {evaluation_results['overall_statistics']['false_negatives']}")
    print(f"\nProcessing Statistics:")

if __name__ == "__main__":
    main()