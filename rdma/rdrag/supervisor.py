import pandas as pd
import numpy as np
import json
import os
from typing import Dict, Any, List, Optional, Union, Tuple
from rdma.rdrag.pipeline import RDPipeline
from rdma.rdrag.entity import LLMRDExtractor
from rdma.rdrag.rd_match import RAGRDMatcher
from rdma.utils.embedding import EmbeddingsManager
from rdma.utils.llm_client import APILLMClient, LocalLLMClient
from rdma.utils.data import NumpyJSONEncoder

# All Defunct for Now, Not-tested
class IterativeSupervisor:
    """Iteratively improve gold standard annotations by validating false positives."""
    
    def __init__(
        self, 
        gold_df: pd.DataFrame, 
        rd_matcher: RAGRDMatcher,
        embedded_documents: List[Dict],
        max_iterations: int = 3,
        output_file: str = None,
        debug: bool = False
    ):
        """
        Initialize the iterative supervisor.
        
        Args:
            gold_df: DataFrame with gold standard annotations
            rd_matcher: Rare disease matcher for verification
            embedded_documents: Embedded rare disease documents
            max_iterations: Maximum number of iterations to run
            output_file: Path to save enhanced gold standard
            debug: Enable detailed debug output
        """
        self.gold_df = gold_df.copy()  # Start with original gold standard
        self.rd_matcher = rd_matcher
        self.embedded_documents = embedded_documents
        self.max_iterations = max_iterations
        self.output_file = output_file
        self.debug = debug
        
        # Ensure consistent document_id types (convert to string)
        self.gold_df['document_id'] = self.gold_df['document_id'].astype(str)
        
        # Initialize index if needed
        print("Ensuring matcher's search index is initialized...")
        if not hasattr(self.rd_matcher, 'index') or self.rd_matcher.index is None:
            print("Matcher index is None, initializing...")
            self.rd_matcher.prepare_index(self.embedded_documents)
        else:
            print("Matcher index already initialized")
            
        # Track history of enhancements
        self.enhancement_history = []
    
    def run_iteration(self, predictions_df: pd.DataFrame, pipeline: RDPipeline) -> Tuple[pd.DataFrame, pd.DataFrame, bool]:
        """
        Run a single iteration of enhancement.
        
        Args:
            predictions_df: DataFrame with pipeline predictions
            pipeline: RD pipeline to use for new predictions
            
        Returns:
            Tuple of (updated gold DataFrame, new predictions DataFrame, whether changes were made)
        """
        # Copy gold_df to avoid modifying the original
        updated_gold_df = self.gold_df.copy()
        
        # Ensure consistent document_id types in predictions
        predictions_df = predictions_df.copy()
        predictions_df['document_id'] = predictions_df['document_id'].astype(str)
        
        # Track changes in this iteration
        all_new_annotations = []
        
        # Stats tracking
        total_docs = len(updated_gold_df)
        docs_processed = 0
        total_false_positives = 0
        entities_not_in_text = 0
        entities_not_rare = 0
        entities_validated = 0
        
        print(f"\nProcessing {total_docs} documents for potential gold standard enhancement...")
        
        # DEBUG: Print column names in predictions_df
        print(f"DEBUG: predictions_df columns: {predictions_df.columns.tolist()}")
        print(f"DEBUG: Sample prediction row: {predictions_df.iloc[0].to_dict()}")
        
        # For each document
        for _, row in updated_gold_df.iterrows():
            doc_id = row['document_id']
            gold_anns = row.get('gold_annotations', []) or []  # Ensure we have a list even if None
            original_text = row.get('clinical_note', '')
            
            docs_processed += 1
            if docs_processed % 10 == 0:
                print(f"Progress: {docs_processed}/{total_docs} documents processed")
            
            # Get predictions for this document
            doc_preds = predictions_df[predictions_df['document_id'] == doc_id]
            
            if doc_preds.empty:
                if self.debug:
                    print(f"No predictions found for document {doc_id}")
                continue
            
            # DEBUG: Print what we found
            print(f"\nDEBUG: Document {doc_id} has {len(doc_preds)} predictions and {len(gold_anns)} gold annotations")
                
            # Find false positives (predicted entities not in gold standard)
            gold_entities = {ann['mention'].lower() for ann in gold_anns}
            
            # Extract predicted entities
            pred_entities = []
            for _, pred_row in doc_preds.iterrows():
                # Check different possible column names for entity
                if 'entity' in pred_row and pd.notna(pred_row['entity']):
                    entity = pred_row['entity'].lower()
                elif 'mention' in pred_row and pd.notna(pred_row['mention']):
                    entity = pred_row['mention'].lower()
                else:
                    # Try to find any column that might contain entity information
                    entity_cols = [col for col in pred_row.index if any(x in col.lower() for x in ['entity', 'mention', 'term'])]
                    if entity_cols and pd.notna(pred_row[entity_cols[0]]):
                        entity = pred_row[entity_cols[0]].lower()
                        print(f"Using alternative column for entity: {entity_cols[0]}")
                    else:
                        continue  # Skip if no entity information found
                
                if entity and entity not in gold_entities:
                    pred_entities.append((entity, pred_row))
            
            # DEBUG: Show entities found
            if pred_entities:
                print(f"DEBUG: Found {len(pred_entities)} potential false positives in document {doc_id}")
                print(f"DEBUG: Gold entities: {gold_entities}")
                print(f"DEBUG: Predicted false positive entities: {[e[0] for e in pred_entities]}")
            
            doc_new_annotations = []
            doc_fps = 0
            
            # Process each potential false positive
            for entity, pred_row in pred_entities:
                doc_fps += 1
                total_false_positives += 1
                
                # This is a false positive - first verify it actually exists in the text
                if not self._entity_exists_in_text(entity, original_text):
                    if self.debug or True:  # Always show this for debugging
                        print(f"✗ '{entity}' not found in original text of document {doc_id} - skipping")
                    entities_not_in_text += 1
                    continue
                
                print(f"\nVerifying potential false positive: '{entity}' in document {doc_id}")
                
                # Get candidates from the index - make sure we have valid candidates
                try:
                    # Double-check that index is initialized
                    if self.rd_matcher.index is None:
                        print("WARNING: Matcher index is None, re-initializing...")
                        self.rd_matcher.prepare_index(self.embedded_documents)
                    
                    candidates = self.rd_matcher._retrieve_candidates(entity)
                    if not candidates:
                        print(f"No candidates found for '{entity}' - skipping verification")
                        continue
                        
                    print(f"Retrieved {len(candidates)} candidates for '{entity}'")
                    if self.debug or True:  # Always show this for debugging
                        # Show top 3 candidates
                        for i, c in enumerate(candidates[:3]):
                            print(f"  {i+1}. {c['metadata']['name']} (Score: {c['similarity_score']:.3f})")
                    
                    # Use the matcher's verification method with less stringent criteria
                    is_rare_disease = self._verify_rare_disease_enhanced(entity, candidates[:5])
                    
                    if is_rare_disease:
                        print(f"✓ '{entity}' verified as a true rare disease - adding to gold standard")
                        entities_validated += 1
                        
                        # Get orpha ID and term if available
                        orpha_id = pred_row.get('orpha_id', '') if pd.notna(pred_row.get('orpha_id', '')) else ''
                        rd_term = pred_row.get('rd_term', '') if pd.notna(pred_row.get('rd_term', '')) else ''
                        
                        # If we don't have an orpha_id yet, try to match with LLM
                        if not orpha_id and self.rd_matcher.llm_client:
                            rd_term_dict = self.rd_matcher._try_llm_match(entity, candidates[:5])
                            if rd_term_dict:
                                orpha_id = rd_term_dict['id']
                                rd_term = rd_term_dict['name']
                                print(f"  Matched to {rd_term} ({orpha_id})")
                        
                        # Create new gold annotation
                        new_annotation = {
                            'mention': entity,
                            'orpha_id': orpha_id,
                            'orpha_desc': rd_term,
                            'document_section': '',  # Not available from predictions
                            'confidence': 0.9,  # High but not perfect confidence
                            'added_by': 'iterative_supervision'
                        }
                        
                        # Add to document-specific and all annotations lists
                        doc_new_annotations.append(new_annotation)
                        
                        # Record for history
                        all_new_annotations.append({
                            'document_id': doc_id,
                            'annotation': new_annotation
                        })
                    else:
                        print(f"✗ '{entity}' confirmed as NOT a rare disease")
                        entities_not_rare += 1
                        
                except Exception as e:
                    print(f"Error verifying entity '{entity}': {str(e)}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            # Add all new annotations for this document to the gold standard
            if doc_new_annotations:
                print(f"Added {len(doc_new_annotations)} new annotations to document {doc_id}")
                # Need to access the original list in the dataframe
                idx = updated_gold_df.index[updated_gold_df['document_id'] == doc_id].tolist()[0]
                updated_gold_anns = gold_anns.copy() if gold_anns else []
                updated_gold_anns.extend(doc_new_annotations)
                updated_gold_df.at[idx, 'gold_annotations'] = updated_gold_anns
        
        # Print summary of processing
        print("\n===== Iteration Processing Summary =====")
        print(f"Documents processed: {docs_processed}/{total_docs}")
        print(f"Total false positives found: {total_false_positives}")
        print(f"Entities not found in text: {entities_not_in_text}")
        print(f"Entities verified as not rare: {entities_not_rare}")
        print(f"Entities validated as rare diseases: {entities_validated}")
        print(f"Total new annotations added: {len(all_new_annotations)}")
        
        # Check if changes were made
        changes_made = len(all_new_annotations) > 0
        
        # Add to history
        if changes_made:
            current_total = sum(len(row.get('gold_annotations', []) or []) for _, row in updated_gold_df.iterrows())
            self.enhancement_history.append({
                'iteration': len(self.enhancement_history) + 1,
                'new_annotations': all_new_annotations,
                'total_gold_annotations': current_total,
                'stats': {
                    'documents_processed': docs_processed,
                    'false_positives': total_false_positives,
                    'entities_not_in_text': entities_not_in_text,
                    'entities_not_rare': entities_not_rare,
                    'entities_validated': entities_validated
                }
            })
            
            # Full pipeline rerun with updated gold standard
            print("\n==== FULL PIPELINE RERUN ====")
            print("Running complete pipeline with enhanced gold standard...")
            print(f"Processing {len(updated_gold_df)} documents...")
            
            # Run the entire pipeline from scratch with updated gold standard
            new_predictions_df = pipeline.process_dataframe(
                df=updated_gold_df,
                embedded_documents=self.embedded_documents,
                text_column='clinical_note',
                id_column='document_id'
            )
            
            print("Pipeline rerun completed")
            print("================================")
        else:
            print("\nNo changes made to gold standard - skipping pipeline rerun")
            new_predictions_df = predictions_df
        
        return updated_gold_df, new_predictions_df, changes_made
    
    def _verify_rare_disease_enhanced(self, term: str, candidates: List[Dict]) -> bool:
        """
        Enhanced verification with more flexible criteria for rare disease determination.
        
        Args:
            term: Term to verify
            candidates: Candidate matches
            
        Returns:
            True if verified as rare disease, False otherwise
        """
        if not self.rd_matcher.llm_client:
            return True  # If no LLM client, assume all terms are valid
            
        # Format candidate context
        context = "\nPotential matches from Orphanet rare disease database:\n" + "\n".join([
            f"{i+1}. {candidate['metadata']['name']} ({candidate['metadata']['id']})"
            for i, candidate in enumerate(candidates[:5])
        ])
        
        prompt = f"""Analyze this medical term and determine if it likely represents a rare disease.

Term: {term}
{context}

Consider the following when making your decision:
1. The term should represent a disease or syndrome, not just a symptom or finding
2. It doesn't need to explicitly mention rarity - the database context shows it's from a rare disease database
3. The term should be semantically similar to entries in the Orphanet database (shown in context)
4. Some medical terms might be variants or synonyms of known rare diseases

Response format:
First line: "DECISION: true" or "DECISION: false"
Next lines: Brief explanation of decision"""

        print(f"Sending verification prompt to LLM for term: '{term}'")
        response = self.rd_matcher.llm_client.query(prompt, self.rd_matcher.system_message).strip().lower()
        result = "decision: true" in response.lower()
        print(f"LLM verification result: {result}")
        if self.debug:
            print(f"LLM response: {response}")
        return result
    
    def _entity_exists_in_text(self, entity: str, text: str) -> bool:
        """
        Check if an entity exists in the original text with improved matching.
        
        Args:
            entity: Entity text to search for
            text: Original text to search in
            
        Returns:
            True if entity is found in text, False otherwise
        """
        if not text or not entity:
            return False
            
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
            
        # Try various entity transformations
        # Check for plurals/singulars
        if entity_normalized.endswith('s') and entity_normalized[:-1] in text_normalized:
            return True
        if not entity_normalized.endswith('s') and f"{entity_normalized}s" in text_normalized:
            return True
            
        # Check for hyphenated vs non-hyphenated versions
        if '-' in entity_normalized:
            non_hyphenated = entity_normalized.replace('-', ' ')
            if non_hyphenated in text_normalized:
                return True
        elif ' ' in entity_normalized:
            hyphenated = entity_normalized.replace(' ', '-')
            if hyphenated in text_normalized:
                return True
            
        return False
    
    def run_iterations(self, pipeline: RDPipeline, initial_predictions_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run multiple iterations of the enhancement process.
        
        Args:
            pipeline: The RD pipeline to use for predictions
            initial_predictions_df: Initial predictions from the first run
            
        Returns:
            Tuple of (enhanced gold standard DataFrame, final predictions DataFrame)
        """
        current_gold_df = self.gold_df.copy()
        current_predictions_df = initial_predictions_df.copy()
        
        print("\n===== Starting Iterative Supervision =====")
        initial_count = sum(len(row.get('gold_annotations', [])) for _, row in current_gold_df.iterrows())
        print(f"Initial gold standard has {initial_count} annotations")
        
        # Initial evaluation
        print("\nInitial evaluation:")
        initial_metrics = evaluate_predictions(current_predictions_df, current_gold_df)
        print(f"Initial metrics - Precision: {initial_metrics['precision']:.3f}, "
              f"Recall: {initial_metrics['recall']:.3f}, F1: {initial_metrics['f1']:.3f}")
        
        # Tracking metrics history
        metrics_history = [initial_metrics]
        
        for iteration in range(1, self.max_iterations + 1):
            print(f"\n----- Iteration {iteration}/{self.max_iterations} -----")
            
            # Run iteration to enhance gold standard and get new predictions
            updated_gold_df, new_predictions_df, changes_made = self.run_iteration(
                current_predictions_df, pipeline
            )
            
            if not changes_made:
                print("\nNo changes made to gold standard. Stopping iterations.")
                break
                
            # Run evaluation with new predictions
            print("\nEvaluating with enhanced gold standard:")
            metrics = evaluate_predictions(new_predictions_df, updated_gold_df)
            metrics_history.append(metrics)
            
            print(f"Updated metrics - Precision: {metrics['precision']:.3f}, "
                  f"Recall: {metrics['recall']:.3f}, F1: {metrics['f1']:.3f}")
            
            # Update for next iteration
            current_gold_df = updated_gold_df
            current_predictions_df = new_predictions_df
            self.gold_df = current_gold_df  # Update internal reference
            
            # Print summary of changes
            latest_history = self.enhancement_history[-1]
            print(f"Added {len(latest_history['new_annotations'])} new annotations")
            print(f"Total gold annotations: {latest_history['total_gold_annotations']}")
        
        # Save final enhanced gold standard if output file specified
        if self.output_file:
            self._save_enhanced_gold_standard(current_gold_df, metrics_history)
        
        return current_gold_df, current_predictions_df
    
    def _save_enhanced_gold_standard(self, gold_df: pd.DataFrame, metrics_history: List[Dict]):
        """Save the enhanced gold standard to a file."""
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        
        # Prepare result dictionary
        result = {
            'documents': [],
            'enhancement_history': self.enhancement_history,
            'metrics_history': metrics_history
        }
        
        # Convert each document to a dictionary
        for _, row in gold_df.iterrows():
            doc = {
                'document_id': row['document_id'],
                'patient_id': row.get('patient_id', ''),
                'admission_id': row.get('admission_id', ''),
                'category': row.get('category', ''),
                'chart_date': row.get('chart_date', ''),
                'gold_annotations': row.get('gold_annotations', [])
            }
            result['documents'].append(doc)
        
        # Save as JSON
        try:
            with open(self.output_file, 'w') as f:
                json.dump(result, f, indent=2, cls=NumpyJSONEncoder)
            print(f"\nEnhanced gold standard saved to {self.output_file}")
        except Exception as e:
            print(f"Error saving enhanced gold standard: {str(e)}")
            
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
        



