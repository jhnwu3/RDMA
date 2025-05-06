from abc import ABC, abstractmethod
import json
import pandas as pd
import numpy as np
import re
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from hporag.entity import BaseEntityExtractor
from hporag.hpo_match import BaseHPOMatcher
from abc import ABC, abstractmethod
import os
import json
import pandas as pd
import numpy as np
import re
import time
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from rdma.hporag.entity import BaseEntityExtractor, LLMEntityExtractor
from rdma.hporag.hpo_match import BaseHPOMatcher

class HPORAG:
    """Main pipeline class combining entity extraction and HPO matching with timing functionality."""
    
    def __init__(self, 
                 entity_extractor: BaseEntityExtractor,
                 hpo_matcher: BaseHPOMatcher,
                 debug: bool = False):
        self.entity_extractor = entity_extractor
        self.hpo_matcher = hpo_matcher
        self.debug = debug
        # Add timing metrics storage
        self.timing_metrics = {
            'extraction_times': [],
            'matching_times': [],
            'context_extraction_times': [],
            'total_times': []
        }
        
    def _debug_print(self, message: str, level: int = 0):
        """Print debug message with timestamp and indentation."""
        if self.debug:
            indent = "  " * level
            print(f"{datetime.now().strftime('%H:%M:%S')} | {indent}{message}")

    def _debug_entity_extraction_llm(self, text: str):
        """Debug entity extraction process for LLM-based extractor."""
        self._debug_print("Sending to LLM:", level=2)
        self._debug_print(f"System message: {self.entity_extractor.system_message}", level=3)
        self._debug_print(f"User input: {text}", level=3)
        
        findings_text = self.entity_extractor.llm_client.query(text, self.entity_extractor.system_message)
        self._debug_print("\nLLM Response:", level=2)
        self._debug_print(findings_text, level=3)
        
        findings = self.entity_extractor._extract_findings_from_response(findings_text)
        self._debug_print("\nExtracted Entities:", level=2)
        for i, finding in enumerate(findings, 1):
            self._debug_print(f"{i}. {finding}", level=3)
        
        return findings

    def _debug_entity_extraction_stanza(self, text: str):
        """Debug entity extraction process for Stanza-based extractor."""
        self._debug_print("Processing with Stanza:", level=2)
        self._debug_print(f"Input text: {text}", level=3)
        
        findings = self.entity_extractor.extract_entities(text)
        self._debug_print("\nExtracted Entities:", level=2)
        for i, finding in enumerate(findings, 1):
            self._debug_print(f"{i}. {finding}", level=3)
        
        return findings

    def process_dataframe(self, df: pd.DataFrame,
                         embedded_documents: List[Dict],
                         text_column: str = 'clinical_note',
                         id_column: str = 'patient_id',
                         batch_size: int = 1) -> pd.DataFrame:
        """Process DataFrame containing clinical notes with timing for each step."""
        self._debug_print("Starting pipeline processing")
        self._debug_print(f"Processing {len(df)} clinical notes...")
        
        results = []
        total_start_time = time.time()
        
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i + batch_size]
            batch_start_time = time.time()
            
            # For debug mode, show detailed process for first sample
            if self.debug and i == 0:
                self._debug_print(f"\nDEBUG SAMPLE (patient {batch.iloc[0][id_column]}):")
                self._debug_print(f"Clinical note: {batch.iloc[0][text_column]}", level=1)
                
                self._debug_print("\nSTEP 1: Entity Extraction", level=1)
                extraction_start = time.time()
                # Choose appropriate debug method based on extractor type
                if isinstance(self.entity_extractor, LLMEntityExtractor):
                    sample_entities = self._debug_entity_extraction_llm(batch.iloc[0][text_column])
                else:
                    sample_entities = self._debug_entity_extraction_stanza(batch.iloc[0][text_column])
                extraction_time = time.time() - extraction_start
                self.timing_metrics['extraction_times'].append(extraction_time)
                self._debug_print(f"Entity extraction took {extraction_time:.2f} seconds", level=1)
                
                self._debug_print("\nSTEP 2: HPO Matching", level=1)
                matching_start = time.time()
                matches = self._debug_hpo_matching(sample_entities, [embedded_documents])
                matching_time = time.time() - matching_start
                self.timing_metrics['matching_times'].append(matching_time)
                self._debug_print(f"HPO matching took {matching_time:.2f} seconds", level=1)
                
                # Store debug sample results with full match information
                if matches:
                    results.append({
                        'patient_id': batch.iloc[0][id_column],
                        'matches': matches,
                        'original_text': batch.iloc[0][text_column],
                        'debug_output': True,  # Flag to identify debug samples
                        'extraction_time': extraction_time,
                        'matching_time': matching_time,
                        'total_time': extraction_time + matching_time
                    })
                    
                # Process remaining samples in first batch if batch_size > 1
                if batch_size > 1:
                    self._process_batch_with_timing(batch.iloc[1:], embedded_documents, results, text_column, id_column)
            else:
                self._process_batch_with_timing(batch, embedded_documents, results, text_column, id_column)
                
            batch_time = time.time() - batch_start_time
            self._debug_print(f"Batch processing took {batch_time:.2f} seconds")
        
        total_time = time.time() - total_start_time
        self.timing_metrics['total_times'].append(total_time)
        self._debug_print(f"Total processing time: {total_time:.2f} seconds")
        
        # Convert results to DataFrame with match details
        results_df = pd.DataFrame(results)
        if not results_df.empty:
            # Add timing summary to dataframe metadata
            results_df.attrs['timing_summary'] = self._calculate_timing_summary()
            
            # Explode matches into separate rows
            results_df = results_df.explode('matches').reset_index(drop=True)
            
            # Extract all fields from the matches dictionary
            match_details = pd.json_normalize(results_df['matches'])
            
            # Add all columns from match_details to results_df
            for column in match_details.columns:
                results_df[column] = match_details[column]
        
        self._debug_print(f"Processed {len(results_df)} matches")
        return results_df

    def _calculate_timing_summary(self) -> Dict[str, Any]:
        """Calculate summary statistics for timing metrics."""
        summary = {}
        
        for metric_name, values in self.timing_metrics.items():
            if values:
                summary[metric_name] = {
                    'mean': np.mean(values),
                    'median': np.median(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'total': np.sum(values)
                }
        
        # Calculate percentages
        if 'extraction_times' in summary and 'matching_times' in summary:
            total_extraction = summary['extraction_times']['total']
            total_matching = summary['matching_times']['total']
            total_context = summary.get('context_extraction_times', {}).get('total', 0)
            grand_total = total_extraction + total_matching + total_context
            
            if grand_total > 0:
                summary['percentages'] = {
                    'extraction': (total_extraction / grand_total) * 100,
                    'matching': (total_matching / grand_total) * 100,
                    'context_extraction': (total_context / grand_total) * 100
                }
                
        return summary
        
    def _debug_hpo_matching(self, entities: List[str], metadata: List[Dict]):
        """Debug HPO matching process."""
        if not entities:
            self._debug_print("No entities to match", level=2)
            return []
            
        # Initialize the HPO matcher's index first
        if self.hpo_matcher.index is None:
            self._debug_print("Initializing HPO matcher index...", level=2)
            self.hpo_matcher.prepare_index(metadata[0])
            
        self._debug_print(f"Processing {len(entities)} entities:", level=2)
        matches = []
        
        for entity in entities:
            # Get candidate terms using vector retrieval
            candidates = self.hpo_matcher._retrieve_candidates(entity)
            match_info = {
                'entity': entity,
                'top_candidates': candidates[:5]
            }
            
            # Try enriched exact matching process
            hpo_term = self.hpo_matcher._try_exact_match(entity, candidates)
            self._debug_print(f"Enriched matching result: {hpo_term}", level=3)
            if hpo_term:
                match_info.update({
                    'hpo_term': hpo_term,
                    'match_method': 'exact',
                    'confidence_score': 1.0
                })
                matches.append(match_info)
                continue
            
            # If no exact match found, try LLM matching
            if hasattr(self.hpo_matcher, 'llm_client') and self.hpo_matcher.llm_client:
                hpo_term = self.hpo_matcher._try_llm_match(entity, candidates)
                self._debug_print(f"LLM matching result: {hpo_term}", level=3)
                if hpo_term:
                    match_info.update({
                        'hpo_term': hpo_term,
                        'match_method': 'llm',
                        'confidence_score': 0.7
                    })
                    matches.append(match_info)
                    
        return matches

    def _extract_sentences(self, text: str) -> List[str]:
        """Split clinical text into sentences."""
        sentences = []
        for part in re.split(r'([.!?])', text):
            if part.strip():
                if part in '.!?':
                    if sentences:
                        sentences[-1] += part
                else:
                    sentences.append(part.strip())
        return sentences

    def _find_entity_context(self, entity: str, sentences: List[str]) -> Optional[str]:
        """Find the sentence containing the given entity."""
        entity_lower = entity.lower()
        
        # Try exact matching first
        for sentence in sentences:
            if entity_lower in sentence.lower():
                return sentence.strip()
        
        # If no exact match, try fuzzy matching
        entity_words = set(re.findall(r'\b\w+\b', entity_lower))
        best_match = None
        best_score = 0
        
        for sentence in sentences:
            sentence_words = set(re.findall(r'\b\w+\b', sentence.lower()))
            common_words = entity_words & sentence_words
            score = len(common_words) / len(entity_words) if entity_words else 0
            
            if score > best_score:
                best_score = score
                best_match = sentence
        
        return best_match.strip() if best_score > 0.5 else None

    def _extract_contexts(self, batch_entities: List[List[str]], texts: List[str]) -> List[List[str]]:
        """Extract original contexts for batched entities with timing."""
        context_start_time = time.time()
        batch_contexts = []
        
        for entities, text in zip(batch_entities, texts):
            # Get sentences for this text once
            sentences = self._extract_sentences(text)
            # Find context for each entity in this batch
            contexts = [self._find_entity_context(entity, sentences) for entity in entities]
            batch_contexts.append(contexts)
        
        context_time = time.time() - context_start_time
        self.timing_metrics['context_extraction_times'].append(context_time)
        self._debug_print(f"Context extraction took {context_time:.2f} seconds")
        
        return batch_contexts

    def _process_batch_with_timing(self, batch, embedded_documents, results, text_column, id_column):
        """Process a batch of samples with timing information."""
        texts = batch[text_column].tolist()
        
        # Extract entities using existing entity extractor
        extraction_start_time = time.time()
        batch_entities = self.entity_extractor.process_batch(texts)
        extraction_time = time.time() - extraction_start_time
        self.timing_metrics['extraction_times'].append(extraction_time)
        self._debug_print(f"Entity extraction batch took {extraction_time:.2f} seconds")
        
        # Extract contexts for all entities
        context_start_time = time.time()
        batch_contexts = self._extract_contexts(batch_entities, texts)
        context_time = time.time() - context_start_time
        self.timing_metrics['context_extraction_times'].append(context_time)
        
        # Process matches with contexts
        matching_start_time = time.time()
        batch_matches = []
        for entities, contexts in zip(batch_entities, batch_contexts):
            # Pass embedded_documents directly, not as a list
            matches = self.hpo_matcher.match_hpo_terms(entities, embedded_documents, contexts)
            batch_matches.append(matches)
        matching_time = time.time() - matching_start_time
        self.timing_metrics['matching_times'].append(matching_time)
        self._debug_print(f"HPO matching batch took {matching_time:.2f} seconds")
        
        # Store results with timing information
        for j, matches in enumerate(batch_matches):
            if matches:  # matches is a list of dictionaries with full match info
                results.append({
                    'patient_id': batch.iloc[j][id_column],
                    'matches': matches,
                    'original_text': batch.iloc[j][text_column],
                    'extraction_time': extraction_time / len(batch),  # Per-record time
                    'matching_time': matching_time / len(batch),      # Per-record time
                    'context_time': context_time / len(batch),        # Per-record time
                    'total_time': (extraction_time + matching_time + context_time) / len(batch)
                })
    
    def save_results(self, df: pd.DataFrame, output_path: str) -> None:
        """Save results to CSV file and include timing summary in a separate report."""
        df.to_csv(output_path, index=False)
        self._debug_print(f"Results saved to {output_path}")
        
        # Create timing report file
        timing_report_path = f"{os.path.splitext(output_path)[0]}_timing_report.json"
        with open(timing_report_path, 'w') as f:
            json.dump(self._calculate_timing_summary(), f, indent=2)
        self._debug_print(f"Timing report saved to {timing_report_path}")
        
        # Print summary timing report
        self._print_timing_summary()
    
    def _print_timing_summary(self):
        """Print a formatted summary of timing metrics."""
        summary = self._calculate_timing_summary()
        
        print("\n" + "="*60)
        print("PIPELINE PERFORMANCE METRICS")
        print("="*60)
        
        # Print time statistics for each phase
        for phase in ['extraction_times', 'matching_times', 'context_extraction_times', 'total_times']:
            if phase in summary:
                phase_name = phase.replace('_times', '').capitalize()
                print(f"\n{phase_name} Phase:")
                print(f"  Average: {summary[phase]['mean']:.2f} seconds")
                print(f"  Median:  {summary[phase]['median']:.2f} seconds")
                print(f"  Min:     {summary[phase]['min']:.2f} seconds")
                print(f"  Max:     {summary[phase]['max']:.2f} seconds")
                print(f"  Total:   {summary[phase]['total']:.2f} seconds")
                
        # Print percentage breakdowns
        if 'percentages' in summary:
            print("\nPercentage of Total Processing Time:")
            print(f"  Entity Extraction:   {summary['percentages']['extraction']:.1f}%")
            print(f"  HPO Term Matching:   {summary['percentages']['matching']:.1f}%")
            print(f"  Context Extraction:  {summary['percentages']['context_extraction']:.1f}%")
        
        print("="*60)