from rdma.rdrag.entity import BaseRDExtractor
from rdma.rdrag.rd_match import BaseRDMatcher
from abc import ABC, abstractmethod
import json
import pandas as pd
import numpy as np
import re
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from fuzzywuzzy import fuzz
class RDPipeline:
    """Main pipeline class combining entity extraction and rare disease matching."""
    
    def __init__(self, 
                 entity_extractor: BaseRDExtractor,
                 rd_matcher: BaseRDMatcher,
                 debug: bool = False):
        self.entity_extractor = entity_extractor
        self.rd_matcher = rd_matcher
        self.debug = debug
        
    def _debug_print(self, message: str, level: int = 0):
        """Print debug message with timestamp and indentation."""
        if self.debug:
            indent = "  " * level
            print(f"{datetime.now().strftime('%H:%M:%S')} | {indent}{message}")
            
    def process_dataframe(self, df: pd.DataFrame,
                         embedded_documents: List[Dict],
                         text_column: str = 'clinical_note',
                         id_column: str = 'document_id',
                         batch_size: int = 1) -> pd.DataFrame:
        """Process DataFrame containing clinical notes."""
        self._debug_print("Starting pipeline processing")
        self._debug_print(f"Processing {len(df)} clinical notes...")
        
        results = []
        
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i + batch_size]
            
            # Debug mode for first sample
            if self.debug and i == 0:
                self._debug_print(f"\nDEBUG SAMPLE (patient {batch.iloc[0][id_column]}):")
                self._debug_print(f"Clinical note: {batch.iloc[0][text_column]}", level=1)
                
                self._debug_print("\nSTEP 1: Entity Extraction", level=1)
                sample_entities = self._debug_entity_extraction(batch.iloc[0][text_column])
                
                self._debug_print("\nSTEP 2: Rare Disease Matching", level=1)
                matches = self._debug_rd_matching(sample_entities, embedded_documents)
                
                if matches:
                    results.append({
                        'document_id': batch.iloc[0][id_column],
                        'matches': matches,
                        'original_text': batch.iloc[0][text_column],
                        'debug_output': True
                    })
                    
                if batch_size > 1:
                    self._process_batch(batch.iloc[1:], embedded_documents, results, text_column, id_column)
            else:
                self._process_batch(batch, embedded_documents, results, text_column, id_column)
                
        # Convert results to DataFrame
        results_df = pd.DataFrame(results)
        if not results_df.empty:
            results_df = results_df.explode('matches').reset_index(drop=True)
            match_details = pd.json_normalize(results_df['matches'])
            for column in match_details.columns:
                results_df[column] = match_details[column]
                
        self._debug_print(f"Processed {len(results_df)} matches")
        return results_df
        
    def _debug_entity_extraction(self, text: str) -> List[str]:
        """Debug entity extraction process."""
        self._debug_print("Sending to LLM:", level=2)
        self._debug_print(f"System message: {self.entity_extractor.system_message}", level=3)
        self._debug_print(f"User input: {text}", level=3)
        
        findings = self.entity_extractor.extract_entities(text)
        self._debug_print("\nExtracted Entities:", level=2)
        for i, finding in enumerate(findings, 1):
            self._debug_print(f"{i}. {finding}", level=3)
            
        return findings
        
    def _debug_rd_matching(self, entities: List[str], metadata: List[Dict]) -> List[Dict]:
        """Debug rare disease matching process."""
        if not entities:
            self._debug_print("No entities to match", level=2)
            return []
            
        if self.rd_matcher.index is None:
            self._debug_print("Initializing matcher index...", level=2)
            self.rd_matcher.prepare_index(metadata)
            
        self._debug_print(f"Processing {len(entities)} entities:", level=2)
        matches = []
        
        for entity in entities:
            self._debug_print(f"\nMatching entity: {entity}", level=2)
            
            candidates = self.rd_matcher._retrieve_candidates(entity)
            self._debug_print("1. Retrieving candidates:", level=3)
            self._debug_print_candidates(candidates[:5])
            
            match = self.rd_matcher._try_enriched_matching(entity, candidates)
            if match:
                self._debug_print(f"Found exact/fuzzy match: {match}", level=3)
                matches.append(match)
                continue
                
            llm_match = self.rd_matcher._try_llm_match(entity, candidates)
            if llm_match:
                self._debug_print(f"Found LLM match: {llm_match}", level=3)
                matches.append(llm_match)
            else:
                self._debug_print("No matches found for entity", level=3)
                
        return matches
        
    def _debug_print_candidates(self, candidates: List[Dict]):
        """Helper to print candidate information."""
        for i, candidate in enumerate(candidates, 1):
            try:
                self._debug_print(
                    f"Candidate {i}: {candidate['metadata']['name']} "
                    f"(ORPHA:{candidate['metadata']['id']}) "
                    f"Score: {candidate['similarity_score']:.3f}", 
                    level=4
                )
            except Exception as e:
                self._debug_print(f"Could not print candidate {i}: {str(e)}", level=4)
                
    def _process_batch(self, batch, embedded_documents, results, text_column, id_column):
        """Process a batch of samples."""
        texts = batch[text_column].tolist()
        batch_entities = self.entity_extractor.process_batch(texts)
        print("-----batch entities --------")
        print(batch_entities)
        batch_metadata = [embedded_documents] * len(batch_entities)
        batch_matches = self.rd_matcher.process_batch(batch_entities, batch_metadata)
        print("------batch matches --------")
        print(batch_matches)
        for j, matches in enumerate(batch_matches):
            if matches:  # matches is a list of dictionaries with full match info
                results.append({
                    'document_id': batch.iloc[j][id_column],
                    'matches': matches,
                    'original_text': batch.iloc[j][text_column]
                })