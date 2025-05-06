from abc import ABC, abstractmethod
import json
import csv
import re
import sys
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
from pathlib import Path
from rdma.utils.embedding import EmbeddingsManager


@dataclass
class VectorizationConfig:
    """Configuration for vectorization process."""
    model_type: str  # 'fastembed' or 'medcpt'
    model_name: Optional[str] = None  # Required for fastembed, not for medcpt
    batch_size: int = 100
    json_file_path: str = 'hpo_data_with_lineage.json'
    csv_file_path: str = 'HPO_addons.csv'
    output_file: str = 'G2GHPO_metadata.npy'
    csv_output_file: str = 'HP_DB.csv'
    device: str = 'cpu'


class HPOVectorizer:
    """Main class for HPO term vectorization."""
    
    def __init__(self, config: VectorizationConfig):
        self.config = config
        self.embedding_manager = EmbeddingsManager(
            model_type=config.model_type,
            model_name=config.model_name,
            device=config.device  # Add this line
        )
        csv.field_size_limit(sys.maxsize)
        self._pattern = re.compile(r'\(.*?\)')
    
    def _clean_text(self, text: str) -> str:
        """Clean text by removing parentheses and standardizing format."""
        return re.sub(self._pattern, '', text).replace('_', ' ').lower()
    
    def _process_json_file(self, csv_data: pd.DataFrame) -> tuple[List[tuple], List[Dict]]:
        """Process JSON file and integrate CSV data."""
        data = []
        csv_rows = []
        
        with open(self.config.json_file_path, 'r') as file:
            hpo_data = json.load(file)
            
        for hp_id, details in tqdm(hpo_data.items(), desc="Processing HPO terms"):
            formatted_hp_id = hp_id.replace('_', ':')
            unique_info = set()
            
            # Process label
            label = details.get('label')
            if label:
                unique_info.add(self._clean_text(label))
            
            # Process synonyms
            synonyms = details.get('synonyms', [])
            for synonym in synonyms:
                unique_info.add(self._clean_text(synonym))
            
            # Process definition
            definition = details.get('definition', '')
            if definition:
                unique_info.add(self._clean_text(definition))
            
            # Add CSV info
            csv_addons = csv_data[csv_data['HP_ID'] == formatted_hp_id]['info'].tolist()
            for addon in csv_addons:
                unique_info.add(self._clean_text(addon))
            
            # Process lineage
            lineages = details.get('lineage', [])
            for info in unique_info:
                data.append((formatted_hp_id, info, ', '.join(lineages)))
                csv_rows.append({
                    'HP_ID': formatted_hp_id,
                    'info': info,
                    'lineage': ', '.join(lineages)
                })
        
        return data, csv_rows
    
    def _save_to_csv(self, csv_rows: List[Dict], output_file: str):
        """Save processed data to CSV for inspection."""
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['HP_ID', 'info', 'lineage'])
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"Processed data saved to {output_file}")
    
    def _calculate_depth(self, lineage: str) -> int:
        """Calculate depth of term in hierarchy."""
        return lineage.count("->") + 1
    
    def _extract_organ_system(self, lineage: str) -> str:
        """Extract organ system from lineage."""
        parts = lineage.split("->")
        return parts[1].strip() if len(parts) > 1 else "Unknown"
    
    def create_vector_database(self, data: List[tuple]) -> List[Dict[str, Any]]:
        """Create vector database from processed data."""
        # Check for existing embeddings
        if os.path.exists(self.config.output_file):
            print("Loading existing embedded documents...")
            embedded_documents = list(np.load(self.config.output_file, allow_pickle=True))
        else:
            print("Starting with new embedded documents list...")
            embedded_documents = []
            
        print(f"Data prepared with {len(data)} terms to embed.")
        
        for i in tqdm(range(0, len(data), self.config.batch_size), 
                     desc="Creating embeddings", 
                     total=(len(data) + self.config.batch_size - 1) // self.config.batch_size):
            
            batch_data = data[i:i + self.config.batch_size]
            
            for hp_id, cleaned_info, lineage in batch_data:
                try:
                    # Generate embedding for single text using document encoder
                    embedding = self.embedding_manager.embed_text(cleaned_info)
                    
                    depth = self._calculate_depth(lineage)
                    organ_system = self._extract_organ_system(lineage)
                    
                    document = {
                        'embedding': embedding,
                        'unique_metadata': {'info': cleaned_info, 'hp_id': hp_id},
                        'lineage': lineage,
                        'organ_system': organ_system,
                        'depth_from_root': depth
                    }
                    embedded_documents.append(document)
                    
                except Exception as e:
                    print(f"Failed to embed text due to {e}")
                    continue
        
        return embedded_documents
    
    def vectorize(self) -> None:
        """Main method to run the vectorization process."""
        # Load CSV data
        csv_data = pd.read_csv(self.config.csv_file_path)
        
        # Process JSON and integrate CSV data
        print("Processing HPO data...")
        data, csv_rows = self._process_json_file(csv_data)
        
        # Save processed data to CSV for inspection
        self._save_to_csv(csv_rows, self.config.csv_output_file)
        print(f"Database contains {len(data)} entries ready for embedding")
        
        # Create and save vector database
        print("Creating vector database...")
        embedded_documents = self.create_vector_database(data)
        
        # Save embeddings
        np.save(self.config.output_file, embedded_documents, allow_pickle=True)
        print(f"Embeddings saved to: {self.config.output_file}")


def create_vectorizer(model_type: str, model_name: Optional[str] = None, 
                     config: Optional[VectorizationConfig] = None) -> HPOVectorizer:
    """Factory function to create a vectorizer with specified model type."""
    if config is None:
        config = VectorizationConfig(model_type=model_type, model_name=model_name)
    return HPOVectorizer(config)