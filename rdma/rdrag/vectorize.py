from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, Any
import json
from pathlib import Path
import numpy as np
from tqdm import tqdm
import torch
from contextlib import nullcontext
from rdma.utils.embedding import EmbeddingModel, FastEmbedModel, SentenceTransformerEmbedModel, MedCPTModel
import re
import os
@dataclass
class RareDiseaseVectorizationConfig:
    """Configuration for rare disease vectorization process."""
    model_name: str
    batch_size: int = 100
    ontology_path: str = 'rare_disease_ontology.jsonl'
    triples_path: Optional[str] = 'RareDisease_Phenotype_Triples.json'
    output_file: str = 'rare_disease_embeddings.npy'
    csv_output_file: str = 'rare_disease_db.csv'
    device: str = 'cpu'  # Added device configuration

class RareDiseaseVectorizer:
    """Main class for rare disease term vectorization."""
    
    def __init__(self, config: RareDiseaseVectorizationConfig, embedding_model: EmbeddingModel):
        self.config = config
        self.embedding_model = embedding_model
        self._pattern = re.compile(r'\(.*?\)')  # For cleaning text
        self.device = torch.device(config.device)
        print(f"Vectorizer initialized with device: {self.device}")
        
    def _clean_text(self, text: str) -> str:
        """Clean text by removing parentheses and standardizing format."""
        return re.sub(self._pattern, '', text).strip().lower()
        
    def _load_jsonl(self, path: str) -> List[Dict]:
        """Load JSONL file line by line."""
        data = []
        with open(path) as f:
            for line in f:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Error decoding line in {path}: {e}")
        return data
        
    def _process_files(self) -> tuple[List[tuple], List[Dict]]:
        """Process both ontology and triples files to extract names and definitions."""
        print("Loading ontology file...")
        ontology_data = self._load_jsonl(self.config.ontology_path)
        
        print("Loading triples file...")
        with open(self.config.triples_path) as f:
            triples_data = json.load(f)
            
        entries = []  # List of (name, orpha_id, definition) tuples
        csv_rows = []
        
        # Process ontology entries
        print("Processing ontology entries...")
        for entry in tqdm(ontology_data):
            orpha_id = entry['id']
            name = self._clean_text(entry['name'])
            definition = self._clean_text(entry.get('definition', ''))
            
            entries.append((name, orpha_id, definition))
            csv_rows.append({
                'name': name,
                'orpha_id': orpha_id,
                'definition': definition,
                'source': 'ontology'
            })
            
        # Process triples entries
        print("Processing triples entries...")
        for triple in tqdm(triples_data):
            orpha_id = triple['source']['id']
            definition = self._clean_text(triple['source'].get('definition', ''))
            
            for name in triple['source']['name']:
                name = self._clean_text(name)
                if not any(e[0] == name and e[1] == orpha_id for e in entries):
                    entries.append((name, orpha_id, definition))
                    csv_rows.append({
                        'name': name,
                        'orpha_id': orpha_id,
                        'definition': definition,
                        'source': 'triples'
                    })
        
        return entries, csv_rows
    
    def _save_to_csv(self, rows: List[Dict], filepath: str) -> None:
        """Save processed data to CSV file."""
        import pandas as pd
        df = pd.DataFrame(rows)
        df.to_csv(filepath, index=False)
        print(f"Saved processed data to: {filepath}")
    
    def create_vector_database(self, entries: List[tuple]) -> List[Dict[str, Any]]:
        """Create vector database from processed entries.
        
        Args:
            entries: List of (name, orpha_id, definition) tuples
            
        Returns:
            List of document dictionaries with embeddings and metadata
        """
        embedded_documents = []
        
        print(f"Processing {len(entries)} entries...")
        
        # Enhanced batch processing with GPU support
        with torch.cuda.device(self.device) if 'cuda' in str(self.device) else nullcontext():
            for i in tqdm(range(0, len(entries), self.config.batch_size)):
                batch_entries = entries[i:i + self.config.batch_size]
                
                try:
                    # Process each text individually since embed_text expects a single string
                    for name, orpha_id, definition in batch_entries:
                        # Create combined text for embedding
                        text = f"{name} {definition}".strip()
                        
                        # Generate embedding for single text
                        embedding = self.embedding_model.embed_text(text)
                        
                        document = {
                            'embedding': embedding,
                            'name': name,
                            'id': orpha_id,
                            'definition': definition
                        }
                        embedded_documents.append(document)
                    
                    # Clear CUDA cache periodically if using GPU
                    if 'cuda' in str(self.device) and i % (self.config.batch_size * 10) == 0:
                        torch.cuda.empty_cache()
                        
                except RuntimeError as e:
                    if 'out of memory' in str(e):
                        # Handle OOM by processing in smaller batches
                        torch.cuda.empty_cache()
                        smaller_batch_size = self.config.batch_size // 2
                        print(f"OOM error, reducing batch size to {smaller_batch_size}")
                        
                        for j in range(i, min(i + self.config.batch_size, len(entries)), smaller_batch_size):
                            sub_batch = entries[j:j + smaller_batch_size]
                            
                            for name, orpha_id, definition in sub_batch:
                                text = f"{name} {definition}".strip()
                                embedding = self.embedding_model.embed_text(text)
                                
                                document = {
                                    'embedding': embedding,
                                    'name': name,
                                    'id': orpha_id,
                                    'definition': definition
                                }
                                embedded_documents.append(document)
                    else:
                        raise e

        return embedded_documents
    
    def vectorize(self) -> None:
        """Main method to run the vectorization process."""
        print("Processing rare disease data...")
        entries, csv_rows = self._process_files()
        
        if self.config.csv_output_file:
            self._save_to_csv(csv_rows, self.config.csv_output_file)
        
        print(f"Database contains {len(entries)} unique name entries")
        
        print("Creating vector database...")
        embedded_documents = self.create_vector_database(entries)
        
        # Save as numpy array of documents
        np.save(self.config.output_file, embedded_documents, allow_pickle=True)
        print(f"Embeddings saved to: {self.config.output_file}")

def create_rare_disease_vectorizer(
    model_type: str,
    model_name: str,
    ontology_path: str,
    triples_path: str,
    batch_size: int = 100,
    output_file: str = 'rare_disease_embeddings.npy',
    csv_output_file: str = 'rare_disease_db.csv',
    device: str = 'cpu'
) -> RareDiseaseVectorizer:
    """Factory function to create a rare disease vectorizer."""
    config = RareDiseaseVectorizationConfig(
        model_name=model_name,
        ontology_path=ontology_path,
        triples_path=triples_path,
        batch_size=batch_size,
        output_file=output_file,
        csv_output_file=csv_output_file,
        device=device
    )
    
    if model_type.lower() == 'fastembed':
        embedding_model = FastEmbedModel(model_name)
    elif model_type.lower() == 'sentence_transformer':
        embedding_model = SentenceTransformerEmbedModel(model_name, device=device)
    elif model_type.lower() == 'medcpt':
        embedding_model = MedCPTModel(device=device)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    return RareDiseaseVectorizer(config, embedding_model)