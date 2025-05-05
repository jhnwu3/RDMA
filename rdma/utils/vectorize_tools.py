#!/usr/bin/env python3
import json
import sys
import os
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
from pathlib import Path

import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from rdma.utils.embedding import EmbeddingsManager
@dataclass
class ToolVectorizationConfig:
    """Configuration for vectorization process."""
    model_type: str  # 'fastembed', 'medcpt', or 'sentence_transformer'
    model_name: Optional[str] = None  # Required for fastembed and sentence_transformer, not for medcpt
    batch_size: int = 100
    abbreviations_file_path: str = 'abbreviations.json'
    lab_table_file_path: str = 'lab_table.json'
    abbreviations_output_file: str = 'abbreviations_vectors.npy'
    lab_table_output_file: str = 'lab_table_vectors.npy'
    device: str = 'cpu'


class AbbreviationsVectorizer:
    """Class for vectorizing abbreviations dictionary."""
    
    def __init__(self, config: ToolVectorizationConfig):
        self.config = config
        self.embedding_manager = EmbeddingsManager(
            model_type=config.model_type,
            model_name=config.model_name,
            device=config.device
        )
    
    def _process_json_file(self) -> List[Dict]:
        """Process JSON file and prepare data for vectorization."""
        data = []
        
        with open(self.config.abbreviations_file_path, 'r') as file:
            abbreviations_dict = json.load(file)
            
        for abbr, meaning in tqdm(abbreviations_dict.items(), desc="Processing abbreviations"):
            data.append({
                'abbr': abbr,
                'meaning': meaning
            })
            
        return data
    
    def create_vector_database(self, data: List[Dict]) -> List[Dict[str, Any]]:
        """Create vector database from abbreviations data."""
        embedded_documents = []
        
        for i in tqdm(range(0, len(data), self.config.batch_size), 
                     desc="Creating embeddings", 
                     total=(len(data) + self.config.batch_size - 1) // self.config.batch_size):
            
            batch_data = data[i:i + self.config.batch_size]
            
            for item in batch_data:
                try:
                    # Use the abbreviation as the primary text to embed
                    abbr = item['abbr']
                    meaning = item['meaning']
                    
                    # Generate embedding for the abbreviation
                    embedding = self.embedding_manager.embed_text(abbr)
                    
                    document = {
                        'embedding': embedding,
                        'unique_metadata': {
                            'info': abbr,     # The term to match against when searching
                            'hp_id': meaning  # The meaning to retrieve when matched
                        }
                    }
                    embedded_documents.append(document)
                    
                except Exception as e:
                    print(f"Failed to embed abbreviation '{item['abbr']}' due to {e}")
                    continue
        
        return embedded_documents
    
    def vectorize(self) -> None:
        """Main method to run the vectorization process."""
        # Process JSON data
        print("Processing abbreviations data...")
        data = self._process_json_file()
        
        print(f"Parsed {len(data)} abbreviation entries")
        
        # Create and save vector database
        print("Creating vector database...")
        embedded_documents = self.create_vector_database(data)
        
        # Save embeddings
        np.save(self.config.abbreviations_output_file, embedded_documents, allow_pickle=True)
        print(f"Abbreviations embeddings saved to: {self.config.abbreviations_output_file}")
        print(f"Total abbreviations vectorized: {len(embedded_documents)}")


class LabTableVectorizer:
    """Class for vectorizing lab table dictionary."""
    
    def __init__(self, config: ToolVectorizationConfig):
        self.config = config
        self.embedding_manager = EmbeddingsManager(
            model_type=config.model_type,
            model_name=config.model_name,
            device=config.device
        )
    
    def _process_json_file(self) -> List[Dict]:
        """Process JSON file and prepare data for vectorization."""
        data = []
        
        with open(self.config.lab_table_file_path, 'r') as file:
            lab_table_dict = json.load(file)
            
        print(f"Loaded lab table with {len(lab_table_dict)} entries")
        # Print sample entry to debug the format
        if lab_table_dict:
            sample_key = next(iter(lab_table_dict))
            sample_value = lab_table_dict[sample_key]
            print(f"Sample entry - Key: {sample_key}, Value type: {type(sample_value)}, Content: {str(sample_value)[:100]}")
        
        for lab_id, lab_info in tqdm(lab_table_dict.items(), desc="Processing lab table"):
            try:
                # Handle case where lab_info is a string
                if isinstance(lab_info, str):
                    data.append({
                        'lab_id': lab_id,
                        'name': lab_info,  # Use the string itself as the name
                        'reference_ranges': []  # Empty reference ranges
                    })
                # Handle case where lab_info is a dictionary
                elif isinstance(lab_info, dict) and 'name' in lab_info:
                    data.append({
                        'lab_id': lab_id,
                        'name': lab_info.get('name', ''),
                        'reference_ranges': lab_info.get('reference_ranges', [])
                    })
                else:
                    print(f"Skipping entry with lab_id '{lab_id}' due to unexpected format: {type(lab_info)}")
            except Exception as e:
                print(f"Error processing lab_id '{lab_id}': {e}")
                    
        print(f"Successfully processed {len(data)} lab table entries")
        return data
    
    def create_vector_database(self, data: List[Dict]) -> List[Dict[str, Any]]:
        """Create vector database from lab table data."""
        embedded_documents = []
        
        for i in tqdm(range(0, len(data), self.config.batch_size), 
                     desc="Creating embeddings", 
                     total=(len(data) + self.config.batch_size - 1) // self.config.batch_size):
            
            batch_data = data[i:i + self.config.batch_size]
            
            for item in batch_data:
                try:
                    # Use the lab name as the primary text to embed
                    lab_id = item['lab_id']
                    name = item['name']
                    reference_ranges = item.get('reference_ranges', [])
                    
                    # Skip items with empty names
                    if not name.strip():
                        print(f"Skipping lab entry with empty name for lab_id {lab_id}")
                        continue
                    
                    # Package the lab information as JSON to store in the hp_id field
                    lab_info = {
                        'lab_id': lab_id,
                        'name': name,
                        'reference_ranges': reference_ranges
                    }
                    
                    # Generate embedding for the lab name
                    embedding = self.embedding_manager.embed_text(name)
                    
                    document = {
                        'embedding': embedding,
                        'unique_metadata': {
                            'info': name,                 # The term to match against when searching
                            'hp_id': json.dumps(lab_info) # The lab info to retrieve when matched
                        }
                    }
                    embedded_documents.append(document)
                    
                except Exception as e:
                    print(f"Failed to embed lab '{item.get('name', '(unknown)')}' due to {e}")
                    continue
        
        return embedded_documents
    
    def vectorize(self) -> None:
        """Main method to run the vectorization process."""
        # Process JSON data
        print("Processing lab table data...")
        data = self._process_json_file()
        
        print(f"Parsed {len(data)} lab table entries")
        
        # Create and save vector database
        print("Creating vector database...")
        embedded_documents = self.create_vector_database(data)
        
        # Save embeddings
        np.save(self.config.lab_table_output_file, embedded_documents, allow_pickle=True)
        print(f"Lab table embeddings saved to: {self.config.lab_table_output_file}")
        print(f"Total lab entries vectorized: {len(embedded_documents)}")


def create_tool_vectorizer(tool_type: str, config: ToolVectorizationConfig):
    """Factory function to create appropriate vectorizer based on tool type."""
    if tool_type == "abbreviations":
        return AbbreviationsVectorizer(config)
    elif tool_type == "lab_table":
        return LabTableVectorizer(config)
    else:
        raise ValueError(f"Unsupported tool type: {tool_type}. Valid options are 'abbreviations' or 'lab_table'")


def main():
    """Main function to run the vectorization process from command line."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Create vector database from tools data')
    
    # Required arguments
    parser.add_argument('--tool_type', 
                       type=str, 
                       choices=['abbreviations', 'lab_table', 'both'],
                       required=True,
                       help='Type of tool to vectorize')
    
    # Support both naming conventions for model type
    model_type_group = parser.add_mutually_exclusive_group(required=True)
    model_type_group.add_argument('--model_type', 
                       type=str, 
                       choices=['fastembed', 'medcpt', 'sentence_transformer'],
                       help='Type of embedding model to use')
    model_type_group.add_argument('--retriever', 
                       type=str, 
                       choices=['fastembed', 'medcpt', 'sentence_transformer'],
                       help='Type of retriever/embedding model to use')
    
    # Support both naming conventions for model name
    model_name_group = parser.add_mutually_exclusive_group()
    model_name_group.add_argument('--model_name',
                       type=str,
                       help='Name of the model (required for fastembed and sentence_transformer, e.g., "BAAI/bge-small-en-v1.5")')
    model_name_group.add_argument('--retriever_model',
                       type=str,
                       help='Name of the retriever model (required for fastembed and sentence_transformer)')
    
    parser.add_argument('--abbreviations_file',
                       type=str,
                       default='abbreviations.json',
                       help='Path to input JSON file containing abbreviations')
    
    parser.add_argument('--lab_table_file',
                       type=str,
                       default='lab_table.json',
                       help='Path to input JSON file containing lab table')
    
    parser.add_argument('--abbreviations_output',
                       type=str,
                       default='abbreviations_vectors.npy',
                       help='Path for output NPY file containing abbreviations embeddings')
    
    parser.add_argument('--lab_table_output',
                       type=str,
                       default='lab_table_vectors.npy',
                       help='Path for output NPY file containing lab table embeddings')
    
    parser.add_argument('--batch_size',
                       type=int,
                       default=100,
                       help='Batch size for processing')
    
    parser.add_argument('--device',
                       type=str,
                       default='cpu',
                       help='Device to use for embedding (e.g., "cuda:0", "cpu")')

    args = parser.parse_args()
    
    # Get model type from either argument
    model_type = args.model_type if args.model_type else args.retriever
    
    # Get model name from either argument
    model_name = args.model_name if args.model_name else args.retriever_model
    
    # Validate model_name is provided if fastembed or sentence_transformer is selected
    if model_type in ['fastembed', 'sentence_transformer'] and not model_name:
        parser.error(f"--model_name or --retriever_model is required when using {model_type}")
    
    # Create configuration
    config = ToolVectorizationConfig(
        model_type=model_type,
        model_name=model_name,
        batch_size=args.batch_size,
        abbreviations_file_path=args.abbreviations_file,
        lab_table_file_path=args.lab_table_file,
        abbreviations_output_file=args.abbreviations_output,
        lab_table_output_file=args.lab_table_output,
        device=args.device
    )
    
    # Process selected tool(s)
    if args.tool_type == 'both' or args.tool_type == 'abbreviations':
        print("\n=== Vectorizing Abbreviations ===")
        abbreviations_vectorizer = AbbreviationsVectorizer(config)
        abbreviations_vectorizer.vectorize()
    
    if args.tool_type == 'both' or args.tool_type == 'lab_table':
        print("\n=== Vectorizing Lab Table ===")
        lab_table_vectorizer = LabTableVectorizer(config)
        lab_table_vectorizer.vectorize()
    
    print("\nVectorization complete!")


if __name__ == "__main__":
    main()