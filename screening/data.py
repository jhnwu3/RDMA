import json
import random
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

class DataLoader:
    def __init__(self, file_path: str, batch_size: Optional[int] = None, seed: Optional[int] = None):
        """
        Initialize the data loader.
        
        Args:
            file_path (str): Path to the JSONL file
            batch_size (int, optional): Size of batches to return
            seed (int, optional): Random seed for reproducibility
        """
        self.file_path = file_path
        self.batch_size = batch_size
        self.data = []
        self.phenotype_to_idx = {}  # Cache for phenotype IDs to indices
        
        # Set random seed if provided
        if seed is not None:
            random.seed(seed)
            
        # Load and process the data
        self._load_data()
        
    def _load_data(self) -> None:
        """Load and preprocess the data from the JSONL file."""
        with open(self.file_path, 'r') as file:
            for line in file:
                record = json.loads(line.strip())
                
                # Extract relevant fields
                processed_record = {
                    'age': record['age'],
                    'disease_id': record['disease_id'],
                    # Get all positive phenotypes
                    'phenotypes': list(record['positive_phenotypes'].keys())
                }
                self.data.append(processed_record)
                
                # Update phenotype vocabulary
                for phenotype in processed_record['phenotypes']:
                    if phenotype not in self.phenotype_to_idx:
                        self.phenotype_to_idx[phenotype] = len(self.phenotype_to_idx)
    
    def _convert_phenotypes_to_indices(self, phenotypes: List[str]) -> List[int]:
        """Convert phenotype IDs to indices."""
        return [self.phenotype_to_idx[p] for p in phenotypes]
    
    def get_vocab_size(self) -> int:
        """Return the size of the phenotype vocabulary."""
        return len(self.phenotype_to_idx)
    
    def get_sample(self, num_samples: int = 1) -> List[Dict]:
        """
        Get random samples from the dataset.
        
        Args:
            num_samples (int): Number of samples to return
            
        Returns:
            List of dictionaries containing age, phenotypes, and disease_id
        """
        return random.sample(self.data, num_samples)
    
    def get_batch(self) -> List[Dict]:
        """
        Get a batch of samples from the dataset.
        
        Returns:
            List of dictionaries containing age, phenotypes, and disease_id
        """
        if self.batch_size is None:
            raise ValueError("Batch size not specified during initialization")
        
        return self.get_sample(self.batch_size)
    
    def get_indexed_sample(self, num_samples: int = 1) -> List[Dict]:
        """
        Get random samples with phenotypes converted to indices.
        
        Args:
            num_samples (int): Number of samples to return
            
        Returns:
            List of dictionaries with phenotypes as indices
        """
        samples = self.get_sample(num_samples)
        for sample in samples:
            sample['phenotype_indices'] = self._convert_phenotypes_to_indices(sample['phenotypes'])
        return samples

# Example usage:
if __name__ == "__main__":
    # Initialize the data loader
    loader = DataLoader("data/dataset/rd_phenotype_simulated_data.jsonl", batch_size=32, seed=42)
    
    # Get a single sample
    single_sample = loader.get_sample(1)
    print("\nSingle sample:")
    print(json.dumps(single_sample, indent=2))
    
    # Get a batch with indexed phenotypes
    batch = loader.get_indexed_sample(3)
    print("\nBatch with indexed phenotypes:")
    print(json.dumps(batch, indent=2))
    
    # Print vocabulary size
    print(f"\nTotal number of unique phenotypes: {loader.get_vocab_size()}")