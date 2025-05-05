from abc import ABC, abstractmethod
import numpy as np
import faiss
import os
import torch
from fastembed import TextEmbedding
from transformers import AutoTokenizer, AutoModel
from typing import Dict, Any, List, Optional, Union, Tuple
from sentence_transformers import SentenceTransformer


class EmbeddingModel(ABC):
    """Abstract base class for embedding models."""

    @abstractmethod
    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single piece of text."""
        pass

    @abstractmethod
    def query_text(self, text: str) -> np.ndarray:
        """Embed a query text, which might use a different encoder."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the dimension of the embedding vectors."""
        pass


class FastEmbedModel(EmbeddingModel):
    """FastEmbed implementation."""  # no device needed surprisingly.

    def __init__(self, model_name: str, device: str = None):
        self.model = TextEmbedding(model_name=model_name)
        # Get dimensions by embedding a sample text
        sample_embedding = list(self.model.embed(["sample text"]))[0]
        self._dimension = len(sample_embedding)

    def _get_embedding(self, text: str) -> np.ndarray:
        """Internal method to get embedding using FastEmbed."""
        return np.array(list(self.model.embed([text]))[0]).astype(np.float32)

    def embed_text(self, text: str) -> np.ndarray:
        """Embed text for document storage."""
        return self._get_embedding(text)

    def query_text(self, text: str) -> np.ndarray:
        """For FastEmbed, query embedding uses same method as document embedding."""
        return self._get_embedding(text)

    @property
    def dimension(self) -> int:
        return self._dimension


class MedCPTModel(EmbeddingModel):
    """MedCPT implementation with separate document and query encoders."""

    def __init__(self, device=None):
        # Initialize document encoder
        self.doc_tokenizer = AutoTokenizer.from_pretrained(
            "ncbi/MedCPT-Article-Encoder"
        )
        self.doc_model = AutoModel.from_pretrained("ncbi/MedCPT-Article-Encoder")

        # Initialize query encoder
        self.query_tokenizer = AutoTokenizer.from_pretrained(
            "ncbi/MedCPT-Query-Encoder"
        )
        self.query_model = AutoModel.from_pretrained("ncbi/MedCPT-Query-Encoder")

        # Set up device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"Using device: {self.device}")

        self.doc_model.to(self.device)
        self.query_model.to(self.device)

        # Enable half precision for GPU
        if self.device.type == "cuda":
            self.doc_model.half()  # Convert to FP16
            self.query_model.half()
            torch.cuda.empty_cache()  # Clear GPU memory

        self._dimension = 768  # MedCPT's dimension

    def embed_text(self, text: str) -> np.ndarray:
        """Embed a document using the Article Encoder."""
        with torch.no_grad():
            # MedCPT expects a list of texts [title, abstract]
            # For single text input, we'll split it into chunks if possible
            text_parts = self._split_text(text)

            encoded = self.doc_tokenizer(
                [text_parts],  # Wrap in list for batch processing
                truncation=True,
                padding=True,
                return_tensors="pt",
                max_length=512,
            ).to(self.device)

            outputs = self.doc_model(**encoded)
            embedding = outputs.last_hidden_state[:, 0, :]  # Get CLS token embedding
            return embedding.cpu().numpy()[0].astype(np.float32)

    def query_text(self, text: str) -> np.ndarray:
        """Embed a query using the Query Encoder."""
        with torch.no_grad():
            encoded = self.query_tokenizer(
                text,
                truncation=True,
                padding=True,
                return_tensors="pt",
                max_length=64,  # Query encoder uses shorter sequences
            ).to(self.device)

            outputs = self.query_model(**encoded)
            embedding = outputs.last_hidden_state[:, 0, :]
            return embedding.cpu().numpy()[0].astype(np.float32)

    def _split_text(self, text: str) -> List[str]:
        """Split text into title-like and content-like parts if possible."""
        # Try to split on first sentence or paragraph break
        splits = text.split("\n", 1)
        if len(splits) > 1:
            return splits

        splits = text.split(". ", 1)
        if len(splits) > 1:
            return splits

        # If no good split found, use same text twice
        return [text, text]

    @property
    def dimension(self) -> int:
        return self._dimension


class SentenceTransformerEmbedModel(EmbeddingModel):
    """Implementation using Sentence Transformers for both document and query embedding."""

    def __init__(self, model_name: Optional[str], device: str = "cpu"):
        """
        Initialize the SentenceTransformer model.

        Args:
            model_name: Name or path of the SentenceTransformer model to use
            device: Device to use (e.g., "cpu", "cuda:0", "cuda:1")

        Raises:
            ValueError: If model_name is None or empty
        """
        if not model_name:
            raise ValueError(
                "model_name must be provided for SentenceTransformer. "
                "Please specify a valid model name (e.g., 'abhinand/MedEmbed-small-v0.1')"
            )

        self.device = device
        print(
            f"Initializing SentenceTransformer with model: {model_name} on device: {device}"
        )

        try:
            # Initialize model and move to specified device
            self.model = SentenceTransformer(model_name)

            # Handle device assignment
            if "cuda" in device:
                if torch.cuda.is_available():
                    # If specific GPU requested (e.g., "cuda:1"), extract the index
                    if ":" in device:
                        gpu_id = int(device.split(":")[-1])
                        if gpu_id >= torch.cuda.device_count():
                            raise ValueError(
                                f"GPU {gpu_id} not found. Available GPUs: {torch.cuda.device_count()}"
                            )
                    self.model.to(device)
                    print(f"Model successfully moved to {device}")
                else:
                    print(
                        f"Warning: CUDA requested but not available. Falling back to CPU."
                    )
                    self.device = "cpu"
                    self.model.to("cpu")
            else:
                self.model.to("cpu")
                print("Model running on CPU")

            # Get dimensions by embedding a sample text
            print("Verifying model by embedding sample text...")
            sample_embedding = self.model.encode("sample text", convert_to_numpy=True)
            self._dimension = len(sample_embedding)
            print(
                f"Model initialized successfully. Embedding dimension: {self._dimension}"
            )

        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize SentenceTransformer model: {str(e)}"
            )

    def embed_text(self, text: str) -> np.ndarray:
        """
        Embed a single piece of text for document storage.

        Args:
            text: Text to embed

        Returns:
            numpy.ndarray: Embedding vector
        """
        if not isinstance(text, str):
            raise ValueError(f"Expected string input, got {type(text)}")

        # Encode text and ensure output is numpy array
        embedding = self.model.encode(text, convert_to_numpy=True, device=self.device)
        return embedding.astype(np.float32)

    def query_text(self, text: str) -> np.ndarray:
        """
        Embed a query text. For SentenceTransformer, we use the same embedding method.

        Args:
            text: Query text to embed

        Returns:
            numpy.ndarray: Embedding vector
        """
        return self.embed_text(text)

    def batch_embed(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Embed multiple texts in batches.

        Args:
            texts: List of texts to embed
            batch_size: Size of batches for processing

        Returns:
            numpy.ndarray: Array of embedding vectors
        """
        if not texts:
            raise ValueError("Cannot embed empty list of texts")

        if not all(isinstance(text, str) for text in texts):
            raise ValueError("All inputs must be strings")

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=True,
            device=self.device,
        )
        return embeddings.astype(np.float32)

    @property
    def dimension(self) -> int:
        """Return the dimension of the embedding vectors."""
        return self._dimension


class EmbeddingsManager:
    """Manager for embedding operations with multiple model support."""

    MODEL_TYPES = {
        "fastembed": FastEmbedModel,
        "medcpt": MedCPTModel,
        "sentence_transformer": SentenceTransformerEmbedModel,  # Add this line
    }

    def __init__(
        self,
        model_type: str,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize with specific model type and name.

        Args:
            model_type: One of ['fastembed', 'medcpt', 'sentence_transformer']
            model_name: Name/path of the specific model (required for fastembed and sentence_transformer)
            device: Device to use ('cuda', 'cuda:0', 'cuda:1', etc. or 'cpu')
        """
        if model_type not in self.MODEL_TYPES:
            raise ValueError(
                f"Unsupported model type. Must be one of {list(self.MODEL_TYPES.keys())}"
            )

        self.model_type = model_type
        self.model_name = model_name
        self.device = device

        print("Loading model...")
        print(f"Model type: {model_type}")
        print(f"Model name: {model_name}")
        print(f"Device: {device}")

        # Initialize model based on type
        if model_type == "medcpt":
            # MedCPT has its own models, only needs device
            self.model = self.MODEL_TYPES[model_type](
                device=device if device else "cpu"
            )
        else:
            # Both fastembed and sentence_transformer require model_name
            if not model_name:
                raise ValueError(
                    f"model_name is required for {model_type}. "
                    f"Please specify a valid model name."
                )
            # Pass both model_name and device consistently for all other models
            self.model = self.MODEL_TYPES[model_type](
                model_name,  # Pass as positional argument
                device=device if device else "cpu",  # Pass device as keyword argument
            )

    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single piece of text for document storage."""
        return self.model.embed_text(text)

    def query_text(self, text: str) -> np.ndarray:
        """Embed a query text, which might use a different encoder."""
        return self.model.query_text(text)

    def load_documents(self, file_path: str) -> List[Dict[str, Any]]:
        """Load embedded documents and verify model compatibility."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Embeddings file not found: {file_path}")

        data = np.load(file_path, allow_pickle=True)
        print(f"Data type: {type(data)}")
        print(f"Shape or length: {data.shape if hasattr(data, 'shape') else len(data)}")
        print(f"First element type: {type(data[0]) if len(data) > 0 else 'empty'}")
        # Verify first embedding dimension matches current model
        if len(data) > 0 and "embedding" in data[0]:
            embedding_dim = data[0]["embedding"].shape[0]
            if embedding_dim != self.model.dimension:
                raise ValueError(
                    f"Embedding dimension mismatch. File has {embedding_dim} dimensions, "
                    f"but model produces {self.model.dimension} dimensions"
                )

        return data.tolist()

    def prepare_embeddings(
        self, embedded_documents: List[Dict[str, Any]]
    ) -> np.ndarray:
        """Prepare embeddings for indexing."""
        embeddings_list = [
            np.array(doc["embedding"])
            for doc in embedded_documents
            if isinstance(doc["embedding"], np.ndarray) and doc["embedding"].size > 0
        ]

        if not embeddings_list:
            raise ValueError("No valid embeddings found")

        return np.vstack(embeddings_list).astype(np.float32)

    def create_index(self, embeddings_array: np.ndarray) -> faiss.Index:
        """Create a FAISS index for the embeddings."""
        dimension = embeddings_array.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings_array)
        return index

    def batch_embed_documents(self, texts: List[str]) -> np.ndarray:
        """Batch embed multiple documents."""
        return np.vstack([self.embed_text(text) for text in texts])

    def batch_embed_queries(self, queries: List[str]) -> np.ndarray:
        """Batch embed multiple queries."""
        return np.vstack([self.query_text(query) for query in queries])

    def search(
        self, query: Union[str, np.ndarray], index: faiss.Index, k: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for similar documents using a query.

        Args:
            query: Query text or pre-computed query embedding
            index: FAISS index to search
            k: Number of results to return

        Returns:
            Tuple of (distances, indices)
        """
        if isinstance(query, str):
            query_vector = self.query_text(query).reshape(1, -1)
        else:
            query_vector = query.reshape(1, -1)

        return index.search(query_vector, k)
