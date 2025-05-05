#!/usr/bin/env python3
import argparse
import json
import numpy as np
import faiss
from typing import List, Dict, Any
from rdma.utils.embedding import EmbeddingsManager


class ToolSearcher:
    """Class for searching vectorized tools."""

    def __init__(
        self,
        model_type: str,
        model_name: str = None,
        device: str = "cpu",
        top_k: int = 5,
    ):
        """
        Initialize the tool searcher.

        Args:
            model_type: Type of embedding model ('fastembed', 'medcpt', 'sentence_transformer')
            model_name: Name of the model (required for fastembed and sentence_transformer)
            device: Device to use for embedding
            top_k: Number of top results to return
        """
        self.embedding_manager = EmbeddingsManager(
            model_type=model_type, model_name=model_name, device=device
        )
        self.top_k = top_k
        self.index = None
        self.embedded_documents = None

    def load_embeddings(self, embeddings_file: str) -> None:
        """
        Load embeddings from file.

        Args:
            embeddings_file: Path to the NPY file containing embeddings
        """
        print(f"Loading embeddings from {embeddings_file}")
        self.embedded_documents = np.load(embeddings_file, allow_pickle=True)
        print(f"Loaded {len(self.embedded_documents)} embedded documents")

        # Prepare embeddings for indexing
        embeddings_list = []
        for doc in self.embedded_documents:
            if isinstance(doc["embedding"], np.ndarray) and doc["embedding"].size > 0:
                embeddings_list.append(doc["embedding"])

        if not embeddings_list:
            raise ValueError("No valid embeddings found")

        embeddings_array = np.vstack(embeddings_list).astype(np.float32)
        dimension = embeddings_array.shape[1]

        # Create FAISS index
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings_array)
        print(
            f"Created FAISS index with {len(embeddings_list)} vectors of dimension {dimension}"
        )

    def search(self, query: str) -> List[Dict[str, Any]]:
        """
        Search for the query in the vectorized tools.

        Args:
            query: Search query

        Returns:
            List of dictionaries with search results
        """
        if self.index is None or self.embedded_documents is None:
            raise ValueError("Embeddings not loaded. Call load_embeddings() first")

        # Embed the query
        query_vector = self.embedding_manager.query_text(query).reshape(1, -1)

        # Search for similar vectors
        distances, indices = self.index.search(query_vector, self.top_k)

        # Format results
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.embedded_documents):
                doc = self.embedded_documents[idx]
                metadata = doc["unique_metadata"]

                # Check if hp_id is a serialized JSON string
                hp_id = metadata["hp_id"]
                try:
                    parsed_hp_id = json.loads(hp_id)
                    metadata_value = parsed_hp_id
                except (json.JSONDecodeError, TypeError):
                    # Not a JSON string, use as is
                    metadata_value = hp_id

                results.append(
                    {
                        "rank": i + 1,
                        "similarity": float(
                            1.0 / (1.0 + dist)
                        ),  # Convert distance to similarity score
                        "query_term": query,
                        "matched_term": metadata["info"],
                        "result": metadata_value,
                    }
                )

        return results


def format_results(results: List[Dict[str, Any]], is_lab_table: bool = False) -> str:
    """
    Format search results for display.

    Args:
        results: List of search results
        is_lab_table: Whether the results are from lab table search

    Returns:
        Formatted string for display
    """
    if not results:
        return "No results found."

    output = []

    for result in results:
        similarity = result["similarity"]
        matched_term = result["matched_term"]

        if is_lab_table:
            result_data = result["result"]
            lab_id = result_data.get("lab_id", "N/A")
            name = result_data.get("name", "N/A")
            ranges = result_data.get("reference_ranges", [])

            range_output = []
            for range_item in ranges:
                age_group = range_item.get("age_group", "N/A")
                male = range_item.get("male", "N/A")
                female = range_item.get("female", "N/A")
                range_output.append(
                    f"    • {age_group}: Male: {male}, Female: {female}"
                )

            output.append(f"[{similarity:.2f}] {name} (ID: {lab_id})")
            output.append("  Reference Ranges:")
            output.extend(range_output)
        else:
            # Abbreviation result
            output.append(f"[{similarity:.2f}] {matched_term} = {result['result']}")

        output.append("")  # Add empty line for readability

    return "\n".join(output)


def main():
    """Main function to run the tool search from command line."""
    parser = argparse.ArgumentParser(description="Search vectorized tools")

    parser.add_argument(
        "--tool_type",
        type=str,
        choices=["abbreviations", "lab_table"],
        required=True,
        help="Type of tool to search",
    )

    parser.add_argument(
        "--embeddings_file",
        type=str,
        required=True,
        help="Path to NPY file containing embeddings",
    )

    # Support both naming conventions for model type
    model_type_group = parser.add_mutually_exclusive_group(required=True)
    model_type_group.add_argument(
        "--model_type",
        type=str,
        choices=["fastembed", "medcpt", "sentence_transformer"],
        help="Type of embedding model to use",
    )
    model_type_group.add_argument(
        "--retriever",
        type=str,
        choices=["fastembed", "medcpt", "sentence_transformer"],
        help="Type of retriever/embedding model to use",
    )

    # Support both naming conventions for model name
    model_name_group = parser.add_mutually_exclusive_group()
    model_name_group.add_argument(
        "--model_name",
        type=str,
        help="Name of the model (required for fastembed and sentence_transformer)",
    )
    model_name_group.add_argument(
        "--retriever_model",
        type=str,
        help="Name of the retriever model (required for fastembed and sentence_transformer)",
    )

    parser.add_argument(
        "--query",
        type=str,
        help="Search query (if not provided, enter interactive mode)",
    )

    parser.add_argument(
        "--top_k", type=int, default=5, help="Number of top results to return"
    )

    parser.add_argument(
        "--device", type=str, default="cpu", help="Device to use for embedding"
    )

    args = parser.parse_args()

    # Get model type from either argument
    model_type = args.model_type if args.model_type else args.retriever

    # Get model name from either argument
    model_name = args.model_name if args.model_name else args.retriever_model

    # Validate model_name is provided if fastembed or sentence_transformer is selected
    if model_type in ["fastembed", "sentence_transformer"] and not model_name:
        parser.error(
            f"--model_name or --retriever_model is required when using {model_type}"
        )

    # Create searcher
    searcher = ToolSearcher(
        model_type=model_type,
        model_name=model_name,
        device=args.device,
        top_k=args.top_k,
    )

    # Load embeddings
    searcher.load_embeddings(args.embeddings_file)

    # Check if we're in interactive mode or one-shot query mode
    if args.query:
        # One-shot query mode
        results = searcher.search(args.query)
        print(format_results(results, is_lab_table=(args.tool_type == "lab_table")))
    else:
        # Interactive mode
        print("\n=== Interactive Search Mode ===")
        print(f"Searching {args.tool_type} with {args.model_type}")
        print("Type 'exit' to quit")

        while True:
            query = input("\nEnter search query: ")
            if query.lower() in ["exit", "quit", "q"]:
                break

            results = searcher.search(query)
            print("\nResults:")
            print(format_results(results, is_lab_table=(args.tool_type == "lab_table")))


if __name__ == "__main__":
    main()
