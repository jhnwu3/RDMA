#!/usr/bin/env python3
import os
import sys
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path to ensure imports work correctly
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Import project modules
from rdma.utils.data import read_json_file
from rdma.utils.embedding import EmbeddingsManager


def enrich_annotations_with_orpha_codes(
    dataset_path: str,
    embeddings_file: str,
    retriever: str = "sentence_transformer",
    retriever_model: str = "abhinand/MedEmbed-small-v0.1",
    output_path: str = None,
    device: str = "cpu",
    top_k: int = 1,
):
    """
    Enrich dataset annotations with Orphanet codes where missing.

    Args:
        dataset_path: Path to the dataset JSON file
        embeddings_file: Path to the pre-computed embeddings file
        retriever: Type of retriever to use
        retriever_model: Name of the retriever model
        output_path: Path to save the enriched dataset
        device: Device to use for embeddings
        top_k: Number of top candidates to retrieve

    Returns:
        The enriched dataset
    """
    print(f"Loading dataset from {dataset_path}")
    data = read_json_file(dataset_path)
    if "documents" in data:
        data = data["documents"]
    print(f"Initializing {retriever} embedding manager with model: {retriever_model}")
    embedding_manager = EmbeddingsManager(
        model_type=retriever, model_name=retriever_model, device=device
    )

    print(f"Loading embeddings from {embeddings_file}")
    try:
        embedded_documents = np.load(embeddings_file, allow_pickle=True)
        print(f"Loaded {len(embedded_documents)} embedded documents")
    except Exception as e:
        print(f"Error loading embeddings file: {e}")
        raise

    # Prepare embeddings for retrieval
    print("Preparing embeddings for retrieval")
    embeddings_array = embedding_manager.prepare_embeddings(embedded_documents)
    index = embedding_manager.create_index(embeddings_array)

    # Track statistics
    total_docs = len(data)
    docs_with_annotations = 0
    total_annotations = 0
    annotations_enriched = 0

    # Process each document
    print(f"Processing {total_docs} documents")
    for doc_id, doc_data in tqdm(data.items()):
        # Skip if no annotations
        if "annotations" not in doc_data or not doc_data["annotations"]:
            continue

        docs_with_annotations += 1

        # Process each annotation
        for i, annotation in enumerate(doc_data["annotations"]):
            total_annotations += 1

            # Check if ordo_with_desc is missing
            if "ordo_with_desc" not in annotation or not annotation["ordo_with_desc"]:
                # Get the mention text
                mention = annotation.get("mention", "")

                if not mention:
                    continue  # Skip if no mention text

                # Retrieve similar Orphanet entries
                try:
                    # Embed the query
                    query_vector = embedding_manager.query_text(mention).reshape(1, -1)

                    # Search for similar items
                    distances, indices = embedding_manager.search(
                        query_vector, index, k=top_k
                    )

                    # Get the top match
                    if indices.size > 0 and indices[0].size > 0:
                        top_match_idx = indices[0][0]
                        top_match = embedded_documents[top_match_idx]

                        # Extract Orphanet ID and name
                        orphanet_id = top_match.get("id", "")
                        orphanet_name = top_match.get("name", "")

                        # Format as "ORPHA_ID NAME"
                        ordo_with_desc = f"{orphanet_id} {orphanet_name}"

                        # Update the annotation
                        annotation["ordo_with_desc"] = ordo_with_desc
                        annotation["auto_enriched"] = True
                        annotation["similarity_score"] = float(
                            1.0 / (1.0 + distances[0][0])
                        )

                        annotations_enriched += 1

                except Exception as e:
                    print(f"Error enriching annotation for mention '{mention}': {e}")

    # Print statistics
    print("\nEnrichment Statistics:")
    print(f"Total documents: {total_docs}")
    print(f"Documents with annotations: {docs_with_annotations}")
    print(f"Total annotations: {total_annotations}")
    print(f"Annotations enriched with Orphanet codes: {annotations_enriched}")

    # Save enriched dataset
    if output_path:
        print(f"Saving enriched dataset to {output_path}")
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

    return data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Enrich dataset annotations with Orphanet codes"
    )

    parser.add_argument(
        "--dataset", required=True, help="Path to the dataset JSON file"
    )
    parser.add_argument(
        "--embeddings_file", required=True, help="Path to the embeddings file"
    )
    parser.add_argument(
        "--output", required=True, help="Path to save the enriched dataset"
    )
    parser.add_argument(
        "--retriever",
        default="sentence_transformer",
        choices=["fastembed", "sentence_transformer", "medcpt"],
        help="Type of retriever to use",
    )
    parser.add_argument(
        "--retriever_model",
        default="abhinand/MedEmbed-small-v0.1",
        help="Name of the retriever model",
    )
    parser.add_argument(
        "--device", default="cpu", help="Device to use (cpu, cuda, cuda:0, etc.)"
    )
    parser.add_argument(
        "--top_k", type=int, default=1, help="Number of top candidates to retrieve"
    )

    args = parser.parse_args()

    enrich_annotations_with_orpha_codes(
        dataset_path=args.dataset,
        embeddings_file=args.embeddings_file,
        retriever=args.retriever,
        retriever_model=args.retriever_model,
        output_path=args.output,
        device=args.device,
        top_k=args.top_k,
    )
