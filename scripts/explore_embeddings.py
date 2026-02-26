#!/usr/bin/env python3
"""
Explore the structure of the ORPHA embeddings .npy file.

Usage (from RDMA repo root):
  python scripts/explore_embeddings.py
  python scripts/explore_embeddings.py --search "disorder"
  python scripts/explore_embeddings.py --search "Fabry disease" --top_k 5
"""

import argparse
import sys
import numpy as np
from pathlib import Path

_RDMA_ROOT = Path("/home/johnwu3/projects/rare_disease/workspace/repos/RDMA")
sys.path.insert(0, str(_RDMA_ROOT))

_DEFAULT_EMBEDDINGS_FILE = (
    _RDMA_ROOT / "data/vector_stores/rd_orpha_medembed.npy"
)
_DEFAULT_MODEL_CACHE_DIR = "/shared/rsaas/jw3/rare_disease/model_cache"


def inspect_entry(doc, idx):
    """Pretty-print a single embedded document."""
    print(f"\n--- Entry [{idx}] ---")
    if isinstance(doc, dict):
        for k, v in doc.items():
            if k == "embedding":
                arr = np.array(v)
                print(f"  embedding : shape={arr.shape}, dtype={arr.dtype}")
            elif isinstance(v, dict):
                print(f"  {k}:")
                for kk, vv in v.items():
                    val_str = str(vv)
                    if len(val_str) > 200:
                        val_str = val_str[:200] + "..."
                    print(f"    {kk}: {val_str!r}")
            else:
                val_str = str(v)
                if len(val_str) > 200:
                    val_str = val_str[:200] + "..."
                print(f"  {k}: {val_str!r}")
    else:
        print(f"  (non-dict entry) type={type(doc)}: {str(doc)[:200]}")


def main():
    parser = argparse.ArgumentParser(description="Explore ORPHA embeddings .npy file")
    parser.add_argument(
        "--embeddings_file",
        type=Path,
        default=_DEFAULT_EMBEDDINGS_FILE,
        help="Path to .npy embeddings file (default: %(default)s)",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=5,
        help="Number of sample entries to print from the start (default: %(default)s)",
    )
    parser.add_argument(
        "--search",
        type=str,
        default=None,
        help="Search for a specific term by name (exact, case-insensitive substring)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Max hits to show for --search (default: %(default)s)",
    )
    parser.add_argument(
        "--retriever_model",
        type=str,
        default="abhinand/MedEmbed-small-v0.1",
        help="Retriever model for semantic search (default: %(default)s)",
    )
    parser.add_argument(
        "--gpu_id",
        type=lambda x: None if x.lower() == "none" else int(x),
        default=None,
        metavar="N|none",
        help="GPU id for embedding model; 'none' for CPU (default: %(default)s)",
    )
    parser.add_argument(
        "--semantic",
        action="store_true",
        help="Use semantic (embedding) search instead of substring search",
    )
    args = parser.parse_args()

    print(f"Loading: {args.embeddings_file}")
    docs = np.load(args.embeddings_file, allow_pickle=True)
    print(f"Total entries: {len(docs)}")

    # ── Field inventory ──────────────────────────────────────────────────
    print("\n=== Field inventory (first 50 entries) ===")
    field_counts: dict = {}
    for doc in docs[:50]:
        if isinstance(doc, dict):
            for k in doc.keys():
                field_counts[k] = field_counts.get(k, 0) + 1
                if isinstance(doc[k], dict):
                    for kk in doc[k].keys():
                        nested = f"{k}.{kk}"
                        field_counts[nested] = field_counts.get(nested, 0) + 1
    for field, count in sorted(field_counts.items()):
        print(f"  {field:40s}: {count}/50")

    # ── Sample entries ────────────────────────────────────────────────────
    print(f"\n=== First {args.n_samples} entries ===")
    for i, doc in enumerate(docs[: args.n_samples]):
        inspect_entry(doc, i)

    # ── Search ────────────────────────────────────────────────────────────
    if args.search:
        query = args.search

        if args.semantic:
            # Semantic search via EmbeddingsManager + FAISS
            from rdma.utils.embedding import EmbeddingsManager

            device = "cpu" if args.gpu_id is None else f"cuda:{args.gpu_id}"
            print(f"\n=== Semantic search: '{query}' (device={device}) ===")
            mgr = EmbeddingsManager(
                model_type="sentence_transformer",
                model_name=args.retriever_model,
                device=device,
            )
            embeddings_array = mgr.prepare_embeddings(docs)
            index = mgr.create_index(embeddings_array)
            q_vec = mgr.query_text(query).reshape(1, -1)
            distances, indices = mgr.search(q_vec, index, k=min(args.top_k, len(docs)))
            hits = list(zip(indices[0], distances[0]))
        else:
            # Substring search over name field
            print(f"\n=== Substring search: '{query}' ===")
            query_lower = query.lower()
            hits_raw = []
            for i, doc in enumerate(docs):
                name = ""
                if isinstance(doc, dict):
                    if "unique_metadata" in doc:
                        name = doc["unique_metadata"].get("info", "")
                    else:
                        name = doc.get("name", "")
                if query_lower in name.lower():
                    hits_raw.append((i, 0.0))  # distance=0 placeholder
            hits = hits_raw[: args.top_k]
            print(f"  Found {len(hits_raw)} substring matches (showing first {args.top_k})")

        for rank, (idx, dist) in enumerate(hits):
            doc = docs[int(idx)]
            print(f"\n  Rank {rank + 1} (dist={dist:.4f}):")
            inspect_entry(doc, int(idx))


if __name__ == "__main__":
    main()
