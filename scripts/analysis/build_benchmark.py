"""
Build a unified NER mining benchmark from two sources:

1. RareDis corpus  - results/train.json + results/test.json  (BRAT-converted)
2. RDD corpus      - RDD_corpus_annotations.json             (flat t1/t2/... format)

Output: results/benchmark.jsonl
Each line:
  {"id": "...", "text": "...", "annotations": ["term1", "term2", ...]}

"annotations" is a deduplicated, sorted list of annotated surface strings
(empty strings and whitespace-only terms are dropped).

Usage:
  python scripts/build_benchmark.py
  python scripts/build_benchmark.py --raredis results/train.json results/test.json \
                                    --rdd RDD_corpus_annotations.json \
                                    --output results/benchmark.jsonl
"""

import json
import argparse
from pathlib import Path


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------


def read_raredis(paths: list[str]) -> list[dict]:
    """Read one or more RareDis JSON files (BRAT-converted format)."""
    documents = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for doc in data:
            terms = []
            for ann in doc.get("annotations", []):
                t = ann.get("text", "").strip()
                if t:
                    terms.append(t)
            documents.append(
                {
                    "id": doc["id"],
                    "text": doc.get("text", ""),
                    "annotations": sorted(set(terms)),
                }
            )
    return documents


def read_rdd(path: str) -> list[dict]:
    """Read the RDD corpus JSON (flat t1/t2/... annotation keys)."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    documents = []
    for entry in data:
        doc_id = entry.get("id", "")
        text = entry.get("text", "")
        terms = []
        for key, value in entry.items():
            if key in ("id", "text") or not key.startswith("t"):
                continue
            if not isinstance(value, dict):
                continue
            t = value.get("text", "").strip()
            if t:
                terms.append(t)
        documents.append(
            {
                "id": doc_id,
                "text": text,
                "annotations": sorted(set(terms)),
            }
        )
    return documents


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Build a unified NER mining benchmark JSONL."
    )
    parser.add_argument(
        "--raredis",
        nargs="+",
        default=["results/train.json", "results/test.json"],
        help="RareDis JSON files (default: results/train.json results/test.json)",
    )
    parser.add_argument(
        "--rdd",
        default="RDD_corpus_annotations.json",
        help="RDD corpus JSON file (default: RDD_corpus_annotations.json)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="results/benchmark.jsonl",
        help="Output JSONL path (default: results/benchmark.jsonl)",
    )
    args = parser.parse_args()

    all_docs = []

    print("Reading RareDis files...")
    raredis_docs = read_raredis(args.raredis)
    print(f"  {len(raredis_docs)} documents")
    all_docs.extend(raredis_docs)

    print("Reading RDD corpus...")
    rdd_docs = read_rdd(args.rdd)
    print(f"  {len(rdd_docs)} documents")
    all_docs.extend(rdd_docs)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for doc in all_docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    total_terms = sum(len(d["annotations"]) for d in all_docs)
    print(f"\nTotal documents : {len(all_docs)}")
    print(f"Total terms     : {total_terms}")
    print(f"Output          : {output_path}")


if __name__ == "__main__":
    main()
