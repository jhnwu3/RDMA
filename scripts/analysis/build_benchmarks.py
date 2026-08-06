"""
Build two separate NER benchmark JSONL files:

  results/benchmark_raredis.jsonl  -- from results/train.json + results/test.json
  results/benchmark_rdd.jsonl      -- from RDD_corpus_annotations.json

Each line in the output JSONL has the format:
  {"id": "...", "text": "...", "annotations": ["term1", "term2", ...]}

Annotations are the unique surface-form strings of all entities, in order of
first appearance.
"""

import json
import argparse
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def dedupe_ordered(items):
    """Return items with duplicates removed, preserving first-seen order."""
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


# ---------------------------------------------------------------------------
# RareDis (brat_to_json output) -> benchmark rows
# ---------------------------------------------------------------------------


def raredis_to_rows(json_paths):
    rows = []
    for path in json_paths:
        with open(path) as f:
            docs = json.load(f)
        for doc in docs:
            text = doc.get("text", "")
            terms = [
                ann["text"]
                for ann in doc.get("annotations", [])
                if ann.get("text", "").strip()
            ]
            rows.append(
                {
                    "id": doc["id"],
                    "text": text,
                    "annotations": dedupe_ordered(terms),
                }
            )
    return rows


# ---------------------------------------------------------------------------
# RDD corpus (flat t1/t2/... keys) -> benchmark rows
# ---------------------------------------------------------------------------


def rdd_to_rows(json_path):
    with open(json_path) as f:
        data = json.load(f)

    rows = []
    for entry in data:
        doc_id = entry.get("id", "")
        text = entry.get("text", "")

        # Collect annotations in key order, skip incomplete ones
        anns = []
        for key, value in entry.items():
            if key in ("id", "text") or not key.startswith("t"):
                continue
            if not isinstance(value, dict):
                continue
            if not all(k in value for k in ("type", "start", "end", "text")):
                continue
            anns.append((value["start"], value["text"]))

        # Sort by start offset so terms appear in reading order
        anns.sort(key=lambda x: x[0])
        terms = dedupe_ordered(t for _, t in anns if t.strip())

        rows.append({"id": doc_id, "text": text, "annotations": terms})

    return rows


# ---------------------------------------------------------------------------
# Write JSONL
# ---------------------------------------------------------------------------


def write_jsonl(rows, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(rows):>5} rows -> {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Build NER benchmark JSONL files.")
    parser.add_argument("--raredis-train", default="results/train.json")
    parser.add_argument("--raredis-test", default="results/test.json")
    parser.add_argument("--rdd", default="RDD_corpus_annotations.json")
    parser.add_argument("--out-raredis", default="results/benchmark_raredis.jsonl")
    parser.add_argument("--out-rdd", default="results/benchmark_rdd.jsonl")
    args = parser.parse_args()

    print("Building RareDis benchmark ...")
    raredis_rows = raredis_to_rows([args.raredis_train, args.raredis_test])
    write_jsonl(raredis_rows, args.out_raredis)

    print("Building RDD benchmark ...")
    rdd_rows = rdd_to_rows(args.rdd)
    write_jsonl(rdd_rows, args.out_rdd)


if __name__ == "__main__":
    main()
