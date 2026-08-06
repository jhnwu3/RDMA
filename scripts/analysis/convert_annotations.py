"""
Convert RDD_corpus_annotations.json from flat t1/t2/... annotation keys
to a standardized format with "text" and "annotations" array.

Input format:
  { "id": "...", "text": "...", "t1": {"type": ..., "start": ..., "end": ..., "text": ...}, ... }

Output format:
  { "id": "...", "text": "...", "annotations": [{"id": "t1", "type": ..., "start": ..., "end": ..., "text": ...}, ...] }
"""

import json
import argparse
from pathlib import Path


def convert(input_path: str, output_path: str) -> None:
    with open(input_path, "r") as f:
        data = json.load(f)

    standardized = []
    skipped_annotations = 0

    for entry in data:
        doc_id = entry.get("id", "")
        text = entry.get("text", "")

        annotations = []
        for key, value in entry.items():
            if key in ("id", "text"):
                continue
            if not key.startswith("t"):
                continue
            if not isinstance(value, dict):
                continue
            # Skip incomplete annotations
            if not all(k in value for k in ("type", "start", "end", "text")):
                skipped_annotations += 1
                continue

            annotations.append(
                {
                    "id": key,
                    "type": value["type"],
                    "start": value["start"],
                    "end": value["end"],
                    "text": value["text"],
                }
            )

        # Sort by start offset
        annotations.sort(key=lambda x: x["start"])

        standardized.append(
            {
                "id": doc_id,
                "text": text,
                "annotations": annotations,
            }
        )

    with open(output_path, "w") as f:
        json.dump(standardized, f, indent=2, ensure_ascii=False)

    total_ann = sum(len(d["annotations"]) for d in standardized)
    print(f"Documents : {len(standardized)}")
    print(f"Annotations kept    : {total_ann}")
    print(f"Annotations skipped : {skipped_annotations}")
    print(f"Output written to   : {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert RDD corpus annotations to standardized JSON."
    )
    parser.add_argument(
        "--input",
        "-i",
        default="RDD_corpus_annotations.json",
        help="Path to the input JSON file (default: RDD_corpus_annotations.json)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="RDD_corpus_standardized.json",
        help="Path to write the output JSON file (default: RDD_corpus_standardized.json)",
    )
    args = parser.parse_args()
    convert(args.input, args.output)


if __name__ == "__main__":
    main()
