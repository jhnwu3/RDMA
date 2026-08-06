"""
Convert BRAT standoff (.txt + .ann) file pairs to a standardized JSON format.

Input (.ann) format:
  T1\tRAREDISEASE 0 19\tIsovaleric Acidemia
  R1\tIs_a Arg1:T1 Arg2:T2

Output JSON format:
  {
    "id": "Acidemia-Isovaleric",
    "text": "...",
    "annotations": [
      {"id": "T1", "type": "RAREDISEASE", "start": 0, "end": 19, "text": "Isovaleric Acidemia"},
      ...
    ],
    "relations": [
      {"id": "R1", "type": "Is_a", "arg1": "T1", "arg2": "T2"},
      ...
    ]
  }

Usage:
  # Convert a single split (test/train/dev) directory:
  python scripts/brat_to_json.py -i repos/RareDis/test -o results/test.json

  # Convert multiple directories at once:
  python scripts/brat_to_json.py -i repos/RareDis/train repos/RareDis/test -o results/combined.json
"""

import json
import argparse
from pathlib import Path


def parse_ann_file(ann_path: Path) -> tuple[list[dict], list[dict]]:
    """Parse a BRAT .ann file into entity and relation lists."""
    entities = []
    relations = []

    with open(ann_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue

            parts = line.split("\t")
            ann_id = parts[0]

            if ann_id.startswith("T") and len(parts) >= 3:
                # Entity: T1\tTYPE start end\ttext
                # Spans may be discontinuous: TYPE start1 end1;start2 end2 ...
                type_span = parts[1].split(" ", 1)
                ann_type = type_span[0]
                raw_spans = type_span[1].split(";") if len(type_span) > 1 else []
                spans = []
                for raw in raw_spans:
                    se = raw.strip().split(" ")
                    spans.append({"start": int(se[0]), "end": int(se[1])})
                text = parts[2] if len(parts) > 2 else ""
                entities.append(
                    {
                        "id": ann_id,
                        "type": ann_type,
                        "spans": spans,
                        "text": text,
                    }
                )

            elif ann_id.startswith("R") and len(parts) >= 2:
                # Relation: R1\tTYPE Arg1:T1 Arg2:T2
                rel_parts = parts[1].split(" ")
                rel_type = rel_parts[0]
                arg1 = rel_parts[1].split(":")[1] if len(rel_parts) > 1 else ""
                arg2 = rel_parts[2].split(":")[1] if len(rel_parts) > 2 else ""
                relations.append(
                    {
                        "id": ann_id,
                        "type": rel_type,
                        "arg1": arg1,
                        "arg2": arg2,
                    }
                )

    # Sort entities by first span start offset
    entities.sort(key=lambda x: x["spans"][0]["start"] if x["spans"] else 0)
    return entities, relations


def convert_directory(input_dir: Path) -> list[dict]:
    """Convert all .txt/.ann pairs in a directory to a list of document dicts."""
    documents = []

    txt_files = sorted(input_dir.glob("*.txt"))
    if not txt_files:
        print(f"  No .txt files found in {input_dir}")
        return documents

    for txt_path in txt_files:
        ann_path = txt_path.with_suffix(".ann")
        if not ann_path.exists():
            print(f"  Warning: no .ann file for {txt_path.name}, skipping")
            continue

        doc_id = txt_path.stem
        text = txt_path.read_text(encoding="utf-8")
        entities, relations = parse_ann_file(ann_path)

        doc = {
            "id": doc_id,
            "text": text,
            "annotations": entities,
        }
        if relations:
            doc["relations"] = relations

        documents.append(doc)

    return documents


def main():
    parser = argparse.ArgumentParser(
        description="Convert BRAT .txt/.ann pairs to standardized JSON."
    )
    parser.add_argument(
        "--input",
        "-i",
        nargs="+",
        required=True,
        help="One or more directories containing .txt and .ann files.",
    )
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Path to write the output JSON file.",
    )
    args = parser.parse_args()

    all_documents = []
    for dir_str in args.input:
        input_dir = Path(dir_str)
        if not input_dir.is_dir():
            print(f"Warning: {input_dir} is not a directory, skipping.")
            continue
        print(f"Processing {input_dir} ...")
        docs = convert_directory(input_dir)
        print(f"  Found {len(docs)} documents")
        all_documents.extend(docs)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_documents, f, indent=2, ensure_ascii=False)

    total_ann = sum(len(d["annotations"]) for d in all_documents)
    total_rel = sum(len(d.get("relations", [])) for d in all_documents)
    print(f"\nDocuments  : {len(all_documents)}")
    print(f"Annotations: {total_ann}")
    print(f"Relations  : {total_rel}")
    print(f"Output     : {output_path}")


if __name__ == "__main__":
    main()
