#!/usr/bin/env python3
"""Process ANN + Original text file pairs into a single JSON file."""

import json
import re
from pathlib import Path

ANN_DIR = Path(__file__).resolve().parents[1] / "repos/RDD Corpus/Corpus/Annotated texts/ANN"
ORIGINAL_DIR = Path(__file__).resolve().parents[1] / "repos/RDD Corpus/Corpus/Original texts"
OUTPUT_JSON = Path(__file__).resolve().parents[1] / "corpus_annotations.json"


def parse_ann_line(line: str) -> dict | None:
    """Parse a single T-line from .ann. Returns None if not a T-annotation."""
    line = line.strip()
    if not line or not line.startswith("T"):
        return None
    # T1\tDisability 27 46\tdevelopmental delay
    parts = line.split("\t")
    if len(parts) < 3:
        return None
    tid = parts[0]  # T1, T2, ...
    if not re.match(r"^T\d+$", tid):
        return None
    middle = parts[1]  # "Disability 27 46"
    text = parts[2]   # "developmental delay"
    tokens = middle.split()
    if len(tokens) < 3:
        return None
    type_name = tokens[0]
    try:
        start, end = int(tokens[1]), int(tokens[2])
    except ValueError:
        return None
    return {
        "id": tid.lower(),
        "type": type_name,
        "start": start,
        "end": end,
        "text": text,
    }


def process_file_pair(ann_path: Path, txt_path: Path) -> dict:
    """Build one JSON entry from an .ann and .txt pair."""
    base_name = ann_path.stem
    annotations = {}
    with open(ann_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parsed = parse_ann_line(line)
            if parsed:
                annotations[parsed["id"]] = {
                    "type": parsed["type"],
                    "start": parsed["start"],
                    "end": parsed["end"],
                    "text": parsed["text"],
                }
    with open(txt_path, "r", encoding="utf-8", errors="replace") as f:
        full_text = f.read()
    return {
        "id": base_name,
        "text": full_text,
        **annotations,
    }


def main():
    ann_files = sorted(ANN_DIR.glob("*.ann"))
    results = []
    skipped = []
    for ann_path in ann_files:
        txt_path = ORIGINAL_DIR / (ann_path.stem + ".txt")
        if not txt_path.exists():
            skipped.append(ann_path.name)
            continue
        results.append(process_file_pair(ann_path, txt_path))
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(results)} entries to {OUTPUT_JSON}")
    if skipped:
        print(f"Skipped {len(skipped)} .ann files (no matching .txt): {skipped[:5]}...")


if __name__ == "__main__":
    main()
