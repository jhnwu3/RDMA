"""
Convert benchmark JSONL files to paired CSV files.

Each JSONL entry has: {id, text, annotations: [...]}

Output per benchmark:
  RDMA/public_data/{benchmark}/texts.csv        -- id, text
  RDMA/public_data/{benchmark}/annotations.csv  -- id, annotation (one row per mention)
"""

import json
import csv
from pathlib import Path

BENCHMARKS = {
    "raredis": "workspace/results/benchmark_raredis.jsonl",
    "rdd":     "workspace/results/benchmark_rdd.jsonl",
}

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # rare_disease/


def convert(benchmark_name: str, jsonl_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    texts_path = out_dir / "texts.csv"
    ann_path   = out_dir / "annotations.csv"

    with (
        open(jsonl_path)          as f_in,
        open(texts_path, "w", newline="") as f_texts,
        open(ann_path,   "w", newline="") as f_ann,
    ):
        texts_writer = csv.writer(f_texts)
        ann_writer   = csv.writer(f_ann)

        texts_writer.writerow(["id", "text"])
        ann_writer.writerow(["id", "annotation"])

        for line in f_in:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            doc_id = entry["id"]
            texts_writer.writerow([doc_id, entry["text"]])
            for mention in entry.get("annotations", []):
                ann_writer.writerow([doc_id, mention])

    print(f"[{benchmark_name}] texts     -> {texts_path}")
    print(f"[{benchmark_name}] annotations -> {ann_path}")


def main():
    for name, rel_path in BENCHMARKS.items():
        jsonl_path = PROJECT_ROOT / rel_path
        out_dir    = PROJECT_ROOT / "RDMA" / "public_data" / name
        convert(name, jsonl_path, out_dir)


if __name__ == "__main__":
    main()
