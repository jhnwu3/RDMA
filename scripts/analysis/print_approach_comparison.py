#!/usr/bin/env python3
"""Print a markdown table comparing zeroshot / rdrag / rdma across benchmarks.

Usage:
    python scripts/print_approach_comparison.py
    python scripts/print_approach_comparison.py --csv results/rare_disease_eval_matrix.csv
    python scripts/print_approach_comparison.py --metric f1   # f1 | precision | recall
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

APPROACHES = {"zeroshot", "rdrag", "rdma"}
APPROACH_ORDER = ["zeroshot", "rdrag", "rdma"]

METRIC_COLS = {"f1", "precision", "recall"}


def load(csv_path: Path, metric: str):
    # (dataset, track, model_type, approach) -> metric value  (keep best if dupes)
    data: dict[tuple, float] = {}
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["approach"] not in APPROACHES:
                continue
            if row["status"] != "evaluated":
                continue
            key = (row["dataset"], row["track"], row["model_type"], row["approach"])
            val = float(row[metric])
            if key not in data or val > data[key]:
                data[key] = val
    return data


def build_table(data: dict[tuple, float], metric: str) -> str:
    combos: set[tuple[str, str, str]] = set()
    for dataset, track, model_type, _ in data:
        combos.add((dataset, track, model_type))

    rows = []
    for dataset, track, model_type in sorted(combos):
        row = {"dataset": dataset, "track": track, "model": model_type}
        for approach in APPROACH_ORDER:
            val = data.get((dataset, track, model_type, approach))
            row[approach] = f"{val*100:.2f}" if val is not None else "-"
        rows.append(row)

    lines = []
    header = f"| dataset | track | model | {' | '.join(APPROACH_ORDER)} |"
    sep    = f"|---------|-------|-------|{'|'.join(['-------']*len(APPROACH_ORDER))}|"

    lines.append(f"\n### {metric.upper()} comparison: zeroshot vs rdrag vs rdma\n")
    lines.append(header)
    lines.append(sep)

    for dataset in sorted(set(r["dataset"] for r in rows)):
        for r in [r for r in rows if r["dataset"] == dataset]:
            vals = " | ".join(r[a] for a in APPROACH_ORDER)
            lines.append(f"| {r['dataset']} | {r['track']} | {r['model']} | {vals} |")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("/home/johnwu3/projects/rare_disease/workspace/results/rare_disease_eval_matrix.csv"),
    )
    parser.add_argument("--metric", default="f1", choices=sorted(METRIC_COLS))
    args = parser.parse_args()

    data = load(args.csv, args.metric)
    print(build_table(data, args.metric))


if __name__ == "__main__":
    main()
