#!/usr/bin/env python3
"""Aggregate MIMIC3 + RareDis eval outputs into CSV and Markdown tables.

Reads the unified manifest and produces:
- results/rare_disease_eval_matrix.csv
- results/rare_disease_eval_matrix.md

Pass --run_evals to sequentially run any missing evals before aggregating.
Pass --force_eval to re-run even already-evaluated entries.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


DEFAULT_WORKSPACE = Path("/home/johnwu3/projects/rare_disease/workspace")
DEFAULT_MANIFEST = (
    DEFAULT_WORKSPACE / "condor" / "rare_disease" / "eval_manifest_mimic3_raredis.tsv"
)
DEFAULT_CSV = DEFAULT_WORKSPACE / "results" / "rare_disease_eval_matrix.csv"
DEFAULT_MD = DEFAULT_WORKSPACE / "results" / "rare_disease_eval_matrix.md"
RDMA_DIR = DEFAULT_WORKSPACE / "repos" / "RDMA"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate rare disease eval matrix")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--csv_out", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--md_out", type=Path, default=DEFAULT_MD)
    parser.add_argument(
        "--run_evals",
        action="store_true",
        help="Sequentially run eval for any row with predictions but no eval output",
    )
    parser.add_argument(
        "--force_eval",
        action="store_true",
        help="Re-run eval even if eval_output already exists (implies --run_evals)",
    )
    return parser.parse_args()


def run_eval(row: Dict[str, str]) -> bool:
    """Run the appropriate eval script for a manifest row. Returns True on success."""
    dataset = row["dataset"]
    track = row["track"]
    approach = row["approach"]
    model_type = row["model_type"]
    predictions_file = row["predictions_file"]

    print(
        f"[eval] dataset={dataset} track={track} approach={approach} model_type={model_type}"
    )

    if dataset == "mimic3" and track == "code":
        cmd = [
            sys.executable,
            str(RDMA_DIR / "scripts" / "mimic3_rd_mining_code" / "eval.py"),
            "--model_type", model_type,
            "--approach", approach,
            "--predictions_file", predictions_file,
        ]
    elif dataset == "mimic3" and track == "text":
        cmd = [
            sys.executable,
            str(RDMA_DIR / "scripts" / "mimic3_rd_mining_text" / "eval.py"),
            "--model_type", model_type,
            "--approach", approach,
            "--predictions_file", predictions_file,
            "--condor",
        ]
    elif dataset == "raredis":
        cmd = [
            sys.executable,
            str(RDMA_DIR / "scripts" / "raredis" / "eval.py"),
            "--model_type", model_type,
            "--approach", approach,
        ]
    elif dataset == "rdd":
        cmd = [
            sys.executable,
            str(RDMA_DIR / "scripts" / "rdd" / "eval.py"),
            "--model_type", model_type,
            "--approach", approach,
        ]
    else:
        print(f"[eval] SKIP unsupported dataset={dataset} track={track}")
        return False

    result = subprocess.run(cmd, cwd=RDMA_DIR)
    if result.returncode != 0:
        print(f"[eval] FAILED returncode={result.returncode} for {model_type}/{approach}")
        return False
    return True


def load_manifest(path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 6:
                raise ValueError(f"Manifest row must have 6 fields: {line}")
            dataset, track, approach, model_type, predictions_file, eval_output = parts
            rows.append(
                {
                    "dataset": dataset,
                    "track": track,
                    "approach": approach,
                    "model_type": model_type,
                    "predictions_file": predictions_file,
                    "eval_output": eval_output,
                }
            )
    return rows


def load_metrics(eval_output: Path) -> Dict[str, object]:
    with eval_output.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    metrics = payload.get("metrics", {})
    return {
        "precision": metrics.get("precision"),
        "recall": metrics.get("recall"),
        "f1": metrics.get("f1"),
        "tp": metrics.get("tp"),
        "fp": metrics.get("fp"),
        "fn": metrics.get("fn"),
        "documents_scored": metrics.get(
            "documents_scored", metrics.get("notes_scored")
        ),
    }


def fmt_float(v: object) -> str:
    if isinstance(v, (int, float)):
        return f"{float(v):.4f}"
    return ""


def write_csv(rows: List[Dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "track",
        "approach",
        "model_type",
        "status",
        "predictions_file",
        "eval_output",
        "precision",
        "recall",
        "f1",
        "tp",
        "fp",
        "fn",
        "documents_scored",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: List[Dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "dataset",
        "track",
        "approach",
        "model_type",
        "status",
        "f1",
        "precision",
        "recall",
        "documents_scored",
        "eval_output",
    ]
    with path.open("w", encoding="utf-8") as f:
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + "|".join(["---"] * len(headers)) + "|\n")
        for row in rows:
            md_row = [
                str(row.get("dataset", "")),
                str(row.get("track", "")),
                str(row.get("approach", "")),
                str(row.get("model_type", "")),
                str(row.get("status", "")),
                fmt_float(row.get("f1")),
                fmt_float(row.get("precision")),
                fmt_float(row.get("recall")),
                str(row.get("documents_scored", "")),
                str(row.get("eval_output", "")),
            ]
            f.write("| " + " | ".join(md_row) + " |\n")


def main() -> None:
    args = parse_args()
    run_evals = args.run_evals or args.force_eval
    manifest_rows = load_manifest(args.manifest)

    out_rows: List[Dict[str, object]] = []
    for row in manifest_rows:
        pred_path = Path(row["predictions_file"])
        eval_path = Path(row["eval_output"])

        status = "pending"
        metrics: Dict[str, object] = {
            "precision": "",
            "recall": "",
            "f1": "",
            "tp": "",
            "fp": "",
            "fn": "",
            "documents_scored": "",
        }

        if not pred_path.exists():
            status = "missing_predictions"
        else:
            need_eval = args.force_eval or not eval_path.exists()
            if need_eval and run_evals:
                run_eval(row)
            if eval_path.exists():
                status = "evaluated"
                metrics = load_metrics(eval_path)
            else:
                status = "missing_eval"
        out_rows.append(
            {
                **row,
                "status": status,
                **metrics,
            }
        )

    out_rows.sort(
        key=lambda r: (
            str(r["dataset"]),
            str(r["track"]),
            str(r["approach"]),
            str(r["model_type"]),
        )
    )

    write_csv(out_rows, args.csv_out)
    write_markdown(out_rows, args.md_out)

    total = len(out_rows)
    evaluated = sum(1 for r in out_rows if r["status"] == "evaluated")
    missing_predictions = sum(
        1 for r in out_rows if r["status"] == "missing_predictions"
    )
    missing_eval = sum(1 for r in out_rows if r["status"] == "missing_eval")

    print(f"Rows: {total}")
    print(f"evaluated: {evaluated}")
    print(f"missing_predictions: {missing_predictions}")
    print(f"missing_eval: {missing_eval}")
    print(f"CSV: {args.csv_out}")
    print(f"Markdown: {args.md_out}")


if __name__ == "__main__":
    main()
