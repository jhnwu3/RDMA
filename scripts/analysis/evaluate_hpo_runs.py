#!/usr/bin/env python3
"""Aggregate Precision/Recall/F1 across all result runs in CSC and BioLarkGSC.

Scans JSONL files in:
- results/csc/*.jsonl
- results/biolarkgsc/*.jsonl

Expected per-line schema (flexible):
{
  "id": "...",
  "predicted": ["HP:...", ...],
  "ground_truth": ["HP:...", ...]
}

Notes:
- Deduplicates predictions and gold per document before counting.
- Handles comma/semicolon-separated HPO IDs in a single string value.
- Normalizes IDs like HP_0001250 -> HP:0001250.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Set


@dataclass
class Metrics:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    n_docs: int = 0
    n_bad_lines: int = 0

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p = self.precision
        r = self.recall
        return (2 * p * r / (p + r)) if (p + r) else 0.0


def normalize_hpo_id(raw: str) -> str:
    s = raw.strip().upper()
    if not s:
        return ""
    if s.startswith("HP_"):
        s = s.replace("_", ":", 1)
    return s


def split_candidate_ids(value: str) -> List[str]:
    # Handles strings like "HP:0001388, HP:0001382".
    parts = [p.strip() for p in value.replace(";", ",").split(",")]
    return [p for p in parts if p]


def flatten_hpo_values(values: object) -> Set[str]:
    out: Set[str] = set()

    if values is None:
        return out

    if isinstance(values, str):
        for part in split_candidate_ids(values):
            hp = normalize_hpo_id(part)
            if hp.startswith("HP:"):
                out.add(hp)
        return out

    if isinstance(values, Sequence) and not isinstance(values, (bytes, bytearray, str)):
        for item in values:
            if isinstance(item, str):
                for part in split_candidate_ids(item):
                    hp = normalize_hpo_id(part)
                    if hp.startswith("HP:"):
                        out.add(hp)
            elif isinstance(item, dict):
                # Be permissive for alternate structures.
                cand = item.get("hpo_id") or item.get("hp_id") or item.get("id")
                if isinstance(cand, str):
                    hp = normalize_hpo_id(cand)
                    if hp.startswith("HP:"):
                        out.add(hp)
    return out


def get_predicted_ids(obj: dict) -> Set[str]:
    for key in ("predicted", "prediction", "predictions"):
        if key in obj:
            return flatten_hpo_values(obj[key])
    return set()


def get_gold_ids(obj: dict) -> Set[str]:
    for key in ("ground_truth", "gold", "labels", "target"):
        if key in obj:
            return flatten_hpo_values(obj[key])
    return set()


def evaluate_jsonl(path: Path):
    """Return (Metrics, per_docs) where per_docs is a list of per-document dicts."""
    m = Metrics()
    per_docs: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                m.n_bad_lines += 1
                continue

            if not isinstance(obj, dict):
                m.n_bad_lines += 1
                continue

            pred = get_predicted_ids(obj)
            gold = get_gold_ids(obj)
            tp_ids = pred & gold
            fp_ids = pred - gold
            fn_ids = gold - pred

            m.tp += len(tp_ids)
            m.fp += len(fp_ids)
            m.fn += len(fn_ids)
            m.n_docs += 1

            per_docs.append({
                "id": str(obj.get("id", m.n_docs)),
                "tp": len(tp_ids),
                "fp": len(fp_ids),
                "fn": len(fn_ids),
                "matched": sorted(tp_ids),
                "fp_ids": sorted(fp_ids),
                "fn_ids": sorted(fn_ids),
            })

    return m, per_docs


def _bootstrap_f1(per_docs: List[dict], n_bootstrap: int, seed: int) -> tuple:
    """Return (ci_lower, ci_upper) via document-level bootstrap resampling."""
    import random
    rng = random.Random(seed)
    n = len(per_docs)
    if n == 0:
        return 0.0, 0.0
    samples = []
    for _ in range(n_bootstrap):
        idx = rng.choices(range(n), k=n)
        tp = sum(per_docs[i]["tp"] for i in idx)
        fp = sum(per_docs[i]["fp"] for i in idx)
        fn = sum(per_docs[i]["fn"] for i in idx)
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        samples.append(2 * p * r / (p + r) if (p + r) > 0 else 0.0)
    samples.sort()
    lo = samples[max(0, int(0.025 * n_bootstrap) - 1)]
    hi = samples[min(n_bootstrap - 1, int(0.975 * n_bootstrap))]
    return round(lo, 6), round(hi, 6)


def _write_audit_json(path: Path, metrics: Metrics, per_docs: List[dict]) -> None:
    """Write per-document audit JSON alongside the prediction file."""
    stem = path.stem
    if stem.endswith("_predictions"):
        stem = stem[: -len("_predictions")]
    audit_path = path.parent / f"eval_{stem}.json"
    payload = {
        "metrics": {
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1": metrics.f1,
            "tp": metrics.tp,
            "fp": metrics.fp,
            "fn": metrics.fn,
            "documents_scored": metrics.n_docs,
        },
        "documents": per_docs,
    }
    audit_path.write_text(json.dumps(payload, indent=2))
    return audit_path


def format_pct(x: float) -> str:
    return f"{x * 100:.2f}"


def render_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    str_rows = [[str(c) for c in row] for row in rows]
    widths = [len(h) for h in headers]
    for row in str_rows:
        for i, c in enumerate(row):
            widths[i] = max(widths[i], len(c))

    def fmt_row(cells: Sequence[str]) -> str:
        return " | ".join(cells[i].ljust(widths[i]) for i in range(len(cells)))

    sep = "-+-".join("-" * w for w in widths)
    lines = [fmt_row(list(headers)), sep]
    lines.extend(fmt_row(row) for row in str_rows)
    return "\n".join(lines)


def find_result_files(dataset_dir: Path) -> List[Path]:
    return sorted(p for p in dataset_dir.glob("*.jsonl") if p.is_file())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate HPO run metrics across result folders"
    )
    parser.add_argument(
        "--results_root",
        type=Path,
        default=Path("/home/johnwu3/projects/rare_disease/workspace/results"),
        help="Root results directory containing csc/ and biolarkgsc/",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=[
            "csc",
            "biolarkgsc",
            "biolarkgsc/ablation",
            "biolarkgsc/revised_hpo",
            "biolarkgsc/revised_hpo_simple",
            "biolarkgsc/ablation_mistral24b",
            "csc/rdma_simple_extractor",
        ],
        help="Dataset subdirectories to include",
    )
    parser.add_argument(
        "--csv_out",
        type=Path,
        default=None,
        help="Optional path to write combined metrics CSV",
    )
    parser.add_argument(
        "--write_audit_jsons",
        action="store_true",
        help="Write per-document audit JSON (eval_<run>.json) next to each prediction file",
    )
    parser.add_argument(
        "--bootstrap",
        action="store_true",
        help="Compute 95%% bootstrap CIs (requires --write_audit_jsons or runs inline)",
    )
    parser.add_argument("--n_bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = []
    for dataset in args.datasets:
        dataset_dir = args.results_root / dataset
        if not dataset_dir.exists():
            continue

        for file_path in find_result_files(dataset_dir):
            metrics, per_docs = evaluate_jsonl(file_path)
            stem = file_path.stem
            approach = stem[: -len("_predictions")] if stem.endswith("_predictions") else stem

            ci_lower, ci_upper = "", ""
            if args.bootstrap and per_docs:
                ci_lower, ci_upper = _bootstrap_f1(per_docs, args.n_bootstrap, args.seed)

            audit_path = ""
            if args.write_audit_jsons and per_docs:
                audit_path = str(_write_audit_json(file_path, metrics, per_docs))

            rows.append(
                {
                    "dataset": dataset,
                    "approach": approach,
                    "file": str(file_path),
                    "audit_json": audit_path,
                    "n_docs": metrics.n_docs,
                    "bad_lines": metrics.n_bad_lines,
                    "tp": metrics.tp,
                    "fp": metrics.fp,
                    "fn": metrics.fn,
                    "precision": metrics.precision,
                    "recall": metrics.recall,
                    "f1": metrics.f1,
                    "f1_ci_lower": ci_lower,
                    "f1_ci_upper": ci_upper,
                }
            )

    if not rows:
        print("No result files found.")
        return

    rows.sort(key=lambda r: (r["dataset"], -r["f1"], r["approach"]))

    show_ci = args.bootstrap and any(r["f1_ci_lower"] != "" for r in rows)
    combined_headers = ["dataset", "approach", "n_docs", "precision", "recall", "f1"]
    if show_ci:
        combined_headers.append("95% CI")
    combined_headers += ["tp", "fp", "fn", "bad_lines"]

    def _ci_str(r):
        if r["f1_ci_lower"] == "":
            return "-"
        return f"[{float(r['f1_ci_lower']):.4f}, {float(r['f1_ci_upper']):.4f}]"

    combined_rows = []
    for r in rows:
        row = [
            r["dataset"],
            r["approach"],
            r["n_docs"],
            format_pct(r["precision"]),
            format_pct(r["recall"]),
            format_pct(r["f1"]),
        ]
        if show_ci:
            row.append(_ci_str(r))
        row += [r["tp"], r["fp"], r["fn"], r["bad_lines"]]
        combined_rows.append(row)

    print("\n=== Combined Metrics (micro, per run file) ===")
    print(render_table(combined_headers, combined_rows))

    # F1 pivot by approach for quick all-at-once comparison.
    dataset_keys = sorted(set(r["dataset"] for r in rows))
    approaches = sorted(set(r["approach"] for r in rows))
    f1_lookup = {(r["approach"], r["dataset"]): format_pct(r["f1"]) for r in rows}

    pivot_headers = ["approach"] + [f"{d}_f1" for d in dataset_keys]
    pivot_rows = []
    for a in approaches:
        row = [a] + [f1_lookup.get((a, d), "-") for d in dataset_keys]
        pivot_rows.append(row)

    print("\n=== F1 Pivot (all approaches at once) ===")
    print(render_table(pivot_headers, pivot_rows))

    if args.csv_out is not None:
        args.csv_out.parent.mkdir(parents=True, exist_ok=True)
        with args.csv_out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "dataset",
                    "approach",
                    "file",
                    "audit_json",
                    "n_docs",
                    "bad_lines",
                    "tp",
                    "fp",
                    "fn",
                    "precision",
                    "recall",
                    "f1",
                    "f1_ci_lower",
                    "f1_ci_upper",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote CSV: {args.csv_out}")


if __name__ == "__main__":
    main()
