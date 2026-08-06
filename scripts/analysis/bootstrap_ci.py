#!/usr/bin/env python3
"""Compute bootstrap 95% CIs for all benchmarks.

Modes (combinable — use any subset together):

  --eval_json PATH          Single audit JSON → prints CI to stdout
  --glob PATTERN            Glob pattern matching audit JSONs
  --manifest PATH           TSV manifest (dataset track approach model_type pred eval)
  --hpo_results_root PATH   Scan JSONL prediction files, evaluate HPO matching inline
                            (covers CSC/BioLarkGSC and all variant subdirs)
  --all                     Shorthand: default manifest + default HPO dirs

Pass --write_audit_jsons with --hpo_results_root to persist eval_*.json audit files
next to each JSONL so future runs can use --glob or --manifest instead.

Usage examples:
  # Manifest-based (mimic3, raredis, rdd)
  python scripts/bootstrap_ci.py --manifest condor/rare_disease/eval_manifest_all_benchmarks.tsv

  # HPO benchmarks (CSC + BioLarkGSC all variants), write audit JSONs alongside
  python scripts/bootstrap_ci.py --hpo_results_root results/ --write_audit_jsons

  # Everything at once
  python scripts/bootstrap_ci.py --all --csv_out results/bootstrap_ci_all.csv --md_out results/bootstrap_ci_all.md

  # Single file check
  python scripts/bootstrap_ci.py --eval_json results/mimic3_rd_mining/eval_rdma_gpt-5-john.json
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

_WORKSPACE = Path("/home/johnwu3/projects/rare_disease/workspace")
_DEFAULT_MANIFEST = (
    _WORKSPACE / "condor" / "rare_disease" / "eval_manifest_all_benchmarks.tsv"
)
_DEFAULT_HPO_ROOT = _WORKSPACE / "results"
_DEFAULT_HPO_DATASETS = [
    "csc",
    "biolarkgsc",
    "biolarkgsc/ablation",
    "biolarkgsc/revised_hpo",
    "biolarkgsc/revised_hpo_simple",
    "biolarkgsc/ablation_mistral24b",
    "csc/rdma_simple_extractor",
]

# ---------------------------------------------------------------------------
# Shared math helpers
# ---------------------------------------------------------------------------

def _f1(tp: int, fp: int, fn: int) -> float:
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def _doc_counts(d: Dict) -> Tuple[int, int, int]:
    """Extract (tp, fp, fn) from a document dict.

    Two storage formats in the wild:
    - Integer fields (csc/biolarkgsc audit JSONs): d["tp"], d["fp"], d["fn"]
    - List fields (mimic3/raredis/rdd audit JSONs): len(matched_pairs/matched),
      len(fp), len(fn)
    """
    if isinstance(d.get("tp"), int):
        return d["tp"], d["fp"], d["fn"]
    tp = len(d.get("matched_pairs") or d.get("matched") or [])
    fp = len(d.get("fp") or [])
    fn = len(d.get("fn") or [])
    return tp, fp, fn


def bootstrap_ci(
    docs: List[Dict],
    n_bootstrap: int = 1000,
    ci: float = 95.0,
    seed: Optional[int] = 42,
) -> Tuple[float, float, float]:
    """Return (point_f1, ci_lower, ci_upper) via document-level bootstrap."""
    rng = random.Random(seed)
    n = len(docs)
    if n == 0:
        return 0.0, 0.0, 0.0

    counts = [_doc_counts(d) for d in docs]
    total_tp = sum(c[0] for c in counts)
    total_fp = sum(c[1] for c in counts)
    total_fn = sum(c[2] for c in counts)
    point = _f1(total_tp, total_fp, total_fn)

    samples: List[float] = []
    for _ in range(n_bootstrap):
        boot_idx = rng.choices(range(n), k=n)
        tp = sum(counts[i][0] for i in boot_idx)
        fp = sum(counts[i][1] for i in boot_idx)
        fn = sum(counts[i][2] for i in boot_idx)
        samples.append(_f1(tp, fp, fn))

    samples.sort()
    alpha = (100.0 - ci) / 2.0
    lo_idx = max(0, int(alpha / 100.0 * n_bootstrap) - 1)
    hi_idx = min(n_bootstrap - 1, int((100.0 - alpha) / 100.0 * n_bootstrap))
    return point, samples[lo_idx], samples[hi_idx]


# ---------------------------------------------------------------------------
# Audit-JSON mode (manifest / glob / single file)
# ---------------------------------------------------------------------------

def process_file(
    path: Path,
    n_bootstrap: int,
    ci: float,
    seed: Optional[int],
    dataset: str = "",
    track: str = "",
    approach: str = "",
    model_type: str = "",
) -> Optional[Dict]:
    """Load an eval JSON and return a result row dict, or None on error."""
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[skip] {path}: {e}", file=sys.stderr)
        return None

    docs = data.get("documents")
    if not docs:
        print(f"[skip] {path}: no 'documents' array", file=sys.stderr)
        return None

    metrics = data.get("metrics", {})
    point, lo, hi = bootstrap_ci(docs, n_bootstrap=n_bootstrap, ci=ci, seed=seed)

    if not dataset:
        parts = path.parts
        for i, p in enumerate(parts):
            if p == "results" and i + 1 < len(parts):
                dataset = parts[i + 1]
                break

    return {
        "dataset": dataset,
        "track": track,
        "approach": approach,
        "model_type": model_type,
        "eval_json": str(path),
        "f1": round(point, 6),
        "f1_ci_lower": round(lo, 6),
        "f1_ci_upper": round(hi, 6),
        "precision": round(float(metrics.get("precision") or 0.0), 6),
        "recall": round(float(metrics.get("recall") or 0.0), 6),
        "tp": metrics.get("tp", ""),
        "fp": metrics.get("fp", ""),
        "fn": metrics.get("fn", ""),
        "n_docs": len(docs),
        "n_bootstrap": n_bootstrap,
    }


def load_manifest(path: Path) -> List[Dict[str, str]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 6:
                continue
            dataset, track, approach, model_type, _pred, eval_output = parts
            rows.append(
                {
                    "dataset": dataset,
                    "track": track,
                    "approach": approach,
                    "model_type": model_type,
                    "eval_output": eval_output,
                }
            )
    return rows


# ---------------------------------------------------------------------------
# HPO inline evaluation (CSC / BioLarkGSC JSONL prediction files)
# ---------------------------------------------------------------------------

def _normalize_hpo_id(raw: str) -> str:
    s = raw.strip().upper()
    if not s:
        return ""
    if s.startswith("HP_"):
        s = s.replace("_", ":", 1)
    return s


def _split_candidate_ids(value: str) -> List[str]:
    parts = [p.strip() for p in value.replace(";", ",").split(",")]
    return [p for p in parts if p]


def _flatten_hpo_values(values: object) -> Set[str]:
    out: Set[str] = set()
    if values is None:
        return out
    if isinstance(values, str):
        for part in _split_candidate_ids(values):
            hp = _normalize_hpo_id(part)
            if hp.startswith("HP:"):
                out.add(hp)
        return out
    if isinstance(values, Sequence) and not isinstance(values, (bytes, bytearray, str)):
        for item in values:
            if isinstance(item, str):
                for part in _split_candidate_ids(item):
                    hp = _normalize_hpo_id(part)
                    if hp.startswith("HP:"):
                        out.add(hp)
            elif isinstance(item, dict):
                cand = item.get("hpo_id") or item.get("hp_id") or item.get("id")
                if isinstance(cand, str):
                    hp = _normalize_hpo_id(cand)
                    if hp.startswith("HP:"):
                        out.add(hp)
    return out


def _get_predicted_ids(obj: dict) -> Set[str]:
    for key in ("predicted", "prediction", "predictions"):
        if key in obj:
            return _flatten_hpo_values(obj[key])
    return set()


def _get_gold_ids(obj: dict) -> Set[str]:
    for key in ("ground_truth", "gold", "labels", "target"):
        if key in obj:
            return _flatten_hpo_values(obj[key])
    return set()


def _evaluate_hpo_jsonl(path: Path) -> Tuple[Dict, List[Dict]]:
    """Evaluate a prediction JSONL against ground truth HPO IDs.

    Returns (aggregate_metrics_dict, per_document_list).
    """
    total_tp = total_fp = total_fn = n_docs = n_bad = 0
    per_docs: List[Dict] = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                n_bad += 1
                continue
            if not isinstance(obj, dict):
                n_bad += 1
                continue

            pred = _get_predicted_ids(obj)
            gold = _get_gold_ids(obj)
            tp_ids = pred & gold
            fp_ids = pred - gold
            fn_ids = gold - pred

            tp, fp, fn = len(tp_ids), len(fp_ids), len(fn_ids)
            total_tp += tp
            total_fp += fp
            total_fn += fn
            n_docs += 1

            per_docs.append({
                "id": str(obj.get("id", n_docs)),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "matched": sorted(tp_ids),
                "fp_ids": sorted(fp_ids),
                "fn_ids": sorted(fn_ids),
            })

    agg = {
        "precision": _f1_precision(total_tp, total_fp),
        "recall": _f1_recall(total_tp, total_fn),
        "f1": _f1(total_tp, total_fp, total_fn),
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "documents_scored": n_docs,
        "bad_lines": n_bad,
    }
    return agg, per_docs


def _f1_precision(tp: int, fp: int) -> float:
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def _f1_recall(tp: int, fn: int) -> float:
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def _write_audit_json(path: Path, agg: Dict, per_docs: List[Dict]) -> Path:
    """Write per-document audit JSON next to the prediction file."""
    stem = path.stem
    if stem.endswith("_predictions"):
        stem = stem[: -len("_predictions")]
    audit_path = path.parent / f"eval_{stem}.json"
    payload = {
        "metrics": {k: v for k, v in agg.items() if k != "bad_lines"},
        "documents": per_docs,
    }
    audit_path.write_text(json.dumps(payload, indent=2))
    return audit_path


def process_hpo_file(
    path: Path,
    n_bootstrap: int,
    ci: float,
    seed: Optional[int],
    dataset: str,
    write_audit_json: bool = False,
) -> Optional[Dict]:
    """Evaluate a JSONL prediction file via HPO ID matching and compute bootstrap CI."""
    try:
        agg, per_docs = _evaluate_hpo_jsonl(path)
    except Exception as e:
        print(f"[skip] {path}: {e}", file=sys.stderr)
        return None

    if not per_docs:
        print(f"[skip] {path}: no valid documents", file=sys.stderr)
        return None

    audit_path = ""
    if write_audit_json:
        audit_path = str(_write_audit_json(path, agg, per_docs))

    point, lo, hi = bootstrap_ci(per_docs, n_bootstrap=n_bootstrap, ci=ci, seed=seed)

    stem = path.stem
    approach = stem[: -len("_predictions")] if stem.endswith("_predictions") else stem

    return {
        "dataset": dataset,
        "track": "hpo",
        "approach": approach,
        "model_type": "",
        "eval_json": audit_path or str(path),
        "f1": round(point, 6),
        "f1_ci_lower": round(lo, 6),
        "f1_ci_upper": round(hi, 6),
        "precision": round(agg["precision"], 6),
        "recall": round(agg["recall"], 6),
        "tp": agg["tp"],
        "fp": agg["fp"],
        "fn": agg["fn"],
        "n_docs": len(per_docs),
        "n_bootstrap": n_bootstrap,
    }


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_csv(rows: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset", "track", "approach", "model_type",
        "f1", "f1_ci_lower", "f1_ci_upper",
        "precision", "recall", "tp", "fp", "fn",
        "n_docs", "n_bootstrap", "eval_json",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "dataset", "track", "approach", "model_type",
        "f1", "95% CI", "precision", "recall", "n_docs",
    ]
    with path.open("w", encoding="utf-8") as f:
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + "|".join(["---"] * len(headers)) + "|\n")
        for row in rows:
            ci_str = f"[{row['f1_ci_lower']:.4f}, {row['f1_ci_upper']:.4f}]"
            f.write(
                f"| {row['dataset']} | {row['track']} | {row['approach']} "
                f"| {row['model_type']} | {row['f1']:.4f} | {ci_str} "
                f"| {row['precision']:.4f} | {row['recall']:.4f} "
                f"| {row['n_docs']} |\n"
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bootstrap CI for all benchmarks (manifest + HPO inline eval)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    src = parser.add_argument_group("input sources (combinable)")
    src.add_argument("--eval_json", type=Path, help="Single audit JSON")
    src.add_argument("--glob", type=str, help="Glob pattern matching audit JSONs")
    src.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help=f"Manifest TSV (default when --all: {_DEFAULT_MANIFEST})",
    )
    src.add_argument(
        "--hpo_results_root",
        type=Path,
        default=None,
        help=f"Scan JSONL prediction files for HPO eval (default when --all: {_DEFAULT_HPO_ROOT})",
    )
    src.add_argument(
        "--hpo_datasets",
        nargs="+",
        default=_DEFAULT_HPO_DATASETS,
        help="Dataset subdirs under --hpo_results_root to scan",
    )
    src.add_argument(
        "--all",
        action="store_true",
        help="Use default manifest + default HPO results root (shorthand)",
    )
    src.add_argument(
        "--write_audit_jsons",
        action="store_true",
        help="Write eval_*.json audit files next to each HPO JSONL (enables future --glob runs)",
    )

    out = parser.add_argument_group("output")
    out.add_argument("--csv_out", type=Path, default=None)
    out.add_argument("--md_out", type=Path, default=None)

    filt = parser.add_argument_group("filtering")
    filt.add_argument(
        "--min_docs",
        type=int,
        default=20,
        help=(
            "Drop any run with fewer than this many documents (useful for excluding "
            "debug/partial runs). Rough minimums by dataset: biolarkgsc≥200, csc≥100, "
            "mimic3≥60, raredis≥1000."
        ),
    )

    boot = parser.add_argument_group("bootstrap")
    boot.add_argument("--n_bootstrap", type=int, default=1000)
    boot.add_argument("--seed", type=int, default=42)
    boot.add_argument("--ci", type=float, default=95.0, help="CI level (default: 95)")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results: List[Dict] = []

    # --- single file ---
    if args.eval_json:
        row = process_file(
            args.eval_json,
            n_bootstrap=args.n_bootstrap,
            ci=args.ci,
            seed=args.seed,
        )
        if row:
            results.append(row)
            print(
                f"F1={row['f1']:.4f}  "
                f"{args.ci:.0f}% CI: [{row['f1_ci_lower']:.4f}, {row['f1_ci_upper']:.4f}]  "
                f"(n_docs={row['n_docs']}, B={args.n_bootstrap})"
            )

    # --- glob ---
    if args.glob:
        import glob as glob_mod
        paths = sorted(Path(p) for p in glob_mod.glob(args.glob, recursive=True))
        print(f"Found {len(paths)} eval JSONs matching '{args.glob}'")
        for p in paths:
            row = process_file(p, n_bootstrap=args.n_bootstrap, ci=args.ci, seed=args.seed)
            if row:
                results.append(row)

    # --- manifest ---
    use_manifest = args.manifest or args.all
    if use_manifest:
        manifest_path = args.manifest or _DEFAULT_MANIFEST
        manifest_rows = load_manifest(manifest_path)
        print(f"Manifest: {manifest_path} ({len(manifest_rows)} rows)")
        for r in manifest_rows:
            eval_path = Path(r["eval_output"])
            if not eval_path.exists():
                print(f"[skip] missing: {eval_path}", file=sys.stderr)
                continue
            row = process_file(
                eval_path,
                n_bootstrap=args.n_bootstrap,
                ci=args.ci,
                seed=args.seed,
                dataset=r["dataset"],
                track=r["track"],
                approach=r["approach"],
                model_type=r["model_type"],
            )
            if row:
                results.append(row)

    # --- HPO inline eval ---
    use_hpo = args.hpo_results_root or args.all
    if use_hpo:
        hpo_root = args.hpo_results_root or _DEFAULT_HPO_ROOT
        total_jsonl = 0
        for dataset in args.hpo_datasets:
            dataset_dir = hpo_root / dataset
            if not dataset_dir.exists():
                continue
            jsonl_files = sorted(p for p in dataset_dir.glob("*.jsonl") if p.is_file())
            for jf in jsonl_files:
                total_jsonl += 1
                row = process_hpo_file(
                    jf,
                    n_bootstrap=args.n_bootstrap,
                    ci=args.ci,
                    seed=args.seed,
                    dataset=dataset,
                    write_audit_json=args.write_audit_jsons,
                )
                if row:
                    results.append(row)
        print(f"HPO root: {hpo_root} — scanned {total_jsonl} JSONL files across {len(args.hpo_datasets)} dirs")

    if not results:
        print("No results to report.", file=sys.stderr)
        return

    if args.min_docs is not None:
        before = len(results)
        results = [r for r in results if r["n_docs"] >= args.min_docs]
        dropped = before - len(results)
        if dropped:
            print(f"[filter] dropped {dropped} runs with n_docs < {args.min_docs} ({before} → {len(results)})")

    results.sort(key=lambda r: (r["dataset"], r["track"], r["approach"], r["model_type"]))

    if args.csv_out:
        write_csv(results, args.csv_out)
        print(f"CSV written: {args.csv_out}")

    if args.md_out:
        write_markdown(results, args.md_out)
        print(f"Markdown written: {args.md_out}")

    if not args.eval_json:
        print(f"\nProcessed {len(results)} entries total")


if __name__ == "__main__":
    main()
