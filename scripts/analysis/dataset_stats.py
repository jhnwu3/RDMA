#!/usr/bin/env python3
"""
Compute corpus statistics for all RDMA benchmarks.

Reports per-dataset and aggregate:
  - # documents
  - # entity/phenotype labels
  - avg / max word count (whitespace-split)

Usage (from workspace root):
    conda activate rd_pyhealth
    python scripts/dataset_stats.py
"""

import pickle
import sys
from pathlib import Path

_RDMA = Path("/home/johnwu3/projects/rare_disease/workspace/repos/RDMA")
_PYHEALTH = Path("/home/johnwu3/projects/rare_disease/workspace/repos/PyHealth")

sys.path.insert(0, str(_PYHEALTH))
sys.path.insert(0, str(_RDMA))

# ── Dataset + task imports ────────────────────────────────────────────────────

from datasets.biolarkgsc import BioLarkGSCDataset
from datasets.csc import CSCDataset
from datasets.raredis import RareDisDataset
from tasks.biolarkgsc import BioLarkGSCNER
from tasks.csc import CSCPhenotypeMining
from tasks.raredis import RareDisNER

# MIMIC-III requires the raw NOTEEVENTS data; guard against missing files.
try:
    from pyhealth.datasets import MIMIC3Dataset
    from tasks.mimic3_rd_mining import MIMIC3RareDiseaseMining
    _MIMIC3_AVAILABLE = True
except Exception:
    _MIMIC3_AVAILABLE = False

_MIMIC3_ROOT = "/srv/local/data/physionet.org/files/mimic-iii-clinical-database-1.4"
_MIMIC3_CACHE_DIR = "/shared/eng/pyhealth/mimic3"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _word_count(text: str) -> int:
    return len(text.split())


def compute_stats(samples, label_field: str) -> dict:
    """Compute stats over a list of task samples.

    Args:
        samples: SampleDataset (iterable of dicts with pickled fields).
        label_field: Key whose unpickled value is the list of entities.

    Returns:
        dict with n_docs, n_entities, avg_words, max_words.
    """
    n_docs = 0
    n_entities = 0
    word_counts = []

    for s in samples:
        text = pickle.loads(s["text"])
        labels = pickle.loads(s[label_field])
        n_docs += 1
        n_entities += len(labels)
        word_counts.append(_word_count(text))

    total_words = sum(word_counts)
    avg_words = total_words / len(word_counts) if word_counts else 0.0
    max_words = max(word_counts) if word_counts else 0

    return {
        "n_docs": n_docs,
        "n_entities": n_entities,
        "total_words": total_words,
        "avg_words": avg_words,
        "max_words": max_words,
    }


def print_row(name, stats):
    print(
        f"  {name:<20s}  docs={stats['n_docs']:>5d}  "
        f"entities={stats['n_entities']:>6d}  "
        f"total_words={stats['total_words']:>8d}  "
        f"avg_words={stats['avg_words']:>7.1f}  "
        f"max_words={stats['max_words']:>6d}"
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    results = {}

    # ── BioLarkGSC ────────────────────────────────────────────────────────────
    print("Loading BioLarkGSC …")
    ds = BioLarkGSCDataset()
    samples = ds.set_task(BioLarkGSCNER())
    results["BioLarkGSC"] = compute_stats(samples, "annotations")

    # ── CSC ───────────────────────────────────────────────────────────────────
    print("Loading CSC …")
    ds = CSCDataset()
    samples = ds.set_task(CSCPhenotypeMining())
    results["CSC"] = compute_stats(samples, "phenotypes")

    # ── RareDis (all splits combined) ─────────────────────────────────────────
    print("Loading RareDis …")
    ds = RareDisDataset()
    samples = ds.set_task(RareDisNER())
    results["RareDis (all)"] = compute_stats(samples, "annotations")

    # Per-split breakdown
    for split in ("train", "dev", "test"):
        split_samples = [s for s in samples if s.get("split") == split]
        if split_samples:
            results[f"  RareDis/{split}"] = compute_stats(split_samples, "annotations")

    # ── MIMIC3-RD ─────────────────────────────────────────────────────────────
    if _MIMIC3_AVAILABLE:
        mimic3_root = Path(_MIMIC3_ROOT)
        if mimic3_root.exists():
            print("Loading MIMIC3-RD …")
            try:
                ds = MIMIC3Dataset(root=str(mimic3_root), tables=["noteevents"], cache_dir=_MIMIC3_CACHE_DIR)
                samples = ds.set_task(MIMIC3RareDiseaseMining())
                results["MIMIC3-RD"] = compute_stats(samples, "annotations")
            except Exception as exc:
                print(f"  [MIMIC3-RD] skipped: {exc}")
        else:
            print(f"  [MIMIC3-RD] skipped: root not found at {_MIMIC3_ROOT}")
    else:
        print("  [MIMIC3-RD] skipped: MIMIC3Dataset not importable")

    # ── Print table ───────────────────────────────────────────────────────────
    print()
    print("=" * 91)
    print(f"  {'Dataset':<20s}  {'docs':>5s}  {'entities':>8s}  {'total_words':>11s}  {'avg_words':>9s}  {'max_words':>9s}")
    print("=" * 91)
    for name, stats in results.items():
        print_row(name, stats)
    print("=" * 91)


if __name__ == "__main__":
    main()
