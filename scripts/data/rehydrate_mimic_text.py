#!/usr/bin/env python3
"""Rebuild the ``context`` windows that were stripped out of public_data/.

The published RDMA annotation files have their MIMIC note excerpts removed (see
``scripts/data/strip_mimic_text.py``). If you hold PhysioNet credentials for
MIMIC-III and/or MIMIC-IV, this script reconstructs those excerpts locally from
your own copy of the corpora.

**The rebuilt context is equivalent, not byte-identical.** The originals stored
snippets with no character offsets, so this script re-locates each annotated
entity inside its source note and re-extracts a window around it. Window edges
will differ from ours; the enclosing sentence(s) will not.

Which corpus each file needs
----------------------------
=====================================================  ==========  ==========================================
File                                                   Corpus      Join
=====================================================  ==========  ==========================================
rare_disease_mining/mimic3_mining_rdma_human_...json   MIMIC-III   top-level key = ``NOTEEVENTS.ROW_ID``
rare_disease_mining/reannoted_rd_annos.json            MIMIC-III   ``document_id`` = ``NOTEEVENTS.ROW_ID``
annotation_tool_input.json                             MIMIC-III   ``document_id`` = ``NOTEEVENTS.ROW_ID``
initial_diff_diagnosis_benchmark.json                  MIMIC-IV    top-level key = ``subject_id``
=====================================================  ==========  ==========================================

Usage
-----
::

    # MIMIC-III files
    python scripts/data/rehydrate_mimic_text.py \
        --mimic3-root /path/to/mimic-iii-clinical-database-1.4 \
        --out rehydrated/

    # MIMIC-IV differential-diagnosis benchmark
    python scripts/data/rehydrate_mimic_text.py \
        --mimic4-note-root /path/to/mimic-iv-note/2.2/note \
        --out rehydrated/

Output goes to ``--out`` (default ``rehydrated/``), never back into
``public_data/``, so a credentialed working copy is never confused with the
publishable one. ``rehydrated/`` is gitignored.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import sys
from pathlib import Path

# Default half-width of the rebuilt context window, in characters.
DEFAULT_WINDOW = 120

MIMIC3_FILES = {
    "rare_disease_mining/mimic3_mining_rdma_human_annotations.json": "gold",
    "rare_disease_mining/reannoted_rd_annos.json": "flat",
    "annotation_tool_input.json": "tool",
}
MIMIC4_FILES = {
    "initial_diff_diagnosis_benchmark.json": "diffdiag",
}


def _open_maybe_gzip(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", newline="", encoding="utf-8", errors="replace")
    return path.open(newline="", encoding="utf-8", errors="replace")


def load_mimic3_notes(root: Path, wanted: set[str]) -> dict[str, str]:
    """Map NOTEEVENTS.ROW_ID -> note text, for the ROW_IDs we need."""
    candidates = [root / "NOTEEVENTS.csv.gz", root / "NOTEEVENTS.csv"]
    source = next((c for c in candidates if c.exists()), None)
    if source is None:
        raise FileNotFoundError(f"NOTEEVENTS.csv[.gz] not found under {root}")

    print(f"reading {source} (looking for {len(wanted):,} notes)...", flush=True)
    notes: dict[str, str] = {}
    csv.field_size_limit(sys.maxsize)
    with _open_maybe_gzip(source) as handle:
        for row in csv.DictReader(handle):
            row_id = (row.get("ROW_ID") or row.get("row_id") or "").strip()
            if row_id in wanted:
                notes[row_id] = row.get("TEXT") or row.get("text") or ""
                if len(notes) == len(wanted):
                    break
    print(f"  matched {len(notes):,}/{len(wanted):,}", flush=True)
    return notes


def load_mimic4_notes(note_root: Path, wanted: set[str]) -> dict[str, str]:
    """Map subject_id -> concatenated discharge-note text, for wanted subjects."""
    candidates = [note_root / "discharge.csv.gz", note_root / "discharge.csv"]
    source = next((c for c in candidates if c.exists()), None)
    if source is None:
        raise FileNotFoundError(f"discharge.csv[.gz] not found under {note_root}")

    print(f"reading {source} (looking for {len(wanted):,} subjects)...", flush=True)
    notes: dict[str, list[str]] = {}
    csv.field_size_limit(sys.maxsize)
    with _open_maybe_gzip(source) as handle:
        for row in csv.DictReader(handle):
            subject = (row.get("subject_id") or "").strip()
            if subject in wanted:
                notes.setdefault(subject, []).append(row.get("text") or "")
    merged = {k: "\n\n".join(v) for k, v in notes.items()}
    print(f"  matched {len(merged):,}/{len(wanted):,}", flush=True)
    return merged


def _normalize(text: str) -> str:
    """Collapse whitespace so entity lookup survives line-wrapping."""
    return re.sub(r"\s+", " ", text).lower()


def find_context(note: str, entity: str, window: int) -> str | None:
    """Locate `entity` in `note` and return a character window around it."""
    if not note or not entity:
        return None

    flat_note = _normalize(note)
    flat_entity = _normalize(entity)
    if not flat_entity:
        return None

    position = flat_note.find(flat_entity)
    if position == -1:
        return None

    start = max(0, position - window)
    end = min(len(flat_note), position + len(flat_entity) + window)
    return flat_note[start:end].strip()


class Counter:
    def __init__(self) -> None:
        self.rebuilt = 0
        self.unlocatable = 0
        self.missing_note = 0

    def report(self) -> str:
        return (
            f"rebuilt {self.rebuilt:,}; "
            f"entity not found in note {self.unlocatable:,}; "
            f"note unavailable {self.missing_note:,}"
        )


def rehydrate_entries(entries, note_for, window, counts, entity_key="entity", doc_key="document_id"):
    """Add a `context` back to each entry in a list of annotation dicts."""
    for entry in entries:
        note = note_for(str(entry.get(doc_key, "")))
        if note is None:
            counts.missing_note += 1
            continue
        context = find_context(note, str(entry.get(entity_key) or ""), window)
        if context is None:
            counts.unlocatable += 1
            continue
        entry["context"] = context
        counts.rebuilt += 1


def rehydrate_file(kind: str, data, notes: dict[str, str], window: int) -> Counter:
    counts = Counter()

    def note_for(doc_id: str):
        return notes.get(doc_id)

    if kind == "gold":
        for doc_id, entry in data.items():
            note = notes.get(str(doc_id))
            for anno in entry.get("annotations", []):
                if note is None:
                    counts.missing_note += 1
                    continue
                context = find_context(note, str(anno.get("mention") or ""), window)
                if context is None:
                    counts.unlocatable += 1
                    continue
                anno["context"] = context
                counts.rebuilt += 1

    elif kind == "flat":
        rehydrate_entries(data.get("corrected_annotations", []), note_for, window, counts)

    elif kind == "tool":
        for bucket in data.get("results", {}).values():
            rehydrate_entries(bucket, note_for, window, counts)

    elif kind == "diffdiag":
        for subject_id, entry in data.items():
            note = notes.get(str(subject_id))
            for match in entry.get("matched_phenotypes", []):
                if note is None:
                    counts.missing_note += 1
                    continue
                target = match.get("original_entity") or match.get("phenotype") or ""
                context = find_context(note, str(target), window)
                if context is None:
                    counts.unlocatable += 1
                    continue
                match["context"] = context
                counts.rebuilt += 1

    return counts


def collect_doc_ids(kind: str, data) -> set[str]:
    if kind in ("gold", "diffdiag"):
        return {str(k) for k in data}
    if kind == "flat":
        return {str(e.get("document_id")) for e in data.get("corrected_annotations", [])}
    if kind == "tool":
        return {
            str(e.get("document_id"))
            for bucket in data.get("results", {}).values()
            for e in bucket
        }
    return set()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--public-data", type=Path, default=Path("public_data"))
    parser.add_argument("--mimic3-root", type=Path, help="MIMIC-III 1.4 directory (contains NOTEEVENTS.csv[.gz]).")
    parser.add_argument("--mimic4-note-root", type=Path, help="mimic-iv-note note/ directory (contains discharge.csv[.gz]).")
    parser.add_argument("--out", type=Path, default=Path("rehydrated"))
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW, help="Context half-width in characters (default: %(default)s).")
    args = parser.parse_args()

    if not (args.mimic3_root or args.mimic4_note_root):
        parser.error("provide --mimic3-root and/or --mimic4-note-root")

    jobs = []
    if args.mimic3_root:
        jobs.append((args.mimic3_root, MIMIC3_FILES, load_mimic3_notes))
    if args.mimic4_note_root:
        jobs.append((args.mimic4_note_root, MIMIC4_FILES, load_mimic4_notes))

    exit_code = 0
    for root, file_map, loader in jobs:
        loaded: dict[str, tuple] = {}
        wanted: set[str] = set()
        for relative, kind in file_map.items():
            path = args.public_data / relative
            if not path.exists():
                print(f"skip {relative} (not found)", file=sys.stderr)
                continue
            with path.open() as handle:
                data = json.load(handle)
            loaded[relative] = (kind, data)
            wanted |= collect_doc_ids(kind, data)

        if not loaded:
            continue

        notes = loader(root, wanted)

        for relative, (kind, data) in loaded.items():
            counts = rehydrate_file(kind, data, notes, args.window)
            destination = args.out / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("w") as handle:
                json.dump(data, handle, indent=1)
                handle.write("\n")
            print(f"{relative}\n  {counts.report()}\n  -> {destination}")
            if counts.unlocatable or counts.missing_note:
                exit_code = 2

    if exit_code:
        print(
            "\nSome entities could not be relocated. This is expected for a few "
            "annotations whose surface form was normalised during annotation; "
            "inspect the counts above before relying on the rebuilt contexts.",
            file=sys.stderr,
        )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
