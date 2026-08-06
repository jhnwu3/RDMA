#!/usr/bin/env python3
"""Strip verbatim MIMIC note text out of RDMA annotation files.

MIMIC-III and MIMIC-IV are distributed under the PhysioNet Credentialed Health
Data Use Agreement, which forbids republishing note text. Three RDMA annotation
files embed note excerpts in ``context`` fields. This script removes those
excerpts while keeping every annotation label and every join key, so that a
credentialed user can rebuild the contexts locally with
``scripts/data/rehydrate_mimic_text.py``.

The originals are archived outside the repository; see
``private_data/mimic3_restricted/README.md``.

Usage
-----
Strip all three files from the private archive into ``public_data/``::

    python scripts/data/strip_mimic_text.py \
        --archive ../../private_data/mimic3_restricted \
        --out public_data

Strip a single file::

    python scripts/data/strip_mimic_text.py \
        --input  /path/to/reannoted_rd_annos.json \
        --output public_data/rare_disease_mining/reannoted_rd_annos.json

The script is deterministic and safe to re-run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Keys whose values are verbatim note text. Removed wherever they appear, at
# any nesting depth.
TEXT_KEYS = frozenset({"context"})

# Keys whose values are free-text fields we replace rather than drop, because
# downstream readers index into them positionally.
REDACTION = "<redacted: see scripts/data/rehydrate_mimic_text.py>"

# Metadata keys that leak absolute paths from the authors' machine.
PATH_KEYS = frozenset(
    {"predictions_file", "ground_truth_file", "evaluation_file", "input_file", "output_file"}
)

# The files this script knows how to strip, and where each one lands.
KNOWN_FILES = {
    "annotation_tool_input.json": "annotation_tool_input.json",
    "reannoted_rd_annos.json": "rare_disease_mining/reannoted_rd_annos.json",
    "initial_diff_diagnosis_benchmark.json": "initial_diff_diagnosis_benchmark.json",
    # Also carries note text: 141 `context` fields under annotations[]. The
    # evaluation tasks only read `mention` and `orpha_code`, so stripping
    # `context` does not affect scoring. This file stays in public_data because
    # its `note_details` block is the join key the other files rehydrate through.
    "mimic3_mining_rdma_human_annotations.json": (
        "rare_disease_mining/mimic3_mining_rdma_human_annotations.json"
    ),
}


class Stats:
    """Counters for what a single strip pass removed."""

    def __init__(self) -> None:
        self.contexts_removed = 0
        self.context_chars_removed = 0
        self.paths_scrubbed = 0
        self.lab_values_removed = 0

    def report(self) -> str:
        return (
            f"contexts removed: {self.contexts_removed:,} "
            f"({self.context_chars_removed:,} chars); "
            f"lab values removed: {self.lab_values_removed:,}; "
            f"paths scrubbed: {self.paths_scrubbed}"
        )


def _scrub_path(value: object) -> object:
    """Reduce an absolute local path to its basename."""
    if isinstance(value, str) and ("/" in value or "\\" in value):
        return Path(value).name
    return value


def strip_node(node: object, stats: Stats) -> object:
    """Recursively remove note text from an arbitrary JSON structure."""
    if isinstance(node, dict):
        out = {}
        for key, value in node.items():
            if key in TEXT_KEYS and isinstance(value, str):
                stats.contexts_removed += 1
                stats.context_chars_removed += len(value)
                continue

            if key in PATH_KEYS and isinstance(value, str):
                scrubbed = _scrub_path(value)
                if scrubbed != value:
                    stats.paths_scrubbed += 1
                out[key] = scrubbed
                continue

            # lab_info holds per-patient measurements. Keep the phenotype-
            # relevant signal (which lab, which direction) and drop the value.
            if key == "lab_info" and isinstance(value, dict):
                lab = dict(value)
                if lab.pop("value", None) is not None:
                    stats.lab_values_removed += 1
                out[key] = lab
                continue

            out[key] = strip_node(value, stats)
        return out

    if isinstance(node, list):
        return [strip_node(item, stats) for item in node]

    return node


def strip_file(src: Path, dst: Path) -> Stats:
    with src.open() as handle:
        data = json.load(handle)

    stats = Stats()
    stripped = strip_node(data, stats)

    # Record provenance so the stripped file is self-describing.
    if isinstance(stripped, dict) and "metadata" in stripped and isinstance(stripped["metadata"], dict):
        stripped["metadata"]["redaction_note"] = (
            "Verbatim MIMIC note excerpts (`context` fields) were removed to comply "
            "with the PhysioNet Data Use Agreement. Rebuild them with "
            "scripts/data/rehydrate_mimic_text.py using your own credentialed copy."
        )

    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w") as handle:
        json.dump(stripped, handle, indent=1, sort_keys=False)
        handle.write("\n")

    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--archive", type=Path, help="Directory holding the restricted originals.")
    parser.add_argument("--out", type=Path, help="public_data/ directory to write stripped copies into.")
    parser.add_argument("--input", type=Path, help="Strip a single file instead of the whole archive.")
    parser.add_argument("--output", type=Path, help="Destination for --input.")
    args = parser.parse_args()

    if args.input:
        if not args.output:
            parser.error("--input requires --output")
        stats = strip_file(args.input, args.output)
        print(f"{args.input.name}: {stats.report()}")
        return 0

    if not (args.archive and args.out):
        parser.error("provide either --input/--output or --archive/--out")

    missing = [name for name in KNOWN_FILES if not (args.archive / name).exists()]
    if missing:
        print(f"error: not found in {args.archive}: {', '.join(missing)}", file=sys.stderr)
        return 1

    for name, relative in KNOWN_FILES.items():
        src = args.archive / name
        dst = args.out / relative
        stats = strip_file(src, dst)
        size_before = src.stat().st_size / 1e6
        size_after = dst.stat().st_size / 1e6
        print(f"{name}\n  {stats.report()}\n  {size_before:.1f} MB -> {size_after:.1f} MB")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
