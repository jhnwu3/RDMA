#!/usr/bin/env python3
"""Gate check: fail if anything under public_data/ looks like clinical note text.

Run this before publishing or pushing. It walks every JSON file under
``public_data/`` and flags:

* any residual ``context`` key (these hold verbatim note excerpts),
* any string value longer than ``--max-len`` characters,
* any string matching a MIMIC / discharge-summary marker.

Exit status is 1 if anything is flagged, so it can be wired into CI or a
pre-push hook.

Usage::

    python scripts/data/check_public_data_leakage.py
    python scripts/data/check_public_data_leakage.py --root public_data --max-len 200
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# Keys that held note text before stripping. Any survivor is a hard failure.
FORBIDDEN_KEYS = frozenset({"context", "note_text", "text_snippet", "raw_text"})

# De-identification artifacts. These are machine-inserted by the MIMIC release
# process and never appear in legitimate annotation labels, so any occurrence is
# a leak regardless of string length.
STRONG_MARKERS = [
    (re.compile(r"\[\*\*.*?\*\*\]"), "MIMIC-III de-identification token [**...**]"),
    (re.compile(r"___\s*\d{1,2}:\d{2}\s*[AP]M", re.I), "MIMIC-IV de-identified timestamp"),
]

# Discharge-summary section headers. These DO occur inside legitimate short
# annotation labels (the corpus contains the extracted entity "past medical
# history"), so they only count as evidence inside a passage long enough to be
# an actual note excerpt.
WEAK_MARKERS = [
    # A bare `___` is a MIMIC-IV de-identification placeholder. It appears
    # legitimately inside short extracted entity labels ("K___-Feil syndrome",
    # "bilateral ___ edema") because the entity was extracted from a
    # de-identified note — and the placeholder marks exactly where PHI was
    # already removed, so it carries no patient information. Inside a long
    # passage, though, it is a reliable sign of copied note text.
    (re.compile(r"___"), "MIMIC-IV de-identification placeholder ___"),
    (re.compile(r"\badmission date\b", re.I), "'Admission Date' note header"),
    (re.compile(r"\bdischarge summary\b", re.I), "'Discharge Summary' note header"),
    (re.compile(r"\bbrief hospital course\b", re.I), "'Brief Hospital Course' note header"),
    (re.compile(r"\bchief complaint\b", re.I), "'Chief Complaint' note header"),
    (re.compile(r"\bpast medical history\b", re.I), "'Past Medical History' note header"),
]

# Minimum length before a section-header match counts as a finding.
WEAK_MARKER_MIN_LEN = 120


# Structured metadata fields whose values legitimately look like note headers.
# `note_details.category` is the MIMIC NOTEEVENTS category column, so its value
# is literally the string "Discharge summary" — a label, not note text.
ALLOWED_MARKER_KEYS = frozenset({"category", "note_type", "description"})

# Fields that hold long text which is *deliberately* public, and so are exempt
# from both the length and marker checks:
#
#   clinical_text  the CSC benchmark case reports, redistributed from RAG-HPO
#                  under MIT (see public_data/README.md)
#   info           Human Phenotype Ontology term labels and definitions, which
#                  are long prose and can themselves contain phrases like
#                  "past medical history"
#
# Everything else is checked. If you add a new public corpus with long text,
# add its field here rather than raising --max-len, so MIMIC leakage stays caught.
PUBLIC_TEXT_KEYS = frozenset({"clinical_text", "info", "definition"})


def walk(node, path, findings, max_len, key=None):
    if isinstance(node, dict):
        for child_key, value in node.items():
            here = f"{path}.{child_key}" if path else child_key
            if child_key in FORBIDDEN_KEYS:
                findings.append((here, f"forbidden key '{child_key}' present"))
            walk(value, here, findings, max_len, child_key)
    elif isinstance(node, list):
        for index, item in enumerate(node):
            walk(item, f"{path}[{index}]", findings, max_len, key)
    elif isinstance(node, str):
        # De-identification artifacts are always a leak, even in exempt fields.
        for pattern, label in STRONG_MARKERS:
            if pattern.search(node):
                findings.append((path, f"{label}: {node[:80]!r}"))
                return

        if key in PUBLIC_TEXT_KEYS:
            return

        if len(node) > max_len:
            findings.append((path, f"string of {len(node)} chars exceeds {max_len}: {node[:80]!r}..."))

        if key in ALLOWED_MARKER_KEYS or len(node) < WEAK_MARKER_MIN_LEN:
            return
        for pattern, label in WEAK_MARKERS:
            if pattern.search(node):
                findings.append((path, f"{label}: {node[:80]!r}"))
                break


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("public_data"))
    parser.add_argument("--max-len", type=int, default=200)
    args = parser.parse_args()

    if not args.root.exists():
        print(f"error: {args.root} does not exist", file=sys.stderr)
        return 1

    total = 0
    for json_path in sorted(args.root.rglob("*.json")):
        with json_path.open() as handle:
            data = json.load(handle)
        findings: list[tuple[str, str]] = []
        walk(data, "", findings, args.max_len, None)

        rel = json_path.relative_to(args.root)
        if findings:
            total += len(findings)
            print(f"FAIL {rel} — {len(findings)} finding(s)")
            for where, why in findings[:10]:
                print(f"     {where}: {why}")
            if len(findings) > 10:
                print(f"     ... and {len(findings) - 10} more")
        else:
            print(f"ok   {rel}")

    if total:
        print(f"\n{total} finding(s) — do not publish until these are resolved.")
        return 1

    print("\nNo clinical text detected in public_data/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
