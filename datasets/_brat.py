"""Shared utilities for parsing BRAT standoff (.ann) files."""

import re
from pathlib import Path

_T_RE = re.compile(r"^T\d+$")
_R_RE = re.compile(r"^R\d+$")


def parse_ann(ann_path: Path, encoding: str = "utf-8") -> tuple:
    """Parse a BRAT .ann file into entity and relation lists.

    Entity lines (T*) are parsed into dicts with the following keys:

    * ``ann_id``         – BRAT identifier, e.g. ``"T1"``
    * ``annotation_type`` – entity type string, e.g. ``"RAREDISEASE"``
    * ``start``          – character offset of the first span start
    * ``end``            – character offset of the last span end
    * ``annotation``     – surface text extracted from the document

    For discontinuous entities (multiple spans joined by ``;``), ``start``
    and ``end`` give the bounding box: first-span start, last-span end.

    Relation lines (R*) are parsed into dicts with the following keys:

    * ``rel_id``         – BRAT identifier, e.g. ``"R1"``
    * ``relation_type``  – relation type string, e.g. ``"Is_a"``
    * ``arg1``           – BRAT id of the first argument, e.g. ``"T1"``
    * ``arg2``           – BRAT id of the second argument, e.g. ``"T2"``

    Attribute lines (A*), note lines (N*), and any other prefixes are
    silently ignored.

    Args:
        ann_path: Path to the ``.ann`` file.
        encoding: File encoding (default ``"utf-8"``).

    Returns:
        ``(entities, relations)`` — two lists of dicts as described above.
    """
    entities = []
    relations = []

    with open(ann_path, encoding=encoding, errors="replace") as fh:
        for raw_line in fh:
            line = raw_line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            ann_id = parts[0]

            # ── Entity line ───────────────────────────────────────────────
            if (
                ann_id.startswith("T")
                and len(parts) >= 3
                and _T_RE.match(ann_id)
            ):
                type_and_spans, entity_text = parts[1], parts[2].strip()
                if not entity_text:
                    continue

                pieces = type_and_spans.split(" ", 1)
                ann_type = pieces[0]
                span_str = pieces[1] if len(pieces) > 1 else ""

                # Parse all spans (handles discontinuous annotations)
                spans = []
                for raw_span in span_str.split(";"):
                    tokens = raw_span.strip().split()
                    if len(tokens) >= 2:
                        try:
                            spans.append(
                                (int(tokens[0]), int(tokens[1]))
                            )
                        except ValueError:
                            pass

                if not spans:
                    continue

                entities.append({
                    "ann_id": ann_id,
                    "annotation_type": ann_type,
                    "start": spans[0][0],
                    "end": spans[-1][1],
                    "annotation": entity_text,
                })

            # ── Relation line ─────────────────────────────────────────────
            elif (
                ann_id.startswith("R")
                and len(parts) >= 2
                and _R_RE.match(ann_id)
            ):
                rel_parts = parts[1].split()
                if len(rel_parts) < 3:
                    continue
                rel_type = rel_parts[0]
                arg1 = (
                    rel_parts[1].split(":")[1]
                    if ":" in rel_parts[1] else ""
                )
                arg2 = (
                    rel_parts[2].split(":")[1]
                    if ":" in rel_parts[2] else ""
                )
                if arg1 and arg2:
                    relations.append({
                        "rel_id": ann_id,
                        "relation_type": rel_type,
                        "arg1": arg1,
                        "arg2": arg2,
                    })

    return entities, relations
