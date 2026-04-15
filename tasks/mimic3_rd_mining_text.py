"""
PyHealth task for MIMIC-III rare-disease text extraction.

Produces one sample per annotated note with gold labels as a flat list of
lowercased entity mention strings.  Unlike ``MIMIC3RareDiseaseMining``, this
task does **not** include ORPHA codes, making it suitable for string-level
NER evaluation (matching the RareDis / RDD task interfaces).

Usage:
    >>> from pyhealth.datasets import MIMIC3Dataset
    >>> from tasks.mimic3_rd_mining_text import MIMIC3RDMiningText
    >>> dataset = MIMIC3Dataset(root="/path/to/mimic-iii/1.4",
    ...                         tables=["noteevents"])
    >>> samples = dataset.set_task(MIMIC3RDMiningText())
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Type, Union

from pyhealth.data.data import Patient
from pyhealth.processors import RawProcessor, TextProcessor
from pyhealth.tasks.base_task import BaseTask

_DEFAULT_ANNOTATION_PATH = (
    Path(__file__).parent.parent
    / "public_data"
    / "rare_disease_mining"
    / "mimic3_mining_rdma_human_annotations.json"
)


class MIMIC3RDMiningText(BaseTask):
    """Text-extraction NER task over MIMIC-III discharge summaries.

    Uses human-reviewed annotations from ``mimic3_mining_rdma_human_annotations.json``
    to supply per-note gold entity strings.  Each sample carries a flat list of
    lowercased mention strings (``entities``), suitable for set-based string
    F1 evaluation.

    Notes that appear in the annotation file but have **no** qualifying
    annotations (negative documents) are included with an empty ``entities``
    list so that false-positive predictions on those notes can be penalised.

    Args:
        annotation_path: Path to ``mimic3_mining_rdma_human_annotations.json``.
        rare_only: If ``True`` (default), include only annotations where
            ``is_rare_disease`` is ``True``.

    Input schema:
        text (TextProcessor): Full clinical note text.

    Output schema:
        entities (RawProcessor): Pickled ``List[str]`` of lowercased mention
            strings (empty list for negative documents).
    """

    task_name: str = "mimic3_rd_mining_text"
    input_schema: Dict[str, Union[str, Type]] = {"text": TextProcessor}
    output_schema: Dict[str, Union[str, Type]] = {"entities": RawProcessor}

    def __init__(
        self,
        annotation_path: str = str(_DEFAULT_ANNOTATION_PATH),
        rare_only: bool = True,
    ) -> None:
        self.annotation_path = annotation_path
        self.rare_only = rare_only
        self._anno_index: Dict[str, List[str]] = self._load_annotations()

    # ------------------------------------------------------------------

    def _load_annotations(self) -> Dict[str, List[str]]:
        """Return a dict mapping ROW_ID → list of lowercased entity strings."""
        with open(self.annotation_path, encoding="utf-8") as fh:
            data = json.load(fh)

        index: Dict[str, List[str]] = {}
        for doc_id, entry in data.items():
            entities: List[str] = []
            for anno in entry.get("annotations", []):
                if self.rare_only and not anno.get("is_rare_disease", False):
                    continue
                mention = anno.get("mention", "").strip()
                if mention:
                    entities.append(mention.lower())
            index[doc_id] = entities

        n_pos = sum(1 for v in index.values() if v)
        n_neg = sum(1 for v in index.values() if not v)
        print(
            f"MIMIC3RDMiningText: loaded {len(index)} annotated notes"
            f" ({n_pos} positive, {n_neg} negative)"
            f" ({self.annotation_path})",
            flush=True,
        )
        return index

    # ------------------------------------------------------------------

    def __call__(self, patient: Patient) -> List[Dict]:
        """Produce one sample per annotated note for this patient.

        Args:
            patient: Patient whose ``noteevents`` carry ``row_id`` and ``text``.

        Returns:
            List of sample dicts (one per matching note).
        """
        note_events = patient.get_events(event_type="noteevents")
        if not note_events:
            return []

        samples = []
        for note in note_events:
            raw_id = note.attr_dict.get("row_id")
            if raw_id is None:
                continue
            try:
                row_id = str(int(float(raw_id)))
            except (ValueError, TypeError):
                continue
            if row_id not in self._anno_index:
                continue

            text = note.attr_dict.get("text", "")
            if not text:
                continue

            entities = self._anno_index[row_id]
            samples.append(
                {
                    "patient_id": patient.patient_id,
                    "note_id": row_id,
                    "text": pickle.dumps(text),
                    "entities": pickle.dumps(entities),
                }
            )

        return samples
