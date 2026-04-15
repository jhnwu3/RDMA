"""
PyHealth task for BioBERT-MRC inference on MIMIC-III rare-disease notes.

Produces one sample per (note, entity_type) pair, using the same MRC-query
format as BioBERTMRCTask (RareDis).  Because MIMIC-III gold annotations are
entity *strings* (not character spans), ``entity_spans`` is always empty —
the samples are intended for inference and string-level evaluation, not for
span-level training.

``gold_entities`` carries the lower-cased gold entity strings from the
human-reviewed annotation file (only for the RAREDISEASE query; all other
entity-type queries get an empty list since MIMIC-III annotations are not
typed by sub-category).

Usage:
    >>> from pyhealth.datasets import MIMIC3Dataset
    >>> from tasks.biobert_mrc_mimic3 import BioBERTMRCMIMIC3Task
    >>> dataset = MIMIC3Dataset(root="/path/to/mimic-iii/1.4",
    ...                         tables=["noteevents"])
    >>> samples = dataset.set_task(BioBERTMRCMIMIC3Task())
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

# Same query strings as BioBERTMRCTask so the trained model sees familiar prompts.
ENTITY_QUERIES: Dict[str, str] = {
    "RAREDISEASE":      "Find rare disease mentions in the text .",
    "SKINRAREDISEASE":  "Find skin rare disease mentions in the text .",
    "DISEASE":          "Find disease mentions in the text .",
    "SIGN":             "Find sign mentions in the text .",
    "SYMPTOM_AND_SIGN": "Find symptom and sign mentions in the text .",
}

# Only the RAREDISEASE query is evaluated against gold (annotations are not
# broken down by sub-category in the MIMIC-III annotation file).
_GOLD_ENTITY_TYPE = "RAREDISEASE"


class BioBERTMRCMIMIC3Task(BaseTask):
    """MRC-NER inference task over MIMIC-III discharge summaries.

    Reads human-reviewed rare-disease annotations from the bundled JSON and
    produces one sample per (note, entity_type) pair.  ``entity_spans`` is
    always ``[]``; the field exists only to satisfy the ``BioBERTMRCNERProcessor``
    interface (which ignores it when building inference batches).

    Args:
        annotation_path: Path to ``mimic3_mining_rdma_human_annotations.json``.
        rare_only: If ``True`` (default), include only annotations where
            ``is_rare_disease`` is ``True``.

    Input schema:
        text (TextProcessor): Full clinical note text.

    Output schema:
        entity_spans  (RawProcessor): Always pickled ``[]``.
        entity_type   (RawProcessor): Pickled entity-type string.
        mrc_query     (RawProcessor): Pickled MRC query string.
        gold_entities (RawProcessor): Pickled list of lower-cased gold entity
            strings (non-empty only for the RAREDISEASE query).
    """

    task_name: str = "biobert_mrc_mimic3"
    input_schema: Dict[str, Union[str, Type]] = {"text": TextProcessor}
    output_schema: Dict[str, Union[str, Type]] = {
        "entity_spans":  RawProcessor,
        "entity_type":   RawProcessor,
        "mrc_query":     RawProcessor,
        "gold_entities": RawProcessor,
    }

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
        """Return a dict mapping ROW_ID → list of lower-cased entity strings."""
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
            f"BioBERTMRCMIMIC3Task: loaded {len(index)} annotated notes"
            f" ({n_pos} positive, {n_neg} negative)"
            f" ({self.annotation_path})",
            flush=True,
        )
        return index

    # ------------------------------------------------------------------

    def __call__(self, patient: Patient) -> List[Dict]:
        """Produce one sample per (note, entity_type) for each annotated note.

        Args:
            patient: Patient whose ``noteevents`` carry ``row_id`` and ``text``.

        Returns:
            List of sample dicts (one per entity type per matching note).
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

            gold_entities = self._anno_index[row_id]

            for entity_type, query in ENTITY_QUERIES.items():
                # Gold is only available for the RAREDISEASE query.
                etype_gold = gold_entities if entity_type == _GOLD_ENTITY_TYPE else []
                samples.append(
                    {
                        "patient_id":    patient.patient_id,
                        "note_id":       row_id,
                        "text":          pickle.dumps(text),
                        "entity_spans":  pickle.dumps([]),
                        "entity_type":   pickle.dumps(entity_type),
                        "mrc_query":     pickle.dumps(query),
                        "gold_entities": pickle.dumps(etype_gold),
                    }
                )

        return samples
