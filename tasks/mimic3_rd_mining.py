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


class MIMIC3RareDiseaseMining(BaseTask):
    """Rare-disease mention extraction task over MIMIC-III discharge summaries.

    Uses human-reviewed annotations from ``mimic3_mining_rdma_human_annotations.json``
    to mark which MIMIC-III NOTEEVENTS rows contain confirmed rare-disease mentions.
    Notes that do not appear in the annotation file are excluded entirely.

    The annotation file is keyed by NOTEEVENTS ``ROW_ID`` (as string).  Each
    entry contains ``note_details`` (without the original note text) and an
    ``annotations`` list.  One sample is produced per annotated note that has
    at least one qualifying annotation.

    The annotation file is loaded and indexed at initialisation time, so the
    lookup during ``__call__`` is O(1) per note.

    Requires the MIMIC-III dataset to be loaded with ``noteevents`` in the
    ``tables`` list and the ``row_id`` column exposed (this is the default
    when using the updated ``mimic3.yaml`` config).

    Args:
        annotation_path (str): Path to the annotations JSON.
            Defaults to the bundled copy inside the RDMA repo.
        rare_only (bool): If ``True`` (default), only annotations where
            ``is_rare_disease`` is ``True`` are included.  Set to ``False``
            to include all reviewed annotations regardless of decision.

    Input schema:
        text (TextProcessor): Full clinical note text.

    Output schema:
        annotations (RawProcessor): List of dicts, each with keys
            ``entity`` (mention string) and ``orpha_code``.

    Examples:
        >>> from pyhealth.datasets import MIMIC3Dataset
        >>> from tasks.mimic3_rd_mining import MIMIC3RareDiseaseMining
        >>> dataset = MIMIC3Dataset(
        ...     root="/path/to/mimic-iii/1.4",
        ...     tables=["noteevents"],
        ... )
        >>> task = MIMIC3RareDiseaseMining()
        >>> samples = dataset.set_task(task)
    """

    task_name: str = "mimic3_rare_disease_mining"
    input_schema: Dict[str, Union[str, Type]] = {"text": TextProcessor}
    output_schema: Dict[str, Union[str, Type]] = {"annotations": RawProcessor}

    def __init__(
        self,
        annotation_path: str = str(_DEFAULT_ANNOTATION_PATH),
        rare_only: bool = True,
    ) -> None:
        self.annotation_path = annotation_path
        self.rare_only = rare_only
        self._anno_index: Dict[str, List[Dict]] = self._load_annotations()

    def _load_annotations(self) -> Dict[str, List[Dict]]:
        """Load and index annotations by document_id (ROW_ID).

        The JSON is keyed by document id at the root level.  Each value has an
        ``annotations`` list whose items use ``mention`` for the entity string.

        Returns:
            Dict mapping document_id (str) to a list of annotation dicts,
            each containing ``entity`` and optionally ``orpha_code``.
        """
        with open(self.annotation_path, encoding="utf-8") as fh:
            data = json.load(fh)

        index: Dict[str, List[Dict]] = {}
        for doc_id, entry in data.items():
            annos = []
            for anno in entry.get("annotations", []):
                if self.rare_only and not anno.get("is_rare_disease", False):
                    continue
                anno_dict: Dict = {"entity": anno["mention"]}
                if "orpha_code" in anno:
                    anno_dict["orpha_code"] = anno["orpha_code"]
                annos.append(anno_dict)
            # Always index the document, including negatives (annos == []).
            # Negatives are documents confirmed to contain no rare-disease
            # mentions; they serve as hard negative examples at evaluation time.
            index[doc_id] = annos

        n_positive = sum(1 for v in index.values() if v)
        n_negative = sum(1 for v in index.values() if not v)
        print(
            f"MIMIC3RareDiseaseMining: loaded {len(index)} annotated notes"
            f" ({n_positive} positive, {n_negative} negative)"
            f" ({self.annotation_path})",
            flush=True,
        )
        return index

    def __call__(self, patient: Patient) -> List[Dict]:
        """Produce one NER sample per annotated note for this patient.

        Args:
            patient: Patient object whose ``noteevents`` events carry a
                ``row_id`` attribute matching the annotation ``document_id``.

        Returns:
            A list of sample dicts (one per matching note), or an empty list
            if the patient has no notes with confirmed rare-disease mentions.
        """
        note_events = patient.get_events(event_type="noteevents")
        if not note_events:
            return []

        samples = []
        for note in note_events:
            raw_id = note.attr_dict.get("row_id")
            if raw_id is None:
                continue
            # ROW_ID is an integer in the CSV but may be read as float (e.g.
            # 287.0) by polars/pandas when the column has a nullable type.
            # Convert via int() to get "287" not "287.0".
            try:
                row_id = str(int(float(raw_id)))
            except (ValueError, TypeError):
                continue
            if row_id not in self._anno_index:
                continue

            text = note.attr_dict.get("text", "")
            if not text:
                continue

            annotations = self._anno_index[row_id]
            # Negative documents (no confirmed rare-disease mentions) are
            # represented with a sentinel so the eval script can score them.
            if not annotations:
                annotations = [{"entity": "None"}]

            samples.append(
                {
                    "patient_id": patient.patient_id,
                    "note_id": row_id,
                    "text": pickle.dumps(text),
                    "annotations": pickle.dumps(annotations),
                }
            )

        return samples
