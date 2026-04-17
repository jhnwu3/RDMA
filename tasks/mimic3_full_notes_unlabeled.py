"""PyHealth task for unlabeled MIMIC-III note-level inference samples.

This task emits one sample per NOTEEVENT row with plain-text note content and
literal metadata fields, without expected label fields.
"""

from typing import Dict, List, Optional, Sequence, Type, Union

from pyhealth.data.data import Patient
from pyhealth.processors import RawProcessor, TextProcessor
from pyhealth.tasks.base_task import BaseTask


class MIMIC3FullNotesUnlabeled(BaseTask):
    """Create unlabeled note-level samples from MIMIC-III NOTEEVENTS.

    Input schema:
        text (TextProcessor): Full clinical note text.

    Output schema:
        category (RawProcessor): NOTEEVENT category.
        description (RawProcessor): NOTEEVENT description.
        chartdate (RawProcessor): Chart date string.
        charttime (RawProcessor): Chart time string.
        storetime (RawProcessor): Store time string.
        hadm_id (RawProcessor): Admission id string.

    Args:
        discharge_only: If True, keep only notes in category
            "Discharge summary".
        note_categories: Optional category allow-list (case-insensitive).
            When provided, only matching categories are included.
        max_notes_per_patient: Optional cap on emitted notes per patient.
    """

    task_name: str = "mimic3_full_notes_unlabeled"
    input_schema: Dict[str, Union[str, Type]] = {"text": TextProcessor}
    output_schema: Dict[str, Union[str, Type]] = {
        "category": RawProcessor,
        "description": RawProcessor,
        "chartdate": RawProcessor,
        "charttime": RawProcessor,
        "storetime": RawProcessor,
        "hadm_id": RawProcessor,
    }

    def __init__(
        self,
        discharge_only: bool = False,
        note_categories: Optional[Sequence[str]] = None,
        max_notes_per_patient: Optional[int] = None,
    ) -> None:
        self.discharge_only = discharge_only
        self.note_categories = (
            {c.strip().lower() for c in note_categories if c and c.strip()}
            if note_categories
            else None
        )
        self.max_notes_per_patient = max_notes_per_patient

    @staticmethod
    def _normalize_row_id(raw_id: object) -> Optional[str]:
        """Normalize MIMIC ROW_ID values to canonical string form."""
        if raw_id is None:
            return None
        try:
            return str(int(float(raw_id)))
        except (TypeError, ValueError):
            return None

    def __call__(self, patient: Patient) -> List[Dict]:
        """Return unlabeled samples for all qualifying notes in a patient."""
        note_events = patient.get_events(event_type="noteevents")
        if not note_events:
            return []

        samples: List[Dict] = []
        for note in note_events:
            note_id = self._normalize_row_id(note.attr_dict.get("row_id"))
            if note_id is None:
                continue

            text = note.attr_dict.get("text", "")
            if not text:
                continue

            category = str(note.attr_dict.get("category", "")).strip()
            category_l = category.lower()

            if self.discharge_only and category_l != "discharge summary":
                continue
            if self.note_categories and category_l not in self.note_categories:
                continue

            samples.append(
                {
                    "patient_id": patient.patient_id,
                    "note_id": note_id,
                    "text": text,
                    "category": category,
                    "description": str(note.attr_dict.get("description", "")),
                    "chartdate": str(note.attr_dict.get("chartdate", "")),
                    "charttime": str(note.attr_dict.get("charttime", "")),
                    "storetime": str(note.attr_dict.get("storetime", "")),
                    "hadm_id": str(note.attr_dict.get("hadm_id", "")),
                }
            )

            if (
                self.max_notes_per_patient is not None
                and len(samples) >= self.max_notes_per_patient
            ):
                break

        return samples
