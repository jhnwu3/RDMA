import logging
import pickle
from typing import Dict, List, Type, Union

from pyhealth.data.data import Patient
from pyhealth.processors import RawProcessor, TextProcessor
from pyhealth.tasks.base_task import BaseTask

logger = logging.getLogger(__name__)

# Entity types from the RareDis annotation scheme that represent
# actual rare-disease mentions.  SKINRAREDISEASE is a dermatological
# subtype that is also included.  Other types (DISEASE, SIGN, ANAPHOR,
# SYMPTOM) are excluded from the NER target.
_NER_TYPES = {"RAREDISEASE", "SKINRAREDISEASE"}


class RareDisNER(BaseTask):
    """Named-entity recognition task for the RareDis benchmark.

    Each document in RareDis is a NORD rare-disease description.  One sample
    is produced per document: the full text as input and the list of annotated
    rare-disease mention strings as output.

    Only annotations whose ``annotation_type`` is ``RAREDISEASE`` or
    ``SKINRAREDISEASE`` are included.  Other types present in the corpus
    (DISEASE, SIGN, ANAPHOR, SYMPTOM) are filtered out.

    Input schema:
        text (TextProcessor): The full document passage.

    Output schema:
        annotations (RawProcessor): List of disease-mention strings.
        split (RawProcessor): Document split (train/dev/test).

    Examples:
        >>> from datasets.raredis import RareDisDataset
        >>> from tasks.raredis import RareDisNER
        >>> dataset = RareDisDataset()
        >>> samples = dataset.set_task(RareDisNER())
    """

    task_name: str = "raredis_ner"
    input_schema: Dict[str, Union[str, Type]] = {"text": TextProcessor}
    output_schema: Dict[str, Union[str, Type]] = {
        "annotations": RawProcessor,
        "split": RawProcessor,
    }

    def __call__(self, patient: Patient) -> List[Dict]:
        """Produce one NER sample per document.

        Args:
            patient: Patient object whose patient_id is the document id.

        Returns:
            A list containing a single sample dict, or an empty list if
            the document has no text or no disease-type annotations.
        """
        texts = patient.get_events(event_type="texts")
        if not texts:
            return []
        text_event = texts[0]
        text = text_event.text
        split = getattr(text_event, "split", None)

        ann_events = patient.get_events(event_type="annotations")
        annotations = [
            e.annotation for e in ann_events if e.annotation_type in _NER_TYPES
        ]

        if not text or not annotations:
            return []

        return [
            {
                "patient_id": patient.patient_id,
                "text": pickle.dumps(text),
                "annotations": pickle.dumps(annotations),
                "split": split,
            }
        ]
