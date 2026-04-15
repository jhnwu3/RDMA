import logging
import pickle
from typing import Dict, List, Type, Union

from pyhealth.data.data import Patient
from pyhealth.processors import RawProcessor, TextProcessor
from pyhealth.tasks.base_task import BaseTask

logger = logging.getLogger(__name__)


class BioLarkGSCNER(BaseTask):
    """Named-entity recognition task for the BioLark GSC benchmark.

    Each document in BioLark GSC is a biomedical abstract.  One sample is
    produced per document: the full text as input and the list of HPO
    annotation dicts (with offsets) as output.

    All annotations in BioLark GSC are HPO phenotype mentions
    (``annotation_type == "HP"``); no filtering by type is required.

    Input schema:
        text (TextProcessor): The full document text.

    Output schema:
        annotations (RawProcessor): List of HPO annotation dicts, each
            containing ``hpo_id``, ``start``, ``end``, and ``annotation``
            (surface string).

    Examples:
        >>> from datasets.biolarkgsc import BioLarkGSCDataset
        >>> from tasks.biolarkgsc import BioLarkGSCNER
        >>> dataset = BioLarkGSCDataset()
        >>> samples = dataset.set_task(BioLarkGSCNER())
    """

    task_name: str = "biolarkgsc_ner"
    input_schema: Dict[str, Union[str, Type]] = {"text": TextProcessor}
    output_schema: Dict[str, Union[str, Type]] = {
        "annotations": RawProcessor,
    }

    def __call__(self, patient: Patient) -> List[Dict]:
        """Produce one NER sample per document.

        Args:
            patient: Patient object whose patient_id is the document id.

        Returns:
            A list containing a single sample dict, or an empty list if
            the document has no text or no HPO annotations.
        """
        texts = patient.get_events(event_type="texts")
        if not texts:
            return []
        text = texts[0].text

        ann_events = patient.get_events(event_type="annotations")
        annotations = [
            {
                "hpo_id": e.hpo_id,
                "start": e.start,
                "end": e.end,
                "annotation": e.annotation,
            }
            for e in ann_events
        ]

        if not text or not annotations:
            return []

        return [
            {
                "patient_id": patient.patient_id,
                "text": pickle.dumps(text),
                "annotations": pickle.dumps(annotations),
            }
        ]
