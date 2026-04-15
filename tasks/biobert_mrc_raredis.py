"""
PyHealth task for BioBERT-MRC fine-tuning on the RareDis corpus.

For each document, one sample is emitted per entity type.  The entity type
is communicated to the model via a natural-language query (MRC-NER style),
so the model predicts span start/end positions conditioned on the query.

Usage:
    >>> from datasets.raredis import RareDisDataset
    >>> from tasks.biobert_mrc_raredis import BioBERTMRCTask
    >>> dataset = RareDisDataset()
    >>> samples = dataset.set_task(BioBERTMRCTask())
"""

import pickle
from typing import Dict, List, Type, Union

from pyhealth.data.data import Patient
from pyhealth.processors import RawProcessor, TextProcessor
from pyhealth.tasks.base_task import BaseTask

# Entity types in the RareDis annotation scheme that we train on.
# Each entry maps annotation_type → MRC query string.
ENTITY_QUERIES: Dict[str, str] = {
    "RAREDISEASE":      "Find rare disease mentions in the text .",
    "SKINRAREDISEASE":  "Find skin rare disease mentions in the text .",
    "DISEASE":          "Find disease mentions in the text .",
    "SIGN":             "Find sign mentions in the text .",
    "SYMPTOM_AND_SIGN": "Find symptom and sign mentions in the text .",
}


class BioBERTMRCTask(BaseTask):
    """MRC-NER task for the RareDis benchmark.

    Each document produces one sample per entity type in ``ENTITY_QUERIES``.
    Samples include the raw text, character-level entity spans filtered to
    that type, the entity type label, and the MRC query string.

    Input schema:
        text (TextProcessor): The full document text.

    Output schema:
        entity_spans (RawProcessor): Pickled list of (start, end) char tuples.
        entity_type  (RawProcessor): Pickled entity type string.
        mrc_query    (RawProcessor): Pickled MRC query string.
        split        (RawProcessor): Document split (train/dev/test).
    """

    task_name: str = "biobert_mrc_raredis"
    input_schema: Dict[str, Union[str, Type]] = {"text": TextProcessor}
    output_schema: Dict[str, Union[str, Type]] = {
        "entity_spans": RawProcessor,
        "entity_type":  RawProcessor,
        "mrc_query":    RawProcessor,
        "split":        RawProcessor,
    }

    def __call__(self, patient: Patient) -> List[Dict]:
        """Produce one NER sample per (document, entity_type) pair.

        Args:
            patient: Patient object whose patient_id is the document id.

        Returns:
            A list of sample dicts, one per entity type.  An empty list is
            returned if the document has no text.
        """
        texts = patient.get_events(event_type="texts")
        if not texts:
            return []
        text_event = texts[0]
        text = text_event.text
        split = getattr(text_event, "split", None)
        if not text:
            return []

        ann_events = patient.get_events(event_type="annotations")

        samples = []
        for entity_type, query in ENTITY_QUERIES.items():
            spans = [
                (int(e.start), int(e.end))
                for e in ann_events
                if e.annotation_type == entity_type
            ]
            samples.append(
                {
                    "patient_id":   patient.patient_id,
                    "text":         pickle.dumps(text),
                    "entity_spans": pickle.dumps(spans),
                    "entity_type":  pickle.dumps(entity_type),
                    "mrc_query":    pickle.dumps(query),
                    "split":        split,
                }
            )

        return samples
