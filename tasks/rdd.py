import logging
import pickle
from typing import Dict, List, Type, Union

from pyhealth.data.data import Patient
from pyhealth.processors import RawProcessor, TextProcessor
from pyhealth.tasks.base_task import BaseTask

logger = logging.getLogger(__name__)

# In the RDD annotation scheme, "Disability" covers full disease/condition
# expressions.  Sub-component types (Impairment, Function) and meta-types
# (Speculation, SpecScope, ABR, Neg, Scope) are excluded from the NER target.
_NER_TYPES = {"Disability"}

_POLARITY_MAP = {"positive": 1, "negative": 0}


class RDDNER(BaseTask):
    """Named-entity recognition task for the RDD benchmark.

    Each document in RDD is a biomedical literature abstract.  One sample is
    produced per document: the abstract text as input and the target entity
    strings as output.

    The ``target`` argument selects what the gold ``annotations`` list
    contains:

    ``"rare_disease"`` *(default)*
        Unique rare-disease surface forms extracted from the
        ``relationships`` table (i.e. the entity strings as they appear in
        the text, deduplicated per document).  Only documents that have at
        least one entry in ``Relationships/positive.csv`` or
        ``negative.csv`` produce a sample.  Requires that the dataset was
        loaded with ``"relationships"`` in its *tables* list (the default).

    ``"disability"``
        Disability / condition expressions from the ``annotations`` table
        whose ``annotation_type`` is ``Disability``.  Sub-component types
        (Impairment, Function) and meta-annotation types (Speculation,
        SpecScope, ABR, Neg, Scope) are excluded.  This is the annotation
        scheme native to the RDD corpus, but note that these entities are
        **symptoms / functional impairments**, not rare-disease names.

    Input schema:
        text (TextProcessor): The full abstract text.

    Output schema:
        annotations (RawProcessor): List of entity mention strings.

    Args:
        target: ``"rare_disease"`` (default) or ``"disability"``.

    Examples:
        >>> from datasets.rdd import RDDDataset
        >>> from tasks.rdd import RDDNER
        >>> dataset = RDDDataset()
        >>> samples = dataset.set_task(RDDNER())            # rare diseases
        >>> samples = dataset.set_task(RDDNER("disability"))  # symptoms
    """

    task_name: str = "rdd_ner"
    input_schema: Dict[str, Union[str, Type]] = {"text": TextProcessor}
    output_schema: Dict[str, Union[str, Type]] = {
        "annotations": RawProcessor
    }

    def __init__(self, target: str = "rare_disease") -> None:
        if target not in ("rare_disease", "disability"):
            raise ValueError(
                f"target must be 'rare_disease' or 'disability', "
                f"got {target!r}"
            )
        self.target = target

    def __call__(self, patient: Patient) -> List[Dict]:
        """Produce one NER sample per document.

        Args:
            patient: Patient object whose patient_id is the document id.

        Returns:
            A list containing a single sample dict, or an empty list if
            the document has no text or no gold entities for the chosen
            target.
        """
        texts = patient.get_events(event_type="texts")
        if not texts:
            return []
        text = texts[0].text
        if not text:
            return []

        if self.target == "rare_disease":
            ship_events = patient.get_events(event_type="relationships")
            seen: set = set()
            annotations: List[str] = []
            for e in ship_events:
                rd = getattr(e, "rare_disease", None)
                if rd and rd not in seen:
                    seen.add(rd)
                    annotations.append(rd)
        else:  # disability
            ann_events = patient.get_events(event_type="annotations")
            annotations = [
                e.annotation
                for e in ann_events
                if e.annotation_type in _NER_TYPES
            ]

        if not annotations:
            return []

        return [
            {
                "patient_id": patient.patient_id,
                "text": pickle.dumps(text),
                "annotations": pickle.dumps(annotations),
            }
        ]


class RDDRelationExtraction(BaseTask):
    """Rare-disease / disability relation classification for RDD.

    Each sample corresponds to one row in ``Relationships/positive.csv``
    or ``Relationships/negative.csv``.  The sentence containing a
    (rare disease, disability) entity pair is the input; whether their
    relationship is real (positive) or spurious (negative) is the label.

    Positive examples are sentences where the rare disease is genuinely
    characterised by the disability.  Negative examples are sentences
    where both entities co-occur but no such relationship holds.

    Input schema:
        text (TextProcessor): The sentence containing the entity pair.

    Output schema:
        label (RawProcessor): 1 for a positive relationship, 0 for negative.

    The sample dict also carries un-pickled span metadata so that models
    can optionally insert entity markers:

    * ``rare_disease`` / ``rd_start`` / ``rd_end``
    * ``disability``  / ``dis_start`` / ``dis_end``
    * ``speculative`` – 1 if the relationship is expressed speculatively
    * ``orphanet_id`` – Orphanet identifier of the rare disease

    Examples:
        >>> from datasets.rdd import RDDDataset
        >>> from tasks.rdd import RDDRelationExtraction
        >>> dataset = RDDDataset()
        >>> samples = dataset.set_task(RDDRelationExtraction())
    """

    task_name: str = "rdd_relation_extraction"
    input_schema: Dict[str, Union[str, Type]] = {"text": TextProcessor}
    output_schema: Dict[str, Union[str, Type]] = {"label": RawProcessor}

    def __call__(self, patient: Patient) -> List[Dict]:
        """Produce one sample per (rare disease, disability) pair.

        Args:
            patient: Patient object whose patient_id is the document id.

        Returns:
            A list of sample dicts, one per relationship event, or an
            empty list if the document has no relationship annotations.
        """
        ship_events = patient.get_events(event_type="relationships")
        if not ship_events:
            return []

        samples = []
        for e in ship_events:
            sentence = getattr(e, "sentence", None)
            polarity = getattr(e, "polarity", None)
            if not sentence or polarity not in _POLARITY_MAP:
                continue

            try:
                rd_start = int(getattr(e, "rd_start", 0))
                rd_end = int(getattr(e, "rd_end", 0))
                dis_start = int(getattr(e, "dis_start", 0))
                dis_end = int(getattr(e, "dis_end", 0))
                speculative = int(getattr(e, "speculative", 0))
            except (ValueError, TypeError):
                logger.warning(
                    "Skipping malformed span offsets for doc %s",
                    patient.patient_id,
                )
                continue

            samples.append(
                {
                    "patient_id": patient.patient_id,
                    "text": pickle.dumps(sentence),
                    "rare_disease": getattr(e, "rare_disease", ""),
                    "rd_start": rd_start,
                    "rd_end": rd_end,
                    "disability": getattr(e, "disability", ""),
                    "dis_start": dis_start,
                    "dis_end": dis_end,
                    "speculative": speculative,
                    "orphanet_id": getattr(e, "orphanet_id", ""),
                    "label": pickle.dumps(_POLARITY_MAP[polarity]),
                }
            )

        return samples
