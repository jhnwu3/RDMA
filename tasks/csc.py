import logging
import pickle
from typing import Dict, List, Type, Union

from pyhealth.data.data import Patient
from pyhealth.processors import RawProcessor, TextProcessor
from pyhealth.tasks.base_task import BaseTask

logger = logging.getLogger(__name__)


class CSCPhenotypeMining(BaseTask):
    """Phenotype mining task for the CSC benchmark.

    Each document in the CSC benchmark is a clinical case report.  One
    sample is produced per document: the full text as input and the list
    of annotated phenotype dicts as output.

    Unlike BioLark GSC, CSC annotations do **not** include character
    offsets.  Each phenotype is represented as a
    ``{"phenotype_name": ..., "hpo_id": ...}`` dict.  HPO IDs use the
    colon-separated form ``HP:XXXXXXX``.

    Documents with an empty phenotype list are included so they can
    serve as negative examples for downstream models.

    Input schema:
        text (TextProcessor): The full clinical case text.

    Output schema:
        phenotypes (RawProcessor): List of phenotype dicts, each
            containing ``phenotype_name`` and ``hpo_id``.

    Examples:
        >>> from datasets.csc import CSCDataset
        >>> from tasks.csc import CSCPhenotypeMining
        >>> dataset = CSCDataset()
        >>> samples = dataset.set_task(CSCPhenotypeMining())
    """

    task_name: str = "csc_phenotype_mining"
    input_schema: Dict[str, Union[str, Type]] = {"text": TextProcessor}
    output_schema: Dict[str, Union[str, Type]] = {
        "phenotypes": RawProcessor,
    }

    def __call__(self, patient: Patient) -> List[Dict]:
        """Produce one phenotype mining sample per document.

        Args:
            patient: Patient object whose patient_id is the document id.

        Returns:
            A list containing a single sample dict, or an empty list if
            the document has no text.
        """
        texts = patient.get_events(event_type="texts")
        if not texts:
            return []
        text = texts[0].text
        if not text:
            return []

        ph_events = patient.get_events(event_type="phenotypes")
        phenotypes = [
            {
                "phenotype_name": e.phenotype_name,
                "hpo_id": e.hpo_id,
            }
            for e in ph_events
        ]

        return [
            {
                "patient_id": patient.patient_id,
                "text": pickle.dumps(text),
                "phenotypes": pickle.dumps(phenotypes),
            }
        ]
