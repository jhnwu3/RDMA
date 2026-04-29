"""
PyHealth task for BioBERT-MRC fine-tuning on the BioLarkGSC corpus.

One sample is emitted per document chunk.  Tokenisation and chunking are
handled by :func:`tasks.utils.mrc_chunk_document`, so the DataLoader
receives ready-to-stack 1-D ``LongTensor`` values with no further
processing needed.

Usage::

    >>> from datasets.biolarkgsc import BioLarkGSCDataset
    >>> from tasks.biobert_mrc_biolarkgsc import BioBERTMRCBioLarkGSCTask
    >>> dataset = BioLarkGSCDataset()
    >>> samples = dataset.set_task(BioBERTMRCBioLarkGSCTask())
"""

import pickle
from typing import Dict, List, Type, Union

from transformers import BertTokenizer

from pyhealth.data.data import Patient
from pyhealth.tasks.base_task import BaseTask

from processors.long_tensor import LongTensorProcessor
from tasks.utils import mrc_chunk_document

_MRC_QUERY = "Find phenotype mentions in the text ."
_ENTITY_TYPE = "PHENOTYPE"

_DEFAULT_MODEL = (
    "/home/johnwu3/projects/rare_disease/workspace/repos/BioBERT-MRC/BioBERTv1.1_P"
)

# ID2LABEL for SpanEntityScore: maps start-logit label id → entity type name.
ID2LABEL: Dict[int, str] = {1: _ENTITY_TYPE}


class BioBERTMRCBioLarkGSCTask(BaseTask):
    """MRC-NER task for the BioLarkGSC benchmark.

    Tokenisation and chunking are delegated to
    :func:`tasks.utils.mrc_chunk_document`.  One sample is emitted per
    document chunk, each carrying flat 1-D ``LongTensor`` fields and pickled
    per-document metadata needed for evaluation.

    Input schema:
        input_ids      (LongTensorProcessor): BERT input token ids.
        attention_mask (LongTensorProcessor): Attention mask.
        segment_ids    (LongTensorProcessor): Segment (token-type) ids.
        input_len      (LongTensorProcessor): Actual sequence length scalar.
        chunk_sub_offset (LongTensorProcessor): Global subword offset for chunk.

    Output schema:
        start_ids (LongTensorProcessor): Per-token start-entity labels.
        end_ids   (LongTensorProcessor): Per-token end-entity labels.

    Passthrough (pickled bytes, not in schema):
        subjects            – gold (label_id, sub_start, sub_end) triples
        words               – whitespace-split word list for the document
        word_char_starts    – character start index per word
        subtoken_word_starts– subword start index per word
        gold_annotations    – list of {start, end, hpo_id} gold dicts
    """

    task_name: str = "biobert_mrc_biolarkgsc"
    input_schema: Dict[str, Union[str, Type]] = {
        "input_ids": LongTensorProcessor,
        "attention_mask": LongTensorProcessor,
        "segment_ids": LongTensorProcessor,
        "input_len": LongTensorProcessor,
        "chunk_sub_offset": LongTensorProcessor,
    }
    output_schema: Dict[str, Union[str, Type]] = {
        "start_ids": LongTensorProcessor,
        "end_ids": LongTensorProcessor,
    }

    def __init__(
        self,
        model_name_or_path: str = _DEFAULT_MODEL,
        max_seq_length: int = 256,
        stride_tokens: int = 64,
    ) -> None:
        self.tokenizer = BertTokenizer.from_pretrained(
            model_name_or_path, do_lower_case=False
        )
        self.max_seq_length = max_seq_length
        self.stride_tokens = stride_tokens

    def __call__(self, patient: Patient) -> List[Dict]:
        """Produce one NER sample per document chunk.

        Args:
            patient: Patient object whose patient_id is the document id.

        Returns:
            A list of sample dicts (one per chunk), or an empty list if
            the document has no text.
        """
        texts = patient.get_events(event_type="texts")
        if not texts:
            return []
        text: str = texts[0].text
        if not text:
            return []

        ann_events = patient.get_events(event_type="annotations")
        entity_spans = [(int(e.start), int(e.end)) for e in ann_events]
        gold_annotations = [
            {
                "start": int(e.start),
                "end": int(e.end),
                "hpo_id": (
                    e.hpo_id.replace("_", ":", 1)
                    if e.hpo_id and e.hpo_id.startswith("HP_")
                    else e.hpo_id
                ),
            }
            for e in ann_events
            if (getattr(e, "hpo_id", None) and getattr(e, "start", None) is not None)
        ]

        chunks = mrc_chunk_document(
            tokenizer=self.tokenizer,
            text=text,
            entity_spans=entity_spans,
            query=_MRC_QUERY,
            max_seq_length=self.max_seq_length,
            stride_tokens=self.stride_tokens,
        )

        gold_annotations_pkl = pickle.dumps(gold_annotations)

        return [
            {
                "patient_id": patient.patient_id,
                **chunk,
                "subjects": pickle.dumps(chunk["subjects"]),
                "words": pickle.dumps(chunk["words"]),
                "word_char_starts": pickle.dumps(chunk["word_char_starts"]),
                "subtoken_word_starts": pickle.dumps(chunk["subtoken_word_starts"]),
                "gold_annotations": gold_annotations_pkl,
            }
            for chunk in chunks
        ]
