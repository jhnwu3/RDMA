"""
PyHealth task for BioBERT-MRC inference on the CSC phenotype-mining benchmark.

Produces one sample per (document, chunk).  Tokenisation and chunking are
handled by :func:`tasks.utils.mrc_chunk_document`, so inference scripts can
iterate directly over pre-chunked tensor samples.

CSC annotations do not include character offsets, so ``entity_spans`` is
always ``[]``; ``start_ids`` / ``end_ids`` are all zeros (inference only).

Usage::

    >>> from datasets.csc import CSCDataset
    >>> from tasks.biobert_mrc_csc import BioBERTMRCCSCTask
    >>> dataset = CSCDataset()
    >>> samples = dataset.set_task(BioBERTMRCCSCTask())
"""

import pickle
from typing import Dict, List, Type, Union

import torch
from transformers import BertTokenizer

from pyhealth.data.data import Patient
from pyhealth.tasks.base_task import BaseTask

from processors.long_tensor import LongTensorProcessor
from tasks.utils import mrc_chunk_document

_MRC_QUERY = "Find phenotype mentions in the text ."

_DEFAULT_MODEL = (
    "/home/johnwu3/projects/rare_disease/workspace/repos/BioBERT-MRC/BioBERTv1.1_P"
)


class BioBERTMRCCSCTask(BaseTask):
    """MRC-NER inference task for the CSC phenotype-mining benchmark.

    Tokenisation and chunking are delegated to
    :func:`tasks.utils.mrc_chunk_document`.  One sample is emitted per
    (document, chunk), each carrying flat 1-D ``LongTensor`` fields and
    pickled per-document metadata needed for span-to-string reconstruction
    and evaluation.

    CSC annotations do not include character offsets, so ``entity_spans``
    is always ``[]``; ``start_ids`` / ``end_ids`` are all zeros
    (inference only).

    Args:
        model_name_or_path: Tokenizer source (must contain ``vocab.txt``).
            Defaults to the base BioBERTv1.1_P directory.
        max_seq_length: Maximum total subword sequence length.
        stride_tokens: Overlap between consecutive chunks in subword tokens.

    Input schema:
        input_ids, attention_mask, segment_ids, input_len, chunk_sub_offset
        (LongTensorProcessor).

    Output schema:
        start_ids, end_ids (LongTensorProcessor) — all zeros (inference only).

    Passthrough (not in schema):
        words, subtoken_word_starts, gold_phenotypes.
    """

    task_name: str = "biobert_mrc_csc"
    input_schema: Dict[str, Union[str, Type]] = {
        "input_ids":        LongTensorProcessor,
        "attention_mask":   LongTensorProcessor,
        "segment_ids":      LongTensorProcessor,
        "input_len":        LongTensorProcessor,
        "chunk_sub_offset": LongTensorProcessor,
    }
    output_schema: Dict[str, Union[str, Type]] = {
        "start_ids": LongTensorProcessor,
        "end_ids":   LongTensorProcessor,
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
        """Produce one sample per (document, chunk).

        Args:
            patient: Patient object whose patient_id is the document id.

        Returns:
            List of per-chunk sample dicts, or empty list if no text.
        """
        texts = patient.get_events(event_type="texts")
        if not texts:
            return []
        text: str = texts[0].text
        if not text:
            return []

        ph_events = patient.get_events(event_type="phenotypes")
        phenotypes = [
            {"phenotype_name": e.phenotype_name, "hpo_id": e.hpo_id}
            for e in ph_events
        ]
        gold_pkl = pickle.dumps(phenotypes)

        chunks = mrc_chunk_document(
            tokenizer=self.tokenizer,
            text=text,
            entity_spans=[],  # CSC has no char-offset annotations
            query=_MRC_QUERY,
            max_seq_length=self.max_seq_length,
            stride_tokens=self.stride_tokens,
        )
        if not chunks:
            return []

        seq_len = self.max_seq_length
        zeros = torch.zeros(seq_len, dtype=torch.long)  # noqa: F821

        return [
            {
                "patient_id":           patient.patient_id,
                "input_ids":            chunk["input_ids"],
                "attention_mask":       chunk["attention_mask"],
                "segment_ids":          chunk["segment_ids"],
                "start_ids":            zeros,
                "end_ids":              zeros,
                "input_len":            chunk["input_len"],
                "chunk_sub_offset":     chunk["chunk_sub_offset"],
                "words":                pickle.dumps(chunk["words"]),
                "subtoken_word_starts": pickle.dumps(chunk["subtoken_word_starts"]),
                "gold_phenotypes":      gold_pkl,
            }
            for chunk in chunks
        ]
