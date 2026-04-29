"""
PyHealth task for BioBERT-MRC inference on MIMIC-III rare-disease notes.

Produces one sample per (note, entity_type, chunk).  Tokenisation and
chunking are handled by :func:`tasks.utils.mrc_chunk_document`, so
inference scripts can iterate directly over pre-chunked tensor samples.

A single ``BertTokenizer`` is shared across all entity types (they all use
the same vocabulary); only the ``query`` string varies per entity type.

Usage::

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

import torch
from transformers import BertTokenizer

from pyhealth.data.data import Patient
from pyhealth.tasks.base_task import BaseTask

from processors.long_tensor import LongTensorProcessor
from tasks.utils import mrc_chunk_document

_DEFAULT_ANNOTATION_PATH = (
    Path(__file__).parent.parent
    / "public_data"
    / "rare_disease_mining"
    / "mimic3_mining_rdma_human_annotations.json"
)

_DEFAULT_MODEL = (
    "/home/johnwu3/projects/rare_disease/workspace/repos/BioBERT-MRC/BioBERTv1.1_P"
)

# Same query strings as the RareDis training task.
ENTITY_QUERIES: Dict[str, str] = {
    "RAREDISEASE":      "Find rare disease mentions in the text .",
    "SKINRAREDISEASE":  "Find skin rare disease mentions in the text .",
    "DISEASE":          "Find disease mentions in the text .",
    "SIGN":             "Find sign mentions in the text .",
    "SYMPTOM_AND_SIGN": "Find symptom and sign mentions in the text .",
}

_GOLD_ENTITY_TYPE = "RAREDISEASE"


class BioBERTMRCMIMIC3Task(BaseTask):
    """MRC-NER inference task over MIMIC-III discharge summaries.

    Tokenisation and chunking are delegated to
    :func:`tasks.utils.mrc_chunk_document`.  One sample is emitted per
    (note, entity_type, chunk), each carrying flat 1-D ``LongTensor`` fields
    and pickled per-document metadata needed for span-to-string
    reconstruction and evaluation.

    A single ``BertTokenizer`` is shared across all entity types — the
    query string is passed as an argument to ``mrc_chunk_document`` at
    call time, so separate tokenizer instances are unnecessary.

    Args:
        annotation_path: Path to ``mimic3_mining_rdma_human_annotations.json``.
        rare_only: If ``True`` (default), include only is_rare_disease annotations.
        model_name_or_path: Tokenizer source (must contain ``vocab.txt``).
        max_seq_length: Maximum total subword sequence length.
        stride_tokens: Overlap between consecutive chunks in subword tokens.

    Input schema:
        input_ids, attention_mask, segment_ids, input_len, chunk_sub_offset
        (LongTensorProcessor).

    Output schema:
        start_ids, end_ids (LongTensorProcessor) — all zeros (inference only).

    Passthrough (not in schema):
        note_id, entity_type, words, subtoken_word_starts, gold_entities.
    """

    task_name: str = "biobert_mrc_mimic3"
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
        annotation_path: str = str(_DEFAULT_ANNOTATION_PATH),
        rare_only: bool = True,
        model_name_or_path: str = _DEFAULT_MODEL,
        max_seq_length: int = 256,
        stride_tokens: int = 64,
    ) -> None:
        self.annotation_path = annotation_path
        self.rare_only = rare_only
        self.max_seq_length = max_seq_length
        self.stride_tokens = stride_tokens
        # One shared tokenizer for all entity types (same vocab, query varies).
        self.tokenizer = BertTokenizer.from_pretrained(
            model_name_or_path, do_lower_case=False
        )
        self._anno_index: Dict[str, List[str]] = self._load_annotations()

    def _load_annotations(self) -> Dict[str, List[str]]:
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

    def __call__(self, patient: Patient) -> List[Dict]:
        """Produce one sample per (note, entity_type, chunk).

        Args:
            patient: Patient whose ``noteevents`` carry ``row_id`` and ``text``.

        Returns:
            List of per-chunk sample dicts, or empty list if no matching notes.
        """
        note_events = patient.get_events(event_type="noteevents")
        if not note_events:
            return []

        zeros = torch.zeros(self.max_seq_length, dtype=torch.long)

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
            gold_pkl = pickle.dumps(gold_entities)
            empty_gold_pkl = pickle.dumps([])

            for entity_type, query in ENTITY_QUERIES.items():
                chunks = mrc_chunk_document(
                    tokenizer=self.tokenizer,
                    text=text,
                    entity_spans=[],  # inference only
                    query=query,
                    max_seq_length=self.max_seq_length,
                    stride_tokens=self.stride_tokens,
                )
                if not chunks:
                    continue

                etype_gold_pkl = (
                    gold_pkl if entity_type == _GOLD_ENTITY_TYPE else empty_gold_pkl
                )

                for chunk in chunks:
                    samples.append(
                        {
                            "patient_id":           patient.patient_id,
                            "note_id":              row_id,
                            "entity_type":          entity_type,
                            "input_ids":            chunk["input_ids"],
                            "attention_mask":       chunk["attention_mask"],
                            "segment_ids":          chunk["segment_ids"],
                            "start_ids":            zeros,
                            "end_ids":              zeros,
                            "input_len":            chunk["input_len"],
                            "chunk_sub_offset":     chunk["chunk_sub_offset"],
                            "words":                pickle.dumps(chunk["words"]),
                            "subtoken_word_starts": pickle.dumps(
                                chunk["subtoken_word_starts"]
                            ),
                            "gold_entities":        etype_gold_pkl,
                        }
                    )

        return samples
