"""
PyHealth task for BioClinicalBERT token-classification NER on BioLarkGSC.

One sample is emitted per document.  The task pre-tokenises the text at
``set_task()`` time so the DataLoader receives ready-to-stack tensors.

BIO label scheme (3 classes):
    O, B-PHENOTYPE, I-PHENOTYPE

Usage::

    >>> from datasets.biolarkgsc import BioLarkGSCDataset
    >>> from tasks.bioclinicalbert_ner_biolarkgsc import (
    ...     BioClinicalBERTNERBioLarkGSCTask,
    ... )
    >>> task = BioClinicalBERTNERBioLarkGSCTask()
    >>> sample_dataset = BioLarkGSCDataset().set_task(task)
"""

import pickle
from typing import Dict, List, Tuple, Type, Union

import torch
from transformers import AutoTokenizer

from pyhealth.data.data import Patient
from pyhealth.tasks.base_task import BaseTask

from processors.long_tensor import LongTensorProcessor
from tasks.utils import word_offsets



def _tokenize_and_align(
    tokenizer,
    words: List[str],
    bio_labels: List[str],
    max_seq_length: int,
) -> Dict[str, torch.Tensor]:
    """Tokenise a word-split document and align BIO labels to subwords."""
    encoding = tokenizer(
        words,
        is_split_into_words=True,
        truncation=True,
        max_length=max_seq_length,
        padding="max_length",
    )

    label_ids: List[int] = []
    word_ids_out: List[int] = []
    prev_word_idx = None

    for wi in encoding.word_ids():
        if wi is None:
            label_ids.append(-100)
            word_ids_out.append(-1)
        elif wi != prev_word_idx:
            lbl = BioClinicalBERTNERBioLarkGSCTask.LABEL2ID[bio_labels[wi]]
            label_ids.append(lbl)
            word_ids_out.append(wi)
        else:
            label_ids.append(-100)
            word_ids_out.append(wi)
        prev_word_idx = wi

    return {
        "input_ids": torch.tensor(encoding["input_ids"], dtype=torch.long),
        "attention_mask": torch.tensor(encoding["attention_mask"], dtype=torch.long),
        "labels": torch.tensor(label_ids, dtype=torch.long),
        "word_ids": torch.tensor(word_ids_out, dtype=torch.long),
    }


class BioClinicalBERTNERBioLarkGSCTask(BaseTask):
    """Token-classification NER task for the BioLarkGSC benchmark.

    One sample per document.  Pre-tokenisation happens at ``set_task()`` time.
    Character offsets from the annotations table are used to assign BIO tags
    to whitespace-split words, then subword-aligned via the HuggingFace
    tokenizer.  Splitting is handled externally via
    ``pyhealth.datasets.split_by_patient``.

    Label scheme: O=0, B-PHENOTYPE=1, I-PHENOTYPE=2.

    Args:
        model_name_or_path: HuggingFace model id or local path for the
            tokenizer.
        max_seq_length: Subword sequence length after padding/truncation.
    """

    ENTITY_TYPES: List[str] = ["PHENOTYPE"]
    LABELS: List[str] = ["O", "B-PHENOTYPE", "I-PHENOTYPE"]
    LABEL2ID: Dict[str, int] = {l: i for i, l in enumerate(LABELS)}
    ID2LABEL: Dict[int, str] = {i: l for i, l in enumerate(LABELS)}
    NUM_LABELS: int = 3

    task_name: str = "bioclinicalbert_ner_biolarkgsc"
    input_schema: Dict[str, Union[str, Type]] = {
        "input_ids": LongTensorProcessor,
        "attention_mask": LongTensorProcessor,
        "word_ids": LongTensorProcessor,
    }
    output_schema: Dict[str, Union[str, Type]] = {
        "labels": LongTensorProcessor,
    }

    def __init__(
        self,
        model_name_or_path: str = "emilyalsentzer/Bio_ClinicalBERT",
        max_seq_length: int = 512,
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, do_lower_case=False
        )
        self.max_seq_length = max_seq_length

    def __call__(self, patient: Patient) -> List[Dict]:
        """Produce one pre-tokenised NER sample per document.

        Args:
            patient: ``Patient`` object whose ``patient_id`` is the document
                id.

        Returns:
            A list with a single sample dict, or an empty list if the
            document has no text.
        """
        texts = patient.get_events(event_type="texts")
        if not texts:
            return []
        text: str = texts[0].text
        if not text:
            return []

        words, word_char_starts = word_offsets(text)
        if not words:
            return []

        bio_labels: List[str] = ["O"] * len(words)

        ann_events = patient.get_events(event_type="annotations")
        # Collect gold HPO IDs for predictions.jsonl (normalise HP_XXXXXXX ->
        # HP:XXXXXXX to match the format used by other biolarkgsc baselines).
        gold_hpo_ids = [
            e.hpo_id.replace("_", ":", 1) if e.hpo_id.startswith("HP_") else e.hpo_id
            for e in ann_events
            if getattr(e, "hpo_id", None)
        ]
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

        # Sort by span length descending so shorter spans win (last written).
        sorted_anns = sorted(
            ann_events,
            key=lambda e: int(e.end) - int(e.start),
            reverse=True,
        )

        for ann in sorted_anns:
            char_start = int(ann.start)
            char_end = int(ann.end)

            span_words = []
            for wi, (ws, word) in enumerate(zip(word_char_starts, words)):
                we = ws + len(word)
                if ws < char_end and we > char_start:
                    span_words.append(wi)

            if not span_words:
                continue

            bio_labels[span_words[0]] = "B-PHENOTYPE"
            for wi in span_words[1:]:
                bio_labels[wi] = "I-PHENOTYPE"

        tokenized = _tokenize_and_align(
            self.tokenizer, words, bio_labels, self.max_seq_length
        )

        return [
            {
                "patient_id": patient.patient_id,
                "input_ids": tokenized["input_ids"],
                "attention_mask": tokenized["attention_mask"],
                "labels": tokenized["labels"],
                "word_ids": tokenized["word_ids"],
                "words": pickle.dumps(words),
                "gold_hpo_ids": pickle.dumps(gold_hpo_ids),
                "gold_annotations": pickle.dumps(gold_annotations),
                "word_char_starts": pickle.dumps(word_char_starts),
            }
        ]
