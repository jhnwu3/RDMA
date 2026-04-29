"""
PyHealth task for BioClinicalBERT token-classification NER on RareDis.

One sample is emitted per document.  The task pre-tokenises the text at
``set_task()`` time (i.e. inside ``__call__``) so that the DataLoader
receives ready-to-stack ``torch.LongTensor`` objects.  No per-batch
tokenisation is needed.

BIO label scheme (13 classes):
    O, B-RAREDISEASE, I-RAREDISEASE, B-SKINRAREDISEASE, I-SKINRAREDISEASE,
    B-DISEASE, I-DISEASE, B-SIGN, I-SIGN, B-ANAPHOR, I-ANAPHOR,
    B-SYMPTOM, I-SYMPTOM

Label constants are class attributes on ``BioClinicalBERTNERTask`` so
downstream evaluation code can do ``BioClinicalBERTNERTask.LABEL2ID``.

Usage::

    >>> from datasets.raredis import RareDisDataset
    >>> from tasks.bioclinicalbert_ner_raredis import BioClinicalBERTNERTask
    >>> task = BioClinicalBERTNERTask()
    >>> sample_dataset = RareDisDataset().set_task(task)
"""

from typing import Dict, List, Optional, Tuple, Type, Union

import torch
from transformers import AutoTokenizer

from pyhealth.data.data import Patient
from pyhealth.tasks.base_task import BaseTask

from processors.long_tensor import LongTensorProcessor
from tasks.utils import word_offsets


# ── Private helpers ────────────────────────────────────────────────────────────



def _tokenize_and_align(
    tokenizer,
    words: List[str],
    bio_labels: List[str],
    max_seq_length: int,
) -> Dict[str, torch.Tensor]:
    """Tokenise a word-split document and align BIO labels to subwords.

    Returns ``torch.LongTensor`` for every field directly — the schema
    ``LongTensorProcessor`` is therefore a no-op passthrough.

    Fields returned:
        ``input_ids``      – token ids, shape ``[max_seq_length]``.
        ``attention_mask`` – 1 for real tokens / 0 for padding, same shape.
        ``labels``         – label id for the *first* subword of each word;
                             ``-100`` for subsequent subwords and special
                             tokens (ignored by the CE loss).
        ``word_ids``       – word index for each subword position; ``-1``
                             for special tokens ([CLS], [SEP], [PAD]).
    """
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
            # First subword of this word — assign the word's label
            label_ids.append(BioClinicalBERTNERTask.LABEL2ID[bio_labels[wi]])
            word_ids_out.append(wi)
        else:
            # Continuation subword — ignore in loss; still record word index
            label_ids.append(-100)
            word_ids_out.append(wi)
        prev_word_idx = wi

    return {
        "input_ids":      torch.tensor(encoding["input_ids"],      dtype=torch.long),
        "attention_mask": torch.tensor(encoding["attention_mask"], dtype=torch.long),
        "labels":         torch.tensor(label_ids,                  dtype=torch.long),
        "word_ids":       torch.tensor(word_ids_out,               dtype=torch.long),
    }


# ── Task ───────────────────────────────────────────────────────────────────────


class BioClinicalBERTNERTask(BaseTask):
    """Token-classification NER task for the RareDis benchmark.

    One sample is emitted per document.  Pre-tokenisation happens inside
    ``__call__`` at ``set_task()`` time, so DataLoaders receive tensors
    directly.

    Label constants (``ENTITY_TYPES``, ``LABELS``, ``LABEL2ID``,
    ``ID2LABEL``, ``NUM_LABELS``) are class attributes so evaluation code
    can reference them as ``BioClinicalBERTNERTask.LABEL2ID[...]``.

    Schema:
        Input:  ``input_ids``, ``attention_mask``, ``word_ids``
                (all ``LongTensorProcessor`` → ``torch.LongTensor [seq]``).
        Output: ``labels`` (``LongTensorProcessor`` → ``torch.LongTensor [seq]``,
                ``-100`` for ignored positions).
        Pass-through: ``split``, ``patient_id`` (not in schema).

    Args:
        model_name_or_path: HuggingFace model id or local path for the
            tokeniser.  Defaults to ``emilyalsentzer/Bio_ClinicalBERT``.
        max_seq_length: Subword sequence length after padding / truncation.
            Defaults to 512.
        entity_types: Entity types to annotate.  Defaults to all six types
            in ``ENTITY_TYPES``.
    """

    # ── Label constants ────────────────────────────────────────────────
    ENTITY_TYPES: List[str] = [
        "RAREDISEASE",
        "SKINRAREDISEASE",
        "DISEASE",
        "SIGN",
        "ANAPHOR",
        "SYMPTOM",
    ]
    LABELS: List[str] = [
        "O",
        "B-RAREDISEASE",  "I-RAREDISEASE",
        "B-SKINRAREDISEASE", "I-SKINRAREDISEASE",
        "B-DISEASE",      "I-DISEASE",
        "B-SIGN",         "I-SIGN",
        "B-ANAPHOR",      "I-ANAPHOR",
        "B-SYMPTOM",      "I-SYMPTOM",
    ]
    LABEL2ID: Dict[str, int] = {l: i for i, l in enumerate(LABELS)}
    ID2LABEL: Dict[int, str] = {i: l for i, l in enumerate(LABELS)}
    NUM_LABELS: int = 13

    # ── PyHealth schema ────────────────────────────────────────────────
    task_name: str = "bioclinicalbert_ner_raredis"
    input_schema: Dict[str, Union[str, Type]] = {
        "input_ids":      LongTensorProcessor,   # torch.LongTensor [seq]
        "attention_mask": LongTensorProcessor,   # torch.LongTensor [seq]
        "word_ids":       LongTensorProcessor,   # torch.LongTensor [seq]; -1=special
    }
    output_schema: Dict[str, Union[str, Type]] = {
        "labels": LongTensorProcessor,   # torch.LongTensor [seq]; -100=ignored
    }
    # split / patient_id are NOT in the schema → passed through unchanged.

    def __init__(
        self,
        model_name_or_path: str = "emilyalsentzer/Bio_ClinicalBERT",
        max_seq_length: int = 512,
        entity_types: Optional[List[str]] = None,
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, do_lower_case=False
        )
        self.max_seq_length = max_seq_length
        self.entity_types = set(entity_types or self.ENTITY_TYPES)

    def __call__(self, patient: Patient) -> List[Dict]:
        """Produce one pre-tokenised NER sample per document.

        Args:
            patient: ``Patient`` object whose ``patient_id`` is the document id.

        Returns:
            A list with a single sample dict, or an empty list if the
            document has no text.
        """
        texts = patient.get_events(event_type="texts")
        if not texts:
            return []
        text_event = texts[0]
        text: str = text_event.text
        split: Optional[str] = getattr(text_event, "split", None)
        if not text:
            return []

        # ── 1. Whitespace-split text → word list + char offsets ───────
        words, word_char_starts = word_offsets(text)
        if not words:
            return []

        # ── 2. Build word-level BIO labels (initialise all to "O") ────
        bio_labels: List[str] = ["O"] * len(words)

        ann_events = patient.get_events(event_type="annotations")
        # Sort by span length descending so shorter spans overwrite later;
        # last-written label wins (consistent with BioBERT-MRC convention).
        sorted_anns = sorted(
            [e for e in ann_events if e.annotation_type in self.entity_types],
            key=lambda e: int(e.end) - int(e.start),
            reverse=True,
        )

        for ann in sorted_anns:
            char_start = int(ann.start)
            char_end   = int(ann.end)
            etype      = ann.annotation_type

            # Find words overlapping this char span
            span_words = []
            for wi, (ws, word) in enumerate(zip(word_char_starts, words)):
                we = ws + len(word)
                if ws < char_end and we > char_start:
                    span_words.append(wi)

            if not span_words:
                continue

            # Assign B- to first word, I- to rest
            bio_labels[span_words[0]] = f"B-{etype}"
            for wi in span_words[1:]:
                bio_labels[wi] = f"I-{etype}"

        # ── 3. Tokenise + align labels ─────────────────────────────────
        tokenized = _tokenize_and_align(
            self.tokenizer, words, bio_labels, self.max_seq_length
        )

        return [
            {
                "patient_id":    patient.patient_id,
                "input_ids":     tokenized["input_ids"],       # torch.LongTensor [seq]
                "attention_mask":tokenized["attention_mask"],  # torch.LongTensor [seq]
                "labels":        tokenized["labels"],          # torch.LongTensor [seq]
                "word_ids":      tokenized["word_ids"],        # torch.LongTensor [seq]
                "split":         split,                        # str, pass-through
            }
        ]
