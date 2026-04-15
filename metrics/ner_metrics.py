"""
NER evaluation metrics for span-extraction MRC NER.

Inlined from BioBERT-MRC/metrics/ner_metrics.py with the following changes:
  - ``SeqEntityScore`` (BIO/BIOS sequence labels) dropped — not used by MRC baselines.
  - ``from tools.common import logger`` replaced with standard ``logging.getLogger``.
  - ``from processors.utils_ner import get_entities`` removed (was only used by
    ``SeqEntityScore``).

``SpanEntityScore`` evaluates span-extraction NER models where both ground-truth
and predictions are lists of ``(label_id, start, end)`` tuples.
"""

import logging
from collections import Counter

logger = logging.getLogger(__name__)


class SpanEntityScore:
    """Accumulates span-level precision / recall / F1 across batches.

    Args:
        id2label: Mapping from integer label id to label string
            (e.g. ``{1: "BioNE"}``).  Used only for per-class reporting in
            :meth:`result`.

    Usage::

        metric = SpanEntityScore(id2label={1: "BioNE"})
        for sample in eval_samples:
            metric.update(true_subject=gold_spans, pred_subject=pred_spans)
        overall, per_class = metric.result()
        # overall: {"precision": ..., "recall": ..., "f1": ...}
        # per_class: {"BioNE": {"precision": ..., "recall": ..., "f1": ...}}
    """

    def __init__(self, id2label):
        self.id2label = id2label
        self.reset()

    def reset(self):
        self.origins = []
        self.founds = []
        self.rights = []

    def compute(self, origin, found, right):
        logger.debug("right=%d  found=%d  origin=%d", right, found, origin)
        recall    = 0 if origin == 0 else right / origin
        precision = 0 if found  == 0 else right / found
        f1 = (
            0.0
            if recall + precision == 0
            else 2 * precision * recall / (precision + recall)
        )
        return recall, precision, f1

    def result(self):
        """Return overall and per-class metrics.

        Returns:
            Tuple of:
                overall (dict): ``{"precision": float, "recall": float, "f1": float}``
                class_info (dict): per-label-name dict with the same keys.
        """
        class_info = {}
        origin_counter = Counter([self.id2label[x[0]] for x in self.origins])
        found_counter  = Counter([self.id2label[x[0]] for x in self.founds])
        right_counter  = Counter([self.id2label[x[0]] for x in self.rights])
        for type_, count in origin_counter.items():
            origin = count
            found  = found_counter.get(type_, 0)
            right  = right_counter.get(type_, 0)
            recall, precision, f1 = self.compute(origin, found, right)
            class_info[type_] = {
                "precision": round(precision, 4),
                "recall":    round(recall,    4),
                "f1":        round(f1,        4),
            }
        origin = len(self.origins)
        found  = len(self.founds)
        right  = len(self.rights)
        recall, precision, f1 = self.compute(origin, found, right)
        return {"precision": precision, "recall": recall, "f1": f1}, class_info

    def update(self, true_subject, pred_subject):
        """Accumulate one sample's spans.

        Args:
            true_subject: List of ``(label_id, start, end)`` gold spans.
            pred_subject: List of ``(label_id, start, end)`` predicted spans.
        """
        self.origins.extend(true_subject)
        self.founds.extend(pred_subject)
        self.rights.extend(
            [span for span in pred_subject if span in true_subject]
        )
