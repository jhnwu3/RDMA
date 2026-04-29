"""
Generic LongTensor schema FeatureProcessor.

This processor's only job is to declare the field type in the PyHealth schema
and ensure every field value is a ``torch.LongTensor``.  All domain logic
(tokenisation, label alignment) lives in the task modules.
"""

import torch

from pyhealth.processors.base_processor import FeatureProcessor


class LongTensorProcessor(FeatureProcessor):
    """Converts a tensor or list field value to ``torch.LongTensor``.

    Shallow / atomic: all domain logic (tokenisation, BIO label alignment)
    lives in the task.  Used in ``input_schema`` and ``output_schema``
    wherever a ``LongTensor`` field is declared.
    """

    def process(self, value) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value.long()
        return torch.tensor(value, dtype=torch.long)

    def is_token(self) -> bool:
        return True

    def schema(self) -> tuple:
        return ("value",)

    def dim(self) -> tuple:
        return (1,)

    def spatial(self) -> tuple:
        return (False,)
