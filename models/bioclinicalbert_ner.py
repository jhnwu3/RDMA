"""
PyHealth BaseModel wrapper around Bio_ClinicalBERT for token-classification NER.

Uses HuggingFace ``AutoModelForTokenClassification`` with the 13-label BIO
scheme defined in ``BioClinicalBERTNERTask``.

Setting ``mode = None`` tells the PyHealth ``Trainer`` not to compute
standard classification metrics; ``Trainer.evaluate()`` will only report
``{"loss": ...}``.  Per-entity-type F1 evaluation is handled separately
by the ``SpanEntityScore`` loop in the baselines script.

Usage (standalone)::

    model = BioClinicalBERTNERModel(dataset=None)
    out   = model(input_ids=ids, attention_mask=mask, labels=lbls)
    # out["loss"], out["logits"]

Usage (PyHealth Trainer)::

    trainer = Trainer(model=model)
    trainer.train(train_dataloader=loader, val_dataloader=val_loader)
"""

import sys
from pathlib import Path
from typing import Dict, Optional

import torch
from transformers import AutoModelForTokenClassification

# ── PyHealth BaseModel ─────────────────────────────────────────────────────────
_PYHEALTH_ROOT = Path(__file__).resolve().parents[2] / "PyHealth"
if str(_PYHEALTH_ROOT) not in sys.path:
    sys.path.insert(0, str(_PYHEALTH_ROOT))

from pyhealth.models.base_model import BaseModel  # noqa: E402

from tasks.bioclinicalbert_ner_raredis import BioClinicalBERTNERTask  # noqa: E402


class BioClinicalBERTNERModel(BaseModel):
    """PyHealth ``BaseModel`` wrapping ``AutoModelForTokenClassification``.

    ``mode = None`` suppresses standard PyHealth metrics so that
    ``Trainer.evaluate()`` only reports loss.  Per-entity-type F1 is
    computed by the ``evaluate()`` function in the baselines script.

    Args:
        dataset: A PyHealth ``SampleDataset`` or ``None`` (for standalone
            use without the PyHealth data pipeline).
        model_name_or_path: HuggingFace model id or local checkpoint path.
            Defaults to ``emilyalsentzer/Bio_ClinicalBERT``.
        num_labels: Number of NER label classes.  Defaults to
            ``BioClinicalBERTNERTask.NUM_LABELS`` (13).
        id2label: Label id → label string mapping.  Defaults to
            ``BioClinicalBERTNERTask.ID2LABEL``.
        label2id: Label string → id mapping.  Defaults to
            ``BioClinicalBERTNERTask.LABEL2ID``.
    """

    mode = None  # suppress standard PyHealth metrics

    def __init__(
        self,
        dataset,
        model_name_or_path: str = "emilyalsentzer/Bio_ClinicalBERT",
        num_labels: int = BioClinicalBERTNERTask.NUM_LABELS,
        id2label: Dict[int, str] = BioClinicalBERTNERTask.ID2LABEL,
        label2id: Dict[str, int] = BioClinicalBERTNERTask.LABEL2ID,
    ) -> None:
        super().__init__(dataset=dataset)
        self._hf_model = AutoModelForTokenClassification.from_pretrained(
            model_name_or_path,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Run ``AutoModelForTokenClassification`` and return a result dict.

        Args:
            input_ids: Token ids ``[batch, seq_len]``.
            attention_mask: Attention mask ``[batch, seq_len]``.
            labels: Label ids ``[batch, seq_len]``, ``-100`` for ignored
                positions.  Omit during inference.
            **kwargs: Ignored extra keys (``word_ids``, ``split``,
                ``patient_id``, …) that a DataLoader may pass.

        Returns:
            During training (``labels`` provided)::

                {"loss": scalar, "logits": [B, L, num_labels]}

            During inference::

                {"logits": [B, L, num_labels]}
        """
        out = self._hf_model(
            input_ids=input_ids.to(self.device),
            attention_mask=attention_mask.to(self.device),
            labels=labels.to(self.device) if labels is not None else None,
        )
        result: Dict[str, torch.Tensor] = {"logits": out.logits}
        if out.loss is not None:
            result["loss"] = out.loss
        return result

    # ------------------------------------------------------------------
    # Checkpoint helpers (HuggingFace format)
    # ------------------------------------------------------------------

    def save_pretrained(self, output_dir: str) -> None:
        """Save the inner HF model as a HuggingFace checkpoint.

        Saves ``config.json`` and ``pytorch_model.bin`` (or
        ``model.safetensors``) so the checkpoint can be reloaded with
        :meth:`load_from_checkpoint`.

        Note:
            The PyHealth ``Trainer`` uses ``save_ckpt`` (plain
            ``state_dict``) for its own checkpointing.  Call this method
            explicitly at the end of training (as done in the baselines
            script) to obtain a HuggingFace-compatible checkpoint.
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self._hf_model.save_pretrained(output_dir)

    @classmethod
    def load_from_checkpoint(
        cls,
        output_dir: str,
        dataset=None,
    ) -> "BioClinicalBERTNERModel":
        """Load a ``BioClinicalBERTNERModel`` from a previously saved checkpoint.

        Args:
            output_dir: Directory written by :meth:`save_pretrained`.
            dataset: Optional PyHealth ``SampleDataset`` (may be ``None``).

        Returns:
            A new ``BioClinicalBERTNERModel`` loaded from ``output_dir``.
        """
        return cls(dataset=dataset, model_name_or_path=output_dir)
