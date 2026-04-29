"""
PyHealth BaseModel wrapper around BertSpanForNer.

All BioBERT-MRC model code (BertSpanForNer, PoolerStartLogits, PoolerEndLogits,
FocalLoss, LabelSmoothingCrossEntropy) is inlined here so the only external
dependencies are HuggingFace ``transformers``, PyTorch, and PyHealth.

Usage (standalone / baselines script):
    model = BertSpanNERModel(dataset=None, model_name_or_path="/path/to/BioBERTv1.1_P")
    out   = model(input_ids=ids, attention_mask=mask, segment_ids=segs,
                  start_ids=starts, end_ids=ends)
    # out["loss"], out["start_logits"], out["end_logits"]

Usage (PyHealth Trainer):
    trainer = Trainer(model=model)
    trainer.train(train_dataloader=dict_loader, val_dataloader=dict_val_loader)
    # Trainer.evaluate() returns {"loss": ...} because model.mode is None.
"""

import sys
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import BertConfig, BertModel, BertPreTrainedModel

# ── PyHealth BaseModel ─────────────────────────────────────────────────────────
_PYHEALTH_ROOT = Path(__file__).resolve().parents[2] / "PyHealth"
if str(_PYHEALTH_ROOT) not in sys.path:
    sys.path.insert(0, str(_PYHEALTH_ROOT))

from pyhealth.models.base_model import BaseModel         # noqa: E402


# ── Inlined from BioBERT-MRC/models/layers/linears.py ─────────────────────────

class PoolerStartLogits(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super().__init__()
        self.dense = nn.Linear(hidden_size, num_classes)

    def forward(self, hidden_states, p_mask=None):
        return self.dense(hidden_states)


class PoolerEndLogits(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super().__init__()
        self.dense_0  = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()
        self.LayerNorm  = nn.LayerNorm(hidden_size)
        self.dense_1  = nn.Linear(hidden_size, num_classes)

    def forward(self, hidden_states, start_positions=None, p_mask=None):
        x = self.dense_0(torch.cat([hidden_states, start_positions], dim=-1))
        x = self.activation(x)
        x = self.LayerNorm(x)
        return self.dense_1(x)


# ── Inlined from BioBERT-MRC/losses/ ──────────────────────────────────────────

class FocalLoss(nn.Module):
    """Multi-class Focal loss."""
    def __init__(self, gamma=2, weight=None, ignore_index=-100):
        super().__init__()
        self.gamma        = gamma
        self.weight       = weight
        self.ignore_index = ignore_index

    def forward(self, input, target):
        logpt = F.log_softmax(input, dim=1)
        pt    = torch.exp(logpt)
        logpt = (1 - pt) ** self.gamma * logpt
        return F.nll_loss(logpt, target, self.weight, ignore_index=self.ignore_index)


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, eps=0.1, reduction="mean", ignore_index=-100):
        super().__init__()
        self.eps          = eps
        self.reduction    = reduction
        self.ignore_index = ignore_index

    def forward(self, output, target):
        c        = output.size()[-1]
        log_preds = F.log_softmax(output, dim=-1)
        if self.reduction == "sum":
            loss = -log_preds.sum()
        else:
            loss = -log_preds.sum(dim=-1)
            if self.reduction == "mean":
                loss = loss.mean()
        return loss * self.eps / c + (1 - self.eps) * F.nll_loss(
            log_preds, target, reduction=self.reduction, ignore_index=self.ignore_index
        )


# ── Inlined from BioBERT-MRC/models/bert_for_ner.py ──────────────────────────

class BertSpanForNer(BertPreTrainedModel):
    """Span-extraction MRC NER head on top of BERT.

    Inlined from BioBERT-MRC with the only change being that
    ``BertPreTrainedModel``, ``BertModel``, and the loss / pooler helpers are
    sourced from this file rather than BioBERT-MRC's vendored copies.
    """

    def __init__(self, config):
        super().__init__(config)
        self.soft_label  = config.soft_label
        self.num_labels  = config.num_labels
        self.loss_type   = config.loss_type
        self.bert        = BertModel(config)
        self.dropout     = nn.Dropout(config.hidden_dropout_prob)
        self.start_fc    = PoolerStartLogits(config.hidden_size, self.num_labels)
        end_input_size   = (
            config.hidden_size + self.num_labels
            if self.soft_label
            else config.hidden_size + 1
        )
        self.end_fc = PoolerEndLogits(end_input_size, self.num_labels)
        self.init_weights()

    def forward(
        self,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
        start_positions=None,
        end_positions=None,
    ):
        outputs         = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        sequence_output = self.dropout(outputs[0])
        start_logits    = self.start_fc(sequence_output)

        if start_positions is not None and self.training:
            if self.soft_label:
                batch_size = input_ids.size(0)
                seq_len    = input_ids.size(1)
                label_logits = torch.zeros(
                    batch_size, seq_len, self.num_labels,
                    device=input_ids.device,
                )
                label_logits.scatter_(2, start_positions.unsqueeze(2), 1)
            else:
                label_logits = start_positions.unsqueeze(2).float()
        else:
            label_logits = F.softmax(start_logits, -1)
            if not self.soft_label:
                label_logits = torch.argmax(label_logits, -1).unsqueeze(2).float()

        end_logits = self.end_fc(sequence_output, label_logits)
        result     = (start_logits, end_logits) + outputs[2:]

        if start_positions is not None and end_positions is not None:
            assert self.loss_type in ("lsr", "focal", "ce")
            if self.loss_type == "lsr":
                loss_fct = LabelSmoothingCrossEntropy()
            elif self.loss_type == "focal":
                loss_fct = FocalLoss()
            else:
                loss_fct = CrossEntropyLoss()

            active_loss         = (attention_mask.view(-1) - token_type_ids.view(-1)) == 1
            active_start_logits = start_logits.view(-1, self.num_labels)[active_loss]
            active_end_logits   = end_logits.view(-1, self.num_labels)[active_loss]
            active_start_labels = start_positions.view(-1)[active_loss]
            active_end_labels   = end_positions.view(-1)[active_loss]

            total_loss = (
                loss_fct(active_start_logits, active_start_labels)
                + loss_fct(active_end_logits, active_end_labels)
            ) / 2
            result = (total_loss,) + result

        return result


# ── PyHealth BaseModel wrapper ─────────────────────────────────────────────────

class BertSpanNERModel(BaseModel):
    """PyHealth ``BaseModel`` wrapping ``BertSpanForNer`` for MRC-NER.

    Setting ``mode = None`` tells the PyHealth ``Trainer`` not to compute
    standard classification metrics; ``Trainer.evaluate()`` will only report
    ``{"loss": ...}``.  Per-entity-type F1 evaluation is handled separately
    by the ``SpanEntityScore`` loop in the baselines script.

    Args:
        dataset: A PyHealth ``SampleDataset`` or ``None`` (for standalone use
            without the PyHealth data pipeline).
        model_name_or_path: Path to a BioBERT checkpoint directory containing
            ``config.json`` and ``pytorch_model.bin`` (e.g. ``BioBERTv1.1_P``).
        num_labels: Number of NER label classes.  Always 2 for MRC-NER
            (``O`` / ``BioNE``).
        loss_type: Loss function passed to ``BertSpanForNer``.  One of
            ``"ce"``, ``"lsr"``, or ``"focal"``.
        soft_label: Whether to use soft (interpolated) labels for the end
            classifier, as in the original BioBERT-MRC implementation.
    """

    mode = None  # suppress standard PyHealth metrics

    def __init__(
        self,
        dataset,
        model_name_or_path: str,
        num_labels: int = 2,
        loss_type: str = "ce",
        soft_label: bool = True,
    ) -> None:
        super().__init__(dataset=dataset)
        config = BertConfig.from_pretrained(
            model_name_or_path,
            num_labels=num_labels,
        )
        # Newer transformers no longer auto-stores arbitrary kwargs as config
        # attributes — set them explicitly so BertSpanForNer.__init__ can read them.
        config.loss_type  = loss_type
        config.soft_label = soft_label
        self._bert_span = BertSpanForNer.from_pretrained(
            model_name_or_path, config=config
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        segment_ids: torch.Tensor,
        start_ids: Optional[torch.Tensor] = None,
        end_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Run ``BertSpanForNer`` and return a result dict.

        Args:
            input_ids: Token ids ``[batch, seq_len]``.
            attention_mask: Attention mask ``[batch, seq_len]``.
            segment_ids: Token type ids (0 = text, 1 = query) ``[batch, seq_len]``.
            start_ids: Start position labels ``[batch, seq_len]`` (training only).
            end_ids: End position labels ``[batch, seq_len]`` (training only).
            **kwargs: Ignored extra keys (``input_len``, ``subjects``,
                ``patient_id``, ``input_lens``, …) that a DataLoader may pass.

        Returns:
            During training (``start_ids`` provided)::

                {"loss": scalar, "start_logits": [B, L, 2], "end_logits": [B, L, 2]}

            During inference::

                {"start_logits": [B, L, 2], "end_logits": [B, L, 2]}
        """
        outputs = self._bert_span(
            input_ids=input_ids.to(self.device),
            attention_mask=attention_mask.to(self.device),
            token_type_ids=segment_ids.to(self.device),
            start_positions=start_ids.to(self.device) if start_ids is not None else None,
            end_positions=end_ids.to(self.device) if end_ids is not None else None,
        )

        if start_ids is not None:
            loss, start_logits, end_logits = outputs[:3]
            return {
                "loss": loss,
                "start_logits": start_logits,
                "end_logits": end_logits,
            }
        else:
            start_logits, end_logits = outputs[:2]
            return {
                "start_logits": start_logits,
                "end_logits": end_logits,
            }

    # ------------------------------------------------------------------
    # Span extraction
    # ------------------------------------------------------------------

    @staticmethod
    def predict(
        start_logits: torch.Tensor,
        end_logits: torch.Tensor,
        input_len: int,
    ):
        """Extract non-overlapping (label_id, start, end) spans from logits.

        Args:
            start_logits: ``[1, seq_len, num_labels]`` tensor.
            end_logits:   ``[1, seq_len, num_labels]`` tensor.
            input_len:    Actual (non-padded) sequence length.

        Returns:
            List of ``(label_id, start, end)`` tuples in 0-based text-token
            space (i.e. after stripping the leading [CLS] and trailing tokens).
        """
        import numpy as np
        start_pred = torch.argmax(start_logits, -1).cpu().numpy()[0][1:input_len - 1]
        end_pred   = torch.argmax(end_logits,   -1).cpu().numpy()[0][1:input_len - 1]
        spans = []
        for i, s_l in enumerate(start_pred):
            if s_l == 0:
                continue
            for j, e_l in enumerate(end_pred[i:]):
                if s_l == e_l:
                    spans.append((s_l, i, i + j))
                    break
        if len(spans) < 2:
            return spans
        result = []
        for i in range(len(spans) - 1):
            if spans[i][2] < spans[i + 1][1]:
                result.append(spans[i])
        result.append(spans[-1])
        return result

    # ------------------------------------------------------------------
    # Checkpoint helpers (HuggingFace format, not plain state_dict)
    # ------------------------------------------------------------------

    def save_pretrained(self, output_dir: str) -> None:
        """Save the inner ``BertSpanForNer`` as a HuggingFace checkpoint.

        Saves both ``config.json`` and ``pytorch_model.bin`` so the checkpoint
        can be reloaded with :meth:`load_from_checkpoint`.

        Note:
            The PyHealth ``Trainer`` uses ``save_ckpt`` (plain ``state_dict``)
            for its own checkpointing.  Call this method explicitly at the end
            of training (as done in the baselines script) to obtain a
            HuggingFace-compatible checkpoint.
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self._bert_span.save_pretrained(output_dir)

    @classmethod
    def load_from_checkpoint(
        cls,
        output_dir: str,
        dataset=None,
        num_labels: int = 2,
        loss_type: str = "ce",
        soft_label: bool = True,
    ) -> "BertSpanNERModel":
        """Load a ``BertSpanNERModel`` from a previously saved checkpoint.

        Args:
            output_dir: Directory written by :meth:`save_pretrained`.
            dataset: Optional PyHealth ``SampleDataset`` (may be ``None``).
            num_labels: Must match the value used during training.
            loss_type: Must match the value used during training.
            soft_label: Must match the value used during training.

        Returns:
            A new ``BertSpanNERModel`` loaded from ``output_dir``.
        """
        return cls(
            dataset=dataset,
            model_name_or_path=output_dir,
            num_labels=num_labels,
            loss_type=loss_type,
            soft_label=soft_label,
        )
