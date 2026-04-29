#!/usr/bin/env python3
"""
BioBERT-MRC fine-tuning baseline for RareDis NER.

Trains BertSpanForNer (span extraction via MRC) on the RareDis corpus using
the PyHealth RareDisDataset + BioBERTMRCTask + BioBERTMRCNERProcessor pipeline.

Usage (from RDMA repo root):
    python baselines/biobert_mrc_raredis.py \\
        --gpu_id 0 \\
        --num_epochs 10 \\
        --batch_size 16 \\
        --output_dir /path/to/save

    # Dry-run (one batch, no training):
    python baselines/biobert_mrc_raredis.py --dry_run
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler

# ── Path setup ────────────────────────────────────────────────────────────────
_RDMA_ROOT = Path("/home/johnwu3/projects/rare_disease/workspace/repos/RDMA")
_BIOBERT_ROOT = Path("/home/johnwu3/projects/rare_disease/workspace/repos/BioBERT-MRC")
_RESULTS_DIR = Path("/home/johnwu3/projects/rare_disease/workspace/results")


# RDMA must have the highest priority so its 'processors' package takes
# precedence over the identically-named package inside BioBERT-MRC.
sys.path.insert(0, str(_BIOBERT_ROOT))
sys.path.insert(0, str(_RDMA_ROOT))  # inserted last → index 0

from datasets.raredis import RareDisDataset  # noqa: E402
from tasks.biobert_mrc_raredis import BioBERTMRCTask  # noqa: E402
from processors.biobert_mrc_ner_processor import (  # noqa: E402
    BioBERTMRCNERProcessor,
    ID2LABEL,
)
from models.biobert_span_ner import BertSpanNERModel  # noqa: E402
from splitter.splitter import split_by_function  # noqa: E402

from callback.optimizater.adamw import AdamW  # noqa: E402
from callback.lr_scheduler import (  # noqa: E402
    get_linear_schedule_with_warmup,
)
from callback.progressbar import ProgressBar  # noqa: E402
from metrics.ner_metrics import SpanEntityScore  # noqa: E402

# ── Inlined from BioBERT-MRC/processors/utils_ner.py ─────────────────────────
# (utils_ner imports BertTokenizer from the vendored transformers copy which
# is not needed for span extraction; we take only what we use.)



# ── Constants ─────────────────────────────────────────────────────────────────
_BIOBERT_MODEL = str(_BIOBERT_ROOT / "BioBERTv1.1_P")

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def ts(msg: str) -> None:
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S} - {msg}", flush=True)


# ── Dataset wrapper ───────────────────────────────────────────────────────────


class MRCNERDataset(Dataset):
    """Wraps a list of raw task samples and applies the processor on-the-fly."""

    def __init__(self, samples: List[Dict], processor: BioBERTMRCNERProcessor) -> None:
        self.samples = samples
        self.processor = processor

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        return self.processor.process(self.samples[idx])


def collate_fn(batch: List[Dict]) -> Tuple[torch.Tensor, ...]:
    """Collate processed samples into padded tensors (tuple form).

    Dynamically trims to the maximum actual sequence length in the batch to
    save memory (mirrors the collate_fn in BioBERT-MRC/processors/).
    Used by the manual training loop.

    RareDis documents always fit in a single chunk, so each processor
    output has shape [1, max_seq_length].  We squeeze out the chunk
    dimension before stacking.
    """
    input_ids = torch.stack([b["input_ids"].squeeze(0) for b in batch])
    attention_mask = torch.stack(
        [b["attention_mask"].squeeze(0) for b in batch]
    )
    segment_ids = torch.stack(
        [b["segment_ids"].squeeze(0) for b in batch]
    )
    start_ids = torch.stack([b["start_ids"].squeeze(0) for b in batch])
    end_ids = torch.stack([b["end_ids"].squeeze(0) for b in batch])
    input_lens = torch.tensor(
        [b["input_len"][0] for b in batch], dtype=torch.long
    )

    max_len = input_lens.max().item()
    return (
        input_ids[:, :max_len],
        attention_mask[:, :max_len],
        segment_ids[:, :max_len],
        start_ids[:, :max_len],
        end_ids[:, :max_len],
        input_lens,
    )


def dict_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate processed samples into a dict of padded tensors.

    Returns the same data as ``collate_fn`` but as a keyword-argument dict,
    enabling ``model(**batch)`` calls and compatibility with the PyHealth
    ``Trainer``.
    """
    input_ids, attention_mask, segment_ids, start_ids, end_ids, input_lens = collate_fn(
        batch
    )
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "segment_ids": segment_ids,
        "start_ids": start_ids,
        "end_ids": end_ids,
        "input_lens": input_lens,
    }


# ── Evaluation ────────────────────────────────────────────────────────────────


def evaluate(
    samples: List[Dict],
    processor: BioBERTMRCNERProcessor,
    model: BertSpanNERModel,
    device: torch.device,
    desc: str = "Eval",
) -> Dict[str, float]:
    """Evaluate *model* on *samples* and return precision/recall/F1."""
    metric = SpanEntityScore(ID2LABEL)
    total_loss = 0.0
    pbar = ProgressBar(n_total=len(samples), desc=desc)

    model.eval()
    for step, sample in enumerate(samples):
        feat = processor.process(sample)
        # RareDis = 1 chunk; squeeze the chunk dimension
        lens = feat["input_len"][0]
        input_ids = feat["input_ids"][0:1, :lens].to(device)
        attention_mask = feat["attention_mask"][0:1, :lens].to(device)
        segment_ids = feat["segment_ids"][0:1, :lens].to(device)
        start_ids = feat["start_ids"][0:1, :lens].to(device)
        end_ids = feat["end_ids"][0:1, :lens].to(device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                segment_ids=segment_ids,
                start_ids=start_ids,
                end_ids=end_ids,
            )

        # Number of text-side tokens (used by bert_extract_item)
        text_len = int(
            torch.sum(
                attention_mask.view(-1) - segment_ids.view(-1)
            ).cpu()
        )
        pred_subjects = BertSpanNERModel.predict(
            outputs["start_logits"], outputs["end_logits"], text_len
        )
        metric.update(
            true_subject=feat["subjects"][0],
            pred_subject=pred_subjects,
        )
        total_loss += outputs["loss"].item()
        pbar(step)

    eval_info, entity_info = metric.result()
    results = dict(eval_info)
    results["loss"] = total_loss / max(len(samples), 1)

    logger.info("\n***** %s results *****", desc)
    logger.info("  ".join(f"{k}={v:.4f}" for k, v in results.items()))
    for etype, info in sorted(entity_info.items()):
        logger.info(
            "  [%s] %s", etype, "  ".join(f"{k}={v:.4f}" for k, v in info.items())
        )

    return results


# ── Training loop ─────────────────────────────────────────────────────────────


def train(
    args: argparse.Namespace,
    train_samples: List[Dict],
    dev_samples: List[Dict],
    model: BertSpanNERModel,
    processor: BioBERTMRCNERProcessor,
    device: torch.device,
    output_dir: Path,
) -> None:
    train_dataset = MRCNERDataset(train_samples, processor)
    train_loader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=args.batch_size,
        collate_fn=collate_fn,
    )

    t_total = len(train_loader) // args.grad_accum * args.num_epochs
    warmup_steps = int(t_total * args.warmup_proportion)

    # Differential learning rates: lower LR for BERT, higher for classifier heads
    no_decay = ["bias", "LayerNorm.weight"]
    bert_params = list(model._bert_span.bert.named_parameters())
    start_params = list(model._bert_span.start_fc.named_parameters())
    end_params = list(model._bert_span.end_fc.named_parameters())

    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in bert_params if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
            "lr": args.lr,
        },
        {
            "params": [p for n, p in bert_params if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
            "lr": args.lr,
        },
        {
            "params": [
                p for n, p in start_params if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
            "lr": args.classifier_lr,
        },
        {
            "params": [p for n, p in start_params if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
            "lr": args.classifier_lr,
        },
        {
            "params": [p for n, p in end_params if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
            "lr": args.classifier_lr,
        },
        {
            "params": [p for n, p in end_params if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
            "lr": args.classifier_lr,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
    )

    ts(f"Training on {len(train_samples)} samples | {len(dev_samples)} dev samples")
    ts(
        f"Epochs={args.num_epochs}  batch={args.batch_size}  lr={args.lr}  classifier_lr={args.classifier_lr}"
    )
    ts(f"t_total={t_total}  warmup={warmup_steps}")

    best_f1 = 0.0
    global_step = 0

    for epoch in range(args.num_epochs):
        ts(f"Epoch {epoch + 1}/{args.num_epochs}")
        model.train()
        epoch_loss = 0.0
        pbar = ProgressBar(n_total=len(train_loader), desc="Training")

        for step, batch in enumerate(train_loader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, attention_mask, segment_ids, start_ids, end_ids, _ = batch

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                segment_ids=segment_ids,
                start_ids=start_ids,
                end_ids=end_ids,
            )
            loss = outputs["loss"]

            if args.grad_accum > 1:
                loss = loss / args.grad_accum
            loss.backward()
            epoch_loss += loss.item()

            pbar(step, {"loss": loss.item()})

            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scheduler.step()
                optimizer.step()
                model.zero_grad()
                global_step += 1

        ts(
            f"  Epoch {epoch + 1} avg loss: {epoch_loss / max(len(train_loader), 1):.4f}"
        )

        # Evaluate on dev set after each epoch
        dev_results = evaluate(
            dev_samples, processor, model, device, desc=f"Dev epoch {epoch+1}"
        )
        if dev_results["f1"] > best_f1:
            best_f1 = dev_results["f1"]
            hf_ckpt_dir = output_dir / "best_hf"
            hf_ckpt_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(hf_ckpt_dir))
            torch.save(args, hf_ckpt_dir / "training_args.bin")
            ts(f"  ✓ New best F1={best_f1:.4f} — checkpoint saved to {hf_ckpt_dir}")

        if "cuda" in str(device):
            torch.cuda.empty_cache()

    ts(f"Training complete.  Best dev F1: {best_f1:.4f}")


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="BioBERT-MRC fine-tuning on RareDis")
    parser.add_argument(
        "--model_path",
        type=str,
        default=_BIOBERT_MODEL,
        help="Path to BioBERT checkpoint (default: %(default)s)",
    )
    parser.add_argument("--max_seq_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument(
        "--lr", type=float, default=3e-5, help="Learning rate for BERT parameters"
    )
    parser.add_argument(
        "--classifier_lr",
        type=float,
        default=1e-3,
        help="Learning rate for span classifier heads",
    )
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_proportion", type=float, default=0.1)
    parser.add_argument(
        "--grad_accum", type=int, default=1, help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--gpu_id",
        type=lambda x: None if x.lower() == "none" else int(x),
        default=0,
        metavar="N|none",
        help="GPU device id; 'none' for CPU",
    )
    parser.add_argument(
        "--condor",
        action="store_true",
        help="Running under HTCondor: use generic 'cuda' device",
    )
    parser.add_argument(
        "--output_dir", type=Path, default=_RESULTS_DIR / "raredis" / "biobert_mrc"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Process one batch and exit (no training)",
    )
    args = parser.parse_args()

    # Device
    if args.condor:
        device = torch.device("cuda")
    elif args.gpu_id is not None and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu_id}")
    else:
        device = torch.device("cpu")
    ts(f"Device: {device}")

    # ── Dataset + task ────────────────────────────────────────────────
    ts("Loading RareDisDataset …")
    dataset = RareDisDataset()
    sample_dataset = dataset.set_task(BioBERTMRCTask())
    ts(f"  Total samples (all splits, all entity types): {len(sample_dataset)}")

    # PyHealth-style split using explicit split metadata emitted by the task
    ts("Partitioning SampleDataset by explicit split field (train/dev/test) …")
    train_samples, dev_samples, test_samples = split_by_function(
        sample_dataset,
        split_key="split",
        splits=("train", "dev", "test"),
    )

    ts(
        f"  Train: {len(train_samples)}  Dev: {len(dev_samples)}  Test: {len(test_samples)}"
    )

    # ── Processor ─────────────────────────────────────────────────────
    ts(f"Initialising BioBERTMRCNERProcessor (model={args.model_path}) …")
    processor = BioBERTMRCNERProcessor(
        model_name_or_path=args.model_path,
        max_seq_length=args.max_seq_length,
    )

    # ── Dry-run ───────────────────────────────────────────────────────
    if args.dry_run:
        ts("Dry-run: processing first batch …")
        sample = train_samples[0]
        feat = processor.process(sample)
        ts(f"  input_ids  shape : {tuple(feat['input_ids'].shape)}")
        ts(f"  input_len        : {feat['input_len']}")
        ts(f"  subjects         : {feat['subjects'][0]}")
        ts("Dry-run complete.")
        return

    # ── Model ─────────────────────────────────────────────────────────
    ts(f"Loading BertSpanNERModel from {args.model_path} …")
    model = BertSpanNERModel(dataset=None, model_name_or_path=args.model_path)
    model.to(device)

    # ── Train ─────────────────────────────────────────────────────────
    train(args, train_samples, dev_samples, model, processor, device, args.output_dir)

    # ── Final test evaluation ──────────────────────────────────────────
    ts("Loading best checkpoint for test evaluation …")
    best_model = BertSpanNERModel.load_from_checkpoint(
        str(args.output_dir / "best_hf")
    )
    best_model.to(device)

    test_results = evaluate(test_samples, processor, best_model, device, desc="Test")

    results_path = args.output_dir / "test_results.json"
    with open(results_path, "w") as fh:
        json.dump(test_results, fh, indent=2)
    ts(f"Test results saved to {results_path}")
    ts(
        f"Test F1={test_results['f1']:.4f}  P={test_results['precision']:.4f}  R={test_results['recall']:.4f}"
    )


if __name__ == "__main__":
    main()
