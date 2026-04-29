#!/usr/bin/env python3
"""
BioBERT-MRC fine-tuning on RareDis NER — PyHealth Trainer version.

Delegates the training loop to the PyHealth ``Trainer``, which handles the
optimizer, gradient clipping, per-epoch evaluation, and loss-based
checkpointing.  A custom ``evaluate()`` using ``SpanEntityScore`` reports
the final per-entity-type F1 on the test set after training.

Trade-offs vs the manual-loop version (biobert_mrc_raredis.py):
  - Optimizer: single AdamW LR for all parameters (no differential LR).
  - Checkpointing: best model selected by lowest val loss, not val F1.
  - LR schedule: none (PyHealth Trainer does not add a warmup scheduler).

Usage (from RDMA repo root):
    python baselines/biobert_mrc_trainer.py \\
        --gpu_id 0 --num_epochs 10 --batch_size 16

    # Dry-run (no training):
    python baselines/biobert_mrc_trainer.py --dry_run
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler

# ── Path setup ────────────────────────────────────────────────────────────────
_RDMA_ROOT = Path("/home/johnwu3/projects/rare_disease/workspace/repos/RDMA")
_BIOBERT_ROOT = Path("/home/johnwu3/projects/rare_disease/workspace/repos/BioBERT-MRC")
_PYHEALTH_ROOT = Path("/home/johnwu3/projects/rare_disease/workspace/repos/PyHealth")
_RESULTS_DIR = Path("/home/johnwu3/projects/rare_disease/workspace/results")

# RDMA must have the highest priority so its 'processors' package takes
# precedence over the identically-named package inside BioBERT-MRC.
sys.path.insert(0, str(_BIOBERT_ROOT))
sys.path.insert(0, str(_PYHEALTH_ROOT))
sys.path.insert(0, str(_RDMA_ROOT))  # inserted last → index 0

from datasets.raredis import RareDisDataset  # noqa: E402
from tasks.biobert_mrc_raredis import BioBERTMRCTask  # noqa: E402
from processors.biobert_mrc_ner_processor import (  # noqa: E402
    BioBERTMRCNERProcessor,
    ID2LABEL,
)
from models.biobert_span_ner import BertSpanNERModel  # noqa: E402
from splitter.splitter import split_by_function  # noqa: E402
from pyhealth.trainer import Trainer  # noqa: E402

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
    """Wraps task samples and applies the processor on-the-fly."""

    def __init__(
        self,
        samples: List[Dict],
        processor: BioBERTMRCNERProcessor,
    ) -> None:
        self.samples = samples
        self.processor = processor

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        return self.processor.process(self.samples[idx])


def collate_fn(batch: List[Dict]) -> Tuple[torch.Tensor, ...]:
    """Collate into padded tensors (tuple form).

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
    """Collate into a keyword-argument dict for ``model(**batch)``."""
    ids, mask, segs, starts, ends, lens = collate_fn(batch)
    return {
        "input_ids": ids,
        "attention_mask": mask,
        "segment_ids": segs,
        "start_ids": starts,
        "end_ids": ends,
        "input_lens": lens,
    }


# ── Evaluation ────────────────────────────────────────────────────────────────


def evaluate(
    samples: List[Dict],
    processor: BioBERTMRCNERProcessor,
    model: BertSpanNERModel,
    device: torch.device,
    desc: str = "Eval",
) -> Dict[str, float]:
    """Evaluate *model* on *samples*, returning precision/recall/F1."""
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
            "  [%s] %s",
            etype,
            "  ".join(f"{k}={v:.4f}" for k, v in info.items()),
        )
    return results


# ── Optuna HPO ────────────────────────────────────────────────────────────────


def _objective(
    trial,
    train_samples: List[Dict],
    dev_samples: List[Dict],
    processor: "BioBERTMRCNERProcessor",
    args,
    device_str: str,
) -> float:
    """Optuna objective: train a trial model, return val F1."""
    lr = trial.suggest_float("lr", 1e-5, 1e-4, log=True)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    patience = trial.suggest_int("patience", 1, 3)

    train_loader = DataLoader(
        MRCNERDataset(train_samples, processor),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=dict_collate_fn,
    )
    dev_loader = DataLoader(
        MRCNERDataset(dev_samples, processor),
        batch_size=batch_size,
        sampler=SequentialSampler(MRCNERDataset(dev_samples, processor)),
        collate_fn=dict_collate_fn,
    )

    model = BertSpanNERModel(dataset=None, model_name_or_path=args.model_path)
    trainer = Trainer(
        model=model,
        device=device_str,
        enable_logging=False,
        output_path=str(args.output_dir / f"trial_{trial.number}"),
        exp_name="tune",
    )
    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=dev_loader,
        epochs=args.tune_epochs,
        optimizer_class=torch.optim.AdamW,
        optimizer_params={"lr": lr},
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        monitor="loss",
        monitor_criterion="min",
        load_best_model_at_last=True,
        patience=patience,
    )

    results = evaluate(
        dev_samples,
        processor,
        model,
        torch.device(device_str),
        desc=f"Trial {trial.number}",
    )
    ts(
        f"Trial {trial.number}: lr={lr:.2e}  batch={batch_size}"
        f"  patience={patience}  → val F1={results['f1']:.4f}"
    )
    return results["f1"]


def tune(
    args,
    train_samples: List[Dict],
    dev_samples: List[Dict],
    processor: "BioBERTMRCNERProcessor",
    device_str: str,
) -> None:
    """Run Optuna study and patch *args* with the best hyperparameters."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    ts(f"Starting Optuna HPO: {args.n_trials} trials, {args.tune_epochs} epochs each")
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: _objective(
            trial, train_samples, dev_samples, processor, args, device_str
        ),
        n_trials=args.n_trials,
    )

    best = study.best_params
    ts(f"Best params: {best}  (val F1={study.best_value:.4f})")

    best_path = args.output_dir / "best_hparams.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(best_path, "w") as fh:
        json.dump({**best, "best_val_f1": study.best_value}, fh, indent=2)
    ts(f"Best hyperparams saved to {best_path}")

    # Patch args so the final training run uses the best values
    args.lr = best["lr"]
    args.batch_size = best["batch_size"]
    args.patience = best["patience"]


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="BioBERT-MRC fine-tuning on RareDis (PyHealth Trainer)"
    )
    parser.add_argument("--model_path", type=str, default=_BIOBERT_MODEL)
    parser.add_argument("--max_seq_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-5,
        help="Learning rate for all parameters (single AdamW)",
    )
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument(
        "--patience",
        type=int,
        default=2,
        help="Early-stopping patience (epochs without val-loss improvement)",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run Optuna HPO before final training",
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=10,
        help="Number of Optuna trials",
    )
    parser.add_argument(
        "--tune_epochs",
        type=int,
        default=5,
        help="Epochs per Optuna trial (lightweight)",
    )
    parser.add_argument(
        "--gpu_id",
        type=lambda x: None if x.lower() == "none" else int(x),
        default=0,
        metavar="N|none",
    )
    parser.add_argument(
        "--condor",
        action="store_true",
        help="Running under HTCondor: use generic 'cuda' device",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=_RESULTS_DIR / "raredis" / "biobert_mrc_trainer",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Process one sample and exit (no training)",
    )
    args = parser.parse_args()

    if args.condor:
        device_str = "cuda"
    elif args.gpu_id is not None and torch.cuda.is_available():
        device_str = f"cuda:{args.gpu_id}"
    else:
        device_str = "cpu"
    device = torch.device(device_str)
    ts(f"Device: {device}")

    # ── Dataset + task ────────────────────────────────────────────────
    ts("Loading RareDisDataset …")
    dataset = RareDisDataset()
    sample_dataset = dataset.set_task(BioBERTMRCTask())
    ts(f"  Total samples: {len(sample_dataset)}")

    ts("Partitioning SampleDataset by explicit split field (train/dev/test) …")
    train_samples, dev_samples, test_samples = split_by_function(
        sample_dataset,
        split_key="split",
        splits=("train", "dev", "test"),
    )
    ts(
        f"  Train: {len(train_samples)}"
        f"  Dev: {len(dev_samples)}"
        f"  Test: {len(test_samples)}"
    )

    # ── Processor ─────────────────────────────────────────────────────
    ts(f"Initialising processor (model={args.model_path}) …")
    processor = BioBERTMRCNERProcessor(
        model_name_or_path=args.model_path,
        max_seq_length=args.max_seq_length,
    )

    # ── Optuna HPO ────────────────────────────────────────────────────
    if args.tune:
        tune(args, train_samples, dev_samples, processor, device_str)

    # ── Dry-run ───────────────────────────────────────────────────────
    if args.dry_run:
        ts("Dry-run: processing first sample …")
        feat = processor.process(train_samples[0])
        ts(f"  input_ids shape : {tuple(feat['input_ids'].shape)}")
        ts(f"  input_len       : {feat['input_len']}")
        ts(f"  subjects        : {feat['subjects'][0]}")
        ts("Dry-run complete.")
        return

    # ── DataLoaders ───────────────────────────────────────────────────
    train_dataset = MRCNERDataset(train_samples, processor)
    dev_dataset = MRCNERDataset(dev_samples, processor)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=dict_collate_fn,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.batch_size,
        sampler=SequentialSampler(dev_dataset),
        collate_fn=dict_collate_fn,
    )

    # ── Model ─────────────────────────────────────────────────────────
    ts(f"Loading BertSpanNERModel from {args.model_path} …")
    model = BertSpanNERModel(dataset=None, model_name_or_path=args.model_path)

    # ── PyHealth Trainer ───────────────────────────────────────────────
    # Trainer handles: optimizer, gradient clipping, per-epoch val
    # evaluation, and loss-based best-model checkpointing.
    # Note: differential LR (BERT encoder vs span heads) is not
    # supported by PyHealth Trainer — all parameters share args.lr.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer = Trainer(
        model=model,
        device=device_str,
        enable_logging=True,
        output_path=str(args.output_dir),
        exp_name="run",
    )

    ts(f"Training: epochs={args.num_epochs}" f"  lr={args.lr}  batch={args.batch_size}")
    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=dev_loader,
        epochs=args.num_epochs,
        optimizer_class=torch.optim.AdamW,
        optimizer_params={"lr": args.lr},
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        monitor="loss",
        monitor_criterion="min",
        load_best_model_at_last=True,
        patience=args.patience,
    )

    # Save in HuggingFace format (Trainer checkpoint is plain state_dict)
    hf_ckpt_dir = args.output_dir / "best_hf"
    ts(f"Saving HuggingFace checkpoint to {hf_ckpt_dir} …")
    model.save_pretrained(str(hf_ckpt_dir))
    torch.save(args, hf_ckpt_dir / "training_args.bin")

    # ── Final test evaluation ──────────────────────────────────────────
    ts("Running test evaluation …")
    test_results = evaluate(test_samples, processor, model, device, desc="Test")

    results_path = args.output_dir / "test_results.json"
    with open(results_path, "w") as fh:
        json.dump(test_results, fh, indent=2)
    ts(f"Test results saved to {results_path}")
    ts(
        f"Test F1={test_results['f1']:.4f}"
        f"  P={test_results['precision']:.4f}"
        f"  R={test_results['recall']:.4f}"
    )


if __name__ == "__main__":
    main()
