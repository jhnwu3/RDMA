#!/usr/bin/env python3
"""
Bio_ClinicalBERT token-classification NER fine-tuning on RareDis.

Delegates the training loop to the PyHealth ``Trainer``, which handles the
optimizer, gradient clipping, per-epoch evaluation, and loss-based
checkpointing.  A custom ``evaluate()`` using ``SpanEntityScore`` reports
the final per-entity-type F1 on the test set after training.

All tokenisation is pre-computed at ``set_task()`` time inside the task's
``__call__``; the DataLoader receives ready-to-stack ``torch.LongTensor``
objects.

Trade-offs vs a manual training loop:
  - Optimizer: single AdamW LR for all parameters (no differential LR).
  - Checkpointing: best model selected by lowest val loss, not val F1.
  - LR schedule: none (PyHealth Trainer does not add a warmup scheduler).

Usage (from RDMA repo root)::

    python baselines/bioclinicalbert_ner_trainer.py \\
        --gpu_id 0 --num_epochs 10 --batch_size 16

    # Dry-run (no training):
    python baselines/bioclinicalbert_ner_trainer.py --dry_run
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
from tqdm import tqdm

# ── Path setup ────────────────────────────────────────────────────────────────
_RDMA_ROOT = Path("/home/johnwu3/projects/rare_disease/workspace/repos/RDMA")
_PYHEALTH_ROOT = Path("/home/johnwu3/projects/rare_disease/workspace/repos/PyHealth")
_RESULTS_DIR = Path("/home/johnwu3/projects/rare_disease/workspace/results")

sys.path.insert(0, str(_PYHEALTH_ROOT))
sys.path.insert(0, str(_RDMA_ROOT))

from datasets.raredis import RareDisDataset  # noqa: E402
from tasks.bioclinicalbert_ner_raredis import BioClinicalBERTNERTask  # noqa: E402
from models.bioclinicalbert_ner import BioClinicalBERTNERModel  # noqa: E402
from metrics.ner_metrics import SpanEntityScore  # noqa: E402
from splitter.splitter import split_by_function  # noqa: E402
from pyhealth.trainer import Trainer  # noqa: E402

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def ts(msg: str) -> None:
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S} - {msg}", flush=True)


# ── DataLoader helpers ────────────────────────────────────────────────────────


class NERDataset(Dataset):
    """Thin wrapper that exposes a list/subset of pre-processed samples."""

    def __init__(self, samples) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        return self.samples[idx]


def collate_fn(batch: List[Dict]) -> Dict:
    """Stack tensor fields; keep string fields as lists."""
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
        "word_ids": torch.stack([b["word_ids"] for b in batch]),
        "split": [b.get("split") for b in batch],
        "patient_id": [b.get("patient_id") for b in batch],
    }


# ── Evaluation ────────────────────────────────────────────────────────────────


def evaluate(
    loader: DataLoader,
    model: BioClinicalBERTNERModel,
    device: torch.device,
    desc: str = "Eval",
) -> Tuple[Dict[str, float], List[Dict]]:
    """Evaluate *model* using ``SpanEntityScore``, returning P/R/F1/loss.

    Word-level predictions are obtained by taking the predicted label at
    the first subword position of each word (identified via ``word_ids``).
    Gold labels are recovered from the ``labels`` tensor in the same way.

    Args:
        loader: DataLoader over a ``NERDataset`` (pre-processed samples).
        model: The ``BioClinicalBERTNERModel`` to evaluate.
        device: Torch device.
        desc: Description string for the tqdm progress bar.

    Returns:
        Tuple of (aggregate_metrics dict, per_doc_list) where per_doc_list
        entries have integer tp/fp/fn for bootstrap CI.
    """
    b_id2label = {
        BioClinicalBERTNERTask.LABEL2ID[f"B-{et}"]: et
        for et in BioClinicalBERTNERTask.ENTITY_TYPES
    }
    metric = SpanEntityScore(b_id2label)
    total_loss = 0.0
    n_batches = 0
    per_docs: List[Dict] = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc, leave=False):
            input_ids = batch["input_ids"].to(device)  # [B, seq]
            attention_mask = batch["attention_mask"].to(device)  # [B, seq]
            labels = batch["labels"].to(device)  # [B, seq]
            word_ids = batch["word_ids"]  # [B, seq] (cpu)
            patient_ids = batch.get("patient_id", [None] * input_ids.size(0))

            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            if "loss" in out:
                total_loss += out["loss"].item()
            n_batches += 1

            # Argmax over label dimension → predicted label ids [B, seq]
            pred_ids = torch.argmax(out["logits"], dim=-1).cpu()  # [B, seq]
            gold_ids = labels.cpu()  # [B, seq]

            for i in range(input_ids.size(0)):
                wids = word_ids[i]  # [seq]
                preds = pred_ids[i]  # [seq]
                golds = gold_ids[i]  # [seq]

                word_preds = BioClinicalBERTNERModel.collapse_to_word_preds(wids, preds)
                word_golds = BioClinicalBERTNERModel.collapse_to_word_preds(wids, golds)

                pred_spans = BioClinicalBERTNERModel.bio_to_spans(word_preds, BioClinicalBERTNERTask.ID2LABEL, BioClinicalBERTNERTask.LABEL2ID)
                gold_spans = BioClinicalBERTNERModel.bio_to_spans(word_golds, BioClinicalBERTNERTask.ID2LABEL, BioClinicalBERTNERTask.LABEL2ID)
                metric.update(true_subject=gold_spans, pred_subject=pred_spans)

                pred_set = set(map(tuple, pred_spans))
                gold_set = set(map(tuple, gold_spans))
                tp = len(pred_set & gold_set)
                fp = len(pred_set - gold_set)
                fn = len(gold_set - pred_set)
                per_docs.append({
                    "id": str(patient_ids[i]) if patient_ids[i] is not None else str(len(per_docs)),
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                })

    eval_info, entity_info = metric.result()
    results = dict(eval_info)
    results["loss"] = total_loss / max(n_batches, 1)

    logger.info("\n***** %s results *****", desc)
    logger.info("  ".join(f"{k}={v:.4f}" for k, v in results.items()))
    for etype, info in sorted(entity_info.items()):
        logger.info(
            "  [%s] %s",
            etype,
            "  ".join(f"{k}={v:.4f}" for k, v in info.items()),
        )
    return results, per_docs


# ── Optuna HPO ────────────────────────────────────────────────────────────────


def _objective(
    trial,
    train_dataset: NERDataset,
    dev_dataset: NERDataset,
    args,
    device_str: str,
) -> float:
    """Optuna objective: train a trial model, return val F1."""
    lr = trial.suggest_float("lr", 1e-5, 1e-4, log=True)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    patience = trial.suggest_int("patience", 1, 3)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        sampler=SequentialSampler(dev_dataset),
        collate_fn=collate_fn,
    )

    model = BioClinicalBERTNERModel(dataset=None, model_name_or_path=args.model_path)
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

    results, _ = evaluate(
        dev_loader,
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
    train_dataset: NERDataset,
    dev_dataset: NERDataset,
    device_str: str,
) -> None:
    """Run Optuna study and patch *args* with the best hyperparameters."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    ts(f"Starting Optuna HPO: {args.n_trials} trials, {args.tune_epochs} epochs each")
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: _objective(trial, train_dataset, dev_dataset, args, device_str),
        n_trials=args.n_trials,
    )

    best = study.best_params
    ts(f"Best params: {best}  (val F1={study.best_value:.4f})")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_path = args.output_dir / "best_hparams.json"
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
        description="Bio_ClinicalBERT NER fine-tuning on RareDis (PyHealth Trainer)"
    )
    parser.add_argument(
        "--model_path", type=str, default="emilyalsentzer/Bio_ClinicalBERT"
    )
    parser.add_argument("--max_seq_length", type=int, default=512)
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
    parser.add_argument("--n_trials", type=int, default=10)
    parser.add_argument("--tune_epochs", type=int, default=5)
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
        default=_RESULTS_DIR / "raredis" / "bioclinicalbert_ner_trainer",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Process one sample and exit (no training)",
    )
    parser.add_argument(
        "--audit_json",
        type=Path,
        default=None,
        help="Write per-document audit JSON (for bootstrap CI) to this path",
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

    ts(f"Running set_task with BioClinicalBERTNERTask (model={args.model_path}) …")
    task = BioClinicalBERTNERTask(
        model_name_or_path=args.model_path,
        max_seq_length=args.max_seq_length,
    )
    sample_dataset = dataset.set_task(task)
    ts(f"  Total samples: {len(sample_dataset)}")
    print(sample_dataset[0])  # Show example pre-processed sample dict
    ts("Partitioning SampleDataset by explicit split field (train/dev/test) …")
    train_split, dev_split, test_split = split_by_function(
        sample_dataset,
        split_key="split",
        splits=("train", "dev", "test"),
    )
    train_dataset = NERDataset(train_split)
    dev_dataset = NERDataset(dev_split)
    test_dataset = NERDataset(test_split)
    ts(
        f"  Train: {len(train_dataset)}"
        f"  Dev: {len(dev_dataset)}"
        f"  Test: {len(test_dataset)}"
    )

    # ── Dry-run ───────────────────────────────────────────────────────
    if args.dry_run:
        ts("Dry-run: inspecting first training sample …")
        sample = train_dataset[0]
        ts(f"  input_ids shape : {sample['input_ids'].shape}")
        ts(f"  labels shape    : {sample['labels'].shape}")
        ts(f"  word_ids shape  : {sample['word_ids'].shape}")
        ts(f"  split           : {sample.get('split')}")
        n_entities = (sample["labels"] >= 0).sum().item() - (
            sample["labels"] == 0
        ).sum().item()
        ts(f"  non-O label tokens: {n_entities}")
        ts("Dry-run complete.")
        return

    # ── Optuna HPO ────────────────────────────────────────────────────
    if args.tune:
        tune(args, train_dataset, dev_dataset, device_str)

    # ── DataLoaders ───────────────────────────────────────────────────
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.batch_size,
        sampler=SequentialSampler(dev_dataset),
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        sampler=SequentialSampler(test_dataset),
        collate_fn=collate_fn,
    )

    # ── Model ─────────────────────────────────────────────────────────
    ts(f"Loading BioClinicalBERTNERModel from {args.model_path} …")
    model = BioClinicalBERTNERModel(dataset=None, model_name_or_path=args.model_path)

    # ── PyHealth Trainer ──────────────────────────────────────────────
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer = Trainer(
        model=model,
        device=device_str,
        enable_logging=True,
        output_path=str(args.output_dir),
        exp_name="run",
    )

    ts(f"Training: epochs={args.num_epochs}  lr={args.lr}  batch={args.batch_size}")
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

    # ── Final test evaluation ─────────────────────────────────────────
    ts("Running test evaluation …")
    test_results, per_docs = evaluate(test_loader, model, device, desc="Test")

    results_path = args.output_dir / "test_results.json"
    with open(results_path, "w") as fh:
        json.dump(test_results, fh, indent=2)
    ts(f"Test results saved to {results_path}")
    ts(
        f"Test F1={test_results['f1']:.4f}"
        f"  P={test_results['precision']:.4f}"
        f"  R={test_results['recall']:.4f}"
    )

    audit_path = args.audit_json or args.output_dir / "eval_bioclinicalbert_ner.json"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with open(audit_path, "w") as fh:
        json.dump({
            "metrics": {
                "precision": test_results["precision"],
                "recall": test_results["recall"],
                "f1": test_results["f1"],
                "tp": sum(d["tp"] for d in per_docs),
                "fp": sum(d["fp"] for d in per_docs),
                "fn": sum(d["fn"] for d in per_docs),
                "documents_scored": len(per_docs),
            },
            "documents": per_docs,
        }, fh, indent=2)
    ts(f"Audit JSON saved to {audit_path}")


if __name__ == "__main__":
    main()
