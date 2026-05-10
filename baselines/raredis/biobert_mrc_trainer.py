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
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from transformers import BertTokenizer

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
from tasks.utils import mrc_chunk_document  # noqa: E402
from models.biobert_span_ner import BertSpanNERModel  # noqa: E402

ID2LABEL = {0: "O", 1: "BioNE"}
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
    """Wraps task samples and tokenises on-the-fly via mrc_chunk_document."""

    def __init__(
        self,
        samples: List[Dict],
        tokenizer: BertTokenizer,
        max_seq_length: int,
    ) -> None:
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        s = self.samples[idx]
        chunks = mrc_chunk_document(
            self.tokenizer,
            pickle.loads(s["text"]),
            pickle.loads(s["entity_spans"]),
            pickle.loads(s["mrc_query"]),
            self.max_seq_length,
        )
        return chunks[0] if chunks else _empty_chunk(self.max_seq_length)


def _empty_chunk(max_seq_length: int) -> Dict:
    """Return a zero-padded chunk for documents that tokenise to nothing."""
    z = torch.zeros(max_seq_length, dtype=torch.long)
    return {
        "input_ids": z, "attention_mask": z, "segment_ids": z,
        "start_ids": z, "end_ids": z,
        "input_len": torch.tensor(0, dtype=torch.long),
        "subjects": [],
    }


def collate_fn(batch: List[Dict]) -> Tuple[torch.Tensor, ...]:
    """Collate 1-D chunk tensors into a padded batch."""
    input_ids = torch.stack([b["input_ids"] for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])
    segment_ids = torch.stack([b["segment_ids"] for b in batch])
    start_ids = torch.stack([b["start_ids"] for b in batch])
    end_ids = torch.stack([b["end_ids"] for b in batch])
    input_lens = torch.tensor(
        [b["input_len"].item() for b in batch], dtype=torch.long
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
    tokenizer: BertTokenizer,
    max_seq_length: int,
    model: BertSpanNERModel,
    device: torch.device,
    desc: str = "Eval",
) -> Tuple[Dict[str, float], List[Dict]]:
    """Evaluate *model* on *samples*, returning (aggregate_metrics, per_doc_list).

    per_doc_list entries have integer tp/fp/fn for bootstrap CI.
    """
    metric = SpanEntityScore(ID2LABEL)
    total_loss = 0.0
    pbar = ProgressBar(n_total=len(samples), desc=desc)
    per_docs: List[Dict] = []

    model.eval()
    for step, sample in enumerate(samples):
        chunks = mrc_chunk_document(
            tokenizer,
            pickle.loads(sample["text"]),
            pickle.loads(sample["entity_spans"]),
            pickle.loads(sample["mrc_query"]),
            max_seq_length,
        )
        chunk = chunks[0] if chunks else _empty_chunk(max_seq_length)
        lens = chunk["input_len"].item()
        input_ids = chunk["input_ids"][:lens].unsqueeze(0).to(device)
        attention_mask = chunk["attention_mask"][:lens].unsqueeze(0).to(device)
        segment_ids = chunk["segment_ids"][:lens].unsqueeze(0).to(device)
        start_ids = chunk["start_ids"][:lens].unsqueeze(0).to(device)
        end_ids = chunk["end_ids"][:lens].unsqueeze(0).to(device)

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
        gold_subjects = chunk["subjects"]
        metric.update(true_subject=gold_subjects, pred_subject=pred_subjects)
        total_loss += outputs["loss"].item()
        pbar(step)

        pred_set = set(map(tuple, pred_subjects))
        gold_set = set(map(tuple, gold_subjects))
        tp = len(pred_set & gold_set)
        fp = len(pred_set - gold_set)
        fn = len(gold_set - pred_set)
        per_docs.append({
            "id": sample.get("patient_id", str(step)),
            "tp": tp,
            "fp": fp,
            "fn": fn,
        })

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
    return results, per_docs


# ── Optuna HPO ────────────────────────────────────────────────────────────────


def _objective(
    trial,
    train_samples: List[Dict],
    dev_samples: List[Dict],
    tokenizer: BertTokenizer,
    args,
    device_str: str,
) -> float:
    """Optuna objective: train a trial model, return val F1."""
    lr = trial.suggest_float("lr", 1e-5, 1e-4, log=True)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    patience = trial.suggest_int("patience", 1, 3)

    train_loader = DataLoader(
        MRCNERDataset(train_samples, tokenizer, args.max_seq_length),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=dict_collate_fn,
    )
    dev_loader = DataLoader(
        MRCNERDataset(dev_samples, tokenizer, args.max_seq_length),
        batch_size=batch_size,
        sampler=SequentialSampler(MRCNERDataset(dev_samples, tokenizer, args.max_seq_length)),
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

    results, _ = evaluate(
        dev_samples,
        tokenizer,
        args.max_seq_length,
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
    tokenizer: BertTokenizer,
    device_str: str,
) -> None:
    """Run Optuna study and patch *args* with the best hyperparameters."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    ts(f"Starting Optuna HPO: {args.n_trials} trials, {args.tune_epochs} epochs each")
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: _objective(
            trial, train_samples, dev_samples, tokenizer, args, device_str
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

    # ── Tokenizer ─────────────────────────────────────────────────────
    ts(f"Loading tokenizer from {args.model_path} …")
    tokenizer = BertTokenizer.from_pretrained(args.model_path)

    # ── Optuna HPO ────────────────────────────────────────────────────
    if args.tune:
        tune(args, train_samples, dev_samples, tokenizer, device_str)

    # ── Dry-run ───────────────────────────────────────────────────────
    if args.dry_run:
        ts("Dry-run: processing first sample …")
        s = train_samples[0]
        chunks = mrc_chunk_document(
            tokenizer, pickle.loads(s["text"]),
            pickle.loads(s["entity_spans"]), pickle.loads(s["mrc_query"]),
            args.max_seq_length,
        )
        chunk = chunks[0] if chunks else _empty_chunk(args.max_seq_length)
        ts(f"  input_ids shape : {tuple(chunk['input_ids'].shape)}")
        ts(f"  input_len       : {chunk['input_len'].item()}")
        ts(f"  subjects        : {chunk['subjects']}")
        ts("Dry-run complete.")
        return

    # ── DataLoaders ───────────────────────────────────────────────────
    train_dataset = MRCNERDataset(train_samples, tokenizer, args.max_seq_length)
    dev_dataset = MRCNERDataset(dev_samples, tokenizer, args.max_seq_length)

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
    test_results, per_docs = evaluate(test_samples, tokenizer, args.max_seq_length, model, device, desc="Test")

    results_path = args.output_dir / "test_results.json"
    with open(results_path, "w") as fh:
        json.dump(test_results, fh, indent=2)
    ts(f"Test results saved to {results_path}")
    ts(
        f"Test F1={test_results['f1']:.4f}"
        f"  P={test_results['precision']:.4f}"
        f"  R={test_results['recall']:.4f}"
    )

    audit_path = args.audit_json or args.output_dir / "eval_biobert_mrc.json"
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
