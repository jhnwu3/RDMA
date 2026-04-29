#!/usr/bin/env python3
"""
Bio_ClinicalBERT token-classification NER fine-tuning on BioLarkGSC.

Trains on the BioLarkGSC HPO phenotype corpus (228 biomedical abstracts,
80/10/10 random split) using the BioClinicalBERT model with a 3-label BIO
scheme: O, B-PHENOTYPE, I-PHENOTYPE.

Usage (from RDMA repo root)::

    python baselines/biolarkgsc/bioclinicalbert_ner.py \\
        --gpu_id 0 --num_epochs 10 --batch_size 16

    # Dry-run (no training):
    python baselines/biolarkgsc/bioclinicalbert_ner.py --dry_run
"""

import argparse
import json
import logging
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from tqdm import tqdm

# ── Path setup ───────────────────────────────────────────────────────────────
_RDMA_ROOT = Path("/home/johnwu3/projects/rare_disease/workspace/repos/RDMA")
_PYHEALTH_ROOT = Path("/home/johnwu3/projects/rare_disease/workspace/repos/PyHealth")
_RESULTS_DIR = Path("/home/johnwu3/projects/rare_disease/workspace/results")
_DEFAULT_EMBEDDINGS_FILE = str(
    _RDMA_ROOT / "data" / "vector_stores" / "G2GHPO_metadata_medembed.npy"
)

sys.path.insert(0, str(_PYHEALTH_ROOT))
sys.path.insert(0, str(_RDMA_ROOT))

from datasets.biolarkgsc import BioLarkGSCDataset  # noqa: E402
from tasks.bioclinicalbert_ner_biolarkgsc import (  # noqa: E402
    BioClinicalBERTNERBioLarkGSCTask,
)
from models.bioclinicalbert_ner import BioClinicalBERTNERModel  # noqa: E402
from metrics.ner_metrics import SpanEntityScore  # noqa: E402
from pyhealth.datasets import split_by_patient  # noqa: E402
from pyhealth.trainer import Trainer  # noqa: E402
from rdma.hpo.embedding_fuzzy_matcher import EmbeddingFuzzyMatcher  # noqa: E402

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def ts(msg: str) -> None:
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S} - {msg}", flush=True)


# ── DataLoader helpers ───────────────────────────────────────────────────────


class NERDataset(Dataset):
    def __init__(self, samples) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        return self.samples[idx]


def collate_fn(batch: List[Dict]) -> Dict:
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
        "word_ids": torch.stack([b["word_ids"] for b in batch]),
        "patient_id": [b.get("patient_id") for b in batch],
        "words": [pickle.loads(b["words"]) if b.get("words") else [] for b in batch],
        "gold_hpo_ids": [pickle.loads(b["gold_hpo_ids"]) if b.get("gold_hpo_ids") else [] for b in batch],
        "gold_annotations": [pickle.loads(b["gold_annotations"]) if b.get("gold_annotations") else [] for b in batch],
        "word_char_starts": [pickle.loads(b["word_char_starts"]) if b.get("word_char_starts") else [] for b in batch],
    }


# ── Span extraction helpers ──────────────────────────────────────────────────

_TASK = BioClinicalBERTNERBioLarkGSCTask


# ── Evaluation ───────────────────────────────────────────────────────────────


def _spans_to_strings(spans: List[Tuple[int, int, int]], words: List[str]) -> List[str]:
    """Convert (label_id, start_word, end_word) spans to surface strings."""
    return [" ".join(words[ws : we + 1]) for _, ws, we in spans]


def evaluate(
    loader: DataLoader,
    model: BioClinicalBERTNERModel,
    device: torch.device,
    desc: str = "Eval",
    predictions_path: Optional[Path] = None,
    code_matcher: Optional[EmbeddingFuzzyMatcher] = None,
) -> Dict[str, float]:
    """Evaluate *model* using ``SpanEntityScore``, returning P/R/F1/loss.

    If *predictions_path* is given, writes a JSONL file with one record per
    document::

        {"id": doc_id, "predicted": ["HP:XXXXXXX", ...],
         "ground_truth": ["HP:XXXXXXX", ...]}
    """
    b_id2label = {_TASK.LABEL2ID[f"B-{et}"]: et for et in _TASK.ENTITY_TYPES}
    metric = SpanEntityScore(b_id2label)
    total_loss = 0.0
    n_batches = 0
    pred_file = open(predictions_path, "w") if predictions_path else None

    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc, leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            word_ids = batch["word_ids"]
            n_items = input_ids.size(0)
            patient_ids = batch.get("patient_id", [None] * n_items)
            words_batch = batch.get("words", [[] for _ in range(n_items)])
            gold_hpo_batch = batch.get("gold_hpo_ids", [[] for _ in range(n_items)])

            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            if "loss" in out:
                total_loss += out["loss"].item()
            n_batches += 1

            pred_ids = torch.argmax(out["logits"], dim=-1).cpu()
            gold_ids = labels.cpu()

            for i in range(input_ids.size(0)):
                wids = word_ids[i]
                preds = pred_ids[i]
                golds = gold_ids[i]
                words = words_batch[i]
                gold_hpo_ids = gold_hpo_batch[i]

                word_preds = BioClinicalBERTNERModel.collapse_to_word_preds(wids, preds)
                word_golds = BioClinicalBERTNERModel.collapse_to_word_preds(wids, golds)

                pred_spans = BioClinicalBERTNERModel.bio_to_spans(word_preds, _TASK.ID2LABEL, _TASK.LABEL2ID)
                gold_spans = BioClinicalBERTNERModel.bio_to_spans(word_golds, _TASK.ID2LABEL, _TASK.LABEL2ID)
                metric.update(true_subject=gold_spans, pred_subject=pred_spans)

                if pred_file is not None and words:
                    span_texts = _spans_to_strings(pred_spans, words)
                    ts(f"[DBG] doc={patient_ids[i]}  pred_spans={len(pred_spans)}  span_texts={span_texts[:5]}")
                    predicted_hpo_ids = []
                    if code_matcher is not None and span_texts:
                        for result in code_matcher.match([{"entity": s} for s in span_texts]):
                            hp_id = result.get("hp_id", "")
                            if hp_id:
                                predicted_hpo_ids.append(hp_id)
                    pred_file.write(
                        json.dumps(
                            {
                                "id": patient_ids[i],
                                "predicted": predicted_hpo_ids,
                                "ground_truth": gold_hpo_ids,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

    if pred_file is not None:
        pred_file.close()

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
    return results


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bio_ClinicalBERT NER fine-tuning on BioLarkGSC"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="emilyalsentzer/Bio_ClinicalBERT",
    )
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument(
        "--patience",
        type=int,
        default=3,
        help="Early-stopping patience (epochs without val-loss improvement)",
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
        default=_RESULTS_DIR / "biolarkgsc" / "bioclinicalbert_ner",
    )
    parser.add_argument(
        "--embeddings_file",
        type=Path,
        default=Path(_DEFAULT_EMBEDDINGS_FILE),
        help="Path to HPO .npy embeddings file",
    )
    parser.add_argument(
        "--retriever",
        type=str,
        choices=["fastembed", "sentence_transformer", "medcpt"],
        default="sentence_transformer",
        help="Retriever type (default: %(default)s)",
    )
    parser.add_argument(
        "--retriever_model",
        type=str,
        default="BAAI/bge-small-en-v1.5",
        help="Retriever model name (default: %(default)s)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=20,
        help="Top-k retrieved candidates per span (default: %(default)s)",
    )
    parser.add_argument(
        "--similarity_threshold",
        type=float,
        default=0.9,
        help="Fuzzy match threshold (default: %(default)s)",
    )
    parser.add_argument(
        "--predictions_path",
        type=Path,
        default=(_RESULTS_DIR / "biolarkgsc" / "bioclinicalbert_ner_predictions.jsonl"),
        help="Output JSONL path for HPO-ID predictions",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Process one sample and exit (no training)",
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Skip training; load checkpoint from --output_dir/best_hf and run test evaluation only",
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
    ts("Loading BioLarkGSCDataset …")
    dataset = BioLarkGSCDataset()

    ts(
        "Running set_task with BioClinicalBERTNERBioLarkGSCTask"
        f" (model={args.model_path}) …"
    )
    task = BioClinicalBERTNERBioLarkGSCTask(
        model_name_or_path=args.model_path,
        max_seq_length=args.max_seq_length,
    )
    sample_dataset = dataset.set_task(task)
    ts(f"  Total samples: {len(sample_dataset)}")

    ts("Splitting by patient (80/10/10, seed=42) …")
    train_split, dev_split, test_split = split_by_patient(
        sample_dataset, ratios=[0.8, 0.1, 0.1], seed=42
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
        n_entities = (sample["labels"] >= 0).sum().item() - (
            sample["labels"] == 0
        ).sum().item()
        ts(f"  non-O label tokens: {n_entities}")
        ts("Dry-run complete.")
        return

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
    hf_ckpt_dir = args.output_dir / "best_hf"
    if args.eval_only:
        ts(f"eval_only: loading checkpoint from {hf_ckpt_dir} …")
        model = BioClinicalBERTNERModel.load_from_checkpoint(str(hf_ckpt_dir))
    else:
        ts(f"Loading BioClinicalBERTNERModel from {args.model_path} …")
        model = BioClinicalBERTNERModel(
            dataset=None,
            model_name_or_path=args.model_path,
            num_labels=_TASK.NUM_LABELS,
            id2label=_TASK.ID2LABEL,
            label2id=_TASK.LABEL2ID,
        )

        # ── PyHealth Trainer ──────────────────────────────────────────────
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

        ts(f"Saving HuggingFace checkpoint to {hf_ckpt_dir} …")
        model.save_pretrained(str(hf_ckpt_dir))
        torch.save(args, hf_ckpt_dir / "training_args.bin")

    model.to(device)

    # ── Final test evaluation ─────────────────────────────────────────
    ts("Running test evaluation …")
    code_matcher = EmbeddingFuzzyMatcher(
        embeddings_file=str(args.embeddings_file),
        retriever=args.retriever,
        retriever_model=args.retriever_model,
        top_k=args.top_k,
        fuzzy_threshold=args.similarity_threshold,
        device=device_str,
    )

    predictions_path = args.predictions_path
    test_results = evaluate(
        test_loader,
        model,
        device,
        desc="Test",
        predictions_path=predictions_path,
        code_matcher=code_matcher,
    )
    ts(f"Per-document predictions saved to {predictions_path}")

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
