#!/usr/bin/env python3
"""
BioBERT-MRC fine-tuning baseline for BioLarkGSC phenotype NER.

Trains BertSpanForNer (span extraction via MRC) on the BioLarkGSC corpus
using the PyHealth BioLarkGSCDataset + BioBERTMRCBioLarkGSCTask +
BioBERTMRCBioLarkGSCTask pipeline.

Single entity type: PHENOTYPE.
Query: "Find phenotype mentions in the text ."

Usage (from RDMA repo root)::

    python baselines/biolarkgsc/biobert_mrc.py \\
        --gpu_id 0 --num_epochs 10 --batch_size 16

    # Dry-run (one batch, no training):
    python baselines/biolarkgsc/biobert_mrc.py --dry_run
"""

import argparse
import bisect
from collections import defaultdict
import json
import logging
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler

# ── Path setup ───────────────────────────────────────────────────────────────
_RDMA_ROOT = Path("/home/johnwu3/projects/rare_disease/workspace/repos/RDMA")
_BIOBERT_ROOT = Path("/home/johnwu3/projects/rare_disease/workspace/repos/BioBERT-MRC")
_RESULTS_DIR = Path("/home/johnwu3/projects/rare_disease/workspace/results")

sys.path.insert(0, str(_BIOBERT_ROOT))
sys.path.insert(0, str(_RDMA_ROOT))

from datasets.biolarkgsc import BioLarkGSCDataset  # noqa: E402
from tasks.biobert_mrc_biolarkgsc import (  # noqa: E402
    BioBERTMRCBioLarkGSCTask,
    ID2LABEL,
)
from models.biobert_span_ner import BertSpanNERModel  # noqa: E402
from pyhealth.datasets import split_by_patient  # noqa: E402

from callback.optimizater.adamw import AdamW  # noqa: E402
from callback.lr_scheduler import (  # noqa: E402
    get_linear_schedule_with_warmup,
)
from callback.progressbar import ProgressBar  # noqa: E402
from metrics.ner_metrics import SpanEntityScore  # noqa: E402
from rdma.hpo.embedding_fuzzy_matcher import EmbeddingFuzzyMatcher  # noqa: E402

_BIOBERT_MODEL = str(_BIOBERT_ROOT / "BioBERTv1.1_P")
_DEFAULT_EMBEDDINGS_FILE = str(
    _RDMA_ROOT / "data" / "vector_stores" / "G2GHPO_metadata_medembed.npy"
)

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def ts(msg: str) -> None:
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S} - {msg}", flush=True)



def _subtoken_to_word_idx(subtoken_idx: int, subtoken_word_starts: List[int]) -> int:
    if not subtoken_word_starts:
        return -1
    return bisect.bisect_right(subtoken_word_starts, subtoken_idx) - 1


def _span_to_char_range(
    local_start: int,
    local_end: int,
    chunk_sub_offset: int,
    subtoken_word_starts: List[int],
    word_char_starts: List[int],
    words: List[str],
) -> Tuple[int, int]:
    global_start = chunk_sub_offset + local_start
    global_end = chunk_sub_offset + local_end
    ws = _subtoken_to_word_idx(global_start, subtoken_word_starts)
    we = _subtoken_to_word_idx(global_end, subtoken_word_starts)
    if ws < 0 or we < 0 or ws >= len(word_char_starts) or we >= len(word_char_starts):
        return -1, -1
    char_start = word_char_starts[ws]
    char_end = word_char_starts[we] + len(words[we])
    return char_start, char_end


def _span_to_text(
    local_start: int,
    local_end: int,
    chunk_sub_offset: int,
    subtoken_word_starts: List[int],
    words: List[str],
) -> str:
    global_start = chunk_sub_offset + local_start
    global_end = chunk_sub_offset + local_end
    ws = _subtoken_to_word_idx(global_start, subtoken_word_starts)
    we = _subtoken_to_word_idx(global_end, subtoken_word_starts)
    if ws < 0 or we < 0 or ws >= len(words) or we >= len(words):
        return ""
    return " ".join(words[ws : we + 1])



# ── Dataset wrapper ──────────────────────────────────────────────────────────


class MRCNERDataset(Dataset):
    """Thin wrapper around a list of chunk-level sample dicts.

    Each sample is already a single chunk with fixed-size tensors,
    as emitted by BioBERTMRCBioLarkGSCTask.__call__.
    """

    def __init__(self, samples: List[Dict]) -> None:
        self.items: List[Dict] = list(samples)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict:
        return self.items[idx]


def collate_fn(batch: List[Dict]) -> Tuple[torch.Tensor, ...]:
    input_ids = torch.stack([b["input_ids"] for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])
    segment_ids = torch.stack([b["segment_ids"] for b in batch])
    start_ids = torch.stack([b["start_ids"] for b in batch])
    end_ids = torch.stack([b["end_ids"] for b in batch])
    input_lens = torch.tensor([b["input_len"] for b in batch], dtype=torch.long)
    max_len = input_lens.max().item()
    return (
        input_ids[:, :max_len],
        attention_mask[:, :max_len],
        segment_ids[:, :max_len],
        start_ids[:, :max_len],
        end_ids[:, :max_len],
        input_lens,
    )


# ── Evaluation ───────────────────────────────────────────────────────────────


def evaluate(
    samples: List[Dict],
    model: BertSpanNERModel,
    device: torch.device,
    desc: str = "Eval",
) -> Dict[str, float]:
    """Evaluate *model* on *samples* (chunk-level) and return P/R/F1."""
    metric = SpanEntityScore(ID2LABEL)
    total_loss = 0.0
    pbar = ProgressBar(n_total=len(samples), desc=desc)

    model.eval()
    for step, sample in enumerate(samples):
        lens = int(sample["input_len"])
        input_ids = sample["input_ids"][:lens].unsqueeze(0).to(device)
        attention_mask = sample["attention_mask"][:lens].unsqueeze(0).to(device)
        segment_ids = sample["segment_ids"][:lens].unsqueeze(0).to(device)
        start_ids = sample["start_ids"][:lens].unsqueeze(0).to(device)
        end_ids = sample["end_ids"][:lens].unsqueeze(0).to(device)
        subjects = pickle.loads(sample["subjects"])

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                segment_ids=segment_ids,
                start_ids=start_ids,
                end_ids=end_ids,
            )

        text_len = int(torch.sum(attention_mask.view(-1) - segment_ids.view(-1)).cpu())
        pred_subjects = BertSpanNERModel.predict(
            outputs["start_logits"], outputs["end_logits"], text_len
        )
        metric.update(
            true_subject=subjects,
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


def export_predictions(
    samples: List[Dict],
    model: BertSpanNERModel,
    device: torch.device,
    predictions_path: Path,
    code_matcher: EmbeddingFuzzyMatcher,
) -> None:
    """Write per-document HPO-ID predictions JSONL for BioLarkGSC eval.

    Samples are chunk-level; we group by patient_id to reconstruct
    per-document predictions across all chunks.
    """
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()

    doc_chunks: Dict[str, List[Dict]] = defaultdict(list)
    for sample in samples:
        doc_chunks[sample["patient_id"]].append(sample)

    with open(predictions_path, "w", encoding="utf-8") as pred_f:
        for doc_id, chunks in doc_chunks.items():
            # Per-document metadata is identical across all chunks of a doc.
            first = chunks[0]
            words = pickle.loads(first["words"])
            subtoken_word_starts = pickle.loads(first["subtoken_word_starts"])
            gold_annotations = pickle.loads(first["gold_annotations"])
            gold_hpo = [
                ann.get("hpo_id") for ann in gold_annotations if ann.get("hpo_id")
            ]

            predicted: List[str] = []
            for chunk in chunks:
                lens = int(chunk["input_len"])
                chunk_offset = int(chunk["chunk_sub_offset"])
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
                    torch.sum(attention_mask.view(-1) - segment_ids.view(-1)).cpu()
                )
                pred_subjects = BertSpanNERModel.predict(
                    outputs["start_logits"],
                    outputs["end_logits"],
                    text_len,
                )
                for _, local_start, local_end in pred_subjects:
                    span_text = _span_to_text(
                        local_start,
                        local_end,
                        chunk_offset,
                        subtoken_word_starts,
                        words,
                    )
                    if span_text:
                        results = code_matcher.match([{"entity": span_text}])
                        hp_id = results[0].get("hp_id", "") if results else ""
                        if hp_id:
                            predicted.append(hp_id)

            pred_f.write(
                json.dumps(
                    {
                        "id": doc_id,
                        "predicted": list({p for p in predicted if p}),
                        "ground_truth": gold_hpo,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


# ── Training loop ────────────────────────────────────────────────────────────


def train(
    args: argparse.Namespace,
    train_samples: List[Dict],
    dev_samples: List[Dict],
    model: BertSpanNERModel,
    device: torch.device,
    output_dir: Path,
) -> None:
    train_dataset = MRCNERDataset(train_samples)
    train_loader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=args.batch_size,
        collate_fn=collate_fn,
    )

    t_total = len(train_loader) // args.grad_accum * args.num_epochs
    warmup_steps = int(t_total * args.warmup_proportion)

    no_decay = ["bias", "LayerNorm.weight"]
    bert_params = list(model._bert_span.bert.named_parameters())
    start_params = list(model._bert_span.start_fc.named_parameters())
    end_params = list(model._bert_span.end_fc.named_parameters())

    def _group(params, lr):
        return [
            {
                "params": [p for n, p in params if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
                "lr": lr,
            },
            {
                "params": [p for n, p in params if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                "lr": lr,
            },
        ]

    optimizer_grouped_parameters = (
        _group(bert_params, args.lr)
        + _group(start_params, args.classifier_lr)
        + _group(end_params, args.classifier_lr)
    )

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=t_total,
    )

    ts(f"Training on {len(train_samples)} samples" f" | {len(dev_samples)} dev samples")
    ts(
        f"Epochs={args.num_epochs}  batch={args.batch_size}"
        f"  lr={args.lr}  classifier_lr={args.classifier_lr}"
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

        avg = epoch_loss / max(len(train_loader), 1)
        ts(f"  Epoch {epoch + 1} avg loss: {avg:.4f}")

        dev_results = evaluate(
            dev_samples,
            model,
            device,
            desc=f"Dev epoch {epoch + 1}",
        )
        if dev_results["f1"] > best_f1:
            best_f1 = dev_results["f1"]
            hf_ckpt_dir = output_dir / "best_hf"
            hf_ckpt_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(hf_ckpt_dir))
            torch.save(args, hf_ckpt_dir / "training_args.bin")
            ts(f"  New best F1={best_f1:.4f}" f" — checkpoint saved to {hf_ckpt_dir}")

        if "cuda" in str(device):
            torch.cuda.empty_cache()

    ts(f"Training complete.  Best dev F1: {best_f1:.4f}")


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="BioBERT-MRC fine-tuning on BioLarkGSC"
    )
    parser.add_argument("--model_path", type=str, default=_BIOBERT_MODEL)
    parser.add_argument("--max_seq_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-5,
        help="LR for BERT parameters",
    )
    parser.add_argument(
        "--classifier_lr",
        type=float,
        default=1e-3,
        help="LR for span classifier heads",
    )
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_proportion", type=float, default=0.1)
    parser.add_argument("--grad_accum", type=int, default=1)
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
        default=_RESULTS_DIR / "biolarkgsc" / "biobert_mrc",
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
        default=_RESULTS_DIR / "biolarkgsc" / "biobert_mrc_predictions.jsonl",
        help="Output JSONL path for HPO-ID predictions",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Process one batch and exit (no training)",
    )
    args = parser.parse_args()

    if args.condor:
        device = torch.device("cuda")
    elif args.gpu_id is not None and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu_id}")
    else:
        device = torch.device("cpu")
    ts(f"Device: {device}")

    # ── Dataset + task ────────────────────────────────────────────────
    ts("Loading BioLarkGSCDataset …")
    dataset = BioLarkGSCDataset()
    ts(f"Initialising BioBERTMRCBioLarkGSCTask (model={args.model_path}) …")
    task = BioBERTMRCBioLarkGSCTask(
        model_name_or_path=args.model_path,
        max_seq_length=args.max_seq_length,
    )
    sample_dataset = dataset.set_task(task)
    ts(f"  Total samples (chunks): {len(sample_dataset)}")

    ts("Splitting by patient (80/10/10, seed=42) …")
    train_split, dev_split, test_split = split_by_patient(
        sample_dataset, ratios=[0.8, 0.1, 0.1], seed=42
    )
    ts(
        f"  Train: {len(train_split)}"
        f"  Dev: {len(dev_split)}"
        f"  Test: {len(test_split)}"
    )

    # ── Dry-run ───────────────────────────────────────────────────────
    if args.dry_run:
        ts("Dry-run: inspecting first sample …")
        sample = list(train_split)[0]
        ts(f"  input_ids shape : {tuple(sample['input_ids'].shape)}")
        ts(f"  input_len       : {int(sample['input_len'])}")
        ts(f"  subjects        : {pickle.loads(sample['subjects'])}")
        ts("Dry-run complete.")
        return

    # ── Model ─────────────────────────────────────────────────────────
    ts(f"Loading BertSpanNERModel from {args.model_path} …")
    model = BertSpanNERModel(dataset=None, model_name_or_path=args.model_path)
    model.to(device)

    # ── Train ─────────────────────────────────────────────────────────
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train(
        args,
        list(train_split),
        list(dev_split),
        model,
        device,
        args.output_dir,
    )

    # ── Final test evaluation ──────────────────────────────────────────
    ts("Loading best checkpoint for test evaluation …")
    best_model = BertSpanNERModel.load_from_checkpoint(str(args.output_dir / "best_hf"))
    best_model.to(device)

    test_results = evaluate(list(test_split), best_model, device, desc="Test")

    code_matcher = EmbeddingFuzzyMatcher(
        embeddings_file=str(args.embeddings_file),
        retriever=args.retriever,
        retriever_model=args.retriever_model,
        top_k=args.top_k,
        fuzzy_threshold=args.similarity_threshold,
        device=str(device),
    )

    export_predictions(
        list(test_split),
        best_model,
        device,
        args.predictions_path,
        code_matcher,
    )
    ts(f"Per-document predictions saved to {args.predictions_path}")

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
