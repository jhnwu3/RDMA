"""
Shared helpers for RDMA task modules.

This module provides pure-Python utility functions used by multiple task
classes.  Nothing here is a PyHealth processor or framework hook — these
are plain functions that task ``__call__`` implementations call directly.

Public API
----------
word_offsets(text)
    Whitespace-split *text* and return (words, char_starts).  Used by all
    NER and MRC tasks for char-offset-aligned tokenisation.

char_span_to_subword_span(...)
    Map a character-level entity span to (sub_start, sub_end) subword
    indices.  Used by ``mrc_chunk_document``.

mrc_chunk_document(tokenizer, text, entity_spans, query,
                   max_seq_length, stride_tokens)
    Tokenise *text*, split into overlapping chunks at word boundaries, and
    return a list of per-chunk sample dicts with flat 1-D ``LongTensor``
    fields.  All three MRC task ``__call__`` implementations delegate to
    this function so the chunking logic lives in exactly one place.
"""

from typing import Dict, List, Optional, Tuple

import torch
from transformers import BertTokenizer


# ── Text helpers ──────────────────────────────────────────────────────────────


def word_offsets(text: str) -> Tuple[List[str], List[int]]:
    """Whitespace-split *text* and return ``(words, char_starts)``.

    Preserves exact character positions so that char-level entity spans can
    be mapped back to word indices.

    Args:
        text: Raw document string.

    Returns:
        words:       List of whitespace-delimited tokens.
        char_starts: Character start index (into *text*) for each word.
    """
    words: List[str] = []
    starts: List[int] = []
    pos = 0
    n = len(text)
    while pos < n:
        while pos < n and text[pos].isspace():
            pos += 1
        if pos >= n:
            break
        word_start = pos
        while pos < n and not text[pos].isspace():
            pos += 1
        words.append(text[word_start:pos])
        starts.append(word_start)
    return words, starts


def char_span_to_subword_span(
    char_start: int,
    char_end: int,
    word_char_starts: List[int],
    words: List[str],
    subtoken_word_starts: List[int],
    word_subtokens: List[List[str]],
) -> Optional[Tuple[int, int]]:
    """Map a character span to a ``(sub_start, sub_end)`` subword index pair.

    Both indices are 0-based relative to the flat text subword token list
    (not counting ``[CLS]``).  Returns ``None`` if the span cannot be
    aligned to any word in the token list.

    Args:
        char_start:          Inclusive character start of the entity span.
        char_end:            Exclusive character end of the entity span.
        word_char_starts:    Character start per word (from :func:`word_offsets`).
        words:               Whitespace-split word list.
        subtoken_word_starts: Cumulative subword start index per word (0-based
                             in the flat text subtoken list, not counting
                             ``[CLS]``).
        word_subtokens:      Per-word list of subword token strings.

    Returns:
        ``(sub_start, sub_end)`` tuple, or ``None`` if alignment fails.
    """
    start_word = end_word = None
    for wi, (ws, word) in enumerate(zip(word_char_starts, words)):
        we = ws + len(word)
        if start_word is None and ws <= char_start < we:
            start_word = wi
        if ws <= char_end - 1 < we:
            end_word = wi
    if start_word is None or end_word is None:
        return None
    sub_start = subtoken_word_starts[start_word]
    sub_end = subtoken_word_starts[end_word] + len(word_subtokens[end_word]) - 1
    return sub_start, sub_end


# ── MRC chunking ──────────────────────────────────────────────────────────────

# Binary label used by BertSpanForNer (matches BioBERT-MRC repo convention).
_LABELS = ["O", "BioNE"]
_LABEL2ID = {l: i for i, l in enumerate(_LABELS)}
_BIONE_ID = _LABEL2ID["BioNE"]


def mrc_chunk_document(
    tokenizer: BertTokenizer,
    text: str,
    entity_spans: List[Tuple[int, int]],
    query: str,
    max_seq_length: int = 256,
    stride_tokens: int = 64,
) -> List[Dict]:
    """Tokenise *text* and split into overlapping chunks for BioBERT-MRC.

    Each chunk is formatted as::

        [CLS] chunk_subwords [SEP] query_subwords [SEP] <pad>…
          seg=0                 0      seg=1                 0

    and returned as a flat per-chunk dict with 1-D ``torch.LongTensor``
    values so that task ``__call__`` methods can append metadata and return
    the list directly.

    Args:
        tokenizer:       A ``BertTokenizer`` loaded from the BioBERT checkpoint.
        text:            Full document text string.
        entity_spans:    List of ``(char_start, char_end)`` gold entity spans.
                         Pass ``[]`` for inference-only tasks.
        query:           MRC query string (e.g. ``"Find phenotype mentions …"``).
        max_seq_length:  Maximum total subword sequence length including all
                         special tokens.
        stride_tokens:   Overlap between consecutive chunks, measured in subword
                         tokens at word boundaries.  Set to ``0`` to disable.

    Returns:
        List of per-chunk dicts.  Each dict contains:

        * ``input_ids``            – ``LongTensor [max_seq_length]``
        * ``attention_mask``       – ``LongTensor [max_seq_length]``
        * ``segment_ids``          – ``LongTensor [max_seq_length]``
        * ``start_ids``            – ``LongTensor [max_seq_length]``
        * ``end_ids``              – ``LongTensor [max_seq_length]``
        * ``input_len``            – 0-d scalar ``LongTensor``
        * ``chunk_sub_offset``     – 0-d scalar ``LongTensor``
        * ``subjects``             – ``List[Tuple[int, int, int]]``
          ``(label_id, local_sub_start, local_sub_end)`` per entity in chunk
        * ``words``                – ``List[str]`` (plain Python)
        * ``word_char_starts``     – ``List[int]`` (plain Python)
        * ``subtoken_word_starts`` – ``List[int]`` (plain Python)

        Returns an empty list if *text* is empty or tokenises to nothing.
    """
    if not text:
        return []

    # ── 1. Whitespace-split text, track char offsets ──────────────────
    words, word_char_starts = word_offsets(text)
    if not words:
        return []

    # ── 2. Subword-tokenise each word ─────────────────────────────────
    word_subtokens: List[List[str]] = []
    for w in words:
        subs = tokenizer.tokenize(w)
        word_subtokens.append(subs if subs else [tokenizer.unk_token])

    # Cumulative subword start index per word (0-based in flat text tokens,
    # not counting [CLS]).
    subtoken_word_starts: List[int] = []
    cum = 0
    for subs in word_subtokens:
        subtoken_word_starts.append(cum)
        cum += len(subs)

    # ── 3. Tokenise query ─────────────────────────────────────────────
    query_tokens: List[str] = tokenizer.tokenize(query) or []

    # Budget of text subword tokens per chunk:
    # [CLS] + text_chunk + [SEP] + query + [SEP]
    max_text_tokens = max_seq_length - len(query_tokens) - 3

    # ── 4. Map entity char-spans → subword index pairs ────────────────
    entity_subword_spans: List[Tuple[int, int]] = []
    for char_start, char_end in entity_spans:
        result = char_span_to_subword_span(
            char_start,
            char_end,
            word_char_starts,
            words,
            subtoken_word_starts,
            word_subtokens,
        )
        if result is not None:
            entity_subword_spans.append(result)

    # ── 5. Chunk at word boundaries with stride ───────────────────────
    cls_tok = tokenizer.cls_token
    sep_tok = tokenizer.sep_token
    pad_id = tokenizer.pad_token_id or 0

    chunks: List[Dict] = []
    word_i = 0
    n_words = len(words)

    while word_i < n_words:
        # Greedily pack whole words into this chunk.
        chunk_subs: List[str] = []
        chunk_sub_count = 0
        start_word = word_i

        while word_i < n_words:
            subs = word_subtokens[word_i]
            if chunk_sub_count + len(subs) > max_text_tokens:
                break
            chunk_subs.extend(subs)
            chunk_sub_count += len(subs)
            word_i += 1
        end_word = word_i  # exclusive

        # Global subword offset of this chunk's first text token.
        chunk_sub_offset = subtoken_word_starts[start_word]

        # Build the BERT sequence: [CLS] chunk [SEP] query [SEP]
        tokens = [cls_tok] + chunk_subs + [sep_tok] + query_tokens + [sep_tok]
        seg_ids = [0] * (len(chunk_subs) + 2) + [1] * (len(query_tokens) + 1)
        input_ids_list = tokenizer.convert_tokens_to_ids(tokens)
        attn_mask_list = [1] * len(input_ids_list)
        input_len = len(input_ids_list)

        # Build start_ids / end_ids and subjects for this chunk.
        start_ids_list = [0] * len(input_ids_list)
        end_ids_list = [0] * len(input_ids_list)
        subjects: List[Tuple[int, int, int]] = []

        chunk_sub_end = chunk_sub_offset + chunk_sub_count  # exclusive
        for global_sub_start, global_sub_end in entity_subword_spans:
            if global_sub_start >= chunk_sub_end or global_sub_end < chunk_sub_offset:
                continue
            local_start = global_sub_start - chunk_sub_offset
            local_end = global_sub_end - chunk_sub_offset
            # Clamp to the chunk's text region (skip cross-boundary spans).
            if not (0 <= local_start < chunk_sub_count):
                continue
            if not (0 <= local_end < chunk_sub_count):
                continue
            start_ids_list[local_start + 1] = _BIONE_ID  # +1 for [CLS]
            end_ids_list[local_end + 1] = _BIONE_ID
            subjects.append((_BIONE_ID, local_start, local_end))

        # Pad to max_seq_length.
        pad_len = max_seq_length - len(input_ids_list)
        input_ids_list += [pad_id] * pad_len
        attn_mask_list += [0] * pad_len
        seg_ids += [0] * pad_len
        start_ids_list += [0] * pad_len
        end_ids_list += [0] * pad_len

        chunks.append(
            {
                "input_ids": torch.tensor(input_ids_list, dtype=torch.long),
                "attention_mask": torch.tensor(attn_mask_list, dtype=torch.long),
                "segment_ids": torch.tensor(seg_ids, dtype=torch.long),
                "start_ids": torch.tensor(start_ids_list, dtype=torch.long),
                "end_ids": torch.tensor(end_ids_list, dtype=torch.long),
                "input_len": torch.tensor(input_len, dtype=torch.long),
                "chunk_sub_offset": torch.tensor(chunk_sub_offset, dtype=torch.long),
                "subjects": subjects,
                "words": words,
                "word_char_starts": word_char_starts,
                "subtoken_word_starts": subtoken_word_starts,
            }
        )

        if end_word >= n_words:
            break

        # Stride back by stride_tokens subwords (at word boundaries).
        rewind = 0
        j = end_word - 1
        while j > start_word and rewind + len(word_subtokens[j]) <= stride_tokens:
            rewind += len(word_subtokens[j])
            j -= 1
        word_i = j + 1
        # Guard: always make forward progress.
        if word_i <= start_word:
            word_i = start_word + 1

    return chunks
