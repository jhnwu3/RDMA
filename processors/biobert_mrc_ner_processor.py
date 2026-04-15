"""
PyHealth SampleProcessor for BioBERT-MRC NER.

Converts a raw sample dict (as produced by ``BioBERTMRCTask``) into the
padded integer tensors expected by ``BertSpanForNer``.

The full note text is tokenised once, then split into overlapping windows
at word boundaries so that no word's subword tokens are ever split across
chunks.  Each chunk is formatted as:

    [CLS] chunk_subwords [SEP] query_subwords [SEP] <pad>…
      segment_id=0         0      segment_id=1           0

and the chunks are stacked into ``[num_chunks, max_seq_length]`` tensors,
enabling all chunks of one note to be forwarded through the model in a
single batched GPU call.

Label mapping (binary):
    0 → "O"
    1 → "BioNE"
"""

import pickle
from typing import Any, Dict, List, Optional, Tuple

import torch
from pyhealth.processors.base_processor import SampleProcessor
from transformers import BertTokenizer


# Fixed binary label list, matching BnerProcessor in BioBERT-MRC
LABELS = ["O", "BioNE"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for i, l in enumerate(LABELS)}

_BIONE_ID = LABEL2ID["BioNE"]


def _word_offsets(text: str) -> Tuple[List[str], List[int]]:
    """Split *text* on whitespace and return (words, char_starts).

    Preserves exact character positions so that char-level entity spans can
    be mapped back to word indices.
    """
    words: List[str] = []
    starts: List[int] = []
    pos = 0
    n = len(text)
    while pos < n:
        # skip whitespace
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


def _char_span_to_subword_span(
    char_start: int,
    char_end: int,
    word_char_starts: List[int],
    words: List[str],
    subtoken_word_starts: List[int],
    word_subtokens: List[List[str]],
) -> Optional[Tuple[int, int]]:
    """Map a character span to a (sub_start, sub_end) subword index pair.

    Both indices are 0-based relative to the flat text subword token list
    (not counting [CLS]).  Returns None if the span cannot be aligned.
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
    sub_end = (
        subtoken_word_starts[end_word] + len(word_subtokens[end_word]) - 1
    )
    return sub_start, sub_end


class BioBERTMRCNERProcessor(SampleProcessor):
    """Convert a ``BioBERTMRCTask`` sample dict into BioBERT-MRC tensors.

    The full note text is tokenised once and split into overlapping chunks at
    word boundaries.  Each chunk is assembled into a complete BERT sequence and
    all chunks are stacked into ``[num_chunks, max_seq_length]`` tensors.

    Args:
        model_name_or_path: Path to a BioBERT checkpoint directory
            containing ``vocab.txt``.  Passed to
            ``BertTokenizer.from_pretrained``.
        max_seq_length: Maximum total subword sequence length including all
            special tokens.
        stride_tokens: Overlap between consecutive chunks, measured in subword
            tokens.  The next chunk starts ``stride_tokens`` subtokens before
            the end of the current chunk (at the nearest word boundary).
            Set to 0 to disable overlap.
    """

    def __init__(
        self,
        model_name_or_path: str,
        max_seq_length: int = 256,
        stride_tokens: int = 64,
    ) -> None:
        self.max_seq_length = max_seq_length
        self.stride_tokens = stride_tokens
        self.tokenizer: BertTokenizer = BertTokenizer.from_pretrained(
            model_name_or_path, do_lower_case=False
        )

    # ------------------------------------------------------------------
    # SampleProcessor interface
    # ------------------------------------------------------------------

    def process(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Tokenise *sample*, chunk the text, and build stacked MRC tensors.

        Args:
            sample: Dict produced by ``BioBERTMRCTask.__call__``.  Values
                for ``text``, ``entity_spans``, ``entity_type``, and
                ``mrc_query`` are pickled bytes.

        Returns:
            Dict with keys:
                ``input_ids``         – LongTensor [num_chunks, max_seq_length]
                ``attention_mask``    – LongTensor [num_chunks, max_seq_length]
                ``segment_ids``       – LongTensor [num_chunks, max_seq_length]
                ``start_ids``         – LongTensor [num_chunks, max_seq_length]
                ``end_ids``           – LongTensor [num_chunks, max_seq_length]
                ``input_len``         – List[int] of length num_chunks
                ``subjects``          – List[List[(label_id, sub_start,
                                        sub_end)]] gold spans per chunk
                                        (0-based within chunk text tokens)
                ``chunk_word_ranges`` – List[(start_word, end_word)] per
                                        chunk (end_word is exclusive)
                ``words``             – List[str] full whitespace-split
                                        word list
        """
        text: str = pickle.loads(sample["text"])
        entity_spans: List[Tuple[int, int]] = pickle.loads(
            sample["entity_spans"]
        )
        query: str = pickle.loads(sample["mrc_query"])

        # ── 1. Whitespace-split text, track char offsets ──────────────
        words, word_char_starts = _word_offsets(text)

        # ── 2. Subword-tokenise each word ─────────────────────────────
        word_subtokens: List[List[str]] = []
        for word in words:
            subs = self.tokenizer.tokenize(word)
            word_subtokens.append(subs if subs else [self.tokenizer.unk_token])

        # Cumulative subword start index per word (0-based in text tokens)
        subtoken_word_starts: List[int] = []
        cum = 0
        for subs in word_subtokens:
            subtoken_word_starts.append(cum)
            cum += len(subs)

        # ── 3. Tokenise query ─────────────────────────────────────────
        query_tokens: List[str] = self.tokenizer.tokenize(query) or []

        # Budget of text subword tokens per chunk
        # [CLS] + text_chunk + [SEP] + query + [SEP]
        max_text_tokens = self.max_seq_length - len(query_tokens) - 3

        # ── 4. Build entity label maps (char-span → subword indices) ──
        # We compute these once over the full word list so we can assign
        # labels per chunk below.
        entity_subword_spans: List[Tuple[int, int]] = []
        for char_start, char_end in entity_spans:
            result = _char_span_to_subword_span(
                char_start, char_end,
                word_char_starts, words,
                subtoken_word_starts, word_subtokens,
            )
            if result is not None:
                entity_subword_spans.append(result)

        # ── 5. Chunk text at word boundaries with stride ──────────────
        cls_tok = self.tokenizer.cls_token
        sep_tok = self.tokenizer.sep_token
        pad_id = self.tokenizer.pad_token_id or 0

        chunk_input_ids: List[List[int]] = []
        chunk_attention: List[List[int]] = []
        chunk_segments: List[List[int]] = []
        chunk_start_ids: List[List[int]] = []
        chunk_end_ids: List[List[int]] = []
        chunk_input_lens: List[int] = []
        chunk_word_ranges: List[Tuple[int, int]] = []
        chunk_subjects: List[List[Tuple[int, int, int]]] = []

        word_i = 0
        n_words = len(words)
        while word_i < n_words:
            # Greedily pack whole words into this chunk
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

            chunk_word_ranges.append((start_word, end_word))

            # Offset of this chunk's first subtoken in the full subtoken list
            chunk_sub_offset = subtoken_word_starts[start_word]

            # Build BERT sequence for this chunk
            tokens = (
                [cls_tok] + chunk_subs + [sep_tok] + query_tokens + [sep_tok]
            )
            seg_ids = (
                [0] * (len(chunk_subs) + 2)
                + [1] * (len(query_tokens) + 1)
            )
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            attn_mask = [1] * len(input_ids)
            input_len = len(input_ids)

            # Build start_ids / end_ids and subjects for this chunk
            start_ids = [0] * len(input_ids)
            end_ids = [0] * len(input_ids)
            subjects: List[Tuple[int, int, int]] = []
            # exclusive global subtoken index of chunk end
            chunk_sub_end = chunk_sub_offset + chunk_sub_count
            for global_sub_start, global_sub_end in entity_subword_spans:
                if (
                    global_sub_start >= chunk_sub_end
                    or global_sub_end < chunk_sub_offset
                ):
                    continue
                local_start = global_sub_start - chunk_sub_offset
                local_end = global_sub_end - chunk_sub_offset
                # Clamp to the chunk's text region
                if local_start < 0 or local_start >= chunk_sub_count:
                    continue
                if local_end < 0 or local_end >= chunk_sub_count:
                    continue
                # +1 for [CLS]
                start_ids[local_start + 1] = _BIONE_ID
                end_ids[local_end + 1] = _BIONE_ID
                subjects.append((_BIONE_ID, local_start, local_end))

            # Pad to max_seq_length
            pad_len = self.max_seq_length - len(input_ids)
            input_ids += [pad_id] * pad_len
            attn_mask += [0] * pad_len
            seg_ids += [0] * pad_len
            start_ids += [0] * pad_len
            end_ids += [0] * pad_len

            chunk_input_ids.append(input_ids)
            chunk_attention.append(attn_mask)
            chunk_segments.append(seg_ids)
            chunk_start_ids.append(start_ids)
            chunk_end_ids.append(end_ids)
            chunk_input_lens.append(input_len)
            chunk_subjects.append(subjects)

            if end_word >= n_words:
                break

            # Stride back by stride_tokens subwords (at word boundaries)
            rewind = 0
            j = end_word - 1
            while (
                j > start_word
                and rewind + len(word_subtokens[j]) <= self.stride_tokens
            ):
                rewind += len(word_subtokens[j])
                j -= 1
            word_i = j + 1
            # Guard: always make forward progress
            if word_i <= start_word:
                word_i = start_word + 1

        return {
            "input_ids": torch.tensor(
                chunk_input_ids, dtype=torch.long
            ),
            "attention_mask": torch.tensor(
                chunk_attention, dtype=torch.long
            ),
            "segment_ids": torch.tensor(
                chunk_segments, dtype=torch.long
            ),
            "start_ids": torch.tensor(
                chunk_start_ids, dtype=torch.long
            ),
            "end_ids": torch.tensor(
                chunk_end_ids, dtype=torch.long
            ),
            "input_len": chunk_input_lens,
            "subjects": chunk_subjects,
            "chunk_word_ranges": chunk_word_ranges,
            "words": words,
        }
