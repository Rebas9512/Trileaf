"""
Text chunking utilities for the writing optimizer pipeline.

Strategy:
  1. Split by natural paragraphs (double newline).
  2. If no paragraphs exist, split by sentences grouping up to MAX_CHUNK_CHARS.
  3. Oversized paragraphs are sub-split at sentence boundaries.
  4. split_finer() provides progressively smaller chunks for AI-detection retries.
"""

from __future__ import annotations

import re
import unicodedata
from typing import List, Tuple

def clean_text(text: str) -> str:
    """
    Lightweight sanitization applied before chunking.

    Removes encoding artifacts and pathological whitespace that would confuse
    downstream models.  Normal prose — including smart quotes, em dashes,
    emojis, and non-ASCII scripts — is left completely unchanged.

    Steps (in order):
    1. Unicode NFC normalization (fix combining-character sequences).
    2. Strip zero-width / invisible format characters (BOM, ZWSP, ZWJ, …).
    3. Remove Unicode replacement character U+FFFD (garbled-byte marker).
    4. Remove ASCII control characters (keep tab, newline, carriage-return).
    5. Normalize line endings to \\n.
    6. Strip trailing whitespace from every line.
    7. Collapse 3+ consecutive blank lines to 2 (one paragraph break).
    8. Collapse runs of 2+ horizontal whitespace chars to a single space.
    9. Cap 5+ consecutive identical punctuation chars to 3
       (e.g. "!!!!!" → "!!!", "....." → "...").
    """
    # 1. NFC normalization
    text = unicodedata.normalize("NFC", text)

    # 2. Zero-width / invisible Unicode format chars
    #    U+FEFF BOM · U+200B ZWSP · U+200C ZWNJ · U+200D ZWJ
    #    U+00AD soft-hyphen · U+2028 line-sep · U+2029 para-sep
    text = re.sub(r"[\ufeff\u200b\u200c\u200d\u00ad\u2028\u2029]", "", text)

    # 3. Encoding-error marker
    text = text.replace("\ufffd", "")

    # 4. ASCII control chars (keep \x09=tab, \x0a=LF, \x0d=CR)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # 5. Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 6. Trailing whitespace per line
    text = "\n".join(line.rstrip() for line in text.split("\n"))

    # 7. Promote standalone line breaks to paragraph breaks.
    #    In prose input (e.g. from a web textarea), a single Enter marks a
    #    new paragraph.  Convert any \n that is not part of a \n\n sequence
    #    so each line is treated as its own paragraph by the chunker.
    text = re.sub(r"(?<!\n)\n(?!\n)", "\n\n", text)

    # 8. Collapse 3+ blank lines → 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 9. Collapse multiple horizontal whitespace to single space
    text = re.sub(r"[ \t]{2,}", " ", text)

    # 10. Cap egregiously repeated identical punctuation (5+ → 3)
    text = re.sub(r"([!?.,;:\-~])\1{4,}", lambda m: m.group(1) * 3, text)

    return text.strip()


# Sentence-ending boundaries (English + Chinese punctuation).
# The no-whitespace alternative only triggers on a capital letter or CJK char
# (not on '"') so that closing dialogue quotes like tonight?" are not split
# off from the preceding sentence, which would create spurious quote fragments.
_SENT_BOUNDARY = re.compile(
    r"(?<=[.!?。！？])\s+"
    r"|(?<=[.!?。！？])(?=[A-Z\u4e00-\u9fff])"
)


def split_text(
    text: str,
    max_chunk_chars: int = 200,
    merge_short_paragraphs: bool = False,
) -> List[str]:
    """
    Main entry point: split *text* into chunks suitable for the pipeline.

    Rules
    -----
    - If text has natural paragraphs (double newline), use them as chunk
      boundaries.  Oversized paragraphs are further split by sentences.
    - If text has no clear paragraph structure, split by sentences grouping
      up to *max_chunk_chars* characters; prefer cuts at punctuation marks.
    - Text shorter than *max_chunk_chars* is returned as a single chunk.
    - When *merge_short_paragraphs* is True (long-text mode), consecutive
      short paragraphs are merged into a single chunk up to *max_chunk_chars*
      so that chunk count stays in a 3-5 range for a ~2 000-char document.
    """
    text = text.strip()
    if not text:
        return []

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    if len(paragraphs) <= 1:
        return _sentence_group_split(text, max_chunk_chars)

    if merge_short_paragraphs:
        return _merge_paragraph_split(paragraphs, max_chunk_chars)

    result: List[str] = []
    for para in paragraphs:
        if len(para) > max_chunk_chars:
            result.extend(_sentence_group_split(para, max_chunk_chars))
        else:
            result.append(para)
    return result


def split_sentences(text: str) -> List[str]:
    """Split text into individual sentences (English + Chinese aware)."""
    parts = _SENT_BOUNDARY.split(text.strip())
    out = [p.strip() for p in parts if p.strip()]
    return out if out else [text.strip()]


def split_finer(text: str, level: int) -> List[str]:
    """
    Return progressively finer sub-chunks for the AI-detection retry loop.

    level 1 → 2-sentence groups
    level 2 → individual sentences
    level 3 → individual sentences (same granularity; orchestrator uses a
               more aggressive Qwen prompt at this level)
    """
    sents = split_sentences(text)
    if len(sents) <= 1 or level <= 0:
        return [text]

    if level == 1:
        groups: List[str] = []
        for i in range(0, len(sents), 2):
            groups.append(" ".join(sents[i : i + 2]))
        return groups

    # level >= 2 → sentence-level
    return sents


def split_text_with_para_idx(
    text: str,
    max_chunk_chars: int = 200,
    merge_short_paragraphs: bool = False,
) -> Tuple[List[str], List[int]]:
    """
    Like split_text but also returns the original paragraph index for each chunk.

    ``para_idx[i]`` is the 0-based index of the original paragraph from which
    ``chunks[i]`` was derived.  Chunks that share the same ``para_idx`` came
    from the same original paragraph and should be joined with a space (not a
    blank line) when reconstructing the output.

    Rules
    -----
    - Texts with no paragraph breaks: all chunks share ``para_idx = 0``.
    - Short-text mode (merge_short_paragraphs=False): each paragraph generates
      one or more chunks, all tagged with that paragraph's index.
    - Long-text mode (merge_short_paragraphs=True): merged chunks are tagged
      with the first original paragraph that entered the merge group.
    """
    text = text.strip()
    if not text:
        return [], []

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    # No paragraph structure — all chunks belong to paragraph 0
    if len(paragraphs) <= 1:
        chunks = _sentence_group_split(text, max_chunk_chars)
        return chunks, [0] * len(chunks)

    if merge_short_paragraphs:
        return _merge_paragraph_split_annotated(paragraphs, max_chunk_chars)

    chunks: List[str] = []
    para_indices: List[int] = []
    for para_idx, para in enumerate(paragraphs):
        if len(para) > max_chunk_chars:
            sub = _sentence_group_split(para, max_chunk_chars)
            chunks.extend(sub)
            para_indices.extend([para_idx] * len(sub))
        else:
            chunks.append(para)
            para_indices.append(para_idx)
    return chunks, para_indices


# ─── Internal helpers ──────────────────────────────────────────────────────────


def _merge_paragraph_split_annotated(
    paragraphs: List[str], max_chunk_chars: int
) -> Tuple[List[str], List[int]]:
    """
    Like _merge_paragraph_split but also returns the para_idx for each chunk.

    The para_idx of a merged chunk is the index of the first original paragraph
    that entered that merge group.  Oversized paragraphs are sub-split at
    sentence boundaries; their sub-chunks all receive the paragraph's own index.
    """
    result: List[str] = []
    para_indices: List[int] = []
    current_parts: List[str] = []
    current_len = 0
    current_start_para: int = 0

    for para_idx, para in enumerate(paragraphs):
        if len(para) > max_chunk_chars:
            if current_parts:
                result.append("\n\n".join(current_parts))
                para_indices.append(current_start_para)
                current_parts = []
                current_len = 0
            sub = _sentence_group_split(para, max_chunk_chars)
            result.extend(sub)
            para_indices.extend([para_idx] * len(sub))
            current_start_para = para_idx + 1
            continue

        separator_overhead = 2 if current_parts else 0
        if current_len + separator_overhead + len(para) > max_chunk_chars and current_parts:
            result.append("\n\n".join(current_parts))
            para_indices.append(current_start_para)
            current_parts = [para]
            current_len = len(para)
            current_start_para = para_idx
        else:
            current_parts.append(para)
            current_len += separator_overhead + len(para)

    if current_parts:
        result.append("\n\n".join(current_parts))
        para_indices.append(current_start_para)

    if not result:
        return ["\n\n".join(paragraphs)], [0]
    return result, para_indices


def _merge_paragraph_split(paragraphs: List[str], max_chunk_chars: int) -> List[str]:
    """
    Group consecutive paragraphs into chunks up to *max_chunk_chars*.

    Paragraphs that are individually oversized are sub-split at sentence
    boundaries before being added to the result.
    """
    result: List[str] = []
    current_parts: List[str] = []
    current_len = 0

    for para in paragraphs:
        if len(para) > max_chunk_chars:
            # Flush the current accumulation first
            if current_parts:
                result.append("\n\n".join(current_parts))
                current_parts = []
                current_len = 0
            result.extend(_sentence_group_split(para, max_chunk_chars))
            continue

        separator_overhead = 2 if current_parts else 0  # "\n\n" joiner
        if current_len + separator_overhead + len(para) > max_chunk_chars and current_parts:
            result.append("\n\n".join(current_parts))
            current_parts = [para]
            current_len = len(para)
        else:
            current_parts.append(para)
            current_len += separator_overhead + len(para)

    if current_parts:
        result.append("\n\n".join(current_parts))

    return result or ["\n\n".join(paragraphs)]


def _sentence_group_split(text: str, max_chars: int) -> List[str]:
    """Group sentences into chunks not exceeding *max_chars*."""
    if len(text) <= max_chars:
        return [text]

    sents = split_sentences(text)
    if len(sents) <= 1:
        return _hard_split(text, max_chars)

    chunks: List[str] = []
    current = ""
    for sent in sents:
        candidate = (current + " " + sent).strip() if current else sent
        if len(candidate) > max_chars and current:
            chunks.append(current.strip())
            current = sent
        else:
            current = candidate
    if current:
        chunks.append(current.strip())
    return chunks or [text]


def _hard_split(text: str, max_chars: int) -> List[str]:
    """Last-resort split: cut at the last space before *max_chars*."""
    chunks: List[str] = []
    while len(text) > max_chars:
        cut = text.rfind(" ", 0, max_chars)
        if cut <= 0:
            cut = max_chars
        chunks.append(text[:cut].strip())
        text = text[cut:].strip()
    if text:
        chunks.append(text)
    return chunks
